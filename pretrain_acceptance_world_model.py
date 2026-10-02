"""Interact -> verifier labels -> replay minibatches -> update world-model weights.

No input archives. Run `--help`; see WORLD_MODEL_PRETRAINING.md for smoke limits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback
import zipfile

import numpy as np
import torch

from world_model_core import (AcceptanceWorldModel, ExperienceReplay, WorldModelLearner,
    acceptance_nll, expected_acceptance, pack_observations, prefix_log_distribution)
from world_model_environment import NativeTrainingEnvironment
from native_elysia_graph import NativeEosWithoutSnapshot
from world_model_hindsight import HindsightLabeler


def append_json(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class ExperienceWriter:
    def __init__(self, output, resume=False):
        self.output = output
        self.pending = []
        self.pending_topk = []
        self.shards = 0
        self.count = 0
        self.edge_count = 0
        self.label_records = {}
        (output / "experience").mkdir(parents=True, exist_ok=resume)
        if resume:
            def read_rows(name):
                path = output / name
                if not path.exists(): return []
                with path.open(encoding="utf-8") as stream:
                    return [json.loads(row) for row in stream if row.strip()]
            states = read_rows("states.jsonl")
            edges = read_rows("edges.jsonl")
            labels = read_rows("labels.jsonl")
            shards = list((output / "experience").glob("shard_*.npz"))
            self.count = len(states)
            self.edge_count = len(edges)
            self.shards = max((int(path.stem.split("_")[-1]) for path in shards), default=-1) + 1
            self.label_records = {record["state_id"]: record for record in labels}

    def add(self, state, parent, action, timing, split):
        o = state.observation
        self.count += 1
        append_json(self.output / "states.jsonl", dict(state_id=o.uid,
            parent_state_id=None if parent is None else parent.observation.uid,
            question=o.question, round_id=o.round_id, split=split, action=action,
            length=o.length, accepted_len=None, label_status="pending_at_capture",
            labels_join="labels.jsonl by state_id (latest record)",
            prefix_token_ids=state.prefix, segment_start=len(state.prior),
            native_unmask_forward_index=state.snapshot["unmask_forward_index"],
            native_hidden_start_offset=state.snapshot["native_hidden_start_offset"],
            native_topk_start_offset=state.snapshot["native_topk_start_offset"],
            hidden_stage="native_forward_pre_counterfactual_fill",
            feature_cache_semantics="latest_native_rows_with_validity_and_transition_age",
            terminal_reason=state.terminal_reason,
            shard=f"experience/shard_{self.shards:04d}.npz", row=len(self.pending), **timing))
        if parent is not None:
            self.edge_count += 1
            append_json(self.output / "edges.jsonl", dict(parent=parent.observation.uid,
                child=o.uid, action=action, question=o.question, split=split,
                delta_acceptance=None, labels_join="labels.jsonl by parent/child state_id"))
        self.pending.append(o)
        self.pending_topk.append((
            np.asarray(state.snapshot["topk_token_ids"], dtype=np.int32),
            np.asarray(state.snapshot["topk_logits"], dtype=np.float16)))

    def label(self, record):
        self.label_records[record["state_id"]] = record
        append_json(self.output / "labels.jsonl", record)

    def teacher(self, state, source='actual_STOP_forward'):
        if state.observation.teacher_margin is not None:
            append_json(self.output / "teacher_targets.jsonl", dict(state_id=state.observation.uid,
                margin=state.observation.teacher_margin.tolist(), source=source,
                role="training_target_only_not_encoder_feature",
                features=(None if state.observation.teacher_features is None
                          else state.observation.teacher_features.tolist()),
                feature_layout=(None if state.observation.teacher_features is None else
                    ['tanh_candidate_rival_margin_div5','candidate_probability','teacher_forced_local_agreement',
                     'final_norm_causal_hidden_projected32_div10']),
                hidden_projection_seed=901 if state.observation.teacher_features is not None else None,
                verifier_distribution=("top_k_logits_plus_exact_logsumexp_in_experience_npz"
                    if state.observation.teacher_topk_ids is not None else None),
                hidden_stage=(None if state.observation.teacher_features is None else
                    "verifier_final_norm_at_causal_position_before_candidate_logit")))

    def flush(self):
        if not self.pending:
            return
        observations = self.pending
        lengths = np.asarray([o.length for o in observations], dtype=np.int32)
        arrays = {name: torch.cat([getattr(o, name) for o in observations]).numpy()
                  for name in ("ids", "hidden", "gaps", "scalars")}
        if all(o.topk_ids is not None for o in observations):
            arrays["aligned_topk_token_ids"] = torch.cat([o.topk_ids for o in observations]).numpy().astype(np.int32)
            arrays["history"] = torch.cat([o.history for o in observations]).numpy()
        arrays["teacher_margin"] = torch.cat([o.teacher_margin if o.teacher_margin is not None
            else torch.zeros(o.length) for o in observations]).numpy()
        arrays["teacher_valid"] = np.concatenate([np.full(o.length,o.teacher_margin is not None,dtype=np.bool_)
                                                  for o in observations])
        if any(o.verifier_history is not None for o in observations):
            history = [o.verifier_history if o.verifier_history is not None else torch.empty(0, 48)
                       for o in observations]
            arrays['verifier_history'] = torch.cat(history).numpy().astype(np.float16)
            arrays['verifier_history_offsets'] = np.concatenate([[0], np.cumsum([len(h) for h in history])])
            arrays['teacher_features'] = torch.cat([o.teacher_features if o.teacher_features is not None
                else torch.zeros(o.length, 35) for o in observations]).numpy().astype(np.float16)
        arrays['teacher_aux_features'] = torch.cat([o.teacher_aux_features if o.teacher_aux_features is not None
            else torch.zeros(o.length, 6) for o in observations]).numpy().astype(np.float16)
        teacher_k = max((o.teacher_topk_ids.shape[-1] for o in observations
                         if o.teacher_topk_ids is not None), default=0)
        if teacher_k:
            arrays['teacher_topk_token_ids'] = torch.cat([
                o.teacher_topk_ids if o.teacher_topk_ids is not None else
                torch.zeros(o.length, teacher_k, dtype=torch.long) for o in observations
            ]).numpy().astype(np.int32)
            arrays['teacher_topk_logits'] = torch.cat([
                o.teacher_topk_logits if o.teacher_topk_logits is not None else
                torch.zeros(o.length, teacher_k, dtype=torch.float16) for o in observations
            ]).numpy().astype(np.float16)
            arrays['teacher_logsumexp'] = torch.cat([
                o.teacher_logsumexp if o.teacher_logsumexp is not None else
                torch.zeros(o.length) for o in observations
            ]).numpy().astype(np.float32)
            arrays['teacher_distribution_valid'] = np.concatenate([
                np.full(o.length, o.teacher_topk_ids is not None, dtype=np.bool_)
                for o in observations
            ])
        arrays['teacher_actual'] = np.asarray([o.teacher_is_actual for o in observations], dtype=np.bool_)
        arrays.update(lengths=lengths, offsets=np.concatenate([[0], np.cumsum(lengths)]),
            context=torch.stack([o.context for o in observations]).numpy(),
            accepted=np.asarray([-1 if o.accepted is None else o.accepted for o in observations], dtype=np.int32),
            label_valid=np.asarray([o.accepted is not None for o in observations], dtype=np.bool_),
            accepted_lower_bound=np.asarray([self.label_records.get(o.uid, {}).get("lower_bound", 0)
                                              for o in observations], dtype=np.int32),
            native_topk_offsets=np.concatenate([[0], np.cumsum([len(p[0]) for p in self.pending_topk])]),
            native_topk_token_ids=np.concatenate([p[0] for p in self.pending_topk]),
            native_topk_logits=np.concatenate([p[1] for p in self.pending_topk]))
        destination = self.output / f"experience/shard_{self.shards:04d}.npz"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
        temporary.replace(destination)
        self.pending = []
        self.pending_topk = []
        self.shards += 1


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(row) for row in stream if row.strip()]


def restore_replays(output, args, checkpoint_value):
    """Rebuild bounded training/holdout graphs from an atomic partial archive."""
    from world_model_core import Observation
    output = Path(output)
    metadata = read_jsonl(output / "states.jsonl")
    edges = read_jsonl(output / "edges.jsonl")
    labels = {row["state_id"]: row for row in read_jsonl(output / "labels.jsonl")}
    teachers = {row["state_id"]: row for row in read_jsonl(output / "teacher_targets.jsonl")}
    selected = {"train": [], "validation": []}
    for split in selected:
        rows = [row for row in metadata if row["split"] == split]
        selected[split] = rows[-args.replay_states:]
    keep = {row["state_id"] for rows in selected.values() for row in rows}
    by_shard = defaultdict(list)
    for row in metadata:
        if row["state_id"] in keep:
            by_shard[row["shard"]].append(row)
    restored = {}
    for shard, rows in by_shard.items():
        with np.load(output / shard, allow_pickle=False) as arrays:
            # NpzFile is lazy and decompresses a member on every access. Cache
            # each selected shard once; otherwise restoring 1k replay rows
            # repeatedly inflates the same compressed arrays thousands of times.
            arrays = {name: arrays[name] for name in arrays.files}
            for meta in rows:
                row = int(meta["row"])
                begin, end = map(int, arrays["offsets"][row:row+2])
                def tensor(name, dtype):
                    return torch.tensor(arrays[name][begin:end], dtype=dtype)
                label = labels.get(meta["state_id"], {})
                observation = Observation(meta["state_id"], meta["question"],
                    int(meta["round_id"]), tensor("ids", torch.long),
                    tensor("hidden", torch.float16), tensor("gaps", torch.float16),
                    tensor("scalars", torch.float32),
                    torch.tensor(arrays["context"][row], dtype=torch.float32),
                    (int(label["accepted_len"]) if label.get("label_valid") else None),
                    torch.tensor(meta["prefix_token_ids"], dtype=torch.long),
                    tensor("aligned_topk_token_ids", torch.long), tensor("history", torch.float32))
                if "verifier_history_offsets" in arrays:
                    h0, h1 = map(int, arrays["verifier_history_offsets"][row:row+2])
                    observation.verifier_history = torch.tensor(
                        arrays["verifier_history"][h0:h1], dtype=torch.float32)
                teacher = teachers.get(meta["state_id"])
                if teacher is not None:
                    observation.teacher_margin = torch.tensor(teacher["margin"], dtype=torch.float32)
                    if teacher.get("features") is not None:
                        observation.teacher_features = torch.tensor(teacher["features"], dtype=torch.float32)
                    observation.teacher_is_actual = teacher.get("source") == 'actual_STOP_forward'
                if "teacher_aux_features" in arrays:
                    observation.teacher_aux_features = tensor("teacher_aux_features", torch.float32)
                if "teacher_topk_token_ids" in arrays and bool(arrays.get(
                        "teacher_distribution_valid", np.zeros(int(arrays["offsets"][row+1]-arrays["offsets"][row]),dtype=np.bool_)
                        )[begin:end].any()):
                    observation.teacher_topk_ids = tensor("teacher_topk_token_ids", torch.long)
                    observation.teacher_topk_logits = tensor("teacher_topk_logits", torch.float16)
                    observation.teacher_logsumexp = tensor("teacher_logsumexp", torch.float32)
                    observation.teacher_is_actual = bool(arrays.get("teacher_actual", np.zeros(len(arrays["lengths"]),dtype=np.bool_))[row])
                restored[meta["state_id"]] = observation
    train = ExperienceReplay(args.replay_states, args.seed, sampling_mode="action_balanced")
    validation = ExperienceReplay(args.replay_states, args.seed + 1)
    for split, replay in (("train", train), ("validation", validation)):
        for meta in selected[split]:
            obs = restored.get(meta["state_id"])
            if obs is not None: replay.add_node(obs)
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        if parent not in keep or child not in keep: continue
        replay = train if edge["split"] == "train" else validation
        replay.add(restored[parent], restored[child], edge["action"])
    if checkpoint_value.get("replay_rng") is not None:
        train.rng.setstate(checkpoint_value["replay_rng"])
    if checkpoint_value.get("replay_state_rng") is not None:
        train.state_rng.setstate(checkpoint_value["replay_state_rng"])
    return train, validation


def package(output, archive, summary):
    """Only this run's small artifacts, never source/LLM/HF cache directories."""
    atomic_json(output / "summary.json", summary)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zip_file:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                zip_file.write(path, path.relative_to(output),
                    compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary) as zip_file:
        bad = zip_file.testzip()
        if bad:
            raise RuntimeError(f"ZIP checksum error: {bad}")
    temporary.replace(archive)
    print(f"[archive] status={summary['status']} {archive} ({archive.stat().st_size/2**20:.1f} MiB)", flush=True)


def checkpoint(learner, replay, args, output, exploration_rng=None, shadow_rng=None):
    value = learner.checkpoint()
    value.update(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        replay_rng=replay.rng.getstate(),
        replay_state_rng=replay.state_rng.getstate(),
        exploration_rng=None if exploration_rng is None else exploration_rng.getstate(),
        shadow_rng=None if shadow_rng is None else shadow_rng.getstate(),
        resume_note="Optimizer/EMA and RNG saved; replay is reconstructed from experience shards on resume")
    temporary = output / "checkpoint.pt.tmp"
    torch.save(value, temporary)
    temporary.replace(output / "checkpoint.pt")


@torch.no_grad()
def evaluate(learner, replay, horizon=3, output=None, baseline_replay=None):
    if not replay.nodes:
        return dict(status="no_holdout_observations")
    learner.model.eval()
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    current_by_length = defaultdict(lambda: dict(n=0, absolute_error=0.0, exact=0, zero_baseline_error=0.0))
    current_by_question = defaultdict(lambda: dict(n=0, absolute_error=0.0, exact=0))
    current_labels = []
    current_labels_by_length = defaultdict(list)
    latent_globals = []
    nodes = list(replay.nodes.values())
    exact = 0
    for begin in range(0, len(nodes), 8):
        batch = pack_observations(nodes[begin:begin+8], learner.token_table, learner.device)
        state = learner.model.encoder(batch)
        latent_globals.append(state.global_state.cpu())
        logits = learner.model.acceptance(state)
        predicted = expected_acceptance(logits, state.lengths)
        distribution = prefix_log_distribution(logits,state.lengths)
        mode = distribution.argmax(-1)
        known = batch["labels"] >= 0
        count = int(known.sum())
        totals["current"][0] += count
        totals["current"][1] += float((predicted[known]-batch["labels"][known]).abs().sum())
        totals["current"][2] += float(acceptance_nll(logits, state.lengths, batch["labels"])) * count
        exact += int((mode[known] == batch["labels"][known]).sum())
        for j, o in enumerate(nodes[begin:begin+8]):
            if o.accepted is None:
                continue
            error = abs(float(predicted[j])-o.accepted)
            length_group = current_by_length[o.length]
            length_group["n"] += 1
            length_group["absolute_error"] += error
            length_group["exact"] += int(int(mode[j]) == o.accepted)
            length_group["zero_baseline_error"] += o.accepted
            question_group = current_by_question[o.question]
            question_group["n"] += 1
            question_group["absolute_error"] += error
            question_group["exact"] += int(int(mode[j]) == o.accepted)
            current_labels.append(o.accepted)
            current_labels_by_length[o.length].append(o.accepted)
        if output is not None:
            for j,o in enumerate(nodes[begin:begin+8]):
                append_json(output/"validation_predictions.jsonl",dict(state_id=o.uid,kind="current",
                    question=o.question,length=o.length,K=o.accepted,predicted_K=int(mode[j]),
                    expected_K=float(predicted[j]),probabilities=distribution[j,:o.length+1].exp().tolist()))
    outgoing = defaultdict(list)
    for p, c, a in replay.edges:
        outgoing[p].append((c, a))
    rollout_results = defaultdict(lambda: dict(n=0, absolute_error=0.0, baseline_error=0.0, gain_edges=0))
    rollout_by_action = defaultdict(lambda: dict(n=0, absolute_error=0.0, baseline_error=0.0, exact=0, gains=0))
    one_step_groups = defaultdict(lambda: dict(n=0, absolute_error=0.0))
    # Every real first edge is represented once. Longer paths choose a stable child,
    # not the child with the best oracle yield. No training on held-out questions.
    for parent, child, action in replay.edges:
        source = pack_observations([replay.nodes[parent]], learner.token_table, learner.device)
        state = learner.model.encoder(source)
        baseline = float(expected_acceptance(learner.model.acceptance(state), state.lengths)[0])
        first_label = replay.nodes[parent].accepted
        for depth in range(1, horizon+1):
            state = learner.model.transition(state,
                torch.tensor([int(action == "E")], device=learner.device), learner.extension_size)
            prediction = float(expected_acceptance(learner.model.acceptance(state), state.lengths)[0])
            lp = prefix_log_distribution(learner.model.acceptance(state),state.lengths)
            mode = int(lp.argmax(-1)[0])
            label = replay.nodes[child].accepted
            if label is not None:
                record = rollout_results[f"h{depth}"]
                record["n"] += 1
                record["absolute_error"] += abs(prediction-label)
                record["baseline_error"] += abs(baseline-label)
                record.setdefault("exact_K",0)
                record["exact_K"] += int(mode == label)
                record["gain_edges"] += int(first_label is not None and label > first_label)
                record.setdefault("gain_comparable_paths", 0)
                record["gain_comparable_paths"] += int(first_label is not None)
                action_record = rollout_by_action[f"h{depth}_{action}"]
                action_record["n"] += 1
                action_record["absolute_error"] += abs(prediction-label)
                action_record["baseline_error"] += abs(baseline-label)
                action_record["exact"] += int(mode == label)
                action_record["gains"] += int(first_label is not None and label > first_label)
            if output is not None:
                row=dict(kind="rollout",source=parent,state_id=child,depth=depth,action=action,
                         K=label,predicted_K=mode,expected_K=prediction)
                if hasattr(state,"mask_probs"):
                    truth = replay.nodes[child].scalars[:,0].to(learner.device)
                    start = int(state.context[0,2]*64)
                    row["active_mask_accuracy"] = float(((state.mask_probs[0,start:len(truth)]>.5)==truth[start:].bool()).float().mean())
                    row["predicted_mask_probabilities"] = state.mask_probs[0,:len(truth)].tolist()
                append_json(output/"validation_predictions.jsonl",row)
            if depth == 1 and label is not None and first_label is not None:
                change = "gain" if label > first_label else "loss" if label < first_label else "same"
                group = one_step_groups[f"{action}_{change}"]
                group["n"] += 1
                group["absolute_error"] += abs(prediction-label)
            children = sorted(outgoing.get(child, []))
            if not children:
                break
            child, action = children[0]
    current = totals["current"]
    train_labels = ([] if baseline_replay is None else
        [o.accepted for o in baseline_replay.nodes.values() if o.accepted is not None])
    train_mean = sum(train_labels)/len(train_labels) if train_labels else None
    train_median = float(torch.tensor(train_labels,dtype=torch.float32).median()) if train_labels else None
    train_by_length = defaultdict(list)
    if baseline_replay is not None:
        for observation in baseline_replay.nodes.values():
            if observation.accepted is not None:
                train_by_length[observation.length].append(observation.accepted)
    return dict(status="evaluated", questions=sorted({o.question for o in nodes}),
        label_coverage=dict(total=len(nodes), exact=int(current[0]), unresolved=len(nodes)-int(current[0])),
        current=dict(n=int(current[0]), mae=current[1]/current[0] if current[0] else None,
                     nll=current[2]/current[0] if current[0] else None,
                     exact_K=exact/current[0] if current[0] else None,
                     train_mean_baseline_mae=(sum(abs(x-train_mean) for x in current_labels)/len(current_labels)
                         if current_labels and train_mean is not None else None),
                     train_median_baseline_mae=(sum(abs(x-train_median) for x in current_labels)/len(current_labels)
                         if current_labels and train_median is not None else None)),
        current_by_proposal_length={str(key): dict(n=value["n"],
            mae=value["absolute_error"]/value["n"], exact_K=value["exact"]/value["n"],
            always_zero_MAE=value["zero_baseline_error"]/value["n"],
            train_mean_baseline_MAE=(sum(abs(x-sum(train_by_length[key])/len(train_by_length[key]))
                for x in current_labels_by_length[key])/value["n"] if train_by_length.get(key) else None))
            for key,value in sorted(current_by_length.items()) if value["n"]},
        current_by_question={key: dict(n=value["n"],mae=value["absolute_error"]/value["n"],
            exact_K=value["exact"]/value["n"])
            for key,value in sorted(current_by_question.items()) if value["n"]},
        global_latent_std_mean=float(torch.cat(latent_globals).std(dim=0, unbiased=False).mean()),
        one_step_by_action_and_change={key: dict(n=v["n"], mae=v["absolute_error"]/v["n"])
                                      for key, v in one_step_groups.items()},
        rollout={key: dict(n=value["n"], mae=value["absolute_error"]/value["n"],
            exact_K=value["exact_K"]/value["n"],
            unchanged_prediction_baseline_mae=value["baseline_error"]/value["n"],
            true_gain_paths=value["gain_edges"], gain_comparable_paths=value["gain_comparable_paths"])
            for key, value in rollout_results.items()},
        rollout_by_horizon_and_action={key:dict(n=value["n"],
            mae=value["absolute_error"]/value["n"],
            unchanged_prediction_baseline_mae=value["baseline_error"]/value["n"],
            exact_K=value["exact"]/value["n"],
            true_gain_rate=value["gains"]/value["n"])
            for key,value in sorted(rollout_by_action.items()) if value["n"]},
        warning="Small-question probe checks learning and rollout signals, not controller quality")


def action_probabilities(actions, args):
    weights = {"S": args.stop_weight, "E": args.extend_weight, "R": args.refine_weight}
    total = sum(weights[a] for a in actions)
    return {a: weights[a]/total for a in actions}


def explore_questions(args, questions, tokenizer, environment, learner, writer, summary, resume=None):
    rng = random.Random(args.seed)
    if resume and resume.get("exploration_rng") is not None:
        rng.setstate(resume["exploration_rng"])
    train_replay = (resume["train_replay"] if resume else
                    ExperienceReplay(args.replay_states, args.seed))
    validation_replay = (resume["validation_replay"] if resume else
                         ExperienceReplay(args.replay_states, args.seed+1))
    train_count = len(questions)-args.validation_questions
    initial = {name: parameter.detach().cpu().clone() for name, parameter in learner.model.named_parameters()}
    next_round_id = defaultdict(int)
    shadow_rng = None
    if resume:
        for state in read_jsonl(args.output_dir / "states.jsonl"):
            next_round_id[state["question"]] = max(next_round_id[state["question"]],
                                                     int(state["round_id"]) + 1)
    completed = set()
    if resume:
        completed = {(str(row["question_id"]), int(row.get("episode", 0)))
                     for row in resume["completed_questions"]}
    try:
        episodes = getattr(args,"episodes_per_question",1)
        itinerary = [(i,episode,q) for i,q in enumerate(questions) for episode in range(episodes)]
        detailed = getattr(args, 'model_architecture', '') == 'two_source'
        itinerary = [item for item in itinerary
                     if (str(item[2]["question_id"]), int(item[1])) not in completed]
        if detailed:
            train_replay.sampling_mode = 'action_balanced'
            # Collect a fixed holdout BEFORE training. Its labels never enter replay.
            itinerary.sort(key=lambda item: item[0] < train_count)
            summary.setdefault('learning_curve', [])
            shadow_rng = random.Random(args.seed+7781)
            if resume and resume.get("shadow_rng") is not None:
                shadow_rng.setstate(resume["shadow_rng"])
            audit_milestones = set(args.audit_milestones)
            train_completed = sum(row.get("split") == "train" for row in resume["completed_questions"]) if resume else 0
            validation_completed = sum(row.get("split") == "validation" for row in resume["completed_questions"]) if resume else 0
            audit_milestones = {m for m in audit_milestones if m > train_completed}

            def audit_stage(count):
                from world_model_training_audit import audit_learning_stage
                result = audit_learning_stage(learner, validation_replay, args.output_dir, count,
                                               args.horizon, args.audit_states_per_question)
                summary['learning_curve'].append(result)
                atomic_json(args.output_dir/'learning_curve.json', summary['learning_curve'])
        for question_index, episode, question in itinerary:
            split = "train" if question_index < train_count else "validation"
            question_id = str(question["question_id"])
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": question["prompt"]}], tokenize=True, add_generation_prompt=True)
            prefix = list(prompt)
            if hasattr(environment, 'reset_history'): environment.reset_history()
            if len(prefix) > args.max_context_tokens:
                raise ValueError(f"Question {question_id} exceeds context cap; no silent truncation")
            generated = []
            labeler = HindsightLabeler(prefix, writer.label,
                lambda record: append_json(args.output_dir / "label_conflicts.jsonl", record))
            buffer = train_replay if split == "train" else validation_replay
            if detailed:
                position = validation_completed+1 if split=='validation' else train_completed+1
                size = args.validation_questions if split=='validation' else train_count
                print(f'[question] {split} {position}/{size} episode={episode+1}/{episodes} {question_id}',flush=True)
            else:
                print(f"[question] {question_index+1}/{len(questions)} episode={episode+1}/{episodes} {question_id} split={split}", flush=True)

            def train_available():
                if split != "train":
                    return
                for _ in range(args.updates_per_transition):
                    metrics = learner.update(buffer, args.batch_sequences, args.horizon)
                    if metrics is None:
                        continue
                    append_json(args.output_dir / "training_metrics.jsonl", metrics)
                    if learner.updates % 10 == 0:
                        print(f"[train] update={learner.updates} loss={metrics['loss']:.4f} "
                              f"h={metrics['horizon']} replay={metrics['replay_states']}", flush=True)

            def on_state(state, parent, action, timing):
                if getattr(args, "watchdog_enabled", False):
                    faulthandler.dump_traceback_later(300, repeat=True)
                writer.add(state, parent, action, timing, split)
                labeler.register(state)
                o = state.observation
                if getattr(args,"model_architecture","legacy") in ("token_dual", "two_source"):
                    learner.model.eval()
                    with torch.no_grad():
                        b=pack_observations([o],learner.token_table,learner.device)
                        z=learner.model.encoder(b)
                        lp=prefix_log_distribution(learner.model.acceptance(z),z.lengths)[0,:o.length+1]
                        append_json(args.output_dir/"online_predictions.jsonl",dict(state_id=o.uid,
                            split=split,updates=learner.updates,predicted_K=int(lp.argmax()),probabilities=lp.exp().tolist(),
                            captured_before_label_and_update=True))
                if parent is not None:
                    buffer.add(parent.observation, o, action)
                    train_available()
                else:
                    buffer.add_node(o)
                if detailed and shadow_rng.random() < args.shadow_verify_probability:
                    elapsed = environment.shadow(state, args.max_proposal_tokens+1)
                    writer.teacher(state, source='shadow_forward')
                    labeler.mark_direct(state, 'shadow_verifier')
                    append_json(args.output_dir/'shadow_verifications.jsonl', dict(state_id=o.uid,
                        split=split, accepted_len=o.accepted, verifier_ms=elapsed,
                        history_updated=False, state_submitted=False))
                print(f"[state] {o.uid} action={action or 'root'} L={o.length} K={o.accepted}", flush=True)

            environment.on_state = on_state
            end_reason = "max_rounds_per_question"
            local_round = 0
            while args.max_rounds_per_question == 0 or local_round < args.max_rounds_per_question:
                round_id = next_round_id[question_id]
                next_round_id[question_id] += 1
                local_round += 1
                # Unlimited answer length does NOT mean an unlimited proposal.
                # A round can emit at most L accepted tokens plus one bonus.
                remaining = (args.max_new_tokens-len(generated) if args.max_new_tokens
                             else args.max_proposal_tokens+1)
                if not args.max_new_tokens and len(prefix)+args.max_proposal_tokens+1 > args.max_context_tokens:
                    raise RuntimeError(f"Context safety limit reached before EOS for {question_id}; "
                                       "result is partial, not a completed answer. Increase max_context_tokens only if VRAM permits.")
                if remaining <= 0 or len(prefix) > args.max_context_tokens:
                    end_reason = "max_new_tokens" if remaining <= 0 else "max_context_tokens"
                    break
                try:
                    state = environment.start(question_id, round_id, prefix, remaining)
                except NativeEosWithoutSnapshot as terminal:
                    # Native Elysia can predict EOS before the raw-snapshot hook
                    # fires. Preserve the actual EOS proposal and obtain its real
                    # verifier outcome, but do not invent hidden/top-k features.
                    accepted, _, emitted, verifier_ms = environment.verifier.score(
                        prefix, terminal.candidate_token_ids, remaining)
                    environment.stats["verifier_calls"] += 1
                    environment.stats["native_eos_without_snapshot"] = (
                        environment.stats.get("native_eos_without_snapshot", 0) + 1)
                    if hasattr(environment, 'remember'):
                        environment.remember(prefix, terminal.candidate_token_ids, accepted, emitted)
                    append_json(args.output_dir / "native_eos_without_snapshot.jsonl", dict(
                        question=question_id, round_id=round_id, split=split,
                        proposal_token_ids=terminal.candidate_token_ids,
                        verifier_accepted_len=accepted, emitted_token_ids=emitted,
                        verifier_ms=verifier_ms, native_stats=terminal.stats,
                        feature_snapshot_available=False,
                        excluded_from_world_model_state_training=True))
                    append_json(args.output_dir / "actions.jsonl", dict(
                        state_id=None, question=question_id, round_id=round_id,
                        split=split, action="forced_stop_native_eos",
                        executed=True, child_state_id=None,
                        feature_snapshot_available=False))
                    labeler.after_emitted(prefix, emitted)
                    prefix += emitted
                    generated += emitted
                    print(f"[native-eos-stop] {question_id} proposal="
                          f"{terminal.candidate_token_ids} accepted={accepted} "
                          f"emitted={len(emitted)}", flush=True)
                    if environment.eos_id in emitted:
                        end_reason = "eos"
                        break
                    continue
                while True:
                    actions = environment.actions(state)
                    probabilities = action_probabilities(actions, args)
                    chosen = rng.choices(actions, weights=[probabilities[a] for a in actions])[0]
                    event = dict(state_id=state.observation.uid, question=question_id,
                                 round_id=round_id, split=split, action=chosen,
                                 legal_probabilities=probabilities)
                    if chosen == "S":
                        verifier_ms = environment.submit(state, remaining)
                        writer.teacher(state)
                        labeler.after_stop(state)
                        append_json(args.output_dir / "actions.jsonl", dict(**event, executed=True))
                        train_available()
                        print(f"[stop] {state.observation.uid} K={state.observation.accepted} "
                              f"emitted={len(state.emitted)}", flush=True)
                        break
                    child = environment.step(state, chosen, remaining)
                    append_json(args.output_dir / "actions.jsonl", dict(**event, executed=child is not None,
                        child_state_id=None if child is None else child.observation.uid))
                    if child is None:
                        continue  # R exhausted: resample remaining legal S/E, never force E.
                    state = child
                if not state.emitted:
                    raise RuntimeError("Verifier emitted no tokens")
                prefix += state.emitted
                generated += state.emitted
                append_json(args.output_dir / "rounds.jsonl", dict(question=question_id,
                    round_id=round_id, split=split, final_state=state.observation.uid,
                    accepted_len=state.observation.accepted, verifier_ms=verifier_ms,
                    emitted=state.emitted))
                if environment.eos_id in state.emitted:
                    end_reason = "eos"
                    break
            if end_reason != "eos" and args.max_new_tokens and len(generated) >= args.max_new_tokens:
                end_reason = "max_new_tokens"
            coverage = labeler.finish(end_reason)
            for key, value in coverage.items():
                summary.setdefault("label_coverage", {}).setdefault(key, 0)
                summary["label_coverage"][key] += value
            summary["questions_completed"] += int(episode==episodes-1)
            summary["episodes_completed"] = summary.get("episodes_completed",0)+1
            append_json(args.output_dir / "questions.jsonl", dict(**question, split=split,episode=episode,
                generated_tokens=generated, decoded=tokenizer.decode(generated),
                collection_end_reason=end_reason, label_coverage=coverage,
                generation_is_bounded_smoke_not_answer_accuracy_benchmark=bool(args.max_new_tokens or args.max_rounds_per_question)))
            writer.flush()
            if detailed and episode == episodes-1:
                if split == 'validation':
                    validation_completed += 1
                    if validation_completed == args.validation_questions: audit_stage(0)
                else:
                    train_completed += 1
                    if train_completed in audit_milestones or train_completed == train_count:
                        audit_stage(train_completed)
            checkpoint(learner, train_replay, args, args.output_dir, rng,
                       shadow_rng=shadow_rng if detailed else None)
            summary.update(updates=learner.updates, dynamics_updates=learner.dynamics_updates,
                           nodes=writer.count, edges=writer.edge_count,
                           environment=environment.stats)
            atomic_json(args.output_dir / "summary.json", summary)
            if getattr(args,"package_every_question",False) and episode==episodes-1:
                # Atomic refresh: a completed-question ZIP survives a later hard
                # notebook timeout (in-progress files after that checkpoint may not).
                package(args.output_dir,args.output_dir.with_suffix(".zip"),dict(summary,status="partial_checkpoint"))
            if shutil.disk_usage(args.output_dir).free < 1024**3:
                raise OSError("Less than 1 GiB free; stop before corrupting artifacts")
        if detailed:
            summary['evaluation'] = json.loads((args.output_dir/'evaluation'/
                f'learning_{train_completed:03d}.json').read_text(encoding='utf-8'))
        else:
            summary["evaluation"] = evaluate(learner, validation_replay, args.horizon,
                args.output_dir if getattr(args,"model_architecture","legacy")=='token_dual' else None,
                baseline_replay=train_replay)
        delta = sum(float((p.detach().cpu()-initial[n]).square().sum())
                    for n, p in learner.model.named_parameters()) ** 0.5
        summary["parameter_l2_change"] = delta
        if learner.updates == 0 or not np.isfinite(delta) or delta == 0:
            raise RuntimeError("Smoke did not actually update world-model weights")
    finally:
        writer.flush()
        checkpoint(learner, train_replay, args, args.output_dir, rng, shadow_rng=shadow_rng)
        summary.update(updates=learner.updates, dynamics_updates=learner.dynamics_updates,
                       nodes=writer.count, edges=writer.edge_count,
                       environment=environment.stats)


def load_questions(args):
    from datasets import load_dataset
    from types import SimpleNamespace
    from utils import get_first_user_msg
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        field = "question"
    else:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        field = "problem"
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.num_questions > len(indices):
        raise ValueError("Requested more questions than available")
    return [dict(question_id=f"{args.dataset}:{index}", dataset_index=index,
        dataset_source=("openai/gsm8k:main:train" if args.dataset == "gsm8k"
                        else "HuggingFaceH4/MATH-500:test"),
        prompt=get_first_user_msg(SimpleNamespace(dataset_name=args.dataset),
                                  {"problem": str(dataset[index][field])}))
        for index in indices[:args.num_questions]]


def load_resume_archive(archive, output, args):
    """Restore an atomic question-boundary checkpoint from a ZIP or mounted folder."""
    archive = Path(archive)
    output = Path(output)
    if not archive.exists():
        raise FileNotFoundError(f"Resume ZIP not found: {archive}")
    if archive.is_dir():
        required = ("config.json", "summary.json", "checkpoint.pt")
        missing = [name for name in required if not (archive / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Resume folder is missing required files: {missing}")
        # Kaggle Dataset inputs often expose ZIP contents as a read-only folder.
        shutil.copytree(archive, output, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad: raise RuntimeError(f"Resume ZIP checksum error: {bad}")
            for member in zf.infolist():
                target = (output / member.filename).resolve()
                if not target.is_relative_to(output.resolve()):
                    raise RuntimeError(f"Unsafe path in resume archive: {member.filename}")
            zf.extractall(output)
    old_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    expected_keys = ("dataset", "num_questions", "validation_questions", "episodes_per_question",
        "model_architecture", "seed", "latent_dim", "replay_states", "horizon", "extend_size",
        "max_proposal_tokens", "max_refinement_steps", "max_context_tokens", "drafter_threshold",
        "updates_per_transition", "learning_rate", "warmup_updates", "horizon_warmup_updates",
        "hidden_layers", "raw_top_k", "dropout", "batch_sequences", "stop_weight", "extend_weight",
        "refine_weight", "max_rounds_per_question", "max_new_tokens", "physical_block_size",
        "small_block_size", "shadow_verify_probability", "audit_milestones", "audit_states_per_question",
        "audit_retrain_updates", "audit_train_states_per_question", "audit_seeds", "audit_variants",
        "target_model_name", "target_gpu_memory_gib")
    for key in expected_keys:
        if old_config.get(key) != getattr(args, key):
            raise ValueError(f"Resume config mismatch for {key}: archive={old_config.get(key)!r}, run={getattr(args,key)!r}")
    if old_config.get("model_architecture") != "two_source":
        raise ValueError("Only two_source archives are supported for resume")
    checkpoint_path = output / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Resume ZIP has no checkpoint.pt")
    checkpoint_value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    completed_questions = read_jsonl(output / "questions.jsonl")
    if not completed_questions:
        raise RuntimeError("Resume archive contains no completed-question ledger")
    if checkpoint_value.get("model") is None or checkpoint_value.get("optimizer") is None:
        raise RuntimeError("Resume archive checkpoint is missing model/optimizer state")
    summary["resumed_from_archive"] = archive.name
    summary["resumed_from_questions_completed"] = len(completed_questions)
    summary["prior_run_error"] = summary.pop("error", None)
    summary["status"] = "resuming"
    summary["source_revision"] = args.source_revision
    if (output / "error.txt").exists():
        (output / "previous_error.txt").write_text(
            (output / "error.txt").read_text(encoding="utf-8"), encoding="utf-8")
        (output / "error.txt").unlink()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    config.update({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
    config["resume_archive"] = archive.name
    atomic_json(output / "config.json", config)
    train_replay, validation_replay = restore_replays(output, args, checkpoint_value)
    return dict(summary=summary, checkpoint=checkpoint_value,
        train_replay=train_replay, validation_replay=validation_replay,
        completed_questions=completed_questions,
        exploration_rng=checkpoint_value.get("exploration_rng"),
        shadow_rng=checkpoint_value.get("shadow_rng"))


def run(args):
    archive = args.output_dir.with_suffix(".zip")
    if args.output_dir.exists() or archive.exists():
        raise FileExistsError("Use a new output directory; existing results will not be overwritten")
    if args.resume_archive:
        args.output_dir.mkdir(parents=True)
        resume = load_resume_archive(args.resume_archive, args.output_dir, args)
        summary = resume["summary"]
        prior_elapsed = float(summary.get("elapsed_seconds", 0.0))
    else:
        args.output_dir.mkdir(parents=True)
        resume = None
        prior_elapsed = 0.0
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        atomic_json(args.output_dir / "config.json", config)
    if resume is None:
        summary = dict(schema=("interactive_acceptance_two_source_v1" if args.model_architecture=='two_source' else
                          "interactive_acceptance_probe_v3" if args.model_architecture=="token_dual"
                           else "interactive_acceptance_pretrain_v2_hindsight"), status="running",
        questions_completed=0, updates=0, nodes=0, edges=0, source_revision=args.source_revision,
        labels="actual STOP verifier + hindsight over emitted greedy tokens; unresolved is NOT zero",
        verifier="full context, KV disabled; invoked only on chosen/forced STOP",
        exploration="single real trajectory; random legal S/E/R; no side branches",
        stop_is_policy_supervision=False,
        drafter="native Elysia, prefix KV inside each generator invocation",
        refine_execution="bounded deterministic segment replay with predecessor assertion",
            timings_are_profiling_only=True, required_input_archives=False)
    began = time.perf_counter()
    args.watchdog_enabled = True
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("This real-LLM run requires 2 GPUs; CPU unit tests are separate")
        questions = load_questions(args)
        question_split = {
            "train": [q["question_id"] for q in questions[:-args.validation_questions]],
            "validation": [q["question_id"] for q in questions[-args.validation_questions:]]}
        split_path = args.output_dir / "question_split.json"
        if resume:
            if not split_path.is_file() or json.loads(split_path.read_text(encoding="utf-8")) != question_split:
                raise RuntimeError("Regenerated question split differs from the partial archive; refusing unsafe resume")
            expected_ids = {q["question_id"] for q in questions}
            if any(row["question_id"] not in expected_ids for row in resume["completed_questions"]):
                raise RuntimeError("Completed-question ledger is inconsistent with regenerated questions")
            regenerated = {q["question_id"]: q for q in questions}
            for row in resume["completed_questions"]:
                if row.get("prompt") != regenerated[row["question_id"]]["prompt"]:
                    raise RuntimeError(f"Question text changed for {row['question_id']}; refusing to mix datasets")
        else:
            atomic_json(split_path, question_split)
        from sparse_extend_world_model_collector import _load_models
        from structured_sparse_collector import FullContextVerifier
        from native_elysia_graph import NativeElysiaRunner
        tokenizer, target, drafter = _load_models(args)
        import transformers
        summary["runtime"] = dict(torch=str(torch.__version__), transformers=transformers.__version__,
            cuda=torch.version.cuda, gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
            target_revision=getattr(target.config, "_commit_hash", None),
            drafter_revision=getattr(drafter.config, "_commit_hash", None))
        target.eval().requires_grad_(False)
        drafter.eval().requires_grad_(False)
        target_devices = {str(p.device) for p in target.parameters()}
        expected_target_devices = {f"cuda:{args.target_device}", f"cuda:{args.drafter_device}"}
        if not target_devices.issubset(expected_target_devices):
            raise RuntimeError(f"Verifier was placed on unexpected devices: {sorted(target_devices)}")
        if len(target_devices) < 2:
            raise RuntimeError(
                "Memory-optimized run expected the FP16 verifier to be sharded across both GPUs; "
                f"actual devices: {sorted(target_devices)}")
        if {str(p.device) for p in drafter.parameters()} != {f"cuda:{args.drafter_device}"}:
            raise RuntimeError("Drafter was not placed exclusively on its requested GPU")
        table = drafter.get_input_embeddings().weight.detach()
        # STOP/verifier labels require exactly matching token IDs, not just vocab size.
        from transformers import AutoTokenizer
        draft_tokenizer = AutoTokenizer.from_pretrained(args.dllm_dir, local_files_only=True,
                                                       trust_remote_code=True)
        # The dLLM adds its MASK token to the common Qwen token vocabulary.
        from world_model_environment import MASK_ID
        target_vocab = {k: v for k, v in tokenizer.get_vocab().items() if v != MASK_ID}
        draft_vocab = {k: v for k, v in draft_tokenizer.get_vocab().items() if v != MASK_ID}
        if target_vocab != draft_vocab:
            raise RuntimeError("Drafter/verifier token ID mappings differ")
        if args.model_architecture=='two_source':
            from world_model_twosource import TwoSourceWorldModel, configure_losses
            model = TwoSourceWorldModel(drafter.config.hidden_size, table.shape[-1], args.raw_top_k,
                dim=args.latent_dim, num_hidden_layers=len(args.hidden_layers), dropout=args.dropout)
        elif args.model_architecture=="token_dual":
            from world_model_probe import ProbeWorldModel
            model = ProbeWorldModel(drafter.config.hidden_size,table.shape[-1],args.raw_top_k,
                dim=args.latent_dim,num_hidden_layers=len(args.hidden_layers),dropout=args.dropout)
        else:
            model = AcceptanceWorldModel(drafter.config.hidden_size, table.shape[-1],
                args.raw_top_k, dim=args.latent_dim, dropout=args.dropout)
        learner = WorldModelLearner(model, table, f"cuda:{args.drafter_device}", args.extend_size,
            args.learning_rate, warmup_updates=args.warmup_updates,
            horizon_warmup=args.horizon_warmup_updates)
        if resume:
            saved = resume["checkpoint"]
            model.load_state_dict(saved["model"])
            learner.target_encoder.load_state_dict(saved["target_encoder"])
            learner.optimizer.load_state_dict(saved["optimizer"])
            learner.updates = int(saved.get("updates", 0))
            learner.dynamics_updates = int(saved.get("dynamics_updates", 0))
        verifier = FullContextVerifier(target, tokenizer, args)
        environment = NativeTrainingEnvironment(NativeElysiaRunner(drafter, tokenizer, args),
            verifier, tokenizer.eos_token_id, drafter.config.hidden_size, args, None)
        if args.model_architecture == 'two_source':
            from world_model_teacher_environment import TeacherVerifier, TeacherTrainingEnvironment
            configure_losses(learner, 'full')
            verifier = TeacherVerifier(target, tokenizer, args)
            environment = TeacherTrainingEnvironment(NativeElysiaRunner(drafter, tokenizer, args),
                verifier, tokenizer.eos_token_id, drafter.config.hidden_size, args, None, token_table=table)
        writer = ExperienceWriter(args.output_dir, resume=bool(resume))
        environment.counter = writer.count
        if resume:
            environment.stats.update(resume["summary"].get("environment", {}))
            # Model and data loading consume RNG; restore the saved streams only
            # after all initialization so the sampling sequence continues.
            saved = resume["checkpoint"]
            if saved.get("python_rng") is not None: random.setstate(saved["python_rng"])
            if saved.get("torch_rng") is not None: torch.set_rng_state(saved["torch_rng"])
            if torch.cuda.is_available() and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        summary["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary["world_model_architecture"] = args.model_architecture
        summary["verifier_teacher"] = ("candidate_vs_best_rival_margin_at_actual_STOP_only" if args.capture_verifier_teacher else "none")
        summary["prototype_limits"] = "No verifier-prefix hidden memory; prefix memory uses frozen token embeddings; no raw verifier-hidden distillation"
        summary["devices"] = dict(verifier=f"cuda:{args.target_device}",
            drafter=f"cuda:{args.drafter_device}", world_model=f"cuda:{args.drafter_device}")
        summary["verifier_devices"] = sorted(target_devices)
        summary["verifier_device_map"] = getattr(target, "hf_device_map", {})
        summary["verifier_memory_mode"] = "FP16 model sharded across both GPUs; 8 GiB placement cap per GPU"
        if args.model_architecture == 'two_source':
            summary['prototype_limits'] = ('Verifier hidden uses fixed 32-dimensional projection; '
                'GRU re-encodes last 8 actual STOP summaries, reset each episode; '
                'shadow targets never change prefix/history; no online weight updates at deployment')
            summary['verifier_teacher'] = 'margin + local agreement + probability + projected final hidden'
            summary['verifier'] = 'full context, KV disabled; actual S and random shadow probes'
            summary['labels'] = 'actual STOP, shadow verifier and hindsight verified greedy stream; unresolved is missing'
            summary['teacher_capture_probability'] = args.shadow_verify_probability
        explore_questions(args, questions, tokenizer, environment, learner, writer, summary, resume=resume)
        if args.model_architecture == 'two_source':
            # Release both large models before matched small-model ablations.
            environment.on_state = None
            del environment, verifier, target, drafter
            import gc
            gc.collect(); torch.cuda.empty_cache()
            from world_model_training_audit import run_retrained_audit
            summary['factor_audit'] = run_retrained_audit(args, learner)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="partial", error=f"{type(error).__name__}: {error}")
        (args.output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        summary["elapsed_seconds"] = prior_elapsed + time.perf_counter()-began
        for name in ("world_model_core.py", "world_model_probe.py", "world_model_environment.py", "world_model_hindsight.py",
                     "pretrain_acceptance_world_model.py", "native_elysia_graph.py",
                     "structured_sparse_collector.py", "sparse_extend_world_model_collector.py", "Fast_dLLM_v2_1_5B/modeling.py"):
            source = Path(__file__).parent / name
            summary.setdefault("source_sha256", {})[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if args.model_architecture == 'two_source':
            for name in ('world_model_twosource.py', 'world_model_teacher_environment.py', 'world_model_training_audit.py'):
                summary['source_sha256'][name] = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        package(args.output_dir, archive, summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--validation_questions", type=int, default=2)
    parser.add_argument("--episodes_per_question",type=int,default=1)
    parser.add_argument("--model_architecture",choices=["legacy","token_dual","two_source"],default="legacy")
    parser.add_argument('--shadow_verify_probability', type=float, default=.15)
    parser.add_argument('--audit_milestones', nargs='+', type=int, default=[10,20,40,60,80])
    parser.add_argument('--audit_states_per_question', type=int, default=32)
    parser.add_argument('--audit_retrain_updates', type=int, default=400)
    parser.add_argument('--audit_train_states_per_question', type=int, default=24)
    parser.add_argument('--audit_seeds', nargs='+', type=int, default=[42,43])
    parser.add_argument('--audit_variants', nargs='+', default=None)
    parser.add_argument("--package_every_question",action="store_true")
    parser.add_argument("--max_rounds_per_question", type=int, default=2, help="0: no round cap; stop on verified EOS")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="0: no answer token cap; stop on verified EOS")
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_proposal_tokens", type=int, default=64)
    parser.add_argument("--extend_size", type=int, default=8)
    parser.add_argument("--max_refinement_steps", type=int, default=3)
    parser.add_argument("--physical_block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.5)
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=None)
    parser.add_argument("--raw_top_k", type=int, default=32)
    parser.add_argument("--stop_weight", type=float, default=1.0)
    parser.add_argument("--extend_weight", type=float, default=1.0)
    parser.add_argument("--refine_weight", type=float, default=1.0)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=1)
    parser.add_argument("--target_gpu_memory_gib", type=int, default=8,
                        help="Maximum verifier weight placement per GPU when sharding")
    parser.add_argument("--dllm_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume_archive", type=Path, default=None,
                        help="Resume a partial two_source result ZIP into a new output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--source_revision', default='local_unpinned')
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_sequences", type=int, default=8)
    parser.add_argument("--replay_states", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--updates_per_transition", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_updates", type=int, default=8)
    parser.add_argument("--horizon_warmup_updates", type=int, default=24)
    args = parser.parse_args(argv)
    if args.hidden_layers is None:
        args.hidden_layers = [7,14,28] if args.model_architecture in ("token_dual", 'two_source') else [14,28]
    if args.model_architecture=="legacy" and len(args.hidden_layers)!=2:
        parser.error("Legacy architecture needs exactly 2 hidden layers")
    if args.model_architecture=="token_dual" and args.latent_dim%8:
        parser.error("Dual latent dimension must be divisible by 8")
    args.capture_verifier_teacher = args.model_architecture in ("token_dual", 'two_source')
    if not 0 <= args.shadow_verify_probability <= 1:
        parser.error('shadow_verify_probability must be within [0,1]')
    if args.model_architecture == 'two_source':
        if args.horizon != 3 or args.latent_dim < 64 or args.latent_dim % 8:
            parser.error('two_source test requires horizon=3 and latent_dim>=64 divisible by 8')
        from world_model_twosource import FEATURE_VARIANTS
        if args.audit_variants is None: args.audit_variants = list(FEATURE_VARIANTS)
        if 'full' not in args.audit_variants or any(v not in FEATURE_VARIANTS for v in args.audit_variants):
            parser.error('Ablations must include full and use supported variants')
        if any(v.startswith('no_layer') for v in args.audit_variants) and args.hidden_layers != [7,14,28]:
            parser.error('Layer-specific ablations require native hidden_layers 7 14 28')
        if not args.audit_seeds or args.audit_retrain_updates < 1 or min(args.audit_states_per_question, args.audit_train_states_per_question) < 4:
            parser.error('Invalid audit sizes/seeds/updates')
    args.target_placement = "auto"
    if not 0 < args.validation_questions < args.num_questions:
        parser.error("Need both training and validation questions")
    if args.target_device == args.drafter_device:
        parser.error("Drafter and verifier must be on distinct GPUs")
    if args.target_gpu_memory_gib < 1:
        parser.error("target_gpu_memory_gib must be positive")
    for key in ("num_questions", "episodes_per_question", "max_context_tokens",
                "extend_size", "batch_sequences", "horizon", "updates_per_transition", "raw_top_k",
                "physical_block_size", "small_block_size", "latent_dim"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.max_rounds_per_question < 0 or args.max_new_tokens < 0:
        parser.error("Round/token caps must be nonnegative (0 means no cap)")
    if args.max_proposal_tokens < args.extend_size or args.max_proposal_tokens % args.extend_size:
        parser.error("Proposal cap must be a multiple of extend_size")
    if args.replay_states < 2 or args.latent_dim % 4 or args.max_refinement_steps < 0:
        parser.error("Invalid replay size, latent dimension or refinement limit")
    if args.physical_block_size % args.small_block_size:
        parser.error("Physical block size must be divisible by small_block_size")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0:
        parser.error("Invalid dropout or learning rate")
    if args.warmup_updates < 0 or args.horizon_warmup_updates < 0:
        parser.error("Warmup update counts cannot be negative")
    if not 0 <= args.drafter_threshold <= 1:
        parser.error("Threshold must be within [0,1]")
    if any(not math.isfinite(w) or w <= 0 for w in (args.stop_weight, args.extend_weight, args.refine_weight)):
        parser.error("S/E/R exploration weights must be finite and positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    run(arguments)
