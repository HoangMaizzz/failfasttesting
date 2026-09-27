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
    acceptance_nll, expected_acceptance, pack_observations)
from world_model_environment import NativeTrainingEnvironment
from world_model_hindsight import HindsightLabeler


def append_json(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class ExperienceWriter:
    def __init__(self, output):
        self.output = output
        self.pending = []
        self.pending_topk = []
        self.shards = 0
        self.count = 0
        self.edge_count = 0
        self.label_records = {}
        (output / "experience").mkdir()

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

    def flush(self):
        if not self.pending:
            return
        observations = self.pending
        lengths = np.asarray([o.length for o in observations], dtype=np.int32)
        arrays = {name: torch.cat([getattr(o, name) for o in observations]).numpy()
                  for name in ("ids", "hidden", "gaps", "scalars")}
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


def checkpoint(learner, replay, args, output, exploration_rng=None):
    value = learner.checkpoint()
    value.update(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        replay_rng=replay.rng.getstate(),
        exploration_rng=None if exploration_rng is None else exploration_rng.getstate(),
        resume_note="Optimizer/EMA saved; rebuild replay from experience shards for a future resume tool")
    temporary = output / "checkpoint.pt.tmp"
    torch.save(value, temporary)
    temporary.replace(output / "checkpoint.pt")


@torch.no_grad()
def evaluate(learner, replay, horizon=3):
    if not replay.nodes:
        return dict(status="no_holdout_observations")
    learner.model.eval()
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    latent_globals = []
    nodes = list(replay.nodes.values())
    for begin in range(0, len(nodes), 8):
        batch = pack_observations(nodes[begin:begin+8], learner.token_table, learner.device)
        state = learner.model.encoder(batch)
        latent_globals.append(state.global_state.cpu())
        logits = learner.model.acceptance(state)
        predicted = expected_acceptance(logits, state.lengths)
        known = batch["labels"] >= 0
        count = int(known.sum())
        totals["current"][0] += count
        totals["current"][1] += float((predicted[known]-batch["labels"][known]).abs().sum())
        totals["current"][2] += float(acceptance_nll(logits, state.lengths, batch["labels"])) * count
    outgoing = defaultdict(list)
    for p, c, a in replay.edges:
        outgoing[p].append((c, a))
    rollout_results = defaultdict(lambda: dict(n=0, absolute_error=0.0, baseline_error=0.0, gain_edges=0))
    one_step_groups = defaultdict(lambda: dict(n=0, absolute_error=0.0))
    # Every real first edge is represented once. Longer paths choose a stable child,
    # not the child with the best oracle yield. No training on held-out questions.
    for parent, child, action in replay.edges:
        source = pack_observations([replay.nodes[parent]], learner.token_table, learner.device)
        state = learner.model.encoder(source)
        baseline = float(expected_acceptance(learner.model.acceptance(state), state.lengths)[0])
        first_label = replay.nodes[parent].accepted
        for depth in range(1, horizon+1):
            state = learner.model.dynamics(state,
                torch.tensor([int(action == "E")], device=learner.device), learner.extension_size)
            prediction = float(expected_acceptance(learner.model.acceptance(state), state.lengths)[0])
            label = replay.nodes[child].accepted
            if label is not None:
                record = rollout_results[f"h{depth}"]
                record["n"] += 1
                record["absolute_error"] += abs(prediction-label)
                record["baseline_error"] += abs(baseline-label)
                record["gain_edges"] += int(first_label is not None and label > first_label)
                record.setdefault("gain_comparable_paths", 0)
                record["gain_comparable_paths"] += int(first_label is not None)
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
    return dict(status="evaluated", questions=sorted({o.question for o in nodes}),
        label_coverage=dict(total=len(nodes), exact=int(current[0]), unresolved=len(nodes)-int(current[0])),
        current=dict(n=int(current[0]), mae=current[1]/current[0] if current[0] else None,
                     nll=current[2]/current[0] if current[0] else None),
        global_latent_std_mean=float(torch.cat(latent_globals).std(dim=0, unbiased=False).mean()),
        one_step_by_action_and_change={key: dict(n=v["n"], mae=v["absolute_error"]/v["n"])
                                      for key, v in one_step_groups.items()},
        rollout={key: dict(n=value["n"], mae=value["absolute_error"]/value["n"],
            unchanged_prediction_baseline_mae=value["baseline_error"]/value["n"],
            true_gain_paths=value["gain_edges"], gain_comparable_paths=value["gain_comparable_paths"])
            for key, value in rollout_results.items()},
        warning="10-question smoke checks the pipeline, not generalization or controller quality")


def action_probabilities(actions, args):
    weights = {"S": args.stop_weight, "E": args.extend_weight, "R": args.refine_weight}
    total = sum(weights[a] for a in actions)
    return {a: weights[a]/total for a in actions}


def explore_questions(args, questions, tokenizer, environment, learner, writer, summary):
    rng = random.Random(args.seed)
    train_replay = ExperienceReplay(args.replay_states, args.seed)
    validation_replay = ExperienceReplay(args.replay_states, args.seed+1)
    train_count = len(questions)-args.validation_questions
    initial = {name: parameter.detach().cpu().clone() for name, parameter in learner.model.named_parameters()}
    try:
        for question_index, question in enumerate(questions):
            split = "train" if question_index < train_count else "validation"
            question_id = str(question["question_id"])
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": question["prompt"]}], tokenize=True, add_generation_prompt=True)
            prefix = list(prompt)
            if len(prefix) > args.max_context_tokens:
                raise ValueError(f"Question {question_id} exceeds context cap; no silent truncation")
            generated = []
            labeler = HindsightLabeler(prefix, writer.label)
            buffer = train_replay if split == "train" else validation_replay
            print(f"[question] {question_index+1}/{len(questions)} {question_id} split={split}", flush=True)

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
                if parent is not None:
                    buffer.add(parent.observation, o, action)
                    train_available()
                else:
                    buffer.add_node(o)
                print(f"[state] {o.uid} action={action or 'root'} L={o.length} K={o.accepted}", flush=True)

            environment.on_state = on_state
            end_reason = "max_rounds_per_question"
            for round_id in range(args.max_rounds_per_question):
                remaining = args.max_new_tokens-len(generated)
                if remaining <= 0 or len(prefix) > args.max_context_tokens:
                    end_reason = "max_new_tokens" if remaining <= 0 else "max_context_tokens"
                    break
                state = environment.start(question_id, round_id, prefix, remaining)
                while True:
                    actions = environment.actions(state)
                    probabilities = action_probabilities(actions, args)
                    chosen = rng.choices(actions, weights=[probabilities[a] for a in actions])[0]
                    event = dict(state_id=state.observation.uid, question=question_id,
                                 round_id=round_id, split=split, action=chosen,
                                 legal_probabilities=probabilities)
                    if chosen == "S":
                        verifier_ms = environment.submit(state, remaining)
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
            if end_reason != "eos" and len(generated) >= args.max_new_tokens:
                end_reason = "max_new_tokens"
            coverage = labeler.finish(end_reason)
            for key, value in coverage.items():
                summary.setdefault("label_coverage", {}).setdefault(key, 0)
                summary["label_coverage"][key] += value
            summary["questions_completed"] += 1
            append_json(args.output_dir / "questions.jsonl", dict(**question, split=split,
                generated_tokens=generated, decoded=tokenizer.decode(generated),
                collection_end_reason=end_reason, label_coverage=coverage,
                generation_is_bounded_smoke_not_answer_accuracy_benchmark=True))
            writer.flush()
            checkpoint(learner, train_replay, args, args.output_dir, rng)
            summary.update(updates=learner.updates, dynamics_updates=learner.dynamics_updates,
                           nodes=writer.count, edges=writer.edge_count,
                           environment=environment.stats)
            atomic_json(args.output_dir / "summary.json", summary)
            if shutil.disk_usage(args.output_dir).free < 1024**3:
                raise OSError("Less than 1 GiB free; stop before corrupting artifacts")
        summary["evaluation"] = evaluate(learner, validation_replay, args.horizon)
        delta = sum(float((p.detach().cpu()-initial[n]).square().sum())
                    for n, p in learner.model.named_parameters()) ** 0.5
        summary["parameter_l2_change"] = delta
        if learner.updates == 0 or not np.isfinite(delta) or delta == 0:
            raise RuntimeError("Smoke did not actually update world-model weights")
    finally:
        writer.flush()
        checkpoint(learner, train_replay, args, args.output_dir, rng)
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


def run(args):
    archive = args.output_dir.with_suffix(".zip")
    if args.output_dir.exists() or archive.exists():
        raise FileExistsError("Use a new output directory; existing results will not be overwritten")
    args.output_dir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    atomic_json(args.output_dir / "config.json", config)
    summary = dict(schema="interactive_acceptance_pretrain_v2_hindsight", status="running",
        questions_completed=0, updates=0, nodes=0, edges=0,
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
        atomic_json(args.output_dir / "question_split.json", {
            "train": [q["question_id"] for q in questions[:-args.validation_questions]],
            "validation": [q["question_id"] for q in questions[-args.validation_questions:]]})
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
        if {str(p.device) for p in target.parameters()} != {f"cuda:{args.target_device}"}:
            raise RuntimeError("Verifier was not placed exclusively on its requested GPU")
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
        model = AcceptanceWorldModel(drafter.config.hidden_size, table.shape[-1],
            args.raw_top_k, dim=args.latent_dim, dropout=args.dropout)
        learner = WorldModelLearner(model, table, f"cuda:{args.drafter_device}", args.extend_size,
            args.learning_rate, warmup_updates=args.warmup_updates,
            horizon_warmup=args.horizon_warmup_updates)
        verifier = FullContextVerifier(target, tokenizer, args)
        environment = NativeTrainingEnvironment(NativeElysiaRunner(drafter, tokenizer, args),
            verifier, tokenizer.eos_token_id, drafter.config.hidden_size, args, None)
        writer = ExperienceWriter(args.output_dir)
        summary["trainable_parameters"] = sum(p.numel() for p in model.parameters())
        summary["devices"] = dict(verifier=f"cuda:{args.target_device}",
            drafter=f"cuda:{args.drafter_device}", world_model=f"cuda:{args.drafter_device}")
        explore_questions(args, questions, tokenizer, environment, learner, writer, summary)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="partial", error=f"{type(error).__name__}: {error}")
        (args.output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        summary["elapsed_seconds"] = time.perf_counter()-began
        for name in ("world_model_core.py", "world_model_environment.py", "world_model_hindsight.py",
                     "pretrain_acceptance_world_model.py", "native_elysia_graph.py",
                     "sparse_extend_world_model_collector.py", "Fast_dLLM_v2_1_5B/modeling.py"):
            source = Path(__file__).parent / name
            summary.setdefault("source_sha256", {})[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        package(args.output_dir, archive, summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--validation_questions", type=int, default=2)
    parser.add_argument("--max_rounds_per_question", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--max_proposal_tokens", type=int, default=64)
    parser.add_argument("--extend_size", type=int, default=8)
    parser.add_argument("--max_refinement_steps", type=int, default=3)
    parser.add_argument("--physical_block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.5)
    parser.add_argument("--hidden_layers", type=int, nargs=2, default=[14, 28])
    parser.add_argument("--raw_top_k", type=int, default=32)
    parser.add_argument("--stop_weight", type=float, default=1.0)
    parser.add_argument("--extend_weight", type=float, default=1.0)
    parser.add_argument("--refine_weight", type=float, default=1.0)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=1)
    parser.add_argument("--dllm_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    args.target_placement = "single"
    if not 0 < args.validation_questions < args.num_questions:
        parser.error("Need both training and validation questions")
    if args.target_device == args.drafter_device:
        parser.error("Drafter and verifier must be on distinct GPUs")
    for key in ("num_questions", "max_rounds_per_question", "max_new_tokens", "max_context_tokens",
                "extend_size", "batch_sequences", "horizon", "updates_per_transition", "raw_top_k",
                "physical_block_size", "small_block_size", "latent_dim"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
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
