"""Question-grouped 5-fold CV for saved TwoSource world-model experiences.

This is an offline experiment: it does not invoke the drafter or verifier.  Each
fold starts from a fresh TwoSource model and trains only on four folds; every
state belonging to the remaining questions is held out for evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch

from world_model_core import (Observation, ExperienceReplay, WorldModelLearner,
                              pack_observations)
from world_model_twosource import TwoSourceWorldModel, configure_losses
from world_model_training_audit import detailed_report, detailed_rows
from persistent_world_model_v1 import GatedFiLMAdapter, grouped_distribution_kl


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_experiences(root: Path):
    root = Path(root)
    required = ("summary.json", "states.jsonl", "edges.jsonl", "labels.jsonl",
                "teacher_targets.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Input is not an extracted TwoSource run; missing {missing}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "interactive_acceptance_two_source_v1":
        raise ValueError(f"Expected TwoSource ZIP/folder, got schema={summary.get('schema')!r}")
    if summary.get("status") != "complete":
        raise ValueError(f"Input run is not complete: status={summary.get('status')!r}")

    metadata_rows = read_jsonl(root / "states.jsonl")
    labels = {row["state_id"]: row for row in read_jsonl(root / "labels.jsonl")}
    teachers = {row["state_id"]: row for row in read_jsonl(root / "teacher_targets.jsonl")}
    edges = read_jsonl(root / "edges.jsonl")
    by_shard = defaultdict(list)
    for meta in metadata_rows:
        by_shard[meta["shard"]].append(meta)

    observations = {}
    for shard, rows in sorted(by_shard.items()):
        shard_path = root / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing experience shard: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as arrays:
            for meta in rows:
                row = int(meta["row"])
                lo, hi = map(int, arrays["offsets"][row:row + 2])

                def seq(name, dtype):
                    return torch.tensor(arrays[name][lo:hi], dtype=dtype)

                label = labels.get(meta["state_id"], {})
                accepted = (int(label["accepted_len"])
                            if label.get("label_valid") and label.get("accepted_len") is not None
                            else None)
                obs = Observation(
                    uid=meta["state_id"], question=str(meta["question"]),
                    round_id=int(meta["round_id"]), ids=seq("ids", torch.long),
                    hidden=seq("hidden", torch.float16), gaps=seq("gaps", torch.float16),
                    scalars=seq("scalars", torch.float32),
                    context=torch.tensor(arrays["context"][row], dtype=torch.float32),
                    accepted=accepted,
                    prefix_ids=torch.tensor(meta["prefix_token_ids"], dtype=torch.long),
                    topk_ids=seq("aligned_topk_token_ids", torch.long),
                    history=seq("history", torch.float32))

                teacher = teachers.get(meta["state_id"])
                if teacher is not None:
                    obs.teacher_margin = torch.tensor(teacher["margin"], dtype=torch.float32)
                    if teacher.get("features") is not None:
                        obs.teacher_features = torch.tensor(teacher["features"], dtype=torch.float32)
                    obs.teacher_is_actual = teacher.get("source") == "actual_STOP_forward"
                if "verifier_history_offsets" in arrays:
                    a, b = map(int, arrays["verifier_history_offsets"][row:row + 2])
                    obs.verifier_history = torch.tensor(
                        arrays["verifier_history"][a:b], dtype=torch.float32)
                if "teacher_aux_features" in arrays:
                    obs.teacher_aux_features = seq("teacher_aux_features", torch.float32)
                if all(name in arrays for name in ("teacher_topk_token_ids",
                        "teacher_topk_logits", "teacher_logsumexp")):
                    valid = arrays.get("teacher_distribution_valid")
                    if valid is not None:
                        valid = torch.tensor(valid[lo:hi], dtype=torch.bool)
                        obs.teacher_distribution_valid = valid
                        if bool(valid.any()):
                            obs.teacher_topk_ids = seq("teacher_topk_token_ids", torch.long)
                            obs.teacher_topk_logits = seq("teacher_topk_logits", torch.float16)
                            obs.teacher_logsumexp = seq("teacher_logsumexp", torch.float32)
                teacher_row = teachers.get(obs.uid, {})
                actual_rows = arrays.get("teacher_actual")
                obs.teacher_is_actual = bool(actual_rows[row]) if actual_rows is not None else (
                    teacher_row.get("source") == "actual_STOP_forward")
                observations[obs.uid] = obs

    question_ids = sorted({obs.question for obs in observations.values()})
    if len(question_ids) != int(summary.get("questions_completed", -1)):
        raise ValueError("Question count in states.jsonl does not match completed run summary")
    if len(question_ids) < 5:
        raise ValueError("At least five distinct questions are required for 5-fold CV")
    normalized_edges = []
    for edge in edges:
        parent, child = edge["parent"], edge["child"]
        if parent not in observations or child not in observations:
            raise ValueError(f"Broken edge references missing state: {edge}")
        if observations[parent].question != observations[child].question:
            raise ValueError(f"Cross-question edge would invalidate grouped CV: {edge}")
        normalized_edges.append((parent, child, edge["action"]))
    return summary, observations, normalized_edges, question_ids


def make_replay(observations, edges, question_ids, seed, sampling_mode="action_balanced"):
    selected = set(question_ids)
    nodes = {uid: obs for uid, obs in observations.items() if obs.question in selected}
    replay = ExperienceReplay(max(2, len(nodes) + 1), seed=seed,
                              sampling_mode=sampling_mode)
    for obs in nodes.values():
        replay.add_node(obs)
    for parent, child, action in edges:
        if parent in nodes and child in nodes:
            replay.add(nodes[parent], nodes[child], action)
    return replay


def grouped_folds(question_ids, folds=5, seed=42):
    if folds != 5:
        raise ValueError("This experiment is intentionally fixed to grouped 5-fold CV")
    shuffled = np.asarray(sorted(set(map(str, question_ids))), dtype=object)
    if len(shuffled) < folds:
        raise ValueError("Need at least one distinct question per fold")
    np.random.default_rng(seed).shuffle(shuffled)
    return [list(map(str, part)) for part in np.array_split(shuffled, folds)]


def fit_embedding_table(model_dir: Path, device: int):
    """Load only what the existing HF loader needs; retain frozen input table."""
    import transformers.modeling_rope_utils as rope_utils
    import transformers.modeling_utils as modeling_utils
    from transformers import AutoModelForCausalLM

    patched_rope = False
    old_tied = getattr(modeling_utils.PreTrainedModel,
                       "get_expanded_tied_weights_keys", None)
    if hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def rope_default(config, target_device, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 1000000.0)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                                  device=target_device) / dim))
            return inv, 1.0
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = rope_default
        patched_rope = True
    if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys"):
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}
    try:
        drafter = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float16, device_map={"": device},
            trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
        table = drafter.get_input_embeddings().weight.detach()
        hidden_dim = int(drafter.config.hidden_size)
        return drafter, table, hidden_dim
    finally:
        if patched_rope and "default" in rope_utils.ROPE_INIT_FUNCTIONS:
            del rope_utils.ROPE_INIT_FUNCTIONS["default"]
        if (hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys")
                and old_tied is not None):
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = old_tied


def save_predictions(path, rows):
    if Path(path).exists():
        Path(path).unlink()
    append_jsonl(path, rows)


@torch.no_grad()
def predict_child_latents(model, observations, edges, question_ids, token_table,
                          device, extension_size, batch_size=16):
    """Predict child z from parent observation + recorded R/E, without child inputs."""
    selected = set(question_ids)
    usable = [(parent, child, action) for parent, child, action in edges
              if observations[parent].question in selected
              and observations[child].question in selected]
    result = {}
    model.eval()
    for start in range(0, len(usable), batch_size):
        group = usable[start:start + batch_size]
        parents = [observations[parent] for parent, _, _ in group]
        packed = pack_observations(parents, token_table, device)
        current = model.encoder(packed)
        actions = torch.tensor([int(action == "E") for _, _, action in group],
                               dtype=torch.long, device=device)
        predicted = model.transition(current, actions, extension_size)
        for index, (_, child, _) in enumerate(group):
            # Multiple parents should not occur in this graph schema. Fail closed
            # rather than silently selecting a latent from a different path.
            if child in result:
                raise ValueError(f"State has multiple incoming transitions: {child}")
            result[child] = predicted.global_state[index].detach().float().cpu()
    return result


def make_film_examples(observations, question_ids, child_latents, hidden_slot,
                       vocab_size):
    selected = set(question_ids)
    examples = []
    for uid, obs in observations.items():
        if (obs.question not in selected or uid not in child_latents
                or obs.accepted is None or obs.teacher_topk_ids is None
                or obs.teacher_distribution_valid is None
                or obs.hidden.ndim != 3 or hidden_slot >= obs.hidden.shape[1]):
            continue
        # The teacher-forced suffix after the first rejection is not causally
        # useful for this proposal. Include the first rejected position itself.
        clean = min(obs.length, obs.accepted + int(obs.accepted < obs.length))
        valid = obs.teacher_distribution_valid.bool()
        for position in range(clean):
            ids = obs.teacher_topk_ids[position].long()
            logits = obs.teacher_topk_logits[position].float()
            normalizer = obs.teacher_logsumexp[position].float()
            if (not bool(valid[position]) or ids.numel() == 0
                    or not bool(torch.isfinite(logits).all())
                    or not bool(torch.isfinite(normalizer))
                    or int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
                continue
            examples.append(dict(hidden=obs.hidden[position, hidden_slot].clone(),
                latent=child_latents[uid].clone(), ids=ids.clone(),
                teacher_logits=obs.teacher_topk_logits[position].clone(),
                teacher_logsumexp=normalizer.clone(), question=obs.question,
                state_id=uid, position=position))
    return examples


def _select_film_eval(examples, limit, seed):
    if len(examples) <= limit:
        return list(examples)
    by_question = defaultdict(list)
    for example in examples:
        by_question[example["question"]].append(example)
    rng = random.Random(seed)
    groups = sorted(by_question)
    per_question = max(1, limit // max(1, len(groups)))
    selected, remaining = [], []
    for question in groups:
        rows = list(by_question[question])
        rng.shuffle(rows)
        selected.extend(rows[:per_question])
        remaining.extend(rows[per_question:])
    if len(selected) > limit:
        rng.shuffle(selected)
        return selected[:limit]
    rng.shuffle(remaining)
    return selected + remaining[:limit - len(selected)]


@torch.no_grad()
def evaluate_film(adapter, drafter, examples, device, batch_tokens=32,
                  max_examples=512, seed=31415):
    available = len(examples)
    examples = _select_film_eval(examples, max_examples, seed)
    if not examples:
        return dict(n=0, base_kl=None, film_kl=None, delta_kl=None,
                    base_top1_match=None, film_top1_match=None, argmax_flip_rate=None)
    adapter.eval()
    head = drafter.get_output_embeddings()
    per_question = defaultdict(lambda: dict(n=0, base_kl=0.0, film_kl=0.0,
                                           base_top1=0, film_top1=0, flips=0))
    total_base = total_film = 0.0
    base_matches = film_matches = flips = 0
    for start in range(0, len(examples), batch_tokens):
        group = examples[start:start + batch_tokens]
        head_dtype = next(head.parameters()).dtype
        hidden = torch.stack([x["hidden"] for x in group]).to(device=device, dtype=head_dtype)
        latent = torch.stack([x["latent"] for x in group]).to(device).float()
        ids = torch.stack([x["ids"] for x in group]).to(device).long()
        teacher = torch.stack([x["teacher_logits"] for x in group]).to(device).float()
        lse = torch.stack([x["teacher_logsumexp"] for x in group]).to(device).float()
        base = head(hidden)
        conditioned = head(adapter(hidden[:, None, :], latent)[:, 0])
        base_kl = grouped_distribution_kl(base, teacher, lse, ids)
        film_kl = grouped_distribution_kl(conditioned, teacher, lse, ids)
        base_top = base.argmax(-1)
        film_top = conditioned.argmax(-1)
        # argmax over top-K returns a support index, not a vocabulary token ID.
        teacher_top = ids.gather(-1, teacher.argmax(-1, keepdim=True)).squeeze(-1)
        match_base = base_top.eq(teacher_top)
        match_film = film_top.eq(teacher_top)
        changed = film_top.ne(base_top)
        total_base += float(base_kl.sum())
        total_film += float(film_kl.sum())
        base_matches += int(match_base.sum())
        film_matches += int(match_film.sum())
        flips += int(changed.sum())
        for i, example in enumerate(group):
            row = per_question[example["question"]]
            row["n"] += 1
            row["base_kl"] += float(base_kl[i])
            row["film_kl"] += float(film_kl[i])
            row["base_top1"] += int(match_base[i])
            row["film_top1"] += int(match_film[i])
            row["flips"] += int(changed[i])
    n = len(examples)
    question_rows = [dict(question=q, n=v["n"],
        base_kl=v["base_kl"] / v["n"], film_kl=v["film_kl"] / v["n"],
        delta_kl=(v["film_kl"] - v["base_kl"]) / v["n"],
        base_top1_match=v["base_top1"] / v["n"],
        film_top1_match=v["film_top1"] / v["n"],
        argmax_flip_rate=v["flips"] / v["n"])
        for q, v in sorted(per_question.items())]
    return dict(n=n, questions=len(question_rows), base_kl=total_base / n,
        film_kl=total_film / n, delta_kl=(total_film-total_base) / n,
        base_top1_match=base_matches / n, film_top1_match=film_matches / n,
        argmax_flip_rate=flips / n,
        question_macro_base_kl=float(np.mean([r["base_kl"] for r in question_rows])),
        question_macro_film_kl=float(np.mean([r["film_kl"] for r in question_rows])),
        per_question=question_rows, max_examples=max_examples,
        available_tokens=available,
        sampling="deterministic question-stratified sample" if n < available else "all usable tokens")


def train_film(adapter, drafter, examples, device, steps, batch_tokens,
               learning_rate, seed, log_prefix):
    if not examples:
        raise RuntimeError("FiLM cannot be evaluated: this fold has no causal verifier-distillation tokens")
    rng = random.Random(seed)
    adapter.train()
    drafter.eval().requires_grad_(False)
    head = drafter.get_output_embeddings()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    history = []
    for step in range(steps):
        group = [examples[rng.randrange(len(examples))] for _ in range(batch_tokens)]
        head_dtype = next(head.parameters()).dtype
        hidden = torch.stack([x["hidden"] for x in group]).to(device=device, dtype=head_dtype)
        latent = torch.stack([x["latent"] for x in group]).to(device).float()
        ids = torch.stack([x["ids"] for x in group]).to(device).long()
        teacher = torch.stack([x["teacher_logits"] for x in group]).to(device).float()
        lse = torch.stack([x["teacher_logsumexp"] for x in group]).to(device).float()
        with torch.no_grad():
            base = head(hidden)
        conditioned = head(adapter(hidden[:, None, :], latent)[:, 0])
        loss = grouped_distribution_kl(conditioned, teacher, lse, ids,
                                       preserve_logits=base).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(f"[{log_prefix}] step={step+1}/{steps} grouped_KL={history[-1]:.5f} "
                  f"tokens={len(examples)}", flush=True)
    adapter.eval()
    return dict(steps=steps, train_tokens=len(examples), first_train_loss=history[0],
                final_train_loss=history[-1], trace=history)


def _bootstrap_mean_ci(values, seed=42, draws=3000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=None, ci95=[None, None])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),
                ci95=[float(x) for x in np.quantile(sampled, [0.025, 0.975])])


def real_verifier_film_test(args, run_dir, output, folds, model_hidden_dim, top_k):
    """Paired, held-out continuation test: same seed prefix, adapter off/on."""
    from types import SimpleNamespace
    from sparse_extend_world_model_collector import _load_models
    from native_elysia_graph import NativeElysiaRunner, NativeEosWithoutSnapshot
    from world_model_teacher_environment import TeacherVerifier, TeacherTrainingEnvironment

    config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    ev = SimpleNamespace(target_model_name=config.get(
            "target_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        target_device=args.target_device, drafter_device=args.device,
        target_gpu_memory_gib=args.target_gpu_memory_gib, target_placement="auto",
        dllm_dir=str(args.dllm_dir), raw_top_k=top_k,
        hidden_layers=config.get("hidden_layers", [7, 14, 28]),
        physical_block_size=int(config.get("physical_block_size", 32)),
        small_block_size=int(config.get("small_block_size", 8)),
        drafter_threshold=float(config.get("drafter_threshold", 0.5)),
        max_refinement_steps=3, extend_size=8, max_proposal_tokens=64,
        capture_verifier_teacher=True, verifier_teacher_top_k=top_k,
        max_context_tokens=4096, output_dir=str(output))
    tokenizer, target, drafter = _load_models(ev)
    target.eval().requires_grad_(False)
    drafter.eval().requires_grad_(False)
    device = torch.device(f"cuda:{args.device}")
    table = drafter.get_input_embeddings().weight.detach()
    verifier = TeacherVerifier(target, tokenizer, ev)
    runner = NativeElysiaRunner(drafter, tokenizer, ev)
    environment = TeacherTrainingEnvironment(runner, verifier, tokenizer.eos_token_id,
        model_hidden_dim, ev, lambda *_: None, token_table=table)

    question_rows = {str(row["question_id"]): row for row in read_jsonl(
        Path(run_dir) / "questions.jsonl") if row.get("prompt")}
    if not question_rows:
        raise FileNotFoundError("No saved question prompts in questions.jsonl; cannot run paired real-verifier test")
    if any(not any(q in question_rows for q in fold) for fold in folds):
        raise ValueError("At least one heldout fold has no question prompt matching states.jsonl IDs")
    per_fold = []
    selected_pairs = []
    verifier_calls = 0
    for fold_id, fold in enumerate(folds, start=1):
        choices = [q for q in fold if q in question_rows]
        chooser = random.Random(args.seed + fold_id * 31337)
        chooser.shuffle(choices)
        chosen = choices[:args.real_film_questions_per_fold]
        per_fold.append(dict(fold=fold_id, heldout_questions=len(fold),
                             selected_questions=chosen))
        if not chosen:
            continue
        checkpoint = torch.load(Path(output) / f"fold_{fold_id}" / "model.pt",
                                map_location="cpu", weights_only=False)
        cfg = checkpoint["model_config"]
        fold_model = TwoSourceWorldModel(hidden_dim=cfg["hidden_dim"],
            token_dim=cfg["token_dim"], top_k=cfg["top_k"], dim=cfg["dim"],
            num_hidden_layers=cfg["num_hidden_layers"], dropout=cfg["dropout"],
            architecture=cfg.get("architecture", "two_source"),
            variant=cfg.get("variant", "full")).to(device)
        fold_model.load_state_dict(checkpoint["model"])
        fold_model.eval()
        film_state = torch.load(Path(output) / f"fold_{fold_id}" / "film_adapter.pt",
                                map_location="cpu", weights_only=False)
        adapter = GatedFiLMAdapter(film_state["hidden_dim"],
            film_state["latent_dim"]).to(device)
        adapter.load_state_dict(film_state["adapter"])
        adapter.eval()

        for qid in chosen:
            question = question_rows[qid]
            prompt = question["prompt"]
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True)
            environment.reset_history()
            drafter._persistent_world_film_adapter = None
            drafter._persistent_world_latent = None
            try:
                first = environment.start(qid, 0, list(prompt_ids), 9)
            except NativeEosWithoutSnapshot as eos:
                accepted, _, emitted, verifier_ms = verifier.score(
                    list(prompt_ids), eos.candidate_token_ids, 9)
                verifier_calls += 1
                selected_pairs.append(dict(question_id=qid, fold=fold_id, paired=False,
                    excluded_reason="seed_eos_without_snapshot", seed_accepted=int(accepted),
                    seed_verifier_ms=float(verifier_ms), emitted_token_ids=emitted))
                continue
            environment.submit(first, 9)
            verifier_calls += 1
            if tokenizer.eos_token_id in first.emitted:
                selected_pairs.append(dict(question_id=qid, fold=fold_id, paired=False,
                    excluded_reason="seed_verifier_emitted_eos",
                    seed_accepted=int(first.observation.accepted),
                    emitted_token_ids=first.emitted))
                continue

            common_prefix = list(prompt_ids) + list(first.emitted)
            common_history = [row.detach().cpu().clone()
                              for row in environment.verifier_history[-8:]]
            conditioned_obs = copy.copy(first.observation)
            conditioned_obs.prefix_ids = torch.tensor(common_prefix, dtype=torch.long)
            conditioned_obs.verifier_history = (torch.stack(common_history)
                if common_history else torch.empty(0, 48))
            batch = pack_observations([conditioned_obs], table, device)
            with torch.no_grad():
                latent = fold_model.encoder(batch).global_state.detach()

            order = ["baseline", "film"]
            random.Random(f"{args.seed}/{qid}/arm-order").shuffle(order)
            pair = {}
            for arm in order:
                environment.verifier_history = [row.clone() for row in common_history]
                if arm == "film":
                    drafter._persistent_world_film_adapter = adapter
                    drafter._persistent_world_latent = latent
                else:
                    drafter._persistent_world_film_adapter = None
                    drafter._persistent_world_latent = None
                began = time.perf_counter()
                try:
                    state = environment.start(qid, 1, common_prefix, 9)
                    generation_ms = (time.perf_counter() - began) * 1000
                    proposal = state.observation.ids[:, 1].tolist()
                    accepted, _, emitted, verifier_ms = verifier.score(
                        common_prefix, proposal, 9)
                    verifier_calls += 1
                    pair[arm] = dict(accepted=int(accepted), proposal_length=len(proposal),
                        acceptance_fraction=int(accepted) / max(1, len(proposal)),
                        proposal_token_ids=proposal, emitted_token_ids=emitted,
                        generation_wall_ms=generation_ms, verifier_ms=float(verifier_ms),
                        total_measured_ms=generation_ms + float(verifier_ms),
                        snapshot_available=True,
                        unmask_forward_index=int(state.snapshot.get("unmask_forward_index", -1)))
                except NativeEosWithoutSnapshot as eos:
                    accepted, _, emitted, verifier_ms = verifier.score(
                        common_prefix, eos.candidate_token_ids, 9)
                    verifier_calls += 1
                    pair[arm] = dict(accepted=int(accepted),
                        proposal_length=len(eos.candidate_token_ids),
                        acceptance_fraction=int(accepted) / max(1, len(eos.candidate_token_ids)),
                        proposal_token_ids=eos.candidate_token_ids, emitted_token_ids=emitted,
                        generation_wall_ms=(time.perf_counter() - began) * 1000,
                        verifier_ms=float(verifier_ms), snapshot_available=False,
                        termination_reason="native_eos_without_snapshot")
            selected_pairs.append(dict(question_id=qid, fold=fold_id, paired=True,
                same_prefix=True, same_world_model_checkpoint=True,
                seed_accepted=int(first.observation.accepted),
                seed_proposal_length=first.observation.length,
                baseline=pair["baseline"], film=pair["film"],
                accepted_delta=pair["film"]["accepted"]-pair["baseline"]["accepted"],
                proposal_same=pair["film"]["proposal_token_ids"] == pair["baseline"]["proposal_token_ids"],
                arm_order=order))
            drafter._persistent_world_film_adapter = None
            drafter._persistent_world_latent = None
            print(f"[film-real fold={fold_id}] {qid}: accepted "
                  f"{pair['baseline']['accepted']} -> {pair['film']['accepted']}; "
                  f"generation_ms {pair['baseline']['generation_wall_ms']:.1f} -> "
                  f"{pair['film']['generation_wall_ms']:.1f}", flush=True)
        del adapter, fold_model, checkpoint, film_state
        gc.collect(); torch.cuda.empty_cache()

    paired = [row for row in selected_pairs if row.get("paired")]
    deltas = [row["accepted_delta"] for row in paired]
    result = dict(status="complete", protocol=(
        "5-fold OOF models; each heldout question gets a seed native draft and real verifier STOP, "
        "then baseline/FiLM generate from the exact same verifier-emitted prefix. FiLM latent is "
        "encoded from the seed observation plus the now-observed verifier-memory row. Arm order is randomized."),
        real_questions_per_fold=args.real_film_questions_per_fold,
        selected_questions_per_fold=per_fold, questions_seen=len(selected_pairs),
        paired_questions=len(paired), excluded_questions=len(selected_pairs)-len(paired),
        verifier_calls=verifier_calls,
        native_drafter_calls=int(environment.stats.get("native_calls", 0)),
        baseline_mean_accepted=(None if not paired else float(np.mean(
            [row["baseline"]["accepted"] for row in paired]))),
        film_mean_accepted=(None if not paired else float(np.mean(
            [row["film"]["accepted"] for row in paired]))),
        paired_delta_accepted=_bootstrap_mean_ci(deltas, args.seed + 712),
        baseline_mean_acceptance_fraction=(None if not paired else float(np.mean(
            [row["baseline"]["acceptance_fraction"] for row in paired]))),
        film_mean_acceptance_fraction=(None if not paired else float(np.mean(
            [row["film"]["acceptance_fraction"] for row in paired]))),
        baseline_generation_wall_ms_median=(None if not paired else float(np.median(
            [row["baseline"]["generation_wall_ms"] for row in paired]))),
        film_generation_wall_ms_median=(None if not paired else float(np.median(
            [row["film"]["generation_wall_ms"] for row in paired]))),
        baseline_verifier_ms_median=(None if not paired else float(np.median(
            [row["baseline"]["verifier_ms"] for row in paired]))),
        film_verifier_ms_median=(None if not paired else float(np.median(
            [row["film"]["verifier_ms"] for row in paired]))),
        proposal_changed_rate=(None if not paired else float(np.mean(
            [not row["proposal_same"] for row in paired]))),
        per_question=selected_pairs,
        caveat="Small paired continuation test, not a full-answer benchmark; generation wall time includes Python/native collector overhead, while verifier latency is timed separately.")
    return result


def package(output: Path, archive: Path, summary):
    atomic_json(output / "cv_summary.json", summary)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != temporary and not path.name.endswith(".tmp"):
                zf.write(path, path.relative_to(output),
                         compress_type=(zipfile.ZIP_STORED if path.suffix == ".npz"
                                        else zipfile.ZIP_DEFLATED))
    with zipfile.ZipFile(temporary) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Result ZIP failed CRC validation: {bad}")
    temporary.replace(archive)
    print(f"[package] {summary['status']} {archive.name} ({archive.stat().st_size / 2**20:.1f} MiB)",
          flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Extracted TwoSource run folder; no input ZIP is required")
    parser.add_argument("--dllm_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 20, 40, 60, 80])
    parser.add_argument("--updates_per_question", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--roots_per_question", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--extension_size", type=int, default=8)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--target_gpu_memory_gib", type=int, default=8)
    parser.add_argument("--film_steps", type=int, default=300)
    parser.add_argument("--film_batch_tokens", type=int, default=32)
    parser.add_argument("--film_learning_rate", type=float, default=1e-4)
    parser.add_argument("--film_eval_tokens", type=int, default=512)
    parser.add_argument("--real_film_questions_per_fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args):
    if args.folds != 5:
        raise ValueError("This experiment is intentionally fixed to grouped 5-fold CV")
    if sorted(args.milestones) != list(args.milestones) or any(x < 1 for x in args.milestones):
        raise ValueError("Milestones must be increasing positive question counts")
    if args.milestones[-1] > 0 and args.milestones[-1] != 80:
        raise ValueError("For 100 questions / 5 folds, the final training milestone must be 80")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.device} is unavailable")
    if args.target_device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA target device {args.target_device} is unavailable")
    if args.film_steps < 1 or args.film_batch_tokens < 1 or args.film_eval_tokens < 1:
        raise ValueError("FiLM training/evaluation sizes must be positive")
    if args.real_film_questions_per_fold < 1:
        raise ValueError("Need at least one paired real-verifier question per fold")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    drafter = token_table = None
    input_summary, observations, edges, question_ids = load_experiences(args.run_dir)
    hidden_dim = int(observations[next(iter(observations))].hidden.shape[-1])
    top_k = int(observations[next(iter(observations))].gaps.shape[-1])
    archive_config_path = Path(args.run_dir) / "config.json"
    archive_config = (json.loads(archive_config_path.read_text(encoding="utf-8"))
                      if archive_config_path.is_file() else {})
    hidden_layers = archive_config.get("hidden_layers", [7, 14, 28])
    if len(hidden_layers) != observations[next(iter(observations))].hidden.shape[1]:
        raise ValueError("Archive hidden_layers metadata does not match cached hidden feature slots")
    final_hidden_slot = max(range(len(hidden_layers)), key=lambda i: hidden_layers[i])
    print(f"[input] q={len(question_ids)} states={len(observations)} edges={len(edges)} "
          f"labeled={sum(o.accepted is not None for o in observations.values())} "
          f"FiLM_topK_states={sum(o.teacher_distribution_valid is not None and bool(o.teacher_distribution_valid.any()) for o in observations.values())}", flush=True)
    drafter, token_table, model_hidden_dim = fit_embedding_table(args.dllm_dir, args.device)
    if hidden_dim != model_hidden_dim:
        raise ValueError(f"Cached hidden dim {hidden_dim} != dLLM embedding model hidden dim {model_hidden_dim}")
    if max(int(o.ids.max()) for o in observations.values()) >= token_table.shape[0]:
        raise ValueError("Archive token IDs exceed frozen dLLM embedding vocabulary")
    distribution_states = sum(o.teacher_distribution_valid is not None
        and bool(o.teacher_distribution_valid.any()) for o in observations.values())
    if distribution_states == 0:
        raise ValueError("Input archive has no saved verifier top-K distributions; FiLM cannot be tested without rerunning collection")
    torch.set_float32_matmul_precision("high")
    folds = grouped_folds(question_ids, args.folds, args.seed)
    if any(not part for part in folds):
        raise ValueError("A CV fold is empty")

    fold_results, pooled_predictions = [], []
    summary = dict(schema="twosource_grouped_5fold_cv_v1", status="running",
        input_schema=input_summary["schema"], input_questions=len(question_ids),
        states=len(observations), edges=len(edges), folds=[], milestones=args.milestones,
        updates_per_new_training_question=args.updates_per_question, horizon=args.horizon,
        split_unit="whole question; no states/edges from heldout question enter training",
        training="fresh TwoSource model per fold; original checkpoint is NOT loaded",
        verifier_calls=0, drafter_calls=0,
        film=dict(status="pending", method="fold-specific GatedFiLM trained only on that fold's train questions; heldout top-K verifier distributions evaluate before/after; small paired actual-verifier continuation test afterwards"),
        caveat="CV evaluates heldout states from the saved adaptive collection policy, not newly sampled deployment states.")
    output = args.output_dir
    archive = output.with_suffix(".zip")
    all_curve = []
    try:
        for fold_id, heldout_questions in enumerate(folds):
            train_questions = [q for q in question_ids if q not in set(heldout_questions)]
            order_rng = random.Random(args.seed + 1009 * fold_id)
            order_rng.shuffle(train_questions)
            if args.milestones[-1] != len(train_questions):
                raise ValueError(f"Fold {fold_id} has {len(train_questions)} train questions, "
                                 f"but final milestone is {args.milestones[-1]}")
            print(f"\n[fold {fold_id + 1}/{args.folds}] train_q={len(train_questions)} "
                  f"validation_q={len(heldout_questions)}", flush=True)
            seed = args.seed + fold_id * 7919
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = TwoSourceWorldModel(hidden_dim, token_table.shape[-1], top_k,
                dim=args.latent_dim, num_hidden_layers=3, dropout=args.dropout)
            learner = WorldModelLearner(model, token_table, f"cuda:{args.device}",
                args.extension_size, args.learning_rate, warmup_updates=16,
                horizon_warmup=64)
            configure_losses(learner, "full")
            train_replay = ExperienceReplay(max(2, len(observations) + 1), seed=seed,
                                           sampling_mode="action_balanced")
            val_replay = make_replay(observations, edges, heldout_questions,
                                     seed=seed + 1, sampling_mode="natural")
            val_ids = [uid for uid, obs in val_replay.nodes.items() if obs.accepted is not None]
            val_replay.audit_roots = val_ids
            roots_by_question = defaultdict(list)
            outgoing_parents = {parent for parent, _, _ in edges}
            for uid in val_ids:
                if uid in outgoing_parents:
                    roots_by_question[val_replay.nodes[uid].question].append(uid)
            rollout_roots = []
            for q in sorted(roots_by_question):
                choices = sorted(roots_by_question[q])
                rr = random.Random(f"{args.seed}/{fold_id}/{q}/rollout")
                rr.shuffle(choices)
                rollout_roots.extend(choices[:args.roots_per_question])

            prior_q = 0
            fold_curve = []
            final_rows = None
            for milestone in args.milestones:
                new_qs = train_questions[prior_q:milestone]
                qset = set(new_qs)
                new_nodes = [obs for obs in observations.values() if obs.question in qset]
                for obs in new_nodes:
                    train_replay.add_node(obs)
                added_ids = {obs.uid for obs in new_nodes}
                for parent, child, action in edges:
                    if parent in added_ids and child in added_ids:
                        train_replay.add(observations[parent], observations[child], action)
                updates = (milestone - prior_q) * args.updates_per_question
                for update_i in range(updates):
                    metrics = learner.update(train_replay, args.batch_size, args.horizon)
                    if metrics is None:
                        raise RuntimeError("Training replay produced no supervised update")
                    if (update_i + 1) % 100 == 0 or update_i + 1 == updates:
                        print(f"[fold {fold_id + 1}] q={milestone}/{len(train_questions)} "
                              f"update={learner.updates} loss={metrics['loss']:.4f}", flush=True)
                prior_q = milestone

                val_replay.audit_roots = val_ids
                current = detailed_rows(learner, val_replay, horizon=0)
                rollout_replay = val_replay
                rollout_replay.audit_roots = rollout_roots
                future = detailed_rows(learner, rollout_replay, horizon=args.horizon)
                rows = current + [row for row in future if row["horizon"] > 0]
                report = detailed_report(rows)
                curve_row = dict(fold=fold_id, train_questions=milestone,
                    updates=learner.updates, validation_questions=len(heldout_questions),
                    report=report)
                fold_curve.append(curve_row); all_curve.append(curve_row)
                print(f"[validation fold={fold_id + 1} train_q={milestone}] " +
                      " ".join(f"{h}_MAE={report['groups'].get(h, {}).get('mae')}"
                               for h in ("h0", "h1", "h2", "h3")), flush=True)
                if milestone == args.milestones[-1]:
                    final_rows = rows
                atomic_json(output / f"fold_{fold_id + 1}" / f"learning_{milestone:03d}.json",
                            curve_row)
                # Partial, CRC-checked artifact remains available between folds.
                summary["folds"] = [dict(fold=i + 1, validation_questions=folds[i],
                    completed_milestones=[x["train_questions"] for x in fold_curve]
                    if i == fold_id else [x["train_questions"] for x in fold_results[i]["learning_curve"]])
                    for i in range(fold_id + 1)]
                summary["learning_curve"] = all_curve
                summary["completed_folds"] = len(fold_results)
                package(output, archive, summary)

            fold_summary = dict(fold=fold_id + 1, validation_questions=heldout_questions,
                train_questions=len(train_questions), updates=learner.updates,
                learning_curve=fold_curve)
            train_latents = predict_child_latents(model, observations, edges,
                train_questions, token_table, learner.device, args.extension_size)
            valid_latents = predict_child_latents(model, observations, edges,
                heldout_questions, token_table, learner.device, args.extension_size)
            train_examples = make_film_examples(observations, train_questions,
                train_latents, final_hidden_slot, token_table.shape[0])
            valid_examples = make_film_examples(observations, heldout_questions,
                valid_latents, final_hidden_slot, token_table.shape[0])
            torch.manual_seed(seed + 97)
            adapter = GatedFiLMAdapter(model_hidden_dim, args.latent_dim).to(
                f"cuda:{args.device}")
            film_before = evaluate_film(adapter, drafter, valid_examples,
                f"cuda:{args.device}", args.film_batch_tokens,
                args.film_eval_tokens, args.seed + fold_id)
            film_fit = train_film(adapter, drafter, train_examples,
                f"cuda:{args.device}", args.film_steps, args.film_batch_tokens,
                args.film_learning_rate, seed + 71,
                f"film fold {fold_id + 1}")
            film_after = evaluate_film(adapter, drafter, valid_examples,
                f"cuda:{args.device}", args.film_batch_tokens,
                args.film_eval_tokens, args.seed + fold_id)
            film_result = dict(fold=fold_id + 1,
                train_questions=len(train_questions),
                validation_questions=len(heldout_questions),
                train_examples=len(train_examples), validation_examples=len(valid_examples),
                hidden_layer=int(hidden_layers[final_hidden_slot]),
                target="saved verifier top-K logits + exact OTHER bucket; only positions through first mismatch",
                conditioning="world-model predicted child latent from parent observation and real R/E action; no child observation or teacher labels enter z",
                before=film_before, after=film_after,
                delta_kl=(None if film_before["film_kl"] is None or film_after["film_kl"] is None
                          else film_after["film_kl"]-film_before["film_kl"]),
                training=film_fit)
            atomic_json(output / f"fold_{fold_id + 1}" / "film_metrics.json", film_result)
            torch.save(dict(adapter={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                hidden_dim=model_hidden_dim, latent_dim=args.latent_dim,
                injection="final_norm_output_before_lm_head", fold=fold_id + 1,
                training_questions=train_questions, validation_questions=heldout_questions,
                seed=seed + 97), output / f"fold_{fold_id + 1}" / "film_adapter.pt")
            fold_summary["film"] = dict(train_examples=len(train_examples),
                validation_examples=len(valid_examples), before=film_before,
                after=film_after, delta_kl=film_result["delta_kl"])
            del adapter, train_latents, valid_latents, train_examples, valid_examples
            gc.collect(); torch.cuda.empty_cache()
            fold_results.append(fold_summary)
            append_jsonl(output / "oof_predictions.jsonl", final_rows or [])
            pooled_predictions.extend(final_rows or [])
            torch.save(dict(model_config=model.config,
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                fold=fold_id + 1, training_questions=train_questions,
                validation_questions=heldout_questions, updates=learner.updates),
                output / f"fold_{fold_id + 1}" / "model.pt")
            del learner, model, train_replay, val_replay
            gc.collect(); torch.cuda.empty_cache()
            summary["completed_folds"] = len(fold_results)
            summary["fold_results"] = fold_results
            summary["pooled_oof"] = detailed_report(pooled_predictions)
            summary["film"]["status"] = "offline_oof_complete"
            summary["film"]["fold_metrics"] = [row["film"] for row in fold_results]
            summary["elapsed_seconds"] = time.perf_counter() - started
            package(output, archive, summary)

        # Fold-averaged learning curves are grouped by training-question count.
        curve_summary = {}
        for milestone in args.milestones:
            rows = [r for r in all_curve if r["train_questions"] == milestone]
            means = {}
            for horizon in ("h0", "h1", "h2", "h3"):
                values = [r["report"]["groups"].get(horizon, {}).get("question_macro_mae")
                          for r in rows]
                values = [float(value) for value in values if value is not None and np.isfinite(value)]
                means[horizon] = float(np.mean(values)) if values else None
            curve_summary[str(milestone)] = dict(folds=len(rows),
                mean_question_macro_mae=means)

        # The paired test invokes native dLLM and the actual Qwen verifier only
        # after all offline folds finish. Release the embedding-only dLLM first
        # so the two-model evaluation fits a T4x2 runtime more safely.
        del token_table, drafter
        token_table = drafter = None
        gc.collect(); torch.cuda.empty_cache()
        print("\nStarting heldout paired FiLM test with real verifier; "
              "two continuation verifier calls per selected question.", flush=True)
        real_film = real_verifier_film_test(args, args.run_dir, output, folds,
            model_hidden_dim, top_k)
        if real_film["paired_questions"] == 0:
            raise RuntimeError("Real-verifier FiLM comparison produced no paired questions; inspect exclusions in the partial ZIP")
        atomic_json(output / "film_real_verifier.json", real_film)
        film_folds = [row["film"] for row in fold_results]
        usable = [row for row in film_folds if row["after"]["n"]]
        film_oof = dict(folds=len(usable),
            validation_tokens=sum(row["after"]["n"] for row in usable),
            mean_fold_base_kl=(None if not usable else float(np.mean(
                [row["before"]["base_kl"] for row in usable]))),
            mean_fold_film_kl_before=(None if not usable else float(np.mean(
                [row["before"]["film_kl"] for row in usable]))),
            mean_fold_film_kl_after=(None if not usable else float(np.mean(
                [row["after"]["film_kl"] for row in usable]))),
            mean_fold_delta_kl=(None if not usable else float(np.mean(
                [row["delta_kl"] for row in usable if row["delta_kl"] is not None]))),
            mean_fold_top1_match_before=(None if not usable else float(np.mean(
                [row["before"]["base_top1_match"] for row in usable]))),
            mean_fold_top1_match_after=(None if not usable else float(np.mean(
                [row["after"]["film_top1_match"] for row in usable]))))
        summary.update(status="complete", fold_results=fold_results,
            learning_curve=all_curve, learning_curve_mean=curve_summary,
            pooled_oof=detailed_report(pooled_predictions),
            verifier_calls=real_film["verifier_calls"],
            drafter_calls=real_film["native_drafter_calls"],
            film=dict(status="complete", offline_oof=film_oof,
                      real_verifier=real_film,
                      steps_per_fold=args.film_steps,
                      real_pairs_requested=args.real_film_questions_per_fold * args.folds),
            elapsed_seconds=time.perf_counter() - started,
            note="OOF world-model and FiLM metrics are generated by fold-specific models that never trained on the heldout question; the paired real-verifier test is a small continuation experiment, not a full-answer benchmark.")
        package(output, archive, summary)
        print("\n[complete]", json.dumps(dict(world_model=summary["pooled_oof"]["groups"],
            film=summary["film"]), indent=2), flush=True)
    except BaseException as exc:
        summary.update(status="partial", error=f"{type(exc).__name__}: {exc}",
                       completed_folds=len(fold_results), learning_curve=all_curve,
                       fold_results=fold_results,
                       elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "error.txt", {"error": summary["error"]})
        package(output, archive, summary)
        raise
    finally:
        del observations, edges
        if token_table is not None:
            del token_table
        if drafter is not None:
            del drafter
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    run(parse_args())
