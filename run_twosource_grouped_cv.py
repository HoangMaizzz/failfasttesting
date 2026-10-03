"""Question-grouped 5-fold CV for saved TwoSource world-model experiences.

This is an offline experiment: it does not invoke the drafter or verifier.  Each
fold starts from a fresh TwoSource model and trains only on four folds; every
state belonging to the remaining questions is held out for evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch

from world_model_core import Observation, ExperienceReplay, WorldModelLearner
from world_model_twosource import TwoSourceWorldModel, configure_losses
from world_model_training_audit import detailed_report, detailed_rows


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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    drafter = token_table = None
    input_summary, observations, edges, question_ids = load_experiences(args.run_dir)
    hidden_dim = int(observations[next(iter(observations))].hidden.shape[-1])
    top_k = int(observations[next(iter(observations))].gaps.shape[-1])
    print(f"[input] q={len(question_ids)} states={len(observations)} edges={len(edges)} "
          f"labeled={sum(o.accepted is not None for o in observations.values())}", flush=True)
    drafter, token_table, model_hidden_dim = fit_embedding_table(args.dllm_dir, args.device)
    if hidden_dim != model_hidden_dim:
        raise ValueError(f"Cached hidden dim {hidden_dim} != dLLM embedding model hidden dim {model_hidden_dim}")
    if max(int(o.ids.max()) for o in observations.values()) >= token_table.shape[0]:
        raise ValueError("Archive token IDs exceed frozen dLLM embedding vocabulary")
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
        summary.update(status="complete", fold_results=fold_results,
            learning_curve=all_curve, learning_curve_mean=curve_summary,
            pooled_oof=detailed_report(pooled_predictions),
            elapsed_seconds=time.perf_counter() - started,
            note="OOF predictions are generated by models that never trained on that question.")
        package(output, archive, summary)
        print("\n[complete]", json.dumps(summary["pooled_oof"]["groups"], indent=2), flush=True)
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
