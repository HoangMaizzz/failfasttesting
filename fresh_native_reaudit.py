"""Retrain corrected, action-conditioned H1-H3 predictors from a native ZIP."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from fresh_native_drafter_test import load_capture, _action_metrics
from latent_sufficiency_audit import (
    ActionDirectPredictor, LatentDynamics, _collect_direct, _collect_rollout,
    _fit_direct, _fit_dynamics, _paths_for_edges, _restrict_edges, _seed,
    normalize_states, question_split,
)


def source_folder(source, output):
    if source.is_dir():
        roots = [p.parent for p in source.rglob("capture_manifest.json")
                 if (p.parent / "states.jsonl").is_file()]
        if len(roots) != 1:
            raise ValueError(f"Expected one fresh-native capture directory, found {roots}")
        return roots[0]
    root = output / "source_capture"
    root.mkdir()
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"Archive member outside capture root: {name}")
            if Path(name).name in ("states.jsonl", "edges.jsonl", "capture_manifest.json") or name.endswith(".npz"):
                archive.extract(name, root)
    return source_folder(root, output)


def label_audit(root, states, edges):
    stored = {}
    metadata = [json.loads(s) for s in (root / "states.jsonl").read_text().splitlines()]
    shards = {}
    for row in metadata:
        if row["shard"] not in shards:
            with np.load(root / row["shard"], allow_pickle=False) as arrays:
                shards[row["shard"]] = arrays["changed"].copy()
        stored[row["state_id"]] = shards[row["shard"]][row["row"]]
    result = Counter()
    for edge in edges:
        if edge.action != 0:
            continue
        valid = edge.changed_valid
        truth = states[edge.dst].changed[valid]
        old = stored[edge.dst][valid]
        result["positions"] += int(valid.sum())
        result["raw_id_positive_labels"] += int(truth.sum())
        result["original_stored_positive_labels"] += int(old.sum())
        result["labels_repaired"] += int(np.count_nonzero(truth != old))
    return dict(result)


def baselines(train_edges, test_edges, states, metadata):
    output, models = {}, {}
    for action, name in ((0, "R"), (1, "E")):
        train = [e for e in train_edges if e.action == action]
        buckets = defaultdict(list)
        for edge in train:
            for pos in np.flatnonzero(edge.valid):
                buckets[(int(pos), metadata[edge.src]["refine_index"])].append(
                    (float(states[edge.dst].mask[pos]), float(states[edge.dst].confidence[pos])))
        values = [v for rows in buckets.values() for v in rows]
        fallback = (float(np.mean([v[0] for v in values])), float(np.median([v[1] for v in values])))
        table = {key: (float(np.mean([v[0] for v in rows])), float(np.median([v[1] for v in rows])))
                 for key, rows in buckets.items()}
        content = np.concatenate([states[e.dst].content[e.valid] for e in train if e.valid.any()])
        mean_content = content.mean(0)
        changes = np.concatenate([states[e.dst].changed[e.changed_valid]
                                  for e in train if e.changed_valid.any()]) if action == 0 else np.array([])
        change_prior = float(changes.mean()) if changes.size else 0.0
        models[name] = (table, fallback, mean_content, change_prior)
        pm, tm, pc, tc, cos, prev = [], [], [], [], [], []
        for edge in test_edges:
            if edge.action != action:
                continue
            for pos in np.flatnonzero(edge.valid):
                p_mask, p_conf = table.get((int(pos), metadata[edge.src]["refine_index"]), fallback)
                pm.append(p_mask); pc.append(p_conf)
                tm.append(states[edge.dst].mask[pos]); tc.append(states[edge.dst].confidence[pos])
                target = states[edge.dst].content[pos]
                def similarity(x):
                    return float(np.dot(x, target) / max(float(np.linalg.norm(x) * np.linalg.norm(target)), 1e-8))
                cos.append(similarity(mean_content)); prev.append(similarity(states[edge.src].content[pos]))
        output[name] = {
            "position_refine_mask_brier": float(np.mean((np.asarray(pm) - tm) ** 2)),
            "position_refine_confidence_mae": float(np.mean(abs(np.asarray(pc) - tc))),
            "train_mean_content_cosine": float(np.mean(cos)),
            "previous_candidate_content_cosine": float(np.mean(prev)),
            "train_change_prevalence": change_prior,
        }
    return output, models


def save_predictions(path, records, states, metadata, baseline_models):
    with path.open("w", encoding="utf-8") as handle:
        for edge, prediction in records:
            target = states[edge.dst]
            valid, cv = edge.valid, edge.changed_valid
            name = "R" if edge.action == 0 else "E"
            table, fallback, mean_content, change_prior = baseline_models[name]
            p_mask = 1 / (1 + np.exp(-np.clip(prediction["mask"], -40, 40)))
            p_change = 1 / (1 + np.exp(-np.clip(prediction["changed"], -40, 40)))
            pc, tc = prediction["content"], target.content
            cosine = (pc * tc).sum(-1) / np.maximum(np.linalg.norm(pc, axis=-1) * np.linalg.norm(tc, axis=-1), 1e-8)
            base = [table.get((i, metadata[edge.src]["refine_index"]), fallback) for i in range(len(valid))]
            row = dict(src=edge.src, dst=edge.dst, question_id=target.qid, action=name,
                       valid=valid.tolist(), changed_valid=cv.tolist(),
                       true_mask=target.mask.tolist(), predicted_mask=p_mask.tolist(),
                       true_confidence=target.confidence.tolist(), predicted_confidence=prediction["confidence"].tolist(),
                       true_changed=target.changed.tolist(), predicted_changed=p_change.tolist(),
                       content_cosine=cosine.tolist(),
                       baseline_mask=[v[0] for v in base], baseline_confidence=[v[1] for v in base],
                       baseline_change=change_prior)
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def run(args):
    started = time.time()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    root = source_folder(args.source, output)
    states, edges, metadata = load_capture(root)
    repair = label_audit(root, states, edges)
    split = question_split([s.qid for s in states.values()], seed=42)
    qsets = {part: {q for q, p in split.items() if p == part} for part in ("train", "validation", "test")}
    mean, std = normalize_states(states, qsets["train"])
    np.savez_compressed(output / "input_normalization.npz", mean=mean, std=std)
    (output / "question_split.json").write_text(json.dumps(split, indent=2))
    with (output / "repaired_change_labels.jsonl").open("w") as handle:
        for sid, state in states.items():
            handle.write(json.dumps({"state_id": sid, "changed": state.changed.tolist()}) + "\n")
    subsets = {part: _restrict_edges(edges, qids, states) for part, qids in qsets.items()}
    paths = _paths_for_edges(edges, states, 3)
    psets = {part: [p for p in paths if states[p[0].src].qid in qids] for part, qids in qsets.items()}
    train_edges, test_edges = subsets["train"], subsets["test"]
    baseline_report, baseline_models = baselines(train_edges, test_edges, states, metadata)
    input_dim, content_dim = next(iter(states.values())).x.shape[-1], next(iter(states.values())).content.shape[-1]
    report = dict(version="fresh_native_reaudit_v2", source=str(args.source), label_audit=repair,
                  state_count=len(states), action_counts=dict(Counter("R" if e.action == 0 else "E" for e in edges)),
                  question_split_counts={p: len(q) for p, q in qsets.items()}, baselines_h1=baseline_report,
                  config=dict(updates=args.updates, seeds=args.seeds, content_loss="0.3*cosine_distance",
                              action_conditioning="all_arms", device=str(args.device)),
                  per_seed={}, training_logs={}, feature_sensitivity={}, matched_rollouts={})
    predictions = output / "predictions"; predictions.mkdir()
    checkpoints = output / "checkpoints"; checkpoints.mkdir()
    print("LABEL_AUDIT", json.dumps(repair), flush=True)
    for seed in args.seeds:
        report["per_seed"][str(seed)] = {}
        for name, width in (("rich_direct_h1", 256), ("latent_direct_h1", 128), ("latent_dynamics_h1_h3", 128)):
            _seed(seed)
            if name == "latent_dynamics_h1_h3":
                model = LatentDynamics(input_dim, content_dim)
                model.content_loss_mode = "cosine"
                model, logs = _fit_dynamics(model, psets["train"], psets["validation"], states,
                                           args.device, args.updates, seed, horizon=3, batch_size=8,
                                           eval_every=30, label=f"{name}/seed{seed}")
                records_by_h = _collect_rollout(model, psets["test"], states, args.device, 3)
            else:
                model = ActionDirectPredictor(input_dim, width, content_dim)
                model, logs = _fit_direct(model, train_edges, subsets["validation"], states,
                                         args.device, args.updates, seed, batch_size=16,
                                         eval_every=30, label=f"{name}/seed{seed}")
                records_by_h = {1: _collect_direct(model, test_edges, states, args.device)}
            report["training_logs"][f"{name}_seed{seed}"] = logs
            metrics = {}
            for horizon, records in records_by_h.items():
                metrics[f"h{horizon}"] = _action_metrics(records, states, train_edges)
                save_predictions(predictions / f"{name}_seed{seed}_h{horizon}.jsonl",
                                 records, states, metadata, baseline_models)
            report["per_seed"][str(seed)][name] = metrics
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            arm=name, seed=seed, input_dim=input_dim, content_dim=content_dim,
                            latent_dim=width, content_loss="cosine"),
                       checkpoints / f"{name}_seed{seed}.pt")
            if name == "latent_dynamics_h1_h3":
                matched = [p for p in psets["test"] if len(p) == 3]
                matched_records = _collect_rollout(model, matched, states, args.device, 3)
                sequence_groups = defaultdict(list)
                for p in matched:
                    sequence_groups["".join("R" if e.action == 0 else "E" for e in p)].append(p)
                report["matched_rollouts"][str(seed)] = {
                    "path_count": len(matched),
                    "by_horizon": {f"h{h}": _action_metrics(v, states, train_edges)
                                   for h, v in matched_records.items()},
                    "action_sequences": {},
                }
                for sequence, seq_paths in sequence_groups.items():
                    rolled = _collect_rollout(model, seq_paths, states, args.device, 3)
                    report["matched_rollouts"][str(seed)]["action_sequences"][sequence] = {
                        "path_count": len(seq_paths),
                        "metrics": {f"h{h}": _action_metrics(v, states, train_edges) for h, v in rolled.items()},
                    }
                report["feature_sensitivity"][str(seed)] = {}
                groups = {"native_hidden": slice(0, content_dim),
                          "current_token_embedding": slice(content_dim, content_dim * 2),
                          "expected_embedding": slice(content_dim * 2, content_dim * 3),
                          "scalars": slice(content_dim * 3, input_dim)}
                for group, columns in groups.items():
                    backup = {sid: s.x[:, columns].copy() for sid, s in states.items() if s.qid in qsets["test"]}
                    for sid in backup:
                        states[sid].x[:, columns] = 0
                    rolled = _collect_rollout(model, psets["test"], states, args.device, 3)
                    report["feature_sensitivity"][str(seed)][group] = {
                        f"h{h}": _action_metrics(rolled[h], states, train_edges) for h in (1, 3)}
                    for sid, values in backup.items():
                        states[sid].x[:, columns] = values
            print("RESULT", seed, name, json.dumps(metrics["h1"]), flush=True)
            del model
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
        report["elapsed_seconds"] = time.time() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file() and "source_capture" not in path.relative_to(output).parts:
                archive.write(path, path.relative_to(output))
    print("OUTPUT_ZIP", archive_path, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("fresh_native_reaudit_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.set_num_threads(4)
    run(args)
