#!/usr/bin/env python3
"""Re-audit a saved score-latent-drafter ZIP without recapturing Fast-dLLM.

The audit evaluates every held-out observed edge at H1 and every available
observed action sequence at H2/H3, then performs fixed-checkpoint input-path
ablations on the full-context model. It reports latent, latent-delta, memory,
scheduler-context, event, and mask metrics. Feature ablations measure reliance
of the trained checkpoint under controlled input removal; they are not retrained
causal feature-value estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from native_latent_dynamics_v2 import Edge, load_teacher
from native_latent_rules_v2 import align_frame
from score_latent_drafter import (
    SCHEMA, ScoreEncoder, ScoreTransition, _merge_batch_state, pack_states,
)


FEATURE_ABLATIONS = {
    "drop_native_residual": ("native_residual",),
    "drop_position_embedding": ("position_embedding",),
    "drop_scheduler_structure": ("scheduler_structure",),
    "drop_scheduler_context": ("scheduler_context",),
    "drop_geometry_embedding": ("geometry_embedding",),
    "drop_prefix_kv": ("prefix_kv",),
    "drop_action_conditioning": ("action_conditioning",),
}
FEATURE_DESCRIPTIONS = {
    "native_residual": "1536D cut-layer native residual representation",
    "position_embedding": "learned physical-position embedding in encoder and transition",
    "scheduler_structure": "8 per-position mask/event/region/valid channels",
    "scheduler_context": "7 native scheduler scalars",
    "geometry_embedding": "5 learned absolute proposal/frame geometry scalars",
    "prefix_kv": "compressed prefix-KV summary from native layers 0,7,15,23",
    "action_conditioning": "learned R/E action embedding and FiLM scale/shift; exact hard geometry rules remain",
}
STRUCTURAL_CHANNELS = [
    "mask_after_commit", "commit_event", "next_region_active",
    "mask_before_forward", "native_active_region", "actual_forward_region",
    "logical_proposal_span", "post_eos_validity",
]
GEOMETRY_FIELDS = [
    "proposal_start_absolute", "physical_frame_start", "prefix_kv_length",
    "proposal_length", "macro_block_index",
]
CONTEXT_NAMES = [
    "draft_position_fraction", "block_index_fraction", "denoising_pass_fraction",
    "current_step_fraction", "active_region_offset_fraction", "eos_in_native_proposal",
    "eos_in_top1_stop_fill",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def locate_run_root(base: Path) -> Path:
    required = ("run_status.json", "config.json", "split.json",
                "latent_score_drafter_report.json", "teacher_features/states.jsonl",
                "teacher_features/edges.jsonl", "models/full_context.pt")
    candidates = [base]
    candidates.extend(path.parent for path in base.rglob("latent_score_drafter_report.json"))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        f"No complete score-latent run found under {base}; expected files include "
        "latent_score_drafter_report.json, teacher_features/states.jsonl and models/full_context.pt")


def safe_extract_selected(archive: zipfile.ZipFile, dest: Path):
    required = {
        "split.json", "config.json", "latent_score_drafter_report.json",
        "run_status.json", "teacher_features/states.jsonl",
        "teacher_features/edges.jsonl", "teacher_features/train_only_normalization.npz",
        "models/native_only.pt", "models/native_structure.pt", "models/full_context.pt",
    }
    members = archive.namelist()
    missing = sorted(required - set(members))
    if missing:
        raise FileNotFoundError("Input ZIP is not a complete score-latent run; missing: " + ", ".join(missing))
    for name in members:
        if not name.startswith("teacher_features/") or not name.endswith(".npz"):
            continue
        required.add(name)
    for name in required:
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"Unsafe ZIP member path: {name!r}")
        target = (dest / Path(*posix.parts)).resolve()
        if dest.resolve() not in target.parents and target != dest.resolve():
            raise ValueError(f"ZIP member escapes extraction directory: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(name) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def load_checkpoint(path: Path, feature_set: str, raw_dim: int, config: dict, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("feature_set") != feature_set:
        raise ValueError(f"Unexpected checkpoint schema/variant at {path}")
    dim = int(checkpoint["latent_dim"])
    encoder = ScoreEncoder(raw_dim, dim, int(config.get("encoder_layers", 2)), feature_set)
    target = ScoreEncoder(raw_dim, dim, int(config.get("encoder_layers", 2)), feature_set)
    transition = ScoreTransition(dim, int(config.get("transition_layers", 2)), feature_set)
    encoder.load_state_dict(checkpoint["online_encoder"], strict=True)
    target.load_state_dict(checkpoint["target_encoder"], strict=True)
    transition.load_state_dict(checkpoint["transition"], strict=True)
    for module in (encoder, target, transition):
        module.to(device).eval()
    return encoder, target, transition, checkpoint


def exact_paths(states, edges, split, horizon, part="test"):
    outgoing = {}
    for edge in edges:
        if split.get(states[edge.src].qid) == part and split.get(states[edge.dst].qid) == part:
            outgoing.setdefault(edge.src, []).append(edge)
    paths = []
    roots = sorted(sid for sid, state in states.items()
                   if split.get(state.qid) == part and outgoing.get(sid))
    for root in roots:
        frontier = [(root, ())]
        for _ in range(horizon):
            next_frontier = []
            for current, chain in frontier:
                for edge in sorted(outgoing.get(current, ()), key=lambda item: item.action):
                    next_frontier.append((edge.dst, chain + (edge,)))
            frontier = next_frontier
            if not frontier:
                break
        if len(frontier) and all(len(chain) == horizon for _, chain in frontier):
            paths.extend((root, chain) for _, chain in frontier)
    return paths


def _auc(y, score):
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum()); negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def _binary_metrics(y, score, hard=None):
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=np.float64)
    if not len(y):
        return {"count": 0}
    if hard is None:
        hard = score >= 0.5
    hard = np.asarray(hard, dtype=bool)
    tp = int(np.sum(hard & (y == 1))); fp = int(np.sum(hard & (y == 0)))
    fn = int(np.sum(~hard & (y == 1))); tn = int(np.sum(~hard & (y == 0)))
    tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
    bins = np.minimum((np.clip(score, 0, 1) * 10).astype(int), 9)
    ece = 0.0
    for idx in range(10):
        mask = bins == idx
        if mask.any():
            ece += float(mask.mean()) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return {
        "count": len(y), "positive_fraction": float(y.mean()),
        "predicted_positive_fraction": float(hard.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float((hard == y).mean()), "precision": tp / max(1, tp + fp),
        "recall": tpr, "specificity": tnr, "balanced_accuracy": (tpr + tnr) / 2,
        "auc": _auc(y, score), "brier": float(np.mean((score - y) ** 2)), "ece_10_bins": ece,
    }


def _mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _mask_metrics(y, pred):
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)
    if not len(y):
        return {"count": 0}
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {"count": len(y), "accuracy": float(np.mean(y == pred)),
            "true_mask_fraction": float(y.mean()), "predicted_mask_fraction": float(pred.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _cosine_mean(a, b, mask):
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    if not mask.any():
        return None
    return float(cos[mask].mean().item())


def _bootstrap_question_ci(per_question, metric_fn, reps, seed):
    qids = sorted(per_question)
    if len(qids) < 2 or reps <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        stats = [metric_fn(per_question[qid]) for qid in sampled]
        stats = [float(v) for v in stats if v is not None and np.isfinite(v)]
        if stats:
            values.append(float(np.mean(stats)))
    if not values:
        return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def _init_bucket():
    return {
        "event_y": [], "event_score": [], "event_hard": [],
        "physical_event_y": [], "physical_event_score": [], "physical_event_hard": [],
        "valid_y": [], "valid_score": [], "valid_hard": [],
        "mask_y": [], "mask_hard": [],
        "z_sq_all": 0.0, "z_n_all": 0, "z_sq_proposal": 0.0, "z_n_proposal": 0,
        "delta_sq": 0.0, "delta_n": 0, "delta_rel": [], "delta_cos": [],
        "mem_sq": 0.0, "mem_n": 0, "mem_cos": [],
        "mem_delta_sq": 0.0, "mem_delta_n": 0,
        "context_abs": [], "context_sq": [],
        "change_y": [], "change_pred": [],
        "terminal_y": [], "terminal_score": [],
        "geometry_exact": [], "question_rows": {}, "records": 0,
    }


def _add_binary(bucket, qbucket, name, y, score, hard):
    y_np = y.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    score_np = score.detach().cpu().numpy().astype(np.float32).reshape(-1)
    hard_np = hard.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    bucket[f"{name}_y"].extend(y_np.tolist())
    bucket[f"{name}_score"].extend(score_np.tolist())
    bucket[f"{name}_hard"].extend(hard_np.tolist())
    qbucket[f"{name}_y"].extend(y_np.tolist())
    qbucket[f"{name}_score"].extend(score_np.tolist())
    qbucket[f"{name}_hard"].extend(hard_np.tolist())


@torch.no_grad()
def evaluate_scenario(encoder, target, transition, paths_by_horizon, states, scalers,
                      device, batch_size, change_thresholds, ablation=(),
                      bootstrap_reps=500, save_predictions=False):
    ablation = tuple(ablation)
    report_groups = {}
    predictions = []
    for horizon, paths in paths_by_horizon.items():
        for action in (0, 1):
            group_key = f"H{horizon}_{'R' if action == 0 else 'E'}"
            selected = [(root, chain) for root, chain in paths if chain[-1].action == action]
            if not selected:
                continue
            print(f"[audit] feature_set={transition.feature_set} ablation={'+'.join(ablation) or 'none'} "
                  f"group={group_key} rollout_paths={len(selected)}", flush=True)
            bucket = _init_bucket()
            per_question = {}
            for offset in range(0, len(selected), batch_size):
                group_paths = selected[offset:offset + batch_size]
                root_ids = [root for root, _ in group_paths]
                root_batch = pack_states(root_ids, states, scalers, device)
                z0, m0 = encoder(root_batch, ablate_features=ablation)
                current = _merge_batch_state((z0, m0), root_batch)
                if "scheduler_context" in ablation:
                    current["context"] = torch.zeros_like(current["context"])
                if "prefix_kv" in ablation:
                    current["memory_z"] = torch.zeros_like(current["memory_z"])
                final_pred = final_aux = final_batch = final_target_z = final_target_mem = None
                final_source_z = final_source_mem = None
                final_parent_prediction = final_parent_memory = final_parent_geometry = None
                for depth in range(horizon):
                    edge_batch = [chain[depth] for _, chain in group_paths]
                    action_tensor = torch.tensor([edge.action for edge in edge_batch],
                                                 dtype=torch.long, device=device)
                    if depth == horizon - 1:
                        final_parent_prediction = current["z"]
                        final_parent_memory = current["memory_z"]
                        final_parent_geometry = current["geometry"]
                    current, aux = transition(current, action_tensor, ablate_features=ablation)
                    if depth == horizon - 1:
                        final_pred, final_aux = current, aux
                        final_batch = pack_states([edge.dst for edge in edge_batch], states,
                                                  scalers, device)
                        source_batch = pack_states([edge.src for edge in edge_batch], states,
                                                   scalers, device)
                        final_target_z, final_target_mem = target(final_batch)
                        final_source_z, final_source_mem = target(source_batch)
                assert final_pred is not None
                proposal = final_batch["span"].bool()
                valid = final_batch["valid"].bool()
                mask_before = final_batch["mask_before"] >= .5
                physical_eligible = mask_before & (final_batch["forward_active"] >= .5) & valid
                proposal_eligible = physical_eligible & proposal
                event_y = final_batch["event"] >= .5
                event_score = final_aux["event_logits"].sigmoid()
                event_hard = final_pred["event"] >= .5
                valid_y = final_batch["valid"] >= .5
                valid_score = final_aux["valid_logits"].sigmoid()
                valid_hard = final_pred["valid"] >= .5
                mask_positions = proposal & valid
                mask_target = final_batch["mask"] >= .5
                mask_pred = final_pred["mask"] >= .5

                z_error = (final_pred["z"] - final_target_z).square().mean(-1)
                z_proposal_mask = proposal & valid
                z_all_sum = (z_error * valid).sum(-1)
                z_all_n = valid.sum(-1)
                z_prop_sum = (z_error * z_proposal_mask).sum(-1)
                z_prop_n = z_proposal_mask.sum(-1)
                source_geometry = torch.stack([
                    torch.as_tensor(states[chain[-1].src].geometry, dtype=torch.long)
                    for _, chain in group_paths
                ]).to(device)
                predicted_parent_aligned, pred_overlap = align_frame(
                    final_parent_prediction, final_parent_geometry, final_batch["geometry"])
                true_parent_aligned, true_overlap = align_frame(
                    final_source_z, source_geometry, final_batch["geometry"])
                delta_valid = (true_overlap > 0) & valid & proposal
                pred_delta = final_pred["z"] - predicted_parent_aligned
                true_delta = final_target_z - true_parent_aligned
                delta_error = (pred_delta - true_delta).square().mean(-1)
                delta_sq_sum = (delta_error * delta_valid).sum(-1)
                delta_n = delta_valid.sum(-1)
                target_delta_norm = true_delta.square().mean(-1).sqrt()
                delta_rel_valid = delta_valid & (target_delta_norm >= .05)
                delta_rel = ((pred_delta - true_delta).square().mean(-1).sqrt()
                             / target_delta_norm.clamp_min(.05))
                delta_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-8)

                mem_valid = final_batch["memory_valid"].bool()
                mem_error = (final_pred["memory_z"] - final_target_mem).square().mean(-1)
                source_mem_valid = pack_states([chain[-1].src for _, chain in group_paths],
                                               states, scalers, device)["memory_valid"].bool()
                mem_delta_valid = mem_valid & source_mem_valid
                true_mem_delta = final_target_mem - final_source_mem
                pred_mem_delta = final_pred["memory_z"] - final_parent_memory
                mem_delta_error = (pred_mem_delta - true_mem_delta).square().mean(-1)

                context_error = final_pred["context"] - final_batch["context"]
                change_labels = []
                change_valids = []
                for _, chain in group_paths:
                    labels, valid_change = _change_label_for_pair(
                        states[chain[-1].src], states[chain[-1].dst], scalers,
                        change_thresholds, device)
                    change_labels.append(labels[0]); change_valids.append(valid_change[0])
                change_labels = torch.stack(change_labels)
                change_valids = torch.stack(change_valids)
                change_pred = final_aux["change_logits"].argmax(-1)
                terminal_target = torch.tensor([
                    bool(states[chain[-1].dst].meta.get("native_terminal", False))
                    for _, chain in group_paths
                ], dtype=torch.float32, device=device)
                geometry_expected = torch.stack([
                    torch.as_tensor(states[chain[-1].dst].geometry, dtype=torch.long)
                    for _, chain in group_paths
                ]).to(device)
                geometry_exact = (final_pred["geometry"] == geometry_expected).all(-1)

                for i, (root, chain) in enumerate(group_paths):
                    edge = chain[-1]; qid = states[root].qid
                    qbucket = per_question.setdefault(qid, _init_bucket())
                    _add_binary(bucket, qbucket, "event", event_y[i][proposal_eligible[i]],
                                event_score[i][proposal_eligible[i]], event_hard[i][proposal_eligible[i]])
                    _add_binary(bucket, qbucket, "physical_event",
                                event_y[i][physical_eligible[i]],
                                event_score[i][physical_eligible[i]], event_hard[i][physical_eligible[i]])
                    _add_binary(bucket, qbucket, "valid", valid_y[i],
                                valid_score[i], valid_hard[i])
                    my = mask_target[i][mask_positions[i]].to(torch.uint8)
                    mh = mask_pred[i][mask_positions[i]].to(torch.uint8)
                    bucket["mask_y"].extend(my.cpu().tolist()); bucket["mask_hard"].extend(mh.cpu().tolist())
                    qbucket["mask_y"].extend(my.cpu().tolist()); qbucket["mask_hard"].extend(mh.cpu().tolist())
                    for name, value in (("z_sq_all", z_all_sum[i]), ("z_n_all", z_all_n[i]),
                                        ("z_sq_proposal", z_prop_sum[i]), ("z_n_proposal", z_prop_n[i]),
                                        ("delta_sq", delta_sq_sum[i]), ("delta_n", delta_n[i])):
                        bucket[name] += float(value.item())
                        qbucket[name] += float(value.item())
                    if delta_rel_valid[i].any():
                        vals = delta_rel[i][delta_rel_valid[i]].detach().cpu().tolist()
                        bucket["delta_rel"].extend(vals); qbucket["delta_rel"].extend(vals)
                        vals = delta_cos[i][delta_rel_valid[i]].detach().cpu().tolist()
                        bucket["delta_cos"].extend(vals); qbucket["delta_cos"].extend(vals)
                    if mem_valid[i]:
                        value = float(mem_error[i].item()) * final_target_mem.shape[-1]
                        bucket["mem_sq"] += value; bucket["mem_n"] += final_target_mem.shape[-1]
                        qbucket["mem_sq"] += value; qbucket["mem_n"] += final_target_mem.shape[-1]
                        value_cos = float(F.cosine_similarity(final_pred["memory_z"][i:i+1],
                                                              final_target_mem[i:i+1], dim=-1).item())
                        bucket["mem_cos"].append(value_cos); qbucket["mem_cos"].append(value_cos)
                    if mem_delta_valid[i]:
                        value = float(mem_delta_error[i].item()) * final_target_mem.shape[-1]
                        bucket["mem_delta_sq"] += value; bucket["mem_delta_n"] += final_target_mem.shape[-1]
                        qbucket["mem_delta_sq"] += value; qbucket["mem_delta_n"] += final_target_mem.shape[-1]
                    for j in range(context_error.shape[-1]):
                        bucket["context_abs"].append(abs(float(context_error[i, j].item())))
                        bucket["context_sq"].append(float(context_error[i, j].square().item()))
                        qbucket["context_abs"].append(abs(float(context_error[i, j].item())))
                        qbucket["context_sq"].append(float(context_error[i, j].square().item()))
                    valid_change = change_valids[i]
                    bucket["change_y"].extend(change_labels[i][valid_change].cpu().tolist())
                    bucket["change_pred"].extend(change_pred[i][valid_change].cpu().tolist())
                    qbucket["change_y"].extend(change_labels[i][valid_change].cpu().tolist())
                    qbucket["change_pred"].extend(change_pred[i][valid_change].cpu().tolist())
                    bucket["terminal_y"].append(int(terminal_target[i].item()))
                    bucket["terminal_score"].append(float(final_pred["terminal"][i].item()))
                    bucket["geometry_exact"].append(int(geometry_exact[i].item()))
                    qbucket["terminal_y"].append(int(terminal_target[i].item()))
                    qbucket["terminal_score"].append(float(final_pred["terminal"][i].item()))
                    qbucket["geometry_exact"].append(int(geometry_exact[i].item()))
                    bucket["records"] += 1; qbucket["records"] += 1
                    if save_predictions:
                        predictions.append({
                            "scenario": "", "question_id": qid, "root": root,
                            "src": edge.src, "dst": edge.dst,
                            "horizon": horizon, "action": "R" if action == 0 else "E",
                            "proposal_eligible": int(proposal_eligible[i].sum().item()),
                            "event_score": event_score[i].detach().cpu().numpy().astype(np.float16),
                            "event_target": event_y[i].cpu().numpy().astype(np.uint8),
                            "proposal_eligible_mask": proposal_eligible[i].cpu().numpy().astype(np.uint8),
                            "valid_score": valid_score[i].detach().cpu().numpy().astype(np.float16),
                            "valid_target": valid_y[i].cpu().numpy().astype(np.uint8),
                            "mask_pred": mask_pred[i].cpu().numpy().astype(np.uint8),
                            "mask_target": mask_target[i].cpu().numpy().astype(np.uint8),
                            "context_pred": final_pred["context"][i].detach().cpu().numpy().astype(np.float32),
                            "context_target": final_batch["context"][i].cpu().numpy().astype(np.float32),
                        })
            report_groups[group_key] = _summarize_bucket(bucket, per_question, bootstrap_reps,
                                                        seed=280 + horizon * 10 + action)
            report_groups[group_key]["rollout_count"] = bucket["records"]
            report_groups[group_key]["question_count"] = len(per_question)
            report_groups[group_key]["edge_ids_unique"] = len({
                (p[1][-1].src, p[1][-1].dst) for p in selected
            })
    result = {"by_horizon_action": report_groups, "predictions": predictions,
              "path_counts_by_horizon": {str(h): len(p) for h, p in paths_by_horizon.items()}}
    return result


def _change_label_for_pair(src, dst, scalers, thresholds, device):
    src_abs = int(src.geometry[1]) + np.arange(32)
    dst_abs = int(dst.geometry[1]) + np.arange(32)
    source_row = dst_abs - int(src.geometry[1])
    overlap = (source_row >= 0) & (source_row < 32)
    aligned = np.zeros_like(dst.u)
    aligned[overlap] = src.u[source_row[overlap]]
    source_valid = np.zeros(32, dtype=bool)
    source_valid[overlap] = src.valid[source_row[overlap]] > 0
    valid = overlap & source_valid & (dst.valid > 0)
    delta = (dst.u - aligned) / scalers["u_std"][None]
    magnitude = np.sqrt(np.mean(np.square(delta), axis=-1))
    labels = np.digitize(magnitude, np.asarray(thresholds, dtype=np.float64)).astype(np.int64)
    return (torch.as_tensor(labels, dtype=torch.long, device=device)[None],
            torch.as_tensor(valid, dtype=torch.bool, device=device)[None])


def _summarize_bucket(bucket, per_question, bootstrap_reps, seed):
    event = _binary_metrics(bucket["event_y"], bucket["event_score"], bucket["event_hard"])
    physical_event = _binary_metrics(bucket["physical_event_y"], bucket["physical_event_score"],
                                     bucket["physical_event_hard"])
    valid = _binary_metrics(bucket["valid_y"], bucket["valid_score"], bucket["valid_hard"])
    mask = _mask_metrics(bucket["mask_y"], bucket["mask_hard"])
    q_event = {qid: _binary_metrics(row["event_y"], row["event_score"], row["event_hard"])
               for qid, row in per_question.items() if row["event_y"]}
    ci_balanced = _bootstrap_question_ci(q_event,
                                         lambda x: x.get("balanced_accuracy"),
                                         bootstrap_reps, seed)
    ci_auc = _bootstrap_question_ci(q_event, lambda x: x.get("auc"), bootstrap_reps, seed + 1)
    change_y = np.asarray(bucket["change_y"], dtype=np.int64)
    change_pred = np.asarray(bucket["change_pred"], dtype=np.int64)
    change_accuracy = float(np.mean(change_y == change_pred)) if len(change_y) else None
    change_counts = {name: int(np.sum(change_y == idx)) for idx, name in enumerate(("small", "medium", "large"))}
    change_pred_counts = {name: int(np.sum(change_pred == idx)) for idx, name in enumerate(("small", "medium", "large"))}
    context_abs = np.asarray(bucket["context_abs"], dtype=np.float64)
    context_sq = np.asarray(bucket["context_sq"], dtype=np.float64)
    return {
        "event_on_proposal_eligible_positions": {**event,
            "question_macro_balanced_accuracy_ci95": ci_balanced,
            "question_macro_auc_ci95": ci_auc},
        "event_on_all_physical_eligible_positions": physical_event,
        "physical_validity": valid,
        "next_mask_on_proposal": mask,
        "latent_state": {
            "valid_physical_rmse": math.sqrt(bucket["z_sq_all"] / max(1, bucket["z_n_all"])),
            "proposal_span_rmse": math.sqrt(bucket["z_sq_proposal"] / max(1, bucket["z_n_proposal"])),
            "valid_physical_positions": int(bucket["z_n_all"]),
            "proposal_span_positions": int(bucket["z_n_proposal"]),
        },
        "latent_delta_same_frame_only": {
            "rmse": math.sqrt(bucket["delta_sq"] / bucket["delta_n"]) if bucket["delta_n"] else None,
            "relative_error_over_change_norm_ge_0_05": _mean(bucket["delta_rel"]),
            "cosine_over_change_norm_ge_0_05": _mean(bucket["delta_cos"]),
            "positions": int(bucket["delta_n"]),
            "supported_positions": len(bucket["delta_rel"]),
        },
        "prefix_memory_latent": {
            "rmse": math.sqrt(bucket["mem_sq"] / bucket["mem_n"]) if bucket["mem_n"] else None,
            "cosine": _mean(bucket["mem_cos"]),
            "elements": int(bucket["mem_n"]),
            "delta_rmse_valid_parent_child": math.sqrt(bucket["mem_delta_sq"] / bucket["mem_delta_n"])
                if bucket["mem_delta_n"] else None,
            "delta_elements": int(bucket["mem_delta_n"]),
        },
        "scheduler_context_7": {
            "names": CONTEXT_NAMES,
            "overall_mae": _mean(context_abs),
            "overall_rmse": math.sqrt(float(context_sq.mean())) if len(context_sq) else None,
            "component_mae": [float(np.mean(context_abs[i::7])) if len(context_abs[i::7]) else None
                              for i in range(7)],
            "component_rmse": [math.sqrt(float(np.mean(context_sq[i::7]))) if len(context_sq[i::7]) else None
                                for i in range(7)],
        },
        "coarse_native_residual_change": {
            "accuracy": change_accuracy, "positions": int(len(change_y)),
            "target_class_counts": change_counts, "predicted_class_counts": change_pred_counts,
        },
        "terminal": _binary_metrics(bucket["terminal_y"], bucket["terminal_score"]),
        "hard_geometry_exact_rate": _mean(bucket["geometry_exact"]),
        "notes": [
            "No same-frame overlap is counted as zero; cross-frame E delta is null/excluded.",
            "Latent scores are compared within this checkpoint's own EMA target coordinate system.",
            "Question bootstrap intervals resample held-out questions, not individual positions.",
        ],
    }


def _feature_impact(base, ablated):
    rows = {}
    for key in sorted(set(base["by_horizon_action"]) & set(ablated["by_horizon_action"])):
        b = base["by_horizon_action"][key]
        a = ablated["by_horizon_action"][key]
        bm = b["event_on_proposal_eligible_positions"]
        am = a["event_on_proposal_eligible_positions"]
        rows[key] = {
            "baseline_rollouts": b["rollout_count"],
            "ablation_rollouts": a["rollout_count"],
            "delta_balanced_accuracy_ablation_minus_full": _subtract(
                am.get("balanced_accuracy"), bm.get("balanced_accuracy")),
            "delta_auc_ablation_minus_full": _subtract(am.get("auc"), bm.get("auc")),
            "delta_brier_ablation_minus_full": _subtract(am.get("brier"), bm.get("brier")),
            "delta_event_specificity_ablation_minus_full": _subtract(
                am.get("specificity"), bm.get("specificity")),
            "delta_mask_accuracy_ablation_minus_full": _subtract(
                a["next_mask_on_proposal"].get("accuracy"), b["next_mask_on_proposal"].get("accuracy")),
            "delta_validity_balanced_accuracy_ablation_minus_full": _subtract(
                a["physical_validity"].get("balanced_accuracy"),
                b["physical_validity"].get("balanced_accuracy")),
            "delta_latent_proposal_rmse_ablation_minus_full": _subtract(
                a["latent_state"].get("proposal_span_rmse"), b["latent_state"].get("proposal_span_rmse")),
            "delta_memory_rmse_ablation_minus_full": _subtract(
                a["prefix_memory_latent"].get("rmse"), b["prefix_memory_latent"].get("rmse")),
            "delta_context_mae_ablation_minus_full": _subtract(
                a["scheduler_context_7"].get("overall_mae"), b["scheduler_context_7"].get("overall_mae")),
        }
    return rows


def _subtract(a, b):
    return None if a is None or b is None else float(a - b)


def _model_architecture(encoder, transition):
    return {
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "transition_parameters": sum(p.numel() for p in transition.parameters()),
        "encoder_trainable_form": "2-layer TransformerEncoder over 32 positions; d=64, 4 heads, FF=256",
        "transition_trainable_form": "action-conditioned 2-layer TransformerEncoder with source cross-attention; d=64, 4 heads, FF=256",
        "outputs": ["32x64 learned latent scores", "64D prefix-memory latent", "32 event scores",
                    "32 validity scores", "32 3-bin change scores", "7 scheduler-context values", "terminal score"],
    }


def audit_archive(args):
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    temp_dir = None
    try:
        if args.input_dir is not None:
            source_root = locate_run_root(args.input_dir)
            source_path = source_root
            source_digest = sha256_tree(source_root)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="score_latent_audit_"))
            with zipfile.ZipFile(args.input_zip) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError(f"ZIP CRC failed at {bad}")
                safe_extract_selected(archive, temp_dir)
            source_root = temp_dir
            source_path = args.input_zip
            source_digest = sha256_file(args.input_zip)
        status = json.loads((source_root / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise RuntimeError(f"Input capture run is not complete: {status}")
        config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
        original_report = json.loads((source_root / "latent_score_drafter_report.json").read_text(encoding="utf-8"))
        split = json.loads((source_root / "split.json").read_text(encoding="utf-8"))
        states, edges = load_teacher(source_root / "teacher_features")
        with np.load(source_root / "teacher_features" / "train_only_normalization.npz",
                     allow_pickle=False) as data:
            scalers = {key: data[key].astype(np.float32) for key in data.files}
        if len(states) != int(status.get("state_count", len(states))):
            raise RuntimeError("Loaded teacher state count does not match run_status.json")
        q_splits = {part: len({s.qid for s in states.values() if split.get(s.qid) == part})
                    for part in ("train", "validation", "test")}
        if q_splits["test"] == 0:
            raise RuntimeError("The saved question split has no test questions")
        paths_by_horizon = {
            h: exact_paths(states, edges, split, h, "test") for h in (1, 2, 3)
        }
        if not paths_by_horizon[1]:
            raise RuntimeError("No test one-step paths found")
        test_edge_ids = {(edge.src, edge.dst, edge.action) for edge in edges
                         if split.get(states[edge.src].qid) == "test"
                         and split.get(states[edge.dst].qid) == "test"}
        h1_edge_ids = {(chain[-1].src, chain[-1].dst, chain[-1].action)
                       for _, chain in paths_by_horizon[1]}
        if h1_edge_ids != test_edge_ids or len(paths_by_horizon[1]) != len(h1_edge_ids):
            raise RuntimeError(
                "H1 audit coverage is not a one-to-one match with held-out native edges: "
                f"paths={len(paths_by_horizon[1])}, unique={len(h1_edge_ids)}, "
                f"expected_edges={len(test_edge_ids)}")
        device = torch.device(args.device if args.device != "auto" else
                              ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available")
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        raw_dim = next(iter(states.values())).u.shape[-1]
        architecture = {}
        scenarios = {}
        predictions_jsonl = []
        full_context_thresholds = None
        for feature_set in ("native_only", "native_structure", "full_context"):
            encoder, target, transition, checkpoint = load_checkpoint(
                source_root / "models" / f"{feature_set}.pt", feature_set, raw_dim, config, device)
            architecture[feature_set] = _model_architecture(encoder, transition)
            result = evaluate_scenario(encoder, target, transition, paths_by_horizon,
                                       states, scalers, device, args.batch_size,
                                       checkpoint["change_bin_thresholds"],
                                       bootstrap_reps=args.bootstrap_reps,
                                       save_predictions=args.save_predictions)
            for row in result["predictions"]:
                row["scenario"] = feature_set
                predictions_jsonl.append(row)
            result.pop("predictions", None)
            scenarios[feature_set] = result
            if feature_set == "full_context":
                full_context_modules = (encoder, target, transition)
                full_context_thresholds = checkpoint["change_bin_thresholds"]
            else:
                del encoder, target, transition
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ablation_results = {}
        encoder, target, transition = full_context_modules
        for name, ablated_features in FEATURE_ABLATIONS.items():
            result = evaluate_scenario(encoder, target, transition, paths_by_horizon,
                                       states, scalers, device, args.batch_size,
                                       full_context_thresholds,
                                       ablation=ablated_features,
                                       bootstrap_reps=args.bootstrap_reps,
                                       save_predictions=args.save_predictions)
            for row in result["predictions"]:
                row["scenario"] = name
                predictions_jsonl.append(row)
            result.pop("predictions", None)
            ablation_results[name] = result
        impact = {
            name: {"removed_feature": ablated_features[0],
                   "description": FEATURE_DESCRIPTIONS[ablated_features[0]],
                   "interpretation": "Full-context fixed checkpoint; this intervention removes the named learned pathway while preserving the hard native action/frame/mask rules.",
                   "by_horizon_action": _feature_impact(scenarios["full_context"], result)}
            for name, (ablated_features, result) in (
                (name, (FEATURE_ABLATIONS[name], ablation_results[name]))
                for name in FEATURE_ABLATIONS
            )
        }
        report = {
            "schema": "score_latent_drafter_audit_v1",
            "source_artifact": str(source_path),
            "source_artifact_sha256": source_digest,
            "source_run_status": status,
            "source_commit": config.get("source_commit"),
            "audit_code_commit": __import__("os").environ.get("WM_SCORE_AUDIT_COMMIT"),
            "reused_capture_and_student_checkpoints": True,
            "dLLM_recaptured": False,
            "verifier_run": False,
            "question_split_counts": q_splits,
            "test_questions": sorted({s.qid for s in states.values() if split.get(s.qid) == "test"}),
            "unique_observed_paths_by_horizon": {str(h): len(paths) for h, paths in paths_by_horizon.items()},
            "heldout_edge_coverage": {
                "expected_unique_edges": len(test_edge_ids),
                "h1_unique_edges_scored": len(h1_edge_ids),
                "h1_one_to_one_complete": True,
            },
            "model_architecture": architecture,
            "feature_inventory": {
                "native_residual": {
                    "shape": [32, raw_dim],
                    "source": "native Fast-dLLM residual immediately before decoder layer 24",
                    "learned_projection": "LayerNorm -> MLP -> 64D per-position score latent",
                },
                "scheduler_structure": {
                    "shape": [32, len(STRUCTURAL_CHANNELS)], "channels": STRUCTURAL_CHANNELS,
                    "note": "The coupled channels are ablated as one group because several are deterministic views of the same scheduler state.",
                },
                "scheduler_context": {
                    "shape": [len(CONTEXT_NAMES)], "channels": CONTEXT_NAMES,
                    "note": "The final two EOS indicators have no positive terminal-transition evidence in this capture.",
                },
                "absolute_geometry": {
                    "shape": [len(GEOMETRY_FIELDS)], "channels": GEOMETRY_FIELDS,
                    "note": "Learned embedding is ablated separately from exact hard frame/position rules.",
                },
                "prefix_kv": {
                    "source_shape": [4, 2, 2, 2, 128],
                    "layers": [0, 7, 15, 23],
                    "learned_projection": "4096D summary -> 64D memory latent",
                },
                "learned_position_embedding": "One trainable physical-position code per slot, used in both encoder and transition.",
                "action_conditioning": "R/E action embedding and FiLM scale/shift in the transition; native exact geometry action remains hard-coded.",
                "model_outputs_and_audits": [
                    "64D latent state and its transition delta", "64D prefix-memory latent and its transition delta",
                    "per-position unmask/commit-event logits", "per-position physical-validity logits",
                    "per-position 3-bin residual-change logits", "per-position next-mask bits",
                    "7-dimension scheduler context", "terminal score (not evaluable positively without EOS rows)",
                    "exact hard geometry consistency",
                ],
            },
            "training_history_from_original_run": {
                name: original_report["variants"][name]["training"]
                for name in ("native_only", "native_structure", "full_context")
            },
            "evaluation_protocol": {
                "horizon": "At each H, the model starts from every test state that has an observed continuation, follows the captured native R/E action sequence, and scores only the H-th prediction. Thus H1 unique source-edge coverage is not truncated by array ordering.",
                "actions": "Groups are labeled by the final native action (R or E); earlier actions in H2/H3 can be mixed.",
                "feature_ablation": "Removes one learned input pathway from the already trained full_context model; it is an inference sensitivity test, not a retrained causal feature-value comparison.",
                "hard_rules": "Exact action, geometry alignment, mask monotonicity, and scheduler constraints remain active under all feature ablations. Geometry ablation removes its learned embedding only.",
                "excluded": ["token IDs", "verifier outputs/acceptance", "latency", "new dLLM forwards"],
            },
            "baselines": scenarios,
            "feature_ablation_sensitivity_vs_full_context": impact,
            "feature_ablation_metrics": ablation_results,
            "limitations": [
                "The saved capture contains no terminal/EOS transition, so terminal metrics have no positive evidence.",
                "Ablation results indicate reliance of this checkpoint; they do not establish that retraining without a feature would be worse.",
                "Each learned latent coordinate has no guaranteed named semantic meaning; this audit identifies input-path contributions, not coordinate meanings.",
                "This experiment still measures drafter latent dynamics and native unmask events; it does not measure verifier acceptance.",
            ],
            "elapsed_seconds": time.time() - started,
        }
        report_path = args.out_dir / "score_latent_drafter_audit_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False), encoding="utf-8")
        rows = []
        for scenario, scenario_report in scenarios.items():
            for group, metrics in scenario_report["by_horizon_action"].items():
                event = metrics["event_on_proposal_eligible_positions"]
                rows.append({"scenario": scenario, "group": group,
                             "rollouts": metrics["rollout_count"],
                             "questions": metrics["question_count"],
                             "event_accuracy": event.get("accuracy"),
                             "event_balanced_accuracy": event.get("balanced_accuracy"),
                             "event_auc": event.get("auc"), "event_brier": event.get("brier"),
                             "event_specificity": event.get("specificity"),
                             "valid_balanced_accuracy": metrics["physical_validity"].get("balanced_accuracy"),
                             "proposal_mask_accuracy": metrics["next_mask_on_proposal"].get("accuracy"),
                             "latent_proposal_rmse": metrics["latent_state"].get("proposal_span_rmse"),
                             "latent_delta_rmse": metrics["latent_delta_same_frame_only"].get("rmse"),
                             "memory_rmse": metrics["prefix_memory_latent"].get("rmse"),
                             "context_mae": metrics["scheduler_context_7"].get("overall_mae"),
                             "coarse_change_accuracy": metrics["coarse_native_residual_change"].get("accuracy")})
        for scenario, scenario_report in ablation_results.items():
            for group, metrics in scenario_report["by_horizon_action"].items():
                event = metrics["event_on_proposal_eligible_positions"]
                rows.append({"scenario": scenario, "group": group,
                             "rollouts": metrics["rollout_count"],
                             "questions": metrics["question_count"],
                             "event_accuracy": event.get("accuracy"),
                             "event_balanced_accuracy": event.get("balanced_accuracy"),
                             "event_auc": event.get("auc"), "event_brier": event.get("brier"),
                             "event_specificity": event.get("specificity"),
                             "valid_balanced_accuracy": metrics["physical_validity"].get("balanced_accuracy"),
                             "proposal_mask_accuracy": metrics["next_mask_on_proposal"].get("accuracy"),
                             "latent_proposal_rmse": metrics["latent_state"].get("proposal_span_rmse"),
                             "latent_delta_rmse": metrics["latent_delta_same_frame_only"].get("rmse"),
                             "memory_rmse": metrics["prefix_memory_latent"].get("rmse"),
                             "context_mae": metrics["scheduler_context_7"].get("overall_mae"),
                             "coarse_change_accuracy": metrics["coarse_native_residual_change"].get("accuracy")})
        if rows:
            with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        if args.save_predictions:
            with (args.out_dir / "position_predictions.jsonl").open("w", encoding="utf-8") as stream:
                for row in predictions_jsonl:
                    clean = {key: value.tolist() if isinstance(value, np.ndarray) else value
                             for key, value in row.items()}
                    stream.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
        (args.out_dir / "README.md").write_text(
            "# Score latent drafter follow-up audit\n\n"
            "This reuses the original capture, split, normalization and checkpoints. It does not reload Fast-dLLM or run the verifier. See `score_latent_drafter_audit_report.json`; `metrics.csv` gives the compact horizon/action table.\n\n"
            "Feature knockout numbers are fixed-checkpoint sensitivity results; they are not retrained ablations.\n",
            encoding="utf-8")
        out_zip = args.out_dir.with_suffix(".zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for file in sorted(args.out_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(args.out_dir).as_posix())
        print(f"[audit-complete] test_questions={q_splits['test']} ",
              f"paths={report['unique_observed_paths_by_horizon']} output={out_zip} ",
              f"elapsed={report['elapsed_seconds']:.1f}s", flush=True)
        return report
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-zip", type=Path)
    input_group.add_argument("--input-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.input_zip is not None and not args.input_zip.is_file():
        raise FileNotFoundError(args.input_zip)
    if args.input_dir is not None and not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    audit_archive(args)


if __name__ == "__main__":
    main()
