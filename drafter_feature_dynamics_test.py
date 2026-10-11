#!/usr/bin/env python3
"""Learn and audit Fast-dLLM feature dynamics without token or verifier targets.

Fresh native R/E observations are collected with the repository's audited
Fast-dLLM runner. Training inputs and targets contain only native hidden
features, scalar scheduler/features, and their temporal changes. Token IDs,
token embeddings, vocabulary logits, verifier labels, and acceptance values are
discarded before training and are not included in the result archive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

from fresh_native_drafter_test import collect as collect_native


SCALAR_NAMES = [
    "mask", "confidence", "top1_top2_margin", "topk_entropy",
    "within_block_position", "proposal_block_index", "native_forward_index",
    "context_length", "newly_unmasked", "feature_valid",
]
CONTINUOUS_SCALARS = (1, 2, 3)


@dataclass
class Observation:
    sid: str
    qid: str
    hidden: np.ndarray
    scalars: np.ndarray
    valid: np.ndarray
    history: np.ndarray


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    action: int  # 0=R, 1=E


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def question_split(qids, seed=42):
    unique = sorted(set(map(str, qids)))
    if len(unique) < 5:
        raise ValueError(f"Need at least 5 questions with native states for a 60/20/20 split; got {len(unique)}")
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train = max(1, int(round(0.60 * len(unique))))
    n_val = max(1, int(round(0.20 * len(unique))))
    n_val = min(n_val, len(unique) - n_train - 1)
    result = {}
    for rank, index in enumerate(order):
        result[unique[int(index)]] = (
            "train" if rank < n_train else
            "validation" if rank < n_train + n_val else "test")
    return result


def trim_capture_to_features(root: Path):
    """Keep hidden/scalars only; do not let token data reach trainer or ZIP."""
    state_rows = _read_jsonl(root / "states.jsonl")
    shard_names = sorted({row["shard"] for row in state_rows})
    hidden_dim = None
    for relative in shard_names:
        path = root / relative
        with np.load(path, allow_pickle=False) as source:
            raw = source["x"]
            if raw.ndim != 3 or (raw.shape[-1] - 10) % 3:
                raise ValueError(f"Unexpected native feature shape in {relative}: {raw.shape}")
            hidden_dim = (raw.shape[-1] - 10) // 3
            hidden = raw[..., :hidden_dim].astype(np.float16, copy=True)
            scalars = raw[..., 3 * hidden_dim:3 * hidden_dim + 10].astype(np.float32)
            valid = source["valid"].astype(bool, copy=True)
        temp = path.with_suffix(".features.tmp.npz")
        np.savez_compressed(temp, hidden=hidden, scalars=scalars, valid=valid)
        temp.replace(path)

    # Question text and prefix token IDs are not needed for the feature audit.
    qpath = root / "questions.jsonl"
    sanitized = []
    for row in _read_jsonl(qpath):
        sanitized.append({key: row[key] for key in (
            "question_id", "source_question_index", "state_ids", "edge_count",
            "state_count", "termination_reason") if key in row})
    qpath.write_text("".join(json.dumps(row) + "\n" for row in sanitized), encoding="utf-8")

    manifest_path = root / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_dynamics_audit"] = {
        "objective": "predict fixed native feature targets after R/E",
        "training_inputs": ["native final-layer hidden", *SCALAR_NAMES, "R-only temporal deltas"],
        "training_targets": ["train-only PCA of native hidden", "confidence", "margin",
                             "top-k entropy", "mask", "newly_unmasked"],
        "excluded_before_training": ["token IDs", "token embeddings", "candidate embeddings",
                                     "vocabulary logits", "verifier", "acceptance", "latency"],
        "prefix_memory": "not captured in this experiment; context length remains an observed scalar",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return hidden_dim


def load_feature_graph(root: Path):
    rows = _read_jsonl(root / "states.jsonl")
    by_shard = defaultdict(list)
    for row in rows:
        by_shard[row["shard"]].append(row)
    states = {}
    for shard, metadata in by_shard.items():
        with np.load(root / shard, allow_pickle=False) as arrays:
            for row in metadata:
                idx = int(row["row"])
                sid = str(row["state_id"])
                states[sid] = Observation(
                    sid=sid, qid=str(row["question_id"]),
                    hidden=arrays["hidden"][idx].astype(np.float32),
                    scalars=arrays["scalars"][idx].astype(np.float32),
                    valid=arrays["valid"][idx].astype(bool),
                    history=np.zeros((8, 5), dtype=np.float32),
                )
    edges = []
    for row in _read_jsonl(root / "edges.jsonl"):
        action = str(row["action"])
        if action not in ("R", "E"):
            continue
        src, dst = str(row["src_state_id"]), str(row["dst_state_id"])
        if src not in states or dst not in states:
            continue
        edges.append(Edge(src, dst, int(action == "E")))

    incoming = {edge.dst: edge for edge in edges}
    for sid, edge in incoming.items():
        if edge.action != 0:
            continue  # segment-local history is meaningful only for Refine
        parent, child = states[edge.src], states[edge.dst]
        valid = parent.valid & child.valid
        child.history[:, 0] = child.scalars[:, 1] - parent.scalars[:, 1]
        child.history[:, 1] = child.scalars[:, 2] - parent.scalars[:, 2]
        child.history[:, 2] = child.scalars[:, 3] - parent.scalars[:, 3]
        a = parent.hidden.astype(np.float64)
        b = child.hidden.astype(np.float64)
        denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cosine = np.divide((a * b).sum(-1), denom, out=np.zeros(8), where=denom > 1e-8)
        child.history[:, 3] = (1.0 - cosine).astype(np.float32)
        child.history[:, 4] = valid.astype(np.float32)
    return states, edges, incoming


def make_paths(states, edges, split, part, max_horizon=3):
    outgoing = defaultdict(list)
    for edge in edges:
        if split.get(states[edge.src].qid) == part and split.get(states[edge.dst].qid) == part:
            outgoing[edge.src].append(edge)
    paths = []
    for root in states.values():
        if split.get(root.qid) != part:
            continue
        path, current = [], root.sid
        while len(path) < max_horizon:
            choices = outgoing.get(current, [])
            if len(choices) != 1:
                break
            edge = choices[0]
            path.append(edge)
            current = edge.dst
        if path:
            paths.append((root.sid, path))
    return paths


def build_transforms(states, split, hidden_components=64):
    train = [state for state in states.values() if split.get(state.qid) == "train"]
    hidden_rows = np.concatenate([s.hidden[s.valid] for s in train if s.valid.any()], axis=0)
    hmean = hidden_rows.mean(0)
    hstd = np.maximum(hidden_rows.std(0), 1e-4)
    zhidden = (hidden_rows - hmean) / hstd
    n_components = min(hidden_components, zhidden.shape[1], zhidden.shape[0] - 1)
    if n_components < 4:
        raise ValueError(f"Too few valid training hidden rows for PCA: {zhidden.shape}")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1729)
    pca.fit(zhidden)
    pca_scores = np.concatenate([
        ((s.hidden[s.valid] - hmean) / hstd) @ pca.components_.T for s in train if s.valid.any()
    ], axis=0)
    pc_mean = pca_scores.mean(0)
    pc_std = np.maximum(pca_scores.std(0), 1e-4)

    scalar_rows = np.concatenate([s.scalars[s.valid] for s in train if s.valid.any()], axis=0)
    scalar_mean = scalar_rows[:, CONTINUOUS_SCALARS].mean(0)
    scalar_std = np.maximum(scalar_rows[:, CONTINUOUS_SCALARS].std(0), 1e-4)

    history_rows = [s.history[s.history[:, 4] > 0, :4]
                    for s in train if (s.history[:, 4] > 0).any()]
    hist = np.concatenate(history_rows, axis=0) if history_rows else np.zeros((0, 4), np.float32)
    if len(hist):
        history_mean, history_std = hist.mean(0), np.maximum(hist.std(0), 1e-4)
    else:
        history_mean, history_std = np.zeros(4, np.float32), np.ones(4, np.float32)

    for state in states.values():
        state.hidden = ((state.hidden - hmean) / hstd).astype(np.float32)
        state.scalars[:, CONTINUOUS_SCALARS] = (
            state.scalars[:, CONTINUOUS_SCALARS] - scalar_mean) / scalar_std
        state.history[:, :4] = (state.history[:, :4] - history_mean) / history_std
        scores = state.hidden @ pca.components_.T
        state._target_hidden = ((scores - pc_mean) / pc_std).astype(np.float32)
        state._target_scalars = state.scalars[:, CONTINUOUS_SCALARS].astype(np.float32)
        state._target_mask = state.scalars[:, 0].astype(np.float32)
        state._target_event = state.scalars[:, 8].astype(np.float32)
    transforms = dict(
        hidden_mean=hmean.astype(np.float32), hidden_std=hstd.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pc_mean.astype(np.float32), pca_std=pc_std.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        scalar_mean=scalar_mean.astype(np.float32), scalar_std=scalar_std.astype(np.float32),
        history_mean=history_mean.astype(np.float32), history_std=history_std.astype(np.float32),
    )
    return transforms


FEATURE_VARIANTS = {
    "uncertainty_scalars": ("uncertainty",),
    "scheduler_scalars": ("scheduler",),
    "unmask_event_only": ("event",),
    "scalars": ("scalars",),
    "hidden_scalars": ("hidden", "scalars"),
    "hidden_scalars_history": ("hidden", "scalars", "history"),
}


def state_input(state: Observation, variant: str):
    parts = []
    for part in FEATURE_VARIANTS[variant]:
        if part == "hidden":
            parts.append(state.hidden)
        elif part == "scalars":
            parts.append(state.scalars)
        elif part == "uncertainty":
            parts.append(state.scalars[:, 1:4])
        elif part == "scheduler":
            parts.append(state.scalars[:, [0, 4, 5, 6, 7, 9]])
        elif part == "event":
            parts.append(state.scalars[:, 8:9])
        else:
            parts.append(state.history)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def target_cont(state: Observation):
    return np.concatenate([state._target_hidden, state._target_scalars], axis=-1).astype(np.float32)


def target_binary(state: Observation):
    return np.stack([state._target_mask, state._target_event], axis=-1).astype(np.float32)


class PositionEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, layers: int = 2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 3, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.randn(1, 8, dim) * 0.01)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        if (~valid).all(-1).any():
            raise ValueError("Every captured native state must have at least one valid row")
        z = self.input(x) + self.position[:, :x.shape[1]]
        z = self.blocks(z, src_key_padding_mask=~valid)
        return self.norm(z) * valid.unsqueeze(-1)


class ActionTransition(nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        self.action = nn.Embedding(2, dim)
        self.position = nn.Parameter(torch.randn(1, 8, dim) * 0.01)
        self.networks = nn.ModuleList()
        for _ in range(2):
            layer = nn.TransformerEncoderLayer(dim, 4, dim * 3, dropout=0.05,
                activation="gelu", batch_first=True, norm_first=True)
            self.networks.append(nn.TransformerEncoder(layer, layers, enable_nested_tensor=False))
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                   nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, action, valid):
        outputs = torch.zeros_like(z)
        for action_id in (0, 1):
            indices = torch.nonzero(action == action_id, as_tuple=False).flatten()
            if not len(indices):
                continue
            value = z.index_select(0, indices)
            mask = valid.index_select(0, indices)
            value = value + self.action.weight[action_id][None, None] + self.position[:, :z.shape[1]]
            value = self.networks[action_id](value, src_key_padding_mask=~mask)
            value = self.norm(z.index_select(0, indices) + self.delta(value)) * mask.unsqueeze(-1)
            outputs = outputs.index_copy(0, indices, value)
        return outputs


class FeatureDecoder(nn.Module):
    def __init__(self, dim: int, continuous_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.continuous = nn.Linear(dim, continuous_dim)
        self.mask = nn.Linear(dim, 1)
        self.event = nn.Linear(dim, 1)

    def forward(self, z):
        h = self.body(z)
        return self.continuous(h), self.mask(h).squeeze(-1), self.event(h).squeeze(-1)


class FeatureDynamics(nn.Module):
    def __init__(self, input_dim, latent_dim, continuous_dim):
        super().__init__()
        self.encoder = PositionEncoder(input_dim, latent_dim)
        self.transition = ActionTransition(latent_dim)
        self.decoder = FeatureDecoder(latent_dim, continuous_dim)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_states(ids, states, variant, device):
    x = torch.as_tensor(np.stack([state_input(states[sid], variant) for sid in ids]),
                        dtype=torch.float32, device=device)
    valid = torch.as_tensor(np.stack([states[sid].valid for sid in ids]),
                            dtype=torch.bool, device=device)
    return x, valid


def _feature_loss(pred_cont, pred_mask, pred_event, targets, target_valid):
    cont = torch.as_tensor(np.stack([target_cont(s) for s in targets]),
                           dtype=torch.float32, device=pred_cont.device)
    binary = torch.as_tensor(np.stack([target_binary(s) for s in targets]),
                             dtype=torch.float32, device=pred_cont.device)
    hidden_dim = cont.shape[-1] - 3
    hidden_err = F.smooth_l1_loss(pred_cont[..., :hidden_dim], cont[..., :hidden_dim],
                                  reduction="none").mean(-1)
    scalar_err = F.smooth_l1_loss(pred_cont[..., hidden_dim:], cont[..., hidden_dim:],
                                  reduction="none").mean(-1)
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask, binary[..., 0], reduction="none")
    event_loss = F.binary_cross_entropy_with_logits(pred_event, binary[..., 1], reduction="none")
    # The destination's observation-valid mask is a supervision mask, not an
    # input to the transition. In particular, EOS/partial snapshots can have
    # fewer valid positions than their source state.
    weights = target_valid.float()
    denom = weights.sum().clamp_min(1)
    return ((hidden_err * weights).sum() / denom
            + 0.5 * (scalar_err * weights).sum() / denom
            + 0.25 * (mask_loss * weights).sum() / denom
            + 0.25 * (event_loss * weights).sum() / denom)


def evaluate_paths(model, paths, states, variant, device, max_horizon=3):
    model.eval()
    rows = []
    with torch.no_grad():
        batch_size = 128
        for start in range(0, len(paths), batch_size):
            path_batch = paths[start:start + batch_size]
            roots = [root for root, _ in path_batch]
            x, valid = _batch_states(roots, states, variant, device)
            z = model.encoder(x, valid)
            for depth in range(1, max_horizon + 1):
                active = [i for i, (_, path) in enumerate(path_batch) if len(path) >= depth]
                if not active:
                    continue
                ix = torch.as_tensor(active, device=device, dtype=torch.long)
                active_paths = [path_batch[i][1] for i in active]
                step_edges = [path[depth - 1] for path in active_paths]
                actions = torch.as_tensor([edge.action for edge in step_edges],
                                          device=device, dtype=torch.long)
                z_next = model.transition(z.index_select(0, ix), actions,
                                          valid.index_select(0, ix))
                z = z.index_copy(0, ix, z_next)
                pred_cont, mask_logit, event_logit = model.decoder(z_next)
                pred_cont = pred_cont.cpu().numpy()
                pred_mask = mask_logit.sigmoid().cpu().numpy()
                pred_event = event_logit.sigmoid().cpu().numpy()
                for local, edge in enumerate(step_edges):
                    target = states[edge.dst]
                    binary = target_binary(target)
                    rows.append(dict(
                        qid=target.qid, edge=f"{edge.src}->{edge.dst}",
                        action="E" if edge.action else "R", horizon=depth,
                        valid=target.valid.copy(), target_cont=target_cont(target),
                        pred_cont=pred_cont[local], target_mask=binary[:, 0],
                        pred_mask=pred_mask[local], target_event=binary[:, 1],
                        pred_event=pred_event[local],
                    ))
    return rows


def _binary_f1(target, pred):
    tp = int(np.sum(target & pred)); fp = int(np.sum(~target & pred)); fn = int(np.sum(target & ~pred))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def metrics(rows):
    if not rows:
        return {"n_edges": 0}
    valid = np.concatenate([row["valid"] for row in rows])
    y = np.concatenate([row["target_cont"] for row in rows])[valid]
    p = np.concatenate([row["pred_cont"] for row in rows])[valid]
    hidden_dim = y.shape[-1] - 3
    err = p - y
    mt = np.concatenate([row["target_mask"] for row in rows])[valid] > 0.5
    mp = np.concatenate([row["pred_mask"] for row in rows])[valid] >= 0.5
    et = np.concatenate([row["target_event"] for row in rows])[valid] > 0.5
    ep = np.concatenate([row["pred_event"] for row in rows])[valid] >= 0.5
    by_question = defaultdict(list)
    for row in rows:
        row_valid = row["valid"]
        if row_valid.any():
            by_question[row["qid"]].append(float(np.abs(row["pred_cont"][row_valid] -
                                                          row["target_cont"][row_valid]).mean()))
    return dict(
        n_edges=len(rows), n_questions=len(by_question), n_positions=int(valid.sum()),
        normalized_hidden_mae=float(np.abs(err[:, :hidden_dim]).mean()),
        normalized_hidden_rmse=float(np.sqrt(np.square(err[:, :hidden_dim]).mean())),
        confidence_mae=float(np.abs(err[:, hidden_dim]).mean()),
        margin_mae=float(np.abs(err[:, hidden_dim + 1]).mean()),
        entropy_mae=float(np.abs(err[:, hidden_dim + 2]).mean()),
        mask_brier=float(np.square(np.concatenate([r["pred_mask"] for r in rows])[valid] - mt).mean()),
        mask_f1=_binary_f1(mt, mp),
        unmask_event_brier=float(np.square(np.concatenate([r["pred_event"] for r in rows])[valid] - et).mean()),
        unmask_event_f1=_binary_f1(et, ep),
        question_macro_normalized_mae=float(np.mean([np.mean(x) for x in by_question.values()])),
        per_question_mae={qid: float(np.mean(values)) for qid, values in by_question.items()},
    )


def _mean_baseline(train_paths, states, action, continuous_dim):
    sums = np.zeros((8, continuous_dim + 2), np.float64)
    counts = np.zeros((8, 1), np.float64)
    mask_sum = np.zeros((8,), np.float64); event_sum = np.zeros((8,), np.float64)
    for _, path in train_paths:
        for edge in path:
            if edge.action != action:
                continue
            dst = states[edge.dst]
            rows = np.flatnonzero(dst.valid)
            target = target_cont(dst)
            sums[rows] += np.concatenate([target[rows], target_binary(dst)[rows]], axis=-1)
            counts[rows] += 1
            mask_sum[rows] += target_binary(dst)[rows, 0]
            event_sum[rows] += target_binary(dst)[rows, 1]
    means = sums / np.maximum(counts, 1)
    mask = mask_sum / np.maximum(counts[:, 0], 1)
    event = event_sum / np.maximum(counts[:, 0], 1)
    return means[:, :continuous_dim].astype(np.float32), mask.astype(np.float32), event.astype(np.float32)


def baseline_rows(paths, states, means):
    rows = []
    for root_id, path in paths:
        for depth, edge in enumerate(path[:3], start=1):
            cont, mask, event = means[edge.action]
            target = states[edge.dst]
            rows.append(dict(qid=target.qid, edge=f"{edge.src}->{edge.dst}",
                action="E" if edge.action else "R", horizon=depth, valid=target.valid.copy(),
                target_cont=target_cont(target), pred_cont=cont,
                target_mask=target_binary(target)[:, 0], pred_mask=mask,
                target_event=target_binary(target)[:, 1], pred_event=event))
    return rows


def _sample_paths(paths, batch_size, rng):
    by_action = {action: [path for path in paths if path[1] and path[1][0].action == action]
                 for action in (0, 1)}
    selected = []
    for i in range(batch_size):
        available = by_action[i % 2]
        if not available:
            available = paths
        selected.append(available[int(rng.integers(0, len(available)))])
    return selected


def _validation_score(rows):
    """Macro-score each observed action/horizon so H1 cannot swamp H3."""
    scores = []
    for horizon in (1, 2, 3):
        for action in ("R", "E"):
            group = [row for row in rows
                     if row["horizon"] == horizon and row["action"] == action]
            if not group:
                continue
            value = metrics(group)
            scores.append(
                value["normalized_hidden_mae"]
                + 0.5 * np.mean([value[name] for name in
                                 ("confidence_mae", "margin_mae", "entropy_mae")])
                + 0.25 * value["mask_brier"]
                + 0.25 * value["unmask_event_brier"]
            )
    return float(np.mean(scores)) if scores else math.inf


def fit_model(states, train_paths, val_paths, variant, latent_dim, device,
              updates, batch_size, seed, learning_rate=3e-4, label="model"):
    seed_all(seed)
    input_dim = len(state_input(next(iter(states.values())), variant)[0])
    continuous_dim = len(next(iter(states.values()))._target_hidden[0]) + 3
    model = FeatureDynamics(input_dim, latent_dim, continuous_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seed + 9001)
    best = math.inf
    best_state = None
    logs = []
    eval_every = max(20, updates // 5)
    for step in range(1, updates + 1):
        model.train()
        batch_paths = _sample_paths(train_paths, batch_size, rng)
        root_ids = [root for root, _ in batch_paths]
        x, valid = _batch_states(root_ids, states, variant, device)
        z = model.encoder(x, valid)
        root_targets = [states[sid] for sid in root_ids]
        pc, pm, pe = model.decoder(z)
        loss = 0.10 * _feature_loss(pc, pm, pe, root_targets, valid)
        max_depth = max(len(path) for _, path in batch_paths)
        current_z, current_valid = z, valid
        active_paths = batch_paths
        for depth in range(max_depth):
            active = [i for i, (_, path) in enumerate(active_paths) if len(path) > depth]
            if not active:
                break
            ix = torch.as_tensor(active, device=device, dtype=torch.long)
            active_paths = [active_paths[i] for i in active]
            current_z = current_z.index_select(0, ix)
            current_valid = current_valid.index_select(0, ix)
            step_edges = [path[depth] for _, path in active_paths]
            actions = torch.as_tensor([edge.action for edge in step_edges], device=device)
            current_z = model.transition(current_z, actions, current_valid)
            children = [states[edge.dst] for edge in step_edges]
            pc, pm, pe = model.decoder(current_z)
            child_valid = torch.as_tensor(
                np.stack([child.valid for child in children]),
                dtype=torch.bool, device=device)
            loss = loss + _feature_loss(pc, pm, pe, children, child_valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {label} loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == updates:
            val_rows = evaluate_paths(model, val_paths, states, variant, device)
            val = metrics(val_rows)
            score = _validation_score(val_rows)
            logs.append(dict(update=step, train_loss=float(loss.detach()),
                             gradient_norm=float(grad), validation_score=float(score),
                             validation_h1=val))
            if score < best:
                best = float(score)
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, logs, best


def _group_edges(paths):
    result = {h: [] for h in (1, 2, 3)}
    for root, path in paths:
        for depth, edge in enumerate(path[:3], start=1):
            result[depth].append(edge)
    return result


def evaluate_model(model, paths, states, variant, device):
    rows = evaluate_paths(model, paths, states, variant, device)
    report = {}
    for horizon in (1, 2, 3):
        for action, name in ((0, "R"), (1, "E")):
            selected = [row for row in rows if row["horizon"] == horizon and row["action"] == name]
            report[f"h{horizon}_{name}"] = metrics(selected)
        selected = [row for row in rows if row["horizon"] == horizon]
        report[f"h{horizon}_all"] = metrics(selected)
    return report, rows


def bootstrap_delta(model_rows, baseline_rows_, samples=1000, seed=43):
    model_by_q, base_by_q = defaultdict(list), defaultdict(list)
    for row in model_rows:
        valid = row["valid"]
        if valid.any():
            model_by_q[row["qid"]].append(float(np.abs(row["pred_cont"][valid] - row["target_cont"][valid]).mean()))
    for row in baseline_rows_:
        valid = row["valid"]
        if valid.any():
            base_by_q[row["qid"]].append(float(np.abs(row["pred_cont"][valid] - row["target_cont"][valid]).mean()))
    qids = sorted(set(model_by_q) & set(base_by_q))
    if not qids:
        return dict(n_questions=0)
    delta = np.asarray([np.mean(model_by_q[q]) - np.mean(base_by_q[q]) for q in qids])
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)])
    return dict(n_questions=len(qids), mean_model_minus_baseline=float(delta.mean()),
                ci95=[float(x) for x in np.quantile(draws, [0.025, 0.975])],
                negative_favors_model=True)


def average_seed_rows(rows_by_seed):
    """Average predictions for identical held-out transitions across seeds."""
    if not rows_by_seed:
        return []
    count = len(rows_by_seed[0])
    if any(len(rows) != count for rows in rows_by_seed):
        raise RuntimeError("Different evaluation row counts across model seeds")
    averaged = []
    for index in range(count):
        source = rows_by_seed[0][index]
        for rows in rows_by_seed[1:]:
            other = rows[index]
            if (source["edge"], source["horizon"], source["action"]) != (
                    other["edge"], other["horizon"], other["action"]):
                raise RuntimeError("Seed predictions are not aligned on the same held-out edge")
        row = dict(source)
        row["pred_cont"] = np.mean([rows[index]["pred_cont"] for rows in rows_by_seed], axis=0)
        row["pred_mask"] = np.mean([rows[index]["pred_mask"] for rows in rows_by_seed], axis=0)
        row["pred_event"] = np.mean([rows[index]["pred_event"] for rows in rows_by_seed], axis=0)
        averaged.append(row)
    return averaged


def run_audit(args, capture_manifest):
    capture_dir = args.out_dir / "capture"
    hidden_dim = trim_capture_to_features(capture_dir)
    states, edges, incoming = load_feature_graph(capture_dir)
    if not states or not edges:
        raise RuntimeError("Native feature capture produced no R/E transitions")
    qids = sorted({state.qid for state in states.values()})
    split = question_split(qids, args.split_seed)
    transforms = build_transforms(states, split, args.hidden_components)
    train_paths = make_paths(states, edges, split, "train")
    val_paths = make_paths(states, edges, split, "validation")
    test_paths = make_paths(states, edges, split, "test")
    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError(f"Need train/validation/test transition paths: "
                           f"{len(train_paths)}/{len(val_paths)}/{len(test_paths)}")
    continuous_dim = len(next(iter(states.values()))._target_hidden[0]) + 3
    recorded_hidden_layers = sorted({
        tuple(row.get("hidden_layer_indices", []))
        for row in _read_jsonl(capture_dir / "states.jsonl")
    })
    means = {action: _mean_baseline(train_paths, states, action, continuous_dim)
             for action in (0, 1)}
    baseline_test = baseline_rows(test_paths, states, means)

    configs = [(name, dim) for name in FEATURE_VARIANTS for dim in (128,)]
    configs += [("hidden_scalars_history", dim) for dim in (64, 256)]
    reports, model_rows_by_config, checkpoints = {}, {}, {}
    for name, dim in configs:
        for seed in args.seeds:
            key = f"{name}_d{dim}_s{seed}"
            print(f"[fit] {key} paths={len(train_paths)} updates={args.updates}", flush=True)
            model, logs, best = fit_model(states, train_paths, val_paths, name, dim,
                torch.device(args.train_device), args.updates, args.batch_size, seed, label=key)
            test_report, rows = evaluate_model(model, test_paths, states, name,
                                               torch.device(args.train_device))
            reports[key] = dict(feature_variant=name, latent_dim=dim, seed=seed,
                                best_validation_score=best, train_updates=logs,
                                test=test_report)
            model_rows_by_config[key] = rows
            checkpoints[key] = {k: v.cpu() for k, v in model.state_dict().items()}
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Nested-data experiment: identical architecture, optimizer steps and fixed validation.
    curve = {}
    train_qids = sorted(q for q, part in split.items() if part == "train")
    order = np.random.default_rng(args.seed).permutation(len(train_qids))
    ordered_train = [train_qids[int(i)] for i in order]
    curve_counts = sorted({min(count, len(train_qids))
                           for count in (10, 20, 40, len(train_qids))})
    for count in curve_counts:
        selected = set(ordered_train[:min(count, len(ordered_train))])
        paths = [p for p in train_paths if states[p[0]].qid in selected]
        key = f"questions_{len(selected)}"
        print(f"[data-curve] {key} paths={len(paths)} updates={args.curve_updates}", flush=True)
        model, logs, best = fit_model(states, paths, val_paths, "hidden_scalars_history", 128,
            torch.device(args.train_device), args.curve_updates, args.batch_size, args.seeds[0],
            label=f"learning_curve_{key}")
        val_report, _ = evaluate_model(model, val_paths, states, "hidden_scalars_history",
                                      torch.device(args.train_device))
        curve[key] = dict(question_count=len(selected), updates=args.curve_updates,
                          best_validation_score=best, validation=val_report,
                          checkpoints=logs)
        del model

    # Aggregate seed-specific validation scores before choosing a family.
    # The held-out test is reserved for the predeclared feature-group/dimension
    # ablations and the final ensemble-vs-baseline comparison.
    family_scores = {}
    family_rows = {}
    for name, dim in configs:
        keys = [f"{name}_d{dim}_s{seed}" for seed in args.seeds]
        family_scores[f"{name}_d{dim}"] = float(np.mean([
            reports[key]["best_validation_score"] for key in keys]))
        family_rows[(name, dim)] = average_seed_rows(
            [model_rows_by_config[key] for key in keys])
    select = min(family_scores, key=family_scores.get)
    selected_name, selected_dim = select.rsplit("_d", 1)
    selected_name, selected_dim = selected_name, int(selected_dim)
    selected_rows = family_rows[(selected_name, selected_dim)]
    delta = bootstrap_delta(selected_rows, baseline_test, seed=args.seed + 99)
    group_ablation = {}
    for name in FEATURE_VARIANTS:
        group_rows = family_rows[(name, 128)]
        group_ablation[name] = {}
        for h in (1, 2, 3):
            for action in ("R", "E"):
                key = f"h{h}_{action}"
                selected_rows_h = [row for row in group_rows
                                   if row["horizon"] == h and row["action"] == action]
                selected_base_h = [row for row in baseline_test
                                   if row["horizon"] == h and row["action"] == action]
                group_ablation[name][key] = dict(
                    metrics=metrics(selected_rows_h),
                    model_minus_train_mean=bootstrap_delta(
                        selected_rows_h, selected_base_h, seed=args.seed + h * 17 + (action == "E")))
    dimension_ablation = {}
    for dim in (64, 128, 256):
        dim_rows = family_rows[("hidden_scalars_history", dim)]
        dimension_ablation[str(dim)] = {}
        for h in (1, 2, 3):
            for action in ("R", "E"):
                key = f"h{h}_{action}"
                selected_rows_h = [row for row in dim_rows
                                   if row["horizon"] == h and row["action"] == action]
                selected_base_h = [row for row in baseline_test
                                   if row["horizon"] == h and row["action"] == action]
                dimension_ablation[str(dim)][key] = dict(
                    metrics=metrics(selected_rows_h),
                    model_minus_train_mean=bootstrap_delta(
                        selected_rows_h, selected_base_h, seed=args.seed + h * 17 + (action == "E")))
    report = dict(
        schema="drafter_feature_dynamics_v1",
        objective="R/E-conditioned prediction of native hidden/scheduler features",
        prohibited_targets=["token IDs", "token embeddings", "vocabulary logits",
                             "verifier output", "accepted token count", "latency"],
        data=dict(question_count=len(qids), state_count=len(states), edge_count=len(edges),
            action_counts={"R": sum(edge.action == 0 for edge in edges),
                           "E": sum(edge.action == 1 for edge in edges)},
            question_counts={part: sum(value == part for value in split.values())
                             for part in ("train", "validation", "test")},
            path_counts={part: len(value) for part, value in
                         (("train", train_paths), ("validation", val_paths), ("test", test_paths))},
            hidden_dim=hidden_dim, hidden_target_components=len(transforms["pca_std"]),
            hidden_target_explained_variance=float(transforms["pca_explained_variance_ratio"].sum()),
            hidden_provenance="native final hidden output captured before counterfactual fill",
            selected_native_hidden_layer_sets=[list(value) for value in recorded_hidden_layers],
            hidden_layer_ablation="not part of V1; the input uses the last selected native layer",
            feature_scalar_names=SCALAR_NAMES,
            feature_variants=FEATURE_VARIANTS,
            split_seed=args.split_seed),
        config=dict(latent_dims=[64, 128, 256], seeds=args.seeds,
                    updates_per_model=args.updates, learning_curve_updates=args.curve_updates,
                    horizon=3, extend_size=8, max_proposal_tokens=64,
                    max_refinement_steps=3, threshold=0.5,
                    evaluation="open-loop latent rollout; real child supplies targets only"),
        variants=reports,
        feature_group_ablation_d128=group_ablation,
        latent_dimension_ablation_full_features=dimension_ablation,
        learning_curve=curve,
        train_mean_baseline={
            f"h{h}_{action}": metrics([r for r in baseline_test
                if r["horizon"] == h and r["action"] == action])
            for h in (1, 2, 3) for action in ("R", "E")},
        selected_by_validation=dict(family=select, mean_score_across_seeds=family_scores[select],
                                    seed_scores={seed: reports[
                                        f"{selected_name}_d{selected_dim}_s{seed}"
                                    ]["best_validation_score"] for seed in args.seeds}),
        selected_model_minus_training_mean_bootstrap=delta,
        capture=capture_manifest,
        interpretation=dict(
            direction="Lower feature error/Brier and higher event F1 favor a latent dynamics model.",
            no_token_claim="The model never predicts or scores token identities.",
            scope="One 8-token native segment is the state. E predicts the next segment under longer prefix context; this does not represent a single 64-slot in-flight proposal.",
            prefix="Native prefix K/V memory is not captured; context-length/geometry scalars are observed.",
            test_policy="Validation selects the final family after averaging scores across seeds. The test split is reported only for the predeclared D=128 feature-group ablation, full-feature latent-dimension ablation, and final ensemble-vs-baseline audit."),
    )
    out = args.out_dir
    (out / "feature_dynamics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez_compressed(out / "feature_transforms.npz", **transforms)
    selected_keys = [f"{selected_name}_d{selected_dim}_s{seed}" for seed in args.seeds]
    torch.save(dict(schema="drafter_feature_dynamics_v1", selected_config=select,
                    seed_states={str(seed): checkpoints[key]
                                 for seed, key in zip(args.seeds, selected_keys)},
                    configs={key: reports[key] for key in selected_keys}),
               out / "selected_model.pt")
    with (out / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected_rows:
            clean = {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in row.items()}
            stream.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return report


def assert_capture_scrubbed(capture_dir: Path):
    manifest_path = capture_dir / "capture_manifest.json"
    states_path = capture_dir / "states.jsonl"
    if not manifest_path.is_file() or not states_path.is_file():
        raise RuntimeError("Refusing to package an incomplete or unsanitized native capture")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "feature_dynamics_audit" not in manifest:
        raise RuntimeError("Refusing to package before token arrays are stripped")
    shard_names = {row["shard"] for row in _read_jsonl(states_path)}
    for relative in shard_names:
        with np.load(capture_dir / relative, allow_pickle=False) as arrays:
            if set(arrays.files) != {"hidden", "scalars", "valid"}:
                raise RuntimeError(f"Refusing to package unsanitized shard: {relative}")
    for row in _read_jsonl(capture_dir / "questions.jsonl"):
        if "question" in row or "prefix_token_ids" in row:
            raise RuntimeError("Refusing to package raw question text or prompt token IDs")


def package(out: Path):
    assert_capture_scrubbed(out / "capture")
    archive = out.with_suffix(".zip")
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != temp:
                zf.write(path, path.relative_to(out).as_posix())
    temp.replace(archive)
    return archive


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dllm-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--extend-size", type=int, default=8)
    p.add_argument("--max-proposal-tokens", type=int, default=64)
    p.add_argument("--max-refinement-steps", type=int, default=3)
    p.add_argument("--updates", type=int, default=160)
    p.add_argument("--curve-updates", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--hidden-components", type=int, default=64)
    p.add_argument("--drafter-device", type=int, default=0)
    p.add_argument("--train-device", default="cuda:1")
    p.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--raw-top-k", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    if args.extend_size != 8 or args.max_proposal_tokens != 64 or args.max_refinement_steps != 3:
        raise ValueError("V1 is pinned to 8-token E segments, max proposal 64, and at most three R steps")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU; two T4s are recommended")
    if args.num_questions < 5:
        raise ValueError("Use at least 5 GSM8K questions; 100 are recommended for this feasibility test")
    if torch.cuda.device_count() == 1:
        args.train_device = "cuda:0"
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    capture_dir = args.out_dir / "capture"
    capture_args = SimpleNamespace(
        out_dir=capture_dir, num_questions=args.num_questions, seed=args.seed,
        threshold=args.threshold, extend_size=args.extend_size, max_proposal_tokens=args.max_proposal_tokens,
        max_refinement_steps=args.max_refinement_steps, physical_block_size=32,
        small_block_size=8, raw_top_k=args.raw_top_k, drafter_device=args.drafter_device,
        shard_rows=32, tokenizer_name=args.tokenizer_name, dllm_dir=args.dllm_dir,
    )
    started = time.time()
    try:
        print("Collecting fresh native Fast-dLLM R/E states. Verifier is not loaded.", flush=True)
        manifest = collect_native(capture_args)
        print(f"[capture-complete] questions={manifest['question_count']} "
              f"states={manifest['state_count']} edges={manifest['edge_count']} "
              f"actions={manifest['action_counts']}", flush=True)
        report = run_audit(args, manifest)
        report["elapsed_total_seconds"] = time.time() - started
        (args.out_dir / "feature_dynamics_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (args.out_dir / "README.md").write_text(
            "# Drafter feature dynamics test\n\n"
            "Fresh native Fast-dLLM GSM8K capture; no verifier is loaded or queried. "
            "The learner predicts only hidden-feature projections, confidence/margin/entropy, "
            "mask and unmask event. It has no token or acceptance output. See "
            "feature_dynamics_report.json for groups, H1-H3 results and scope.\n",
            encoding="utf-8")
        archive = package(args.out_dir)
        print(f"[result-zip] {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    except Exception as exc:
        (args.out_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        try:
            print("[partial-zip]", package(args.out_dir), flush=True)
        except Exception as package_exc:
            print("[partial-zip-skipped] raw capture is not confirmed scrubbed; "
                  f"kept error.txt in working directory ({package_exc})", flush=True)
        raise


if __name__ == "__main__":
    main()
