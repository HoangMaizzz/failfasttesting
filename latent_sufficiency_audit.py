"""Question-disjoint feasibility audit for a learned dLLM latent drafter.

The Refine arm uses the 100-question native capture archive and compares:
  (1) a wider direct observation-to-next-behavior predictor,
  (2) an end-to-end 128D predictive latent without a learned transition, and
  (3) a 128D latent transition trained and evaluated in free-running rollouts.

The optional Extend arm uses the existing five-question structured graph in
leave-one-question-out H1 evaluation. It is explicitly reported separately;
it is too small to support a multi-step/generalization claim.
No verifier, target model, or dLLM weights are loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LATENT_DIM = 128
SEED_SPLIT = 42
R_WIDTH = 32
E_WIDTH = 64


@dataclass
class State:
    sid: str
    qid: str
    x: np.ndarray
    mask: np.ndarray
    eligible: np.ndarray
    confidence: np.ndarray
    changed: np.ndarray
    content: np.ndarray
    proposal_len: int


@dataclass
class Edge:
    src: str
    dst: str
    action: int  # 0=Refine, 1=Extend
    valid: np.ndarray
    changed_valid: np.ndarray
    new_suffix: np.ndarray


def question_split(qids: list[str], seed: int = SEED_SPLIT) -> dict[str, str]:
    qids = sorted(set(qids))
    order = np.random.default_rng(seed).permutation(len(qids))
    n_train = max(1, min(int(.70 * len(qids)), len(qids) - 2))
    n_val = max(1, min(int(.15 * len(qids)), len(qids) - n_train - 1))
    result = {}
    for rank, idx in enumerate(order):
        result[qids[int(idx)]] = "train" if rank < n_train else (
            "validation" if rank < n_train + n_val else "test")
    return result


def _load_npz_member(archive: zipfile.ZipFile, name: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _r_features(meta: dict, row: dict[str, np.ndarray], index: int) -> np.ndarray:
    mask = row["mask"][index].astype(np.float32)
    eligible = row["eligible"][index].astype(np.float32)
    confidence = row["confidence"][index].astype(np.float32)
    entropy = row["entropy"][index].astype(np.float32)
    margin = row["margin"][index].astype(np.float32)
    width = len(mask)
    positions = np.arange(width, dtype=np.float32)
    active_ratio = float(mask[eligible.astype(bool)].mean()) if eligible.any() else 0.0
    structure = np.stack([
        mask, eligible, positions / max(1, width - 1),
        (float(meta.get("block_start", 0)) + positions) / 4096.0,
        np.full(width, float(meta.get("context_len", 0)) / 4096.0, np.float32),
        np.full(width, float(meta.get("step", 0)) / 8.0, np.float32),
        np.full(width, float(meta.get("small_block_index", 0)) / 3.0, np.float32),
        np.full(width, active_ratio, np.float32),
    ], axis=-1)
    temporal = np.stack([
        confidence - row["prev_confidence"][index].astype(np.float32),
        entropy - row["prev_entropy"][index].astype(np.float32),
        margin - row["prev_margin"][index].astype(np.float32),
        row["candidate_changed"][index].astype(np.float32),
    ], axis=-1)
    return np.concatenate([
        row["hidden"][index].astype(np.float32),
        row["token_emb"][index].astype(np.float32),
        row["candidate_emb"][index].astype(np.float32),
        structure,
        np.stack([confidence, entropy, margin], axis=-1),
        temporal,
    ], axis=-1).astype(np.float16)


def load_refine(path: Path) -> tuple[dict[str, State], list[Edge], list[list[str]], dict]:
    states: dict[str, State] = {}
    groups: dict[str, list[dict]] = {}
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("capture_manifest.json"))
        if manifest.get("status") != "complete":
            raise ValueError("Refine archive capture_manifest.json is not complete")
        cache_name, cache = None, None
        for meta in manifest["states"]:
            member = meta.get("npz") or meta.get("npz_path")
            if member != cache_name:
                cache = _load_npz_member(archive, member)
                cache_name = member
            index = int(meta["row"])
            sid = str(meta["uid"])
            states[sid] = State(
                sid=sid, qid=str(meta["question_id"]),
                x=_r_features(meta, cache, index),
                mask=cache["mask"][index].astype(bool),
                eligible=cache["eligible"][index].astype(bool),
                confidence=cache["confidence"][index].astype(np.float32),
                changed=cache["candidate_changed"][index].astype(bool),
                content=cache["candidate_emb"][index].astype(np.float32),
                proposal_len=R_WIDTH,
            )
            groups.setdefault(str(meta["group_id"]), []).append(meta)

    edges: list[Edge] = []
    paths: list[list[str]] = []
    for rows in groups.values():
        rows.sort(key=lambda item: int(item["step"]))
        chain = [str(rows[0]["uid"])] if rows else []
        for parent, child in zip(rows, rows[1:]):
            if int(child["forward_id"]) != int(parent["forward_id"]) + 1:
                if len(chain) > 1:
                    paths.append(chain)
                chain = [str(child["uid"])]
                continue
            a, b = states[str(parent["uid"])], states[str(child["uid"])]
            active = a.mask & a.eligible
            edges.append(Edge(a.sid, b.sid, 0, active.copy(), active.copy(),
                              np.zeros(R_WIDTH, dtype=bool)))
            chain.append(b.sid)
        if len(chain) > 1:
            paths.append(chain)
    if not edges:
        raise ValueError("No adjacent native Refine transitions found in archive")
    info = {
        "archive": path.name, "schema": manifest.get("schema_version"),
        "status": manifest.get("status"), "question_count": len({s.qid for s in states.values()}),
        "state_count": len(states), "refine_edge_count": len(edges),
        "rollout_path_count": len(paths), "threshold": manifest.get("config", {}).get("threshold"),
        "observation_stage": manifest.get("provenance", {}).get("phase"),
        "hidden_provenance": manifest.get("provenance", {}).get("hidden"),
    }
    return states, edges, paths, info


def _topk_distribution(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x -= np.max(x, axis=-1, keepdims=True)
    exp = np.exp(np.clip(x, -80, 0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def load_extend(path: Path) -> tuple[dict[str, State], list[Edge], dict]:
    states: dict[str, State] = {}
    edges: list[Edge] = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("graph_manifest.json"))
        nodes = [json.loads(line) for line in archive.read("nodes.jsonl").decode("utf-8").splitlines()]
        graph_edges = [json.loads(line) for line in archive.read("edges.jsonl").decode("utf-8").splitlines()]
        node_map = {str(node["state_id"]): node for node in nodes}
        shard_cache: dict[str, dict[str, np.ndarray]] = {}
        for node in nodes:
            shard = str(node["shard"])
            if shard not in shard_cache:
                shard_cache[shard] = _load_npz_member(archive, f"raw/gsm8k_structured/{shard}")
            arr = shard_cache[shard]
            r = int(node["row"])
            length = min(E_WIDTH, int(node["proposal_length"]))
            hidden = arr["hidden_states"][r, -1].astype(np.float32)
            dist = _topk_distribution(arr["topk_logits"][r])
            conf_raw = arr["drafter_observed_prob"][r].astype(np.float32)
            mask_raw = arr["proposal_mask"][r].astype(bool)
            conf = np.zeros(E_WIDTH, np.float32)
            mask = np.zeros(E_WIDTH, bool)
            dist_full = np.zeros((E_WIDTH, dist.shape[-1]), np.float32)
            conf[:length] = conf_raw[:length]
            mask[:length] = mask_raw[:length]
            dist_full[:length] = dist[:length]
            x = np.zeros((E_WIDTH, hidden.shape[-1] + dist.shape[-1] + 9), np.float32)
            x[:length, :hidden.shape[-1]] = hidden[:length]
            x[:length, hidden.shape[-1]:hidden.shape[-1] + dist.shape[-1]] = dist_full[:length]
            off = hidden.shape[-1] + dist.shape[-1]
            x[:length, off] = conf[:length]
            x[:length, off + 1] = mask[:length]
            pos = np.arange(E_WIDTH, dtype=np.float32)
            x[:, off + 2] = pos / max(1, E_WIDTH - 1)
            x[:, off + 3] = length / E_WIDTH
            x[:, off + 4] = float(node.get("masks_remaining", 0)) / E_WIDTH
            x[:, off + 5] = float(node.get("extend_depth", 0)) / 8.0
            x[:, off + 6] = float(node.get("refine_steps_since_extend", 0)) / 4.0
            x[:, off + 7] = (pos >= length).astype(np.float32)
            x[:, off + 8] = (pos < length).astype(np.float32)
            states[str(node["state_id"])] = State(
                sid=str(node["state_id"]), qid=str(node["problem_id"]),
                x=x.astype(np.float16), mask=mask, eligible=np.arange(E_WIDTH) < length,
                confidence=conf, changed=np.zeros(E_WIDTH, bool),
                content=dist_full, proposal_len=length,
            )
        for edge in graph_edges:
            if edge.get("action") != "E":
                continue
            src, dst = str(edge["src_state_id"]), str(edge["dst_state_id"])
            if src not in states or dst not in states:
                continue
            a, b = states[src], states[dst]
            parent_node, child_node = node_map[src], node_map[dst]
            parent_len, child_len = a.proposal_len, b.proposal_len
            active = np.zeros(E_WIDTH, bool)
            active[:min(parent_len, child_len)] = a.mask[:min(parent_len, child_len)] | b.mask[:min(parent_len, child_len)]
            if child_len > parent_len:
                active[parent_len:child_len] = True
            changed_valid = np.zeros(E_WIDTH, bool)
            changed_valid[:parent_len] = a.mask[:parent_len]
            child_arr = shard_cache[str(child_node["shard"])]
            src_arr = shard_cache[str(parent_node["shard"])]
            child_ids = child_arr["topk_token_ids"][int(child_node["row"]), :, 0]
            src_ids = src_arr["topk_token_ids"][int(parent_node["row"]), :, 0]
            changed = np.zeros(E_WIDTH, bool)
            upto = min(parent_len, child_len)
            changed[:upto] = child_ids[:upto] != src_ids[:upto]
            b.changed = changed
            edges.append(Edge(src, dst, 1, active, changed_valid,
                              np.arange(E_WIDTH) >= parent_len))
    if not edges:
        raise ValueError("No Extend edges found in structured graph archive")
    qids = sorted({s.qid for s in states.values()})
    info = {
        "archive": path.name, "schema": manifest.get("schema_version"),
        "status": manifest.get("status"), "question_count": len(qids),
        "problem_ids": qids, "state_count": len(states), "extend_edge_count": len(edges),
        "transition_backend": manifest.get("transition_backend"),
        "threshold": manifest.get("config", {}).get("threshold"),
        "protocol_note": "Legacy structured sparse capture: full-context replay, threshold and E mechanics are those recorded in this archive.",
    }
    return states, edges, info


def normalize_states(states: dict[str, State], train_qids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    selected = [state.x.astype(np.float32) for state in states.values() if state.qid in train_qids]
    if not selected:
        raise ValueError("No training observations for normalization")
    total = np.zeros(selected[0].shape[-1], np.float64)
    square = np.zeros_like(total)
    count = 0
    for x in selected:
        total += x.sum(axis=0, dtype=np.float64)
        square += np.square(x.astype(np.float64)).sum(axis=0)
        count += x.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1e-5))
    for state in states.values():
        state.x = ((state.x.astype(np.float32) - mean) / std).astype(np.float16)
    return mean.astype(np.float32), std.astype(np.float32)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, width: int, layers: int = 1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.pos = nn.Embedding(E_WIDTH, width)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.shape[1]
        idx = torch.arange(p, device=x.device)
        return self.context(self.proj(x) + self.pos(idx)[None])


class ForecastHead(nn.Module):
    def __init__(self, width: int, content_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mask = nn.Linear(width, 1)
        self.conf = nn.Linear(width, 1)
        self.changed = nn.Linear(width, 1)
        self.content = nn.Linear(width, content_dim)

    def forward(self, z):
        z = self.norm(z)
        return {
            "mask": self.mask(z).squeeze(-1),
            "confidence": self.conf(z).squeeze(-1).sigmoid(),
            "changed": self.changed(z).squeeze(-1),
            "content": self.content(z),
        }


class DirectPredictor(nn.Module):
    def __init__(self, input_dim: int, width: int, content_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, width, layers=2 if width >= 256 else 1)
        self.head = ForecastHead(width, content_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


class LatentDynamics(nn.Module):
    def __init__(self, input_dim: int, content_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, layers=1)
        self.action = nn.Embedding(2, latent_dim)
        block = nn.TransformerEncoderLayer(latent_dim, 4, latent_dim * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transition = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
        self.delta = nn.Linear(latent_dim, latent_dim)
        self.head = ForecastHead(latent_dim, content_dim)

    def advance(self, z, action):
        action_z = self.action(action)[:, None, :]
        delta = self.delta(self.transition(z + action_z))
        return z + delta


def device_batch(states: list[State], device: torch.device):
    x = torch.as_tensor(np.stack([s.x for s in states]), dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.stack([s.mask for s in states]), dtype=torch.bool, device=device)
    conf = torch.as_tensor(np.stack([s.confidence for s in states]), dtype=torch.float32, device=device)
    change = torch.as_tensor(np.stack([s.changed for s in states]), dtype=torch.float32, device=device)
    content = torch.as_tensor(np.stack([s.content for s in states]), dtype=torch.float32, device=device)
    return x, mask, conf, change, content


def edge_loss(pred, edges: list[Edge], states: dict[str, State], device: torch.device):
    sources = [states[e.src] for e in edges]
    targets = [states[e.dst] for e in edges]
    valid = torch.as_tensor(np.stack([e.valid for e in edges]), dtype=torch.bool, device=device)
    change_valid = torch.as_tensor(np.stack([e.changed_valid for e in edges]), dtype=torch.bool, device=device)
    ymask = torch.as_tensor(np.stack([s.mask for s in targets]), dtype=torch.float32, device=device)
    yconf = torch.as_tensor(np.stack([s.confidence for s in targets]), dtype=torch.float32, device=device)
    ychanged = torch.as_tensor(np.stack([s.changed for s in targets]), dtype=torch.float32, device=device)
    ycontent = torch.as_tensor(np.stack([s.content for s in targets]), dtype=torch.float32, device=device)
    if not bool(valid.any()):
        return pred["mask"].sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(pred["mask"][valid], ymask[valid])
    loss = loss + F.smooth_l1_loss(pred["confidence"][valid], yconf[valid])
    if bool(change_valid.any()):
        loss = loss + F.binary_cross_entropy_with_logits(pred["changed"][change_valid], ychanged[change_valid])
    pred_content = pred["content"][valid]
    true_content = ycontent[valid]
    if pred_content.shape[-1] == true_content.shape[-1]:
        loss = loss + .15 * F.smooth_l1_loss(pred_content, true_content)
    return loss


def _paths_for_edges(edges: list[Edge], states: dict[str, State], max_horizon: int = 3) -> list[list[Edge]]:
    by_src: dict[str, list[Edge]] = {}
    for edge in edges:
        by_src.setdefault(edge.src, []).append(edge)
    result = []
    for edge in edges:
        path, cur = [edge], edge.dst
        while len(path) < max_horizon:
            nxt = by_src.get(cur, [])
            if len(nxt) != 1:
                break
            path.append(nxt[0])
            cur = nxt[0].dst
        result.append(path)
    return result


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(bool).reshape(-1)
    score = score.astype(np.float64).reshape(-1)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(bool).reshape(-1)
    score = score.reshape(-1)
    if not y.any():
        return None
    order = np.argsort(-score, kind="mergesort")
    hits = np.cumsum(y[order])
    return float(np.mean(hits[y[order]] / (np.flatnonzero(y[order]) + 1)))


def _collect_direct(model, edges: list[Edge], states: dict[str, State], device, batch_size=16):
    model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(edges), batch_size):
            batch = edges[start:start + batch_size]
            src = [states[e.src] for e in batch]
            x = torch.as_tensor(np.stack([s.x for s in src]), dtype=torch.float32, device=device)
            pred = model(x)
            for i, edge in enumerate(batch):
                records.append((edge, {k: v[i].float().cpu().numpy() for k, v in pred.items()}))
    return records


def _collect_rollout(model, paths: list[list[Edge]], states: dict[str, State], device,
                     max_horizon=3, batch_size=8):
    model.eval()
    by_h = {h: [] for h in range(1, max_horizon + 1)}
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            initial = [states[p[0].src] for p in batch_paths]
            x = torch.as_tensor(np.stack([s.x for s in initial]), dtype=torch.float32, device=device)
            z = model.encoder(x)
            for h in range(1, max_horizon + 1):
                active = [i for i, path in enumerate(batch_paths) if len(path) >= h]
                if not active:
                    continue
                edge_batch = [batch_paths[i][h - 1] for i in active]
                actions = torch.as_tensor([e.action for e in edge_batch], dtype=torch.long, device=device)
                z_next = model.advance(z[active], actions)
                pred = model.head(z_next)
                for local_i, edge in enumerate(edge_batch):
                    by_h[h].append((edge, {k: v[local_i].float().cpu().numpy() for k, v in pred.items()}))
                z = z.clone()
                z[active] = z_next
    return by_h


def score_records(records, states: dict[str, State], train_edges: list[Edge]) -> dict:
    truth_mask, pred_mask, truth_conf, pred_conf = [], [], [], []
    truth_change, pred_change, truth_content, pred_content = [], [], [], []
    questions = set()
    for edge, pred in records:
        src, dst = states[edge.src], states[edge.dst]
        valid = edge.valid
        if valid.any():
            truth_mask.extend(dst.mask[valid].tolist())
            pred_mask.extend((1.0 / (1.0 + np.exp(-np.clip(pred["mask"][valid], -40, 40)))).tolist())
            truth_conf.extend(dst.confidence[valid].tolist())
            pred_conf.extend(pred["confidence"][valid].tolist())
            truth_content.extend(dst.content[valid].tolist())
            pred_content.extend(pred["content"][valid].tolist())
        change_valid = edge.changed_valid
        if change_valid.any():
            truth_change.extend(dst.changed[change_valid].tolist())
            prob = 1.0 / (1.0 + np.exp(-np.clip(pred["changed"][change_valid], -40, 40)))
            pred_change.extend(prob.tolist())
        questions.add(src.qid)
    if not truth_mask:
        return {"edge_count": len(records), "position_count": 0}
    tm, pm = np.asarray(truth_mask, bool), np.asarray(pred_mask, np.float64)
    tc, pc = np.asarray(truth_conf, np.float64), np.asarray(pred_conf, np.float64)
    result = {
        "edge_count": len(records), "question_count": len(questions), "position_count": len(tm),
        "next_mask_prevalence": float(tm.mean()),
        "mask_brier": float(np.mean((pm - tm) ** 2)),
        "mask_auc": _auc(tm, pm), "mask_average_precision": _average_precision(tm, pm),
        "mask_accuracy_at_0_5": float(np.mean((pm >= .5) == tm)),
        "confidence_mae": float(np.mean(np.abs(pc - tc))),
    }
    # Separate stable baselines with exactly the same scored positions.
    persisted_conf = np.concatenate([states[e.src].confidence[e.valid] for e, _ in records if e.valid.any()])
    result["confidence_persistence_mae"] = float(np.mean(np.abs(persisted_conf - tc)))
    if truth_change:
        ty, py = np.asarray(truth_change, bool), np.asarray(pred_change, np.float64)
        prevalence = float(ty.mean())
        train_labels = [states[e.dst].changed[e.changed_valid] for e in train_edges if e.changed_valid.any()]
        train_prevalence = float(np.concatenate(train_labels).mean()) if train_labels else None
        result.update({
            "candidate_change_prevalence": prevalence,
            "candidate_change_brier": float(np.mean((py - ty) ** 2)),
            "candidate_change_train_prevalence": train_prevalence,
            "candidate_change_train_constant_brier": (
                float(np.mean((train_prevalence - ty) ** 2)) if train_prevalence is not None else None),
            "candidate_change_auc": _auc(ty, py),
            "candidate_change_average_precision": _average_precision(ty, py),
        })
    else:
        result.update({"candidate_change_count": 0, "candidate_change_brier": None,
                       "candidate_change_auc": None})
    yt, yp = np.asarray(truth_content, np.float64), np.asarray(pred_content, np.float64)
    denom = np.linalg.norm(yt, axis=-1) * np.linalg.norm(yp, axis=-1)
    result["content_cosine"] = float(np.mean(np.sum(yt * yp, axis=-1) / np.maximum(denom, 1e-8)))
    result["mask_persistence_brier"] = float(np.mean((np.concatenate([
        states[e.src].mask[e.valid] for e, _ in records if e.valid.any()]).astype(float) - tm) ** 2))
    # Native current-confidence rule, including the forced argmax commit.
    native_probs = []
    for edge, _ in records:
        src = states[edge.src]
        active = edge.valid
        scores = src.confidence.copy()
        commit = (scores > .5) & active
        if active.any() and not commit.any():
            commit[np.flatnonzero(active)[np.argmax(scores[active])]] = True
        native_probs.extend((~commit[active]).astype(float).tolist())
    result["current_native_rule_mask_brier"] = float(np.mean((np.asarray(native_probs) - tm) ** 2))
    return result


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_direct(model, train_edges, val_edges, states, device, updates, seed,
                batch_size=16, eval_every=40, label="direct"):
    _seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best_state, stale = float("inf"), None, 0
    logs = []
    for step in range(1, updates + 1):
        model.train()
        picks = rng.integers(0, len(train_edges), size=min(batch_size, len(train_edges)))
        batch = [train_edges[int(i)] for i in picks]
        src = [states[e.src] for e in batch]
        x = torch.as_tensor(np.stack([s.x for s in src]), dtype=torch.float32, device=device)
        loss = edge_loss(model(x), batch, states, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == updates:
            if val_edges:
                metrics = score_records(_collect_direct(model, val_edges, states, device), states, train_edges)
                score = metrics["mask_brier"] + metrics["confidence_mae"]
                logs.append({"update": step, "train_loss": float(loss.detach().cpu()), "val_score": score})
                print(f"[{label}] update={step}/{updates} val_mask_brier={metrics['mask_brier']:.4f} "
                      f"val_conf_mae={metrics['confidence_mae']:.4f}", flush=True)
                if score < best_score:
                    best_score, best_state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    stale += 1
                    if stale >= 5:
                        break
            elif step % eval_every == 0:
                print(f"[{label}] update={step}/{updates} train_loss={float(loss.detach().cpu()):.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), logs


def _fit_dynamics(model, train_paths, val_paths, states, device, updates, seed,
                  horizon=3, batch_size=8, eval_every=40, label="dynamics"):
    _seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best_state, stale = float("inf"), None, 0
    logs = []
    trainable_paths = [path for path in train_paths if path]
    if not trainable_paths:
        raise ValueError("No paths available to train latent dynamics")
    for step in range(1, updates + 1):
        model.train()
        chosen = [trainable_paths[int(rng.integers(len(trainable_paths)))] for _ in range(batch_size)]
        x0 = torch.as_tensor(np.stack([states[path[0].src].x for path in chosen]),
                             dtype=torch.float32, device=device)
        z = model.encoder(x0)
        loss = z.sum() * 0.0
        used = 0
        for h in range(1, horizon + 1):
            active = [i for i, path in enumerate(chosen) if len(path) >= h]
            if not active:
                break
            edge_batch = [chosen[i][h - 1] for i in active]
            actions = torch.as_tensor([edge.action for edge in edge_batch], dtype=torch.long, device=device)
            z_next = model.advance(z[active], actions)
            pred = model.head(z_next)
            loss = loss + edge_loss(pred, edge_batch, states, device)
            next_z = z.clone()
            next_z[active] = z_next
            z = next_z
            used += 1
        loss = loss / max(1, used)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == updates:
            val_rollouts = _collect_rollout(model, val_paths, states, device, horizon)
            metrics = score_records(val_rollouts.get(1, []), states, train_paths[0] if False else [])
            score = metrics.get("mask_brier", 1.0) + metrics.get("confidence_mae", 1.0)
            logs.append({"update": step, "train_loss": float(loss.detach().cpu()), "val_score": score})
            print(f"[{label}] update={step}/{updates} val_mask_brier={metrics.get('mask_brier', float('nan')):.4f} "
                  f"val_conf_mae={metrics.get('confidence_mae', float('nan')):.4f}", flush=True)
            if score < best_score:
                best_score, best_state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= 5:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), logs


def _average_metric_dict(rows: list[dict]) -> dict:
    keys = sorted({key for row in rows for key, value in row.items()
                   if isinstance(value, (int, float)) and value is not None})
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            result[key] = float(np.mean(values))
            if len(values) > 1:
                result[key + "_std"] = float(np.std(values, ddof=1))
    return result


def _restrict_edges(edges, allowed: set[str], states):
    return [edge for edge in edges if states[edge.src].qid in allowed and states[edge.dst].qid in allowed]


def run_refine(path: Path, device, updates, seeds):
    states, edges, paths, info = load_refine(path)
    split = question_split([s.qid for s in states.values()])
    train_q = {q for q, part in split.items() if part == "train"}
    val_q = {q for q, part in split.items() if part == "validation"}
    test_q = {q for q, part in split.items() if part == "test"}
    normalize_states(states, train_q)
    train_edges = _restrict_edges(edges, train_q, states)
    val_edges = _restrict_edges(edges, val_q, states)
    test_edges = _restrict_edges(edges, test_q, states)
    edge_map = {(edge.src, edge.dst): edge for edge in edges}
    def convert_paths(qset):
        result = []
        for chain in paths:
            if not chain or states[chain[0]].qid not in qset:
                continue
            # Every state boundary is a valid rollout start. Keeping only the
            # first boundary per group would undercount H1 and waste most data.
            for start in range(len(chain) - 1):
                suffix = chain[start:]
                result.append([edge_map[(a, b)] for a, b in zip(suffix, suffix[1:])
                               if (a, b) in edge_map])
        return [p for p in result if p]
    train_paths, val_paths, test_paths = convert_paths(train_q), convert_paths(val_q), convert_paths(test_q)
    train_paths = [p for p in train_paths if p]
    val_paths = [p for p in val_paths if p]
    test_paths = [p for p in test_paths if p]
    in_dim, content_dim = next(iter(states.values())).x.shape[-1], next(iter(states.values())).content.shape[-1]
    info["split_question_counts"] = {part: sum(v == part for v in split.values())
                                     for part in ("train", "validation", "test")}
    info["split_edge_counts"] = {"train": len(train_edges), "validation": len(val_edges), "test": len(test_edges)}
    info["split_paths_h1_h3_counts"] = {
        part: {f"h{h}": sum(len(p) >= h for p in paths_part)
               for h in (1, 2, 3)}
        for part, paths_part in (("train", train_paths), ("validation", val_paths), ("test", test_paths))
    }
    result = {"data": info, "split_seed": SEED_SPLIT, "targets": [
        "next mask state", "next confidence", "candidate-change event", "native candidate embedding"],
        "arms": {}, "per_seed": {}, "training_logs": {}}
    arm_rows = {name: [] for name in ("rich_direct_h1", "latent_direct_h1", "latent_dynamics_h1_h3")}
    for seed in seeds:
        print(f"\n[Refine] seed={seed} training questions={len(train_q)} validation={len(val_q)} test={len(test_q)}", flush=True)
        models = {
            "rich_direct_h1": DirectPredictor(in_dim, 256, content_dim),
            "latent_direct_h1": DirectPredictor(in_dim, LATENT_DIM, content_dim),
            "latent_dynamics_h1_h3": LatentDynamics(in_dim, content_dim),
        }
        seed_metrics = {}
        for name in ("rich_direct_h1", "latent_direct_h1"):
            model, logs = _fit_direct(models[name], train_edges, val_edges, states, device,
                                      updates, seed, label=f"Refine/{name}/s{seed}")
            records = _collect_direct(model, test_edges, states, device)
            seed_metrics[name] = {"h1": score_records(records, states, train_edges)}
            result["training_logs"][f"{name}_seed{seed}"] = logs
            del model
        dyn, logs = _fit_dynamics(models["latent_dynamics_h1_h3"], train_paths, val_paths,
                                  states, device, updates, seed, horizon=3,
                                  label=f"Refine/latent_dynamics/s{seed}")
        rolled = _collect_rollout(dyn, test_paths, states, device, max_horizon=3)
        seed_metrics["latent_dynamics_h1_h3"] = {
            f"h{h}": score_records(rolled[h], states, train_edges) for h in range(1, 4)}
        result["training_logs"][f"latent_dynamics_h1_h3_seed{seed}"] = logs
        result["per_seed"][str(seed)] = seed_metrics
        for name, horizon_metrics in seed_metrics.items():
            for horizon, metrics in horizon_metrics.items():
                arm_rows[name].append({"seed": seed, "horizon": horizon, **metrics})
        del dyn, models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, rows in arm_rows.items():
        grouped = {}
        for row in rows:
            h = row["horizon"]
            grouped.setdefault(h, []).append(row)
        result["arms"][name] = {h: _average_metric_dict(hrows) for h, hrows in grouped.items()}
    result["interpretation_rules"] = {
        "compression_gap": "Compare rich_direct_h1 with latent_direct_h1 on the same held-out questions.",
        "transition_gap": "Compare latent_direct_h1 with latent_dynamics_h1_h3 at H1.",
        "rollout_error": "Compare latent_dynamics H1, H2, H3; metrics are free-running from the original state.",
        "scope": "Drafter state dynamics only; no token-ID generation, verifier acceptance, latency, or action-policy claim.",
    }
    return result


def run_extend(path: Path, device, updates):
    states, edges, info = load_extend(path)
    qids = sorted({s.qid for s in states.values()})
    in_dim, content_dim = next(iter(states.values())).x.shape[-1], next(iter(states.values())).content.shape[-1]
    # Leave one problem out. The five-fold result is explicitly a small-sample pilot.
    out = {"data": info, "validation": "leave-one-problem-out; one-step Extend only",
           "arms": {}, "folds": {}, "limitations": [
               "Five GSM8K problems only; report is exploratory, not a stable generalization estimate.",
               "Legacy E archive uses its recorded threshold/replay protocol, not a new native 0.5 collection.",
               "Only state features are predicted; this does not predict exact generated token IDs or verifier acceptance."]}
    per_arm = {name: [] for name in ("rich_direct_h1", "latent_direct_h1", "latent_dynamics_h1")}
    raw_x = {sid: state.x.copy() for sid, state in states.items()}
    for fold, heldout in enumerate(qids):
        train_q = set(qids) - {heldout}
        tr = _restrict_edges(edges, train_q, states)
        te = _restrict_edges(edges, {heldout}, states)
        for sid, value in raw_x.items():
            states[sid].x = value.copy()
        normalize_states(states, train_q)
        out["folds"][heldout] = {"train_edges": len(tr), "test_edges": len(te)}
        if not tr or not te:
            continue
        print(f"\n[Extend] heldout problem={heldout} train={len(tr)} test={len(te)}", flush=True)
        models = {
            "rich_direct_h1": DirectPredictor(in_dim, 256, content_dim),
            "latent_direct_h1": DirectPredictor(in_dim, LATENT_DIM, content_dim),
            "latent_dynamics_h1": LatentDynamics(in_dim, content_dim),
        }
        for name in ("rich_direct_h1", "latent_direct_h1"):
            model, _ = _fit_direct(models[name], tr, [], states, device, updates, 1000 + fold,
                                   label=f"Extend/{name}/fold{fold}")
            per_arm[name].append(score_records(_collect_direct(model, te, states, device), states, tr))
            del model
        # E graph is branched; one-step transition is evaluated without stitching unrelated branches.
        model = models["latent_dynamics_h1"]
        _seed(2000 + fold)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        rng = np.random.default_rng(2000 + fold)
        for step in range(1, updates + 1):
            batch = [tr[int(i)] for i in rng.integers(0, len(tr), size=min(16, len(tr)))]
            x = torch.as_tensor(np.stack([states[e.src].x for e in batch]), dtype=torch.float32, device=device)
            z = model.encoder(x)
            action = torch.as_tensor([e.action for e in batch], dtype=torch.long, device=device)
            loss = edge_loss(model.head(model.advance(z, action)), batch, states, device)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        records = []
        with torch.no_grad():
            model.eval()
            for start in range(0, len(te), 16):
                batch = te[start:start + 16]
                x = torch.as_tensor(np.stack([states[e.src].x for e in batch]), dtype=torch.float32, device=device)
                z = model.encoder(x)
                a = torch.as_tensor([e.action for e in batch], dtype=torch.long, device=device)
                pred = model.head(model.advance(z, a))
                for i, edge in enumerate(batch):
                    records.append((edge, {k: v[i].cpu().numpy() for k, v in pred.items()}))
        per_arm["latent_dynamics_h1"].append(score_records(records, states, tr))
        del model, models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["arms"] = {name: _average_metric_dict(rows) for name, rows in per_arm.items()}
    return out


def package_results(output: Path):
    archive = output.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != temporary:
                zf.write(file, file.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-zip", type=Path, required=True)
    parser.add_argument("--extend-zip", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    if args.updates < 1 or not args.seeds:
        raise ValueError("updates and seeds must be nonempty positive values")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle GPU accelerator before running this audit")
    device = torch.device("cuda:0")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, flush=True)
    print("No dLLM or verifier weights will be loaded; only captured observation archives are used.", flush=True)
    started = time.time()
    report = {
        "schema_version": "latent_sufficiency_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "config": {"updates_per_model": args.updates, "seeds": args.seeds,
                   "latent_dim": LATENT_DIM, "question_split": "70/15/15 fixed seed 42",
                   "refine_horizons": [1, 2, 3], "extend_validation": "5-fold question-disjoint H1"},
    }
    report["refine"] = run_refine(args.refine_zip, device, args.updates, args.seeds)
    if args.extend_zip:
        report["extend"] = run_extend(args.extend_zip, device, max(80, args.updates // 2))
    report["elapsed_seconds"] = time.time() - started
    (args.out_dir / "latent_sufficiency_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    readme = [
        "# Latent Drafter Sufficiency Audit", "",
        "This audit consumes existing capture ZIPs. It does not load dLLM/verifier weights or rerun generation.",
        "The main Refine experiment holds out whole questions and compares rich direct, predictive latent direct, and free-running latent transition.",
        "The optional Extend experiment is a separate five-question leave-one-problem-out H1 pilot.",
        "See latent_sufficiency_report.json for raw counts, per-seed results and limitations.", "",
        "Interpretation: rich-direct vs latent-direct diagnoses a 128D bottleneck; latent-direct vs transition H1 diagnoses the learned step; H1/H2/H3 diagnoses error accumulation.",
        "All reported outcomes are drafter observation dynamics. No metric here is verifier acceptance, exact next-token generation, or end-to-end latency.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    archive = package_results(args.out_dir)
    print("\nREPORT:", args.out_dir / "latent_sufficiency_report.json", flush=True)
    print("RESULT ZIP:", archive, f"({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print("ELAPSED:", f"{report['elapsed_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
