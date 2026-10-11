#!/usr/bin/env python3
"""Train/evaluate action-conditioned Fast-dLLM native latent dynamics (V2).

The learner predicts only native intermediate representations, a compressed
summary of the real prefix KV cache, and structural scheduler state. It has no
vocabulary head and never receives token IDs, token embeddings, token logits,
token confidence, verifier labels, accepted-token counts, or latency targets.
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
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fresh_native_drafter_test import collect as collect_native


SCHEMA = "native_latent_dynamics_v2"
BLOCK_SIZE = 32
KV_SHAPE = (4, 2, 2, 2, 128)
CONTEXT_SIZE = 7
ACTION_TO_ID = {"R": 0, "E": 1}


@dataclass
class State:
    sid: str
    qid: str
    u: np.ndarray
    memory: np.ndarray
    mask: np.ndarray
    event: np.ndarray
    active: np.ndarray
    context: np.ndarray
    memory_valid: bool
    meta: dict


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    action: int


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_questions(qids, seed=42):
    unique = sorted(set(map(str, qids)))
    if len(unique) < 5:
        raise ValueError(f"Need at least five questions with captured states, got {len(unique)}")
    order = np.random.default_rng(seed).permutation(unique)
    n_train = max(1, int(round(0.60 * len(unique))))
    n_val = max(1, int(round(0.20 * len(unique))))
    n_val = min(n_val, len(unique) - n_train - 1)
    result = {}
    for i, qid in enumerate(order):
        result[str(qid)] = "train" if i < n_train else (
            "validation" if i < n_train + n_val else "test"
        )
    return result


def load_teacher(root: Path):
    metadata = read_jsonl(root / "states.jsonl")
    grouped = defaultdict(list)
    for row in metadata:
        grouped[row["shard"]].append(row)
    states = {}
    for shard, rows in grouped.items():
        with np.load(root / shard, allow_pickle=False) as arrays:
            required = {
                "native_latent_block", "physical_mask", "physical_event",
                "active_region", "prefix_kv_summary", "prefix_kv_valid",
                "wm_v2_context",
            }
            missing = required.difference(arrays.files)
            if missing:
                raise RuntimeError(f"Native capture shard {shard} misses {sorted(missing)}")
            for row in rows:
                index = int(row["row"])
                state = State(
                    sid=str(row["state_id"]), qid=str(row["question_id"]),
                    u=arrays["native_latent_block"][index].astype(np.float32),
                    memory=arrays["prefix_kv_summary"][index].reshape(-1).astype(np.float32),
                    mask=arrays["physical_mask"][index].astype(np.float32),
                    event=arrays["physical_event"][index].astype(np.float32),
                    active=arrays["active_region"][index].astype(np.float32),
                    context=arrays["wm_v2_context"][index].astype(np.float32),
                    memory_valid=bool(arrays["prefix_kv_valid"][index]),
                    meta=row,
                )
                if state.u.ndim != 2 or state.u.shape[0] != BLOCK_SIZE:
                    raise ValueError(f"{state.sid}: latent must be [32, hidden], got {state.u.shape}")
                if state.memory.size != int(np.prod(KV_SHAPE)):
                    raise ValueError(f"{state.sid}: unexpected prefix-KV summary size {state.memory.size}")
                if any(x.shape != (BLOCK_SIZE,) for x in (state.mask, state.event, state.active)):
                    raise ValueError(f"{state.sid}: physical structural arrays must have 32 positions")
                if state.context.shape != (CONTEXT_SIZE,):
                    raise ValueError(f"{state.sid}: expected {CONTEXT_SIZE} scheduler features")
                if not all(np.isfinite(x).all() for x in
                           (state.u, state.memory, state.mask, state.event, state.active, state.context)):
                    raise ValueError(f"{state.sid}: non-finite teacher feature")
                states[state.sid] = state

    edges = []
    for row in read_jsonl(root / "edges.jsonl"):
        src, dst, action = str(row["src_state_id"]), str(row["dst_state_id"]), str(row["action"])
        if src in states and dst in states and action in ACTION_TO_ID:
            if states[src].qid != states[dst].qid:
                raise RuntimeError("An R/E edge crosses question IDs")
            edges.append(Edge(src, dst, ACTION_TO_ID[action]))
    if not states or not edges:
        raise RuntimeError(f"Insufficient native data: {len(states)} states, {len(edges)} edges")
    return states, edges


def build_paths(states, edges, split, part, max_horizon=3):
    outgoing = defaultdict(list)
    for edge in edges:
        if split.get(states[edge.src].qid) == part and split.get(states[edge.dst].qid) == part:
            outgoing[edge.src].append(edge)
    paths = []
    for state in states.values():
        if split.get(state.qid) != part:
            continue
        chain, current = [], state.sid
        while len(chain) < max_horizon:
            children = outgoing.get(current, ())
            if len(children) != 1:
                break
            chain.append(children[0])
            current = children[0].dst
        if chain:
            paths.append((state.sid, chain))
    return paths


def fit_scalers(states, split):
    train = [s for s in states.values() if split[s.qid] == "train"]
    if not train:
        raise ValueError("Empty training split")
    u_sum = np.zeros(train[0].u.shape[-1], dtype=np.float64)
    u_sq = np.zeros_like(u_sum)
    u_count = 0
    m_sum = np.zeros(train[0].memory.shape, dtype=np.float64)
    m_sq = np.zeros_like(m_sum)
    m_count = 0
    for state in train:
        u = state.u.astype(np.float64).reshape(-1, state.u.shape[-1])
        u_sum += u.sum(0)
        u_sq += np.square(u).sum(0)
        u_count += len(u)
        if state.memory_valid:
            m = state.memory.astype(np.float64)
            m_sum += m
            m_sq += np.square(m)
            m_count += 1
    u_mean = u_sum / max(1, u_count)
    u_std = np.sqrt(np.maximum(u_sq / max(1, u_count) - u_mean * u_mean, 1e-8))
    m_mean = m_sum / max(1, m_count)
    m_std = np.sqrt(np.maximum(m_sq / max(1, m_count) - m_mean * m_mean, 1e-8))
    return {
        "u_mean": u_mean.astype(np.float32), "u_std": u_std.astype(np.float32),
        "memory_mean": m_mean.astype(np.float32), "memory_std": m_std.astype(np.float32),
    }


def pack_state(state, scalers, device):
    u = np.clip((state.u - scalers["u_mean"]) / scalers["u_std"], -12, 12)
    memory = np.clip((state.memory - scalers["memory_mean"]) / scalers["memory_std"], -12, 12)
    return {
        "u": torch.as_tensor(u[None], dtype=torch.float32, device=device),
        "memory": torch.as_tensor(memory[None], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(state.mask[None], dtype=torch.float32, device=device),
        "event": torch.as_tensor(state.event[None], dtype=torch.float32, device=device),
        "active": torch.as_tensor(state.active[None], dtype=torch.float32, device=device),
        "context": torch.as_tensor(state.context[None], dtype=torch.float32, device=device),
        "memory_valid": torch.as_tensor([float(state.memory_valid)], dtype=torch.float32, device=device),
    }


def stack_states(ids, states, scalers, device):
    packed = [pack_state(states[sid], scalers, device) for sid in ids]
    return {key: torch.cat([row[key] for row in packed], dim=0) for key in packed[0]}


class NativeLatentDynamics(nn.Module):
    """Action-conditioned latent transition in the dLLM's native residual space."""

    def __init__(self, hidden_size: int, memory_size: int, dim: int = 128,
                 layers: int = 3, feature_set: str = "full"):
        super().__init__()
        if feature_set not in {"native_only", "native_structure", "full"}:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.feature_set = feature_set
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.u_in = nn.Linear(hidden_size, dim)
        self.structure_in = nn.Linear(3, dim, bias=False)
        self.memory_in = nn.Sequential(nn.LayerNorm(memory_size), nn.Linear(memory_size, dim), nn.GELU())
        self.context_in = nn.Sequential(nn.Linear(CONTEXT_SIZE, dim), nn.GELU(), nn.Linear(dim, dim))
        self.position = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.01)
        self.action = nn.Embedding(2, dim)
        self.action_affine = nn.Embedding(2, dim * 2)
        nn.init.normal_(self.action.weight, std=0.02)
        nn.init.zeros_(self.action_affine.weight)
        self.encoder_norm = nn.LayerNorm(dim)
        enc_layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 4, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, 2, enable_nested_tensor=False)
        dyn_layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 4, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(dyn_layer, layers, enable_nested_tensor=False)
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.transition_norm = nn.LayerNorm(dim)
        self.u_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_size))
        self.memory_out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, memory_size),
        )
        self.mask_out = nn.Linear(dim, 1)
        self.event_out = nn.Linear(dim, 1)
        self.active_out = nn.Linear(dim, 1)
        self.context_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, CONTEXT_SIZE))

    def encode(self, state):
        pieces = self.u_in(state["u"]) + self.position
        if self.feature_set != "native_only":
            structure = torch.stack((state["mask"], state["event"], state["active"]), dim=-1)
            pieces = pieces + self.structure_in(structure)
            pieces = pieces + self.context_in(state["context"]).unsqueeze(1)
        if self.feature_set == "full":
            memory_latent = self.memory_in(state["memory"])
            pieces = pieces + (memory_latent * state["memory_valid"].unsqueeze(-1)).unsqueeze(1)
        return self.encoder_norm(self.encoder(pieces))

    def transition(self, z, action):
        action_embed = self.action(action).unsqueeze(1)
        scale, shift = self.action_affine(action).chunk(2, dim=-1)
        prepared = z * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        h = self.dynamics(prepared + action_embed)
        return self.transition_norm(z + self.delta(h))

    def decode(self, z):
        pooled = z.mean(dim=1)
        return {
            "u": self.u_out(z),
            "memory": self.memory_out(pooled),
            "mask": self.mask_out(z).squeeze(-1),
            "event": self.event_out(z).squeeze(-1),
            "active": self.active_out(z).squeeze(-1),
            "context": self.context_out(pooled),
        }

    def step(self, z, action):
        z_next = self.transition(z, action)
        return z_next, self.decode(z_next)


def state_target_batch(ids, states, scalers, device):
    return stack_states(ids, states, scalers, device)


def supervised_loss(pred, target, memory_valid):
    loss_u = F.smooth_l1_loss(pred["u"], target["u"])
    memory_mask = memory_valid.float().view(-1, 1)
    memory_error = F.smooth_l1_loss(pred["memory"], target["memory"], reduction="none")
    loss_memory = (memory_error * memory_mask).sum() / (memory_mask.sum() * pred["memory"].shape[-1]).clamp_min(1)
    loss_mask = F.binary_cross_entropy_with_logits(pred["mask"], target["mask"])
    loss_event = F.binary_cross_entropy_with_logits(pred["event"], target["event"])
    loss_active = F.binary_cross_entropy_with_logits(pred["active"], target["active"])
    loss_context = F.smooth_l1_loss(pred["context"], target["context"])
    total = loss_u + 0.25 * loss_memory + 0.15 * loss_mask + 0.15 * loss_event + 0.10 * loss_active + 0.10 * loss_context
    return total, {
        "u": float(loss_u.detach()), "memory": float(loss_memory.detach()),
        "mask": float(loss_mask.detach()), "event": float(loss_event.detach()),
        "active": float(loss_active.detach()), "context": float(loss_context.detach()),
    }


def fit_model(model, paths, states, scalers, device, updates=300, batch_size=12,
              seed=42, eval_callback=None):
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    viable = [path for path in paths if path[1]]
    if not viable:
        raise RuntimeError("Training split contains no R/E paths")
    curve = []
    checkpoints = sorted(set([max(1, updates // 4), max(1, updates // 2), updates]))
    for step in range(1, updates + 1):
        model.train()
        picks = rng.integers(0, len(viable), size=min(batch_size, len(viable)))
        batch_paths = [viable[int(i)] for i in picks]
        root_ids = [root for root, _ in batch_paths]
        state = stack_states(root_ids, states, scalers, device)
        z = model.encode(state)
        total_loss = torch.zeros((), device=device)
        loss_parts = defaultdict(float)
        max_h = max(len(path) for _, path in batch_paths)
        for depth in range(max_h):
            active = [i for i, (_, path) in enumerate(batch_paths) if len(path) > depth]
            if not active:
                break
            indices = torch.as_tensor(active, dtype=torch.long, device=device)
            edges = [batch_paths[i][1][depth] for i in active]
            actions = torch.as_tensor([edge.action for edge in edges], dtype=torch.long, device=device)
            z_child, pred = model.step(z.index_select(0, indices), actions)
            target = state_target_batch([edge.dst for edge in edges], states, scalers, device)
            mem_valid = torch.as_tensor(
                [states[edge.dst].memory_valid for edge in edges], dtype=torch.bool, device=device
            )
            step_loss, parts = supervised_loss(pred, target, mem_valid)
            total_loss = total_loss + step_loss / max_h
            for key, value in parts.items():
                loss_parts[key] += value / max_h
            z = z.index_copy(0, indices, z_child)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            row = {"update": step, "train_loss": float(total_loss.detach()),
                   "loss_components": dict(loss_parts)}
            if eval_callback:
                row["validation"] = eval_callback(model)
            curve.append(row)
            print(f"[train:{model.feature_set}] update={step}/{updates} "
                  f"loss={row['train_loss']:.5f}", flush=True)
    return curve


def _f1(target, pred):
    target = np.asarray(target, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(target & pred)); fp = int(np.sum(~target & pred)); fn = int(np.sum(target & ~pred))
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def _state_metrics(pred, target, source, scalers, memory_valid, predicted_source=None):
    pu = pred["u"]
    tu = target["u"]
    su = source["u"]
    diff = pu - tu
    out = {
        "latent_nrmse": float(np.sqrt(np.mean(np.square(diff)))),
        "latent_nmae": float(np.mean(np.abs(diff))),
        "latent_native_rmse": float(np.sqrt(np.mean(np.square(diff * scalers["u_std"])))),
    }
    pflat, tflat = pu.reshape(-1), tu.reshape(-1)
    denom = np.linalg.norm(pflat) * np.linalg.norm(tflat)
    out["latent_cosine"] = float(np.dot(pflat, tflat) / denom) if denom > 1e-12 else None
    true_delta = tu - su
    predicted_source = source if predicted_source is None else predicted_source
    pred_delta = pu - predicted_source["u"]
    delta_norm = float(np.linalg.norm(true_delta))
    out["delta_relative_error"] = (
        float(np.linalg.norm(pred_delta - true_delta) / delta_norm) if delta_norm > 1e-6 else None
    )
    out["mask_accuracy"] = float(np.mean((pred["mask"] >= 0.5) == (target["mask"] >= 0.5)))
    out["mask_f1"] = _f1(target["mask"] >= 0.5, pred["mask"] >= 0.5)
    out["commit_event_f1"] = _f1(target["event"] >= 0.5, pred["event"] >= 0.5)
    out["active_region_f1"] = _f1(target["active"] >= 0.5, pred["active"] >= 0.5)
    out["scheduler_context_mae"] = float(np.mean(np.abs(pred["context"] - target["context"])))
    out["native_eos_accuracy"] = float(
        (pred["context"][5] >= 0.5) == (target["context"][5] >= 0.5)
    )
    out["stop_fill_eos_accuracy"] = float(
        (pred["context"][6] >= 0.5) == (target["context"][6] >= 0.5)
    )
    target_std = float(np.std(tu))
    out["latent_prediction_std_ratio"] = (
        float(np.std(pu) / target_std) if target_std > 1e-8 else None
    )
    if memory_valid:
        dm = pred["memory"] - target["memory"]
        out["prefix_kv_nrmse"] = float(np.sqrt(np.mean(np.square(dm))))
        out["prefix_kv_nmae"] = float(np.mean(np.abs(dm)))
        out["prefix_kv_native_rmse"] = float(np.sqrt(np.mean(np.square(dm * scalers["memory_std"]))))
        true_memory_delta = target["memory"] - source["memory"]
        pred_memory_delta = pred["memory"] - predicted_source["memory"]
        true_memory_norm = float(np.linalg.norm(true_memory_delta))
        out["prefix_kv_delta_relative_error"] = (
            float(np.linalg.norm(pred_memory_delta - true_memory_delta) / true_memory_norm)
            if true_memory_norm > 1e-6 else None
        )
    else:
        out["prefix_kv_nrmse"] = None
        out["prefix_kv_nmae"] = None
        out["prefix_kv_native_rmse"] = None
        out["prefix_kv_delta_relative_error"] = None
    return out


def aggregate(records):
    if not records:
        return {"n_predictions": 0}
    keys = [k for k in records[0]["metrics"] if k not in
            ("delta_relative_error", "prefix_kv_delta_relative_error")]
    result = {"n_predictions": len(records)}
    for key in keys:
        vals = [r["metrics"][key] for r in records if r["metrics"].get(key) is not None]
        result[key] = float(np.mean(vals)) if vals else None
    delta = [r["metrics"]["delta_relative_error"] for r in records
             if r["metrics"].get("delta_relative_error") is not None]
    result["delta_relative_error_mean"] = float(np.mean(delta)) if delta else None
    result["delta_relative_error_median"] = float(np.median(delta)) if delta else None
    result["n_delta_nontrivial"] = len(delta)
    memory_delta = [r["metrics"]["prefix_kv_delta_relative_error"] for r in records
                    if r["metrics"].get("prefix_kv_delta_relative_error") is not None]
    result["prefix_kv_delta_relative_error_mean"] = (
        float(np.mean(memory_delta)) if memory_delta else None
    )
    result["prefix_kv_delta_nontrivial"] = len(memory_delta)
    return result


@torch.no_grad()
def evaluate(model, paths, states, scalers, device):
    model.eval()
    records = []
    for root, chain in paths:
        current = stack_states([root], states, scalers, device)
        z = model.encode(current)
        previous_prediction = None
        for depth, edge in enumerate(chain, 1):
            action = torch.as_tensor([edge.action], dtype=torch.long, device=device)
            z, output = model.step(z, action)
            target = pack_state(states[edge.dst], scalers, device)
            source = pack_state(states[edge.src], scalers, device)
            pred_u = output["u"][0].cpu().numpy()
            pred_memory = output["memory"][0].cpu().numpy()
            pred_mask = output["mask"][0].sigmoid().cpu().numpy()
            pred_event = output["event"][0].sigmoid().cpu().numpy()
            pred_active = output["active"][0].sigmoid().cpu().numpy()
            pred_context = output["context"][0].cpu().numpy()
            target_np = {key: value[0].cpu().numpy() for key, value in target.items()}
            source_np = {key: value[0].cpu().numpy() for key, value in source.items()}
            pred_np = {
                "u": pred_u, "memory": pred_memory,
                "mask": pred_mask, "event": pred_event,
                "active": pred_active, "context": pred_context,
            }
            predicted_source = source_np if previous_prediction is None else previous_prediction
            metrics = _state_metrics(
                pred_np, target_np, source_np, scalers, states[edge.dst].memory_valid,
                predicted_source=predicted_source,
            )
            records.append({"qid": states[edge.dst].qid, "src": edge.src, "dst": edge.dst,
                            "action": "E" if edge.action else "R", "horizon": depth,
                            "metrics": metrics})
            previous_prediction = pred_np
            # Roll the predicted latent, not the true child features. The action
            # sequence is supplied to isolate transition learning from policy.
    return records


def persistence_baseline(paths, states, scalers):
    rows = []
    for root, chain in paths:
        source = pack_state(states[root], scalers, torch.device("cpu"))
        source_np = {key: value[0].numpy() for key, value in source.items()}
        for depth, edge in enumerate(chain, 1):
            target = pack_state(states[edge.dst], scalers, torch.device("cpu"))
            target_np = {key: value[0].numpy() for key, value in target.items()}
            pred = {
                "u": source_np["u"], "memory": source_np["memory"],
                "mask": source_np["mask"], "event": np.zeros_like(source_np["event"]),
                "active": source_np["active"], "context": source_np["context"],
            }
            true_source = pack_state(states[edge.src], scalers, torch.device("cpu"))
            true_source_np = {key: value[0].numpy() for key, value in true_source.items()}
            rows.append({"qid": states[edge.dst].qid, "action": "E" if edge.action else "R",
                         "horizon": depth, "metrics": _state_metrics(
                             pred, target_np, true_source_np, scalers,
                             states[edge.dst].memory_valid, predicted_source=source_np,
                         )})
    return rows


def grouped_report(records):
    def summarize(rows):
        result = aggregate(rows)
        per_question = {
            qid: aggregate([row for row in rows if row["qid"] == qid])
            for qid in sorted({row["qid"] for row in rows})
        }
        result["question_count"] = len(per_question)
        metric_names = [key for key, value in result.items()
                        if key not in {"n_predictions", "question_count"}
                        and isinstance(value, (int, float))]
        result["question_macro"] = {
            key: float(np.mean([summary[key] for summary in per_question.values()
                                if summary.get(key) is not None]))
            if any(summary.get(key) is not None for summary in per_question.values())
            else None
            for key in metric_names
        }
        return result

    groups = {}
    for action in ("R", "E"):
        for horizon in (1, 2, 3):
            key = f"action_{action}_h{horizon}"
            groups[key] = summarize([
                r for r in records if r["action"] == action and r["horizon"] == horizon
            ])
    groups["all"] = summarize(records)
    return groups


def sanitize_capture(capture: Path, out: Path, states, edges, split, scalers):
    """Write only permitted native latent/structural features into final ZIP."""
    dest = out / "teacher_features"
    dest.mkdir(parents=True, exist_ok=True)
    meta_rows = sorted(states.values(), key=lambda s: (s.qid, s.sid))
    shard_size = 64
    state_meta = []
    for start in range(0, len(meta_rows), shard_size):
        rows = meta_rows[start:start + shard_size]
        name = f"latent_{start // shard_size:04d}.npz"
        np.savez_compressed(
            dest / name,
            native_residual=np.stack([s.u for s in rows]).astype(np.float16),
            prefix_kv_summary=np.stack([s.memory.reshape(KV_SHAPE) for s in rows]).astype(np.float16),
            physical_mask=np.stack([s.mask for s in rows]).astype(np.bool_),
            commit_event=np.stack([s.event for s in rows]).astype(np.bool_),
            active_region=np.stack([s.active for s in rows]).astype(np.bool_),
            scheduler_context=np.stack([s.context for s in rows]).astype(np.float32),
            prefix_kv_valid=np.asarray([s.memory_valid for s in rows], dtype=np.bool_),
        )
        for offset, state in enumerate(rows):
            m = state.meta
            state_meta.append({
                "state_id": state.sid, "question_id": state.qid,
                "parent_state_id": m.get("parent_state_id"),
                "action_from_parent": m.get("action_from_parent"),
                "block_index": int(m.get("block_index", 0)),
                "refine_index": int(m.get("refine_index", 0)),
                "native_forward_index": int(m.get("native_unmask_forward_index", 0)),
                "physical_block_start": m.get("physical_block_start"),
                "active_region_offset": m.get("active_region_offset"),
                "prefix_kv_length": m.get("prefix_kv_length"),
                "prefix_kv_valid": state.memory_valid,
                "split": split[state.qid], "shard": name, "row": offset,
            })
    (dest / "states.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in state_meta), encoding="utf-8"
    )
    clean_edges = [{"src_state_id": edge.src, "dst_state_id": edge.dst,
                    "action": "E" if edge.action else "R"}
                   for edge in edges]
    (dest / "edges.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in clean_edges), encoding="utf-8"
    )
    np.savez_compressed(dest / "train_only_normalization.npz", **scalers)


def train_and_audit(args, states, edges, split, out):
    device = torch.device(args.train_device)
    hidden_size = next(iter(states.values())).u.shape[-1]
    memory_size = next(iter(states.values())).memory.size
    scalers = fit_scalers(states, split)
    train_paths = build_paths(states, edges, split, "train", args.horizon)
    val_paths = build_paths(states, edges, split, "validation", args.horizon)
    test_paths = build_paths(states, edges, split, "test", args.horizon)
    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError(
            f"Need nonempty train/validation/test paths; got {len(train_paths)}/"
            f"{len(val_paths)}/{len(test_paths)}"
        )
    variants = ("native_only", "native_structure", "full")
    report = {
        "schema": SCHEMA,
        "objective": "predict native residual/prefix-KV-summary/structural feature dynamics after R/E",
        "excluded_targets_and_inputs": [
            "token IDs", "token embeddings", "vocabulary logits", "token probabilities",
            "confidence", "margin", "entropy", "verifier labels", "accepted-token counts", "latency",
        ],
        "native_cut": {"layer_index": args.cut_layer, "semantic": "input residual to the final four native dLLM decoder layers"},
        "capture": {"physical_block": BLOCK_SIZE, "scheduler_subblock": 8,
                    "prefix_kv": "summary of mean K/V over last 128 prefix positions and last 8, suffix layers only; not full KV cache"},
        "data": {"states": len(states), "edges": len(edges),
                 "action_edges": {a: sum(e.action == aid for e in edges) for a, aid in ACTION_TO_ID.items()},
                 "questions": {part: sum(value == part for value in split.values())
                               for part in ("train", "validation", "test")}},
        "split_by_question": split,
        "config": {"dim": args.latent_dim, "layers": args.model_layers,
                   "updates_per_variant": args.updates, "batch_size": args.batch_size,
                   "horizon": args.horizon, "seed": args.seed},
        "source_commit": __import__("os").environ.get("WM_V2_SOURCE_COMMIT"),
        "variants": {},
    }
    all_groups = {}
    test_metric_rows = []
    for variant_index, variant in enumerate(variants):
        torch.manual_seed(args.seed + variant_index * 101)
        model = NativeLatentDynamics(hidden_size, memory_size, args.latent_dim,
                                     args.model_layers, variant)
        def validation(model_to_eval):
            rows = evaluate(model_to_eval, val_paths, states, scalers, device)
            return grouped_report(rows)
        started = time.time()
        curve = fit_model(
            model, train_paths, states, scalers, device,
            updates=args.updates, batch_size=args.batch_size,
            seed=args.seed + len(all_groups) * 101, eval_callback=validation,
        )
        test_rows = evaluate(model, test_paths, states, scalers, device)
        test_report = grouped_report(test_rows)
        test_metric_rows.extend({"model": variant, **row} for row in test_rows)
        report["variants"][variant] = {
            "training_curve": curve,
            "test": test_report,
            "elapsed_seconds": time.time() - started,
        }
        all_groups[variant] = test_rows
        torch.save({"schema": SCHEMA, "feature_set": variant,
                    "state_dict": model.cpu().state_dict()}, out / f"model_{variant}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = persistence_baseline(test_paths, states, scalers)
    report["persistence_baseline_test"] = grouped_report(baseline)
    test_metric_rows.extend({"model": "persistence_baseline", **row} for row in baseline)
    (out / "test_rollout_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
               for row in test_metric_rows), encoding="utf-8"
    )
    report["interpretation_guide"] = {
        "latent_nrmse": "per-coordinate RMSE after train-only channel normalization; 0 is exact, about 1 means error comparable to a training standard deviation",
        "latent_native_rmse": "RMSE in native residual units",
        "delta_relative_error_mean": "error in predicted state change divided by actual change magnitude; excludes near-zero true changes",
        "prefix_kv_nrmse": "error in the stored compressed prefix-KV summary, not full-cache equivalence",
        "horizon": "number of free-running R/E transitions from a real teacher root; child teacher observations are never fed back",
        "action_sequence": "teacher R/E action sequence is supplied; this isolates dynamics from action-policy learning",
        "feature_ablation": "native_only vs native_structure isolates geometry/mask/event/context; full additionally includes the prefix-KV summary",
    }
    (out / "latent_dynamics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report, scalers


def package(out: Path):
    archive = out.with_suffix(".zip")
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != temp:
                zf.write(path, path.relative_to(out).as_posix())
    temp.replace(archive)
    return archive


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dllm-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--extend-size", type=int, default=8)
    parser.add_argument("--max-proposal-tokens", type=int, default=64)
    parser.add_argument("--max-refinement-steps", type=int, default=3)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--model-layers", type=int, default=3)
    parser.add_argument("--cut-layer", type=int, default=24)
    parser.add_argument("--suffix-layers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--drafter-device", type=int, default=0)
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--raw-top-k", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_questions < 20:
        raise ValueError("Use >=20 questions for a meaningful question-disjoint 60/20/20 split")
    if args.extend_size != 8 or args.max_proposal_tokens != 64 or args.max_refinement_steps > 3:
        raise ValueError("V2 is pinned to E=8, max proposal=64, and at most three R steps per segment")
    if args.horizon not in (1, 2, 3):
        raise ValueError("Supported rollout horizon is 1, 2, or 3")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle GPU T4 x2: native capture on GPU0 and latent training on GPU1")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    capture_dir = args.out_dir.parent / (args.out_dir.name + "_raw_capture")
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True)
    capture_args = argparse.Namespace(
        out_dir=capture_dir, num_questions=args.num_questions, seed=args.seed,
        threshold=args.threshold, extend_size=args.extend_size,
        max_proposal_tokens=args.max_proposal_tokens,
        max_refinement_steps=args.max_refinement_steps,
        physical_block_size=32, small_block_size=8, raw_top_k=args.raw_top_k,
        drafter_device=args.drafter_device, shard_rows=16,
        tokenizer_name=args.tokenizer_name, dllm_dir=args.dllm_dir,
        wm_v2_capture_native_latent=True, wm_v2_cut_layer=args.cut_layer,
        wm_v2_suffix_layers=args.suffix_layers, wm_v2_memory_window=128,
    )
    started = time.time()
    try:
        print("[capture] Fresh greedy native Fast-dLLM R/E trajectories; no verifier is loaded.", flush=True)
        capture_manifest = collect_native(capture_args)
        states, edges = load_teacher(capture_dir)
        split = split_questions({s.qid for s in states.values()}, args.split_seed)
        print(f"[capture-done] states={len(states)} edges={len(edges)} "
              f"R={sum(e.action == 0 for e in edges)} E={sum(e.action == 1 for e in edges)}", flush=True)
        report, scalers = train_and_audit(args, states, edges, split, args.out_dir)
        sanitize_capture(capture_dir, args.out_dir, states, edges, split, scalers)
        report["elapsed_total_seconds"] = time.time() - started
        report["capture_manifest_summary"] = {
            "question_count": capture_manifest.get("question_count"),
            "state_count": capture_manifest.get("state_count"),
            "edge_count": capture_manifest.get("edge_count"),
            "action_counts": capture_manifest.get("action_counts"),
            "terminal_counts": capture_manifest.get("terminal_counts"),
        }
        (args.out_dir / "latent_dynamics_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        (args.out_dir / "README.md").write_text(
            "# Native latent-dynamics V2 feasibility test\n\n"
            "Fresh GSM8K Fast-dLLM native trajectories, threshold 0.5, physical block 32, "
            "logical E=8, cap 64, and up to three sampled R transitions per segment. "
            "The model predicts only cut-layer residual dynamics, a compressed prefix-KV "
            "summary, and scheduler mask/event features. No token or verifier target is used. "
            "See latent_dynamics_report.json for question-disjoint H1-H3 metrics and feature ablations.\n",
            encoding="utf-8",
        )
        archive = package(args.out_dir)
        print(f"[result-zip] {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    except Exception as exc:
        (args.out_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        if any(args.out_dir.iterdir()):
            try:
                print(f"[partial-zip] {package(args.out_dir)}", flush=True)
            except Exception as package_exc:
                print(f"[partial-zip-skipped] {package_exc}", flush=True)
        raise
    finally:
        if capture_dir.exists():
            shutil.rmtree(capture_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
