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
import traceback
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fresh_native_drafter_test import collect as collect_native
from native_latent_rules_v2 import (
    align_frame, constrained_commit, initial_segment_mask, next_geometry, proposal_span, scheduler_region,
)


SCHEMA = "native_latent_dynamics_v2_native_rules_v3"
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
    geometry: np.ndarray | None = None
    mask_before: np.ndarray | None = None
    native_active: np.ndarray | None = None
    forward_active: np.ndarray | None = None
    valid: np.ndarray | None = None


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


def load_teacher(root: Path, require_edges=True):
    if (root / "teacher_features").is_dir():
        root = root / "teacher_features"
    sanitized = (root / "train_only_normalization.npz").is_file()
    aliases = {"native_latent_block": "native_residual", "physical_event": "commit_event",
               "wm_v2_context": "scheduler_context", "wm_v2_geometry": "geometry",
               "wm_v2_valid_positions": "valid_positions"} if sanitized else {}
    metadata = read_jsonl(root / "states.jsonl")
    grouped = defaultdict(list)
    for row in metadata:
        grouped[row["shard"]].append(row)
    states = {}
    for shard, rows in grouped.items():
        with np.load(root / shard, allow_pickle=False) as source_arrays:
            required = {
                "native_latent_block", "physical_mask", "physical_event",
                "active_region", "prefix_kv_summary", "prefix_kv_valid",
                "wm_v2_context",
                "wm_v2_geometry", "mask_before_forward", "proposal_span",
                "native_active_region", "forward_active_region",
                "wm_v2_valid_positions",
            }
            missing = [key for key in required if aliases.get(key, key) not in source_arrays.files]
            if missing:
                raise RuntimeError(f"Native capture shard {shard} misses {sorted(missing)}")
            arrays = {key: source_arrays[aliases.get(key, key)] for key in required}
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
                    geometry=arrays["wm_v2_geometry"][index].astype(np.int64),
                    mask_before=arrays["mask_before_forward"][index].astype(np.float32),
                    native_active=arrays["native_active_region"][index].astype(np.float32),
                    forward_active=arrays["forward_active_region"][index].astype(np.float32),
                    valid=arrays["wm_v2_valid_positions"][index].astype(np.float32),
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
                if state.geometry.shape != (5,) or any(
                    x.shape != (BLOCK_SIZE,) for x in (state.mask_before, state.native_active, state.forward_active)
                ):
                    raise ValueError(f"{state.sid}: invalid native geometry/scheduler capture")
                expected_span = proposal_span(torch.as_tensor(state.geometry[None]))[0].numpy()
                if not np.array_equal(expected_span, arrays["proposal_span"][index]):
                    raise ValueError(f"{state.sid}: proposal span does not match absolute geometry")
                states[state.sid] = state

    edges = []
    edge_path = root / "edges.jsonl"
    edge_rows = read_jsonl(edge_path) if edge_path.exists() or require_edges else []
    for row in edge_rows:
        src, dst, action = str(row["src_state_id"]), str(row["dst_state_id"]), str(row["action"])
        if src not in states or dst not in states or action not in ACTION_TO_ID:
            raise RuntimeError(f"Broken/unknown native edge: {row}")
        if src in states and dst in states:
            if states[src].qid != states[dst].qid:
                raise RuntimeError("An R/E edge crosses question IDs")
            edges.append(Edge(src, dst, ACTION_TO_ID[action]))
    if not states or (require_edges and not edges):
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


def audit_native_contract(states, edges):
    """Fail closed on captured native-rule violations before training."""
    counts = defaultdict(int)
    for s in states.values():
        geometry = torch.as_tensor(s.geometry[None])
        before = s.mask_before >= .5
        after = s.mask >= .5
        if np.any(after & ~before):
            raise ValueError(f"{s.sid}: native capture remasks a committed position")
        if not np.array_equal(s.event >= .5, before & ~after):
            raise ValueError(f"{s.sid}: full physical event != same-forward pre/post mask difference")
        expected_region = scheduler_region(torch.as_tensor(s.mask_before[None]), geometry)[0].numpy()
        if not np.array_equal(expected_region, s.forward_active >= .5):
            raise ValueError(f"{s.sid}: forward region differs from native first remaining 8-subblock")
        eligible = before & expected_region
        if not eligible.any() or not (s.event >= .5).any() or np.any((s.event >= .5) & ~eligible):
            raise ValueError(f"{s.sid}: native commit violates eligibility/progress rule")
        if int(s.geometry[1]) != (int(s.geometry[0]) + int(s.geometry[3]) - 1) // 32 * 32:
            raise ValueError(f"{s.sid}: snapshot is in a different physical frame")
        refine_index = int(s.meta.get("refine_index", 0))
        if refine_index == 0 and not np.array_equal(before, initial_segment_mask(geometry)[0].numpy()):
            # Do not turn an unverified E-pre-mask assumption into a hard rule.
            raise ValueError(f"{s.sid}: first exposed pre-mask differs from native seeded segment geometry")
        counts["valid_physical_snapshots"] += 1
        counts["partially_visible_proposal"] += int(proposal_span(geometry).sum() < 8)
    for edge in edges:
        source, child = states[edge.src], states[edge.dst]
        planned = next_geometry(torch.as_tensor(source.geometry[None]), torch.tensor([edge.action]))[0].numpy()
        if not np.array_equal(planned, child.geometry):
            raise ValueError(f"{edge.dst}: R/E geometry differs from native segment contract")
        if edge.action == 0:
            if not np.array_equal(source.mask, child.mask_before):
                raise ValueError(f"{edge.dst}: adjacent R pre-mask is not parent's post-mask")
            if source.memory_valid and child.memory_valid and not np.array_equal(source.memory, child.memory):
                raise ValueError(f"{edge.dst}: same-block R changed prefix KV")
            a = int(source.meta.get("native_unmask_forward_index", source.meta.get("native_forward_index", 0)))
            b = int(child.meta.get("native_unmask_forward_index", child.meta.get("native_forward_index", 0)))
            if a and b and b != a + 1:
                raise ValueError(f"{edge.dst}: R skips an unobserved native unmask forward")
            counts["valid_adjacent_R"] += 1
        else:
            counts["valid_macro_E"] += 1
            counts["E_crossing_frame"] += int(source.geometry[1] != child.geometry[1])
    return {"status": "passed", "counts": dict(counts), "scope": "captured native physical state and R/E contract; not a claim of simulator accuracy"}


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
    if any(x is None for x in (state.geometry, state.mask_before, state.native_active, state.forward_active, state.valid)):
        raise ValueError("Fresh native-rules capture required; old V2 omitted full physical events/geometry")
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
        "geometry": torch.as_tensor(state.geometry[None], dtype=torch.long, device=device),
        "mask_before": torch.as_tensor(state.mask_before[None], dtype=torch.float32, device=device),
        "native_active": torch.as_tensor(state.native_active[None], dtype=torch.float32, device=device),
        "forward_active": torch.as_tensor(state.forward_active[None], dtype=torch.float32, device=device),
        "span": proposal_span(torch.as_tensor(state.geometry[None], device=device)).float(),
        "valid": torch.as_tensor(state.valid[None], dtype=torch.float32, device=device),
        "terminal": torch.as_tensor([float(state.meta.get("native_terminal", False))], dtype=torch.float32, device=device),
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
        self.geometry_in = nn.Linear(5, dim)
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
        self.source_attention = nn.MultiheadAttention(dim, 4, dropout=0., batch_first=True)
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.transition_norm = nn.LayerNorm(dim)
        self.u_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_size))
        self.memory_out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, memory_size),
        )
        self.event_out = nn.Linear(dim, 1)
        self.context_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, CONTEXT_SIZE))
        self.terminal_out = nn.Linear(dim, 1)
        nn.init.constant_(self.terminal_out.bias, -3.)
        # Native-state identity path is not squeezed through a 128-D decoder.
        nn.init.zeros_(self.u_out[-1].weight)
        nn.init.zeros_(self.u_out[-1].bias)
        nn.init.zeros_(self.memory_out[-1].weight)
        nn.init.zeros_(self.memory_out[-1].bias)

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

    def transition(self, z, action, state):
        geometry = next_geometry(state["geometry"], action)
        source_z = z
        z, _ = align_frame(z, state["geometry"], geometry)
        action_embed = self.action(action).unsqueeze(1)
        scale, shift = self.action_affine(action).chunk(2, dim=-1)
        prepared = z * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        coords = geometry.float() / torch.tensor([4096., 4096., 4096., 8., 8.], device=z.device)
        prepared = prepared + self.geometry_in(coords).unsqueeze(1)
        # Alignment is only the identity carry path. A new frame still needs
        # source semantics and distinct target-position queries, otherwise all
        # source rows vanish and Extend collapses to a broadcast constant.
        prepared = prepared + self.position
        if self.feature_set != "native_only":
            prepared = prepared + self.context_in(state["context"]).unsqueeze(1)
        if self.feature_set == "full":
            prepared = prepared + (self.memory_in(state["memory"]) * state["memory_valid"][:, None]).unsqueeze(1)
        source_context, _ = self.source_attention(prepared + action_embed, source_z, source_z, need_weights=False)
        h = self.dynamics(prepared + source_context + action_embed)
        return self.transition_norm(z + self.delta(h))

    def decode(self, z, source, action):
        pooled = z.mean(dim=1)
        geometry = next_geometry(source["geometry"], action)
        extend = action[:, None].bool()
        base_u, frame_overlap = align_frame(source["u"], source["geometry"], geometry)
        # E includes fresh prefill/possible boundary bridges. Its first exposed
        # pre-mask is nonetheless fixed by native geometry, audited on capture.
        before = torch.where(extend, initial_segment_mask(geometry).float(), source["mask"])
        forward_active = scheduler_region(before, geometry)
        # Native right-shifts logits: [row0, row0, row1, ...]. The structural
        # event readout uses the same positional convention, with no vocab head.
        shifted = torch.cat((z[:, :1], z[:, :-1]), dim=1)
        event_scores = self.event_out(shifted).squeeze(-1)
        event, eligible = constrained_commit(event_scores, before, forward_active)
        mask = before * (1.0 - event)
        native_active = scheduler_region(mask, geometry, logical_only=True)
        span = proposal_span(geometry)
        terminal = self.terminal_out(pooled).squeeze(-1).sigmoid()
        native_active = native_active & (terminal[:, None] < .5)
        active = native_active
        learned_context = self.context_out(pooled)
        context = source["context"].clone()
        context[:, 0] = (geometry[:, 0].float() / 4096.).clamp(max=1)
        context[:, 1] = torch.where(action.bool(),
            (geometry[:, 1] // 32 - geometry[:, 0] // 32).float() / 2., source["context"][:, 1])
        # Only E's number of micro-forwards and EOS flags are unknown; do not
        # ask the model to relearn exact R counters/coordinates.
        crosses = geometry[:, 1] > geometry[:, 0]
        bridge_max = (32 - geometry[:, 0].remainder(32)).float()
        first_denoising_count = torch.where(crosses,
            2. + (bridge_max - 1.) * learned_context[:, 2].sigmoid(),
            torch.ones_like(bridge_max))
        context[:, 2] = torch.where(action.bool(), first_denoising_count / 16., source["context"][:, 2] + 1./16.)
        context[:, 3] = torch.where(action.bool(), torch.full_like(action, 1., dtype=torch.float32) / 16., source["context"][:, 3] + 1./16.)
        pos = torch.arange(BLOCK_SIZE, device=z.device)[None]
        first = torch.where(active, pos, BLOCK_SIZE).min(-1).values
        context[:, 4] = torch.where(first < BLOCK_SIZE, first.float() / 31., -1.)
        context[:, 5:] = learned_context[:, 5:].sigmoid()
        memory_delta = self.memory_out(pooled)
        return {
            "u": base_u + self.u_out(z),
            "memory": source["memory"] + memory_delta * extend,
            "mask": mask, "event": event, "active": active.float(),
            "native_active": native_active.float(), "forward_active": forward_active.float(),
            "mask_before": before, "span": span.float(), "geometry": geometry,
            "context": context, "memory_valid": torch.where(action.bool(), (geometry[:, 2] > 0).float(), source["memory_valid"]),
            "event_scores": event_scores,
            "eligible": eligible.float(), "frame_overlap": frame_overlap.float(),
            "action": action,
            # Unknown future EOS location is not teacher-fed into rollout.
            "valid": torch.ones_like(mask),
            "terminal": terminal,
        }

    def step(self, z, action, source):
        z_next = self.transition(z, action, source)
        return z_next, self.decode(z_next, source, action)


def state_target_batch(ids, states, scalers, device):
    return stack_states(ids, states, scalers, device)


def supervised_loss(pred, target, memory_valid):
    latent_error = F.smooth_l1_loss(pred["u"], target["u"], reduction="none")
    valid = target["valid"]
    loss_u = (latent_error * valid[..., None]).sum() / (valid.sum() * pred["u"].shape[-1]).clamp_min(1)
    memory_mask = memory_valid.float().view(-1, 1)
    memory_error = F.smooth_l1_loss(pred["memory"], target["memory"], reduction="none")
    loss_memory = (memory_error * memory_mask).sum() / (memory_mask.sum() * pred["memory"].shape[-1]).clamp_min(1)
    # Score event logits only where the real native forward could commit. This
    # avoids hundreds of trivial negative positions hiding failed unmasking.
    eligible = target["mask_before"] * target["forward_active"] * valid
    event_error = F.binary_cross_entropy_with_logits(pred["event_scores"], target["event"], reduction="none")
    loss_event = (event_error * eligible).sum() / eligible.sum().clamp_min(1)
    mask_error = F.smooth_l1_loss(pred["mask"], target["mask"], reduction="none")
    loss_mask = (mask_error * valid).sum() / valid.sum().clamp_min(1)
    loss_context = F.smooth_l1_loss(pred["context"][:, 2:4], target["context"][:, 2:4])
    loss_eos = F.binary_cross_entropy(pred["context"][:, 5:].clamp(1e-6, 1-1e-6), target["context"][:, 5:])
    loss_terminal = F.binary_cross_entropy(pred["terminal"].clamp(1e-6, 1-1e-6), target["terminal"])
    total = loss_u + .25 * loss_memory + .30 * loss_event + .15 * loss_mask + .05 * loss_context + .05 * loss_eos + .05 * loss_terminal
    return total, {
        "u": float(loss_u.detach()), "memory": float(loss_memory.detach()),
        "mask": float(loss_mask.detach()), "event": float(loss_event.detach()),
        "context": float(loss_context.detach()), "eos": float(loss_eos.detach()), "terminal": float(loss_terminal.detach()),
    }


def fit_model(model, paths, states, scalers, device, updates=300, batch_size=12,
              seed=42, eval_callback=None, terminal_sources=()):
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
    best_score, best_weights, best_update = float("inf"), None, None
    for step in range(1, updates + 1):
        model.train()
        picks = rng.integers(0, len(viable), size=min(batch_size, len(viable)))
        batch_paths = [viable[int(i)] for i in picks]
        root_ids = [root for root, _ in batch_paths]
        state = stack_states(root_ids, states, scalers, device)
        z = model.encode(state)
        total_loss = torch.zeros((), device=device)
        loss_parts = defaultdict(float)
        horizon_limit = 1 if step <= updates // 3 else (2 if step <= 2 * updates // 3 else 3)
        max_h = min(horizon_limit, max(len(path) for _, path in batch_paths))
        total_cases = sum(min(max_h, len(path)) for _, path in batch_paths)
        for depth in range(max_h):
            active = [i for i, (_, path) in enumerate(batch_paths) if len(path) > depth]
            if not active:
                break
            indices = torch.as_tensor(active, dtype=torch.long, device=device)
            edges = [batch_paths[i][1][depth] for i in active]
            actions = torch.as_tensor([edge.action for edge in edges], dtype=torch.long, device=device)
            predicted_source = {key: value.index_select(0, indices) for key, value in state.items()}
            z_child, pred = model.step(z.index_select(0, indices), actions, predicted_source)
            target = state_target_batch([edge.dst for edge in edges], states, scalers, device)
            mem_valid = torch.as_tensor(
                [states[edge.dst].memory_valid for edge in edges], dtype=torch.bool, device=device
            )
            step_loss, parts = supervised_loss(pred, target, mem_valid)
            weight = len(active) / total_cases
            total_loss = total_loss + step_loss * weight
            for key, value in parts.items():
                loss_parts[key] += value * weight
            z = z.index_copy(0, indices, z_child)
            # Feedback contains only our own predicted U/M/G, never targets.
            state = {key: value.index_copy(0, indices, pred[key]) for key, value in state.items()}
        optimizer.zero_grad(set_to_none=True)
        if terminal_sources:
            # EOS before a raw snapshot has no hidden target, but its parent
            # still supplies a valid termination-supervision example for E.
            sid = terminal_sources[int(rng.integers(0, len(terminal_sources)))]
            observed = stack_states([sid], states, scalers, device)
            _, terminal_pred = model.step(model.encode(observed), torch.ones(1, dtype=torch.long, device=device), observed)
            total_loss = total_loss + .05 * F.binary_cross_entropy(terminal_pred["terminal"], torch.ones_like(terminal_pred["terminal"]))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in checkpoints:
            row = {"update": step, "trained_horizon": max_h, "train_loss": float(total_loss.detach()),
                   "loss_components": dict(loss_parts)}
            if eval_callback:
                row["validation"] = eval_callback(model)
                # Select using validation only, H1 prioritized. Test is unseen.
                h1 = [row["validation"][f"action_{a}_h1"] for a in ("R", "E")]
                score = np.mean([r["latent_nrmse"] + .25 * r["proposal_wrong_positions"] for r in h1 if r["n_predictions"]])
                if score < best_score:
                    best_score, best_update = float(score), step
                    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            curve.append(row)
            print(f"[train:{model.feature_set}] update={step}/{updates} "
                  f"loss={row['train_loss']:.5f}", flush=True)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.selected_update = best_update or updates
    return curve


def _f1(target, pred):
    target = np.asarray(target, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(target & pred)); fp = int(np.sum(~target & pred)); fn = int(np.sum(target & ~pred))
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None


def _state_metrics(pred, target, source, scalers, memory_valid, predicted_source=None):
    pu = pred["u"]
    tu = target["u"]
    su = source["u"]
    diff = pu - tu
    valid = target["valid"] >= .5
    if not valid.any():
        raise ValueError("No valid native residual positions")
    out = {
        "latent_nrmse": float(np.sqrt(np.mean(np.square(diff[valid])))),
        "latent_nmae": float(np.mean(np.abs(diff[valid]))),
        "latent_native_rmse": float(np.sqrt(np.mean(np.square(diff[valid] * scalers["u_std"])))),
    }
    pflat, tflat = pu[valid].reshape(-1), tu[valid].reshape(-1)
    denom = np.linalg.norm(pflat) * np.linalg.norm(tflat)
    out["latent_cosine"] = float(np.dot(pflat, tflat) / denom) if denom > 1e-12 else None
    def aligned_parent(row):
        if "geometry" not in row or "geometry" not in target:
            return row["u"]
        offset = int(target["geometry"][1] - row["geometry"][1])
        indices = np.arange(BLOCK_SIZE) + offset
        base = np.zeros_like(row["u"])
        valid = (indices >= 0) & (indices < BLOCK_SIZE)
        base[valid] = row["u"][indices[valid]]
        return base
    offset = int(target["geometry"][1] - source["geometry"][1])
    mapped = ((np.arange(BLOCK_SIZE) + offset >= 0) & (np.arange(BLOCK_SIZE) + offset < BLOCK_SIZE)) & valid
    out["frame_overlap_positions"] = int(mapped.sum())
    out["new_frame_latent_nrmse"] = float(np.sqrt(np.mean(np.square(diff[valid & ~mapped])))) if (valid & ~mapped).any() else None
    true_delta = tu - aligned_parent(source)
    predicted_source = source if predicted_source is None else predicted_source
    pred_delta = pu - aligned_parent(predicted_source)
    delta_norm = float(np.linalg.norm(true_delta[mapped]))
    out["delta_relative_error"] = (
        float(np.linalg.norm((pred_delta - true_delta)[mapped]) / delta_norm) if delta_norm > 1e-6 else None
    )
    out["mask_accuracy"] = float(np.mean((pred["mask"][valid] >= 0.5) == (target["mask"][valid] >= 0.5)))
    out["mask_f1"] = _f1(target["mask"][valid] >= 0.5, pred["mask"][valid] >= 0.5)
    out["commit_event_f1"] = _f1(target["event"][valid] >= 0.5, pred["event"][valid] >= 0.5)
    out["active_region_f1"] = _f1(target["active"][valid] >= 0.5, pred["active"][valid] >= 0.5)
    span = (target["span"] >= .5) & valid
    wrong = (pred["mask"] >= .5) != (target["mask"] >= .5)
    out["proposal_visible_positions"] = int(span.sum())
    out["proposal_wrong_positions"] = int(wrong[span].sum())
    out["proposal_mask_accuracy"] = float((~wrong[span]).mean()) if span.any() else None
    out["proposal_exact_mask"] = float(not wrong[span].any())
    out["proposal_latent_nrmse"] = float(np.sqrt(np.mean(np.square(diff[span])))) if span.any() else None
    eligible = (target["mask_before"] >= .5) & (target["forward_active"] >= .5) & valid
    truth_event = target["event"] >= .5
    predicted_event = pred["event"] >= .5
    out["event_eligible_positions"] = int(eligible.sum())
    out["event_tp"] = int((truth_event & predicted_event & eligible).sum())
    out["event_fp"] = int((~truth_event & predicted_event & valid).sum())
    out["event_fp_on_real_eligible"] = int((~truth_event & predicted_event & eligible).sum())
    out["event_fn"] = int((truth_event & ~predicted_event & eligible).sum())
    out["event_outside_real_eligible"] = int((predicted_event & ~eligible & valid).sum())
    out["true_commit_count"] = int((truth_event & valid).sum())
    out["predicted_commit_count"] = int((predicted_event & valid).sum())
    out["commit_count_absolute_error"] = abs(out["predicted_commit_count"] - out["true_commit_count"])
    out["scheduler_context_mae"] = float(np.mean(np.abs(pred["context"] - target["context"])))
    out["denoising_forward_count_absolute_error"] = float(abs(pred["context"][2] - target["context"][2]) * 16.)
    out["native_eos_accuracy"] = float(
        (pred["context"][5] >= 0.5) == (target["context"][5] >= 0.5)
    )
    out["stop_fill_eos_accuracy"] = float(
        (pred["context"][6] >= 0.5) == (target["context"][6] >= 0.5)
    )
    out["native_eos_positive"] = int(target["context"][5] >= .5)
    out["stop_fill_eos_positive"] = int(target["context"][6] >= .5)
    out["terminal_accuracy"] = float((pred["terminal"] >= .5) == (target["terminal"] >= .5))
    out["terminal_positive"] = int(target["terminal"] >= .5)
    out["remasked_positions"] = int(((predicted_source["mask"] < .5) & (pred["mask"] >= .5)).sum()) if np.array_equal(target["geometry"], predicted_source["geometry"]) else None
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
    for flag in ("native_eos", "stop_fill_eos", "terminal"):
        positives = sum(r["metrics"][flag + "_positive"] for r in records)
        result[flag + "_positive_count"] = positives
        result[flag + "_positive_coverage"] = "present" if positives else "not_tested_no_positive"
    tp, fp, fn = (sum(r["metrics"][key] for r in records) for key in ("event_tp", "event_fp", "event_fn"))
    result["commit_event_micro_precision"] = tp / (tp + fp) if tp + fp else None
    result["commit_event_micro_recall"] = tp / (tp + fn) if tp + fn else None
    result["commit_event_micro_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    result["commit_event_total_tp_fp_fn"] = [tp, fp, fn]
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
            z, output = model.step(z, action, current)
            target = pack_state(states[edge.dst], scalers, device)
            source = pack_state(states[edge.src], scalers, device)
            pred_u = output["u"][0].cpu().numpy()
            pred_memory = output["memory"][0].cpu().numpy()
            pred_mask = output["mask"][0].cpu().numpy()
            pred_event = output["event"][0].cpu().numpy()
            pred_active = output["active"][0].cpu().numpy()
            pred_context = output["context"][0].cpu().numpy()
            target_np = {key: value[0].cpu().numpy() for key, value in target.items()}
            source_np = {key: value[0].cpu().numpy() for key, value in source.items()}
            pred_np = {
                "u": pred_u, "memory": pred_memory,
                "mask": pred_mask, "event": pred_event,
                "active": pred_active, "context": pred_context,
                "geometry": output["geometry"][0].cpu().numpy(),
                "terminal": float(output["terminal"][0].cpu()),
            }
            predicted_source = source_np if previous_prediction is None else previous_prediction
            metrics = _state_metrics(
                pred_np, target_np, source_np, scalers, states[edge.dst].memory_valid,
                predicted_source=predicted_source,
            )
            metrics["action_inconsistent_with_predicted_state"] = float(
                (predicted_source["terminal"] >= .5) or
                (edge.action == 1 and (predicted_source["context"][6] >= .5 or predicted_source["geometry"][4] >= 7)) or
                (edge.action == 0 and not np.any((predicted_source["mask"] >= .5) & (source_np["span"] >= .5)))
            )
            records.append({"qid": states[edge.dst].qid, "src": edge.src, "dst": edge.dst,
                            "action": "E" if edge.action else "R", "horizon": depth,
                            "sequence": "".join("E" if e.action else "R" for e in chain[:depth]),
                            "crossed_physical_frame": bool(states[edge.src].geometry[1] != states[edge.dst].geometry[1]),
                            "metrics": metrics})
            previous_prediction = pred_np
            current = {key: output[key] for key in current}
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
                "geometry": source_np["geometry"],
                "terminal": source_np["terminal"],
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
    groups["sequences"] = {
        seq: summarize([r for r in records if r.get("sequence") == seq])
        for seq in sorted({r["sequence"] for r in records if "sequence" in r})
    }
    groups["E_frame_crossing"] = {
        name: summarize([r for r in records if r["action"] == "E" and r.get("crossed_physical_frame") == crossing])
        for name, crossing in (("same_frame", False), ("new_frame", True))
    }
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
            geometry=np.stack([s.geometry for s in rows]).astype(np.int64),
            mask_before_forward=np.stack([s.mask_before for s in rows]).astype(np.bool_),
            proposal_span=np.stack([proposal_span(torch.as_tensor(s.geometry[None]))[0].numpy() for s in rows]),
            native_active_region=np.stack([s.native_active for s in rows]).astype(np.bool_),
            forward_active_region=np.stack([s.forward_active for s in rows]).astype(np.bool_),
            valid_positions=np.stack([s.valid for s in rows]).astype(np.bool_),
        )
        for offset, state in enumerate(rows):
            m = state.meta
            state_meta.append({
                "state_id": state.sid, "question_id": state.qid,
                "parent_state_id": m.get("parent_state_id"),
                "action_from_parent": m.get("action_from_parent"),
                "block_index": int(m.get("block_index", 0)),
                "refine_index": int(m.get("refine_index", 0)),
                "native_forward_index": int(m.get("native_unmask_forward_index", m.get("native_forward_index", 0))),
                "physical_block_start": m.get("physical_block_start"),
                "active_region_offset": m.get("active_region_offset"),
                "prefix_kv_length": m.get("prefix_kv_length"),
                "prefix_kv_valid": state.memory_valid,
                "proposal_start_absolute": int(state.geometry[0]),
                "proposal_length": int(state.geometry[3]),
                "visible_proposal_positions": int(proposal_span(torch.as_tensor(state.geometry[None])).sum()),
                "native_hidden_stage": "native_denoising_pre_commit_pre_layer24",
                "mask_stage": "same_forward_post_commit",
                "event_scope": "full_physical_block",
                "segment_micro_forward_counts": m.get("segment_micro_forward_counts"),
                "eos_present": m.get("eos_present"),
                "native_terminal": m.get("native_terminal"),
                "masks_before_eos": m.get("masks_before_eos"),
                "prefix_kv_layer_indices": [0, 7, 15, 23],
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
    # Terminal edges with no exposed hidden state must not be silently censored,
    # or turned into fake zero-valued hidden targets.
    terminal_file = capture / "terminal_transitions.jsonl"
    if not terminal_file.exists():
        terminal_file = capture / "teacher_features" / "terminal_transitions.jsonl"
    if terminal_file.exists():
        terminal_rows = read_jsonl(terminal_file)
        allowed = {"question_id", "src_state_id", "action", "reason", "hidden_available",
                   "segment_index", "native_terminal", "segment_micro_forward_counts"}
        cleaned = [{**r, "reason": r.get("reason", r.get("terminal_reason")),
                    "native_terminal": bool(r.get("native_terminal", r.get("candidate_contains_eos", False))),
                    "segment_micro_forward_counts": r.get("segment_micro_forward_counts", r.get("forward_pass_breakdown"))}
                   for r in terminal_rows]
        (dest / "terminal_transitions.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in r.items() if k in allowed}) + "\n" for r in cleaned), encoding="utf-8"
        )


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
                    "prefix_kv": "upstream layers 0/7/15/23: mean K/V over last 128 and last 8 prefix positions; not full KV cache",
                    "E_semantics": "top1 fill selected parent, fresh prefill, bridge if needed, first exposed native unmask; macro action, not native in-place extension"},
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
        "native_contract_audit": audit_native_contract(states, edges),
    }
    all_groups = {}
    terminal_file = out / "teacher_features" / "terminal_transitions.jsonl"
    terminal_rows = read_jsonl(terminal_file) if terminal_file.exists() else []
    terminal_sources = [r["src_state_id"] for r in terminal_rows if r.get("native_terminal") and r.get("src_state_id") in states and split[states[r["src_state_id"]].qid] == "train"]
    report["terminal_without_hidden"] = {"captured": len(terminal_rows), "train_with_observed_parent": len(terminal_sources),
        "handling": "termination-only supervision; no fabricated hidden/mask target"}
    test_metric_rows = []
    for variant_index, variant in enumerate(variants):
        torch.manual_seed(args.seed)
        model = NativeLatentDynamics(hidden_size, memory_size, args.latent_dim,
                                     args.model_layers, variant)
        def validation(model_to_eval):
            rows = evaluate(model_to_eval, val_paths, states, scalers, device)
            return grouped_report(rows)
        started = time.time()
        curve = fit_model(
            model, train_paths, states, scalers, device,
            updates=args.updates, batch_size=args.batch_size,
            seed=args.seed, eval_callback=validation, terminal_sources=terminal_sources,
        )
        test_rows = evaluate(model, test_paths, states, scalers, device)
        test_report = grouped_report(test_rows)
        test_metric_rows.extend({"model": variant, **row} for row in test_rows)
        report["variants"][variant] = {
            "training_curve": curve,
            "test": test_report,
            "elapsed_seconds": time.time() - started,
            "selected_update": model.selected_update,
        }
        all_groups[variant] = test_rows
        torch.save({"schema": SCHEMA, "feature_set": variant,
                    "model_config": {"hidden_size": hidden_size, "memory_size": memory_size,
                                     "dim": args.latent_dim, "layers": args.model_layers},
                    "normalization": {k: torch.from_numpy(v) for k, v in scalers.items()},
                    "selected_update": model.selected_update,
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
        "hard_rules": "R no remask; constant R prefix KV; event restricted to current native 8-subblock; at least one eligible commit; deterministic absolute geometry, counters, next active region",
        "event_scores": "predicted structural commit-event likelihood, not native token confidence; forcing progress does not make learned event ordering exact",
        "coordinate_delta": "mapped by absolute coordinates; delta scored only on overlap, new-frame error reported separately; unaligned source latent retained by cross-attention",
        "partial_observation": "U is pre-commit and G is post-commit; replacement token embeddings/full KV are absent, so this is a compressed partially observed simulator, not exact native generator execution",
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
    parser.add_argument("--dllm-dir", type=Path)
    parser.add_argument("--capture-input", type=Path, help="Reuse a corrected native-rules folder (raw or teacher_features); no drafter rerun")
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
    if args.extend_size != 8 or args.max_proposal_tokens != 64 or not 0 <= args.max_refinement_steps <= 3:
        raise ValueError("V2 is pinned to E=8, max proposal=64, and at most three R steps per segment")
    if args.threshold != .5 or args.cut_layer != 24 or args.suffix_layers != 4:
        raise ValueError("This audited native-rules schema pins threshold=.5, cut-layer=24, memory groups=4")
    if args.horizon not in (1, 2, 3):
        raise ValueError("Supported rollout horizon is 1, 2, or 3")
    if args.capture_input is None and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        raise RuntimeError("Select Kaggle GPU T4 x2: native capture on GPU0 and latent training on GPU1")
    if args.capture_input is None and args.dllm_dir is None:
        raise ValueError("Provide --dllm-dir for fresh capture or --capture-input for feature-only replay")
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
    features_persisted = False
    stage = "capture"
    try:
        print("[capture] Fresh greedy native Fast-dLLM R/E trajectories; no verifier is loaded.", flush=True)
        capture_root = args.capture_input or capture_dir
        if args.capture_input is not None:
            print(f"[replay] Reuse corrected native features: {capture_root}", flush=True)
        capture_manifest = collect_native(capture_args) if args.capture_input is None else {"source": str(capture_root)}
        states, edges = load_teacher(capture_root)
        contract = audit_native_contract(states, edges)
        split = split_questions({s.qid for s in states.values()}, args.split_seed)
        scalers = fit_scalers(states, split)
        # Preserve reusable features before training/reporting can fail.
        sanitize_capture(capture_root, args.out_dir, states, edges, split, scalers)
        features_persisted = True
        (args.out_dir / "native_contract_audit.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        stage = "training"
        (args.out_dir / "run_status.json").write_text(json.dumps({"status": "partial", "stage": stage, "schema": SCHEMA}), encoding="utf-8")
        print(f"[capture-done] states={len(states)} edges={len(edges)} "
              f"R={sum(e.action == 0 for e in edges)} E={sum(e.action == 1 for e in edges)}", flush=True)
        report, scalers = train_and_audit(args, states, edges, split, args.out_dir)
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
        (args.out_dir / "run_status.json").write_text(json.dumps({"status": "complete", "schema": SCHEMA}), encoding="utf-8")
        archive = package(args.out_dir)
        print(f"[result-zip] {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    except Exception as exc:
        (args.out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if not features_persisted:
            try:
                capture_root = args.capture_input or capture_dir
                states, edges = load_teacher(capture_root, require_edges=False)
                qids = {s.qid for s in states.values()}
                split = split_questions(qids, args.split_seed) if len(qids) >= 5 else {q: "train" for q in qids}
                sanitize_capture(capture_root, args.out_dir, states, edges, split, fit_scalers(states, split))
                features_persisted = True
            except Exception as preserve_exc:
                (args.out_dir / "capture_preservation_error.txt").write_text(repr(preserve_exc), encoding="utf-8")
        (args.out_dir / "run_status.json").write_text(json.dumps({"status": "partial", "failed_stage": stage, "schema": SCHEMA,
            "sanitized_features_preserved": features_persisted, "raw_capture_retained": None if features_persisted else str(capture_dir)}), encoding="utf-8")
        if any(args.out_dir.iterdir()):
            try:
                print(f"[partial-zip] {package(args.out_dir)}", flush=True)
            except Exception as package_exc:
                print(f"[partial-zip-skipped] {package_exc}", flush=True)
        raise
    finally:
        if features_persisted and capture_dir.exists():
            shutil.rmtree(capture_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
