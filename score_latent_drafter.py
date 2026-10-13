#!/usr/bin/env python3
"""Learn and evaluate score-valued latent dynamics for native Fast-dLLM.

The student receives one real native observation at a rollout root, encodes it
to per-position latent scores and a small memory code, then predicts latent
states and scheduler events after supplied R/E actions. It never reconstructs
the 1536-wide residual, token IDs, vocabulary logits, token probabilities,
verifier labels, accepted lengths, or latency. Future teacher observations are
used only to make stopped-gradient representation/event targets and metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
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
from native_latent_dynamics_v2 import (
    ACTION_TO_ID, KV_SHAPE, audit_native_contract, fit_scalers, load_teacher,
    package as package_teacher, read_jsonl, sanitize_capture, split_questions,
)
from native_latent_rules_v2 import (
    align_frame, constrained_commit, initial_segment_mask, next_geometry,
    proposal_span, scheduler_region,
)


SCHEMA = "score_latent_drafter_v1"
BLOCK_SIZE = 32
LATENT_DIM_DEFAULT = 64
MEMORY_DIM = 64


def geometry_features(geometry: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor([4096.0, 4096.0, 4096.0, 64.0, 16.0],
                         dtype=torch.float32, device=geometry.device)
    return geometry.float() / scale


def structural_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack((batch["mask"], batch["event"], batch["active"],
                        batch["mask_before"], batch["native_active"],
                        batch["forward_active"], batch["span"], batch["valid"]), dim=-1)


def pack_states(ids, states, scalers, device) -> dict[str, torch.Tensor]:
    rows = [states[sid] for sid in ids]
    u_mean = torch.as_tensor(scalers["u_mean"], dtype=torch.float32)
    u_std = torch.as_tensor(scalers["u_std"], dtype=torch.float32).clamp_min(1e-4)
    m_mean = torch.as_tensor(scalers["memory_mean"], dtype=torch.float32)
    m_std = torch.as_tensor(scalers["memory_std"], dtype=torch.float32).clamp_min(1e-4)
    geometries = torch.as_tensor(np.stack([s.geometry for s in rows]), dtype=torch.long)
    span = proposal_span(geometries).float()
    return {
        "u": ((torch.as_tensor(np.stack([s.u for s in rows]), dtype=torch.float32)
               - u_mean[None, None]) / u_std[None, None]).to(device),
        "memory": ((torch.as_tensor(np.stack([s.memory for s in rows]), dtype=torch.float32)
                    - m_mean[None]) / m_std[None]).to(device),
        "memory_valid": torch.tensor([s.memory_valid for s in rows], dtype=torch.bool, device=device),
        "mask": torch.as_tensor(np.stack([s.mask for s in rows]), dtype=torch.float32, device=device),
        "event": torch.as_tensor(np.stack([s.event for s in rows]), dtype=torch.float32, device=device),
        "active": torch.as_tensor(np.stack([s.active for s in rows]), dtype=torch.float32, device=device),
        "mask_before": torch.as_tensor(np.stack([s.mask_before for s in rows]), dtype=torch.float32, device=device),
        "native_active": torch.as_tensor(np.stack([s.native_active for s in rows]), dtype=torch.float32, device=device),
        "forward_active": torch.as_tensor(np.stack([s.forward_active for s in rows]), dtype=torch.float32, device=device),
        "valid": torch.as_tensor(np.stack([s.valid for s in rows]), dtype=torch.float32, device=device),
        "context": torch.as_tensor(np.stack([s.context for s in rows]), dtype=torch.float32, device=device),
        "geometry": geometries.to(device),
        "span": span.to(device),
        "terminal": torch.tensor([bool(s.meta.get("native_terminal", False)) for s in rows],
                                  dtype=torch.float32, device=device),
    }


def split_edges(states, edges, split):
    return {
        part: [e for e in edges if split[states[e.src].qid] == part
               and split[states[e.dst].qid] == part]
        for part in ("train", "validation", "test")
    }


def enumerate_paths(states, edges, split, part, horizon=3, paths_per_root=12):
    """Enumerate bounded observed action sequences; never invent unobserved edges."""
    outgoing = defaultdict(list)
    for edge in edges:
        if split[states[edge.src].qid] == part and split[states[edge.dst].qid] == part:
            outgoing[edge.src].append(edge)
    paths, seen = [], set()
    roots = sorted(sid for sid, state in states.items() if split[state.qid] == part)
    for root in roots:
        local_count = 0
        stack = [(root, tuple())]
        while stack and local_count < paths_per_root:
            current, chain = stack.pop()
            if chain:
                signature = tuple((e.src, e.dst, e.action) for e in chain)
                if signature not in seen:
                    paths.append((root, chain))
                    seen.add(signature)
                    local_count += 1
            if len(chain) >= horizon:
                continue
            children = outgoing.get(current, [])
            # Reverse so DFS visits R before E in deterministic order.
            for edge in reversed(sorted(children, key=lambda e: e.action)):
                stack.append((edge.dst, chain + (edge,)))
    return paths


def path_pool(paths, horizon):
    return [(root, chain[:horizon]) for root, chain in paths if len(chain) >= horizon]


class ScoreEncoder(nn.Module):
    """Encode native residuals into a score vector at each physical position."""
    def __init__(self, raw_dim: int, dim: int, layers: int = 2,
                 feature_set: str = "full_context"):
        super().__init__()
        if feature_set not in {"native_only", "native_structure", "full_context"}:
            raise ValueError(f"Unknown feature set {feature_set!r}")
        self.feature_set = feature_set
        self.raw_dim = raw_dim
        self.dim = dim
        self.u_norm = nn.LayerNorm(raw_dim)
        self.u_in = nn.Sequential(nn.Linear(raw_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.position = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.02)
        self.structure_in = nn.Linear(8, dim, bias=False)
        self.global_in = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory_in = nn.Sequential(nn.LayerNorm(int(np.prod(KV_SHAPE))),
                                       nn.Linear(int(np.prod(KV_SHAPE)), dim), nn.GELU())
        block = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.context = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.z_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.memory_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, batch, ablate_features=()):
        ablate_features = set(ablate_features)
        residual = self.u_in(self.u_norm(batch["u"]))
        if "native_residual" in ablate_features:
            residual = torch.zeros_like(residual)
        h = residual + (0.0 if "position_embedding" in ablate_features else self.position)
        if self.feature_set != "native_only":
            if "scheduler_structure" not in ablate_features:
                h = h + self.structure_in(structural_features(batch))
            context = batch["context"]
            geometry = geometry_features(batch["geometry"])
            if "scheduler_context" in ablate_features:
                context = torch.zeros_like(context)
            if "geometry_embedding" in ablate_features:
                geometry = torch.zeros_like(geometry)
            global_input = torch.cat((context, geometry), dim=-1)
            h = h + self.global_in(global_input)[:, None]
        if self.feature_set == "full_context":
            if "prefix_kv" in ablate_features:
                m = torch.zeros((len(h), self.dim), dtype=h.dtype, device=h.device)
            else:
                m = self.memory_in(batch["memory"])
                m = m * batch["memory_valid"][:, None]
            h = h + m[:, None]
        valid = batch["valid"].bool().clone()
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            valid[all_invalid, 0] = True
        h = self.context(h, src_key_padding_mask=~valid)
        z = self.z_out(h) * batch["valid"][..., None]
        if self.feature_set == "full_context":
            if "prefix_kv" in ablate_features:
                memory_z = torch.zeros((len(z), self.dim), dtype=z.dtype, device=z.device)
            else:
                memory_z = self.memory_out(self.memory_in(batch["memory"]))
                memory_z = memory_z * batch["memory_valid"][:, None]
        else:
            memory_z = torch.zeros((len(z), self.dim), dtype=z.dtype, device=z.device)
        return z, memory_z


class ScoreTransition(nn.Module):
    """Predict the next latent score vector and native structural event."""
    def __init__(self, dim: int, layers: int = 2, feature_set: str = "full_context"):
        super().__init__()
        self.dim = dim
        self.feature_set = feature_set
        self.position = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.02)
        self.action = nn.Embedding(2, dim)
        self.action_film = nn.Embedding(2, dim * 2)
        self.geometry_in = nn.Linear(5, dim)
        self.context_in = nn.Linear(7, dim)
        self.structure_in = nn.Linear(8, dim, bias=False)
        self.memory_in = nn.Linear(dim, dim, bias=False)
        self.source_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        block = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.dynamics = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                   nn.Linear(dim * 2, dim))
        self.z_norm = nn.LayerNorm(dim)
        self.event_out = nn.Linear(dim, 1)
        self.change_out = nn.Linear(dim, 3)
        self.valid_out = nn.Linear(dim, 1)
        self.memory_delta = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim * 2),
                                          nn.GELU(), nn.Linear(dim * 2, dim))
        self.context_out = nn.Sequential(nn.LayerNorm(dim + 8),
                                         nn.Linear(dim + 8, dim), nn.GELU(),
                                         nn.Linear(dim, 7), nn.Tanh())
        self.terminal_out = nn.Linear(dim, 1)
        nn.init.constant_(self.terminal_out.bias, -3.0)

    def forward(self, state, action, ablate_features=()):
        ablate_features = set(ablate_features)
        geometry = next_geometry(state["geometry"], action)
        aligned, overlap = align_frame(state["z"], state["geometry"], geometry)
        action_embed = self.action(action)
        scale, shift = self.action_film(action).chunk(2, dim=-1)
        if "action_conditioning" in ablate_features:
            action_embed = torch.zeros_like(action_embed)
            scale = torch.zeros_like(scale)
            shift = torch.zeros_like(shift)
        geometry_embedding = geometry_features(geometry)
        context_input = state["context"]
        if "geometry_embedding" in ablate_features:
            geometry_embedding = torch.zeros_like(geometry_embedding)
        if "scheduler_context" in ablate_features:
            context_input = torch.zeros_like(context_input)
        global_features = (self.geometry_in(geometry_embedding)
                           + self.context_in(context_input) + action_embed)
        query = aligned * (1.0 + 0.1 * torch.tanh(scale[:, None])) + shift[:, None]
        if "position_embedding" not in ablate_features:
            query = query + self.position
        query = query + global_features[:, None]
        if self.feature_set != "native_only" and "scheduler_structure" not in ablate_features:
            source_struct = torch.stack((state["mask"], state["event"], state["active"],
                                         state["mask_before"], state["native_active"],
                                         state["forward_active"], state["span"],
                                         state["valid"]), dim=-1)
            query = query + self.structure_in(source_struct)
        if self.feature_set == "full_context" and "prefix_kv" not in ablate_features:
            query = query + self.memory_in(state["memory_z"])[:, None]
        source_valid = state["valid"].bool().clone()
        all_invalid = ~source_valid.any(dim=1)
        if all_invalid.any():
            source_valid[all_invalid, 0] = True
        source, _ = self.source_attention(query, state["z"], state["z"],
                                          key_padding_mask=~source_valid, need_weights=False)
        h = self.dynamics(query + source)
        valid_logits = self.valid_out(h).squeeze(-1)
        valid_prob = valid_logits.sigmoid()
        action_extend = action.bool()
        before = torch.where(action_extend[:, None], initial_segment_mask(geometry).float(),
                             state["mask"])
        base_region = scheduler_region(before, geometry)
        proposal_positions = proposal_span(geometry)
        candidate_valid = (before >= 0.5) & base_region & proposal_positions
        has_candidate = candidate_valid.any(dim=1)
        has_valid_candidate = (candidate_valid & (valid_prob >= 0.5)).any(dim=1)
        first_candidate = candidate_valid.float().argmax(dim=1)
        force_valid = has_candidate & ~has_valid_candidate
        forced_valid = torch.zeros_like(valid_prob, dtype=torch.bool).scatter(
            1, first_candidate[:, None], force_valid[:, None])
        valid_hard = ((valid_prob >= 0.5) | forced_valid).to(valid_prob.dtype)
        valid = valid_hard + valid_prob - valid_prob.detach()
        z_next = self.z_norm(aligned + self.delta(h)) * valid[..., None]

        region = base_region & valid_hard.bool()
        shifted = torch.cat((z_next[:, :1], z_next[:, :-1]), dim=1)
        event_logits = self.event_out(shifted).squeeze(-1)
        event, eligible = constrained_commit(event_logits, before, region)
        mask = before * (1.0 - event)
        active = scheduler_region(mask, geometry, logical_only=True).float() * valid
        source_valid = state["valid"].to(h.dtype)
        pooled_h = (h * source_valid[..., None]).sum(dim=1) / source_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        terminal_logit = self.terminal_out(pooled_h).squeeze(-1)
        terminal_prob = terminal_logit.sigmoid()
        active = active * (terminal_prob < 0.5)[:, None]
        if "prefix_kv" in ablate_features:
            memory_z = torch.zeros_like(state["memory_z"])
        else:
            memory_candidate = state["memory_z"] + self.memory_delta(
                torch.cat((pooled_h, state["memory_z"]), dim=-1))
            memory_z = torch.where(action_extend[:, None], memory_candidate, state["memory_z"])
        context_state = state["context"]
        action_context = action.float()[:, None]
        if "scheduler_context" in ablate_features:
            context_state = torch.zeros_like(context_state)
        if "action_conditioning" in ablate_features:
            action_context = torch.zeros_like(action_context)
        context_input = torch.cat((pooled_h, context_state, action_context), dim=-1)
        next_context = self.context_out(context_input)
        next_state = {
            "z": z_next, "memory_z": memory_z, "mask": mask,
            "event": event, "active": active, "context": next_context,
            "geometry": geometry, "memory_valid": torch.where(
                action_extend, (geometry[:, 2] > 0), state["memory_valid"]),
            "mask_before": before, "native_active": active,
            "forward_active": region.float(), "valid": valid,
            "span": proposal_span(geometry).float(), "terminal": terminal_prob,
        }
        aux = {"event_logits": event_logits, "eligible": eligible.float(),
               "change_logits": self.change_out(h), "valid_logits": valid_logits,
               "terminal_logit": terminal_logit,
               "overlap": overlap.float()}
        return next_state, aux


def variance_covariance_loss(z, valid, variance_target=0.5):
    rows = z[valid.bool()]
    if rows.shape[0] < 4:
        return z.sum() * 0.0
    centered = rows - rows.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
    var_loss = F.relu(variance_target - std).square().mean()
    cov = centered.T @ centered / max(1, rows.shape[0] - 1)
    offdiag = cov - torch.diag(torch.diagonal(cov))
    return var_loss + 0.01 * offdiag.square().sum() / z.shape[-1]


def fit_change_thresholds(states, train_edges, scalers):
    u_std = scalers["u_std"]
    magnitudes = []
    for edge in train_edges:
        src, dst = states[edge.src], states[edge.dst]
        src_abs = int(src.geometry[1]) + np.arange(BLOCK_SIZE)
        dst_abs = int(dst.geometry[1]) + np.arange(BLOCK_SIZE)
        source_row = dst_abs - int(src.geometry[1])
        overlap = (source_row >= 0) & (source_row < BLOCK_SIZE)
        aligned_source = np.zeros_like(dst.u)
        aligned_source[overlap] = src.u[source_row[overlap]]
        source_valid = np.zeros(BLOCK_SIZE, dtype=bool)
        source_valid[overlap] = src.valid[source_row[overlap]] > 0
        valid = overlap & source_valid & (dst.valid > 0)
        if not valid.any():
            continue
        delta = (dst.u - aligned_source) / u_std[None]
        mag = np.sqrt(np.mean(np.square(delta), axis=-1))
        magnitudes.extend(mag[valid].tolist())
    if len(magnitudes) < 8:
        raise RuntimeError("Not enough same-frame R positions to define coarse change-score bins")
    q1, q2 = np.quantile(np.asarray(magnitudes), [1/3, 2/3]).tolist()
    return [float(q1), float(q2)]


def target_change_labels(src, dst, scalers, thresholds, device):
    src_abs = int(src.geometry[1]) + np.arange(BLOCK_SIZE)
    dst_abs = int(dst.geometry[1]) + np.arange(BLOCK_SIZE)
    source_row = dst_abs - int(src.geometry[1])
    overlap = (source_row >= 0) & (source_row < BLOCK_SIZE)
    aligned_source = np.zeros_like(dst.u)
    aligned_source[overlap] = src.u[source_row[overlap]]
    source_valid = np.zeros(BLOCK_SIZE, dtype=bool)
    source_valid[overlap] = src.valid[source_row[overlap]] > 0
    valid = overlap & source_valid & (dst.valid > 0)
    delta = (dst.u - aligned_source) / scalers["u_std"][None]
    magnitude = np.sqrt(np.mean(np.square(delta), axis=-1))
    labels = np.digitize(magnitude, thresholds).astype(np.int64)
    return (torch.as_tensor(labels, dtype=torch.long, device=device)[None],
            torch.as_tensor(valid, dtype=torch.bool, device=device)[None])


def _target_latents(target_encoder, batch):
    with torch.no_grad():
        return target_encoder(batch)


def _state_from_batch(batch, encoded):
    z, memory_z = encoded
    return {"z": z, "memory_z": memory_z, **{
        key: batch[key] for key in (
            "mask", "event", "active", "context", "geometry", "memory_valid",
            "mask_before", "native_active", "forward_active", "valid", "span", "terminal")
    }}


def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def _balanced_event_weight(train_edges, states):
    positive = negative = 0
    for edge in train_edges:
        dst = states[edge.dst]
        eligible = (dst.mask_before >= 0.5) & (dst.forward_active >= 0.5) & (dst.valid >= 0.5)
        positive += int(((dst.event >= 0.5) & eligible).sum())
        negative += int((eligible & (dst.event < 0.5)).sum())
    return float(np.clip(negative / max(positive, 1), 1.0, 6.0))


def loss_for_step(pred, aux, target_batch, target_z, source_state,
                  predicted_source_z, source_target_z, scalers, thresholds,
                  event_pos_weight):
    valid = target_batch["valid"]
    z_err = F.smooth_l1_loss(pred["z"], target_z, reduction="none")
    z_loss = _masked_mean(z_err, valid)

    source_geometry = torch.as_tensor(source_state.geometry[None], dtype=torch.long,
                                      device=target_z.device)
    true_parent_aligned, overlap = align_frame(source_target_z, source_geometry,
                                               target_batch["geometry"])
    pred_parent_aligned, _ = align_frame(predicted_source_z, source_geometry,
                                         target_batch["geometry"])
    # Compare the transition's change against the native teacher change. For
    # H2/H3, predicted_source_z is the model's own previous rollout, not a
    # teacher-forced child/source representation.
    pred_delta = pred["z"] - pred_parent_aligned
    true_delta = target_z - true_parent_aligned
    delta_valid = (overlap > 0) & (valid > 0)
    delta_loss = _masked_mean(F.smooth_l1_loss(pred_delta, true_delta, reduction="none"),
                              delta_valid)

    eligible = target_batch["mask_before"] * target_batch["forward_active"] * valid
    event_bce = F.binary_cross_entropy_with_logits(
        aux["event_logits"], target_batch["event"], reduction="none",
        pos_weight=torch.tensor(event_pos_weight, device=target_z.device))
    event_loss = _masked_mean(event_bce, eligible)

    change_labels, change_valid = target_change_labels(
        source_state, target_batch["_state_obj"], scalers, thresholds, target_z.device)
    change_ce = F.cross_entropy(aux["change_logits"].reshape(-1, 3),
                                change_labels.reshape(-1), reduction="none").reshape_as(change_valid)
    change_loss = _masked_mean(change_ce, change_valid)

    context_loss = F.smooth_l1_loss(pred["context"], target_batch["context"])
    mem_valid = target_batch["memory_valid"].float()[:, None]
    memory_loss = _masked_mean(F.smooth_l1_loss(pred["memory_z"],
                                                target_batch["_memory_z"], reduction="none"), mem_valid)
    terminal_loss = F.binary_cross_entropy_with_logits(aux["terminal_logit"],
                                                        target_batch["terminal"])
    valid_loss = _masked_mean(F.binary_cross_entropy_with_logits(
        aux["valid_logits"], target_batch["valid"], reduction="none"),
        torch.ones_like(valid))
    total = z_loss + 0.5 * delta_loss + 0.5 * event_loss + 0.2 * change_loss \
        + 0.1 * context_loss + 0.1 * memory_loss + 0.05 * terminal_loss + 0.1 * valid_loss
    return total, {"latent": z_loss, "delta": delta_loss, "event": event_loss,
                   "change": change_loss, "context": context_loss,
                   "memory": memory_loss, "terminal": terminal_loss, "valid": valid_loss}


def _augment_target_batch(batch, state_objects, ids, target_encoder):
    z, memory_z = _target_latents(target_encoder, batch)
    out = dict(batch)
    out["_memory_z"] = memory_z
    out["_state_obj"] = state_objects[ids[0]] if len(ids) == 1 else [state_objects[sid] for sid in ids]
    return z, out


def _ema_update(target, online, momentum=0.995):
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)
        for target_buffer, online_buffer in zip(target.buffers(), online.buffers()):
            target_buffer.copy_(online_buffer)


def _sample_paths(rng, paths, questions, horizon, batch_size):
    candidates = path_pool(paths, horizon)
    if not candidates:
        horizon = max((len(chain) for _, chain in paths), default=0)
        candidates = path_pool(paths, horizon)
    if not candidates:
        raise RuntimeError(f"No train paths for requested horizon {horizon}")
    by_question = defaultdict(list)
    for path in candidates:
        by_question[questions[path[0]]].append(path)
    qids = sorted(by_question)
    chosen = []
    order = rng.permutation(qids)
    while len(chosen) < batch_size:
        qid = order[len(chosen) % len(order)]
        pool = by_question[qid]
        chosen.append(pool[int(rng.integers(0, len(pool)))])
    return chosen, horizon


def _merge_batch_state(encoded, batch):
    return _state_from_batch(batch, encoded)


def train_variant(feature_set, train_paths, val_paths, states, scalers,
                  args, change_thresholds, event_pos_weight, device):
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed + {"native_only": 1, "native_structure": 2,
                                                 "full_context": 3}[feature_set])
    raw_dim = next(iter(states.values())).u.shape[-1]
    online = ScoreEncoder(raw_dim, args.latent_dim, args.encoder_layers, feature_set).to(device)
    transition = ScoreTransition(args.latent_dim, args.transition_layers, feature_set).to(device)
    target = copy.deepcopy(online).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(list(online.parameters()) + list(transition.parameters()),
                                  lr=args.learning_rate, weight_decay=1e-4)
    best_score, best_payload, best_update = float("inf"), None, 0
    curve = []
    checkpoints = sorted(set([max(1, args.updates // 4), max(1, args.updates // 2),
                              max(1, 3 * args.updates // 4), args.updates]))
    train_qids = sorted({states[root].qid for root, _ in train_paths})
    for update in range(1, args.updates + 1):
        horizon = 1 if update <= args.updates // 3 else (2 if update <= 2 * args.updates // 3 else 3)
        chosen, horizon = _sample_paths(np_rng, train_paths, {sid: s.qid for sid, s in states.items()},
                                        horizon, args.batch_size)
        root_ids = [root for root, _ in chosen]
        root_batch = pack_states(root_ids, states, scalers, device)
        z0, m0 = online(root_batch)
        with torch.no_grad():
            root_target_z, root_target_memory = target(root_batch)
        latent_state = _merge_batch_state((z0, m0), root_batch)
        optimizer.zero_grad(set_to_none=True)
        step_losses, parts = [], defaultdict(float)
        current_ids = root_ids
        for depth in range(horizon):
            edge_batch = [chain[depth] for _, chain in chosen]
            actions = torch.tensor([edge.action for edge in edge_batch], dtype=torch.long, device=device)
            child_ids = [edge.dst for edge in edge_batch]
            source_ids = [edge.src for edge in edge_batch]
            child_batch = pack_states(child_ids, states, scalers, device)
            source_target_batch = pack_states(source_ids, states, scalers, device)
            target_z, target_mem = _target_latents(target, child_batch)
            source_z_teacher, _ = _target_latents(target, source_target_batch)
            child_batch_loss = dict(child_batch)
            child_batch_loss["_memory_z"] = target_mem
            next_state, aux = transition(latent_state, actions)

            # Per-edge coarse hidden-change target; it teaches change scale,
            # not exact native channels or token/vocabulary outputs.
            losses = []
            for i, edge in enumerate(edge_batch):
                src_obj, dst_obj = states[edge.src], states[edge.dst]
                target_one = {k: v[i:i+1] for k, v in child_batch_loss.items()
                              if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(edge_batch)}
                pred_one = {k: v[i:i+1] for k, v in next_state.items()
                            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(edge_batch)}
                aux_one = {k: v[i:i+1] for k, v in aux.items()
                           if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(edge_batch)}
                # Source target code is only used to form the supervised delta;
                # rollout state itself is the student prediction.
                target_one["_state_obj"] = dst_obj
                total, items = loss_for_step(pred_one, aux_one, target_one, target_z[i:i+1],
                                             src_obj, latent_state["z"][i:i+1],
                                             source_z_teacher[i:i+1],
                                             scalers, change_thresholds, event_pos_weight)
                losses.append(total)
                for name, value in items.items():
                    parts[name] += float(value.detach()) / (horizon * len(edge_batch))
            step_loss = torch.stack(losses).mean()
            step_losses.append(step_loss)
            latent_state = next_state
            current_ids = child_ids

        root_valid = root_batch["valid"]
        root_consistency = _masked_mean(
            F.smooth_l1_loss(z0, root_target_z, reduction="none"), root_valid)
        root_mem_valid = root_batch["memory_valid"].float()[:, None]
        root_memory_consistency = _masked_mean(
            F.smooth_l1_loss(m0, root_target_memory, reduction="none"), root_mem_valid)
        root_var = variance_covariance_loss(z0, root_valid)
        total_loss = (torch.stack(step_losses).mean() + 0.2 * root_consistency
                      + 0.1 * root_memory_consistency + 0.05 * root_var)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Nonfinite loss in {feature_set} update {update}")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(online.parameters()) + list(transition.parameters()), 1.0)
        optimizer.step()
        _ema_update(target, online, args.ema_momentum)

        if update % args.log_every == 0 or update == 1:
            summary = {k: round(v, 5) for k, v in parts.items()}
            summary.update(root_consistency=round(float(root_consistency.detach()), 5),
                           root_variance=round(float(root_var.detach()), 5))
            print(f"[train] variant={feature_set} update={update}/{args.updates} "
                  f"horizon={horizon} loss={float(total_loss.detach()):.5f} "
                  f"parts={json.dumps(summary)}", flush=True)

        if update in checkpoints:
            val = evaluate_rollouts(online, transition, target, val_paths, states, scalers,
                                    change_thresholds, device, max_paths=args.eval_paths,
                                    save_arrays=False)
            score = validation_selection_score(val)
            row = {"update": update, "horizon_trained": horizon,
                   "selection_score": score, "validation": val["by_horizon_action"]}
            curve.append(row)
            print(f"[validation] variant={feature_set} update={update} score={score:.5f}", flush=True)
            if score < best_score:
                best_score, best_update = score, update
                best_payload = {
                    "online": {k: v.detach().cpu().clone() for k, v in online.state_dict().items()},
                    "target": {k: v.detach().cpu().clone() for k, v in target.state_dict().items()},
                    "transition": {k: v.detach().cpu().clone() for k, v in transition.state_dict().items()},
                }

    if best_payload is None:
        raise RuntimeError("Training produced no validation checkpoint")
    online.load_state_dict(best_payload["online"])
    target.load_state_dict(best_payload["target"])
    transition.load_state_dict(best_payload["transition"])
    return online, transition, target, {"best_update": best_update,
                                       "best_validation_score": best_score,
                                       "training_curve": curve}


@torch.no_grad()
def evaluate_rollouts(online, transition, target, paths, states, scalers,
                      change_thresholds, device, max_paths=None, save_arrays=False):
    online.eval(); transition.eval(); target.eval()
    if max_paths:
        paths = paths[:max_paths]
    rows = []
    arrays = defaultdict(list)
    by_group = defaultdict(list)
    for root, chain in paths:
        root_batch = pack_states([root], states, scalers, device)
        current_z, current_mem = online(root_batch)
        current = _merge_batch_state((current_z, current_mem), root_batch)
        for depth, edge in enumerate(chain, start=1):
            action = torch.tensor([edge.action], dtype=torch.long, device=device)
            predicted, aux = transition(current, action)
            child_batch = pack_states([edge.dst], states, scalers, device)
            true_z, true_mem = target(child_batch)
            source_batch = pack_states([edge.src], states, scalers, device)
            true_source_z, _ = target(source_batch)
            valid = child_batch["valid"].bool()
            err = (predicted["z"] - true_z).square()
            z_mse = _masked_mean(err, valid).item()
            z_rmse = math.sqrt(max(z_mse, 0.0))
            z_norm = torch.linalg.vector_norm(predicted["z"], dim=-1)
            t_norm = torch.linalg.vector_norm(true_z, dim=-1)
            cosine_rows = F.cosine_similarity(predicted["z"], true_z, dim=-1, eps=1e-6)
            cosine = _masked_mean(cosine_rows, valid).item()
            true_parent_aligned, overlap = align_frame(true_source_z, source_batch["geometry"],
                                                       child_batch["geometry"])
            pred_parent_aligned, _ = align_frame(current["z"], current["geometry"],
                                                 child_batch["geometry"])
            delta_valid = (overlap > 0) & valid
            pred_delta = predicted["z"] - pred_parent_aligned
            true_delta = true_z - true_parent_aligned
            delta_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-6)
            delta_cosine = _masked_mean(delta_cos, delta_valid).item()
            delta_abs = (pred_delta - true_delta).square().mean(-1).sqrt()
            true_mag = true_delta.square().mean(-1).sqrt()
            rel = delta_abs / true_mag.clamp_min(0.05)
            delta_rel = _masked_mean(rel, delta_valid & (true_mag >= 0.05)).item()

            eligible = ((child_batch["mask_before"] >= .5)
                        & (child_batch["forward_active"] >= .5) & valid)
            event_target = child_batch["event"] >= .5
            event_score = aux["event_logits"].sigmoid() >= .5
            event_pred = predicted["event"] >= .5
            tp = int((event_score & event_target & eligible).sum())
            fp = int((event_score & ~event_target & eligible).sum())
            fn = int((~event_score & event_target & eligible).sum())
            precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
            event_acc = float(((event_pred == event_target) & eligible).sum().item()
                              / max(1, int(eligible.sum().item())))
            mask_positions = child_batch["span"].bool() & valid
            mask_acc = float((((predicted["mask"] >= .5) == (child_batch["mask"] >= .5))
                              & mask_positions).sum().item() / max(1, int(mask_positions.sum().item())))
            valid_acc = float(((predicted["valid"] >= .5) == (child_batch["valid"] >= .5))
                              .float().mean().item())
            new_logical = (child_batch["span"].bool() & ~source_batch["span"].bool()
                           & valid)
            new_latent_rmse = (math.sqrt(_masked_mean(err, new_logical).item())
                               if new_logical.any() else None)
            change_labels, change_valid = target_change_labels(
                states[edge.src], states[edge.dst], scalers, change_thresholds, device)
            change_pred = aux["change_logits"].argmax(-1)
            change_acc = float(((change_pred == change_labels) & change_valid).sum().item()
                               / max(1, int(change_valid.sum().item())))
            horizon_action = (depth, "R" if edge.action == 0 else "E")
            record = {
                "root": root, "question_id": states[root].qid, "src": edge.src,
                "dst": edge.dst, "action": "R" if edge.action == 0 else "E",
                "horizon": depth, "latent_rmse": z_rmse, "latent_cosine": cosine,
                "latent_delta_cosine": delta_cosine, "latent_delta_relative_error": delta_rel,
                "event_precision": precision, "event_recall": recall,
                "event_accuracy_eligible": event_acc, "mask_accuracy_proposal": mask_acc,
                "valid_position_accuracy": valid_acc,
                "extend_new_slot_rmse": new_latent_rmse,
                "coarse_change_bin_accuracy": change_acc,
                "eligible_positions": int(eligible.sum().item()),
                "event_positives": int(event_target[eligible].sum().item()),
                "new_extend_positions": int(new_logical.sum().item()),
                "terminal_probability": float(predicted["terminal"].item()),
                "terminal_target": bool(states[edge.dst].meta.get("native_terminal", False)),
            }
            rows.append(record)
            by_group[horizon_action].append(record)
            if save_arrays:
                arrays["predicted_z"].append(predicted["z"][0].detach().cpu().numpy().astype(np.float16))
                arrays["target_z"].append(true_z[0].detach().cpu().numpy().astype(np.float16))
                arrays["predicted_delta"].append(pred_delta[0].detach().cpu().numpy().astype(np.float16))
                arrays["target_delta"].append(true_delta[0].detach().cpu().numpy().astype(np.float16))
                arrays["event_scores"].append(aux["event_logits"][0].sigmoid().detach().cpu().numpy().astype(np.float16))
            current = predicted

    def avg(items, key):
        values = [float(r[key]) for r in items if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else None
    groups = {}
    for (horizon, action), group in sorted(by_group.items()):
        groups[f"H{horizon}_{action}"] = {
            "edge_count": len(group), "question_count": len({r["question_id"] for r in group}),
            **{key: avg(group, key) for key in (
                "latent_rmse", "latent_cosine", "latent_delta_cosine",
                "latent_delta_relative_error", "event_precision", "event_recall",
                "event_accuracy_eligible", "mask_accuracy_proposal",
                "valid_position_accuracy",
                "extend_new_slot_rmse", "coarse_change_bin_accuracy", "terminal_probability")},
            "eligible_positions": sum(r["eligible_positions"] for r in group),
            "native_event_positives": sum(r["event_positives"] for r in group),
            "extend_new_positions": sum(r["new_extend_positions"] for r in group),
        }
    result = {"by_horizon_action": groups, "edge_count": len(rows),
              "question_count": len({r["question_id"] for r in rows}),
              "rows": rows}
    if save_arrays:
        result["arrays"] = {k: np.stack(v) if v else np.empty((0,)) for k, v in arrays.items()}
    return result


def validation_selection_score(report):
    groups = report["by_horizon_action"]
    values = []
    # Give R and E equal weight whenever each action is represented.
    for horizon in (1, 3):
        selected = [groups.get(f"H{horizon}_{action}") for action in ("R", "E")]
        selected = [row for row in selected if row and row["latent_rmse"] is not None]
        if selected:
            values.append(float(np.mean([row["latent_rmse"] for row in selected])))
    if not values:
        raise RuntimeError("Validation set has no H1/H3 paths")
    return float(np.mean(values))


def latent_collapse_report(z_rows):
    if not z_rows:
        return {"count": 0}
    x = np.concatenate(z_rows, axis=0).astype(np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 2:
        return {"count": len(x)}
    std = x.std(axis=0)
    centered = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular.square() if hasattr(singular, "square") else singular ** 2
    probs = energy / max(float(energy.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    return {"count": len(x), "mean_dimension_std": float(std.mean()),
            "p10_dimension_std": float(np.quantile(std, .1)),
            "p90_dimension_std": float(np.quantile(std, .9)),
            "near_constant_dimensions_std_lt_0_05": int((std < .05).sum()),
            "effective_rank": effective_rank}


def evaluate_root_latent(encoder, states, ids, scalers, device, max_states=512):
    encoder.eval()
    selected = ids[:max_states]
    z_rows = []
    for start in range(0, len(selected), 16):
        batch = pack_states(selected[start:start+16], states, scalers, device)
        z, _ = encoder(batch)
        valid = batch["valid"].bool()
        z_rows.append(z[valid].detach().cpu().numpy())
    return latent_collapse_report(z_rows)


def save_eval_arrays(path: Path, report):
    arrays = report.get("arrays", {})
    if not arrays:
        return
    np.savez_compressed(path, **arrays)
    public_rows = [{k: v for k, v in row.items() if k != "question_id"}
                   for row in report["rows"]]
    path.with_suffix(".jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in public_rows), encoding="utf-8")


def train_and_evaluate(args, states, edges, split, scalers, out):
    device = torch.device(args.train_device)
    edge_splits = split_edges(states, edges, split)
    path_splits = {part: enumerate_paths(states, edge_splits[part], split, part,
                                         args.horizon, args.paths_per_root)
                   for part in ("train", "validation", "test")}
    if not all(path_splits[p] for p in path_splits):
        counts = {p: len(path_splits[p]) for p in path_splits}
        raise RuntimeError(f"Need R/E paths in all question splits; got {counts}")
    change_thresholds = fit_change_thresholds(states, edge_splits["train"], scalers)
    event_pos_weight = _balanced_event_weight(edge_splits["train"], states)
    report = {
        "schema": SCHEMA,
        "objective": "action-conditioned per-position latent score dynamics and native event prediction",
        "excluded": ["token IDs", "token embeddings as targets", "vocabulary logits/probabilities",
                     "verifier inputs/labels", "accepted length", "latency", "exact 1536D reconstruction"],
        "data": {"states": len(states), "edges": len(edges),
                 "actions": {name: sum(e.action == action_id for e in edges)
                             for name, action_id in ACTION_TO_ID.items()},
                 "questions": {p: sum(v == p for v in split.values())
                              for p in ("train", "validation", "test")},
                 "paths": {p: len(path_splits[p]) for p in path_splits}},
        "question_split": split,
        "native_contract": audit_native_contract(states, edges),
        "config": {"latent_dim": args.latent_dim, "encoder_layers": args.encoder_layers,
                   "transition_layers": args.transition_layers, "updates": args.updates,
                   "batch_size": args.batch_size, "horizon": args.horizon,
                   "seed": args.seed, "ema_momentum": args.ema_momentum,
                   "feature_sets": ["native_only", "native_structure", "full_context"],
                   "change_bin_thresholds": change_thresholds,
                   "event_positive_weight": event_pos_weight},
        "feature_semantics": "64 signed learned score coordinates per physical position; individual coordinates are not probabilities or guaranteed human-named concepts",
        "variants": {},
        "feature_impact_vs_native_only": {},
    }

    for feature_set in ("native_only", "native_structure", "full_context"):
        started = time.time()
        online, transition, target, training = train_variant(
            feature_set, path_splits["train"], path_splits["validation"], states,
            scalers, args, change_thresholds, event_pos_weight, device)
        test = evaluate_rollouts(online, transition, target, path_splits["test"],
                                 states, scalers, change_thresholds, device,
                                 max_paths=args.test_paths, save_arrays=True)
        root_ids = sorted(sid for sid, state in states.items() if split[state.qid] == "test")
        collapse = evaluate_root_latent(target, states, root_ids, scalers, device)
        variant_dir = out / "models"
        variant_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": SCHEMA, "feature_set": feature_set,
                    "latent_dim": args.latent_dim,
                    "raw_hidden_dim": next(iter(states.values())).u.shape[-1],
                    "online_encoder": online.cpu().state_dict(),
                    "target_encoder": target.cpu().state_dict(),
                    "transition": transition.cpu().state_dict(),
                    "normalization": {k: torch.as_tensor(v) for k, v in scalers.items()},
                    "change_bin_thresholds": change_thresholds},
                   variant_dir / f"{feature_set}.pt")
        evaluation_dir = out / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        save_eval_arrays(evaluation_dir / f"{feature_set}_test_predictions.npz", test)
        report["variants"][feature_set] = {
            "training": training, "test": {"by_horizon_action": test["by_horizon_action"],
                                             "edge_count": test["edge_count"],
                                             "question_count": test["question_count"]},
            "test_latent_variance": collapse,
            "elapsed_seconds": time.time() - started,
        }
        del online, transition, target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[variant-done] {feature_set} elapsed={report['variants'][feature_set]['elapsed_seconds']:.1f}s",
              flush=True)

    # Direct feature value is measured on identical splits and architecture.
    base = report["variants"]["native_only"]["test"]["by_horizon_action"]
    for variant in ("native_structure", "full_context"):
        current = report["variants"][variant]["test"]["by_horizon_action"]
        metric_deltas = {}
        for key in sorted(set(base) | set(current)):
            base_row, current_row = base.get(key, {}), current.get(key, {})
            metric_deltas[key] = {}
            for metric in ("latent_rmse", "latent_cosine", "latent_delta_cosine",
                           "latent_delta_relative_error", "event_precision", "event_recall",
                           "event_accuracy_eligible", "mask_accuracy_proposal",
                           "valid_position_accuracy",
                           "extend_new_slot_rmse", "coarse_change_bin_accuracy"):
                a, b = base_row.get(metric), current_row.get(metric)
                metric_deltas[key][metric] = (None if a is None or b is None else float(b - a))
        report["feature_impact_vs_native_only"][variant] = metric_deltas
    report["interpretation"] = {
        "latent_rmse": "prediction error on the learned per-position score vector; not token accuracy",
        "latent_delta_relative_error": "error in score-vector change divided by real edge change magnitude; excludes near-zero changes",
        "event_scores": "probability of a native structural commit event; not token confidence",
        "H1_H3": "only the real root is encoded; later transitions consume predicted latent and predicted structure",
        "feature_ablation": "native_only vs native_structure measures structural context; full_context additionally uses the captured prefix-KV summary",
        "limitations": "one seed per feature ablation; the test evaluates latent drafter dynamics, not verifier acceptance or deployment latency",
    }
    (out / "latent_score_drafter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _write_config(out, args, capture_manifest, commit):
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(source_commit=commit, capture_manifest=capture_manifest)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dllm-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--extend-size", type=int, default=8)
    parser.add_argument("--max-proposal-tokens", type=int, default=64)
    parser.add_argument("--max-refinement-steps", type=int, default=3)
    parser.add_argument("--raw-top-k", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--transition-layers", type=int, default=2)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.995)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--paths-per-root", type=int, default=12)
    parser.add_argument("--eval-paths", type=int, default=256)
    parser.add_argument("--test-paths", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--drafter-device", type=int, default=0)
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    return parser.parse_args()


def make_capture_args(args, capture_dir):
    return argparse.Namespace(
        out_dir=capture_dir, num_questions=args.num_questions, seed=args.seed,
        threshold=args.threshold, extend_size=args.extend_size,
        max_proposal_tokens=args.max_proposal_tokens,
        max_refinement_steps=args.max_refinement_steps,
        physical_block_size=32, small_block_size=8, raw_top_k=args.raw_top_k,
        drafter_device=args.drafter_device, shard_rows=16,
        tokenizer_name=args.tokenizer_name, dllm_dir=args.dllm_dir,
        wm_v2_capture_native_latent=True, wm_v2_cut_layer=24,
        wm_v2_suffix_layers=4, wm_v2_memory_window=128,
    )


def main():
    args = parse_args()
    if args.num_questions < 20:
        raise ValueError("Need at least 20 questions for a question-disjoint 60/20/20 split")
    if args.threshold != 0.5 or args.extend_size != 8 or args.max_proposal_tokens != 64:
        raise ValueError("This test is pinned to native threshold .5, E=8, and proposal cap 64")
    if not 0 <= args.max_refinement_steps <= 3 or args.horizon != 3:
        raise ValueError("This test uses up to 3 native R steps and H1-H3 rollout")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {args.out_dir}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle GPU T4 x2: native dLLM capture on GPU0, latent training on GPU1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = args.out_dir.parent / (args.out_dir.name + "_raw_capture")
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    status = {"status": "partial", "stage": "capture", "schema": SCHEMA}
    (args.out_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    features_saved = False
    try:
        print("[capture] Fresh greedy Fast-dLLM native GSM8K R/E states; verifier is not loaded.", flush=True)
        capture_manifest = collect_native(make_capture_args(args, capture_dir))
        states, edges = load_teacher(capture_dir)
        if len({s.qid for s in states.values()}) < 20:
            raise RuntimeError("Native capture produced fewer than 20 questions with states")
        split = split_questions({s.qid for s in states.values()}, args.split_seed)
        scalers = fit_scalers(states, split)
        contract = audit_native_contract(states, edges)
        sanitize_capture(capture_dir, args.out_dir, states, edges, split, scalers)
        features_saved = True
        (args.out_dir / "native_contract_audit.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.out_dir / "split.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
        status.update(stage="training", question_count=len(split), state_count=len(states), edge_count=len(edges))
        (args.out_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
        _write_config(args.out_dir, args, capture_manifest,
                      __import__("os").environ.get("WM_SCORE_SOURCE_COMMIT"))
        report = train_and_evaluate(args, states, edges, split, scalers, args.out_dir)
        report["capture"] = {k: capture_manifest.get(k) for k in (
            "schema_version", "question_count", "state_count", "edge_count",
            "action_counts", "terminal_counts", "terminal_transition_count", "elapsed_capture_seconds")}
        (args.out_dir / "latent_score_drafter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (args.out_dir / "README.md").write_text(
            "# Score-valued latent drafter dynamics test\n\n"
            "100-question GSM8K native Fast-dLLM capture. The student emits 64 learned scalar coordinates per physical position, predicts those coordinates and scheduler events after R/E, and rolls out H1-H3 from a real root without feeding child observations. No verifier or token-generation target is used. See latent_score_drafter_report.json.\n",
            encoding="utf-8")
        status.update(status="complete", stage="complete")
        (args.out_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
        archive = args.out_dir.with_suffix(".zip")
        tmp = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for path in sorted(args.out_dir.rglob("*")):
                if path.is_file() and path != tmp:
                    zf.write(path, path.relative_to(args.out_dir).as_posix())
        tmp.replace(archive)
        print(f"[result-zip] {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    except Exception:
        (args.out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if not features_saved:
            try:
                states, edges = load_teacher(capture_dir, require_edges=False)
                qids = {s.qid for s in states.values()}
                if len(qids) >= 5:
                    split = split_questions(qids, args.split_seed)
                    scalers = fit_scalers(states, split)
                    sanitize_capture(capture_dir, args.out_dir, states, edges, split, scalers)
                    features_saved = True
                    status.update(partial_capture_questions=len(qids), partial_capture_states=len(states))
            except Exception as preserve_error:
                (args.out_dir / "capture_preservation_error.txt").write_text(
                    repr(preserve_error), encoding="utf-8")
        status.update(status="partial", failed_stage=status.get("stage"),
                      sanitized_features_preserved=features_saved)
        (args.out_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if any(args.out_dir.iterdir()):
            archive = args.out_dir.with_suffix(".zip")
            tmp = archive.with_suffix(".zip.tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
                for path in sorted(args.out_dir.rglob("*")):
                    if path.is_file() and path != tmp:
                        zf.write(path, path.relative_to(args.out_dir).as_posix())
            tmp.replace(archive)
            print(f"[partial-zip] {archive}", flush=True)
        raise
    finally:
        # Raw capture contains teacher token IDs used only to create E prefixes;
        # the packaged sanitized feature set contains none of them.
        if features_saved and capture_dir.exists():
            shutil.rmtree(capture_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
