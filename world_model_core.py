"""Small task-relevant latent dynamics. No LLM weights, timing or oracle inputs."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import copy
import random

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Observation:
    uid: str
    question: str
    round_id: int
    ids: torch.Tensor             # CPU int64 [L, 2]: native and same-forward STOP
    hidden: torch.Tensor          # CPU fp16 [L, 2, hidden_dim], cached native rows
    gaps: torch.Tensor            # CPU fp16 [L, top_k], top-k minus top-1
    scalars: torch.Tensor         # CPU fp32 [L, 16], explicit validity/provenance
    context: torch.Tensor         # CPU fp32 [8], no verifier-derived information
    accepted: int | None = None   # target ONLY
    prefix_ids: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    history: torch.Tensor | None = None
    teacher_margin: torch.Tensor | None = None  # target ONLY; never encoder input
    teacher_features: torch.Tensor | None = None  # current verifier, target ONLY
    verifier_history: torch.Tensor | None = None  # preceding actual STOPs only
    teacher_aux_features: torch.Tensor | None = None  # verifier distribution summaries, target only
    teacher_topk_ids: torch.Tensor | None = None      # verifier distillation support, target only
    teacher_topk_logits: torch.Tensor | None = None   # verifier distillation support, target only
    teacher_logsumexp: torch.Tensor | None = None     # exact full-vocabulary normalizer
    teacher_is_actual: bool = False                   # false for counterfactual shadow probes

    @property
    def length(self):
        return len(self.ids)


@dataclass
class Latent:
    tokens: torch.Tensor
    global_state: torch.Tensor
    lengths: torch.Tensor
    context: torch.Tensor

    def take(self, indices):
        return Latent(*(x.index_select(0, indices) for x in (
            self.tokens, self.global_state, self.lengths, self.context)))


def valid_positions(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def positions(width, dim, device):
    pos = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, dim, 2, device=device).float()
                            * (-9.210340371976184 / dim))
    output = torch.empty(width, dim, device=device)
    output[:, 0::2] = torch.sin(pos * frequencies)
    output[:, 1::2] = torch.cos(pos * frequencies)
    return output


def transformer(dim, layers, dropout):
    layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout,
                                       activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim),
                                 enable_nested_tensor=False)


def contextualize(network, tokens, global_state, lengths):
    valid = valid_positions(lengths, tokens.shape[1])
    sequence = torch.cat([global_state[:, None], tokens], dim=1)
    padding = torch.cat([torch.zeros(len(tokens), 1, dtype=torch.bool,
                                     device=tokens.device), ~valid], dim=1)
    encoded = network(sequence, src_key_padding_mask=padding)
    return encoded[:, 1:] * valid[..., None], encoded[:, 0]


class NativeEncoder(nn.Module):
    def __init__(self, hidden_dim, token_dim, top_k=32, dim=128, layers=2, dropout=0.1):
        super().__init__()
        self.hidden_projection = nn.Linear(hidden_dim * 2, 128)
        self.token_projection = nn.Linear(token_dim, 32)
        self.gap_projection = nn.Linear(top_k, 32)
        self.scalar_projection = nn.Linear(16, 16)
        self.fuse = nn.Linear(128 + 64 + 32 + 16, dim)
        self.global_projection = nn.Linear(8, dim)
        self.network = transformer(dim, layers, dropout)
        self.dim = dim

    def forward(self, batch):
        features = torch.cat([
            self.hidden_projection(batch["hidden"].flatten(2)),
            self.token_projection(batch["token_vectors"]).flatten(2),
            self.gap_projection(batch["gaps"]),
            self.scalar_projection(batch["scalars"]),
        ], dim=-1)
        tokens = self.fuse(features)
        tokens = tokens + positions(tokens.shape[1], self.dim, tokens.device)
        global_state = self.global_projection(batch["context"])
        tokens, global_state = contextualize(self.network, tokens, global_state,
                                            batch["lengths"])
        return Latent(tokens, global_state, batch["lengths"], batch["context"])


class ActionDynamics(nn.Module):
    def __init__(self, dim=128, layers=2, dropout=0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(2, dim)  # R=0, E=1
        self.delta_embedding = nn.Linear(1, dim)
        self.extension_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.context_projection = nn.Linear(8, dim)
        self.network = transformer(dim, layers, dropout)
        self.dim = dim

    def forward(self, state, actions, extension_size):
        if bool(((actions != 0) & (actions != 1)).any()) or extension_size < 1:
            raise ValueError("Expected R=0 or E=1 and a positive extension size")
        increments = actions * extension_size
        lengths = state.lengths + increments
        width = int(lengths.max().item())
        old = F.pad(state.tokens, (0, 0, 0, max(0, width-state.tokens.shape[1])))[:, :width]
        old_valid = valid_positions(state.lengths, width)
        queries = self.extension_query + positions(width, self.dim, old.device)
        tokens = torch.where(old_valid[..., None], old, queries[None])
        action = self.action_embedding(actions) + self.delta_embedding(increments.float()[:, None] / 64)
        context = state.context.clone()
        context[:, 1] = lengths.float() / 64
        context[:, 2] = torch.where(actions.bool(), state.lengths.float() / 64, context[:, 2])
        context[:, 3] = torch.where(actions.bool(), 0.0, context[:, 3] + 1 / 3)
        tokens, global_state = contextualize(self.network, tokens + action[:, None],
            state.global_state + action + self.context_projection(context), lengths)
        return Latent(tokens, global_state, lengths, context)


def prefix_log_distribution(logits, lengths):
    """Proper P(K=0..L); padding classes get -inf. No independence assumption."""
    log_q = F.logsigmoid(logits)
    survival = F.pad(log_q.cumsum(-1), (1, 0), value=0.0)
    failure = F.pad(F.logsigmoid(-logits), (0, 1), value=0.0)
    classes = torch.arange(logits.shape[1] + 1, device=logits.device)[None]
    result = survival + failure
    result = torch.where(classes == lengths[:, None], survival, result)
    return result.masked_fill(classes > lengths[:, None], -torch.inf)


def acceptance_nll(logits, lengths, labels):
    valid = labels >= 0
    if not bool(valid.any()):
        return logits.sum() * 0
    if bool((labels[valid] > lengths[valid]).any()):
        raise ValueError("Accepted prefix exceeds proposal length")
    log_p = prefix_log_distribution(logits[valid], lengths[valid])
    return -log_p.gather(1, labels[valid, None]).mean()


def expected_acceptance(logits, lengths):
    survival = F.logsigmoid(logits).cumsum(-1).exp()
    return (survival * valid_positions(lengths, logits.shape[1])).sum(-1)


class AcceptanceWorldModel(nn.Module):
    def __init__(self, hidden_dim, token_dim, top_k=32, dim=128, layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, token_dim=token_dim, top_k=top_k,
                           dim=dim, layers=layers, dropout=dropout)
        self.encoder = NativeEncoder(**self.config)
        self.dynamics = ActionDynamics(dim, layers, dropout)
        self.head = nn.Sequential(nn.Linear(2 * dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def acceptance(self, state):
        global_rows = state.global_state[:, None].expand(-1, state.tokens.shape[1], -1)
        return self.head(torch.cat([state.tokens, global_rows], dim=-1)).squeeze(-1)

    def transition(self, state, actions, extension_size):
        return self.dynamics(state, actions, extension_size)


def pack_observations(observations, token_table, device, include_candidates=False):
    """The verifier label is deliberately separated from all encoder features."""
    width = max(o.length for o in observations)
    def stack(name):
        arrays = []
        for observation in observations:
            value = getattr(observation, name)
            pad = torch.zeros((width - observation.length, *value.shape[1:]), dtype=value.dtype)
            arrays.append(torch.cat([value, pad]))
        return torch.stack(arrays).to(device)
    ids = stack("ids").long()
    with torch.no_grad():
        vectors = F.embedding(ids.to(token_table.device), token_table).to(device).float()
    result = dict(hidden=stack("hidden").float(), gaps=stack("gaps").float(),
        scalars=stack("scalars").float(), token_vectors=vectors,
        context=torch.stack([o.context for o in observations]).to(device),
        lengths=torch.tensor([o.length for o in observations], device=device),
        labels=torch.tensor([-1 if o.accepted is None else o.accepted for o in observations],
                            dtype=torch.long, device=device))
    if all(o.prefix_ids is not None and o.topk_ids is not None for o in observations):
        topk = stack("topk_ids").long()
        prefix_width = max(len(o.prefix_ids) for o in observations)
        prefix_ids = torch.stack([F.pad(o.prefix_ids, (0, prefix_width-len(o.prefix_ids)))
                                  for o in observations]).to(device)
        with torch.no_grad():
            alternatives = F.embedding(topk.to(token_table.device), token_table).to(device).float()
            # Conditional top-K weights only, NOT full-vocabulary probabilities.
            weights = torch.softmax(result["gaps"], dim=-1)
            top_summary = (alternatives * weights[..., None]).sum(-2)
            prefix_vectors = F.embedding(prefix_ids.to(token_table.device), token_table).to(device).float()
        result.update(topk_vectors=top_summary, prefix_vectors=prefix_vectors,
                      prefix_lengths=torch.tensor([len(o.prefix_ids) for o in observations], device=device),
                      history=stack("history").float())
        if include_candidates:
            result['candidate_vectors'] = alternatives
    margins = torch.zeros(len(observations), width, device=device)
    teacher_valid = torch.zeros(len(observations), width, dtype=torch.bool, device=device)
    for row, observation in enumerate(observations):
        if observation.teacher_margin is not None:
            margins[row, :observation.length] = observation.teacher_margin.to(device)
            teacher_valid[row, :observation.length] = True
    result.update(teacher_margin=margins, teacher_valid=teacher_valid)
    if any(o.verifier_history is not None for o in observations):
        memories = [o.verifier_history if o.verifier_history is not None
                    else torch.empty(0, 48) for o in observations]
        memory_width = max(1, max(len(x) for x in memories))
        result['verifier_history'] = torch.stack([
            F.pad(x, (0, 0, 0, memory_width-len(x))) for x in memories]).to(device).float()
        result['verifier_history_lengths'] = torch.tensor([len(x) for x in memories], device=device)
    if any(o.teacher_features is not None for o in observations):
        features = torch.zeros(len(observations), width, 35, device=device)
        for row, observation in enumerate(observations):
            if observation.teacher_features is not None:
                features[row, :observation.length] = observation.teacher_features.to(device)
        result['teacher_features'] = features
    if any(o.teacher_aux_features is not None for o in observations):
        features = torch.zeros(len(observations), width, 6, device=device)
        for row, observation in enumerate(observations):
            if observation.teacher_aux_features is not None:
                features[row, :observation.length] = observation.teacher_aux_features.to(device)
        result['teacher_aux_features'] = features
    if any(o.teacher_topk_ids is not None for o in observations):
        k = max((int(o.teacher_topk_ids.shape[-1]) for o in observations
                 if o.teacher_topk_ids is not None), default=1)
        ids = torch.zeros(len(observations), width, k, dtype=torch.long, device=device)
        logits = torch.zeros(len(observations), width, k, dtype=torch.float32, device=device)
        normalizer = torch.zeros(len(observations), width, dtype=torch.float32, device=device)
        valid = torch.zeros(len(observations), width, dtype=torch.bool, device=device)
        for row, observation in enumerate(observations):
            if observation.teacher_topk_ids is None:
                continue
            count = min(k, observation.teacher_topk_ids.shape[-1])
            ids[row, :observation.length, :count] = observation.teacher_topk_ids[:, :count].to(device)
            logits[row, :observation.length, :count] = observation.teacher_topk_logits[:, :count].to(device).float()
            normalizer[row, :observation.length] = observation.teacher_logsumexp.to(device).float()
            valid[row, :observation.length] = True
        result.update(teacher_topk_ids=ids, teacher_topk_logits=logits,
                      teacher_logsumexp=normalizer, teacher_distribution_valid=valid)
    result['teacher_actual'] = torch.tensor(
        [bool(o.teacher_is_actual) for o in observations], dtype=torch.bool, device=device)
    return result


class ExperienceReplay:
    """Bounded real graph: no edges across questions/rounds and no synthetic labels."""
    def __init__(self, max_states=512, seed=42, sampling_mode="natural"):
        self.nodes = OrderedDict()
        self.edges = []
        self.max_states = max_states
        self.rng = random.Random(seed)
        # Keep current-state supervision identical when transition sampling changes.
        self.state_rng = random.Random(seed + 1_000_003)
        self.sampling_mode = sampling_mode
        self.sampled_edge_counts = defaultdict(int)

    def add_node(self, observation):
        self.nodes[observation.uid] = observation
        while len(self.nodes) > self.max_states:
            removed, _ = self.nodes.popitem(last=False)
            self.edges = [e for e in self.edges if removed not in e[:2]]

    def add(self, parent, child, action):
        if (parent.question, parent.round_id) != (child.question, child.round_id):
            raise ValueError("Cannot cross a verification round or question")
        if action not in ("R", "E"):
            raise ValueError("Unknown action")
        if (action == "R" and parent.length != child.length) or (
                action == "E" and parent.length >= child.length):
            raise ValueError("Action/length mismatch")
        for o in (parent, child):
            self.add_node(o)
        edge = (parent.uid, child.uid, action)
        if edge not in self.edges:
            self.edges.append(edge)

    def _sample_root_edge(self):
        if self.sampling_mode == "natural":
            edge = self.rng.choice(self.edges)
        else:
            if self.sampling_mode not in ("action_balanced", "change_balanced"):
                raise ValueError(f"Unknown replay sampling mode: {self.sampling_mode}")
            by_action = defaultdict(list)
            by_action_change = defaultdict(lambda: defaultdict(list))
            for item in self.edges:
                parent_id, child_id, action = item
                by_action[action].append(item)
                if self.sampling_mode == "change_balanced":
                    parent, child = self.nodes[parent_id], self.nodes[child_id]
                    if parent.accepted is None or child.accepted is None:
                        change = "unknown"
                    else:
                        delta = child.accepted-parent.accepted
                        change = "gain" if delta > 0 else ("loss" if delta < 0 else "same")
                    by_action_change[action][change].append(item)

            action = self.rng.choice(sorted(by_action))
            if self.sampling_mode == "action_balanced":
                edge = self.rng.choice(by_action[action])
            else:
                change = self.rng.choice(sorted(by_action_change[action]))
                edge = self.rng.choice(by_action_change[action][change])
        parent, child, action = edge
        source, target = self.nodes[parent], self.nodes[child]
        if source.accepted is None or target.accepted is None:
            change = "unknown"
        else:
            delta = target.accepted-source.accepted
            change = "gain" if delta > 0 else ("loss" if delta < 0 else "same")
        self.sampled_edge_counts[f"{action}/{change}"] += 1
        return edge

    def sample(self, batch_size, horizon):
        if not self.edges:
            raise ValueError("Replay has no real transitions")
        outgoing = {}
        for p, c, a in self.edges:
            outgoing.setdefault(p, []).append((c, a))
        paths = []
        for _ in range(batch_size):
            parent, _, _ = self._sample_root_edge()
            nodes, actions = [self.nodes[parent]], []
            for _ in range(self.rng.randint(1, horizon)):
                if parent not in outgoing:
                    break
                child, action = self.rng.choice(outgoing[parent])
                nodes.append(self.nodes[child])
                actions.append(action)
                parent = child
            paths.append((nodes, actions))
        return paths


class WorldModelLearner:
    def __init__(self, model, token_table, device, extension_size=8, lr=3e-4,
                 ema_decay=0.99, latent_weight=0.1, warmup_updates=8, horizon_warmup=24,
                 teacher_weight=0.1, structure_weight=0.1, delta_weight=0.0):
        self.model = model.to(device)
        self.target_encoder = copy.deepcopy(model.encoder).eval().requires_grad_(False)
        self.token_table = token_table.detach()
        self.device, self.extension_size = device, extension_size
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.ema_decay, self.latent_weight = ema_decay, latent_weight
        self.warmup_updates, self.horizon_warmup = warmup_updates, horizon_warmup
        self.teacher_weight, self.structure_weight, self.delta_weight = teacher_weight, structure_weight, delta_weight
        self.updates = 0
        self.dynamics_updates = 0

    def pack(self, observations):
        return pack_observations(observations, self.token_table, self.device,
                                 include_candidates=getattr(self.model, 'needs_candidates', False))

    def update(self, replay, batch_size=8, max_horizon=3):
        horizon = 1 if self.updates < self.horizon_warmup else max_horizon
        labeled = [o for o in replay.nodes.values() if o.accepted is not None]
        dynamics_ready = bool(replay.edges) and self.updates >= self.warmup_updates
        if not labeled and not dynamics_ready:
            return None  # No supervised signal yet: no fake zero-label/optimizer update.
        paths = replay.sample(batch_size, horizon) if dynamics_ready else []
        path_depth_counts = dict((str(depth), sum(len(p[1]) == depth for p in paths))
                                 for depth in range(1, max_horizon+1))
        self.model.train()
        current_losses = []
        if labeled:
            # Include STOP-at-root and terminal states even when they have no R/E edge.
            batch = self.pack(replay.state_rng.choices(labeled, k=batch_size))
            actual = self.model.encoder(batch)
            current_losses.append(acceptance_nll(self.model.acceptance(actual), actual.lengths,
                                                batch["labels"]))
        rollout_losses, latent_losses, auxiliary_losses, structural_losses = [], [], [], []
        delta_losses = []
        if labeled and hasattr(self.model, "teacher_loss"):
            auxiliary_losses.append(self.model.teacher_loss(actual, batch))
        if paths:
            batch = self.pack([p[0][0] for p in paths])
            imagined = self.model.encoder(batch)
            for step in range(max(len(p[1]) for p in paths)):
                active = [i for i, p in enumerate(paths) if len(p[1]) > step]
                imagined = imagined.take(torch.tensor(active, device=self.device))
                paths = [paths[i] for i in active]
                actions = torch.tensor([int(p[1][step] == "E") for p in paths], device=self.device)
                if self.delta_weight:
                    source_prediction = expected_acceptance(self.model.acceptance(imagined), imagined.lengths)
                imagined = self.model.transition(imagined, actions, self.extension_size)
                future = self.pack([p[0][step+1] for p in paths])
                if self.delta_weight:
                    source_labels = torch.tensor([-1 if p[0][step].accepted is None else p[0][step].accepted
                                                  for p in paths], device=self.device)
                    labeled_pair = (source_labels >= 0) & (future['labels'] >= 0)
                    if labeled_pair.any():
                        delta_prediction = expected_acceptance(self.model.acceptance(imagined), imagined.lengths)-source_prediction
                        delta_target = future['labels']-source_labels
                        delta_losses.append(F.smooth_l1_loss(delta_prediction[labeled_pair],
                                                            delta_target[labeled_pair].float()))
                # True child ONLY supplies targets/current-observation supervision.
                if not torch.equal(imagined.lengths, future["lengths"]):
                    raise RuntimeError("Imagined/real length mismatch")
                actual = self.model.encoder(future)
                if hasattr(self.model, "teacher_loss"):
                    auxiliary_losses.append(self.model.teacher_loss(actual, future))
                    if getattr(self.model, 'teach_imagined', False):
                        auxiliary_losses.append(self.model.teacher_loss(imagined, future))
                    structural_losses.append(self.model.structural_loss(imagined, future))
                current_losses.append(acceptance_nll(self.model.acceptance(actual),
                                                       actual.lengths, future["labels"]))
                with torch.no_grad():
                    target = self.target_encoder(future)
                mask = valid_positions(imagined.lengths, imagined.tokens.shape[1])
                distance = 1 - F.cosine_similarity(imagined.tokens, target.tokens, dim=-1)
                if hasattr(self.model, "region_mean"):
                    start = imagined.context[:, 2] * 64
                    focus = mask & (torch.arange(mask.shape[1], device=self.device)[None] >= start[:, None])
                    token_loss = self.model.region_mean(distance, focus) + .1*self.model.region_mean(distance, mask & ~focus)
                else:
                    token_loss = ((distance * mask).sum(-1) / imagined.lengths).mean()
                latent_losses.append(token_loss + (1-F.cosine_similarity(imagined.global_state, target.global_state)).mean())
                rollout_losses.append(acceptance_nll(self.model.acceptance(imagined),
                                                       imagined.lengths, future["labels"]))
        zero = next(self.model.parameters()).sum() * 0
        current = torch.stack(current_losses).mean() if current_losses else zero
        latent = torch.stack(latent_losses).mean() if latent_losses else zero
        rollout = torch.stack(rollout_losses).mean() if rollout_losses else zero
        teacher = torch.stack(auxiliary_losses).mean() if auxiliary_losses else zero
        structure = torch.stack(structural_losses).mean() if structural_losses else zero
        delta = torch.stack(delta_losses).mean() if delta_losses else zero
        loss = (current + self.latent_weight * latent + getattr(self, 'rollout_weight', 1.)*rollout + self.teacher_weight*teacher
                + self.structure_weight*structure + self.delta_weight*delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite world-model loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        if hasattr(self.model, 'update_teacher_ema'):
            self.model.update_teacher_ema(self.ema_decay)
        with torch.no_grad():
            for dst, src in zip(self.target_encoder.parameters(), self.model.encoder.parameters()):
                dst.lerp_(src, 1 - self.ema_decay)
        self.updates += 1
        self.dynamics_updates += int(bool(paths))
        return dict(update=self.updates, loss=float(loss.detach()), current_nll=float(current.detach()),
                    rollout_nll=float(rollout.detach()), latent_loss=float(latent.detach()),
                    grad_norm=float(grad_norm), horizon=horizon if paths else 0,
                    teacher_loss=float(teacher.detach()), structural_loss=float(structure.detach()),
                    delta_loss=float(delta.detach()),
                    sampled_path_depth_counts=path_depth_counts,
                    dynamics_trained=bool(paths), labeled_replay_states=len(labeled),
                    replay_states=len(replay.nodes))

    @torch.no_grad()
    def predict(self, observations):
        self.model.eval()
        batch = self.pack(observations)
        state = self.model.encoder(batch)
        logits = self.model.acceptance(state)
        return expected_acceptance(logits, state.lengths).cpu().tolist()

    def checkpoint(self):
        return dict(schema=getattr(self.model, "schema", "acceptance_world_model_v1"), model_config=self.model.config,
            model=self.model.state_dict(), target_encoder=self.target_encoder.state_dict(),
            optimizer=self.optimizer.state_dict(), updates=self.updates,
            dynamics_updates=self.dynamics_updates,
            token_embedding_included=False,
            token_embedding_source="frozen drafter input embeddings; load separately")
