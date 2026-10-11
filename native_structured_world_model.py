"""A structured latent probe anchored to Fast-dLLM's native state.

This is deliberately not a token generator. The frozen native Fast-dLLM shell
creates real R/E child states; this model learns (a) verifier prefix acceptance
and (b) action-conditioned latent transitions from those observed children.
Token content comes from the pretrained dLLM embedding table, never random
token identity codes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from world_model_core import Latent, positions, transformer, valid_positions


@dataclass
class NativeStructuredLatent(Latent):
    mask_probs: torch.Tensor
    frontier: torch.Tensor
    event_probs: torch.Tensor

    def take(self, indices):
        return NativeStructuredLatent(*(value.index_select(0, indices) for value in (
            self.tokens, self.global_state, self.lengths, self.context,
            self.mask_probs, self.frontier, self.event_probs)))


class NativeStructuredEncoder(nn.Module):
    """Fuse native layers, candidate/content, scheduler state and prefix memory."""

    VARIANTS = {
        "full", "no_hidden", "no_content", "no_candidates", "no_prefix",
        "no_logits", "no_geometry", "no_history",
    }

    def __init__(self, hidden_dim, token_dim, top_k, dim, num_hidden_layers, dropout):
        super().__init__()
        if dim % 8:
            raise ValueError("Native structured latent width must be divisible by 8")
        self.dim = dim
        self.variant = "full"
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, dim)
        self.layer_identity = nn.Parameter(torch.randn(num_hidden_layers, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.layer_mixer = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(dim),
                                                  enable_nested_tensor=False)
        self.native_content = nn.Linear(token_dim, dim)
        self.stop_content = nn.Linear(token_dim, dim)
        self.candidate_content = nn.Linear(token_dim, dim)
        self.logit_features = nn.Sequential(nn.LayerNorm(top_k), nn.Linear(top_k, dim))
        self.scalar_features = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.history_features = nn.Linear(4, dim)
        self.context_features = nn.Linear(8, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 64, dim) * 0.01)
        self.prefix_projection = nn.Linear(token_dim, dim)
        self.prefix_queries = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.prefix_attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.proposal_prefix_attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.network = transformer(dim, 3, dropout)
        self.dropout = nn.Dropout(dropout)

    def _prefix_window(self, batch):
        vectors = batch["prefix_vectors"]
        lengths = batch["prefix_lengths"]
        window = 512
        rows, row_lengths = [], []
        max_len = max(1, min(window, int(lengths.max().item())))
        for index, length_value in enumerate(lengths):
            length = int(length_value.item())
            begin = max(0, length - max_len)
            row = vectors[index, begin:length]
            valid = len(row)
            if valid == 0:
                row = vectors.new_zeros((1, vectors.shape[-1]))
                valid = 1
            row = F.pad(row, (0, 0, 0, max_len-len(row)))
            rows.append(row)
            row_lengths.append(valid)
        return torch.stack(rows), torch.tensor(row_lengths, device=vectors.device)

    def forward(self, batch):
        variant = self.variant
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown native feature lesion: {variant}")
        scalars = batch["scalars"].clone()
        history = batch["history"]
        gaps = batch["gaps"]
        token_vectors = batch["token_vectors"]

        if variant == "no_geometry":
            # Keep mask/commit/validity explicit; remove only positional/block
            # coordinates and newly-unmasked timing fields.
            scalars[:, :, 9:13] = 0
        if variant == "no_history":
            history = torch.zeros_like(history)

        hidden = batch["hidden"]
        if variant == "no_hidden":
            hidden = torch.zeros_like(hidden)
        batch_size, width, layer_count, _ = hidden.shape
        hidden = self.hidden_projection(self.hidden_norm(hidden))
        hidden = hidden + self.layer_identity[:layer_count][None, None]
        hidden = self.layer_mixer(hidden.reshape(batch_size * width, layer_count, self.dim))
        hidden = hidden.mean(dim=1).reshape(batch_size, width, self.dim)
        hidden = hidden * scalars[:, :, 3:4]
        if variant == "no_hidden":
            hidden = torch.zeros_like(hidden)

        if variant == "no_content":
            native = torch.zeros_like(token_vectors[:, :, 0])
            filled = torch.zeros_like(token_vectors[:, :, 1])
        else:
            native = token_vectors[:, :, 0]
            filled = token_vectors[:, :, 1]
        content = self.native_content(native) + self.stop_content(filled)
        if variant == "no_content":
            content = torch.zeros_like(content)

        candidate = batch["topk_vectors"]
        if variant == "no_candidates":
            candidate = torch.zeros_like(candidate)
        candidate = self.candidate_content(candidate) * scalars[:, :, 4:5]
        if variant == "no_candidates":
            candidate = torch.zeros_like(candidate)

        if variant == "no_logits":
            gaps = torch.zeros_like(gaps)
            candidate = torch.zeros_like(candidate)
        logits = self.logit_features(gaps.clamp(-40, 0) / 10)
        if variant == "no_logits":
            logits = torch.zeros_like(logits)
        history_features = self.history_features(history)
        if variant == "no_history":
            history_features = torch.zeros_like(history_features)
        position = self.position_embedding[:, :width]
        x = (hidden + content + candidate + logits
             + self.scalar_features(scalars)
             + history_features
             + position)

        global_state = self.context_features(batch["context"])
        if variant == "no_prefix":
            prefix_summary = torch.zeros_like(x)
        else:
            prefix_vectors, prefix_lengths = self._prefix_window(batch)
            prefix_memory = self.prefix_projection(prefix_vectors)
            prefix_memory = prefix_memory + positions(prefix_memory.shape[1], self.dim,
                                                       prefix_memory.device)
            queries = self.prefix_queries.expand(batch_size, -1, -1)
            prefix_memory, _ = self.prefix_attention(queries, prefix_memory, prefix_memory,
                key_padding_mask=~valid_positions(prefix_lengths, prefix_memory.shape[1]),
                need_weights=False)
            prefix_summary, _ = self.proposal_prefix_attention(x, prefix_memory, prefix_memory,
                need_weights=False)
        valid = valid_positions(batch["lengths"], width)
        sequence = torch.cat([global_state[:, None], x + prefix_summary], dim=1)
        padding = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device),
                             ~valid], dim=1)
        encoded = self.network(sequence, src_key_padding_mask=padding)
        tokens = encoded[:, 1:] * valid[:, :, None]
        mask_probs = scalars[:, :, 0].clamp(0, 1) * valid
        positions_ = torch.arange(width, device=x.device)[None]
        first_mask = torch.where(mask_probs > 0.5, positions_, width).min(-1).values
        frontier = torch.minimum(first_mask, batch["lengths"])
        event_probs = scalars[:, :, 2].clamp(0, 1) * valid
        return NativeStructuredLatent(tokens, encoded[:, 0], batch["lengths"], batch["context"],
                                      mask_probs, frontier, event_probs)


class NativeActionDynamics(nn.Module):
    """Separate R and E transitions; E commits old fill then adds new segment slots."""

    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.variant = "full"
        self.refine_action = nn.Parameter(torch.randn(dim) * 0.02)
        self.extend_action = nn.Parameter(torch.randn(dim) * 0.02)
        self.extension_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.context_projection = nn.Linear(8, dim)
        self.refine_network = transformer(dim, 2, dropout)
        self.extend_network = transformer(dim, 2, dropout)
        self.mask_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                       nn.GELU(), nn.Linear(dim // 2, 1))
        self.unmask_event_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                               nn.GELU(), nn.Linear(dim // 2, 1))

    def _run(self, tokens, lengths, global_state, context, action_embedding, network):
        action = action_embedding[None].expand(len(tokens), -1)
        conditioned_context = context.clone()
        conditioned_context[:, 1] = lengths.float() / 64
        global_state = global_state + action + self.context_projection(conditioned_context)
        tokens = tokens + action[:, None]
        encoded = network(torch.cat([global_state[:, None], tokens], dim=1),
            src_key_padding_mask=torch.cat([
                torch.zeros(len(tokens), 1, dtype=torch.bool, device=tokens.device),
                ~valid_positions(lengths, tokens.shape[1])], dim=1))
        return encoded[:, 1:], encoded[:, 0]

    def forward(self, state, actions, extension_size):
        if bool(((actions != 0) & (actions != 1)).any()) or extension_size < 1:
            raise ValueError("Expected R=0 or E=1 and a positive extension size")
        old_lengths = state.lengths
        new_lengths = old_lengths + actions * extension_size
        width = int(new_lengths.max().item())
        old = F.pad(state.tokens, (0, 0, 0, max(0, width-state.tokens.shape[1])))[:, :width]
        old_valid = valid_positions(old_lengths, width)
        query = self.extension_query + positions(width, self.dim, old.device)
        source = torch.where(old_valid[:, :, None], old, query[None])

        if self.variant not in ("full", "shared_no_action"):
            raise ValueError(f"Unknown dynamics lesion: {self.variant}")
        tokens = source.new_zeros(source.shape)
        global_state = state.global_state.new_zeros(state.global_state.shape)
        for action_id in (0, 1):
            indices = torch.nonzero(actions == action_id, as_tuple=False).flatten()
            if not len(indices):
                continue
            if self.variant == "shared_no_action":
                action_embedding = source.new_zeros(self.dim)
                network = self.refine_network
            elif action_id == 0:
                action_embedding, network = self.refine_action, self.refine_network
            else:
                action_embedding, network = self.extend_action, self.extend_network
            lengths = old_lengths.index_select(0, indices) if action_id == 0 else new_lengths.index_select(0, indices)
            child_tokens, child_global = self._run(source.index_select(0, indices), lengths,
                state.global_state.index_select(0, indices), state.context.index_select(0, indices),
                action_embedding, network)
            tokens = tokens.index_copy(0, indices, child_tokens)
            global_state = global_state.index_copy(0, indices, child_global)

        previous_mask = F.pad(state.mask_probs, (0, max(0, width-state.mask_probs.shape[1])))[:, :width]
        predicted_mask = torch.sigmoid(self.mask_head(tokens).squeeze(-1))
        position = torch.arange(width, device=old.device)[None]
        extension_region = position >= old_lengths[:, None]
        refine_mask = previous_mask * predicted_mask * valid_positions(old_lengths, width)
        extend_mask = predicted_mask * extension_region * valid_positions(new_lengths, width)
        mask_probs = torch.where(actions[:, None].bool(), extend_mask, refine_mask)
        event_prediction = torch.sigmoid(self.unmask_event_head(tokens).squeeze(-1))
        refine_event = event_prediction * (previous_mask > 0).float() * valid_positions(old_lengths, width)
        extend_event = event_prediction * extension_region * valid_positions(new_lengths, width)
        event_probs = torch.where(actions[:, None].bool(), extend_event, refine_event)
        first_mask = torch.where(mask_probs > 0.5, position, width).min(-1).values
        frontier = torch.minimum(first_mask, new_lengths)
        context = state.context.clone()
        context[:, 1] = new_lengths.float() / 64
        context[:, 2] = torch.where(actions.bool(), old_lengths.float() / 64, context[:, 2])
        context[:, 3] = torch.where(actions.bool(), 0.0, context[:, 3] + 1 / 3)
        return NativeStructuredLatent(tokens, global_state, new_lengths, context,
                                      mask_probs, frontier, event_probs)


class NativeStructuredWorldModel(nn.Module):
    """Acceptance + H1-H3 R/E latent dynamics using real Fast-dLLM observations."""

    schema = "native_structured_acceptance_world_model_v1"

    def __init__(self, hidden_dim, token_dim, top_k=32, dim=256,
                 num_hidden_layers=3, dropout=0.1):
        super().__init__()
        if dim % 8:
            raise ValueError("Native structured latent width must be divisible by 8")
        self.config = dict(hidden_dim=hidden_dim, token_dim=token_dim, top_k=top_k,
            dim=dim, num_hidden_layers=num_hidden_layers, dropout=dropout,
            architecture="native_structured")
        self.encoder = NativeStructuredEncoder(hidden_dim, token_dim, top_k, dim,
                                               num_hidden_layers, dropout)
        self.dynamics = NativeActionDynamics(dim, dropout)
        self.accept_head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim),
                                         nn.GELU(), nn.Linear(dim, 1))

    def acceptance(self, state):
        global_rows = state.global_state[:, None].expand(-1, state.tokens.shape[1], -1)
        return self.accept_head(torch.cat([state.tokens, global_rows], -1)).squeeze(-1)

    def transition(self, state, actions, extension_size):
        return self.dynamics(state, actions, extension_size)

    @staticmethod
    def region_mean(values, region):
        counts = region.sum(-1)
        per_row = (values * region).sum(-1) / counts.clamp_min(1)
        return (per_row * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)

    def structural_loss(self, predicted_state, future_batch):
        valid = valid_positions(future_batch["lengths"], predicted_state.mask_probs.shape[1])
        mask_target = future_batch["scalars"][:, :, 0]
        event_target = future_batch["scalars"][:, :, 2]
        mask_loss = F.binary_cross_entropy(predicted_state.mask_probs.clamp(1e-6, 1-1e-6),
                                           mask_target, reduction="none")
        event_loss = F.binary_cross_entropy(predicted_state.event_probs.clamp(1e-6, 1-1e-6),
                                            event_target, reduction="none")
        return self.region_mean(mask_loss + 0.5 * event_loss, valid)
