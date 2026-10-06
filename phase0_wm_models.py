"""Tokenwise latent dynamics. No vocabulary decoder and no LLM forward.

The verifier reads only Z plus padding lengths. Native fields remain observation
inputs or auxiliary D targets; verifier teachers never enter encoder/transition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F


def valid_positions(lengths, width=64):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def positions(width, dim, device, start=0):
    p = torch.arange(start, start + width, device=device).float()[:, None]
    f = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000) / dim))
    result = torch.zeros(width, dim, device=device)
    result[:, 0::2] = (p * f).sin()
    result[:, 1::2] = (p * f[:result[:, 1::2].shape[1]]).cos()
    return result


@dataclass
class ObservationBatch:
    hidden: torch.Tensor
    ids: torch.Tensor
    topk_ids: torch.Tensor
    gaps: torch.Tensor
    c: torch.Tensor              # scalars16 + history4
    context: torch.Tensor
    prefix_ids: torch.Tensor
    prefix_lengths: torch.Tensor
    lengths: torch.Tensor

    def to(self, device):
        return ObservationBatch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


@dataclass
class LatentState:
    z: torch.Tensor
    lengths: torch.Tensor
    c: torch.Tensor
    context: torch.Tensor

    def to(self, device):
        return LatentState(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))

    def detach(self):
        return LatentState(*(getattr(self, f).detach() for f in self.__dataclass_fields__))


def slice_latent(state, start, count):
    return LatentState(*(getattr(state, f)[start:start + count] for f in state.__dataclass_fields__))


class TokenEmbedding(nn.Module):
    def __init__(self, weights, vocabulary=None, learned=False):
        super().__init__()
        weights = torch.as_tensor(weights).clone()
        self.table = nn.Embedding.from_pretrained(weights, freeze=not learned, padding_idx=None)
        self.register_buffer('vocabulary', torch.tensor([] if vocabulary is None else vocabulary, dtype=torch.long))
        self.dim = weights.shape[1]
        self.learned = learned

    def forward(self, ids):
        if not self.vocabulary.numel():
            if bool((ids < 0).any()) or bool((ids >= self.table.num_embeddings).any()):
                raise ValueError('Token ID is outside pretrained embedding vocabulary')
            index = ids
        else:
            index = torch.searchsorted(self.vocabulary, ids.contiguous()).clamp_max(len(self.vocabulary) - 1)
            matched = self.vocabulary[index] == ids
            index = torch.where(matched, index + 1, torch.zeros_like(index))
        return self.table(index).float()


class ObservationFusion(nn.Module):
    """All saved native layers, token identities, top32 identities and full prefix."""
    def __init__(self, embedding, dim):
        super().__init__()
        self.embedding = embedding
        self.hidden = nn.ModuleList([nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, dim // 2), nn.GELU()) for _ in range(3)])
        self.rank = nn.Embedding(32, embedding.dim)
        self.topk_score = nn.Sequential(nn.Linear(embedding.dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))
        self.input = nn.Linear(3 * (dim // 2) + 3 * embedding.dim + 20 + 8 + 6 + 32, dim)
        self.prefix_project = nn.Linear(embedding.dim, dim)
        self.prefix_attention = nn.MultiheadAttention(dim, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, b):
        valid = valid_positions(b.lengths)
        raw = b.hidden.float()
        h = torch.cat([layer(raw[:, :, k]) for k, layer in enumerate(self.hidden)], -1)
        h = h * b.c[..., 3:4]
        moments = torch.cat([raw.mean(-1), raw.std(-1, unbiased=False)], -1) * b.c[..., 3:4]
        top = self.embedding(b.topk_ids)
        rank = self.rank(torch.arange(32, device=top.device))[None, None]
        gaps = b.gaps.clamp(-40, 0) / 10
        scores = self.topk_score(torch.cat([top + rank, gaps[..., None]], -1)).squeeze(-1)
        k = (scores.softmax(-1)[..., None] * top).sum(-2) * b.c[..., 4:5]
        native, stop = self.embedding(b.ids[..., 0]), self.embedding(b.ids[..., 1])
        context = b.context[:, None].expand(-1, 64, -1)
        x = self.input(torch.cat([h, native, stop, k, b.c, context, moments, gaps * b.c[..., 4:5]], -1))
        x = x + positions(64, x.shape[-1], x.device)[None]
        # Full ordered prefix is supplied; attention is O(proposal_length*prefix).
        p = self.prefix_project(self.embedding(b.prefix_ids))
        p = p + positions(p.shape[1], p.shape[2], p.device)[None]
        pv = valid_positions(b.prefix_lengths.clamp_min(1), p.shape[1])
        p = p * (b.prefix_lengths > 0)[:, None, None]
        attended = self.prefix_attention(x, p, p, key_padding_mask=~pv, need_weights=False)[0]
        return self.norm(x + attended) * valid[..., None]


def transformer(dim, layers, dropout):
    block = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout,
        activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, nn.LayerNorm(dim), enable_nested_tensor=False)


class LatentEncoder(nn.Module):
    def __init__(self, embedding, dim=128, layers=2, dropout=.05):
        super().__init__()
        self.fusion = ObservationFusion(embedding, dim)
        self.network = transformer(dim, layers, dropout)
        self.dim = dim

    def forward(self, b):
        valid = valid_positions(b.lengths)
        z = self.network(self.fusion(b), src_key_padding_mask=~valid)
        return LatentState(z * valid[..., None], b.lengths, b.c, b.context)


class VerifierReadout(nn.Module):
    """Reads only token latent; context/teacher/IDs are deliberately inaccessible."""
    def __init__(self, dim, layers=1, dropout=.05):
        super().__init__()
        self.network = transformer(dim, layers, dropout)
        self.output = nn.Linear(dim, 4)

    def forward(self, z, lengths):
        valid = valid_positions(lengths)
        causal = torch.triu(torch.ones(64, 64, dtype=torch.bool, device=z.device), 1)
        h = self.network(z, mask=causal, src_key_padding_mask=~valid)
        out = self.output(h)
        return dict(hazard=out[..., 0], tf=out[..., 1],
                    probability=out[..., 2].sigmoid(), margin=out[..., 3].tanh())


class RawObservationVerifier(nn.Module):
    """Same native information, larger direct readout, no frozen latent bottleneck."""
    def __init__(self, embedding, dim=192, dropout=.05):
        super().__init__()
        self.fusion = ObservationFusion(embedding, dim)
        self.readout = VerifierReadout(dim, layers=1, dropout=dropout)

    def forward(self, batch):
        return self.readout(self.fusion(batch), batch.lengths)


class NativeReconstruction(nn.Module):
    def __init__(self, dim, target_width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, target_width))

    def forward(self, state):
        return self.network(state.z)


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def verifier_loss(heads, rows, device):
    """Conditional hazard is censored after first mismatch; TF suffix is separate."""
    n = torch.tensor([r['length'] for r in rows], device=device)
    labels = torch.tensor([-1 if r['accepted'] is None else r['accepted'] for r in rows], device=device)
    pos = torch.arange(64, device=device)[None]
    valid = valid_positions(n)
    observed = valid & (labels >= 0)[:, None]
    risk = observed & (pos <= labels[:, None])
    yy = (pos < labels[:, None]).float()
    bce = F.binary_cross_entropy_with_logits(heads['hazard'], yy, reduction='none')
    loss = masked_mean(bce, risk)
    survival = F.logsigmoid(heads['hazard']).cumsum(-1).exp()
    loss = loss + .5 * masked_mean((survival - yy).square(), observed)
    target = torch.zeros(len(rows), 64, 3, device=device)
    tv = torch.zeros_like(valid)
    for i, r in enumerate(rows):
        if r['teacher'] is not None:
            target[i, :r['length']] = r['teacher'][:, :3].to(device)
            tv[i, :r['length']] = True
    if tv.any():
        loss = loss + .3 * masked_mean(F.binary_cross_entropy_with_logits(heads['tf'], target[..., 2], reduction='none'), tv)
        loss = loss + .15 * masked_mean((heads['probability'] - target[..., 1]).square(), tv)
        loss = loss + .1 * masked_mean(F.smooth_l1_loss(heads['margin'], target[..., 0], reduction='none'), tv)
    return loss


def native_loss(prediction, target, mask):
    return masked_mean(F.smooth_l1_loss(prediction, target, reduction='none'), mask)


def advance_structure(state, action):
    """Only deterministic action bookkeeping; missing native outcomes are predicted."""
    if action not in ('R', 'E'):
        raise ValueError('Only R/E have refinement transitions')
    c = state.c.clone()
    context = state.context.clone()
    lengths = state.lengths + (8 if action == 'E' else 0)
    if bool((lengths > 64).any()):
        raise ValueError('E exceeds max64')
    pos = torch.arange(64, device=c.device)[None]
    old = valid_positions(state.lengths)
    valid = valid_positions(lengths)
    c[..., 5:7] = torch.where(old[..., None], c[..., 5:7] + .125, c[..., 5:7])
    if action == 'E':
        c[..., 0] = torch.where(old, torch.zeros_like(c[..., 0]), c[..., 0])
        c[..., 1] = torch.where(old, torch.ones_like(c[..., 1]), c[..., 1])
        c[..., 14] = torch.where(old, torch.zeros_like(c[..., 14]), c[..., 14])
        context[:, 2] = state.lengths / 64
        context[:, 3] = 0
    else:
        context[:, 3] = context[:, 3] + 1 / 3
    context[:, 1] = lengths / 64
    active = pos >= (context[:, 2:3] * 64).round()
    c[..., 9] = pos / lengths[:, None]
    absolute = pos + context[:, 0:1] * 1024
    physical = (context[:, 5:6] * 64).clamp_min(1)
    small = (context[:, 6:7] * 64).clamp_min(1)
    c[..., 10] = absolute.remainder(physical) / physical
    c[..., 11] = absolute.remainder(small) / small
    c[..., 12] = active.float()
    c[..., 13] = 1
    c = c * valid[..., None]
    return LatentState(state.z, lengths, c, context), active & valid


class ActionDynamics(nn.Module):
    """An independent learned model for one native R or E action."""
    def __init__(self, dim, action, layers=2, dropout=.05):
        super().__init__()
        self.action = action
        self.condition = nn.Linear(28, dim)
        self.materialize = nn.Sequential(nn.Linear(dim + 20, dim), nn.GELU(), nn.Linear(dim, dim)) if action == 'E' else None
        self.new_queries = nn.Parameter(torch.randn(8, dim) * .02) if action == 'E' else None
        self.network = transformer(dim, layers, dropout)
        self.delta = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)
        self.native = nn.Linear(dim, 9)  # mask, validH/L, ageH/L, confidence, fresh-unmask, histories
        self.history = nn.Linear(dim, 4)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.gate.bias, -1)

    def forward(self, state):
        base, active = advance_structure(state, self.action)
        z = state.z.clone()
        if self.action == 'E':
            z = z + self.materialize(torch.cat([state.z, state.c], -1)) * valid_positions(state.lengths)[..., None]
            for i in range(len(z)):
                start = int(state.lengths[i])
                z[i, start:start + 8] = self.new_queries + positions(8, z.shape[-1], z.device, start)
        condition = torch.cat([base.c, base.context[:, None].expand(-1, 64, -1)], -1)
        valid = valid_positions(base.lengths)
        x = z + self.condition(condition) + positions(64, z.shape[-1], z.device)[None]
        h = self.network(x, src_key_padding_mask=~valid)
        predicted_z = (z + self.gate(h).sigmoid() * self.delta(h)) * valid[..., None]
        p = self.native(h)
        c = base.c.clone()
        mask_probability = p[..., 0].sigmoid()
        if self.action == 'R':
            mask_probability = mask_probability * state.c[..., 0]
        c[..., 0] = torch.where(active, mask_probability, c[..., 0])
        c[..., 1] = 1 - c[..., 0]
        c[..., 2] = torch.where(active, p[..., 6].sigmoid(), torch.zeros_like(c[..., 2]))
        c[..., 3] = torch.where(active, p[..., 1].sigmoid(), c[..., 3])
        c[..., 4] = torch.where(active, p[..., 2].sigmoid(), c[..., 4])
        c[..., 5] = torch.where(active, F.softplus(p[..., 3]) / 8, c[..., 5])
        c[..., 6] = torch.where(active, F.softplus(p[..., 4]) / 8, c[..., 6])
        c[..., 7] = torch.where(active, p[..., 5].sigmoid(), c[..., 7])
        c[..., 8] = torch.where(active, p[..., 7].sigmoid(), c[..., 8])
        c[..., 14] = torch.where(active, p[..., 8].sigmoid(), c[..., 14])
        c[..., 16:20] = self.history(h)
        return LatentState(predicted_z, base.lengths, c * valid[..., None], base.context)


class DynamicsPair(nn.Module):
    def __init__(self, dim, layers=2, dropout=.05):
        super().__init__()
        self.refine = ActionDynamics(dim, 'R', layers, dropout)
        self.extend = ActionDynamics(dim, 'E', layers, dropout)

    def forward(self, state, actions):
        # Split heterogeneous action batches; both action models are independent.
        out = []
        for action in ('R', 'E'):
            indices = torch.where(actions == int(action == 'E'))[0]
            if not len(indices):
                continue
            part = LatentState(*(getattr(state, f)[indices] for f in state.__dataclass_fields__))
            pred = (self.refine if action == 'R' else self.extend)(part)
            out.append((indices, pred))
        fields = {}
        for f in state.__dataclass_fields__:
            template = getattr(out[0][1], f)
            value = template.new_zeros((len(actions), *template.shape[1:]))
            for idx, pred in out:
                value = value.index_copy(0, idx, getattr(pred, f))
            fields[f] = value
        return LatentState(**fields)


class PriorDynamics(nn.Module):
    def __init__(self, new_token_latent):
        super().__init__()
        self.register_buffer('new_token_latent', new_token_latent.clone())

    def forward(self, state, actions):
        parts = []
        for i, action in enumerate(actions.tolist()):
            s = slice_latent(state, i, 1)
            base, _ = advance_structure(s, 'E' if action else 'R')
            z = s.z.clone()
            if action:
                start = int(s.lengths[0]); z[:, start:start + 8] = self.new_token_latent
            parts.append(replace(base, z=z * valid_positions(base.lengths)[..., None]))
        return LatentState(*(torch.cat([getattr(s, f) for s in parts]) for f in state.__dataclass_fields__))


def rollout(model, state, sequences):
    if not sequences:
        return state
    if len({len(seq) for seq in sequences}) != 1:
        raise ValueError('A rollout batch must have one horizon')
    for step in range(len(sequences[0])):
        actions = torch.tensor([s[step] for s in sequences], device=state.z.device)
        state = model(state, actions)
    return state


class DirectOutcome(nn.Module):
    """Source Z plus complete action sequence -> future V heads, no latent rollout."""
    def __init__(self, dim, dropout=.05):
        super().__init__()
        self.actions = nn.Embedding(3, dim)
        self.sequence = nn.GRU(dim, dim, batch_first=True)
        self.input = nn.Linear(dim + 28, dim)
        self.queries = nn.Parameter(torch.randn(64, dim) * .02)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.readout = VerifierReadout(dim, 1, dropout)

    def forward(self, source, sequences):
        seq = torch.as_tensor(sequences, device=source.z.device, dtype=torch.long)
        target_length = source.lengths + (seq == 1).sum(-1) * 8
        if bool((target_length > 64).any()):
            raise ValueError('Direct action sequence exceeds max64')
        action_inputs = self.actions(seq) + positions(seq.shape[1], source.z.shape[-1], source.z.device)[None]
        _, code = self.sequence(action_inputs)
        code = code[-1]
        x = self.input(torch.cat([source.z, source.c, source.context[:, None].expand(-1, 64, -1)], -1))
        queries = self.queries[None].expand(len(x), -1, -1) + code[:, None]
        h = self.attention(queries, x, x, key_padding_mask=~valid_positions(source.lengths), need_weights=False)[0]
        return self.readout(h + queries, target_length)


def latent_loss(pred, truth, recon=None, native_target=None, native_mask=None):
    """D-only targets. Frozen G is deliberately never called by this loss."""
    valid = valid_positions(truth.lengths)
    if not torch.equal(pred.lengths, truth.lengths):
        raise ValueError('Action length disagrees with native child')
    # Frozen encoder LayerNorm defines the coordinate scale. Balance the native
    # active segment against the older prefix so many unchanged prefix positions
    # cannot swamp the eight new E positions. These are D-target weights only;
    # no future boundary/mask is passed into the imagined transition.
    active = valid & (torch.arange(64, device=valid.device)[None] >=
                      (truth.context[:, 2] * 64).round().long()[:, None])
    older = valid & ~active
    def balanced(value):
        active_mean = (value * active).sum(-1) / active.sum(-1).clamp_min(1)
        older_mean = (value * older).sum(-1) / older.sum(-1).clamp_min(1)
        count = active.any(-1).float() + older.any(-1).float()
        return ((active_mean + older_mean) / count.clamp_min(1)).mean()
    distance = balanced(F.smooth_l1_loss(pred.z, truth.z, reduction='none').mean(-1))
    cosine = balanced(1 - F.cosine_similarity(pred.z, truth.z, dim=-1))
    c_loss = balanced(F.smooth_l1_loss(pred.c, truth.c, reduction='none').mean(-1))
    loss = distance + .2 * cosine + .2 * c_loss
    if recon is not None:
        loss = loss + .1 * native_loss(recon(pred), native_target, native_mask)
    return loss
