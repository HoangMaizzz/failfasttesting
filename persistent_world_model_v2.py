"""Behavior-first, fixed-coordinate world model. Current verifier is target-only.

Reuses V1 hazard utilities and the native Observation schema; the drafter input
allowlist is deliberately small. Shadow verifier rows never update runtime state.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from world_model_core import valid_positions, acceptance_nll, prefix_log_distribution
from persistent_world_model_v1 import expected_yield, grouped_distribution_kl

VARIANTS = ('full', 'drafter_only', 'outcome_only', 'no_verifier_logits',
            'no_verifier_hidden', 'no_drafter_hidden')


def semantic_verifier(obs):
    """40 interpretable channels, observed prefix through first reject only."""
    out = torch.zeros(40)
    if obs.accepted is None or obs.teacher_features is None:
        return out
    y, length = int(obs.accepted), obs.length
    clean = min(length, y + int(y < length))
    f = obs.teacher_features[:clean].float()
    if not len(f):
        return out
    margin = obs.teacher_margin if obs.teacher_margin is not None else torch.zeros(length)
    out[:4] = torch.tensor([y / length, min(y, length) / length,
                           float(f[:, 1].mean()), float((-margin[:clean]).clamp_min(0).mean()) / 10])
    out[4:36] = f[-1, 3:35]  # projected hidden at preceding causal position
    out[36:] = torch.tensor([length / 64, float(y < length), float(obs.context[3]),
                            float(obs.scalars[:, 0].mean())])
    return out


def pack_v2(observations, prior_memory, device):
    """No frozen vocab embeddings/top-K summaries are allocated for WM training."""
    width = max(o.length for o in observations)
    def pad(values, shape):
        out = torch.zeros(len(values), width, *shape)
        for i, x in enumerate(values):
            out[i, :len(x)] = x.float()
        return out.to(device)
    memory = [prior_memory.get(o.uid, torch.empty(0, 40)) for o in observations]
    memory_width = max(1, max(map(len, memory)))
    history = torch.stack([F.pad(x, (0, 0, 0, memory_width-len(x))) for x in memory]).to(device)
    hshape = observations[0].hidden.shape[1:]
    structure = torch.tensor([[o.length/64, float(o.context[3]),
        float(o.scalars[:, 0].mean()), float(o.context[2]),
        float(o.context[2]), float(o.scalars[0, 10])] for o in observations], device=device)
    features = [o.teacher_features if o.teacher_features is not None
                else torch.zeros(o.length, 35) for o in observations]
    margins = [o.teacher_margin if o.teacher_margin is not None
               else torch.zeros(o.length) for o in observations]
    teacher_valid = torch.tensor([o.teacher_features is not None and o.accepted is not None
                                 for o in observations], device=device)
    return dict(hidden=pad([o.hidden for o in observations], hshape),
        mask=pad([o.scalars[:, 0] for o in observations], ()),
        hidden_valid=pad([o.scalars[:, 3] for o in observations], ()),
        lengths=torch.tensor([o.length for o in observations], device=device),
        labels=torch.tensor([-1 if o.accepted is None else o.accepted for o in observations], device=device),
        structure=structure, prior_verifier=history,
        prior_lengths=torch.tensor(list(map(len, memory)), device=device),
        current_verifier=torch.stack([semantic_verifier(o) for o in observations]).to(device),
        teacher_features=pad(features, (35,)), teacher_margin=pad(margins, ()),
        teacher_present=teacher_valid,
        teacher_actual=torch.tensor([o.teacher_is_actual for o in observations], device=device))


@dataclass
class StateV2:
    z: torch.Tensor
    structure: torch.Tensor

    @property
    def lengths(self):
        return (self.structure[:, 0]*64).round().long()

    def take(self, indices):
        return StateV2(self.z.index_select(0, indices), self.structure.index_select(0, indices))


class BehavioralWorldModelV2(nn.Module):
    schema = 'persistent_behavioral_world_model_v2'

    def __init__(self, hidden_dim, hidden_layers=3, variant='full', dropout=.05):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.config = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                           variant=variant, dropout=dropout)
        self.variant = variant
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim*hidden_layers, 64)
        self.token_structure = nn.Linear(4, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout,
            batch_first=True, norm_first=True, activation='gelu')
        self.d_encoder = nn.TransformerEncoder(layer, 1, nn.LayerNorm(64),
                                               enable_nested_tensor=False)
        self.d_summary = nn.Sequential(nn.Linear(70, 64), nn.GELU(), nn.LayerNorm(64))
        self.v_projection = nn.Sequential(nn.Linear(40, 64), nn.GELU(), nn.LayerNorm(64))
        self.v_memory = nn.GRU(64, 64, batch_first=True)
        self.pre_fuse = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))
        self.grounding = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, 128))
        # The post teacher remains in the same coordinate system as pre.
        nn.init.zeros_(self.grounding[-1].weight); nn.init.zeros_(self.grounding[-1].bias)
        self.readout = nn.Sequential(nn.Linear(146, 128), nn.GELU(), nn.Linear(128, 3))

    def _v_features(self, x):
        x = x.clone()
        if self.variant == 'outcome_only':
            x[..., 2:36] = 0
        if self.variant == 'no_verifier_logits':
            x[..., 2:4] = 0
        if self.variant == 'no_verifier_hidden':
            x[..., 4:36] = 0
        return x

    def pre(self, b):
        # Explicit allowlist: current_verifier, labels, teacher_* are NEVER read.
        hidden = b['hidden'].float()
        if self.variant == 'no_drafter_hidden':
            hidden = torch.zeros_like(hidden)
        hidden = self.hidden_projection(self.hidden_norm(hidden).flatten(2))
        hidden *= b['hidden_valid'][..., None]
        valid = valid_positions(b['lengths'], hidden.shape[1])
        pos = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(len(hidden), -1)
        relative = pos / b['lengths'].clamp_min(1)[:, None]
        active = relative >= b['structure'][:, 3, None]*64 / b['lengths'][:, None]
        meta = torch.stack([relative, b['mask'], active.float(), b['hidden_valid']], -1)
        tokens = self.d_encoder(hidden + self.token_structure(meta), src_key_padding_mask=~valid)
        pooled = (tokens*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
        d = self.d_summary(torch.cat([pooled, b['structure']], -1))
        v = torch.zeros_like(d)
        if self.variant != 'drafter_only':
            seq, _ = self.v_memory(self.v_projection(self._v_features(b['prior_verifier'])))
            idx = (b['prior_lengths']-1).clamp_min(0)
            v = seq[torch.arange(len(d), device=d.device), idx] * (b['prior_lengths']>0)[:, None]
        return StateV2(self.pre_fuse(torch.cat([d, v], -1)), b['structure'])

    def post(self, pre, b):
        if self.variant == 'drafter_only':
            return pre
        v = self.v_projection(self._v_features(b['current_verifier']))
        delta = self.grounding(torch.cat([pre.z, v], -1))
        return StateV2(pre.z + delta*b['teacher_present'][:, None], pre.structure)

    def heads(self, state, width=None):
        width = int(width or state.lengths.max())
        pos = (torch.arange(width, device=state.z.device)[None]+1) / state.lengths.clamp_min(1)[:, None]
        freq = torch.arange(1, 9, device=state.z.device).float()
        angle = pos[..., None]*freq*3.141592653589793
        length = state.lengths[:, None, None].expand(-1, width, 1).float()/64
        x = torch.cat([state.z[:, None].expand(-1, width, -1),
                       angle.sin(), angle.cos(), pos[..., None], length], -1)
        out = self.readout(x)
        return dict(hazard=out[..., 0], probability=out[..., 1].sigmoid(),
                    gap=F.softplus(out[..., 2])*5)

    def representation_loss(self, b):
        pre = self.pre(b); post = self.post(pre, b)
        hp, hpost = self.heads(pre, b['hidden'].shape[1]), self.heads(post, b['hidden'].shape[1])
        main, metrics = behavioral_loss(hp, b)
        grounded, _ = behavioral_loss(hpost, b)
        loss = main + .25*grounded + .001*(post.z-pre.z).square().mean()
        return loss, {**metrics, 'posterior_behavior': float(grounded.detach())}

    def freeze_representation(self):
        self.eval().requires_grad_(False)


def action_descriptor(state, actions, extend_size=8):
    """Source-only metadata: future mask count/hidden/verifier are unavailable."""
    actions = actions.to(state.z.device).long()
    if bool(((actions<0)|(actions>1)).any()):
        raise ValueError('R=0 or E=1 expected')
    old = state.structure
    new = old.clone()
    new[:, 0] += actions*extend_size/64
    if bool((new[:, 0]>1.00001).any()):
        raise ValueError('Proposal exceeds 64 tokens')
    new[:, 1] = torch.where(actions.bool(), torch.zeros_like(old[:, 1]), old[:, 1]+1/3)
    new[:, 3] = torch.where(actions.bool(), old[:, 0], old[:, 3])
    new[:, 4] += actions/8
    descriptor = torch.stack([old[:, 0], new[:, 0], actions*extend_size/64,
        new[:, 3], new[:, 4], old[:, 1], old[:, 2], actions.float()], -1)
    return descriptor, new


class FixedDynamicsV2(nn.Module):
    def __init__(self, generic=False):
        super().__init__()
        self.generic = generic
        self.maps = nn.ModuleList([nn.Sequential(nn.Linear(136, 256), nn.GELU(),
                                                 nn.Linear(256, 128)) for _ in range(2)])
        if not generic:
            nn.init.zeros_(self.maps[0][-1].weight); nn.init.zeros_(self.maps[0][-1].bias)
        self.mask_head = nn.Sequential(nn.Linear(136, 128), nn.GELU(), nn.Linear(128, 1))
        self.effect_head = nn.Sequential(nn.Linear(136, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, state, actions, extend_size=8):
        desc, structure = action_descriptor(state, actions, extend_size)
        x = torch.cat([state.z, desc], -1)
        deltas = torch.stack([m(x) for m in self.maps], 1)
        selected = deltas[torch.arange(len(x), device=x.device), actions.long()]
        z = selected if self.generic else state.z+selected
        structure[:, 2] = self.mask_head(x).sigmoid().squeeze(-1)
        return StateV2(z, structure), self.effect_head(x).squeeze(-1)*8, selected


class DirectOutcomeV2(nn.Module):
    """Predict endpoint hazards directly from z and up to three R/E descriptors."""
    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(nn.Linear(128+24, 256), nn.GELU(), nn.Linear(256, 128))
        self.readout = nn.Sequential(nn.Linear(146, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, source, actions):
        structure = source
        desc = []
        for action in actions:
            d, s = action_descriptor(structure, action)
            desc.append(d)
            structure = StateV2(source.z, s)
        sequence = torch.stack(desc, 1)
        sequence = F.pad(sequence, (0, 0, 0, 3-len(desc))).flatten(1)
        z = self.map(torch.cat([source.z, sequence], -1))
        length = structure.lengths
        width = int(length.max())
        pos = (torch.arange(width, device=z.device)[None]+1)/length[:, None]
        angle = pos[..., None]*torch.arange(1, 9, device=z.device)*3.141592653589793
        x = torch.cat([z[:, None].expand(-1, width, -1), angle.sin(), angle.cos(),
                       pos[..., None], length[:, None, None].expand(-1, width, 1)/64], -1)
        return self.readout(x).squeeze(-1), length


class GatedFiLMV2(nn.Module):
    def __init__(self, hidden_dim, latent_dim=128):
        super().__init__()
        self.hidden_dim, self.latent_dim = hidden_dim, latent_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(latent_dim, 2*hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim+latent_dim+2, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.projection.weight); nn.init.zeros_(self.projection.bias)

    def forward(self, hidden, z, positions, newly_extended, active):
        original = hidden.float(); normalized = self.norm(original)
        if z.ndim == 1:
            z = z[None]
        z = z.to(hidden.device).float()
        zr = z[:, None].expand(-1, hidden.shape[1], -1)
        gate = self.gate(torch.cat([normalized, zr, positions[..., None].float(),
                                   newly_extended[..., None].float()], -1)).sigmoid()
        gate = gate*active[..., None]
        gamma, beta = self.projection(z).chunk(2, -1)
        delta = gate*(gamma[:, None]*normalized+beta[:, None])
        self.regularization = delta.norm(dim=-1)/original.detach().norm(dim=-1).clamp_min(1e-6)
        # Diagnostics are detached; the actual conditioned path retains gradients.
        self.diagnostics = dict(gate=gate.detach(), delta_norm=delta.detach().norm(dim=-1),
            relative_delta=delta.detach().norm(dim=-1)/original.detach().norm(dim=-1).clamp_min(1e-6))
        return (original+delta).to(hidden.dtype)


def teacher_token_ids(ids, logits):
    return ids.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


def behavioral_targets(batch):
    """Observable GT, NOT a learned posterior. Suffix local labels are censored."""
    length, y = batch['lengths'], batch['labels']
    width = batch['teacher_features'].shape[1]
    pos = torch.arange(width, device=length.device)[None]
    valid = valid_positions(length, width) & (y >= 0)[:, None]
    clean = valid & (pos <= y[:, None]) & batch['teacher_present'][:, None]
    return dict(accepted=y, yield_fraction=y.float()/length.clamp_min(1),
        reject_position=torch.minimum(y.clamp_min(0), length),
        # These are actual prefix-emission labels, not local suffix agreement.
        survival=(pos < y[:, None]).float(), valid=valid, clean=clean,
        candidate_probability=batch['teacher_features'][..., 1].detach(),
        gap=(-batch['teacher_margin']).clamp_min(0).detach())


def behavioral_loss(heads, batch):
    target = behavioral_targets(batch)
    hazard = acceptance_nll(heads['hazard'], batch['lengths'], batch['labels'])
    survival = F.logsigmoid(heads['hazard']).cumsum(-1).exp()
    valid, clean = target['valid'], target['clean']
    survival_loss = ((survival-target['survival']).square()*valid).sum()/valid.sum().clamp_min(1)
    probability = ((heads['probability']-target['candidate_probability']).square()*clean).sum()/clean.sum().clamp_min(1)
    gap = (F.smooth_l1_loss(heads['gap']/5, target['gap']/5, reduction='none')*clean).sum()/clean.sum().clamp_min(1)
    loss = hazard + survival_loss + probability + gap
    return loss, dict(hazard=float(hazard.detach()), survival_brier=float(survival_loss.detach()),
                      probability_mse=float(probability.detach()), gap_huber=float(gap.detach()))


def film_acceptance_loss(logits, base, draft_ids, target_ids, accepted_region,
                         first_reject, support_ids, teacher_logits, lse,
                         relative_delta, kl_only=False):
    if kl_only:
        return grouped_distribution_kl(logits, teacher_logits, lse, support_ids,
                                       preserve_logits=base).mean()
    keep = F.cross_entropy(logits.float(), draft_ids, reduction='none')
    fix = F.cross_entropy(logits.float(), target_ids, reduction='none')
    keep_loss = (keep*accepted_region).sum()/accepted_region.sum().clamp_min(1)
    fix_loss = (fix*first_reject).sum()/first_reject.sum().clamp_min(1)
    target_score = logits.gather(-1, target_ids[:, None]).squeeze(-1).float()
    draft_score = logits.gather(-1, draft_ids[:, None]).squeeze(-1).float()
    margin = ((1-(target_score-draft_score)).clamp_min(0)*first_reject).sum()/first_reject.sum().clamp_min(1)
    kl = grouped_distribution_kl(logits, teacher_logits, lse, support_ids).mean()
    return keep_loss + fix_loss + .25*margin + .05*kl + .01*relative_delta.square().mean()
