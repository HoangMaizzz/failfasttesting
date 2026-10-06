"""Acceptance-oriented verifier teachers and capacity-matched D-to-V students.

Projected Qwen hidden is a training teacher. The deployment-facing forward
accepts only D latent, native bookkeeping, lengths and known context.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from phase0_wm_models import positions, transformer, valid_positions


ORACLE_METHODS = ('V_hidden_only', 'V_joint_probe', 'V_joint')
STUDENT_METHODS = ('Direct', 'Bridge_state', 'Bridge_behavior', 'Direct_distill')


class AcceptanceHead(nn.Module):
    def __init__(self, dim=128, dropout=.05):
        super().__init__()
        self.network = transformer(dim, 1, dropout)
        self.output = nn.Linear(dim, 1)

    def forward(self, z, lengths):
        width = z.shape[1]
        mask = torch.triu(torch.ones(width, width, device=z.device, dtype=torch.bool), 1)
        h = self.network(z, mask=mask, src_key_padding_mask=~valid_positions(lengths, width))
        return self.output(h).squeeze(-1)


class NativeVerifier(nn.Module):
    """Native ceiling using projected causal Qwen hidden and, optionally, d_i.

    The source has already projected full final Qwen hidden to 32 dimensions.
    A 128D learned latent here is NOT a 128D compression test of full Qwen.
    """
    def __init__(self, method, dim=128, hidden_dim=32, embedding_dim=64, dropout=.05):
        super().__init__()
        if method not in ORACLE_METHODS:
            raise ValueError('Unknown native verifier method')
        self.method = method
        # Identical head initialization across different teacher input widths.
        self.readout = AcceptanceHead(dim, dropout)
        width = hidden_dim + (embedding_dim if method != 'V_hidden_only' else 0)
        self.encoder = (nn.Sequential(nn.Linear(width, dim * 2), nn.GELU(),
                                     nn.Linear(dim * 2, dim), nn.LayerNorm(dim))
                        if method == 'V_joint' else
                        nn.Sequential(nn.Linear(width, dim), nn.LayerNorm(dim)))

    def encode(self, teacher_hidden, candidate_embedding, lengths):
        if self.method == 'V_hidden_only':
            x = teacher_hidden
        else:
            x = torch.cat([teacher_hidden, candidate_embedding], -1)
        valid = valid_positions(lengths, x.shape[1])
        z = self.encoder(x) + positions(x.shape[1], self.readout.output.in_features, x.device)[None]
        return z * valid[..., None]

    def forward(self, teacher_hidden, candidate_embedding, lengths):
        z = self.encode(teacher_hidden, candidate_embedding, lengths)
        return dict(hazard=self.readout(z, lengths), z_V=z)


class PairedStudent(nn.Module):
    """All four students have EXACTLY the same inference architecture.

    Bridge heads are frozen copies of G_V. Direct heads can adapt to the D
    representation. Training parameter counts differ and are reported.
    Teacher hidden, token teachers and parent K never enter this forward.
    """
    def __init__(self, method, dim=128, layers=2, dropout=.05, native_head=None):
        super().__init__()
        if method not in STUDENT_METHODS:
            raise ValueError('Unknown student method')
        self.method = method
        self.readout = AcceptanceHead(dim, dropout)
        self.input = nn.Sequential(nn.Linear(dim + 28, dim), nn.LayerNorm(dim))
        self.network = transformer(dim, layers, dropout)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        if native_head is not None:
            self.readout.load_state_dict(copy.deepcopy(native_head.state_dict()))
        if method.startswith('Bridge'):
            self.readout.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.method.startswith('Bridge'):
            self.readout.eval()
        return self

    def forward(self, z_D, c, context, lengths):
        if z_D.shape[-1] != 128 or c.shape[-1] != 20 or context.shape[-1] != 8:
            raise ValueError('Requires frozen 128D D latent and native 20+8 bookkeeping')
        width = z_D.shape[1]
        valid = valid_positions(lengths, width)
        known = torch.cat([z_D, c, context[:, None].expand(-1, width, -1)], -1)
        x = self.input(known) + positions(width, z_D.shape[-1], z_D.device)[None]
        h = self.network(x, src_key_padding_mask=~valid)
        z = self.output(h) * valid[..., None]
        return dict(hazard=self.readout(z, lengths), z_V=z)


def outputs(hazard, lengths):
    valid = valid_positions(lengths, hazard.shape[1])
    q = F.logsigmoid(hazard).cumsum(-1).exp() * valid
    return dict(q=q, hazards=hazard.sigmoid(), K=q.sum(-1))


def acceptance_loss(hazard, lengths, accepted):
    """Censored hazard NLL plus cumulative survival Brier and expected-K Huber."""
    valid = valid_positions(lengths, hazard.shape[1]) & (accepted >= 0)[:, None]
    pos = torch.arange(hazard.shape[1], device=hazard.device)[None]
    risk = valid & (pos <= accepted[:, None])
    truth = (pos < accepted[:, None]).float()
    prediction = outputs(hazard, lengths)
    zero = hazard.sum() * 0
    def mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1)
    nll = mean(F.binary_cross_entropy_with_logits(hazard, truth, reduction='none'), risk)
    brier = mean((prediction['q'] - truth).square(), valid)
    observed = accepted >= 0
    kloss = (F.smooth_l1_loss(prediction['K'][observed], accepted[observed].float())
             if observed.any() else zero)
    return nll + .5 * brier + .05 * kloss, dict(nll=nll, brier=brier, K_huber=kloss)


def objective(method, prediction, batch, target_z, cfg):
    behavioral, parts = acceptance_loss(prediction['hazard'], batch['lengths'], batch['accepted'])
    zero = prediction['z_V'].sum() * 0
    latent = zero
    if target_z is not None:
        valid = valid_positions(batch['lengths']) & batch['teacher_valid']
        distance = F.smooth_l1_loss(prediction['z_V'], target_z.detach(), reduction='none').mean(-1)
        latent = (distance * valid).sum() / valid.sum().clamp_min(1)
    if method in ORACLE_METHODS or method == 'Direct':
        loss = behavioral
    elif method == 'Bridge_state':
        loss = cfg['bridge_state_weight'] * latent
    elif method == 'Bridge_behavior':
        loss = cfg['bridge_state_weight'] * latent + cfg['bridge_behavior_weight'] * behavioral
    elif method == 'Direct_distill':
        loss = behavioral + cfg['distill_weight'] * latent
    else:
        raise ValueError('Unknown method')
    parts.update(latent=latent, behavioral=behavioral)
    return loss, parts


def parameter_counts(model):
    return dict(inference=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
