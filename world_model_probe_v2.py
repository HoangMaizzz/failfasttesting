"""Experimental token candidate attention and residual R/E latent dynamics.

Reuse the native probe's heads, carry rules, and training targets. These are
hypotheses tested by retrained ablations, not asserted performance improvements.
"""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from world_model_core import ActionDynamics, valid_positions
from world_model_probe import ProbeEncoder, ProbeWorldModel


class CandidateEncoder(ProbeEncoder):
    def __init__(self, *args, candidate_attention=False, drop_feature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_attention = candidate_attention
        self.drop_feature = drop_feature
        if candidate_attention:
            self.candidate_query = nn.Linear(32, 32, bias=False)
            self.candidate_key = nn.Linear(32, 32, bias=False)
            self.candidate_gap = nn.Linear(1, 32, bias=False)

    def candidate_features(self, b):
        if not self.candidate_attention:
            return super().candidate_features(b)
        candidates = self.token_projection(b['candidate_vectors'])
        gaps = b['gaps'].clamp(-40, 0)
        keys = self.candidate_key(candidates) + self.candidate_gap(gaps[..., None]/10)
        query = self.candidate_query(self.token_projection(b['token_vectors'][:, :, 1]))
        scores = (keys*query[:, :, None]).sum(-1)/math.sqrt(32) + gaps
        return (scores.softmax(-1)[..., None]*candidates).sum(-2)

    def forward(self, b):
        # Retrained feature removals, never reading labels/teacher targets.
        b = dict(b)
        if self.drop_feature == 'hidden':
            b['hidden'] = torch.zeros_like(b['hidden'])
        elif self.drop_feature == 'gaps':
            b['gaps'] = torch.zeros_like(b['gaps'])
            if 'candidate_vectors' in b:
                b['topk_vectors'] = b['candidate_vectors'].mean(-2)
        elif self.drop_feature == 'prefix_history':
            b['prefix_vectors'] = torch.zeros_like(b['prefix_vectors'])
            b['history'] = torch.zeros_like(b['history'])
        return super().forward(b)


class ResidualDynamics(ActionDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Parameter(torch.full((2,), -2.2))
        self.mask_condition = nn.Linear(2, self.dim, bias=False)
        nn.init.zeros_(self.mask_condition.weight)

    def forward(self, state, actions, extension_size):
        pos = torch.arange(state.tokens.shape[1], device=actions.device)[None]
        active = (pos >= state.context[:, 2, None]*64).float()
        condition = self.mask_condition(torch.stack([state.mask_probs, active], -1))
        prediction = super().forward(replace(state, tokens=state.tokens+condition), actions, extension_size)
        width = prediction.tokens.shape[1]
        old = F.pad(state.tokens, (0, 0, 0, max(0, width-state.tokens.shape[1])))[:, :width]
        old_region = valid_positions(state.lengths, width)
        gate = self.gate[actions].sigmoid()
        prediction.tokens = torch.where(old_region[..., None],
            old + gate[:, None, None]*(prediction.tokens-old), prediction.tokens)
        prediction.global_state = state.global_state + gate[:, None]*(prediction.global_state-state.global_state)
        return prediction


class ImprovedProbeWorldModel(ProbeWorldModel):
    schema = 'acceptance_probe_v2_experiment'

    def __init__(self, *args, candidate_attention=True, residual_dynamics=True, drop_feature=None, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.config
        self.encoder = CandidateEncoder(c['hidden_dim'], c['token_dim'], c['top_k'], c['dim'],
            c['num_hidden_layers'], c['dropout'], candidate_attention=candidate_attention, drop_feature=drop_feature)
        if residual_dynamics:
            self.dynamics = ResidualDynamics(c['dim'], 2, c['dropout'])
        self.needs_candidates = candidate_attention or drop_feature == 'gaps'
        self.config = dict(c, candidate_attention=candidate_attention,
                          residual_dynamics=residual_dynamics, drop_feature=drop_feature)


def experiment_variants():
    base = dict(candidate_attention=False, residual_dynamics=False, delta_weight=0.,
                teacher_weight=0., latent_weight=.1, structure_weight=.1, drop_feature=None)
    full = dict(base, candidate_attention=True, residual_dynamics=True, delta_weight=.1, teacher_weight=.1)
    return {
        'baseline': base,
        'teacher_only': dict(base, teacher_weight=.1),
        'attention_only': dict(base, candidate_attention=True),
        'residual_only': dict(base, residual_dynamics=True),
        'delta_only': dict(base, delta_weight=.1),
        'improved': full,
        # Training-only replay sampling arms. They reuse the exact same model;
        # accepted-token labels are used only to stratify real root edges.
        'action_balanced': dict(full, sampling_mode='action_balanced'),
        'change_balanced': dict(full, sampling_mode='change_balanced'),
        'change_balanced_delta': dict(full, sampling_mode='change_balanced', delta_weight=.3),
        'no_attention': dict(full, candidate_attention=False),
        'no_residual': dict(full, residual_dynamics=False),
        'no_delta': dict(full, delta_weight=0.),
        'no_teacher': dict(full, teacher_weight=0.),
        'no_latent_loss': dict(full, latent_weight=0.),
        'no_structure_loss': dict(full, structure_weight=0.),
        'no_hidden': dict(full, drop_feature='hidden'),
        'no_gaps': dict(full, drop_feature='gaps'),
        'no_prefix_history': dict(full, drop_feature='prefix_history'),
    }
