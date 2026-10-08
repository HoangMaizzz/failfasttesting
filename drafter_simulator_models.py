"""Frozen observation encoder and small native-refinement simulators.

No vocabulary generator or verifier is used here. Each latent row represents
one position of the physical canvas, including positions outside the active
eight-token span. Diagnostic targets describe the next *real forward*.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(),
                                    nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim))
        self.reconstruct = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        return self.encode(x)


class BehaviorHeads(nn.Module):
    def __init__(self, dim, content_dim=64):
        super().__init__()
        self.decode = nn.Sequential(nn.Linear(dim, 128), nn.GELU(),
                                    nn.Linear(128, 4 + content_dim))
        self.change = nn.Sequential(nn.Linear(2 * dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z, parent=None):
        y = self.decode(z)
        result = dict(mask_logits=y[..., 0], confidence=y[..., 1].sigmoid(),
                      entropy=y[..., 2].sigmoid(), margin=y[..., 3].sigmoid(),
                      content=y[..., 4:])
        if parent is not None:
            result['stability_logits'] = self.change(torch.cat([parent, z], -1)).squeeze(-1)
        return result


class Transition(nn.Module):
    def __init__(self, dim=128, kind='transformer', layers=2, gated=False):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.net = nn.Linear(dim * 2, dim)
        elif kind == 'mlp':
            self.net = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, dim))
        elif kind == 'transformer':
            layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.,
                                               batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.delta = nn.Linear(dim, dim)
        else:
            raise ValueError(f'Unknown transition: {kind}')
        self.gate = nn.Linear(dim, 1) if gated else None

    def forward(self, z):
        if self.kind == 'transformer':
            delta = self.delta(self.net(z))
        else:
            global_z = z.mean(1, keepdim=True).expand_as(z)
            delta = self.net(torch.cat([z, global_z], -1))
        if self.gate is not None:
            delta = delta * self.gate(z).sigmoid()
        return z + delta


class DirectBehaviorPredictor(nn.Module):
    """Matched input, one-step behavior prediction without a latent target."""
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = ObservationEncoder(input_dim, latent_dim)
        self.transition = Transition(latent_dim, 'transformer')
        self.heads = BehaviorHeads(latent_dim)

    def forward(self, x):
        parent = self.encoder(x)
        return self.heads(self.transition(parent), parent)


def behavior_loss(pred, truth, eligible, previous_mask=None, stability=None):
    """Discrete supervision is confined to currently eligible masked positions.

    Confidence/content are recorded by the *next* native forward. Already
    committed positions do not generate easy positive commit/stability labels.
    """
    use = eligible.bool()
    if previous_mask is not None:
        use = use & previous_mask.bool()
    if not bool(use.any()):
        return pred['mask_logits'].sum() * 0.
    loss = F.binary_cross_entropy_with_logits(pred['mask_logits'][use], truth['mask'][use].float())
    for key in ('confidence', 'entropy', 'margin'):
        loss = loss + F.smooth_l1_loss(pred[key][use], truth[key][use].float())
    # Native semantic embeddings, never random token identity codes.
    loss = loss + .1 * F.smooth_l1_loss(pred['content'][use], truth['content'][use].float())
    if stability is not None and 'stability_logits' in pred:
        loss = loss + .25 * F.binary_cross_entropy_with_logits(
            pred['stability_logits'][use], stability[use].float())
    return loss


def project_masks(probability, previous_mask, eligible):
    """Reporting projection enforces irreversible commits and native progress.

    Raw probabilities are evaluated separately; this cannot improve their
    calibration score by hiding violations. No token identities are invented.
    """
    mask = previous_mask.bool().clone()
    active = eligible.bool() & mask
    commit = active & (probability < .5)
    for row in range(mask.shape[0]):
        if bool(active[row].any()) and not bool(commit[row].any()):
            score = probability[row].masked_fill(~active[row], float('inf'))
            commit[row, score.argmin()] = True
    return mask & ~commit
