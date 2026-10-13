"""Readout probes and direct H1 predictors; no latent rollout or LLM weights."""
import torch
from torch import nn


class ProfileProbe(nn.Module):
    def __init__(self, sizes, groups, width=256, layers=2, max_length=64,
                 profile_input=False, residual=False, margin_scale=1., dropout=.1):
        super().__init__()
        if width % 4:
            raise ValueError("Probe width must be divisible by four")
        self.groups = tuple(groups)
        self.profile_input = profile_input
        self.residual = residual
        self.margin_scale = margin_scale
        self.config = dict(sizes=sizes, groups=list(groups), width=width, layers=layers,
                           max_length=max_length, profile_input=profile_input,
                           residual=residual, margin_scale=margin_scale, dropout=dropout)
        self.projections = nn.ModuleDict({g: nn.Linear(sizes[g], width) for g in groups})
        if profile_input:
            self.profile_projection = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.contextual = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width),
                                               enable_nested_tensor=False)
        self.margin_head = nn.Linear(width, 1)
        self.current_head = nn.Linear(width, 1)
        self.K_head = nn.Linear(width, max_length + 1)

    def forward(self, batch):
        # Only named present-time feature channels are read. Targets are separate.
        x = sum(self.projections[g](batch[g]) for g in self.groups)
        if self.profile_input:
            x = x + self.profile_projection(batch["profile"][:, :, None] / self.margin_scale)
        x = self.contextual(x, src_key_padding_mask=~batch["valid"])
        margin = self.margin_head(x).squeeze(-1) * self.margin_scale
        if self.residual:
            margin = margin + batch["profile"]
        pooled = (x * batch["valid"][:, :, None]).sum(1) / batch["lengths"][:, None]
        K_logits = self.K_head(pooled)
        K_logits = K_logits.masked_fill(
            torch.arange(K_logits.shape[1], device=x.device)[None] > batch["lengths"][:, None],
            -1e9)
        return dict(margin=margin, current_margin=self.current_head(x).squeeze(-1) * self.margin_scale,
                    K_logits=K_logits)


def state_mean(values, valid):
    return ((values * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


def loss_for(pred, target, valid, scale, task="margin", current=None, sign_weight=0.):
    from torch.nn import functional as F
    margin_loss = state_mean(F.smooth_l1_loss(pred["margin"] / scale,
                                             target["margin"] / scale, reduction="none"), valid)
    if task == "K":
        return F.cross_entropy(pred["K_logits"], target["K"])
    if task == "K_profile":
        current_loss = state_mean(F.smooth_l1_loss(pred["current_margin"] / scale,
                                                 current["margin"] / scale, reduction="none"), valid)
        return F.cross_entropy(pred["K_logits"], target["K"]) + .5 * (margin_loss + current_loss)
    sign_loss = state_mean(F.binary_cross_entropy_with_logits(
        pred["margin"] / scale, target["match"], reduction="none"), valid)
    return margin_loss + sign_weight * sign_loss
