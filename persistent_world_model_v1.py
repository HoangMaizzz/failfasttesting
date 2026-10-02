"""Persistent drafter/verifier latent and zero-start gated FiLM for Fast-dLLM V1."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from world_model_core import positions, valid_positions
from world_model_probe import ProbeEncoder


def hazard_nll(logits, lengths, accepted):
    """Prefix-censored Bernoulli loss: only accepted tokens and first reject count."""
    if logits.ndim != 2 or lengths.shape != accepted.shape:
        raise ValueError("Expected logits [B,L] and lengths/accepted [B]")
    total = logits.sum() * 0.0
    count = 0
    for row in range(len(lengths)):
        length, y = int(lengths[row]), int(accepted[row])
        if y < 0:
            continue
        if y > length:
            raise ValueError("Accepted prefix exceeds proposal length")
        observed = min(length, y + int(y < length))
        if observed == 0:
            continue
        target = torch.zeros(observed, device=logits.device)
        target[:y] = 1.0
        total = total + F.binary_cross_entropy_with_logits(
            logits[row, :observed].float(), target, reduction="sum")
        count += observed
    return total / max(1, count)


def expected_yield(logits, lengths):
    survival = torch.sigmoid(logits.float()).cumprod(-1)
    return (survival * valid_positions(lengths, logits.shape[-1])).sum(-1)


def prefix_survival(logits):
    return torch.sigmoid(logits.float()).cumprod(-1)


def hazard_mode(logits):
    q = torch.sigmoid(logits.float())
    survival = torch.cumprod(q, dim=-1)
    before = F.pad(survival[..., :-1], (1, 0), value=1.0)
    probabilities = torch.cat([before * (1-q), survival[..., -1:]], dim=-1)
    return probabilities.argmax(-1)


class PersistentWorldModelV1(nn.Module):
    """D(64) + verifier(64) posterior, R/E dynamics, correction and token hazards."""

    schema = "persistent_drafter_verifier_film_v1"

    def __init__(self, hidden_dim, token_dim, top_k=32, num_hidden_layers=3,
                 dropout=0.1, model_dim=128):
        super().__init__()
        if model_dim != 128:
            raise ValueError("V1 fixes the persistent world latent at 128 dimensions")
        self.config = dict(hidden_dim=hidden_dim, token_dim=token_dim, top_k=top_k,
                           num_hidden_layers=num_hidden_layers, dropout=dropout,
                           model_dim=model_dim)
        self.d_encoder = ProbeEncoder(hidden_dim, token_dim, top_k, 128,
                                      num_hidden_layers, dropout)
        self.d_projection = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64))
        # 35 current verifier features + 6 distribution/gap/validity summaries + position.
        self.v_projection = nn.Linear(42, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.v_encoder = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(64),
                                               enable_nested_tensor=False)
        self.v_summary = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
        self.posterior = nn.Sequential(nn.Linear(128, 256), nn.GELU(),
                                       nn.LayerNorm(256), nn.Linear(256, 128))
        self.unverified = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128))
        self.transitions = nn.ModuleDict({
            "R": nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.GELU(),
                               nn.Linear(256, 128)),
            "E": nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.GELU(),
                               nn.Linear(256, 128)),
        })
        self.corrector = nn.Sequential(nn.Linear(192, 256), nn.GELU(),
                                       nn.LayerNorm(256), nn.Linear(256, 128))
        self.hazard_head = nn.Sequential(nn.Linear(128 + 17, 160), nn.GELU(),
                                         nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))
        self.model_dim = model_dim

    def encode_d(self, batch, ablate_hidden=False):
        if not ablate_hidden:
            encoded = self.d_encoder(batch)
        else:
            copy = dict(batch)
            copy["hidden"] = torch.zeros_like(batch["hidden"])
            encoded = self.d_encoder(copy)
        valid = valid_positions(encoded.lengths, encoded.tokens.shape[1]).float()
        pooled = (encoded.tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.d_projection(torch.cat([encoded.global_state, pooled], dim=-1))

    def encode_v(self, batch, ablate_hidden=False, ablate_logits=False):
        features = batch.get("teacher_features")
        if features is None:
            return torch.zeros(len(batch["lengths"]), 64, device=batch["lengths"].device)
        features = features.float().clone()
        if ablate_hidden:
            features[:, :, 3:] = 0
        if ablate_logits:
            features[:, :, :3] = 0
        auxiliary = batch.get("teacher_aux_features")
        if auxiliary is None:
            auxiliary = torch.zeros(*features.shape[:2], 6, device=features.device)
        else:
            auxiliary = auxiliary.float().clone()
            if ablate_logits:
                auxiliary[:, :, :4] = 0
        labels = batch["labels"].clamp_min(0)
        lengths = batch["lengths"]
        # A verifier sees causal rows through the first rejected token, not the
        # suffix after rejection. For full acceptance it sees all proposal rows.
        v_lengths = torch.minimum(lengths, labels + (labels < lengths).long())
        teacher_valid = batch.get("teacher_valid", torch.zeros_like(features[:, :, 0], dtype=torch.bool))
        v_lengths = torch.where(teacher_valid.any(-1), v_lengths, torch.zeros_like(v_lengths))
        position = torch.arange(features.shape[1], device=features.device)[None, :]
        relative = position.float() / lengths.clamp_min(1)[:, None]
        sequence_features = torch.cat([features, auxiliary,
                                       relative[:, :, None]], dim=-1)
        sequence = self.v_projection(sequence_features)
        valid = valid_positions(v_lengths, sequence.shape[1])
        # An all-masked row can yield NaNs in Transformer attention. Give rows
        # with no verifier observation a harmless zero sentinel for the pass;
        # the real pooling mask below still makes their V embedding exactly 0.
        attention_valid = valid.clone()
        no_teacher = ~teacher_valid.any(-1)
        if bool(no_teacher.any()) and attention_valid.shape[1]:
            attention_valid[no_teacher, 0] = True
            sequence = sequence.clone()
            sequence[no_teacher, 0] = 0
        encoded = self.v_encoder(sequence, src_key_padding_mask=~attention_valid)
        pooled = (encoded * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        summary = torch.stack([labels.float() / lengths.clamp_min(1),
                               (labels + (labels < lengths).long()).float() /
                               lengths.clamp_min(1)], dim=-1)
        return pooled + self.v_summary(summary) * teacher_valid.any(-1)[:, None]

    def encode(self, batch, *, ablate_hidden=False, ablate_verifier_hidden=False,
               ablate_verifier_logits=False):
        d = self.encode_d(batch, ablate_hidden=ablate_hidden)
        v = self.encode_v(batch, ablate_hidden=ablate_verifier_hidden,
                          ablate_logits=ablate_verifier_logits)
        z_post = self.posterior(torch.cat([d, v], dim=-1))
        z_d = self.unverified(d)
        has_v = batch.get("teacher_valid", torch.zeros(
            len(d), 1, dtype=torch.bool, device=d.device)).any(-1)
        z = torch.where(has_v[:, None], z_post, z_d)
        return {"d": d, "v": v, "posterior": z_post, "unverified": z_d,
                "state": z, "has_verifier": has_v}

    def transition_prior(self, z, action):
        if action not in ("R", "E"):
            raise ValueError("World transition action must be R or E")
        return z + self.transitions[action](z)

    def correct(self, prior, d_next):
        return prior + self.corrector(torch.cat([prior, d_next], dim=-1))

    def root_observation(self, previous_z, d_root):
        """Assimilate a new native draft after the preceding verifier call."""
        return self.correct(previous_z, d_root)

    def transition(self, z, action, d_next):
        return self.correct(self.transition_prior(z, action), d_next)

    def hazards(self, z, lengths, width=None):
        width = int(width or lengths.max().item())
        p = positions(width, 16, z.device)[None].expand(len(z), -1, -1)
        length = lengths.float().clamp_min(1)[:, None, None].expand(-1, width, 1) / 64
        state = z[:, None, :].expand(-1, width, -1)
        return self.hazard_head(torch.cat([state, p, length], dim=-1)).squeeze(-1)

    def posterior_from_batch(self, batch):
        encoded = self.encode(batch)
        return encoded["posterior"]


class GatedFiLMAdapter(nn.Module):
    """Per-token gated FiLM on the final normalized Fast-dLLM representation."""

    def __init__(self, hidden_dim, latent_dim=128, gate_dim=128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.film = nn.Linear(self.latent_dim, 2 * self.hidden_dim)
        self.gate = nn.Sequential(nn.Linear(self.hidden_dim + self.latent_dim, gate_dim),
                                  nn.GELU(), nn.Linear(gate_dim, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, hidden, latent):
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError("FiLM hidden width does not match the Fast-dLLM layer")
        z = latent.to(device=hidden.device, dtype=torch.float32)
        if z.ndim == 1:
            z = z[None]
        if z.shape[0] == 1 and hidden.shape[0] > 1:
            z = z.expand(hidden.shape[0], -1)
        if z.shape != (hidden.shape[0], self.latent_dim):
            raise ValueError("Expected one 128-dimensional persistent latent per batch row")
        original_dtype = hidden.dtype
        h = hidden.float()
        normalized = self.norm(h)
        gamma, beta = self.film(z).chunk(2, dim=-1)
        z_rows = z[:, None, :].expand(-1, hidden.shape[1], -1)
        if getattr(self, "force_gate", None) is None:
            gate = torch.sigmoid(self.gate(torch.cat([normalized, z_rows], dim=-1)))
        else:
            gate = torch.full_like(hidden[..., :1], float(self.force_gate))
        conditioned = h + gate * (gamma[:, None] * normalized + beta[:, None])
        return conditioned.to(original_dtype)


def grouped_distribution_kl(student_logits, teacher_topk_logits, teacher_logsumexp,
                            support_ids, preserve_logits=None):
    """KL on verifier top-K token categories plus one exact OTHER bucket."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    selected = student_logp.gather(-1, support_ids.long())
    student_p = selected.exp()
    student_other = (1.0 - student_p.sum(-1, keepdim=True)).clamp_min(1e-8)
    student_categories = torch.cat([student_p, student_other], dim=-1).clamp_min(1e-8)
    student_categories = student_categories / student_categories.sum(-1, keepdim=True)

    teacher_p = torch.exp(teacher_topk_logits.float() - teacher_logsumexp.float()[..., None])
    teacher_other = (1.0 - teacher_p.sum(-1, keepdim=True)).clamp_min(1e-8)
    teacher_categories = torch.cat([teacher_p, teacher_other], dim=-1).clamp_min(1e-8)
    teacher_categories = teacher_categories / teacher_categories.sum(-1, keepdim=True)
    target_kl = (teacher_categories * (teacher_categories.log() - student_categories.log())).sum(-1)
    if preserve_logits is None:
        return target_kl
    base_logp = F.log_softmax(preserve_logits.float(), dim=-1).gather(-1, support_ids.long())
    base_p = base_logp.exp()
    base_other = (1.0 - base_p.sum(-1, keepdim=True)).clamp_min(1e-8)
    base_categories = torch.cat([base_p, base_other], dim=-1).clamp_min(1e-8)
    base_categories = base_categories / base_categories.sum(-1, keepdim=True)
    preserve_kl = (base_categories * (base_categories.log() - student_categories.log())).sum(-1)
    return target_kl + 0.05 * preserve_kl
