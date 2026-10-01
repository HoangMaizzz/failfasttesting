"""Token latent learned from native observations, privileged teacher and past STOPs.

Teacher data at the current state is used only in loss functions. History is an
immutable, bounded window of preceding actual submissions in the same episode.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from world_model_core import ActionDynamics, Latent, transformer, valid_positions
from world_model_probe import ProbeEncoder, ProbeWorldModel

FEATURE_VARIANTS = ('full', 'no_hidden', 'no_token_ids', 'no_topk', 'no_confidence',
    'no_masks', 'no_prefix', 'no_native_history', 'no_provenance',
    'no_verifier_memory', 'no_teacher', 'no_verifier_hidden',
    'no_latent_loss', 'no_structure_loss', 'no_rollout_loss', 'no_residual',
    'no_layer7', 'no_layer14', 'no_layer28', 'no_block_alignment', 'no_agreement_readout')


class MemoryEncoder(ProbeEncoder):
    def __init__(self, *args, variant='full', **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.memory = nn.GRU(48, self.dim, batch_first=True)
        self.memory_fuse = nn.Linear(self.dim, self.dim)

    def forward(self, batch):
        b = dict(batch)
        variant = self.variant
        scalars = b['scalars'].clone()
        if variant == 'no_hidden': b['hidden'] = torch.zeros_like(b['hidden'])
        if variant in ('no_layer7', 'no_layer14', 'no_layer28'):
            # Default experiment has the native layers [7,14,28] in this order.
            b['hidden'] = b['hidden'].clone()
            b['hidden'][:, :, ('no_layer7','no_layer14','no_layer28').index(variant)] = 0
        if variant == 'no_token_ids': b['token_vectors'] = torch.zeros_like(b['token_vectors'])
        if variant == 'no_topk':
            b['topk_vectors'] = torch.zeros_like(b['topk_vectors'])
            b['gaps'] = torch.zeros_like(b['gaps'])
        if variant == 'no_confidence': scalars[:, :, 7:9] = 0
        if variant == 'no_provenance': scalars[:, :, 2:7] = 0; scalars[:, :, 12:14] = 0
        if variant == 'no_block_alignment':
            scalars[:, :, 10:12] = 0
            b['context'] = b['context'].clone(); b['context'][:, 5:7] = 0
        if variant == 'no_masks':
            scalars[:, :, :3] = 0; scalars[:, :, 14] = 0
            b['token_vectors'] = b['token_vectors'].clone()
            b['token_vectors'][:, :, 0] = b['token_vectors'][:, :, 1]
        if variant == 'no_native_history': b['history'] = torch.zeros_like(b['history'])
        if variant == 'no_prefix': b['prefix_vectors'] = torch.zeros_like(b['prefix_vectors'])
        b['scalars'] = scalars
        z = super().forward(b)
        z.context = batch['context']  # factual transition metadata stays exact
        # Mask/frontier are exact legal-action metadata, not learned feature cues.
        z.mask_probs = batch['scalars'][:, :, 0] * valid_positions(z.lengths, z.tokens.shape[1])
        index = torch.arange(z.tokens.shape[1], device=z.tokens.device)
        z.frontier = torch.minimum(torch.where(z.mask_probs.bool(), index,
            z.tokens.shape[1]).min(-1).values, z.lengths)
        if variant != 'no_verifier_memory' and 'verifier_history' in b:
            history = b['verifier_history']
            if variant == 'no_verifier_hidden':
                history = history.clone(); history[:, :, 8:40] = 0
            lengths = b['verifier_history_lengths']
            # Gather each example's last REAL step; padding cannot update memory.
            sequence, _ = self.memory(history)
            row = torch.arange(len(lengths), device=lengths.device)
            memory = sequence[row, (lengths-1).clamp_min(0)] * (lengths > 0)[:, None]
            memory = self.memory_fuse(memory) * (lengths > 0)[:, None]
            z.global_state = z.global_state + memory
            z.tokens = z.tokens + memory[:, None] * valid_positions(z.lengths, z.tokens.shape[1])[:, :, None]
        return z


class ResidualDynamics(ActionDynamics):
    def __init__(self, dim, dropout, residual=True, variant='full'):
        super().__init__(dim, 2, dropout)
        self.residual = residual
        self.variant = variant
        self.token_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)

    def forward(self, state, actions, extension_size):
        source = state
        if self.variant == 'no_block_alignment':
            context = state.context.clone(); context[:,5:7] = 0
            source = Latent(state.tokens, state.global_state, state.lengths, context)
        pred = super().forward(source, actions, extension_size)
        context = pred.context.clone()
        context[:,5:7] = state.context[:,5:7]
        pred.context = context
        if self.residual:
            width = pred.tokens.shape[1]
            old = F.pad(state.tokens, (0, 0, 0, max(0, width-state.tokens.shape[1])))[:, :width]
            pred.tokens = self.token_norm(pred.tokens + old)
            pred.global_state = self.global_norm(pred.global_state + state.global_state)
            pred.tokens *= valid_positions(pred.lengths, width)[:, :, None]
        return pred


class VerifierTeacher(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.input = nn.Linear(35, dim)
        self.network = transformer(dim, 1, dropout)
        self.output = nn.Linear(dim, 32)
        self.margin = nn.Linear(32, 1)
        self.agreement = nn.Linear(32, 1)

    def forward(self, features, lengths):
        valid = valid_positions(lengths, features.shape[1])
        h = self.network(self.input(features), src_key_padding_mask=~valid)
        return self.output(h) * valid[:, :, None]


class TwoSourceWorldModel(ProbeWorldModel):
    schema = 'acceptance_two_source_memory_v1'
    teach_imagined = True

    def __init__(self, hidden_dim, token_dim, top_k=32, dim=128,
                 num_hidden_layers=3, dropout=.1, architecture='two_source', variant='full'):
        if architecture != 'two_source' or variant not in FEATURE_VARIANTS:
            raise ValueError('Invalid two-source model configuration')
        super().__init__(hidden_dim, token_dim, top_k, dim, num_hidden_layers, dropout)
        if dim < 64: raise ValueError('Need at least 64 latent channels: 32 agreement + native')
        self.config.update(architecture=architecture, variant=variant)
        self.variant = variant
        self.encoder = MemoryEncoder(hidden_dim, token_dim, top_k, dim, num_hidden_layers,
                                     dropout, variant=variant)
        self.dynamics = ResidualDynamics(dim, dropout, residual=variant != 'no_residual', variant=variant)
        self.accept_head = nn.Sequential(nn.Linear(dim+32, dim//2), nn.GELU(), nn.Linear(dim//2, 1))
        self.margin_head = nn.Linear(32, 1)
        self.local_head = nn.Linear(32, 1)
        self.teacher = VerifierTeacher(dim, dropout)
        self.teacher_ema = copy.deepcopy(self.teacher).eval().requires_grad_(False)

    def acceptance(self, z):
        agreement = z.tokens[:, :, -32:]
        if self.variant == 'no_agreement_readout': agreement = torch.zeros_like(agreement)
        logits = self.accept_head(torch.cat([agreement,
            z.global_state[:, None].expand(-1, agreement.shape[1], -1)], -1)).squeeze(-1)
        if z.carry_logits is not None:
            logits = torch.where(z.carry_valid, z.carry_logits, logits)
        return logits

    def local_agreement(self, z):
        return self.local_head(z.tokens[:, :, -32:]).squeeze(-1)

    @torch.no_grad()
    def update_teacher_ema(self, decay):
        for dst, src in zip(self.teacher_ema.parameters(), self.teacher.parameters()):
            dst.lerp_(src, 1-decay)

    def teacher_loss(self, z, batch):
        if self.variant == 'no_teacher': return z.tokens.sum()*0
        valid = batch['teacher_valid']
        margin = torch.tanh(batch['teacher_margin'].detach()/5)
        if 'teacher_features' in batch:
            features = batch['teacher_features'].detach().clone()
            local = features[:, :, 2]
        else:
            features = torch.zeros(*margin.shape, 35, device=margin.device)
            features[:, :, 0] = margin
            features[:, :, 2] = (margin >= 0).float()
            local = features[:, :, 2]
        if self.variant == 'no_verifier_hidden': features[:, :, 3:] = 0
        # Never allow dropout in EMA targets, even when model.train() is called.
        self.teacher_ema.eval()
        with torch.no_grad(): target = self.teacher_ema(features, batch['lengths'])
        teacher = self.teacher(features, batch['lengths'])
        teacher_fit = F.smooth_l1_loss(torch.tanh(self.teacher.margin(teacher).squeeze(-1)),
                                      margin, reduction='none')
        teacher_fit += F.binary_cross_entropy_with_logits(self.teacher.agreement(teacher).squeeze(-1),
                                                         local, reduction='none')
        belief = z.tokens[:, :, -32:]
        distillation = 1-F.cosine_similarity(belief, target, dim=-1)
        predicted_margin = torch.tanh(self.margin_head(belief).squeeze(-1))
        task = F.smooth_l1_loss(predicted_margin, margin, reduction='none')
        task += F.binary_cross_entropy_with_logits(self.local_agreement(z), local, reduction='none')
        return self.region_mean(teacher_fit + task + .25*distillation, valid)


def configure_losses(learner, variant):
    learner.teacher_weight = 0 if variant == 'no_teacher' else .25
    learner.latent_weight = 0 if variant == 'no_latent_loss' else .1
    learner.structure_weight = 0 if variant == 'no_structure_loss' else .1
    # Existing learner supports a separately configurable rollout loss below.
    learner.rollout_weight = 0 if variant == 'no_rollout_loss' else 1.
