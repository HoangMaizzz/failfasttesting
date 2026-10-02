"""Privileged verifier targets and prior-round memory without retaining GPU graphs."""
import torch
from structured_sparse_collector import FullContextVerifier
from world_model_environment import NativeTrainingEnvironment


def fixed_projection(width, output, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (width, output), generator=generator).float()*2-1)/width**.5


class TeacherVerifier(FullContextVerifier):
    @torch.inference_mode()
    def score(self, prefix, proposal, remaining_output_budget):
        captured = {}
        norm = self.model.model.norm
        if not hasattr(self, 'hidden_projection'):
            self.hidden_projection = fixed_projection(self.model.config.hidden_size, 32, 901)

        def hidden_hook(module, inputs, output):
            rows = output[0, len(prefix)-1:len(prefix)+len(proposal)].float()
            projection = self.hidden_projection.to(rows.device)
            captured['hidden'] = (rows @ projection).clamp(-30, 30).cpu()

        def logits_hook(module, inputs, output):
            # logits_to_keep already restricts the LM head to L+1 positions.
            rows = output[0, :len(proposal)].float()
            ids = torch.tensor(proposal, device=rows.device)
            candidate = rows.gather(1, ids[:, None]).squeeze(1)
            captured['probability'] = (candidate-torch.logsumexp(rows, -1)).exp().cpu()
            captured['agreement'] = rows.argmax(-1).eq(ids).float().cpu()

        handles = [norm.register_forward_hook(hidden_hook), self.model.lm_head.register_forward_hook(logits_hook)]
        try:
            result = super().score(prefix, proposal, remaining_output_budget)
        finally:
            for handle in handles: handle.remove()
        if set(captured) != {'hidden', 'probability', 'agreement'}:
            raise RuntimeError('Verifier hooks failed; refusing incomplete privileged features')
        margin = torch.tensor(self.last_teacher['margin'])
        features = torch.cat([torch.tanh(margin[:, None]/5), captured['probability'][:, None],
                              captured['agreement'][:, None], captured['hidden'][:len(proposal)]/10], -1)
        self.last_teacher.update(features=features, boundary_hidden=captured['hidden'][result[0]]/10,
            hidden_stage='final_norm_at_preceding_causal_position',
            hidden_format='fixed_seed_901_rademacher_projection_32_not_raw_hidden',
            local_agreement_semantics='teacher_forced_on_candidate_prefix_not_post_correction_reference')
        return result


class TeacherTrainingEnvironment(NativeTrainingEnvironment):
    def __init__(self, *args, token_table=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_table = token_table
        self.content_projection = fixed_projection(token_table.shape[-1], 8, 902)
        self.reset_history()

    def reset_history(self):
        self.verifier_history = []

    def make_state(self, *args, **kwargs):
        callback = self.on_state
        def with_history(state, *rest):
            state.observation.verifier_history = (torch.stack(self.verifier_history[-8:]).clone()
                if self.verifier_history else torch.empty(0, 48))
            callback(state, *rest)
        self.on_state = with_history
        try: return super().make_state(*args, **kwargs)
        finally: self.on_state = callback

    def capture_teacher(self, state, source):
        teacher = self.verifier.last_teacher
        state.observation.teacher_margin = torch.tensor(teacher['margin'], dtype=torch.float32)
        state.observation.teacher_features = teacher['features'].clone()
        # Older test doubles and previously saved runs can expose only the compact
        # teacher features. Keep that path usable; V1 FiLM training separately
        # requires the full top-K distribution targets and checks for them clearly.
        topk_available = all(key in teacher for key in ('topk_ids', 'topk_logits', 'logsumexp'))
        if topk_available:
            state.observation.teacher_topk_ids = torch.as_tensor(
                teacher['topk_ids'], dtype=torch.long).cpu()
            state.observation.teacher_topk_logits = torch.as_tensor(
                teacher['topk_logits'], dtype=torch.float16).cpu()
            state.observation.teacher_logsumexp = torch.as_tensor(
                teacher['logsumexp'], dtype=torch.float32).cpu()
        else:
            state.observation.teacher_topk_ids = None
            state.observation.teacher_topk_logits = None
            state.observation.teacher_logsumexp = None
        p_top1 = torch.as_tensor(teacher.get('top1_probability',
            state.observation.teacher_features[:, 1]), dtype=torch.float32).cpu()
        entropy = torch.as_tensor(teacher.get('normalized_entropy',
            torch.zeros(state.observation.length)), dtype=torch.float32).cpu()
        rank = torch.as_tensor(teacher.get('candidate_rank',
            torch.zeros(state.observation.length)), dtype=torch.float32).cpu()
        in_topk = torch.as_tensor(teacher.get('candidate_in_topk',
            torch.zeros(state.observation.length, dtype=torch.bool)), dtype=torch.bool).cpu()
        candidate_prob = state.observation.teacher_features[:, 1].clamp_min(1e-12)
        draft_prob = torch.zeros(state.observation.length, dtype=torch.float32)
        draft_valid = state.observation.scalars[:, 4].bool().cpu() & in_topk
        for position in range(state.observation.length):
            if not bool(draft_valid[position]):
                continue
            ids = state.observation.topk_ids[position]
            matches = (ids == state.observation.ids[position, 1]).nonzero(as_tuple=False)
            if not len(matches):
                draft_valid[position] = False
                continue
            conditional = torch.softmax(state.observation.gaps[position].float(), dim=-1)
            draft_prob[position] = conditional[int(matches[0, 0])].clamp_min(1e-12)
        log_gap = (candidate_prob.log() - draft_prob.clamp_min(1e-12).log()).clamp(-20, 20)
        state.observation.teacher_aux_features = torch.stack([
            p_top1, entropy, rank, log_gap, draft_valid.float(), in_topk.float()
        ], dim=-1)
        state.observation.teacher_is_actual = source == 'actual_STOP_forward'

    @torch.inference_mode()
    def remember(self, prefix, candidate, accepted, emitted, masks=0, refinement=0):
        teacher = self.verifier.last_teacher
        hidden = teacher['boundary_hidden'] if teacher is not None else torch.zeros(32)
        token = self.token_table[int(emitted[-1])].detach().float().cpu()
        content = token @ self.content_projection
        length = len(candidate)
        stats = torch.tensor([accepted/max(1, length), length/64, len(emitted)/65,
            float(self.eos_id in emitted), len(prefix)/4096, masks/max(1, length),
            refinement/3, float(accepted == length)])
        self.verifier_history.append(torch.cat([stats, hidden.cpu(), content]))
        self.verifier_history = self.verifier_history[-8:]

    def submit(self, state, remaining):
        elapsed = super().submit(state, remaining)
        self.capture_teacher(state, 'actual_STOP_forward')
        self.remember(state.prefix, state.observation.ids[:, 1].tolist(), state.observation.accepted,
                      state.emitted, int(state.observation.scalars[:, 0].sum()), state.snapshot_index)
        return elapsed

    def shadow(self, state, remaining):
        accepted, _, _, elapsed = self.verifier.score(state.prefix, state.observation.ids[:, 1].tolist(), remaining)
        state.observation.accepted = int(accepted)
        self.capture_teacher(state, 'shadow_forward')
        self.stats['shadow_verifier_calls'] = self.stats.get('shadow_verifier_calls', 0)+1
        # Crucially: no emitted tokens, submitted flag or history update here.
        return elapsed
