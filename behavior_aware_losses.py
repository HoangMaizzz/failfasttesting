"""Stable survival losses on the unchanged Phase0 conditional-hazard heads."""
import torch
from torch.nn import functional as F
from phase0_wm_models import valid_positions, masked_mean


def stage_a_targets(rows, device):
    """Reuse Phase0 Stage-A indexing: zero-based position i survives iff i<K.

    Stage A originally constructs these tensors inside verifier_loss. This
    named extraction is regression-tested against that unchanged function.
    Missing K stays unobserved; conditional hazard is censored AFTER rejection.
    """
    lengths = torch.tensor([r['length'] for r in rows], device=device)
    labels = torch.tensor([-1 if r['accepted'] is None else r['accepted'] for r in rows], device=device)
    if bool(((labels > lengths) | (labels < -1)).any()):
        raise ValueError('Invalid accepted length')
    pos = torch.arange(64, device=device)[None]
    observed = valid_positions(lengths) & (labels >= 0)[:, None]
    return dict(lengths=lengths, labels=labels, observed=observed,
                survival=(pos < labels[:, None]).float(), risk=observed & (pos <= labels[:, None]))


def log_survival(heads):
    return F.logsigmoid(heads['hazard']).cumsum(-1)


def survival_bce(logq, target, mask):
    # log(1-exp(logq)) using expm1 remains stable near q=1. Positive targets
    # retain gradient even when a long prefix makes q extremely small.
    log1mq = torch.log(-torch.expm1(logq.clamp_max(-torch.finfo(logq.dtype).tiny)))
    return masked_mean(-(target * logq + (1-target) * log1mq), mask)


def behavior_losses(heads, target_rows, parent_rows, mode, oracle=None):
    logq = log_survival(heads); q = logq.exp()
    labels = stage_a_targets(target_rows, logq.device)
    valid = valid_positions(labels['lengths'])
    expected = (q * valid).sum(-1)
    if mode == 'B1':
        if oracle is None or oracle.requires_grad:
            raise ValueError('B1 requires detached frozen-real-child survival')
        qloss = survival_bce(logq, oracle, valid)
        kloss = F.smooth_l1_loss(expected, (oracle * valid).sum(-1))
    elif mode in ('B2', 'C'):
        qloss = survival_bce(logq, labels['survival'], labels['observed'])
        # Parent truth occurs ONLY in loss. Algebraically the two subtractions
        # cancel, but observed-parent masking is preserved for the action loss.
        parent = torch.tensor([-1 if r['accepted'] is None else r['accepted'] for r in parent_rows], device=q.device)
        observed_delta = (parent >= 0) & (labels['labels'] >= 0)
        delta_pred = expected - parent
        delta_true = labels['labels'] - parent
        kloss = masked_mean(F.smooth_l1_loss(delta_pred, delta_true.float(), reduction='none'), observed_delta)
    else:
        raise ValueError('Only B1/B2/C have behavior losses')
    return qloss, kloss


def objective(state_loss, heads, children, parents, mode, lambda_q, lambda_K, oracle=None):
    # Exactly zero weights execute EXACTLY the A path, including no G forward
    # at the call site. No behavior loss can change RNG, dropout, or optimizer.
    if mode == 'A' or (lambda_q == 0 and lambda_K == 0):
        return state_loss, None, None
    q, k = behavior_losses(heads, children, parents, mode, oracle)
    return state_loss + lambda_q * q + lambda_K * k, q, k


def gradient_norm(loss, parameters):
    if loss is None or not loss.requires_grad:
        return None
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [g.detach().square().sum() for g in gradients if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.
