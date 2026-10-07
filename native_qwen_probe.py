"""Tiny native-Qwen survival probes and an exactly resumable training API.

Capture, split ownership, cohort metrics, and test evaluation belong to the caller.
``pack_batch(uids, spec, device)`` supplies frozen observations; the validation
callback ``evaluate_validation(model, spec)`` returns a lower-is-better score and
cohort dictionary. It must close over validation data only. Parallel jobs should
run in separate processes: PyTorch dropout uses process-global RNG state.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paired_latent_models import PairedStudent, acceptance_loss, outputs


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _spec(spec):
    result = dict(kind=_get(spec, 'kind'), variant=_get(spec, 'variant', 'HC'),
                  depth=_get(spec, 'depth'), dim=_get(spec, 'dim', 128),
                  lambda_K=_get(spec, 'lambda_K', .1),
                  lambda_rec=_get(spec, 'lambda_rec', 0.),
                  structural_input=_get(spec, 'structural_input', False))
    if result['kind'] not in ('raw', 'direct', 'latent'):
        raise ValueError('spec.kind must be raw, direct, or latent')
    if result['variant'] not in ('H', 'HC'):
        raise ValueError('spec.variant must be H or HC')
    if not isinstance(result['structural_input'], bool):
        raise ValueError('spec.structural_input must be a boolean')
    if result['kind'] == 'latent' and result['dim'] not in (64, 128, 256):
        raise ValueError('Verifier latent dimension must be 64, 128, or 256')
    for name in ('lambda_K', 'lambda_rec'):
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if result['lambda_rec'] and result['kind'] != 'latent':
        raise ValueError('Reconstruction is a separate latent-only ablation')
    return result


def _valid(lengths, width):
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('lengths must be a one-dimensional integer tensor')
    if bool(((lengths < 1) | (lengths > width)).any()):
        raise ValueError('Each proposal length must be between 1 and width')
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


class NativeProbe(nn.Module):
    """Identical capacity across depths; independent survival, never hazards."""
    def __init__(self, spec, hidden_dim, dropout):
        super().__init__()
        self.spec = _spec(spec)
        self.hidden_dim = hidden_dim
        self.variant = self.spec['variant']
        self.uses_structural = (self.spec['kind'] == 'latent' or
                               (self.spec['kind'] == 'raw' and self.spec['structural_input']))
        width = hidden_dim * (2 if self.variant == 'HC' else 1)
        if self.uses_structural:
            width += 4
        final_dim = self.spec['dim'] if self.spec['kind'] == 'latent' else 1
        self.encoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(256, final_dim))
        self.head = (nn.Sequential(nn.LayerNorm(final_dim), nn.Linear(final_dim, 1))
                     if self.spec['kind'] == 'latent' else None)
        self.reconstruction = (nn.Linear(final_dim, hidden_dim)
                               if self.spec['lambda_rec'] > 0 else None)

    def forward(self, hidden, candidate_embedding, structural, lengths):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError('hidden must have shape [B, positions, hidden_dim]')
        valid = _valid(lengths, hidden.shape[1])
        if hidden.shape[0] != lengths.shape[0]:
            raise ValueError('Hidden/length batch sizes differ')
        dtype = self.encoder[1].weight.dtype
        values = [hidden.detach().to(dtype=dtype)]
        if self.variant == 'HC':
            if candidate_embedding.shape != hidden.shape:
                raise ValueError('Candidate embeddings must be frozen own-Qwen [B, positions, d]')
            values.append(candidate_embedding.detach().to(dtype=dtype))
        if self.uses_structural:
            if structural is None or structural.shape != (*hidden.shape[:2], 4):
                raise ValueError('structural must contain exactly four positional scalars')
            values.append(structural.detach().to(dtype=dtype))
        # Padding is not an observation. Zero it before LN as well as afterward.
        x = torch.cat(values, -1).masked_fill(~valid[..., None], 0.)
        encoded = self.encoder(x)
        logits = (self.head(encoded) if self.head is not None else encoded).squeeze(-1)
        q = logits.sigmoid().masked_fill(~valid, 0.)
        result = dict(logits=logits, q=q, K=q.sum(-1))
        if self.head is not None:
            result['z'] = encoded.masked_fill(~valid[..., None], 0.)
        if self.reconstruction is not None:
            result['reconstructed_hidden'] = self.reconstruction(result['z'])
        return result


def make_model(spec, cfg, hidden_dim):
    """Create fresh models; depth is metadata, never an architectural choice.

    Raw ``structural_input=True`` appends the same four structural scalars used
    by the latent encoder, for a matched-input compression reference. Its default
    is False; latent probes always use structure and Direct uses only D inputs.
    """
    normalized = _spec(spec)
    dropout = _get(cfg, 'dropout', .05)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('dropout must be in [0, 1)')
    if normalized['kind'] == 'direct':
        model = PairedStudent('Direct', dim=128, layers=2, dropout=dropout)
        model.probe_spec = normalized
        return model
    if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 1:
        raise ValueError('Native hidden_dim must be positive')
    return NativeProbe(normalized, hidden_dim, dropout)


def call_model(model, spec, batch):
    """The forward interface never reads accepted, parent K, or final logits."""
    normalized = _spec(spec)
    stored = model.probe_spec if isinstance(model, PairedStudent) else model.spec
    if normalized != stored:
        raise ValueError('Model spec differs from call spec')
    if normalized['kind'] == 'direct':
        dtype = next(model.parameters()).dtype
        pred = model(batch['z_D'].detach().to(dtype=dtype),
                     batch['c'].detach().to(dtype=dtype),
                     batch['context'].detach().to(dtype=dtype), batch['lengths'])
        return dict(**outputs(pred['hazard'], batch['lengths']),
                    hazard=pred['hazard'], logits=pred['hazard'], z=pred['z_V'])
    return model(batch['hidden'], batch.get('candidate_embedding'),
                 batch.get('structural'), batch['lengths'])


def survival_truth(accepted, width):
    """Zero-based position i survives exactly when K >= i + 1."""
    return (accepted[:, None] >= torch.arange(1, width + 1, device=accepted.device)[None]).float()


def probe_loss(spec, prediction, batch):
    """Native: all-valid BCE + lambda_K Huber; Direct: current legacy loss."""
    spec = _spec(spec)
    if spec['kind'] == 'direct':
        return acceptance_loss(prediction['hazard'], batch['lengths'], batch['accepted'])
    logits, lengths, accepted = prediction['logits'], batch['lengths'], batch['accepted']
    valid = _valid(lengths, logits.shape[1])
    if accepted.shape != lengths.shape or bool((~torch.isfinite(accepted)).any()):
        raise ValueError('Native training requires finite accepted counts for every state')
    if bool(((accepted < 0) | (accepted > lengths) | (accepted != accepted.round())).any()):
        raise ValueError('Native accepted counts must be integers in [0, length]')
    truth = survival_truth(accepted.detach(), logits.shape[1])
    bce = F.binary_cross_entropy_with_logits(logits[valid], truth[valid])
    khuber = F.smooth_l1_loss(prediction['K'], accepted.detach().float())
    reconstruction = logits[valid].sum() * 0.
    if spec['lambda_rec']:
        target = batch['hidden'].detach().to(dtype=logits.dtype)
        reconstruction = F.mse_loss(prediction['reconstructed_hidden'][valid], target[valid])
    total = bce + spec['lambda_K'] * khuber + spec['lambda_rec'] * reconstruction
    return total, dict(bce=bce, K_huber=khuber, reconstruction=reconstruction)


def parameter_counts(model):
    return dict(inference=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


class QuestionActionSampler:
    """Action-uniform, question-uniform within action, state-uniform within question."""
    def __init__(self, uids, rows, seed):
        self.groups = defaultdict(lambda: defaultdict(list))
        if not uids or len(set(uids)) != len(uids):
            raise ValueError('Training UID list must be nonempty and unique')
        for uid in sorted(uids, key=str):
            row = rows[uid]
            question = row.get('question_id', row.get('question'))
            if question is None:
                raise ValueError('Training rows require question_id or question')
            if row.get('split') in ('val', 'validation', 'test'):
                raise ValueError('Non-training row entered the training sampler')
            self.groups[row.get('action', 'root')][question].append(uid)
        self.actions = sorted(self.groups, key=str)
        self.questions = {a: sorted(self.groups[a], key=str) for a in self.actions}
        self.rng = random.Random(seed)

    def sample(self, count):
        result = []
        for _ in range(count):
            action = self.rng.choice(self.actions)
            question = self.rng.choice(self.questions[action])
            result.append(self.rng.choice(self.groups[action][question]))
        return result


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, '__dict__'):
        return _jsonable(vars(value))
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in _weights(model).items():
        digest.update(name.encode())
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state(device):
    array_state = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(), python=random.getstate(),
                numpy=[array_state[0], torch.tensor(array_state[1].astype(np.int64)),
                       *array_state[2:]],
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _restore_rng(state, device):
    torch.set_rng_state(state['torch_cpu'])
    random.setstate(state['python'])
    array_state = state['numpy']
    np.random.set_state((array_state[0], array_state[1].numpy().astype(np.uint32),
                         *array_state[2:]))
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def _save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _training_config(cfg):
    values = dict(updates=_get(cfg, 'updates', _get(cfg, 'max_updates')),
                  eval_every=_get(cfg, 'eval_every'), batch_size=_get(cfg, 'batch_size'),
                  lr=_get(cfg, 'lr', _get(cfg, 'learning_rate')), dropout=_get(cfg, 'dropout', .05),
                  study_signature=_get(cfg, 'study_signature'))
    for name in ('updates', 'eval_every', 'batch_size'):
        if not isinstance(values[name], int) or isinstance(values[name], bool) or values[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if values['lr'] is None or not math.isfinite(values['lr']) or values['lr'] <= 0:
        raise ValueError('lr (or learning_rate) must be finite and positive')
    if not values['study_signature']:
        raise ValueError('cfg.study_signature is required for source/capture/split identity')
    return values


def train_model(spec, seed, train_uids, rows, pack_batch, evaluate_validation,
                cfg, folder, device):
    """Train fixed updates and return the model selected by validation only.

    ``last.pt`` and ``best.pt`` both include optimizer, RNG, sampler, signature,
    and curve. Resume restores the last committed validation checkpoint. The
    study_signature must identify all observation contents and the fixed split;
    local signatures also cover config, spec, training IDs, and implementation.
    """
    normalized, config = _spec(spec), _training_config(cfg)
    device, folder = torch.device(device), Path(folder)
    train_uids = list(train_uids)
    sampler = QuestionActionSampler(train_uids, rows, seed + 700)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    # Inspect without consuming any of the actual training RNG stream.
    before_pack = _rng_state(device)
    if normalized['kind'] == 'direct':
        hidden_dim = 0
    else:
        sample = pack_batch([sorted(train_uids, key=str)[0]], spec, device)
        hidden_dim = sample['hidden'].shape[-1]
        del sample
    _restore_rng(before_pack, device)
    model = make_model(spec, cfg, hidden_dim).to(device)
    initial_hash = _weight_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('native_qwen_probe.py', 'paired_latent_models.py', 'phase0_wm_models.py')}
    signature = _digest(dict(schema='native_qwen_probe_v1', spec=normalized, config=_jsonable(cfg),
                             seed=seed, train_uids=sorted(train_uids, key=str), hidden_dim=hidden_dim,
                             groups={str(a): {str(q): v for q, v in g.items()}
                                     for a, g in sampler.groups.items()}, implementation=implementation,
                             device=str(device), torch_version=str(torch.__version__)))
    folder.mkdir(parents=True, exist_ok=True)
    last_path, best_path = folder / 'last.pt', folder / 'best.pt'
    curve, batch_digest, start, best_score = [], '', 1, math.inf
    if last_path.exists():
        old = torch.load(last_path, map_location='cpu', weights_only=True)
        if old['signature'] != signature or old['initial_hash'] != initial_hash:
            raise ValueError('Resume signature differs: source, split, config, spec, or initialization')
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        sampler.rng.setstate(old['sampler_rng'])
        _restore_rng(old['rng'], device)
        curve, batch_digest, start = old['curve'], old['batch_digest'], old['step'] + 1
        best_score = old['best_score']
        # The last checkpoint is authoritative if an interruption occurred
        # between writing best.pt and last.pt at a validation boundary.
        best_payload = old['best_checkpoint']
        _save(best_path, best_payload)
    else:
        if best_path.exists() or (folder / 'selection.json').exists():
            raise ValueError('Existing output has no resumable last checkpoint')
        best_payload = None
    for step in range(start, config['updates'] + 1):
        model.train()
        uids = sampler.sample(config['batch_size'])
        batch_digest = _digest([batch_digest, uids])
        batch = pack_batch(uids, spec, device)
        prediction = call_model(model, spec, batch)
        loss, parts = probe_loss(spec, prediction, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite probe loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite probe gradient')
        optimizer.step()
        if step % config['eval_every'] != 0 and step != config['updates']:
            continue
        model.eval()
        with torch.no_grad():
            score, cohorts = evaluate_validation(model, spec)
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Validation score must be finite and lower-is-better')
        item = dict(step=step, loss=float(loss.detach()),
                    **{k: float(v.detach()) for k, v in parts.items()},
                    validation_score=score, validation_cohorts=_jsonable(cohorts),
                    gradient_norm=float(norm), batch_digest=batch_digest,
                    selection_split='validation', test_access=False)
        curve.append(item)
        payload = dict(schema='native_qwen_probe_v1', model=_weights(model),
                       optimizer=optimizer.state_dict(), sampler_rng=sampler.rng.getstate(),
                       rng=_rng_state(device), step=step, curve=list(curve),
                       validation_score=score, validation_cohorts=_jsonable(cohorts),
                       initial_hash=initial_hash, batch_digest=batch_digest, signature=signature,
                       spec=normalized, seed=seed, parameter_counts=parameter_counts(model),
                       test_access=False, selection_split='validation')
        improved = score < best_score
        if improved:
            best_score = score
            # torch.save is immediate, but the retained best optimizer must not
            # alias optimizer state tensors changed by subsequent updates.
            best_payload = copy.deepcopy(payload)
        payload['best_score'] = best_score
        payload['best_checkpoint'] = best_payload
        _save(last_path, payload)
        if improved:
            _save(best_path, best_payload)
        _write_json(folder / 'learning_curve.json', curve)
    if best_payload is None:
        raise RuntimeError('No validation-selected checkpoint')
    model.load_state_dict(best_payload['model'])
    model.eval()
    record = dict(spec=normalized, seed=seed, updates=config['updates'],
                  selected_step=best_payload['step'], validation_score=best_payload['validation_score'],
                  validation_cohorts=best_payload['validation_cohorts'],
                  study_signature=config['study_signature'], signature=signature,
                  initial_hash=initial_hash, batch_digest=batch_digest,
                  parameter_counts=parameter_counts(model), hidden_dim=hidden_dim,
                  selection_split='validation', test_access=False)
    _write_json(folder / 'learning_curve.json', curve)
    _write_json(folder / 'selection.json', record)
    _write_json(folder / 'complete.json', record)
    return model, record
