"""Frozen Phase0 inputs for the projected32_teacher_pilot feasibility test.

Teacher columns 0:3 (margin/probability/agreement) are never exported. The
remaining 32 columns are supervision or an explicit hidden-only oracle input;
they are not native LLM hidden states and imply no full-Qwen compression claim.
``pack_paired_rows`` includes targets; select ``D_INPUT_FIELDS`` for D inputs.
No teacher, acceptance label, or parent truth belongs in that selection.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import tempfile
import zipfile

import torch
from torch.nn import functional as F

import behavior_aware_source as source
from phase0_wm_data import dataset_digest, load_dataset


TEACHER_INPUT_DIM = 32
EMBEDDING_DIM = 64
LATENT_DIM = 128
D_INPUT_FIELDS = ('z_D', 'c', 'context', 'lengths')
DYNAMICS_MEMBER = 'latent_128/dynamics/H1/best.pt'


def _float_tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}')
    if not value.is_floating_point():
        raise ValueError(f'{name} must be floating point')
    result = value.detach().to(device='cpu', dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must be finite')
    return result


def project_teacher_hidden(features, length):
    """Copy only columns 3:35; absence remains absence, never a negative."""
    if features is None:
        return None
    features = _float_tensor(features, (length, 35), 'teacher_features')
    return features[:, 3:35].clone()


def lookup_candidate_embedding(embedding, ids):
    """NativeVerifier teacher input only, from frozen same-forward STOP identities.

    Real endpoint identities must never enter a D/student rollout forward.
    """
    if embedding.get('learned', False) or embedding.get('vocabulary') is not None:
        raise ValueError('Candidate embeddings require the frozen pretrained full vocabulary')
    weights = embedding.get('weights')
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 2
            or weights.shape[1] != EMBEDDING_DIM or not weights.is_floating_point()):
        raise ValueError('Frozen candidate embedding table must have shape [vocabulary,64]')
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] != 2
            or ids.dtype != torch.long):
        raise ValueError('Candidate ids must be long [L,2]')
    stop = ids[:, 1].detach().cpu()
    if bool(((stop < 0) | (stop >= len(weights))).any()):
        raise ValueError('STOP token ID is outside pretrained embedding vocabulary')
    result = weights.detach().cpu()[stop].float().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Candidate embeddings must be finite')
    return result


def fit_normalization(rows, train_questions, field, width):
    """Population moments over real TRAIN tokens only, before any mutation.

    An absent train teacher gets an explicitly unfitted identity transform;
    validation/test teachers can never supply replacement fitting statistics.
    Moments are accumulated in float64 without concatenating the dataset.
    """
    allowed = set(train_questions)
    mean = torch.zeros(width, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    count = states = 0
    questions = set()
    for uid in sorted(rows):
        row = rows[uid]
        value = row.get(field)
        if row['question'] not in allowed or value is None:
            continue
        x = _float_tensor(value, (row['length'], width), field).double()
        n = len(x)
        delta = x.mean(0) - mean
        total = count + n
        m2 += ((x - x.mean(0)) ** 2).sum(0) + delta.square() * count * n / total
        mean += delta * n / total
        count = total
        states += 1
        questions.add(row['question'])
    std = (m2 / count).clamp_min(0).sqrt().clamp_min(1e-6) if count else torch.ones_like(mean)
    return dict(mean=mean.float().tolist(), std=std.float().tolist(), tokens=count,
                states=states, questions=sorted(questions), fit_split='train',
                fitted=bool(count), unbiased=False, epsilon=1e-6)


def apply_normalization(value, normalization):
    """Return a fresh float tensor; raw fields survive until fitting finishes."""
    if value is None:
        return None
    mean = torch.tensor(normalization['mean'], dtype=torch.float32)
    std = torch.tensor(normalization['std'], dtype=torch.float32)
    result = (value.detach().cpu().float() - mean) / std
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite normalized paired field')
    return result


def _validate_archive(archive):
    seen = set()
    for info in archive.infolist():
        name = info.orig_filename
        portable = PurePosixPath(name.replace('\\', '/'))
        windows = PureWindowsPath(name)
        if (portable.is_absolute() or windows.drive or windows.root
                or '\\' in name or ':' in name or '\x00' in name
                or any(p in ('', '.', '..') for p in name.rstrip('/').split('/'))
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError('Unsafe Phase0 archive path')
        key = portable.as_posix().casefold()
        if key in seen:
            raise ValueError('Duplicate archive member')
        seen.add(key)


def materialize_dynamics(phase0_input, cache_dir):
    """Copy only optional H1 weights, then load safely on CPU; never extract all.

    Missing weights return None even if an earlier invocation populated cache.
    A present invalid checkpoint fails rather than pretending it was absent.
    """
    root = source.resolve_phase0_input(phase0_input)
    provenance = dict(source=str(root), member=DYNAMICS_MEMBER, available=False,
                      frozen=True, retrained=False, LLM_forwards=0)
    with tempfile.TemporaryFile() as buffer:
        if root.is_dir():
            member = root / DYNAMICS_MEMBER
            if not member.resolve().is_relative_to(root):
                raise ValueError('Dynamics member escapes extracted Phase0 root')
            if not member.exists():
                return None, provenance
            if member.is_symlink() or not member.is_file():
                raise ValueError('Dynamics member must be a regular file')
            with member.open('rb') as stream:
                shutil.copyfileobj(stream, buffer)
        else:
            with zipfile.ZipFile(root) as archive:
                _validate_archive(archive)
                prefix = source._zip_prefix(root)
                name = prefix + DYNAMICS_MEMBER
                if name not in archive.namelist():
                    return None, provenance
                if archive.getinfo(name).is_dir():
                    raise ValueError('Dynamics member must be a regular file')
                with archive.open(name) as stream:
                    shutil.copyfileobj(stream, buffer)
        buffer.seek(0)
        weights = torch.load(buffer, map_location='cpu', weights_only=True)
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                       or not bool(torch.isfinite(v).all()) for k, v in weights.items())):
            raise ValueError('H1 dynamics checkpoint must be a finite tensor state dict')
        buffer.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: buffer.read(2**20), b''):
            digest.update(block)
        cache = Path(cache_dir).resolve()
        # Content-addressing avoids reusing stale optional weights across sources.
        target = cache / 'paired_dynamics' / digest.hexdigest() / 'best.pt'
        if not target.resolve().is_relative_to(cache):
            raise ValueError('Dynamics cache path escapes cache root')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            buffer.seek(0)
            try:
                shutil.copyfileobj(buffer, stream)
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    provenance.update(available=True, cache_path=str(target), sha256=digest.hexdigest(),
                      weight_hash=source.weight_hash(weights), horizon=1, latent_dim=LATENT_DIM)
    return weights, provenance


def _teacher_checksum(dataset, hidden_only=False):
    digest = hashlib.sha256()
    for uid, state in sorted(dataset.states.items()):
        digest.update((uid + '\0').encode())
        value = state.teacher_features
        if value is None:
            digest.update(b'None')
        else:
            value = value[:, 3:35] if hidden_only else value
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_paired_source(original_input, phase0_input, cache_dir,
                          verification_samples=32, device='cpu') -> dict:
    """Reuse the exact source checks and frozen latent train verification.

    Evaluation-only incoming-edge metadata uses the immediate parent. Ambiguous
    multiple parents cannot be represented by one UID row and are rejected.
    Returned normalizations are JSON-compatible population mean/std reports.
    """
    if (isinstance(verification_samples, bool) or not isinstance(verification_samples, int)
            or verification_samples < 0):
        raise ValueError('verification_samples must be a nonnegative integer')
    payload = source.prepare_source(original_input, phase0_input, cache_dir,
                                    verification_samples=verification_samples, device=device)
    source.split_check(payload['split'])
    dataset = load_dataset(original_input, max_questions=100, cache_dir=Path(cache_dir) / 'raw')
    if dataset_digest(dataset) != payload['provenance']['original_data_digest']:
        raise ValueError('Raw dataset changed after frozen-source verification')
    if set(dataset.states) != set(payload['rows']):
        raise ValueError('Paired dataset states differ from frozen-source rows')
    root = Path(payload['provenance']['source'])
    materialized = root if root.is_dir() else Path(cache_dir) / 'phase0'
    preprocessing_path = materialized / 'preprocessing.pt'
    if source.sha_file(preprocessing_path) != payload['provenance']['artifact_hashes']['preprocessing.pt']:
        raise ValueError('Frozen preprocessing changed after source verification')
    preprocessing = torch.load(preprocessing_path, map_location='cpu', weights_only=True)
    incoming = {}
    for parent, child, action in dataset.edges:
        # Raw/cache edges are R/E strings; enumerate_paths separately encodes
        # known rollout actions as 0/1. Keep evaluation metadata in string form.
        if not isinstance(action, str) or action not in ('R', 'E'):
            raise ValueError(f'Invalid raw evaluation action: {action!r}')
        if child in incoming:
            raise ValueError(f'Ambiguous evaluation parent for {child}')
        incoming[child] = (parent, action)
    rows = {}
    for uid, state in dataset.states.items():
        previous = payload['rows'][uid]
        row = {key: previous[key] for key in ('uid', 'question', 'length', 'accepted', 'ids', 'c', 'context')}
        row.update(action='root', parent_uid=None, parent_length=None,
                   parent_accepted=None, parent_token_changed=None)
        if uid in incoming:
            parent_uid, action = incoming[uid]
            parent = dataset.states[parent_uid]
            row.update(action=action, parent_uid=parent_uid, parent_length=parent.length,
                       parent_accepted=parent.accepted,
                       parent_token_changed=bool((parent.ids[:, 1] != state.ids[:parent.length, 1]).any()))
        row['teacher_hidden'] = project_teacher_hidden(state.teacher_features, state.length)
        row['candidate_embedding'] = lookup_candidate_embedding(preprocessing['embedding'], state.ids)
        rows[uid] = row
    # Both fits complete while every row still contains unnormalized values.
    normalization = {field: fit_normalization(rows, payload['split']['train'], field, width)
                     for field, width in (('teacher_hidden', TEACHER_INPUT_DIM),
                                          ('candidate_embedding', EMBEDDING_DIM))}
    teacher_states = {}
    teacher_tokens = {}
    observation_ids = {}
    coverage = {}
    for name, questions in payload['split'].items():
        selected = [r for r in rows.values() if r['question'] in set(questions)]
        teachers = [r for r in selected if r['teacher_hidden'] is not None]
        observed = [r for r in teachers if r['accepted'] is not None]
        teacher_states[name] = len(teachers)
        teacher_tokens[name] = sum(r['length'] for r in teachers)
        observation_ids[name] = sorted(r['uid'] for r in observed)
        coverage[name] = dict(states=len(selected), teacher_states=len(teachers),
                              teacher_tokens=teacher_tokens[name], observation_states=len(observed),
                              labeled_states=sum(r['accepted'] is not None for r in selected))
    teacher_checksum = _teacher_checksum(dataset)
    hidden_checksum = _teacher_checksum(dataset, hidden_only=True)
    for row in rows.values():
        for field, report in normalization.items():
            row[field] = apply_normalization(row[field], report)
    weights, dynamics_provenance = materialize_dynamics(root, cache_dir)
    provenance = dict(payload['provenance'])
    provenance.update(summary_marker='projected32_teacher_pilot', full_qwen_compression_claimed=False,
                      teacher_columns=[3, 35], teacher_input_dim=TEACHER_INPUT_DIM,
                      teacher_checksum=teacher_checksum, teacher_hidden_checksum=hidden_checksum,
                      teacher_checksum_format='sorted uid+NUL, shape/dtype, exact raw tensor bytes; None retained',
                      teacher_coverage=coverage, teacher_states_per_split=teacher_states,
                      teacher_tokens_per_split=teacher_tokens, teacher_selection='latest loader record',
                      candidate_identity='ids[:,1] same-forward STOP',
                      candidate_embedding_source='frozen preprocessing.pt pretrained embeddings',
                      normalization_fit='train questions only, real tokens, population moments',
                      D_input_fields=list(D_INPUT_FIELDS), parent_fields='evaluation only', LLM_forwards=0)
    payload.update(rows=rows, provenance=provenance, normalization=normalization,
                   observation_ids=observation_ids, teacher_input_dim=TEACHER_INPUT_DIM,
                   embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM,
                   data_audit=dict(dataset.audit, paired_teacher_coverage=coverage),
                   teacher_states_per_split=teacher_states, dynamics_weights=weights,
                   dynamics_provenance=dynamics_provenance,
                   summary_marker='projected32_teacher_pilot')
    return payload


def pack_paired_rows(rows, cachez, device):
    """Pack inputs plus explicitly masked targets; no evaluation parent reads.

    Zero teacher padding is storage only: ``teacher_valid`` is false there and
    for every missing teacher. Missing acceptance is -1, not a negative label.
    ``z_D`` is copied unchanged, including the source's frozen padding values.
    """
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot pack an empty paired batch')
    packed = {key: [] for key in ('z_D', 'c', 'context', 'teacher_hidden', 'candidate_embedding')}
    lengths, accepted, valid = [], [], []
    for row in rows:
        n = row['length']
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 64:
            raise ValueError('Paired row length must be an integer in [1,64]')
        label = row['accepted']
        if label is not None and (isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= n):
            raise ValueError('Observed acceptance must be an integer in [0,length] or None')
        packed['z_D'].append(_float_tensor(cachez[row['uid']], (64, LATENT_DIM), 'frozen z_D'))
        packed['c'].append(F.pad(_float_tensor(row['c'], (n, 20), 'c'), (0, 0, 0, 64 - n)))
        packed['context'].append(_float_tensor(row['context'], (8,), 'context'))
        candidate = _float_tensor(row['candidate_embedding'], (n, EMBEDDING_DIM), 'candidate_embedding')
        packed['candidate_embedding'].append(F.pad(candidate, (0, 0, 0, 64 - n)))
        teacher = row['teacher_hidden']
        mask = torch.zeros(64, dtype=torch.bool)
        if teacher is None:
            teacher = torch.zeros(n, TEACHER_INPUT_DIM)
        else:
            teacher = _float_tensor(teacher, (n, TEACHER_INPUT_DIM), 'teacher_hidden')
            mask[:n] = True
        packed['teacher_hidden'].append(F.pad(teacher, (0, 0, 0, 64 - n)))
        valid.append(mask)
        lengths.append(n)
        accepted.append(-1 if label is None else label)
    result = {key: torch.stack(values).to(device=device, dtype=torch.float32)
              for key, values in packed.items()}
    result.update(lengths=torch.tensor(lengths, device=device, dtype=torch.long),
                  accepted=torch.tensor(accepted, device=device, dtype=torch.long),
                  teacher_valid=torch.stack(valid).to(device))
    return result
