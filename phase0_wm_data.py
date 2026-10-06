"""Original trace adapter and fixed native reconstruction targets for Phase 0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F

from factorized_wm_data import load_dataset, enumerate_paths, split_questions
from phase0_wm_models import ObservationBatch, LatentState


EMBEDDING_REPO = 'Efficient-Large-Model/Fast_dLLM_v2_1.5B'


def training_token_ids(dataset, questions):
    allowed = set(questions)
    ids = {151665}
    for state in dataset.states.values():
        if state.question in allowed:
            ids.update(state.ids.flatten().tolist())
            ids.update(state.topk_ids.flatten().tolist())
            ids.update(state.prefix_ids.tolist())
    return sorted(ids)


def prepare_embedding(dataset, questions, cfg, cache_dir, device):
    """Extract ONLY native input embeddings. Never instantiate the language model."""
    vocabulary = training_token_ids(dataset, questions)
    dim = cfg['embedding_dim']
    if cfg['token_embedding'] == 'learned':
        generator = torch.Generator().manual_seed(cfg['seed'])
        weights = torch.randn(len(vocabulary) + 1, dim, generator=generator) * .02
        return dict(weights=weights, vocabulary=vocabulary, learned=True), dict(
            mode='learned_train_vocabulary', pretrained=False, token_decoder=False,
            vocabulary=len(vocabulary), OOV_behavior='UNK input embedding', projection_fit='none')
    if cfg['token_embedding'] != 'pretrained':
        raise ValueError('token_embedding must be pretrained or explicitly learned')
    from safetensors import safe_open
    filename = cfg.get('embedding_path')
    if filename:
        filename = Path(filename)
        if filename.is_dir():
            filename = filename / 'model.safetensors'
        if not filename.is_file():
            raise FileNotFoundError(f'Embedding checkpoint not found: {filename}')
    else:
        from huggingface_hub import hf_hub_download
        filename = Path(hf_hub_download(cfg.get('embedding_repo_id', EMBEDDING_REPO),
            'model.safetensors', revision=cfg['embedding_revision'], cache_dir=str(cache_dir)))
    with safe_open(str(filename), framework='pt', device='cpu') as archive:
        keys = archive.keys()
        key = next((key for key in ('model.embed_tokens.weight', 'embed_tokens.weight',
                    'transformer.wte.weight') if key in keys), None)
        if key is None:
            raise ValueError('Checkpoint has no native input embedding tensor')
        table = archive.get_slice(key)
        size, native_dim = table.get_shape()
        if max(vocabulary) >= size:
            raise ValueError('Native embedding vocabulary does not cover trace token IDs')
        # PCA uses IDs observed in TRAIN questions only. Pretrained table itself
        # may embed every tokenizer ID, including held-out IDs without fitting.
        selected = random.Random(cfg['seed']).sample(vocabulary, min(len(vocabulary), cfg['embedding_fit_tokens']))
        selected.sort()
        raw = torch.stack([table[t:t + 1][0].float() for t in selected]).to(device)
        mean = raw.mean(0)
        centered = raw - mean
        torch.manual_seed(cfg['seed'])
        _, _, basis = torch.pca_lowrank(centered, q=min(dim, min(centered.shape)), center=False, niter=3)
        if basis.shape[1] < dim:
            basis = F.pad(basis, (0, dim - basis.shape[1]))
        train_projected = centered @ basis
        std = train_projected.std(0, unbiased=False).clamp_min(.01)
        projected = torch.empty(size, dim, dtype=torch.float16)
        for start in range(0, size, 4096):
            block = table[start:start + 4096].float().to(device)
            projected[start:start + len(block)] = ((block - mean) @ basis / std).clamp(-10, 10).half().cpu()
    digest = hashlib.sha256(projected.numpy().tobytes()).hexdigest()
    report = dict(mode='frozen_native_embedding_train_PCA', pretrained=True,
        repo_id=cfg.get('embedding_repo_id', EMBEDDING_REPO), revision=cfg['embedding_revision'],
        local_path=str(filename), tensor=key, vocabulary=size, native_dim=native_dim,
        projection_dim=dim, projection_fit='train token IDs only', fit_tokens=len(selected),
        projected_table_sha256=digest, LLM_forwards=0)
    return dict(weights=projected, vocabulary=None, learned=False), report


class NativeTargets:
    """Fixed PCA coordinates fitted before encoder training, plus observed D fields."""
    def __init__(self, mean, basis, std):
        self.mean, self.basis, self.std = mean.cpu(), basis.cpu(), std.cpu()

    @classmethod
    def fit(cls, dataset, questions, dim=16, max_rows=2048, device='cpu', seed=42):
        groups = {}
        allowed = set(questions)
        for state in dataset.states.values():
            if state.question in allowed and (state.scalars[:, 3] > 0).any():
                groups.setdefault(state.question, []).append(state)
        if not groups:
            raise ValueError('No train hidden rows for fixed reconstruction targets')
        rng = random.Random(seed)
        raw = []
        qids = sorted(groups)
        for _ in range(max_rows):
            state = rng.choice(groups[rng.choice(qids)])
            positions = torch.where(state.scalars[:, 3] > 0)[0].tolist()
            raw.append(state.hidden[rng.choice(positions)])
        raw = torch.stack(raw).float().to(device)
        means, bases, stds = [], [], []
        for layer in range(3):
            x = raw[:, layer]
            mean = x.mean(0)
            torch.manual_seed(seed + layer)
            _, _, basis = torch.pca_lowrank(x - mean, q=min(dim, min(x.shape)), center=False, niter=3)
            if basis.shape[1] < dim:
                basis = F.pad(basis, (0, dim - basis.shape[1]))
            std = ((x - mean) @ basis).std(0, unbiased=False).clamp_min(.05)
            means.append(mean); bases.append(basis); stds.append(std)
        return cls(torch.stack(means), torch.stack(bases), torch.stack(stds))

    @property
    def width(self):
        return 3 * self.basis.shape[-1] + 32 + 20

    def target(self, state):
        raw = state.hidden.float()
        h = torch.stack([((raw[:, k] - self.mean[k]) @ self.basis[k] / self.std[k]).clamp(-10, 10)
                         for k in range(3)], 1).flatten(1)
        c = torch.cat([state.scalars, state.surface[:, 32:]], -1)
        target = torch.cat([h, state.gaps.clamp(-40, 0) / 10, c.clamp(-8, 8)], -1)
        mask = torch.ones_like(target, dtype=torch.bool)
        hd = h.shape[-1]
        mask[:, :hd] = state.scalars[:, 3:4] > 0
        mask[:, hd:hd + 32] = state.scalars[:, 4:5] > 0
        mask[:, hd + 32 + 7] = state.scalars[:, 8] > 0
        return target, mask


def prepare_rows(dataset, targets):
    rows = {}
    for uid, state in dataset.states.items():
        native_target, native_mask = targets.target(state)
        rows[uid] = dict(uid=uid, question=state.question, length=state.length,
            hidden=state.hidden, ids=state.ids, topk_ids=state.topk_ids, gaps=state.gaps,
            c=torch.cat([state.scalars, state.surface[:, 32:]], -1), context=state.context,
            prefix_ids=state.prefix_ids, accepted=state.accepted, teacher=state.teacher_features,
            native_target=native_target, native_mask=native_mask)
    return rows


def pack_observations(rows, device, prefix_max_tokens=4096):
    if any(len(r['prefix_ids']) > prefix_max_tokens for r in rows):
        raise ValueError('Trace prefix exceeds prefix_max_tokens; no silent truncation')
    pwidth = max(1, max(len(r['prefix_ids']) for r in rows))
    def token_field(name):
        x = [r[name] for r in rows]
        return torch.stack([F.pad(t, (0, 0) * (t.ndim - 1) + (0, 64 - len(t))) for t in x]).to(device)
    return ObservationBatch(hidden=token_field('hidden'), ids=token_field('ids'),
        topk_ids=token_field('topk_ids'), gaps=token_field('gaps'), c=token_field('c'),
        context=torch.stack([r['context'] for r in rows]).to(device),
        prefix_ids=torch.stack([F.pad(r['prefix_ids'], (0, pwidth - len(r['prefix_ids']))) for r in rows]).to(device),
        prefix_lengths=torch.tensor([len(r['prefix_ids']) for r in rows], device=device),
        lengths=torch.tensor([r['length'] for r in rows], device=device))


def pack_latents(rows, cache, device):
    return LatentState(torch.stack([cache[r['uid']] for r in rows]).to(device).float(),
        torch.tensor([r['length'] for r in rows], device=device),
        torch.stack([F.pad(r['c'], (0, 0, 0, 64 - r['length'])) for r in rows]).to(device),
        torch.stack([r['context'] for r in rows]).to(device))


def pack_native_targets(rows, device):
    targets = torch.stack([F.pad(r['native_target'], (0, 0, 0, 64 - r['length'])) for r in rows]).to(device)
    mask = torch.stack([F.pad(r['native_mask'], (0, 0, 0, 64 - r['length'])) for r in rows]).to(device)
    return targets, mask


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for uid, state in sorted(dataset.states.items()):
        digest.update(json.dumps([uid, state.question, state.round_id, state.accepted]).encode())
        for key in ('hidden', 'ids', 'topk_ids', 'surface', 'scalars', 'context', 'prefix_ids',
                    'teacher_features', 'teacher_margin'):
            value = getattr(state, key)
            digest.update(key.encode())
            digest.update(b'None' if value is None else value.numpy().tobytes())
    digest.update(json.dumps(dataset.edges).encode())
    return digest.hexdigest()
