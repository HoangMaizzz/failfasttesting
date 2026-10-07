"""Exact saved-state reconstruction and lossless, mmap-backed native captures.

Old projected Qwen teachers are never loaded as features. Only frozen drafter
latents are reused; Qwen hidden and its candidate embeddings come from a new
frozen real-verifier forward. Parent acceptance is evaluation metadata only.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

import behavior_aware_source as phase0
from factorized_wm_data import _resolve_run, _Reader
from factorized_wm_metrics import write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verifier_identity(config, summary):
    name = config.get('target_model_name')
    runtime = summary.get('runtime', {})
    revision = runtime.get('target_revision')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Saved target_model_name is missing; refusing to assume a different verifier')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Saved immutable Qwen target_revision is missing; no silent main/latest fallback')
    if config.get('target_quantization', 'none') not in ('none', None):
        raise ValueError('This native FP16 study cannot silently replace quantized historical labels')
    return dict(model_id=name, revision=revision, dtype='float16', quantization='none',
                historical_transformers=runtime.get('transformers'),
                original_mode=summary.get('verifier'), identity_source='original config + summary.runtime')


def prepare_native_source(original_input, phase0_input, cache_dir, verification_samples=16, device='cpu'):
    payload = phase0.prepare_source(original_input, phase0_input, cache_dir,
                                   verification_samples=verification_samples, device=device)
    run = _resolve_run(original_input)
    with _Reader(run) as reader:
        config, summary = reader.json('config.json'), reader.json('summary.json')
        metadata = {r['state_id']: r for r in reader.lines('states.jsonl')}
        incoming = {}
        for edge in reader.lines('edges.jsonl'):
            child = edge['child']
            if child in incoming:
                raise ValueError(f'Ambiguous parent for {child}')
            incoming[child] = (edge['parent'], edge['action'])
    identity = verifier_identity(config, summary)
    if set(metadata) != set(payload['rows']):
        raise ValueError('Raw metadata and verified frozen drafter source differ')
    rows = {}
    for uid, old in payload['rows'].items():
        meta = metadata[uid]
        prefix = meta.get('prefix_token_ids')
        if not isinstance(prefix, list) or not prefix or any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError(f'Invalid concrete verifier prefix for {uid}')
        n = old['length']
        candidate = old['ids'][:, 1].tolist()
        if not 1 <= n <= 64 or len(candidate) != n:
            raise ValueError('No proposal truncation is permitted')
        parent_uid, action = incoming.get(uid, (None, 'root'))
        if action not in ('root', 'R', 'E'):
            raise ValueError('Unsupported native action')
        parent = None if parent_uid is None else payload['rows'][parent_uid]
        if parent is not None and parent['question'] != old['question']:
            raise ValueError('Native edge crosses questions')
        parent_length = None if parent is None else parent['length']
        parent_K = None if parent is None else parent['accepted']
        segment = int(meta.get('segment_start', 0))
        if not 0 <= segment < n:
            raise ValueError('Invalid known native segment boundary')
        if action == 'E' and (parent_length is None or not parent_length < n or segment != parent_length):
            raise ValueError('Extend metadata disagrees with native segment geometry')
        rows[uid] = dict(uid=uid, question=old['question'], round_id=int(meta['round_id']),
            prefix=prefix, candidate=candidate, length=n, accepted=old['accepted'],
            action=action, parent_uid=parent_uid, parent_length=parent_length, parent_K=parent_K,
            segment_start=segment, c=old['c'], context=old['context'],
            is_parent_full_prefix=parent_K is not None and parent_K == parent_length)
    for ids in payload['split'].values():
        if not set(ids) <= {r['question'] for r in rows.values()}:
            raise ValueError('Split contains questions absent from reconstructed states')
    return dict(rows=rows, cachez=payload['cachez'], split=payload['split'],
        provenance=payload['provenance'], identity=identity,
        historical_config=config, historical_summary=summary)


def select_capture_uids(rows, states_per_question=0, seed=42):
    """Full coverage, or a label-blind action-diverse pipeline smoke subset."""
    if type(states_per_question) is not int or states_per_question < 0:
        raise ValueError('capture_states_per_question must be a nonnegative integer')
    groups = defaultdict(lambda: defaultdict(list))
    for uid, row in sorted(rows.items()):
        if row['accepted'] is not None:
            groups[row['question']][row['action']].append(uid)
    selected = []
    for q, actions in sorted(groups.items()):
        buckets = {a: sorted(u, key=lambda x: digest([seed, x])) for a, u in actions.items()}
        if not states_per_question:
            selected.extend(u for bucket in buckets.values() for u in bucket)
            continue
        # Smoke must not always select root/R and accidentally contain no E.
        count = 0
        while count < states_per_question and any(buckets.values()):
            for action in ('E', 'R', 'root'):
                if count >= states_per_question:
                    break
                if buckets.get(action):
                    selected.append(buckets[action].pop(0)); count += 1
    return sorted(selected)


def structural_features(row):
    n, start = row['length'], row['segment_start']
    p = torch.arange(n).float()
    is_new = p >= start
    rel = torch.where(is_new, (p-start)/max(1, n-start-1), torch.zeros_like(p))
    return torch.stack([p/max(1, n-1), torch.full_like(p, n/64), is_new.float(), rel], -1)


class NativeHiddenStore:
    """One FP16 NPY mmap per depth; no random projection/PCA or lossy codec.

    A progress record is committed only after the hidden pages are flushed.
    Missing/failed captures never become zeros masquerading as observations.
    """
    def __init__(self, output, rows, uids, depths, hidden_dim, signature, readonly=False):
        self.root = Path(output).resolve(); self.rows = rows
        self.uids = list(uids); self.depths = list(depths); self.hidden_dim = hidden_dim
        self.signature = signature; self.readonly = readonly
        self.offsets = {}; total = 0
        for uid in uids:
            if ('accepted' in rows[uid] and (type(rows[uid]['accepted']) is not int or
                    not 0 <= rows[uid]['accepted'] <= rows[uid]['length'])):
                raise ValueError('Known saved acceptance must be a valid integer, not an unlabeled placeholder')
            if 'accepted' not in rows[uid] and not readonly:
                raise ValueError('Writable captures require saved verifier labels')
            self.offsets[uid] = (total, total + rows[uid]['length']); total += rows[uid]['length']
        self.total_positions = total
        manifest = dict(schema='native_qwen_capture_v1', signature=signature,
            uids=list(uids), depths=list(depths), hidden_dim=hidden_dim, positions=total,
            offsets={k:list(v) for k,v in self.offsets.items()}, dtype='float16',
            normalization='none', projection='none', compression='lossless NPY')
        path = self.root / 'capture_manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Capture identity/input/subset/geometry signature differs')
        elif readonly:
            raise FileNotFoundError('Native capture_manifest.json is missing')
        else:
            self.root.mkdir(parents=True, exist_ok=True); write_json(path, manifest)
        self.arrays = {}
        for slot, depth in zip(('layer25','layer50','layer75','layer100'), depths):
            file = self.root / 'native_hidden' / slot / 'hidden.npy'
            if file.exists():
                array = np.load(file, mmap_mode='r' if readonly else 'r+', allow_pickle=False)
                if array.shape != (total, hidden_dim) or array.dtype != np.float16:
                    raise ValueError('Native hidden file has incompatible shape/dtype')
            elif readonly:
                raise FileNotFoundError(f'Missing raw native layer file: {slot}')
            else:
                file.parent.mkdir(parents=True, exist_ok=True)
                array = np.lib.format.open_memmap(file, mode='w+', dtype=np.float16,
                                                 shape=(total, hidden_dim))
            self.arrays[depth] = array
        progress = self.root / 'capture_progress.json'
        self.progress = json.loads(progress.read_text()) if progress.exists() else {}
        if set(self.progress) - set(uids):
            raise ValueError('Capture progress contains foreign states')
        for uid, rec in self.progress.items():
            known_K = rows[uid].get('accepted')
            if rec.get('signature') != signature or (known_K is not None and rec.get('saved_K') != known_K):
                raise ValueError('Capture progress is not bound to the saved verifier labels')
            if known_K is None and not readonly:
                raise ValueError('Writable captures require saved verifier labels')
            saved, rerun = rec.get('saved_K'), rec.get('rerun_K')
            if (type(saved) is not int or type(rerun) is not int or
                    not 0 <= saved <= rows[uid]['length'] or not 0 <= rerun <= rows[uid]['length'] or
                    type(rec.get('matches')) is not bool or rec['matches'] != (saved == rerun) or
                    not rec.get('alignment', {}).get('passed') is True):
                raise ValueError('Replayed reproduction or causal alignment record is inconsistent')

    def record(self, uid, result):
        if self.readonly:
            raise RuntimeError('Native hidden store is read only')
        row = self.rows[uid]; n = row['length']
        if type(result['K']) is not int or not 0 <= result['K'] <= n:
            raise ValueError('Rerun verifier acceptance is outside proposal geometry')
        if not result['alignment']['passed']:
            raise ValueError(f'Causal hidden-to-logit alignment failed for {uid}')
        if set(result['hidden']) != set(self.depths):
            raise ValueError('Capture omitted a native layer')
        start, end = self.offsets[uid]
        hasher = hashlib.sha256()
        for depth in self.depths:
            h = result['hidden'][depth].detach().cpu()
            if tuple(h.shape) != (n, self.hidden_dim) or not bool(torch.isfinite(h).all()):
                raise ValueError('Raw native hidden is incomplete/nonfinite')
            value = h.half().numpy()
            self.arrays[depth][start:end] = value
            self.arrays[depth].flush(); hasher.update(value.tobytes())
        self.progress[uid] = dict(signature=self.signature, saved_K=row['accepted'],
            rerun_K=int(result['K']), matches=int(result['K']) == row['accepted'],
            alignment=result['alignment'], raw_hidden_sha256=hasher.hexdigest())
        # Avoid rewriting a growing 10k-state manifest on every forward.
        if len(self.progress)%64==0 or len(self.progress)==1:
            self.flush_progress()

    def flush_progress(self):
        if not self.readonly:
            write_json(self.root / 'capture_progress.json',self.progress)

    def verify_saved_rows(self):
        for uid, rec in self.progress.items():
            begin,end=self.offsets[uid]; hasher=hashlib.sha256()
            for depth in self.depths:
                x=np.asarray(self.arrays[depth][begin:end])
                if not np.isfinite(x).all():
                    raise ValueError('Cached native hidden became nonfinite')
                hasher.update(x.tobytes())
            if hasher.hexdigest()!=rec['raw_hidden_sha256']:
                raise ValueError(f'Native hidden checksum failed for {uid}')

    def qualified(self):
        return sorted(uid for uid, rec in self.progress.items() if rec['matches'])

    def close(self):
        self.flush_progress()
        for array in self.arrays.values():
            if not self.readonly:
                array.flush()
            if hasattr(array,'_mmap'):
                array._mmap.close()
        self.arrays.clear()


def pack_native_batch(uids, spec, rows, cachez, store, candidate_table, device):
    group=[rows[u] for u in uids]
    lengths=torch.tensor([r['length'] for r in group], dtype=torch.long, device=device)
    accepted=torch.tensor([r['accepted'] for r in group], dtype=torch.long, device=device)
    if bool((accepted < 0).any()):
        raise ValueError('Unlabeled captures cannot be supervised as rejection')
    batch=dict(lengths=lengths, accepted=accepted)
    if spec['kind']=='direct':
        # No teacher hidden, own-Qwen candidate embedding, labels or parent K is
        # passed into the Direct predictor's forward interface.
        z=[]
        for u,r in zip(uids,group):
            latent=cachez[u].float()
            if tuple(latent.shape) not in ((r['length'],128),(64,128)):
                raise ValueError('Frozen direct D latent must be tokenwise [L,128] or source-padded [64,128]')
            z.append(F.pad(latent[:r['length']],(0,0,0,64-r['length'])))
        batch.update(z_D=torch.stack(z).to(device),
            c=torch.stack([F.pad(r['c'],(0,0,0,64-r['length'])) for r in group]).to(device),
            context=torch.stack([r['context'] for r in group]).to(device))
        return batch
    depth=spec['depth']; h=[]; embeddings=[]; structural=[]
    for uid,row in zip(uids,group):
        if uid not in store.progress or not store.progress[uid]['matches']:
            raise ValueError('An unavailable native capture entered a probe batch')
        begin,end=store.offsets[uid]; n=row['length']
        hidden=torch.tensor(np.array(store.arrays[depth][begin:end],copy=True),dtype=torch.float32)
        h.append(F.pad(hidden,(0,0,0,64-n)))
        ids=[candidate_table['index'][t] for t in row['candidate']]
        e=torch.tensor(np.array(candidate_table['values'][ids],copy=True),dtype=torch.float32)
        embeddings.append(F.pad(e,(0,0,0,64-n)))
        structural.append(F.pad(structural_features(row),(0,0,0,64-n)))
    batch.update(hidden=torch.stack(h).to(device),
                 candidate_embedding=torch.stack(embeddings).to(device),
                 structural=torch.stack(structural).to(device))
    return batch


def load_candidate_table(root):
    root=Path(root)
    ids=json.loads((root/'candidate_embedding_ids.json').read_text())
    if len(set(ids))!=len(ids) or any(type(t) is not int or t<0 for t in ids):
        raise ValueError('Candidate embedding lookup has duplicate/invalid token IDs')
    values=np.load(root/'candidate_embeddings.npy',mmap_mode='r',allow_pickle=False)
    if values.ndim!=2 or len(values)!=len(ids) or values.dtype!=np.float16 or not np.isfinite(values).all():
        raise ValueError('Frozen own-Qwen embedding subset is malformed')
    return dict(index={t:i for i,t in enumerate(ids)},values=values)
