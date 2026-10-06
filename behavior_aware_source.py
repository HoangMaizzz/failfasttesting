"""Content-based Phase0 checkpoint discovery and frozen-source provenance."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile
import torch

from phase0_wm_data import (load_dataset, dataset_digest, NativeTargets, prepare_rows, pack_observations)
from phase0_wm_models import TokenEmbedding, LatentEncoder
from run_latent_wm_phase0 import select_paths, cpu_weights

REQUIRED = ('config.json', 'summary.json', 'split_manifest.json', 'study_manifest.json',
            'source_hashes.json', 'preprocessing.pt', 'latent_128/stage_A/best.pt',
            'latent_128/stage_A/complete.json', 'latent_128/frozen_latents.pt')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):h.update(block)
    return h.hexdigest()


def weight_hash(weights):
    h=hashlib.sha256()
    for name,tensor in sorted(weights.items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def split_check(split):
    if set(split)!= {'train','val','test'} or [len(split[k]) for k in ('train','val','test')]!=[70,15,15]:
        raise ValueError('Requires exact Phase0 100-question 70/15/15 split')
    ids=sum([split[k] for k in ('train','val','test')],[])
    if len(set(ids))!=100:raise ValueError('Split overlap or duplicate question IDs')


def _metadata_ok(read):
    try:
        cfg=json.loads(read('config.json'));summary=json.loads(read('summary.json'))
        split=json.loads(read('split_manifest.json'));split_check(split)
        return (summary.get('schema')=='latent_world_model_phase0_v1' and cfg.get('num_questions')==100
                and 128 in cfg.get('latent_dims',[]) and cfg.get('token_embedding')=='pretrained')
    except (ValueError, KeyError, OSError, zipfile.BadZipFile):return False


def _zip_prefix(path):
    try:
        with zipfile.ZipFile(path) as z:
            names=set(z.namelist());matches=[]
            for name in names:
                if name.endswith('summary.json'):
                    prefix=name[:-len('summary.json')]
                    if all(prefix+r in names for r in REQUIRED) and _metadata_ok(lambda r:z.read(prefix+r)):
                        matches.append(prefix)
            if len(matches)>1:raise ValueError('Multiple full Phase0 roots in ZIP')
            return matches[0] if matches else None
    except zipfile.BadZipFile:return None


def resolve_phase0_input(path):
    """Original Phase0 result only; arbitrary ZIP names or extracted folders."""
    path=Path(path).resolve()
    if not path.exists():raise FileNotFoundError(f'Phase0 input not mounted: {path}')
    candidates=[]
    if path.is_file():
        if _zip_prefix(path) is not None:return path
    else:
        roots=[path]+[p.parent for p in path.rglob('summary.json')]
        for root in set(roots):
            if all((root/r).is_file() for r in REQUIRED) and _metadata_ok(lambda r:(root/r).read_bytes()):
                candidates.append(root)
        for item in path.rglob('*'):
            if item.is_file() and item.suffix.lower() not in ('.npz','.pt','.json','.jsonl','.csv','.md'):
                if zipfile.is_zipfile(item) and _zip_prefix(item) is not None:candidates.append(item)
    candidates=sorted(set(candidates))
    if len(candidates)!=1:
        raise ValueError(f'Need exactly one full pretrained Phase0 100q/128 source; found {candidates}. '
                         'Upload Phase0 result ZIP, not only original experience/report.')
    return candidates[0]


def materialize_phase0(path, cache):
    path=resolve_phase0_input(path)
    if path.is_dir():return path
    prefix=_zip_prefix(path);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        seen=set()
        for info in z.infolist():
            portable=PurePosixPath(info.filename.replace('\\','/')); windows=PureWindowsPath(info.filename)
            normalized=portable.as_posix()
            if portable.is_absolute() or windows.drive or '..' in portable.parts or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe Phase0 archive path')
            if normalized in seen:raise ValueError('Duplicate archive member')
            seen.add(normalized)
        for member in REQUIRED:
            target=cache/member;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(prefix+member) as src,target.open('wb') as dst:
                for block in iter(lambda:src.read(2**20),b''):dst.write(block)
    return cache


def prepare_source(original_input, phase0_input, cache_dir, verification_samples=32, device='cpu'):
    root=materialize_phase0(phase0_input,Path(cache_dir)/'phase0')
    cfg=json.loads((root/'config.json').read_text());split=json.loads((root/'split_manifest.json').read_text())
    split_check(split)
    hashes=json.loads((root/'source_hashes.json').read_text())
    # Do not use an accidentally modified dynamics/context implementation.
    for name in ('phase0_wm_models.py','phase0_wm_data.py','factorized_wm_data.py'):
        raw=(Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(raw).hexdigest()!=hashes[name]:
            raise ValueError(f'Phase0 source implementation hash mismatch: {name}')
    dataset=load_dataset(original_input,max_questions=100,cache_dir=Path(cache_dir)/'raw')
    if sorted(dataset.question_ids)!=sorted(sum(split.values(),[])):raise ValueError('Original input question IDs differ')
    digest=dataset_digest(dataset)
    expected=hashlib.sha256(json.dumps(dict(config=cfg,source_hashes=hashes,data=digest),sort_keys=True).encode()).hexdigest()
    manifest=json.loads((root/'study_manifest.json').read_text())
    if expected!=manifest['fingerprint']:raise ValueError('Original raw trace does not match Phase0 fingerprint')
    preprocessing=torch.load(root/'preprocessing.pt',map_location='cpu',weights_only=True)
    frozen=torch.load(root/'latent_128/stage_A/best.pt',map_location='cpu',weights_only=True)
    cachez=torch.load(root/'latent_128/frozen_latents.pt',map_location='cpu',weights_only=True)
    if set(cachez)!=set(dataset.states):raise ValueError('Incomplete/mismatched frozen latent cache')
    targets=NativeTargets(**preprocessing['native_targets'])
    rows=prepare_rows(dataset,targets)
    encoder=LatentEncoder(TokenEmbedding(**preprocessing['embedding']),128,cfg['encoder_layers'],cfg['dropout'])
    encoder.load_state_dict(frozen['encoder']);encoder.to(device).eval().requires_grad_(False)
    uids=sorted(u for u,r in rows.items() if r['question'] in set(split['train']))[:verification_samples]
    with torch.no_grad():
        for start in range(0,len(uids),16):
            group=[rows[u] for u in uids[start:start+16]]
            z=encoder(pack_observations(group,device,cfg['prefix_max_tokens'])).z.cpu()
            for i,r in enumerate(group):
                if cachez[r['uid']].shape!=(64,128) or not torch.allclose(z[i],cachez[r['uid']].float(),atol=.006,rtol=.004):
                    raise ValueError('Frozen latent cache differs from source encoder')
    encoder.cpu()
    # Raw native fields are targets only. Workers need no prefix/token encoder,
    # embedding weights, or raw 1536D hidden storage after this preparation.
    keep={'uid','question','length','c','context','accepted','ids','native_target','native_mask'}
    rows={u:{k:v for k,v in r.items() if k in keep} for u,r in rows.items()}
    paths={k:select_paths(dataset,q,include_contradictions=k!='train') for k,q in split.items()}
    provenance=dict(source=str(resolve_phase0_input(phase0_input)),fingerprint=expected,
        artifact_hashes={r:sha_file(root/r) for r in REQUIRED},
        encoder_weight_hash=weight_hash(frozen['encoder']),verifier_weight_hash=weight_hash(frozen['readout']),
        native_reconstruction_weight_hash=weight_hash(frozen['reconstruction']),
        checked_encoder_train_samples=len(uids),test_encoder_verifications=0,
        original_data_digest=digest,source_code_hashes=hashes,
        latent_dim=128,encoder_retrained=False,LLM_forwards=0)
    return dict(source_config=cfg,split=split,rows=rows,cachez=cachez,paths=paths,
                frozen={k:v for k,v in frozen.items() if k!='encoder'},provenance=provenance)
