"""Frozen-checkpoint feature sensitivity; never imports or runs either LLM.

Reads an existing training ZIP or extracted directory. Labels remain targets.
Default: baseline on all questions, ablations ONLY on held-out questions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from world_model_core import Observation, pack_observations, expected_acceptance, prefix_log_distribution
from world_model_probe import ProbeWorldModel


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        names = self.zip.namelist() if self.zip else [str(p.relative_to(path)).replace('\\', '/') for p in self.path.rglob('*') if p.is_file()]
        roots = [n[:-len('checkpoint.pt')] for n in names if n.endswith('checkpoint.pt')]
        if len(roots) != 1:
            raise ValueError('Expected one checkpoint.pt in the input run')
        self.root = roots[0]

    def read(self, name):
        return self.zip.read(self.root + name) if self.zip else (self.path / self.root / name).read_bytes()

    def lines(self, name):
        try:
            return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]
        except (KeyError, FileNotFoundError):
            return []


def load_observations(archive, metadata, labels):
    grouped = defaultdict(list)
    for row in metadata:
        grouped[row['shard']].append(row)
    result = {}
    for shard, rows in grouped.items():
        with np.load(io.BytesIO(archive.read(shard)), allow_pickle=False) as source:
            # Load each NPZ member ONCE, not once per state (large hidden arrays).
            arrays = {k: source[k] for k in ('ids', 'hidden', 'gaps', 'scalars', 'context',
                      'offsets', 'aligned_topk_token_ids', 'history')}
        for row in rows:
            i = row['row']; start, end = arrays['offsets'][i:i+2]
            def tensor(key):
                return torch.from_numpy(arrays[key][start:end].copy())
            label = labels.get(row['state_id'], {})
            accepted = label.get('accepted_len') if label.get('label_valid') else None
            if accepted is not None and not 0 <= accepted <= end-start:
                raise ValueError('Invalid acceptance label')
            result[row['state_id']] = Observation(
                row['state_id'], row['question'], row['round_id'], tensor('ids').long(),
                tensor('hidden'), tensor('gaps'), tensor('scalars'),
                torch.from_numpy(arrays['context'][i].copy()), accepted,
                torch.tensor(row['prefix_token_ids'], dtype=torch.long),
                tensor('aligned_topk_token_ids').long(), tensor('history'))
    return result


def perturb(batch, group):
    """Input-zero sensitivity, not a retrained ablation or causal importance.

Keep lengths, legality, mask/frontier and validity flags factual in every arm.
"""
    b = dict(batch)
    def zero(key):
        b[key] = torch.zeros_like(batch[key])
    if group == 'full':
        return b
    if group == 'hidden_all':
        zero('hidden')
    elif group.startswith('hidden_layer_'):
        index = int(group.rsplit('_', 1)[1])
        b['hidden'] = batch['hidden'].clone(); b['hidden'][:, :, index] = 0
    elif group in ('token_native', 'token_stop'):
        b['token_vectors'] = batch['token_vectors'].clone()
        b['token_vectors'][:, :, int(group == 'token_stop')] = 0
    elif group == 'topk':
        zero('gaps'); zero('topk_vectors')
    elif group == 'prefix_content':
        zero('prefix_vectors')  # preserve prefix length and positional encoding
    elif group == 'history':
        zero('history')
    elif group in ('confidence', 'position_age', 'mask_encoder'):
        b['scalars'] = batch['scalars'].clone()
        cols = {'confidence': [7], 'position_age': [2, 5, 6, 9, 10, 11, 12],
                'mask_encoder': [0, 1, 14]}[group]
        b['scalars'][:, :, cols] = 0
    elif group != 'encoder_context':
        raise ValueError(group)
    return b


def encode(model, batch, group):
    handle = None
    if group == 'encoder_context':
        # Do not corrupt context used as factual metadata by transition/carry.
        handle = model.encoder.context.register_forward_hook(lambda m, args, out: torch.zeros_like(out))
    try:
        z = model.encoder(perturb(batch, group))
        if group == 'mask_encoder':
            # Hide mask signal from learned encoder, NOT from known dynamics rules.
            pos = torch.arange(z.tokens.shape[1], device=z.tokens.device)[None]
            z.mask_probs = batch['scalars'][:, :, 0] * (pos < z.lengths[:, None])
            first = torch.where(z.mask_probs.bool(), pos, z.tokens.shape[1]).min(-1).values
            z.frontier = torch.minimum(first, z.lengths)
        return z
    finally:
        if handle is not None:
            handle.remove()


def metric(rows):
    error = np.asarray([r['expected_K']-r['K'] for r in rows]); absolute = abs(error)
    result = dict(n=len(rows), mae=float(absolute.mean()), rmse=float(np.sqrt((error**2).mean())),
        median=float(np.median(absolute)), p90=float(np.quantile(absolute, .9)),
        p95=float(np.quantile(absolute, .95)), bias=float(error.mean()),
        within_1=float((absolute<=1).mean()), within_2=float((absolute<=2).mean()),
        within_4=float((absolute<=4).mean()), over_4=float((error>4).mean()),
        under_4=float((error < -4).mean()), nll=float(np.mean([r['nll'] for r in rows])))
    for name in ('persistence_error', 'latent_global_cosine', 'latent_token_mse', 'active_mask_brier'):
        values = [r[name] for r in rows if name in r]
        if values:
            result[name] = float(np.mean(values))
    return result


def record(model, z, index, observation, meta, group, source, depth, actions, baseline=None, truth=None):
    if observation.accepted is None:
        return None
    logits = model.acceptance(z)[index:index+1]
    lengths = z.lengths[index:index+1]
    if int(lengths[0]) != observation.length:
        raise ValueError('Rollout length does not match real child; do not score misaligned states')
    lp = prefix_log_distribution(logits, lengths)
    row = dict(variant=group, split=meta['split'], question=observation.question,
        source=source, state_id=observation.uid, depth=depth, actions=actions,
        group='current' if depth == 0 else f'h{depth}_{actions[-1]}',
        length=observation.length, K=observation.accepted,
        expected_K=float(expected_acceptance(logits, lengths)[0]),
        nll=float(-lp[0, observation.accepted]), mode_K=int(lp.argmax(-1)[0]))
    if baseline is not None:
        row['persistence_error'] = abs(baseline-observation.accepted)
    if truth is not None:
        row['latent_global_cosine'] = float(F.cosine_similarity(z.global_state[index:index+1], truth.global_state)[0])
        row['latent_token_mse'] = float((z.tokens[index, :observation.length]-truth.tokens[0, :observation.length]).square().mean())
        start = int(round(float(observation.context[2])*64))
        if start < observation.length:
            actual = observation.scalars[start:, 0].to(z.tokens.device)
            row['active_mask_brier'] = float((z.mask_probs[index, start:observation.length]-actual).square().mean())
    return row


def make_report(rows, bootstrap=1000):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], r['variant'], r['group'])].append(r)
        if r['depth']:
            groups[(r['split'], r['variant'], f"h{r['depth']}_all")].append(r)
    output = {}; rng = np.random.default_rng(42)
    def key(r):
        return r['source'], r['state_id'], r['depth'], r['actions']
    for (split, variant, group), values in sorted(groups.items()):
        stats = metric(values)
        per_q = defaultdict(list)
        for r in values:
            per_q[r['question']].append(abs(r['expected_K']-r['K']))
        stats['question_macro_mae'] = float(np.mean([np.mean(v) for v in per_q.values()]))
        stats['questions'] = len(per_q)
        if variant != 'full':
            reference = {key(r): r for r in groups[(split, 'full', group)]}
            paired = [(r, reference[key(r)]) for r in values]
            base = metric([b for _, b in paired])
            stats['delta_mae'] = stats['mae']-base['mae']
            stats['delta_p90'] = stats['p90']-base['p90']
            stats['delta_within_2'] = stats['within_2']-base['within_2']
            differences = defaultdict(list)
            for r, b in paired:
                differences[r['question']].append(abs(r['expected_K']-r['K'])-abs(b['expected_K']-b['K']))
            d = np.asarray([np.mean(v) for v in differences.values()])
            stats['delta_question_macro_mae'] = float(d.mean())
            draws = d[rng.integers(0, len(d), size=(bootstrap, len(d)))].mean(1)
            stats['delta_question_macro_mae_ci95'] = np.quantile(draws, [.025, .975]).tolist()
        output['/'.join((split, variant, group))] = stats
    return output


def load_embeddings(path, repo, revision, cache):
    """Read only embedding tensor; never construct the drafter model."""
    from safetensors import safe_open
    provenance = dict(repo=repo, requested_revision=revision)
    if path:
        candidates = [Path(path)]
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().model_info(repo, revision=revision)
        provenance['resolved_revision'] = info.sha
        names = [s.rfilename for s in info.siblings]
        index = 'model.safetensors.index.json'
        if index in names:
            p = hf_hub_download(repo, index, revision=info.sha, cache_dir=cache)
            weights = json.loads(Path(p).read_text())['weight_map']
            matches = [v for k, v in weights.items() if k.endswith('embed_tokens.weight')]
            if len(matches) != 1:
                raise ValueError('Cannot identify input embedding shard')
            filename = matches[0]
        else:
            filename = 'model.safetensors'
        print('[embedding] Download/cache weight file, then read ONLY input embedding; no LLM inference.', flush=True)
        candidates = [Path(hf_hub_download(repo, filename, revision=info.sha, cache_dir=cache))]
    with safe_open(str(candidates[0]), framework='pt', device='cpu') as file:
        keys = [k for k in file.keys() if k.endswith('embed_tokens.weight')]
        if len(keys) != 1:
            raise ValueError('Expected one embed_tokens.weight')
        table = file.get_tensor(keys[0]).contiguous()
    provenance.update(tensor_key=keys[0], shape=list(table.shape),
        tensor_sha256=hashlib.sha256(table.view(torch.uint8).numpy().tobytes()).hexdigest())
    return table, provenance


@torch.inference_mode()
def audit(args):
    if not args.trust_checkpoint:
        raise ValueError('Use --trust-checkpoint only for your own trusted training archive (PyTorch pickle).')
    started = time.monotonic(); archive = Archive(args.input)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(io.BytesIO(archive.read('checkpoint.pt')), map_location='cpu', weights_only=False)
    model = ProbeWorldModel(**checkpoint['model_config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    table, provenance = load_embeddings(args.embeddings, args.embedding_repo, args.embedding_revision, args.cache)
    if table.shape[1] != checkpoint['model_config']['token_dim']:
        raise ValueError('Embedding dimension mismatch')
    # CPU table avoids occupying GPU RAM with 151K token embeddings.
    states = archive.lines('states.jsonl'); labels = {r['state_id']: r for r in archive.lines('labels.jsonl')}
    metadata = {r['state_id']: r for r in states}
    questions = defaultdict(list)
    for r in states:
        questions[r['question']].append(r)
    outgoing = {}
    for e in archive.lines('edges.jsonl'):
        if e['parent'] in outgoing:
            raise ValueError('This audit expects linear exploratory trajectories, not branching trees')
        if e['action'] not in ('R', 'E'):
            raise ValueError('Unexpected transition action')
        outgoing[e['parent']] = (e['child'], e['action'])
    variants = ['full', 'hidden_all', *[f'hidden_layer_{i}' for i in range(model.config['num_hidden_layers'])],
        'token_native', 'token_stop', 'topk', 'prefix_content', 'history', 'confidence', 'position_age', 'mask_encoder', 'encoder_context']
    extension = checkpoint['args'].get('extend_size', 8)
    rows = []
    saved = {}
    for r in archive.lines('validation_predictions.jsonl'):
        k = (r.get('source', r['state_id']), r['state_id'], r.get('depth', 0))
        saved[k] = r['expected_K']
    parity_errors = []
    for qi, (question, qmeta) in enumerate(questions.items(), 1):
        if len({r['split'] for r in qmeta}) != 1:
            raise ValueError('A question appears in both train and validation')
        observations = load_observations(archive, qmeta, labels)
        nodes = list(observations.values()); factual = {}
        for begin in range(0, len(nodes), args.batch_size):
            chunk = nodes[begin:begin+args.batch_size]
            b = pack_observations(chunk, table, args.device)
            z = model.encoder(b)
            for i, o in enumerate(chunk):
                factual[o.uid] = z.take(torch.tensor([i], device=args.device))
        qrows = []
        for begin in range(0, len(nodes), args.batch_size):
            chunk = nodes[begin:begin+args.batch_size]
            b = pack_observations(chunk, table, args.device)
            validation = qmeta[0]['split'] != 'train'
            for variant in variants if validation or args.ablate_train else ['full']:
                z = encode(model, b, variant)
                persistence = expected_acceptance(model.acceptance(z), z.lengths).tolist()
                current = [o.uid for o in chunk]; source = current.copy(); actions = ['']*len(chunk)
                for depth in range(args.horizon+1):
                    for i, uid in enumerate(current):
                        r = record(model, z, i, observations[uid], metadata[uid], variant, source[i], depth,
                            actions[i], persistence[i] if depth else None, factual[uid] if depth else None)
                        if r is not None:
                            qrows.append(r)
                            key = (source[i], uid, depth)
                            if variant == 'full' and key in saved:
                                parity_errors.append(abs(r['expected_K']-saved[key]))
                    if depth == args.horizon:
                        break
                    keep = [i for i, uid in enumerate(current) if uid in outgoing]
                    if not keep:
                        break
                    children = [outgoing[current[i]] for i in keep]
                    if any(uid not in observations for uid, _ in children):
                        raise ValueError('Edge crosses question boundary')
                    z = model.transition(z.take(torch.tensor(keep, device=args.device)),
                        torch.tensor([int(a=='E') for _, a in children], device=args.device), extension)
                    current = [uid for uid, _ in children]
                    source = [source[i] for i in keep]; persistence = [persistence[i] for i in keep]
                    actions = [actions[i]+a for i, (_, a) in zip(keep, children)]
        rows.extend(qrows)
        with (out/'predictions.jsonl').open('a', encoding='utf-8') as f:
            for r in qrows:
                f.write(json.dumps(r, allow_nan=False)+'\n')
        print(f'[audit] {qi}/{len(questions)} {question} split={qmeta[0]["split"]} states={len(nodes)} scored={len(qrows)}', flush=True)
        del observations, nodes, factual, b, z
    parity = dict(n=len(parity_errors), max_error=max(parity_errors, default=None),
        mean_error=float(np.mean(parity_errors)) if parity_errors else None,
        passed=bool(parity_errors) and max(parity_errors) < .05)
    results = make_report(rows, args.bootstrap)
    summary = dict(status='complete' if parity['passed'] else 'baseline_not_verified', elapsed_seconds=time.monotonic()-started,
        llm_forward_calls=0, training_updates=0, input=str(args.input), embedding=provenance,
        hidden_layer_indices=checkpoint['args'].get('hidden_layers'),
        variants=variants, ablate_train=args.ablate_train,
        parity_with_saved_predictions=parity, metrics=results,
        caveats=['Zero-input sensitivity is not retrained feature utility; perturbations can be OOD.',
                 'Train results are in-sample; draw conclusions from validation only.',
                 'All archived states are evaluated, including ones evicted by original replay cap.',
                 'Horizon groups differ in sample composition. Action suffix means last action.',
                 'Mask/frontier/length/validity and transition context remain factual.',
                 'Latent distance is a diagnostic, not proof of semantic state fidelity.',
                 'Fixed threshold/block-size effects cannot be inferred from this single run.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Offline feature audit', '', f'Baseline parity: {parity}', '',
        'No drafter/verifier forwards; no training updates. Positive delta MAE = worse without feature.', '',
        '| Split / variant / target | n | MAE | P90 | within 2 | delta MAE |', '|---|---:|---:|---:|---:|---:|']
    for key, m in results.items():
        lines.append(f'| {key} | {m["n"]} | {m["mae"]:.3f} | {m["p90"]:.3f} | {m["within_2"]:.1%} | {m.get("delta_mae", 0):+.3f} |')
    lines += ['', *summary['caveats']]
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    destination = out.with_suffix('.zip')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            zf.write(p, p.name)
    print(f'[done] {destination}', flush=True)
    if not parity['passed']:
        raise RuntimeError('Baseline parity was not confirmed; keep report but do not interpret feature ranking yet.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--embeddings', help='Local safetensors containing the original drafter input embeddings')
    p.add_argument('--embedding-repo', default='Efficient-Large-Model/Fast_dLLM_v2_1.5B')
    p.add_argument('--embedding-revision', default='main')
    p.add_argument('--cache', default='/kaggle/temp/wm_audit_hf')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--ablate-train', action='store_true')
    p.add_argument('--trust-checkpoint', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.bootstrap) < 1 or not 0 <= args.horizon <= 3:
        p.error('Positive batch/bootstrap required; horizon must be 0..3')
    return args


if __name__ == '__main__':
    audit(parse_args())
