"""Question-count learning curves from cached experience; zero LLM calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import io
import json
from pathlib import Path
import random
import time
import zipfile
import numpy as np
import torch
from world_model_core import (ExperienceReplay, WorldModelLearner, pack_observations,
    expected_acceptance, prefix_log_distribution)
from world_model_probe import ProbeWorldModel
from offline_feature_audit import Archive, load_observations, make_report, load_embeddings


def uniform_sample(rows, limit):
    if len(rows) <= limit:
        return rows
    return [rows[i] for i in np.linspace(0, len(rows)-1, limit, dtype=int)]


def build_replay(qids, metadata, observations, edges, max_states, seed):
    replay = ExperienceReplay(max_states=max_states, seed=seed)
    add_questions(replay, qids, metadata, observations, edges)
    return replay


def add_questions(replay, qids, metadata, observations, edges):
    for q in qids:
      for row in metadata[q]:
        uid = row['state_id']
        parent = row.get('parent_state_id')
        edge = edges.get((parent, uid)) if parent else None
        if edge:
            replay.add(observations[parent], observations[uid], edge)
        else:
            replay.add_node(observations[uid])


def make_learner(checkpoint, table, device, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ProbeWorldModel(**checkpoint['model_config'])
    cfg = checkpoint['args']
    learner = WorldModelLearner(model, table, device,
        extension_size=int(cfg.get('extend_size', 8)),
        lr=float(cfg.get('learning_rate', 3e-4)),
        ema_decay=float(cfg.get('ema_decay', .99)),
        latent_weight=float(cfg.get('latent_weight', .1)),
        warmup_updates=int(cfg.get('warmup_updates', 16)),
        horizon_warmup=int(cfg.get('horizon_warmup_updates', 64)))
    return learner


@torch.inference_mode()
def validation_rows(model, observations, metadata, edges, limit_per_question=24, roots_per_question=4):
    model.eval()
    by_question = defaultdict(list)
    for uid, meta in metadata.items():
        if meta['split'] == 'validation':
            by_question[meta['question']].append(uid)
    rows = []
    for question, ids in sorted(by_question.items()):
        ids.sort(key=lambda uid: metadata[uid]['row_order'])
        selected = uniform_sample(ids, limit_per_question)
        for begin in range(0, len(selected), 4):
            chunk = [observations[uid] for uid in selected[begin:begin+4]]
            batch = pack_observations(chunk, model._audit_token_table, model._audit_device)
            z = model.encoder(batch)
            logits = model.acceptance(z)
            prediction = expected_acceptance(logits, z.lengths).tolist()
            distributions = prefix_log_distribution(logits, z.lengths)
            for i, obs in enumerate(chunk):
                if obs.accepted is None:
                    continue
                rows.append(dict(split='validation', variant='full', question=question,
                    source=obs.uid, state_id=obs.uid, depth=0, actions='', group='current',
                    length=obs.length, K=obs.accepted, expected_K=prediction[i],
                    nll=float(-distributions[i, obs.accepted])))
        roots = [uid for uid in ids if metadata[uid].get('parent_state_id') is None and uid in edges]
        roots = uniform_sample(roots, roots_per_question)
        for root in roots:
            uid = root
            obs = observations[uid]
            batch = pack_observations([obs], model._audit_token_table, model._audit_device)
            z = model.encoder(batch)
            pred = float(expected_acceptance(model.acceptance(z), z.lengths)[0])
            actions = ''
            for depth in range(1, 4):
                edge = edges.get(uid)
                if edge is None:
                    break
                child, action = edge
                child_obs = observations[child]
                z = model.transition(z, torch.tensor([int(action == 'E')], device=model._audit_device),
                                     int(model._audit_extension_size))
                child_pred = float(expected_acceptance(model.acceptance(z), z.lengths)[0])
                if child_obs.accepted is not None and int(z.lengths[0]) == child_obs.length:
                    actions += action
                    true_delta = child_obs.accepted-obs.accepted if obs.accepted is not None else None
                    row = dict(split='validation', variant='full', question=question,
                        source=root, state_id=child, depth=depth, actions=actions,
                        group=f'h{depth}_{action}', length=child_obs.length,
                        source_length=obs.length, K=child_obs.accepted,
                        expected_K=child_pred,
                        nll=float(-prefix_log_distribution(model.acceptance(z), z.lengths)[0, child_obs.accepted]),
                        persistence_error=abs(pred-child_obs.accepted))
                    if true_delta is not None:
                        row.update(source_K=obs.accepted, true_delta_K=true_delta,
                            predicted_delta_K=child_pred-pred,
                            change='gain' if true_delta > 0 else ('loss' if true_delta < 0 else 'same'))
                    rows.append(row)
                else:
                    break
                uid, obs, pred = child, child_obs, child_pred
    return rows


def aggregate(points):
    result = {}
    keys = sorted({(p['regime'], p['questions'], metric) for p in points for metric in
                   ('current_mae', 'h1_R_mae', 'h1_E_mae', 'h3_all_mae')
                   if metric in p['metrics']})
    for regime, count, metric in keys:
        vals = [p['metrics'][metric] for p in points if p['regime'] == regime
                and p['questions'] == count and p['metrics'].get(metric) is not None]
        if not vals:
            continue
        result[f'{regime}/q{count}/{metric}'] = dict(seeds=len(vals), mean=float(np.mean(vals)),
            std=float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0, values=vals)
    return result


def parse_ints(value):
    return [int(x) for x in value.split(',') if x.strip()]


def run(args):
    started = time.monotonic()
    archive = Archive(args.input)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(io.BytesIO(archive.read('checkpoint.pt')), map_location='cpu', weights_only=False)
    states = archive.lines('states.jsonl')
    labels = {r['state_id']: r for r in archive.lines('labels.jsonl')}
    train_q = sorted({r['question'] for r in states if r['split'] == 'train'})
    val_q = sorted({r['question'] for r in states if r['split'] != 'train'})
    sizes = parse_ints(args.question_sizes); seeds = parse_ints(args.seeds)
    if not sizes or any(n < 2 or n > len(train_q) for n in sizes):
        raise ValueError(f'Question sizes must be between 2 and {len(train_q)}')
    metadata_by_q = defaultdict(list)
    metadata = {}
    for order, row in enumerate(states):
        row['row_order'] = order
        metadata[row['state_id']] = row
        metadata_by_q[row['question']].append(row)
    print(f'[data] {len(train_q)} train questions; {len(val_q)} fixed validation questions; loading cached tensors', flush=True)
    observations = load_observations(archive, states, labels)
    edge_by_parent = {}
    edge_by_child = {}
    for edge in archive.lines('edges.jsonl'):
        edge_by_parent[edge['parent']] = (edge['child'], edge['action'])
        edge_by_child[(edge['parent'], edge['child'])] = edge['action']
    table, embedding_meta = load_embeddings(args.embeddings, args.embedding_repo,
                                              args.embedding_revision, args.cache)
    expected_embedding_hash = args.expected_embedding_sha256
    if expected_embedding_hash and embedding_meta['tensor_sha256'] != expected_embedding_hash:
        raise ValueError('Input embedding hash differs from the verified 100q audit; refusing a mismatched learning curve')
    device = args.device
    table = table.to(device)
    cfg = checkpoint['args']
    # Keep every selected train observation in this experiment. Capping at the
    # original 4096 online replay slots would make 40/80-question points silently
    # discard data and would defeat the data-volume comparison.
    max_states = sum(len(metadata_by_q[q]) for q in train_q)
    batch_size = int(cfg.get('batch_sequences', 8))
    max_horizon = int(cfg.get('horizon', 3))
    # Retain every example at every size. The proportional regime uses a constant
    # update budget per added question; compute-matched fits reset and get fixed steps.
    regimes = ('proportional', 'fixed_updates')
    points = []
    for seed in seeds:
        order = list(train_q); random.Random(seed).shuffle(order)
        nested = {size: order[:size] for size in sizes}
        for regime in regimes:
            cumulative = None
            if regime == 'proportional':
                learner = make_learner(checkpoint, table, device, seed)
                cumulative = ExperienceReplay(max_states, seed)
                previous_size = 0
            for size in sizes:
                selected_q = nested[size]
                if regime == 'fixed_updates':
                    learner = make_learner(checkpoint, table, device, seed)
                    cumulative = build_replay(selected_q, metadata_by_q, observations,
                                              edge_by_child, max_states, seed)
                    updates = args.fixed_updates
                else:
                    add_questions(cumulative, order[previous_size:size], metadata_by_q,
                                  observations, edge_by_child)
                    updates = args.updates_per_question * (size-previous_size)
                learner.model._audit_token_table = table
                learner.model._audit_device = device
                learner.model._audit_extension_size = int(cfg.get('extend_size', 8))
                train_losses = []
                for _ in range(updates):
                    metric = learner.update(cumulative, batch_size=batch_size, max_horizon=max_horizon)
                    if metric is not None:
                        train_losses.append(metric['loss'])
                validation = validation_rows(learner.model, observations, metadata, edge_by_parent,
                    args.validation_states_per_question, args.validation_roots_per_question)
                report = make_report(validation, args.bootstrap)
                selected = {k.split('/')[-1]: v for k, v in report.items()}
                metric_row = {
                    'current_mae': selected.get('current', {}).get('mae'),
                    'h1_R_mae': selected.get('h1_R', {}).get('mae'),
                    'h1_E_mae': selected.get('h1_E', {}).get('mae'),
                    'h3_all_mae': selected.get('h3_all', {}).get('mae'),
                    'h1_R_persistence_delta': selected.get('h1_R', {}).get('dynamics_minus_persistence_macro'),
                    'h1_E_persistence_delta': selected.get('h1_E', {}).get('dynamics_minus_persistence_macro'),
                    'h1_R_changed_n': selected.get('h1_R_changed', {}).get('n', 0),
                    'h1_E_changed_n': selected.get('h1_E_changed', {}).get('n', 0),
                }
                point = dict(regime=regime, questions=size, seed=seed,
                    selected_question_ids=selected_q, optimizer_updates=learner.updates,
                    new_updates=updates, replay_states=len(cumulative.nodes), replay_edges=len(cumulative.edges),
                    mean_recent_train_loss=float(np.mean(train_losses)) if train_losses else None,
                    metrics=metric_row, validation_metrics=report)
                points.append(point)
                with (out/'learning_curve.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(point, allow_nan=False)+'\n')
                with (out/'validation_predictions.jsonl').open('a', encoding='utf-8') as f:
                    for row in validation:
                        row.update(regime=regime, questions=size, seed=seed)
                        f.write(json.dumps(row, allow_nan=False)+'\n')
                print(f"[curve] regime={regime} seed={seed} questions={size} "
                      f"updates={learner.updates} current_MAE={metric_row['current_mae']:.3f} "
                      f"h1R={metric_row['h1_R_mae']} h1E={metric_row['h1_E_mae']}", flush=True)
                if regime == 'proportional':
                    previous_size = size
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    (out/'summary.json').write_text(json.dumps(dict(status='complete', elapsed_seconds=time.monotonic()-started,
        input=str(args.input), questions_train=len(train_q), validation_questions=len(val_q),
        question_sizes=sizes, seeds=seeds, updates_per_question=args.updates_per_question,
        fixed_updates=args.fixed_updates, total_optimizer_updates=sum(p['new_updates'] for p in points),
        fixed_validation_sample_per_question=args.validation_states_per_question,
        fixed_validation_roots_per_question=args.validation_roots_per_question,
        llm_forward_calls=0, verifier_replayed=False, training_regimes={
            'proportional': 'updates increase proportionally with question count; measures added data under fixed updates per question.',
            'fixed_updates': 'same number of gradient updates at each question count; isolates benefit of adding examples under matched compute.'},
        embedding=embedding_meta, points=aggregate(points), caveats=[
            'Uses one fixed question-level train/validation split from the archive.',
            'Validation evaluation uses the same deterministic stratified subset at every checkpoint.',
            'Fixed-update models retrain from the same random initialization per seed and get equal optimizer steps.',
            'Proportional runs are nested incremental training; report includes the number of optimizer updates.',
            'Three seeds and 20 validation questions give an exploratory learning curve, not a final scaling law.',
            'World-model training updates are run; drafter/verifier calls remain zero.']), indent=2), encoding='utf-8')
    report_lines = ['# World-model learning curve', '',
        'Each point is evaluated on the same held-out questions and state sample.',
        '`fixed_updates` isolates data volume at matched optimizer steps; `proportional` scales steps with questions.', '',
        '| Regime / train questions | Current MAE | R1 MAE | E1 MAE | H3 MAE | Seeds |',
        '|---|---:|---:|---:|---:|---:|']
    aggregated = aggregate(points)
    for regime in regimes:
        for size in sizes:
            def show(metric):
                v = aggregated.get(f'{regime}/q{size}/{metric}')
                return '—' if v is None else f"{v['mean']:.3f} ± {v['std']:.3f}"
            report_lines.append(f"| {regime} / {size} | {show('current_mae')} | {show('h1_R_mae')} | "
                f"{show('h1_E_mae')} | {show('h3_all_mae')} | {len(seeds)} |")
    report_lines += ['', 'Values are mean ± standard deviation across independent initialization/data-order seeds. '
        'Positive persistence delta means the learned transition has higher MAE than carrying the source prediction.', '',
        'Three seeds and one fixed 20-question holdout are exploratory. Do not call a noisy/non-monotonic curve a scaling law.']
    (out/'report.md').write_text('\n'.join(report_lines), encoding='utf-8')
    dest = out.with_suffix('.zip')
    with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in out.iterdir():
            if path.is_file(): z.write(path, path.name)
    print(f'[done] {dest}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--embeddings'); p.add_argument('--embedding-repo', default='Efficient-Large-Model/Fast_dLLM_v2_1.5B')
    p.add_argument('--embedding-revision', default='25093b6f63300adfd57f72145083c8a528fe4f16')
    p.add_argument('--expected-embedding-sha256', default='a1d834b8219ee8040d8a32a019e166d10c5bb0546d6d2e2e9b2a0041b8a2e73b')
    p.add_argument('--cache', default='/kaggle/temp/wm_audit_hf')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--question-sizes', default='10,20,40,80'); p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--updates-per-question', type=int, default=8); p.add_argument('--fixed-updates', type=int, default=512)
    p.add_argument('--validation-states-per-question', type=int, default=24)
    p.add_argument('--validation-roots-per-question', type=int, default=4); p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--trust-checkpoint', action='store_true'); a = p.parse_args()
    if not a.trust_checkpoint:
        p.error('Use --trust-checkpoint only with your own training archive')
    if min(a.updates_per_question, a.fixed_updates, a.validation_states_per_question,
           a.validation_roots_per_question, a.bootstrap) < 1:
        p.error('All counts must be positive')
    if sorted(set(parse_ints(a.question_sizes))) != parse_ints(a.question_sizes):
        p.error('Question sizes must be strictly increasing and unique')
    run(a)


if __name__ == '__main__':
    main()
