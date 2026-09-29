"""Retrain full and layer-28-only world models from the same cached archive."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
import io
import json
from pathlib import Path
import random
import time
import zipfile
import numpy as np
import torch

from world_model_core import ExperienceReplay, WorldModelLearner
from world_model_learning_curve import (Archive, add_questions,
    load_observations, make_learner, parse_ints, validation_rows)
from offline_feature_audit import load_embeddings, make_report


def feature_observations(observations, selected_index=None):
    """Copy the observation map, optionally retaining exactly one hidden layer."""
    if selected_index is None:
        return observations
    result = {}
    for uid, obs in observations.items():
        if obs.hidden.ndim != 3 or not 0 <= selected_index < obs.hidden.shape[1]:
            raise ValueError(f'Hidden layer index {selected_index} invalid for state {uid}: {obs.hidden.shape}')
        result[uid] = replace(obs, hidden=obs.hidden[:, selected_index:selected_index + 1].contiguous())
    return result


def group_metrics(report, variant, group):
    return report.get(f'validation/{variant}/{group}', {})


def summarize_seed(rows, seed, update, train_questions, model_parameters):
    report = make_report(rows, bootstrap=1000)
    groups = ('current', 'h1_R', 'h1_E', 'h3_all')
    metrics = {}
    for group in groups:
        full = group_metrics(report, 'full', group)
        compact = group_metrics(report, 'hidden28', group)
        metrics[group] = {
            'n': full.get('n', 0),
            'full_mae': full.get('mae'),
            'hidden28_mae': compact.get('mae'),
            'full_question_macro_mae': full.get('question_macro_mae'),
            'hidden28_question_macro_mae': compact.get('question_macro_mae'),
            'hidden28_minus_full_question_macro_mae': compact.get('delta_question_macro_mae'),
            'paired_question_bootstrap_ci95': compact.get('delta_question_macro_mae_ci95'),
            'full_dynamics_minus_persistence': full.get('dynamics_minus_persistence_macro'),
            'hidden28_dynamics_minus_persistence': compact.get('dynamics_minus_persistence_macro'),
        }
    return dict(seed=seed, optimizer_updates=update, train_questions=train_questions,
        trainable_parameters_by_model=model_parameters, metrics=metrics)


def run(args):
    started = time.monotonic()
    seeds = parse_ints(args.seeds)
    if not seeds or args.updates < 1:
        raise ValueError('At least one seed and a positive update count are required')
    archive = Archive(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(io.BytesIO(archive.read('checkpoint.pt')), map_location='cpu', weights_only=False)
    state_rows = archive.lines('states.jsonl')
    labels = {r['state_id']: r for r in archive.lines('labels.jsonl')}
    train_q = sorted({r['question'] for r in state_rows if r['split'] == 'train'})
    val_q = sorted({r['question'] for r in state_rows if r['split'] != 'train'})
    if len(train_q) != args.expected_train_questions or len(val_q) != args.expected_validation_questions:
        raise ValueError(f"Expected {args.expected_train_questions}/{args.expected_validation_questions} train/validation questions; "
                         f"archive has {len(train_q)}/{len(val_q)}")
    metadata_by_q = defaultdict(list); metadata = {}
    for order, row in enumerate(state_rows):
        row['row_order'] = order
        metadata[row['state_id']] = row
        metadata_by_q[row['question']].append(row)
    observations = load_observations(archive, state_rows, labels)
    layer_indices = checkpoint.get('args', {}).get('hidden_layers')
    if layer_indices is None:
        layer_indices = [7, 14, 28]
    if 28 not in layer_indices:
        raise ValueError(f'Hidden layer 28 is not present in archive layer list: {layer_indices}')
    hidden28_index = layer_indices.index(28)
    if any(o.hidden.shape[1] != len(layer_indices) for o in observations.values()):
        raise ValueError('Archive hidden-state count does not match its configured hidden layer list')
    observations_28 = feature_observations(observations, hidden28_index)
    edges = archive.lines('edges.jsonl')
    edge_by_parent = {r['parent']: (r['child'], r['action']) for r in edges}
    edge_by_child = {(r['parent'], r['child']): r['action'] for r in edges}
    table, embedding_meta = load_embeddings(args.embeddings, args.embedding_repo,
        args.embedding_revision, args.cache)
    if table.shape[1] != checkpoint['model_config']['token_dim']:
        raise ValueError('Frozen token-embedding dimension differs from checkpoint configuration')
    if args.expected_embedding_sha256 and embedding_meta['tensor_sha256'] != args.expected_embedding_sha256:
        raise ValueError('Embedding hash differs from the verified archive/audit embedding')
    table = table.to(args.device)
    cfg = checkpoint['args']
    extension_size = int(cfg.get('extend_size', 8))
    batch_size = int(cfg.get('batch_sequences', 8))
    horizon = int(cfg.get('horizon', 3))
    max_states = sum(len(metadata_by_q[q]) for q in train_q)
    seed_results = []
    seed_fit_stats = []
    prediction_path = out / 'validation_predictions.jsonl'
    for seed in seeds:
        per_arch_rows = {}
        parameter_counts = {}
        fit_stats = {}
        for architecture in ('full', 'hidden28'):
            selected_observations = observations if architecture == 'full' else observations_28
            model_checkpoint = dict(checkpoint)
            model_checkpoint['model_config'] = dict(checkpoint['model_config'])
            if architecture == 'hidden28':
                model_checkpoint['model_config']['num_hidden_layers'] = 1
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            learner = make_learner(model_checkpoint, table, args.device, seed)
            learner.model._audit_token_table = table
            learner.model._audit_device = args.device
            learner.model._audit_extension_size = extension_size
            replay = ExperienceReplay(max_states=max_states, seed=seed)
            add_questions(replay, train_q, metadata_by_q, selected_observations, edge_by_child)
            parameter_count = sum(p.numel() for p in learner.model.parameters())
            parameter_counts[architecture] = parameter_count
            print(f'[fit] seed={seed} architecture={architecture} questions={len(train_q)} '
                  f'updates={args.updates} trainable_parameters={parameter_count:,}', flush=True)
            # Capture the random-initialization baseline and trained checkpoint on
            # the exact same validation rows. The primary comparison is at args.updates.
            checkpoints = sorted(set([0, args.updates]))
            for update in checkpoints:
                losses = []
                while learner.updates < update:
                    result = learner.update(replay, batch_size=batch_size, max_horizon=horizon)
                    if result is not None: losses.append(result['loss'])
                rows = validation_rows(learner.model, selected_observations, metadata,
                    edge_by_parent, args.validation_states_per_question,
                    args.validation_roots_per_question)
                for row in rows:
                    row.update(variant=architecture, seed=seed, optimizer_updates=learner.updates)
                per_arch_rows[(architecture, update)] = rows
                with prediction_path.open('a', encoding='utf-8') as f:
                    for row in rows: f.write(json.dumps(row, allow_nan=False) + '\n')
                if update == args.updates:
                    # Save train loss separately for convergence interpretation.
                    recent_loss = float(np.mean(losses)) if losses else None
                    fit_stats[architecture] = dict(mean_train_loss=recent_loss,
                        trainable_parameters=parameter_count)
                    print(f'[score] seed={seed} architecture={architecture} update={update} '
                          f'current_MAE={np.mean([abs(r["expected_K"]-r["K"]) for r in rows if r["depth"]==0]):.3f} '
                          f'recent_train_loss={recent_loss}', flush=True)
            del learner, replay
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        combined = per_arch_rows[('full', args.updates)] + per_arch_rows[('hidden28', args.updates)]
        result = summarize_seed(combined, seed, args.updates, len(train_q), parameter_counts)
        result['fit'] = fit_stats
        seed_results.append(result)
        with (out/'seed_metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(result, allow_nan=False) + '\n')
        # Keep initialization metrics too; no optimizer update has occurred at 0.
        init = summarize_seed(per_arch_rows[('full', 0)] + per_arch_rows[('hidden28', 0)],
            seed, 0, len(train_q), parameter_counts)
        with (out/'initialization_metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(init, allow_nan=False) + '\n')
        seed_fit_stats.append(dict(seed=seed, models=fit_stats))

    groups = ('current', 'h1_R', 'h1_E', 'h3_all')
    aggregate = {}
    for group in groups:
        aggregate[group] = {}
        for key in ('full_question_macro_mae', 'hidden28_question_macro_mae',
                    'hidden28_minus_full_question_macro_mae', 'full_mae', 'hidden28_mae'):
            values = [r['metrics'][group][key] for r in seed_results if r['metrics'][group].get(key) is not None]
            aggregate[group][key] = dict(mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, values=values)
    summary = dict(status='complete', elapsed_seconds=time.monotonic()-started,
        input=str(args.input), train_questions=train_q, validation_questions=val_q,
        hidden_layer_indices=layer_indices, selected_hidden_layer=28, topk_gaps_preserved=True,
        other_features_preserved=['candidate embeddings', 'token streams', 'prefix', 'history', 'scalars'],
        seeds=seeds, optimizer_updates_per_model=args.updates,
        llm_forward_calls=0, verifier_replayed=False, embedding=embedding_meta,
        model_comparison=aggregate, seed_results=seed_results,
        fit_stats=seed_fit_stats,
        caveats=['Feature comparison is between retrained models, not zero-input masking.',
                 'The same question split and validation sample are used for both models.',
                 'The 20-question holdout and seed count are exploratory; confirm with more questions/seeds before deployment.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Retrained full vs hidden-layer-28 world model', '',
        f"Training: {len(train_q)} questions, {args.updates} updates/model/seed; validation: {len(val_q)} fixed questions.",
        'Both models retain top-K logit gaps and every non-hidden feature. Lower MAE is better.', '',
        '| Target | Full-feature MAE | Hidden-28-only MAE | Hidden-28 minus full |',
        '|---|---:|---:|---:|']
    for group in groups:
        a = aggregate[group]
        def fmt(k):
            v=a[k]
            return f"{v['mean']:.3f} ± {v['std']:.3f}"
        lines.append(f"| {group} | {fmt('full_question_macro_mae')} | {fmt('hidden28_question_macro_mae')} | {fmt('hidden28_minus_full_question_macro_mae')} |")
    parameter_rows = [r['trainable_parameters_by_model'] for r in seed_results]
    full_params = int(np.mean([r['full'] for r in parameter_rows]))
    compact_params = int(np.mean([r['hidden28'] for r in parameter_rows]))
    lines += ['', 'Negative final column favors the hidden-28-only model; positive favors the full model.',
        f"Trainable parameters: full={full_params:,}; hidden28={compact_params:,} "
        f"(reduction={(1-compact_params/full_params)*100:.1f}%).",
        'Per-seed results and paired question-bootstrap intervals are in `seed_metrics.jsonl`; all raw validation predictions are retained.',
        'This measures retrained feature utility and small-model parameter count, not production latency.']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    dest = out.with_suffix('.zip')
    with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in out.iterdir():
            if p.is_file(): z.write(p, p.name)
    print(f'[done] {dest}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--embeddings'); p.add_argument('--embedding-repo', default='Efficient-Large-Model/Fast_dLLM_v2_1.5B')
    p.add_argument('--embedding-revision', default='25093b6f63300adfd57f72145083c8a528fe4f16')
    p.add_argument('--expected-embedding-sha256', default='a1d834b8219ee8040d8a32a019e166d10c5bb0546d6d2e2e9b2a0041b8a2e73b')
    p.add_argument('--cache', default='/kaggle/temp/wm_audit_hf')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds', default='42,43,44'); p.add_argument('--updates', type=int, default=512)
    p.add_argument('--validation-states-per-question', type=int, default=24)
    p.add_argument('--validation-roots-per-question', type=int, default=4)
    p.add_argument('--expected-train-questions', type=int, default=80)
    p.add_argument('--expected-validation-questions', type=int, default=20)
    p.add_argument('--trust-checkpoint', action='store_true'); args=p.parse_args()
    if not args.trust_checkpoint: p.error('Use --trust-checkpoint only with your own training archive')
    run(args)


if __name__ == '__main__': main()
