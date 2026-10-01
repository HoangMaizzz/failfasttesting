"""Fixed-question learning curves, detailed H0..H3 errors and matched ablations.

All audits reuse real saved experiences. They never run drafter/verifier again.
Removal at inference measures sensitivity; separately retrained removals measure
how much each factor helps after the small model adapts. Neither is a factorial
interaction study or proof of a causal effect in a deployed controller.
"""
from collections import defaultdict
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from world_model_core import (Observation, ExperienceReplay, WorldModelLearner,
    expected_acceptance, prefix_log_distribution, valid_positions)
from world_model_twosource import TwoSourceWorldModel, FEATURE_VARIANTS, configure_losses


def lines(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(row) for row in stream if row.strip()]


def write_json(path, value):
    from pretrain_acceptance_world_model import atomic_json
    atomic_json(path, value)


def load_saved_replay(output, split, roots_per_question=32, horizon=3, seed=9102):
    """Select roots once, then include their real descendants to preserve paths."""
    output = Path(output)
    metadata = {r['state_id']: r for r in lines(output/'states.jsonl') if r['split'] == split}
    labels = {r['state_id']: r for r in lines(output/'labels.jsonl')}
    teacher = {r['state_id']: r for r in lines(output/'teacher_targets.jsonl')}
    edges = [r for r in lines(output/'edges.jsonl') if r['split'] == split]
    outgoing = {e['parent']: (e['child'], e['action']) for e in edges}
    groups = defaultdict(list)
    for uid, meta in metadata.items(): groups[meta['question']].append(uid)
    roots, keep = [], set()
    for question, uids in sorted(groups.items()):
        rng = random.Random(f'{seed}/{question}')
        # Mix random boundaries and directly teacher-labelled boundaries.
        with_teacher = [uid for uid in uids if uid in teacher]
        first = rng.sample(with_teacher, min(len(with_teacher), max(1, roots_per_question//4)))
        remaining = [uid for uid in uids if uid not in first]
        selected = first + rng.sample(remaining, min(len(remaining), roots_per_question-len(first)))
        roots.extend(selected)
        for uid in selected:
            keep.add(uid)
            for _ in range(horizon):
                if uid not in outgoing: break
                uid = outgoing[uid][0]; keep.add(uid)
    shards = defaultdict(list)
    for uid in sorted(keep): shards[metadata[uid]['shard']].append(uid)
    nodes = {}
    for shard, uids in sorted(shards.items()):
        with np.load(output/shard, allow_pickle=False) as arrays:
            for uid in uids:
                meta = metadata[uid]; row = meta['row']
                begin, end = map(int, arrays['offsets'][row:row+2])
                def tensor(name, dtype=None):
                    # Copy selected rows; do not retain the whole decompressed shard.
                    return torch.tensor(arrays[name][begin:end], dtype=dtype)
                label = labels.get(uid, {})
                observation = Observation(uid, meta['question'], meta['round_id'],
                    tensor('ids', torch.long), tensor('hidden', torch.float16),
                    tensor('gaps', torch.float16), tensor('scalars', torch.float32),
                    torch.tensor(arrays['context'][row], dtype=torch.float32),
                    label.get('accepted_len') if label.get('label_valid', False) else None,
                    torch.tensor(meta['prefix_token_ids'], dtype=torch.long),
                    tensor('aligned_topk_token_ids', torch.long), tensor('history', torch.float32))
                t = teacher.get(uid)
                if t is not None:
                    observation.teacher_margin = torch.tensor(t['margin'])
                    if t.get('features') is not None: observation.teacher_features = torch.tensor(t['features'])
                if 'verifier_history_offsets' in arrays:
                    h0, h1 = map(int, arrays['verifier_history_offsets'][row:row+2])
                    observation.verifier_history = torch.tensor(arrays['verifier_history'][h0:h1], dtype=torch.float32)
                nodes[uid] = observation
    replay = ExperienceReplay(max(2, len(nodes)+1), seed=seed, sampling_mode='action_balanced')
    for observation in nodes.values(): replay.add_node(observation)
    for e in edges:
        if e['parent'] in nodes and e['child'] in nodes:
            replay.add(nodes[e['parent']], nodes[e['child']], e['action'])
    replay.audit_roots = roots
    replay.audit_sampling = dict(method='fixed random roots + quarter teacher roots; retain H3 descendants',
        roots=len(roots), states=len(nodes), questions=len(groups), roots_per_question=roots_per_question)
    return replay


def auc(labels, probabilities):
    y = np.asarray(labels, dtype=bool); scores = np.asarray(probabilities)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0: return None
    order = np.argsort(scores, kind='stable'); ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and scores[order[end]] == scores[order[start]]: end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))


def token_metrics(labels, probabilities):
    if not labels: return dict(n=0)
    y = np.asarray(labels, dtype=float); p = np.clip(np.asarray(probabilities), 1e-7, 1-1e-7)
    calibration = []
    for lower in np.arange(0, 1, .1):
        mask = (p >= lower) & (p < lower+.1)
        if mask.any(): calibration.append(dict(n=int(mask.sum()), mean_p=float(p[mask].mean()), rate=float(y[mask].mean())))
    return dict(n=len(y), positive_rate=float(y.mean()), brier=float(((p-y)**2).mean()),
        log_loss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
        accuracy=float(((p >= .5) == y).mean()), auc=auc(y, p),
        ece=float(sum(r['n']*abs(r['mean_p']-r['rate']) for r in calibration)/len(y)),
        calibration=calibration)


def summarize(rows):
    known = [r for r in rows if r['K'] is not None]
    if not known: return dict(n=0, total=len(rows))
    error = np.array([r['expected_K']-r['K'] for r in known])
    per_q = defaultdict(list)
    for row, e in zip(known, error): per_q[row['question']].append(abs(float(e)))
    result = dict(n=len(known), total=len(rows), mae=float(abs(error).mean()),
        question_macro_mae=float(np.mean([np.mean(v) for v in per_q.values()])),
        rmse=float(np.sqrt((error**2).mean())), bias=float(error.mean()),
        p50_abs_error=float(np.quantile(abs(error), .5)), p90_abs_error=float(np.quantile(abs(error), .9)),
        p95_abs_error=float(np.quantile(abs(error), .95)),
        within_1=float((abs(error) <= 1).mean()), within_2=float((abs(error) <= 2).mean()),
        exact_mode=float(np.mean([r['mode_K'] == r['K'] for r in known])),
        nll=float(np.mean([r['nll'] for r in known])),
        mean_prediction=float(np.mean([r['expected_K'] for r in known])),
        mean_K=float(np.mean([r['K'] for r in known])),
        question_mae={q:float(np.mean(v)) for q,v in per_q.items()})
    if all('persistence_K' in r for r in known):
        result['predicted_persistence_mae'] = float(np.mean([abs(r['persistence_K']-r['K']) for r in known]))
    if all('train_length_mean' in r and r['train_length_mean'] is not None for r in known):
        result['train_length_mean_mae'] = float(np.mean([abs(r['train_length_mean']-r['K']) for r in known]))
    latent = [r['latent_cosine_distance'] for r in known if r.get('latent_cosine_distance') is not None]
    if latent: result['latent_cosine_distance'] = float(np.mean(latent))
    mask = [r['mask_brier'] for r in known if r.get('mask_brier') is not None]
    if mask: result['mask_brier'] = float(np.mean(mask))
    return result


@torch.no_grad()
def detailed_rows(learner, replay, horizon=3, baseline_replay=None):
    learner.model.eval(); learner.target_encoder.eval()
    outgoing = {p:(c,a) for p,c,a in replay.edges}
    roots = getattr(replay, 'audit_roots', list(replay.nodes))
    train_by_length = defaultdict(list)
    if baseline_replay is not None:
        for o in baseline_replay.nodes.values():
            if o.accepted is not None: train_by_length[o.length].append(o.accepted)
    overall = [x for values in train_by_length.values() for x in values]
    rows = []

    def record(z, target, source, depth, sequence, baseline):
        logits = learner.model.acceptance(z)
        lp = prefix_log_distribution(logits, z.lengths)[0]
        survival = F.logsigmoid(logits).cumsum(-1).exp()[0, :target.length].cpu().tolist()
        row = dict(source=source.uid, state_id=target.uid, question=target.question,
            round_id=target.round_id, kind='current' if depth == 0 else 'rollout',
            horizon=depth, actions=sequence, length=target.length,
            K=target.accepted, expected_K=float(expected_acceptance(logits,z.lengths)[0]),
            mode_K=int(lp.argmax()), nll=None if target.accepted is None else float(-lp[target.accepted]),
            acceptance_survival=survival,
            mask_count=int(target.scalars[:,0].sum()), refinement=float(target.context[3])*3,
            teacher_available=target.teacher_margin is not None)
        if depth:
            b = learner.pack([target]); truth = learner.target_encoder(b)
            row['latent_cosine_distance'] = float((1-F.cosine_similarity(z.tokens,truth.tokens,-1))[0,:target.length].mean())
            row['mask_brier'] = float((z.mask_probs[0,:target.length]-b['scalars'][0,:,0]).square().mean())
            row['persistence_K'] = baseline
        if train_by_length:
            row['train_length_mean'] = float(np.mean(train_by_length[target.length])) if train_by_length[target.length] else float(np.mean(overall))
        if target.teacher_features is not None:
            row['local_agreement_truth'] = target.teacher_features[:,2].tolist()
            row['local_agreement_probability'] = learner.model.local_agreement(z)[0,:target.length].sigmoid().cpu().tolist()
            row['margin_truth'] = target.teacher_margin.tolist()
            row['margin_prediction_scaled'] = learner.model.margin_head(z.tokens[:,:,-32:]).squeeze(-1)[0,:target.length].tanh().cpu().tolist()
        if depth == 0:
            confidence = target.scalars[:,7].clamp(1e-6,1-1e-6)
            row['drafter_confidence'] = confidence.tolist()
            row['confidence_survival'] = confidence.log().cumsum(0).exp().tolist()
            row['confidence_expected_K'] = float(confidence.log().cumsum(0).exp().sum())
        rows.append(row)

    # Batch current encoding; rollouts remain open-loop: actual children ONLY targets.
    for begin in range(0, len(roots), 8):
        source_nodes = [replay.nodes[uid] for uid in roots[begin:begin+8]]
        z = learner.model.encoder(learner.pack(source_nodes))
        for i, source in enumerate(source_nodes):
            zi = z.take(torch.tensor([i], device=learner.device))
            record(zi, source, source, 0, '', None)
            baseline = float(expected_acceptance(learner.model.acceptance(zi),zi.lengths)[0])
            uid = source.uid; sequence = ''
            for depth in range(1, horizon+1):
                if uid not in outgoing: break
                uid, action = outgoing[uid]; sequence += action
                zi = learner.model.transition(zi,torch.tensor([int(action=='E')],device=learner.device),learner.extension_size)
                record(zi,replay.nodes[uid],source,depth,sequence,baseline)
    return rows


def detailed_report(rows):
    groups = defaultdict(list)
    for r in rows:
        group = f"h{r['horizon']}"
        groups[group].append(r)
        groups[f"{group}/L{r['length']}"] .append(r)
        groups[f"{group}/sequence/{r['actions'] or 'current'}"].append(r)
        groups[f"{group}/mask/{'present' if r['mask_count'] else 'resolved'}"].append(r)
    report = dict(groups={key:summarize(values) for key,values in sorted(groups.items())}, tokens={})
    for depth in range(4):
        selected = [r for r in rows if r['horizon']==depth]
        y, p, local_y, local_p, conf_y, conf_p = [], [], [], [], [], []
        first_ranks, margin_error = [], []
        for row in selected:
            if row['K'] is not None:
                prefix = [float(i < row['K']) for i in range(row['length'])]
                y.extend(prefix); p.extend(row['acceptance_survival'])
                if row['K'] < row['length'] and row.get('local_agreement_probability') is not None:
                    scores = row['local_agreement_probability']; k = row['K']
                    first_ranks.append((sum(v<scores[k] for v in scores)+.5*sum(v==scores[k] for v in scores))/len(scores))
                if depth == 0:
                    conf_y.extend(prefix); conf_p.extend(row['confidence_survival'])
            if 'local_agreement_truth' in row:
                local_y.extend(row['local_agreement_truth']); local_p.extend(row['local_agreement_probability'])
                margin_error.extend(abs(a-b) for a,b in zip(row['margin_prediction_scaled'],np.tanh(np.array(row['margin_truth'])/5)))
        report['tokens'][f'h{depth}'] = dict(prefix_acceptance=token_metrics(y,p),
            local_teacher_forced_agreement=token_metrics(local_y,local_p),
            confidence_prefix_baseline=token_metrics(conf_y,conf_p),
            first_mismatch_weakness_percentile_mean=float(np.mean(first_ranks)) if first_ranks else None,
            first_mismatch_samples=len(first_ranks),
            scaled_margin_mae=float(np.mean(margin_error)) if margin_error else None)
    report['interpretation'] = ('Prefix acceptance ends at first mismatch. Local teacher-forced agreement '
        'after mismatch is a separate target; not an accepted or recoverable corrected suffix. '
        'H1/H2/H3 count individual R or E actions. Smaller MAE/Brier/NLL is better.')
    return report


def save_rows(path, rows):
    from pretrain_acceptance_world_model import append_json
    for row in rows: append_json(path,row)


def audit_learning_stage(learner, replay, output, train_questions, horizon=3, states_per_question=32):
    if not hasattr(learner, '_fixed_audit_holdout'):
        learner._fixed_audit_holdout = load_saved_replay(output,'validation',states_per_question,horizon)
        write_json(output/'audit_validation_ids.json', learner._fixed_audit_holdout.audit_roots)
    replay = learner._fixed_audit_holdout
    rows = detailed_rows(learner,replay,horizon)
    report = detailed_report(rows)
    if train_questions:
        baseline_rows = lines(output/'evaluation'/'learning_000_predictions.jsonl')
        report['paired_change_from_initial'] = paired_question_ci(baseline_rows, rows)
        report['paired_change_note'] = 'Negative delta means lower heldout MAE than the untrained model'
    report.update(train_questions=train_questions, updates=learner.updates,
                  sampling=replay.audit_sampling, verifier_calls_for_evaluation=0)
    audit_dir = output/'evaluation'; audit_dir.mkdir(exist_ok=True)
    write_json(audit_dir/f'learning_{train_questions:03d}.json',report)
    save_rows(audit_dir/f'learning_{train_questions:03d}_predictions.jsonl',rows)
    snapshots = output/'learning_checkpoints'; snapshots.mkdir(exist_ok=True)
    torch.save(dict(model_config=learner.model.config,
        model={k:v.detach().cpu() for k,v in learner.model.state_dict().items()},
        updates=learner.updates,train_questions=train_questions),snapshots/f'questions_{train_questions:03d}.pt')
    print(f"[learning-audit] train_questions={train_questions} updates={learner.updates} "
          f"metrics={ {h:report['groups'].get(h,{}) .get('mae') for h in ('h0','h1','h2','h3')} }",flush=True)
    return dict(train_questions=train_questions,updates=learner.updates,
                groups={h:report['groups'].get(h,{'n':0}) for h in ('h0','h1','h2','h3')})


def paired_question_ci(full_rows, other_rows, seed=72, repetitions=500):
    lookup = {(r['source'],r['state_id'],r['horizon']):r for r in full_rows if r['K'] is not None}
    groups = defaultdict(lambda:defaultdict(list))
    for r in other_rows:
        key=(r['source'],r['state_id'],r['horizon'])
        if key not in lookup or r['K'] is None: continue
        base=lookup[key]
        delta=abs(r['expected_K']-r['K'])-abs(base['expected_K']-base['K'])
        groups[f"h{r['horizon']}"][r['question']].append(delta)
    result={}; rng=np.random.default_rng(seed)
    for group, per_q in groups.items():
        values=np.array([np.mean(v) for v in per_q.values()])
        samples=values[rng.integers(0,len(values),(repetitions,len(values)))].mean(-1)
        result[group]=dict(questions=len(values), delta_macro_mae=float(values.mean()),
            ci95=[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],
            meaning='positive means removal is worse than full')
    return result


def run_retrained_audit(args, online_learner):
    output=args.output_dir; root=output/'factor_audit'; root.mkdir(exist_ok=True)
    train=load_saved_replay(output,'train',args.audit_train_states_per_question,args.horizon)
    validation=online_learner._fixed_audit_holdout
    write_json(root/'sampling.json',dict(train=train.audit_sampling,validation=validation.audit_sampling,
        warning='Matched offline retraining on a bounded subset, not the online learning-curve weights'))
    # Inference-only lesions share the final ONLINE model and identical heldout rows.
    full_online=detailed_rows(online_learner,validation,args.horizon,train)
    sensitivity={}
    try:
        for variant in args.audit_variants:
            if variant in ('no_latent_loss','no_structure_loss','no_rollout_loss','no_teacher','no_residual','no_agreement_readout'):
                continue  # Loss/removal requires retraining, not a feature lesion.
            online_learner.model.encoder.variant=variant
            online_learner.model.dynamics.variant=variant
            rows=detailed_rows(online_learner,validation,args.horizon,train)
            sensitivity[variant]=dict(report=detailed_report(rows),
                paired_delta=paired_question_ci(full_online,rows))
    finally:
        online_learner.model.encoder.variant='full'
        online_learner.model.dynamics.variant='full'
    write_json(root/'inference_sensitivity.json',sensitivity)
    results=[]
    for seed in args.audit_seeds:
        full_rows=None
        # Full must precede removal variants for paired comparison.
        for variant in ['full']+[v for v in args.audit_variants if v!='full']:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            cfg=dict(online_learner.model.config,variant=variant)
            model=TwoSourceWorldModel(**cfg)
            learner=WorldModelLearner(model,online_learner.token_table,online_learner.device,
                args.extend_size,args.learning_rate,warmup_updates=16,horizon_warmup=64)
            configure_losses(learner,variant)
            # All variants see exactly the same ordered minibatch draws.
            train.rng=random.Random(seed); train.state_rng=random.Random(seed+1_000_003)
            print(f'[factor-fit] seed={seed} variant={variant} updates={args.audit_retrain_updates}',flush=True)
            for update in range(args.audit_retrain_updates):
                metrics=learner.update(train,args.batch_sequences,args.horizon)
                if metrics is None: raise RuntimeError('Offline audit lacks valid train targets')
                if (update+1)%100==0: print(f"[factor-fit] {variant} update={update+1} loss={metrics['loss']:.4f}",flush=True)
            rows=detailed_rows(learner,validation,args.horizon,train)
            if variant=='full': full_rows=rows
            report=detailed_report(rows)
            record=dict(seed=seed,variant=variant,updates=learner.updates,report=report,
                paired_delta=paired_question_ci(full_rows,rows))
            results.append(record)
            folder=root/f'seed_{seed}_{variant}'; folder.mkdir(exist_ok=True)
            write_json(folder/'report.json',record); save_rows(folder/'predictions.jsonl',rows)
            torch.save(dict(model_config=model.config,model={k:v.detach().cpu() for k,v in model.state_dict().items()}),folder/'model.pt')
            write_json(root/'retrained_comparisons.json',results)
            del learner,model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            from pretrain_acceptance_world_model import package
            # A durable ZIP survives a timeout midway through the ablation sweep.
            prior_summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            package(output,output.with_suffix('.zip'),dict(prior_summary,status='partial_audit',audit_completed=len(results)))
    summary=dict(seeds=args.audit_seeds,variants=args.audit_variants,
        updates_per_variant=args.audit_retrain_updates,variants_completed=len(results),
        question_split='80 training / 20 heldout by default; never train on heldout',
        horizon_units='individual native R/E decisions, E includes first native unmask',
        verifier_calls_for_audit=0, results=[dict(seed=r['seed'],variant=r['variant'],
            paired_delta=r['paired_delta'],groups={h:r['report']['groups'].get(h,{'n':0})
            for h in ('h0','h1','h2','h3')}) for r in results],
        caveat='Single-factor removals with matched seeds and draws; interactions need a later factorial study')
    write_json(root/'summary.json',summary)
    export_audit_tables_and_plots(output, summary)
    return summary


def export_audit_tables_and_plots(output, summary):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    table=[]
    for result in summary['results']:
        for horizon, metrics in result['groups'].items():
            delta=result['paired_delta'].get(horizon,{})
            table.append(dict(seed=result['seed'],variant=result['variant'],horizon=horizon,
                n=metrics.get('n'),mae=metrics.get('mae'),macro_mae=metrics.get('question_macro_mae'),
                p90_error=metrics.get('p90_abs_error'),within_1=metrics.get('within_1'),
                paired_delta_macro_mae=delta.get('delta_macro_mae'),
                ci_low=delta.get('ci95',[None,None])[0],ci_high=delta.get('ci95',[None,None])[1]))
    with (output/'factor_audit'/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    curve=json.loads((output/'learning_curve.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for horizon in ('h0','h1','h2','h3'):
        x=[stage['train_questions'] for stage in curve]
        y=[stage['groups'].get(horizon,{}).get('question_macro_mae',np.nan) for stage in curve]
        axes[0].plot(x,y,marker='o',label=horizon)
    axes[0].set(xlabel='Training questions seen',ylabel='Heldout question-macro MAE (tokens)',title='Fixed 20-question holdout')
    axes[0].legend();axes[0].grid(alpha=.2)
    values=defaultdict(list)
    for result in summary['results']:
        if 'h3' in result['paired_delta']:
            values[result['variant']].append(result['paired_delta']['h3']['delta_macro_mae'])
    names=[v for v in summary['variants'] if v in values and v!='full']
    means=[np.mean(values[v]) for v in names]
    axes[1].barh(names,means,color=['#d95f02' if x>0 else '#1b9e77' for x in means])
    axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Removal MAE minus full MAE (tokens)',title='H3 matched retraining; mean across seeds')
    axes[1].tick_params(axis='y',labelsize=7)
    fig.tight_layout();fig.savefig(output/'learning_and_factor_impact.png',dpi=170);plt.close(fig)
    (output/'READ_RESULTS.md').write_text(
        'Use evaluation/learning_*.json for the SAME heldout roots at every stage. '
        'H0 is current acceptance; H1/H2/H3 are individual R/E actions. '
        'Check MAE, p90, bias, within-1/2, prefix Brier and separate local agreement AUC.\n\n'
        'factor_audit/comparison.csv reports independently retrained single-factor removals. '
        'A positive paired delta means the removed factor helped full. Confidence intervals '
        'resample questions, not correlated tokens. inference_sensitivity.json is an immediate '
        'input lesion, not the independently retrained experiment.\n\n'
        'Direct verifier targets are sampled: only actual STOP and 15% shadow probes supply '
        'margin/hidden/local-agreement labels. Other K labels can come from verified greedy '
        'hindsight. Missing labels are never zero. Root sampling includes a quarter teacher '
        'states, so this audit is not a natural-frequency prevalence estimate.\n\n'
        'Verifier final hidden is a fixed 32D projection, not full raw hidden. The GRU '
        're-encodes the preceding 8 actual STOPs. Masks and proposal length also remain legal '
        'transition metadata when mask-input features are ablated. This is a representation '
        'and dynamics study; it does not yet measure scheduler throughput.\n',encoding='utf-8')
