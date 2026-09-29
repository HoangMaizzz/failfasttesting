"""Reports and paired question-bootstrap comparisons for the retraining study."""
from collections import defaultdict
import gzip
import json
import numpy as np


def question_errors(out, point):
    by_q=defaultdict(list)
    identities=set()
    with gzip.open(out/point['predictions'],'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            key=(r['scope'],r['group'],r['question'])
            by_q[key].append(abs(r['expected_K']-r['K']))
            if r['depth']==1:
                change=r.get('change')
                if change in ('gain','same','loss'):
                    by_q[(r['scope'],r['group']+'_'+change,r['question'])].append(
                        abs(r['expected_K']-r['K']))
            identities.add((r['scope'],r['group'],r['question'],r['source'],r['state_id'],r['depth'],r['actions']))
    return {k:float(np.mean(v)) for k,v in by_q.items()}, identities


def paired_difference(out, points_a, points_b, cache, bootstraps=2000):
    """Average seeds within each question, then bootstrap paired questions.

    This CI is conditional on the available seeds/split. Seed spread is reported
    separately; it is not a claim of population certainty or a corrected p-value.
    """
    if not points_a or [p['seed'] for p in points_a]!=[p['seed'] for p in points_b]:
        raise ValueError('Paired comparisons require identical nonempty seed sets')
    groups=defaultdict(lambda:defaultdict(dict))
    for a,b in zip(points_a,points_b):
        if a['seed']!=b['seed']: raise ValueError('Unpaired seeds')
        def load(p):
            if p['predictions'] not in cache: cache[p['predictions']]=question_errors(out,p)
            return cache[p['predictions']]
        ea,ia=load(a); eb,ib=load(b)
        # Full cohort occurs only at the final update. Compare identical scopes.
        common_scopes={k[0] for k in ea}&{k[0] for k in eb}
        if {i for i in ia if i[0] in common_scopes}!={i for i in ib if i[0] in common_scopes}:
            raise ValueError('Evaluation populations differ in a paired comparison')
        for scope,group,q in ea.keys()&eb.keys():
            groups[scope+'/'+group][a['seed']][q]=ea[(scope,group,q)]-eb[(scope,group,q)]
    result={}; rng=np.random.default_rng(1742)
    for group,seeds in groups.items():
        common_q=sorted(set.intersection(*(set(q) for q in seeds.values())))
        d=np.array([[seed[q] for q in common_q] for seed in seeds.values()])
        avg=d.mean(0)
        draws=avg[rng.integers(0,len(avg),(bootstraps,len(avg)))].mean(1)
        result[group]=dict(delta_mae=float(avg.mean()),questions=len(common_q),seeds=len(seeds),
            ci95_questions=np.quantile(draws,[.025,.975]).tolist(),
            delta_per_seed=d.mean(1).tolist(),favorable_seeds=int((d.mean(1)<0).sum()))
    return result


def create_report(out, summary):
    points=summary['points']; names=list(summary['variants'])
    final=max(p['update'] for p in points)
    cache={}; impacts={}; progress={}
    def select(name,update):
        return sorted([p for p in points if p['variant']==name and p['update']==update],key=lambda p:p['seed'])
    for name in names:
        reference='improved' if (name.startswith('no_') or name in
            ('action_balanced','change_balanced','change_balanced_delta')) else 'baseline'
        if name!=reference and reference in names:
            impacts[name+'_minus_'+reference]=paired_difference(out,select(name,final),select(reference,final),cache)
        steps=sorted({p['update'] for p in points if p['variant']==name})
        for before,after in zip(steps,steps[1:]):
            progress[f'{name}/{before}_to_{after}']=paired_difference(out,select(name,after),select(name,before),cache)
    (out/'paired_factor_impacts.json').write_text(json.dumps(impacts,indent=2),encoding='utf-8')
    (out/'paired_learning_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
    curves=[]
    for name in names:
        for update in sorted({p['update'] for p in points if p['variant']==name}):
            ps=select(name,update); metrics={}
            for group in sorted({k for p in ps for k in p['metrics']}):
                values=[p['metrics'][group]['question_macro_mae'] for p in ps if group in p['metrics']]
                metrics[group]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                                   seeds=len(values),n_per_seed=[p['metrics'][group]['n'] for p in ps if group in p['metrics']])
            curves.append(dict(variant=name,update=update,metrics=metrics))
    (out/'curves.json').write_text(json.dumps(curves,indent=2),encoding='utf-8')
    groups=['full/current','full/h1_R','full/h1_E','rollout_panel/h3']
    def fmt(name,group):
        entry=next(c for c in curves if c['variant']==name and c['update']==final)['metrics'].get(group)
        return '-' if entry is None else f"{entry['mean']:.3f} +/- {entry['std']:.3f}"
    lines=['# Retrained world-model improvement study','',
        'Results below use question-macro acceptance MAE (tokens); smaller is better.',
        'Current and R/E final columns include ALL labeled held-out states/edges. H3 uses the fixed real-path panel.',
        'Intermediate curves always use the same panel. Do not compare full-population final MAE to earlier panel MAE.', '',
        f"Train: {summary['train_questions']} questions; validation: {summary['validation_questions']}; final update: {final}.",
        f"Teacher targets restored: train {summary['teacher_train']}, validation {summary['teacher_validation']}.",
        f"Validation edges: {summary['full_validation_edges']}; changed-K edges: {summary['changed_validation_edges']}.", '',
        '| Retrained arm | Current | R1 | E1 | H3 |','|---|---:|---:|---:|---:|']
    for name in names: lines.append('| '+name+' | '+' | '.join(fmt(name,g) for g in groups)+' |')
    lines+=['','## Paired factor effects','',
        'Delta = first arm minus reference. Negative is lower error. Intervals bootstrap questions after averaging seeds.',
        'For removal arms, positive delta means removing that component hurt the combined model.', '',
        '| Comparison | Current delta [95% CI] | R1 delta [95% CI] | E1 delta [95% CI] |',
        '|---|---:|---:|---:|']
    for pair,metrics in impacts.items():
        cells=[]
        for group in groups[:3]:
            m=metrics.get(group)
            cells.append('-' if m is None else f"{m['delta_mae']:+.3f} [{m['ci95_questions'][0]:+.3f}, {m['ci95_questions'][1]:+.3f}]")
        lines.append('| '+pair+' | '+' | '.join(cells)+' |')
    sampler_names=('action_balanced','change_balanced','change_balanced_delta')
    if 'improved' in names and any(n in names for n in sampler_names):
        lines+=['','## Sampling effects on changed-K edges','',
            'Each sampler is compared with `improved` (natural edge sampling). Negative delta means lower MAE.',
            '| Sampler | R gain ΔMAE | R same ΔMAE | R loss ΔMAE | E gain ΔMAE | E same ΔMAE | E loss ΔMAE |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in sampler_names:
            if name not in impacts: continue
            metrics=impacts[name+'_minus_improved']; cells=[]
            for group in ('full/h1_R_gain','full/h1_R_same','full/h1_R_loss',
                          'full/h1_E_gain','full/h1_E_same','full/h1_E_loss'):
                m=metrics.get(group)
                cells.append('-' if m is None else
                    f"{m['delta_mae']:+.3f} [{m['ci95_questions'][0]:+.3f}, {m['ci95_questions'][1]:+.3f}]")
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        lines+=['','Root-edge draws actually used by each sampler at the final update, summed over seeds:','',
            '| Sampler | Root edge action/change counts |','|---|---|']
        for name in ('improved',)+sampler_names:
            if name not in names: continue
            counts=defaultdict(int)
            for point in select(name,final):
                for key,value in point.get('sampled_root_edge_counts',{}).items(): counts[key]+=value
            lines.append(f"| {name} | `"+json.dumps(dict(sorted(counts.items())),sort_keys=True)+'` |')
    lines+=['','## Learning over updates','',
        '| Arm | Update | Panel current MAE | Panel R1 | Panel E1 | Panel H3 |','|---|---:|---:|---:|---:|---:|']
    for c in curves:
        cells=[]
        for group in ('panel/current','panel/h1_R','panel/h1_E','rollout_panel/h3'):
            v=c['metrics'].get(group); cells.append('-' if v is None else f"{v['mean']:.3f} +/- {v['std']:.3f}")
        lines.append(f"| {c['variant']} | {c['update']} | "+' | '.join(cells)+' |')
    lines+=['','## Interpretation and limits','',
        '- Baseline reproduces the previous offline target setup: no verifier-teacher loss. Teacher-only isolates restoring those targets.',
        '- Every arm is retrained from matched shared initialization with the same question split, replay seed, and update budget.',
        '- Single-addition arms estimate effects against baseline; removal arms estimate effects conditional on the combined model. This is not a full factorial interaction experiment.',
        '- Dropping gaps removes explicit gap features and candidate probability weights. Scalar confidence/history and ordered top-K membership can still carry related information.',
        '- Hidden removal and prefix/history removal are retrained input ablations; masks, lengths, and validity remain factual.',
        '- Report each loss, NLL, P90, within-1/2, bias, delta-K MAE, persistence MAE, mask Brier and active-token latent cosine in summary.json.',
        '- Latent cosine is a diagnostic within a representation; representations from different arms need not use the same coordinate system.',
        '- Teacher is an auxiliary training target only, never an encoder input. No teacher is recomputed.',
        '- Changed-edge diagnostics are selected by true labels for analysis only; they do not enter the train sampler or inference policy.',
        '- Balanced sampling uses training labels only to select a root edge. Current-state supervision uses an independent seeded RNG; validation sampling is untouched.',
        '- Confidence intervals are exploratory, conditional on this split/seeds, without multiple-comparison correction. This validation set was previously inspected; a new question-level test set is needed for a final claim.',
        '- Parameter count and wall/training times are recorded; equal optimizer updates do not imply equal compute across architectures.',
        '- No new LLM inference, latency speedup experiment, or online controller is run.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    focus=[n for n in ('baseline','improved','action_balanced','change_balanced',
                       'change_balanced_delta','attention_only','residual_only','teacher_only','delta_only') if n in names]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,group in zip(axes.flat,('panel/current','panel/h1_R','panel/h1_E','rollout_panel/h3')):
        for name in focus:
            data=[c for c in curves if c['variant']==name and group in c['metrics']]
            x=[c['update'] for c in data]; y=np.array([c['metrics'][group]['mean'] for c in data])
            sd=np.array([c['metrics'][group]['std'] for c in data])
            ax.plot(x,y,marker='.',label=name); ax.fill_between(x,y-sd,y+sd,alpha=.1)
        ax.set(title=group,xlabel='Optimizer updates',ylabel='Acceptance MAE (tokens)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for name in ('baseline','improved'):
        if name not in names: continue
        for group,style in (('train_monitor/current','--'),('panel/current','-')):
            data=[c for c in curves if c['variant']==name and group in c['metrics']]
            ax.plot([c['update'] for c in data],[c['metrics'][group]['mean'] for c in data],style,label=f'{name}: {group}')
    ax.set(xlabel='Optimizer updates',ylabel='Acceptance MAE (tokens)',title='Fixed train/validation panels')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out/'train_validation.png',dpi=160); plt.close(fig)
