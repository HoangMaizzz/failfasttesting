"""Question-level paired native-verifier probe metrics; no LLM/model imports.

q is marginal prefix SURVIVAL, never a local post-rejection agreement or hazard.
E_NEW_BLOCK uses the same globally defined q, not a reset product at the boundary.
"""
from __future__ import annotations
from collections import defaultdict
import math
import random

from factorized_wm_metrics import binary_metrics, calibration

COHORTS=('ALL','R_ALL','R_GAIN','R_LOSS','E_ALL','E_PARENT_FULL_PREFIX','E_NEW_BLOCK')


def cohort(rows, name):
    if name not in COHORTS:
        raise ValueError('Unknown native cohort')
    if name=='ALL': return list(rows)
    if name.startswith('R'):
        result=[r for r in rows if r['action']=='R']
        if name=='R_GAIN':
            result=[r for r in result if r['parent_K'] is not None and r['K_true']>r['parent_K']]
        if name=='R_LOSS':
            result=[r for r in result if r['parent_K'] is not None and r['K_true']<r['parent_K']]
        return result
    result=[r for r in rows if r['action']=='E']
    if name=='E_PARENT_FULL_PREFIX':
        return [r for r in result if r['parent_K'] is not None and r['parent_K']==r['parent_length']]
    if name=='E_NEW_BLOCK':
        return [r for r in result if r['parent_length'] is not None and r['parent_length']<r['proposal_length']]
    return result


def validate_records(rows):
    seen=set()
    for r in rows:
        key=(r['method'],r['seed'],r['state_id'])
        if key in seen: raise ValueError('Duplicate native prediction key')
        seen.add(key)
        n=r['proposal_length']; k=r['K_true']; q=r['q_pred']
        if type(n) is not int or not 1<=n<=64 or type(k) is not int or not 0<=k<=n:
            raise ValueError('Invalid native length/acceptance')
        if len(q)!=n or any(not math.isfinite(v) or not 0<=v<=1 for v in q):
            raise ValueError('Invalid marginal survival vector')
        if not math.isfinite(r['K_pred']) or abs(r['K_pred']-sum(q))>1e-4:
            raise ValueError('Expected accepted length must equal sum of marginal survival')


def mean(values): return sum(values)/len(values) if values else None


def state_metrics(rows, positions_only_new=False):
    errors=defaultdict(list); bias=defaultdict(list)
    truth=[]; probabilities=[]; violations=0; positions=0
    for r in rows:
        errors[r['question_id']].append(abs(r['K_pred']-r['K_true']))
        bias[r['question_id']].append(r['K_pred']-r['K_true'])
        start=r['parent_length'] if positions_only_new else 0
        if start is None: raise ValueError('New-block metrics require the parent boundary')
        q=r['q_pred']; k=r['K_true']
        truth.extend(int(i<k) for i in range(start,len(q)))
        probabilities.extend(q[start:]); positions+=len(q)-start
        violations+=sum(q[i+1]>q[i]+1e-6 for i in range(len(q)-1))
    b=binary_metrics(truth,probabilities)
    return dict(status='ok' if rows else 'empty',
        num_states=len(rows),num_positions=positions,num_questions=len(errors),
        K_MAE_question_macro=None if positions_only_new else mean([mean(v) for v in errors.values()]),
        K_bias_question_macro=None if positions_only_new else mean([mean(v) for v in bias.values()]),
        K_bias_state_micro=None if positions_only_new else mean([v for arr in bias.values() for v in arr]),
        survival_brier=b.get('brier'),survival_auc=b.get('roc_auc'),
        survival_nll=b.get('nll'),survival_ece=b.get('ece'),
        survival_counts=dict(total=len(truth),positive=sum(truth),negative=len(truth)-sum(truth)),
        monotonicity_violations=violations,
        semantics='E_NEW_BLOCK: globally defined survival restricted to new positions; no K metric'
            if positions_only_new else 'q_i=P(K>=i+1); Khat=sum_i q_i; no monotonic projection',
        calibration=calibration(truth,probabilities))


def report(rows):
    validate_records(rows)
    return {name:state_metrics(cohort(rows,name),name=='E_NEW_BLOCK') for name in COHORTS}


def per_method_report(rows):
    validate_records(rows)
    groups=defaultdict(list)
    for r in rows: groups[(r['method'],r['seed'])].append(r)
    coverage=defaultdict(set)
    for r in rows:coverage[r['method']].add((r['seed'],r['state_id']))
    if coverage and any(v != next(iter(coverage.values())) for v in coverage.values()):
        raise ValueError('Methods must have identical seed/state coverage; unmatched predictions refused')
    return {method:{str(seed):report(group) for (name,seed),group in groups.items() if name==method}
            for method in sorted({key[0] for key in groups})}


def seed_summary(reports, method, cohort_name='ALL'):
    values=[v[cohort_name]['K_MAE_question_macro'] for v in reports.get(method,{}).values()]
    values=[v for v in values if v is not None]
    avg=mean(values)
    return dict(mean=avg,seed_count=len(values),
        std=None if not values else math.sqrt(mean([(v-avg)**2 for v in values])),
        values=values)


def paired_bootstrap(rows, method_a, method_b, cohort_name='ALL', samples=2000, seed=42):
    """Average matched seed errors per state, then resample whole questions.

    This averages errors across random seeds; it does NOT silently evaluate a
    probability-averaged ensemble, nor multiply the effective question count.
    """
    validate_records(rows)
    coverage_a={(r['seed'],r['state_id']) for r in rows if r['method']==method_a}
    coverage_b={(r['seed'],r['state_id']) for r in rows if r['method']==method_b}
    if coverage_a != coverage_b:
        raise ValueError('Paired comparisons require identical seed/state coverage before cohort filtering')
    selected=cohort(rows,cohort_name)
    a={(r['seed'],r['state_id']):r for r in selected if r['method']==method_a}
    b={(r['seed'],r['state_id']):r for r in selected if r['method']==method_b}
    keys=set(a)&set(b)
    if set(a) != set(b):
        raise ValueError('Paired comparisons require identical seed/state coverage; unmatched predictions refused')
    states=defaultdict(list)
    for key in sorted(keys):
        left,right=a[key],b[key]
        if (left['action'],left.get('split')) != (right['action'],right.get('split')):
            raise ValueError('Paired methods disagree about state action or split')
        if (left['question_id'],left['K_true'],left['proposal_length'],left['parent_length'],left['parent_K']) != (
                right['question_id'],right['K_true'],right['proposal_length'],right['parent_length'],right['parent_K']):
            raise ValueError('Paired methods disagree about the real state')
        states[(left['question_id'],left['state_id'])].append((
            abs(left['K_pred']-left['K_true']),abs(right['K_pred']-right['K_true'])))
    questions=defaultdict(list)
    for (question,uid),values in states.items():
        questions[question].append((mean([v[0] for v in values]),mean([v[1] for v in values])))
    units=[dict(question=q,num_states=len(values),mae_a=mean([v[0] for v in values]),
                mae_b=mean([v[1] for v in values])) for q,values in sorted(questions.items())]
    deltas=[v['mae_a']-v['mae_b'] for v in units]
    rng=random.Random(seed)
    draws=sorted(mean([rng.choice(deltas) for _ in deltas]) for _ in range(samples)) if deltas else []
    def quantile(p):
        if not draws:return None
        x=p*(len(draws)-1);i=int(x);j=min(i+1,len(draws)-1)
        return draws[i]+(draws[j]-draws[i])*(x-i)
    return dict(method_a=method_a,method_b=method_b,cohort=cohort_name,
        direction='A-minus-B; negative favors A',status='ok' if units else 'empty',
        matched_seed_state_count=len(keys),unmatched_a=len(set(a)-set(b)),unmatched_b=len(set(b)-set(a)),
        num_states=len(states),num_questions=len(units),mae_a=mean([u['mae_a'] for u in units]),
        mae_b=mean([u['mae_b'] for u in units]),delta_MAE=mean(deltas),
        ci95=[quantile(.025),quantile(.975)] if draws else None,
        bootstrap_unit='question',samples=samples,seed=seed,question_units=units)


def select_layers(validation_rows, depths, method_for_depth, count=2):
    """Mean seed-level ALL/E-full-prefix question-macro MAE, validation only."""
    if any(r.get('split','val') not in ('val','validation') for r in validation_rows):
        raise ValueError('Layer selection can only use validation rows')
    reports=per_method_report(validation_rows);rank=[]
    for depth in depths:
        method=method_for_depth[depth]
        all_score=seed_summary(reports,method,'ALL')['mean']
        e_score=seed_summary(reports,method,'E_PARENT_FULL_PREFIX')['mean']
        if all_score is None:raise ValueError('A native layer lacks validation predictions')
        rank.append(dict(depth=depth,method=method,ALL=all_score,E_PARENT_FULL_PREFIX=e_score,
                         score=mean([v for v in (all_score,e_score) if v is not None])))
    rank.sort(key=lambda r:(r['score'],r['depth']))
    return dict(selection_split='val',criterion='equal mean ALL/E-full-prefix question-macro K-MAE across seeds',
                selected=[r['depth'] for r in rank[:count]],ranking=rank)


def compression_comparison(reports, latent_method, raw_method):
    result={}
    for name in ('ALL','E_PARENT_FULL_PREFIX'):
        native=seed_summary(reports,raw_method,name)
        compact=seed_summary(reports,latent_method,name)
        raw,small=native['mean'],compact['mean']
        result[name]=dict(raw=native,latent=compact,
            degradation_fraction=None if raw is None or small is None or raw<=0 else (small-raw)/raw)
    return result


def feasibility_gate(validation_rows, raw_method, latent_method, direct_method, cfg, checks):
    if any(r.get('split','val') not in ('val','validation') for r in validation_rows):
        raise ValueError('Feasibility gate can only use validation rows')
    validate_records(validation_rows)
    coverage = {method:{(r['seed'],r['state_id']) for r in validation_rows if r['method']==method}
                for method in (raw_method,latent_method,direct_method)}
    if not coverage[raw_method] or any(v != coverage[raw_method] for v in coverage.values()):
        raise ValueError('Feasibility gate requires identical direct/raw/latent seed/state coverage')
    r=per_method_report(validation_rows)
    raw=seed_summary(r,raw_method,'E_PARENT_FULL_PREFIX')['mean']
    latent=seed_summary(r,latent_method,'E_PARENT_FULL_PREFIX')['mean']
    direct=seed_summary(r,direct_method,'E_PARENT_FULL_PREFIX')['mean']
    comparison=paired_bootstrap(validation_rows,raw_method,direct_method,'ALL',
                                cfg['bootstrap_samples'],42)
    e_comparison=paired_bootstrap(validation_rows,latent_method,direct_method,'E_PARENT_FULL_PREFIX',
                                 cfg['bootstrap_samples'],42)
    all_compression=compression_comparison(r,latent_method,raw_method)
    counts=[v['E_PARENT_FULL_PREFIX'] for v in r.get(latent_method,{}).values()]
    sufficient=bool(counts) and all(c['num_states']>=cfg['gate_min_E_states'] and
        c['num_questions']>=cfg['gate_min_E_questions'] for c in counts)
    seed_checks=[];bias=[]
    for seed,values in r.get(latent_method,{}).items():
        d=r.get(direct_method,{}).get(seed,{}).get('E_PARENT_FULL_PREFIX',{}).get('K_MAE_question_macro')
        v=values['E_PARENT_FULL_PREFIX']['K_MAE_question_macro']
        seed_checks.append(d is not None and d>0 and v is not None and v<d)
        bias.extend(values[name]['K_bias_question_macro'] for name in ('ALL','E_PARENT_FULL_PREFIX'))
    retained=None if direct is None or raw is None or latent is None or direct<=raw else (direct-latent)/(direct-raw)
    conditions=dict(
        required_sanity_checks=all(checks.values()),
        native_beats_direct_ALL_CI=comparison['ci95'] is not None and comparison['ci95'][1]<0,
        native_beats_direct_E=raw is not None and direct is not None and raw<direct,
        latent_improves_E_15pct=direct is not None and direct>0 and latent is not None and
            (direct-latent)/direct>=cfg['gate_E_improvement'],
        latent_E_paired_CI=e_comparison['ci95'] is not None and e_comparison['ci95'][1]<0,
        compression_within_10pct=all(v['degradation_fraction'] is not None and
            v['degradation_fraction']<=cfg['gate_compression_degradation'] for v in all_compression.values()),
        retains_native_advantage=retained is not None and retained>=cfg['gate_advantage_retention'],
        no_catastrophic_positive_bias=bool(bias) and all(v is not None and v<=cfg['gate_max_positive_bias'] for v in bias),
        three_seed_consistency=len(seed_checks)>=3 and all(seed_checks),
        enough_E_full_prefix=sufficient)
    status='PIPELINE_ONLY' if cfg.get('pipeline_check_only') else 'INVALID' if not all(checks.values()) else (
        'INCONCLUSIVE' if not sufficient else 'PASS' if all(conditions.values()) else 'FAIL')
    return dict(status=status,selection_split='val',raw_method=raw_method,latent_method=latent_method,
        direct_method=direct_method,conditions=conditions,checks=checks,
        E_improvement_fraction=None if direct is None or direct<=0 or latent is None else (direct-latent)/direct,
        retained_advantage_fraction=retained,compression=all_compression,
        native_vs_direct_ALL=comparison,latent_vs_direct_E=e_comparison,
        interpretation='Feasibility gate is preregistered on validation; held-out test is a separate confirmation')
