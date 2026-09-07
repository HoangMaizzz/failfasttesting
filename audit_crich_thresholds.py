"""Read-only logged-support threshold audit; not counterfactual policy replay."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def is_true(value):
    return str(value).lower() in ('true', '1', '1.0')


def clean_rows(records, ids, exclude_cold=False):
    result = []
    for row in records:
        if not is_true(row.get('pair_resolved')):
            continue
        if int(row['problem_id']) not in ids:
            continue
        if exclude_cold and row['action_source'].startswith('cold_start'):
            continue
        score = number(row['continue_score_before_update'])
        dj = number(row['delta_J_ms_per_token'])
        prop = number(row['behavior_continue_probability'])
        if not (math.isfinite(score) and math.isfinite(dj) and math.isfinite(prop)
                and 0 <= score <= 1 and 0 < prop <= 1):
            raise ValueError('Invalid resolved pair score, utility or propensity')
        result.append(dict(pid=int(row['problem_id']), score=score, dj=dj, prop=prop))
    return result


def tail(records, threshold):
    selected = [r for r in records if r['score'] > threshold]
    if not selected:
        return dict(n=0, problems=0, mean_dj=None, ipw_mean_dj=None,
                    good_c=0, tie=0, effective_sample_size=0.)
    weights = [min(50., 1 / r['prop']) for r in selected]
    total = sum(weights)
    return dict(n=len(selected), problems=len({r['pid'] for r in selected}),
                mean_dj=sum(r['dj'] for r in selected)/len(selected),
                ipw_mean_dj=sum(r['dj']*w for r,w in zip(selected,weights))/total,
                good_c=sum(r['dj'] < -1 for r in selected),
                tie=sum(abs(r['dj']) <= 1 for r in selected),
                effective_sample_size=total**2/sum(w*w for w in weights))


def eligible(stats):
    return all(s['n'] >= 10 and s['problems'] >= 3 for s in stats)


def profitable(stats):
    return eligible(stats) and all(s['mean_dj'] < 0 and s['ipw_mean_dj'] < 0 for s in stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('stable_crich50_report.zip'))
    parser.add_argument('--output', type=Path, default=Path('outputs_crich_threshold_audit'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, scan = [], []
    with zipfile.ZipFile(args.archive) as archive:
        for attempt in range(1,5):
            sequence = rows(archive, f'confirm/attempt_{attempt}/candidate_sequence.csv')
            ids = [int(r['problem_id']) for r in sorted(sequence, key=lambda r:int(r['run_position']))]
            if len(set(ids)) != 50:
                raise ValueError('Expected 50 unique IDs')
            for scope in ('eval30', 'eval30_non_cold', 'full50'):
                data = []
                for seed in (42,43):
                    record = rows(archive, f'confirm/attempt_{attempt}/seed_{seed}/u1_current_tau05/adaptive_full_stream_transitions.csv')
                    data.append(clean_rows(record, set(ids if scope == 'full50' else ids[20:]), scope.endswith('non_cold')))
                # Every observed score is a change point for the strict > rule.
                thresholds = sorted({0., 0.5, 1., *[i/200 for i in range(201)],
                                     *[r['score'] for d in data for r in d]})
                supported, passing = [], []
                for threshold in thresholds:
                    stats = [tail(d, threshold) for d in data]
                    item = dict(attempt=attempt, scope=scope, threshold=threshold,
                                seed42=stats[0], seed43=stats[1], sufficient_support=eligible(stats),
                                passes_local_gate=profitable(stats))
                    scan.append(item)
                    if eligible(stats):
                        supported.append(item)
                    if profitable(stats):
                        passing.append(item)
                # Least-bad diagnostic only, never auto-promoted to a new policy.
                best = min(supported, key=lambda r: max(r['seed42']['mean_dj'], r['seed43']['mean_dj'])) if supported else None
                summary.append(dict(attempt=attempt, scope=scope, scanned=len(thresholds),
                                    supported=len(supported), passing=len(passing),
                                    least_bad=best, fixed05=[tail(d,.5) for d in data]))
    report = dict(source=str(args.archive), source_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  rule='CONTINUE iff pre-update score > threshold',
                  sign='negative delta_J is locally beneficial', include_ties=True,
                  support_gate='>=10 resolved pairs across >=3 problems in EACH seed',
                  propensity_sensitivity='self-normalized inverse action probability, clipped at 50',
                  caveat='Conditional on logged states and resolved feedback; not unbiased policy value, not speedup. Censoring and changed trajectories are uncorrected. Threshold search is post-hoc, not held-out validation.',
                  summaries=summary)
    (args.output/'audit_summary.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    (args.output/'all_thresholds.json').write_text(json.dumps(scan,indent=2), encoding='utf-8')
    lines = ['# C-rich threshold audit', '', report['caveat'], '',
             'The scan preserves all four existing 50-ID sequences and their order.',
             'It does not reselect problems to maximize the observed outcome.',
             'Primary scope: last 30 problems. Ties are included in utility, not discarded.', '',
             '| Attempt | Supported cutoffs | Passing cutoffs | Least-bad cutoff | Mean dJ seed42 | Mean dJ seed43 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        if row['scope'] != 'eval30':
            continue
        best = row['least_bad']
        if best:
            lines.append(f"| {row['attempt']} | {row['supported']} | {row['passing']} | {best['threshold']:.6f} | {best['seed42']['mean_dj']:.3f} | {best['seed43']['mean_dj']:.3f} |")
    passes = sum(r['passing'] for r in summary if r['scope'] == 'eval30')
    lines.extend(['', f'Primary passing cutoffs across attempts: {passes}.',
                  'No alternate threshold or winning 50-ID set is frozen automatically.',
                  'Do not interpret sums or means of local dJ as end-to-end milliseconds.',
                  'A supported negative mean would only justify a prospective GPU test, not prove superiority.'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
