"""Fresh online positive control on a frozen, post-hoc selected ID order."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import run_math_learner_positive_control as pc

IDS = [299,179,474,96,100,218,441,459,71,194,88,442,188,80,385,119,
       396,217,237,34,488,388,168,489,343,315,151,356,26,49,32,431,
       329,374,312,139,136,404,248,37,321,357,122,368,366,394,418,351,393,166]
METHODS = ('always_stop', 'u1', 'probe_only')


def command(args, method, dest):
    cmd = pc.method_cmd(args, 'always_stop' if method == 'always_stop' else 'u1',
                        IDS, dest, args.seed, .5)
    if method == 'probe_only':
        cmd.append('--adaptive-hindsight-logistic-probe-only')
    return cmd


def compare(left, right):
    a, b = pc.load_bench(left), pc.load_bench(right)
    pairs = a.merge(b, on='problem_id', suffixes=('_baseline', '_method'), validate='one_to_one')
    def mspt(frame):
        return float(1000 * frame.actual_e2e_time.sum() / frame.output_tokens.sum())
    same = pairs.output_token_hash_baseline.astype(str) == pairs.output_token_hash_method.astype(str)
    return {'baseline_ms_per_token': mspt(a), 'method_ms_per_token': mspt(b),
            'speedup': mspt(a) / mspt(b), 'output_matches': int(same.sum()),
            'problems': len(pairs), 'lossless_comparison': bool(same.all())}, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dllm_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--target_quantization', default='int8', choices=['int8', 'none'])
    p.add_argument('--target_device', type=int, default=0)
    p.add_argument('--drafter_device', type=int, default=0)
    p.add_argument('--drafter_threshold', type=float, default=.5)
    p.add_argument('--lowconf_threshold', type=float, default=.7)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--log_level', default='INFO')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    args.output_dir = str(out)
    assert len(IDS) == len(set(IDS)) == 50
    if args.dry_run:
        for method in METHODS:
            print(' '.join(map(str, command(args, method, out / 'raw' / method))))
        return
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'dry_run')}
    config['problem_ids'] = IDS
    config['source_hashes'] = {
        name: hashlib.sha256((pc.ROOT / name).read_bytes()).hexdigest()
        for name in ('adaptive_td.py', 'failfast.py', 'run_math50_witness_online.py',
                     'run_math_learner_positive_control.py', 'patch_fastdllm_frontier.py')}
    config_path = out / 'CONFIG.json'
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise RuntimeError('Existing run differs: use its original configuration or a new output_dir')
    pc.jdump(config_path, config)
    pd.DataFrame({'position': range(50), 'problem_id': IDS}).to_csv(out / 'FIXED_SEQUENCE.csv', index=False)
    subprocess.run([sys.executable, str(pc.ROOT / 'patch_fastdllm_frontier.py'), args.dllm_dir], check=True)
    results = {}
    try:
        for method in METHODS:
            case = out / 'raw' / method
            if not (args.resume and pc.complete(case, IDS)):
                if case.exists():
                    case.rename(case.with_name(method + '_incomplete_' + str(time.time_ns())))
                pc.run(command(args, method, case), out / 'logs' / (method + '.log'))
            if not pc.complete(case, IDS):
                raise RuntimeError('Incomplete or reordered benchmark: ' + str(case))
            result = pc.summary(case, IDS)
            bench = pc.load_bench(case)
            result['e2e_ms_per_token'] = float(bench.actual_e2e_time.sum() * 1000 / bench.output_tokens.sum())
            result['total_e2e_seconds'] = float(bench.actual_e2e_time.sum())
            result['method'] = method
            decisions_path = case / 'adaptive_td_decisions.csv'
            if decisions_path.exists():
                decisions = pd.read_csv(decisions_path)
                if 'action_source' in decisions:
                    counts = decisions.action_source.value_counts().to_dict()
                    result['action_sources'] = counts
                    if method == 'probe_only' and counts.get('learned_continue', 0):
                        raise RuntimeError('Probe-only unexpectedly executed learned CONTINUE')
            results[method] = result
            pc.jdump(out / 'LEARNING_SUMMARY.json', results)
            pd.DataFrame(results.values()).to_csv(out / 'summary.csv', index=False)
            shutil.make_archive(str(out), 'zip', root_dir=out)
        comparisons = {}
        for baseline in ('always_stop', 'probe_only'):
            metrics, pairs = compare(out / 'raw' / baseline, out / 'raw' / 'u1')
            comparisons[baseline] = metrics
            pairs.to_csv(out / ('paired_vs_' + baseline + '.csv'), index=False)
        pc.jdump(out / 'COMPARISONS.json', comparisons)
        (out / 'INTERPRETATION.txt').write_text(
            'Post-hoc selected positive control, not held-out generalization.\n'
            'Learners start from zero; no archived labels or weights are loaded.\n'
            'Local deltaJ sums are not additive E2E milliseconds.\n'
            'E2E includes transfer and controller cost. INT8 does not guarantee output equality.\n'
            'Resume skips only complete methods; an incomplete method restarts from zero.\n', encoding='utf-8')
    finally:
        print('Archive:', shutil.make_archive(str(out), 'zip', root_dir=out), flush=True)


if __name__ == '__main__':
    main()
