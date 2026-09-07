#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from oracle_run_guard import guard, preserve
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Run ONE broad FP16/no-KV MATH discovery stream. The stream uses the real '
            'U1 learner and native stochastic probes; hindsight logging records resolved '
            'STOP-vs-CONTINUE delta-J pairs whenever a CONTINUE branch is physically observed. '
            'The later optimizer chooses a 50-problem action-faithful witness from this large pool.'
        )
    )
    p.add_argument('--dllm_dir', default='/kaggle/working/Fast_dLLM_v2_1.5B')
    p.add_argument('--output_dir', default='/kaggle/working/fp16_nokv_single_discovery')
    p.add_argument('--target_device', type=int, default=0)
    p.add_argument('--drafter_device', type=int, default=1)
    p.add_argument('--target_model_name', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--two_gpu', action='store_true')
    p.add_argument('--pool_size', type=int, default=180,
                   help='Number of unique MATH problems in the single discovery stream; recommended 150-200.')
    p.add_argument('--candidate_id_min', type=int, default=1)
    p.add_argument('--candidate_id_max', type=int, default=499)
    p.add_argument('--pool_ids_file', default=None,
                   help='Optional text/CSV file containing candidate MATH problem IDs. If omitted, sample deterministically from the ID range.')
    p.add_argument('--seed', type=int, default=42,
                   help='Single discovery/generation/probe seed. No multi-seed discovery is performed.')
    p.add_argument('--pool_seed', type=int, default=20260907,
                   help='Seed only for selecting/shuffling the pool IDs when --pool_ids_file is omitted.')
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--drafter_threshold', type=float, default=0.50)
    p.add_argument('--lowconf_threshold', type=float, default=0.70)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log_level', default='INFO')
    return p.parse_args()


def read_pool_ids(a):
    if not 50 <= a.pool_size <= 499 or not 1 <= a.candidate_id_min <= a.candidate_id_max <= 499:
        raise ValueError('Use 50..499 MATH IDs in range 1..499 (0 reserved for warm-up)')
    if not 150 <= a.pool_size <= 200:
        print(f'[WARN] pool_size={a.pool_size}; recommended range is 150..200.', flush=True)
    if a.pool_ids_file:
        text = Path(a.pool_ids_file).read_text(encoding='utf-8')
        vals = []
        for token in text.replace(',', ' ').replace('\n', ' ').split():
            try:
                vals.append(int(token))
            except ValueError:
                pass
        vals = list(dict.fromkeys(vals))
        if len(vals) < a.pool_size:
            raise ValueError(f'pool_ids_file has only {len(vals)} unique integer IDs, need {a.pool_size}')
        if any(not 1 <= pid <= 499 for pid in vals):
            raise ValueError('Pool IDs must be in 1..499')
        return vals[:a.pool_size]

    universe = list(range(a.candidate_id_min, a.candidate_id_max + 1))
    if len(universe) < a.pool_size:
        raise ValueError('candidate ID range smaller than pool_size')
    rng = random.Random(a.pool_seed)
    pool = rng.sample(universe, a.pool_size)
    # The order is part of the observed discovery realization, so freeze it.
    rng.shuffle(pool)
    return pool


def complete(case: Path, ids):
    p = case / 'benchmark_results.csv'
    if not p.exists():
        return False
    try:
        import pandas as pd
        d = pd.read_csv(p)
        if 'mode' in d.columns:
            d = d[d['mode'] == 'dllm_ar']
        return len(d) == len(ids) and d.problem_id.astype(int).tolist() == list(ids)
    except Exception:
        return False


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 116)
    print('RUN:', ' '.join(map(str, cmd)))
    print('LOG:', log_path)
    print('=' * 116, flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def cmd_for(a, ids, dest):
    return [
        sys.executable, '-u', 'failfast.py',
        '--dataset_name', 'math',
        '--num_questions', str(len(ids)),
        '--problem_ids', *map(str, ids),
        '--warmup_questions', '1',
        '--benchmark_modes', 'dllm_ar',
        '--dllm_variant', 'failfast',
        '--decoding_strategy', 'greedy',
        '--max_new_tokens', str(a.max_new_tokens),
        '--spec_len', '8', '--block_size', '32', '--small_block_size', '8',
        '--target_model_name', str(a.target_model_name),
        '--dllm_dir', str(a.dllm_dir),
        '--target_device', str(a.target_device),
        '--drafter_device', str(a.drafter_device),
        '--target_quantization', 'none',
        '--unquantized_dtype', 'float16',
        *(['--target_two_gpu_fp16'] if a.two_gpu else []),
        '--disable_reusing_drafter_kvs',
        '--drafter_thresholds', str(a.drafter_threshold),
        '--sweep_lowconf_threshold', str(a.lowconf_threshold),
        '--sweep_max_spec_len', '64', '--sweep_incr_len', '8',
        '--seed', str(a.seed), '--quiet_generation', '--disable_progress',
        '--skip_artifacts', '--skip_plots', '--overwrite',
        '--output_dir', str(dest), '--log_level', a.log_level,
        '--adaptive-td',
        '--adaptive-feature-schema', 'otrc_v2_2_compact_td',
        '--adaptive-credit-assignment', 'hindsight_delta_j_logistic_f2',
        '--adaptive-policy-mode', 'hindsight_delta_j_logistic_f2',
        '--adaptive-policy-ablation', 'learned',
        '--adaptive-hindsight-logistic-tie-ms-per-token', '1.0',
        '--no-adaptive-hindsight-logistic-use-class-weight',
        '--no-adaptive-hindsight-logistic-use-prefix-feature',
        '--no-adaptive-hindsight-logistic-dynamic-threshold',
        '--adaptive-hindsight-logistic-utility-weighting', 'raw_abs',
        '--adaptive-hindsight-logistic-learning-rate', '0.05',
        '--adaptive-hindsight-logistic-continue-threshold', '0.5',
        '--adaptive-hindsight-delta-j-min-pairs', '30',
        '--adaptive-hindsight-delta-j-min-continue-pairs', '3',
        '--adaptive-hindsight-logistic-min-positive-problems', '2',
        '--adaptive-hindsight-delta-j-structural-probe', '0.08',
        '--adaptive-hindsight-delta-j-floor-probe', '0.02',
        '--adaptive-hindsight-logistic-replay-stop-to-continue-ratio', '0',
        '--no-adaptive-hindsight-logistic-balance-utility-mass',
        '--adaptive-hindsight-logistic-replay-batch-size', '16',
        '--adaptive-hindsight-logistic-replay-buffer-size', '100',
        '--adaptive-log-decisions', '--adaptive-profile-overhead',
        '--adaptive-hindsight-state-fingerprint',
    ]


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = read_pool_ids(a)
    guard(out, {**vars(a), 'ids': ids}, a.resume)
    subprocess.run([sys.executable, str(ROOT/'patch_fastdllm_frontier.py'), a.dllm_dir], check=True)
    case = out / f'u1_current_tau05_seed_{a.seed}_pool_{len(ids)}'

    manifest = {
        'discovery_design': 'single_broad_stream',
        'setting': {
            'target_quantization': 'none',
            'target_dtype': 'fp16',
            'drafter_dtype': 'fp16',
            'verifier_use_cache': False,
            'reusable_drafter_kvs': False,
            'structural_probe_probability': 0.08,
            'floor_probe_probability': 0.02,
            'u1_threshold': 0.5,
            'learning_rate': 0.05,
            'replay_batch': 16,
            'replay_buffer': 100,
        },
        'pool_size': len(ids),
        'problem_ids': ids,
        'seed': a.seed,
        'case': case.name,
        'notes': [
            'Only one discovery seed/stream is executed.',
            'Hindsight delta-J pairs are resolved for physically observed CONTINUE refinements; unresolved/tie rows are not treated as clean labels.',
            'The later search may reorder/select problems only when the stitched replay remains action-faithful; the final 50-problem witness is rerun physically from zero weights.',
        ],
    }
    (out / 'DISCOVERY_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'POOL_IDS.txt').write_text('\n'.join(map(str, ids)) + '\n', encoding='utf-8')

    if a.resume and complete(case, ids):
        print('[RESUME] reuse', case)
    else:
        preserve(case)
        run(cmd_for(a, ids, case), out / 'logs' / f'{case.name}.log')
    if not complete(case, ids):
        raise RuntimeError('Discovery output incomplete or reordered')

    print('\nSingle discovery complete:', out)
    print('Unique candidate problems:', len(ids))
    print('Use this directory as --sources for run_oracle_total_benefit_control.py')


if __name__ == '__main__':
    main()
