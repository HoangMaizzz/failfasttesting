"""Single-GPU INT8 counterpart of the FP16/no-KV Kaggle positive control."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dllm_dir', required=True)
    p.add_argument('--target_model_name', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--output_dir', default=str(ROOT/'outputs_int8_nokv_math_pool'))
    p.add_argument('--pool_size', type=int, default=180)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pool_seed', type=int, default=20260907)
    p.add_argument('--max_candidates', type=int, default=5)
    p.add_argument('--search_restarts', type=int, default=16)
    p.add_argument('--search_iterations', type=int, default=5000)
    p.add_argument('--seed_build_attempts', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    if not 50 <= a.pool_size <= 499:
        p.error('pool_size must be 50..499')
    if min(a.max_candidates,a.search_restarts,a.search_iterations,a.seed_build_attempts) <= 0:
        p.error('Search budgets must be positive')
    return a


def commands(a):
    out = Path(a.output_dir).resolve()
    common = ['--dllm_dir', str(Path(a.dllm_dir).resolve()),
              '--target_model_name', a.target_model_name,
              '--target_device', '0', '--drafter_device', '0',
              '--target_quantization', 'int8', '--seed', str(a.seed),
              '--drafter_threshold', '.5', '--lowconf_threshold', '.7']
    if a.resume:
        common += ['--resume']
    discovery = [sys.executable, '-u', str(ROOT/'collect_fp16_nokv_discovery.py'), *common,
                 '--output_dir', str(out/'discovery'), '--pool_size', str(a.pool_size),
                 '--pool_seed', str(a.pool_seed)]
    search = [sys.executable, '-u', str(ROOT/'run_oracle_total_benefit_control.py'), *common,
              '--output_dir', str(out/'witness'), '--sources', str(out/'discovery'),
              '--target_dtype', 'fp16', '--drafter_dtype', 'fp16',
              '--replay_seeds', str(a.seed), '--max_candidates', str(a.max_candidates),
              '--search_restarts', str(a.search_restarts), '--search_iterations', str(a.search_iterations),
              '--seed_build_attempts', str(a.seed_build_attempts), '--offline_min_learned_c', '8',
              '--min_learned_c', '5', '--min_total_learned_benefit', '1.0',
              '--probe_soft_weight', '.5', '--nonprobe_natural_weight', '.25',
              '--benefit_reward', '.01', '--learned_c_reward', '.02']
    return discovery, search


def main():
    a = parse_args()
    cmds = commands(a)
    if a.dry_run:
        for cmd in cmds:
            print(' '.join(cmd))
        return
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0', WANDB_MODE='disabled',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    try:
        for index, cmd in enumerate(cmds):
            print('\n>>>', ' '.join(cmd), flush=True)
            with (out/'session.log').open('a', encoding='utf-8') as log:
                log.write('\n>>> ' + ' '.join(cmd) + '\n')
                with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f'Stage {index+1} failed ({code}); see {out}/session.log')
            if index == 0:
                shutil.make_archive(str(out/'discovery'), 'zip', root_dir=out/'discovery')
    finally:
        print('Report archive:', shutil.make_archive(str(out), 'zip', root_dir=out), flush=True)


if __name__ == '__main__':
    main()
