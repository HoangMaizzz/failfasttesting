"""Run from a pinned checkout with Kaggle Internet and GPU T4 x2 enabled."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path('/kaggle/working')
OUTPUT = WORK / 'math50_fp16_nokv_pool_results'
MODELS = WORK / 'math50_fp16_models'
POOL_SIZE = 180
MAX_CANDIDATES = 5
SEARCH_RESTARTS = 16
SEARCH_ITERATIONS = 5000
SEED_BUILD_ATTEMPTS = 10

env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0', WANDB_MODE='disabled',
           PYTORCH_ALLOC_CONF='expandable_segments:True')
OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def run(cmd):
    print('\n>>>', ' '.join(map(str, cmd)), flush=True)
    with (OUTPUT/'session.log').open('a', encoding='utf-8') as log:
        log.write('\n>>> ' + ' '.join(map(str, cmd)) + '\n')
        with subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f'Command failed ({code}); see session.log')


try:
    run(['nvidia-smi'])
    run([sys.executable, '-c',
         'import torch; print(torch.__version__); '
         'assert torch.cuda.device_count() >= 2, "Select GPU T4 x2"; '
         'print([(i,torch.cuda.get_device_name(i)) for i in range(2)])'])
    run([sys.executable, '-m', 'pip', 'install', 'transformers==4.53.3',
         'accelerate==1.10.1', 'datasets==4.2.0', 'einops==0.8.1',
         'pandas', 'numpy', 'matplotlib', 'safetensors'])
    target, drafter = MODELS/'target', MODELS/'drafter'
    for repo_id, destination in [('Qwen/Qwen2.5-7B-Instruct', target),
                                ('Efficient-Large-Model/Fast_dLLM_v2_1.5B', drafter)]:
        run([sys.executable, '-c',
             'from huggingface_hub import snapshot_download; '
             f'snapshot_download({repo_id!r}, local_dir={str(destination)!r})'])
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
         '-p', 'test_fp16_oracle_pool.py'])
    common = ['--dllm_dir', drafter, '--target_model_name', target,
              '--target_device', '0', '--drafter_device', '1', '--two_gpu',
              '--seed', '42', '--drafter_threshold', '.5', '--lowconf_threshold', '.7',
              '--resume']
    run([sys.executable, '-u', 'collect_fp16_nokv_discovery.py', *common,
         '--output_dir', OUTPUT/'discovery', '--pool_size', str(POOL_SIZE)])
    shutil.make_archive(str(OUTPUT/'discovery'), 'zip', root_dir=OUTPUT/'discovery')
    run([sys.executable, '-u', 'run_oracle_total_benefit_control.py', *common,
         '--output_dir', OUTPUT/'witness', '--sources', OUTPUT/'discovery',
         '--target_quantization', 'none', '--target_dtype', 'fp16', '--drafter_dtype', 'fp16',
         '--replay_seeds', '42', '--max_candidates', str(MAX_CANDIDATES),
         '--search_restarts', str(SEARCH_RESTARTS), '--search_iterations', str(SEARCH_ITERATIONS),
         '--seed_build_attempts', str(SEED_BUILD_ATTEMPTS), '--offline_min_learned_c', '8',
         '--min_learned_c', '5', '--min_total_learned_benefit', '1.0',
         '--probe_soft_weight', '.5', '--nonprobe_natural_weight', '.25',
         '--benefit_reward', '.01', '--learned_c_reward', '.02'])
finally:
    archive = shutil.make_archive(str(OUTPUT), 'zip', root_dir=OUTPUT)
    print('\nReport archive:', archive, flush=True)
    from IPython.display import FileLink, display
    display(FileLink(archive))
