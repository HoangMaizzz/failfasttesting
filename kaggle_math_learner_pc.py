"""Kaggle entry point: public repo, INT8 target, FP16 drafter, live logs and ZIP."""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from IPython.display import display, FileLink

work = Path('/kaggle/working')
repo = work / 'failfasttesting_learner_pc_41903ae'
output = work / 'math_learner_positive_control_run'
env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0', WANDB_MODE='disabled',
           PYTORCH_ALLOC_CONF='expandable_segments:True')


def run(command, cwd=None):
    print('\n>>>', ' '.join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f'Command failed: {status}; inspect the preceding log')


if not repo.exists():
    run(['git', 'clone', '--branch', 'codex/frontier-stop-controller',
         'https://github.com/HoangMaizzz/failfasttesting.git', repo])
run(['git', 'fetch', 'origin', 'codex/frontier-stop-controller'], repo)
run(['git', 'checkout', '41903ae'], repo)
run([sys.executable, '-m', 'pip', 'install', 'transformers==4.53.3',
     'accelerate==1.10.1', 'bitsandbytes==0.47.0', 'datasets==4.2.0',
     'einops==0.8.1', 'pandas', 'numpy', 'matplotlib', 'safetensors'])
run(['nvidia-smi'])
drafter = work / 'Fast_dLLM_v2_1.5B'
run([sys.executable, '-c', 'from huggingface_hub import snapshot_download; '
     f"snapshot_download('Efficient-Large-Model/Fast_dLLM_v2_1.5B', local_dir={str(drafter)!r})"])
try:
    run([sys.executable, '-u', 'run_math_learner_positive_control.py',
         '--dllm_dir', drafter, '--output_dir', output,
         '--target_quantization', 'int8', '--target_device', '0', '--drafter_device', '0',
         '--thresholds', '.25', '.30', '.35', '.40', '.45', '.50',
         '--max_validate_thresholds', '3', '--include_probe_control', '--resume'], repo)
finally:
    if output.exists():
        archive = shutil.make_archive(str(output), 'zip', root_dir=output)
        display(FileLink(archive))
