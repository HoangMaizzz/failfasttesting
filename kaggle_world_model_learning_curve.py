"""Offline learning curve from a saved world-model training ZIP; no LLM inference."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

working = Path('/kaggle/working'); working.mkdir(exist_ok=True); os.chdir(working)
input_path = globals().get('INPUT_PATH', '/kaggle/input/datasets/yumesakihikari/math100')
root = Path(input_path)
if not root.exists():
    raise FileNotFoundError(f'Kaggle dataset path not found: {root}')
if root.is_dir():
    candidates = list(root.rglob('checkpoint.pt'))
    if len(candidates) == 1:
        input_path = str(candidates[0].parent)
    elif not candidates:
        zips = list(root.rglob('wm_token_dual_math_100q_*.zip'))
        if len(zips) != 1:
            raise RuntimeError(f'Specify INPUT_PATH to one training ZIP or extracted run. Found: {zips}')
        input_path = str(zips[0])
    else:
        raise RuntimeError(f'Multiple training runs found; set INPUT_PATH exactly: {candidates}')
print('Training archive:', input_path, flush=True)
ref = globals().get('SOURCE_REF', 'codex/sparse-extend-world-model')
temp = Path('/kaggle/temp'); temp.mkdir(exist_ok=True)
repo = Path(tempfile.mkdtemp(prefix='wm_learning_curve_', dir=temp))
base = f'https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{ref}/'
for filename in ('world_model_learning_curve.py', 'offline_feature_audit.py',
                 'world_model_core.py', 'world_model_probe.py'):
    (repo/filename).write_bytes(urlopen(base+filename, timeout=60).read())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors', 'huggingface_hub'], check=True)
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
output = working/f'world_model_learning_curve_{stamp}'
cmd = [sys.executable, '-u', str(repo/'world_model_learning_curve.py'),
    '--input', input_path, '--output', str(output), '--device', device, '--trust-checkpoint',
    '--question-sizes', globals().get('QUESTION_SIZES', '10,20,40,80'),
    '--seeds', globals().get('SEEDS', '42,43,44'),
    '--updates-per-question', str(globals().get('UPDATES_PER_QUESTION', 8)),
    '--fixed-updates', str(globals().get('FIXED_UPDATES', 512)),
    '--progress-updates', globals().get('PROGRESS_UPDATES', '0,16,64,128,256,512'),
    '--validation-states-per-question', str(globals().get('VALIDATION_STATES_PER_QUESTION', 24)),
    '--validation-roots-per-question', str(globals().get('VALIDATION_ROOTS_PER_QUESTION', 4))]
embedding = globals().get('EMBEDDING_FILE', '')
if not embedding and Path('/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors').exists():
    embedding = '/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors'
if embedding:
    cmd += ['--embeddings', embedding]
print(f'Running on {device}; training only the small world model from cached states.', flush=True)
print('Drafter forwards: 0 | verifier forwards: 0 | question-counts/seeds: 10,20,40,80 × 3', flush=True)
print('Fixed-data progress checkpoints:', globals().get('PROGRESS_UPDATES', '0,16,64,128,256,512'), flush=True)
subprocess.run(cmd, cwd=repo, check=True)
summary = json.loads((output/'summary.json').read_text())
print('World-model optimizer updates:', summary['total_optimizer_updates'])
print('DOWNLOAD:', output.with_suffix('.zip'))
from IPython.display import display, FileLink
display(FileLink(output.with_suffix('.zip').name))
