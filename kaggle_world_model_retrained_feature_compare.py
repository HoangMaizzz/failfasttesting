"""Kaggle launcher for paired retraining: full features vs only hidden layer 28."""
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
if not root.exists(): raise FileNotFoundError(f'Kaggle dataset path not found: {root}')
if root.is_dir():
    candidates = list(root.rglob('checkpoint.pt'))
    if len(candidates) == 1:
        input_path = str(candidates[0].parent)
    elif not candidates:
        zips = list(root.rglob('wm_token_dual_math_100q_*.zip'))
        if len(zips) != 1: raise RuntimeError(f'Specify INPUT_PATH to exactly one full training ZIP. Found: {zips}')
        input_path = str(zips[0])
    else:
        raise RuntimeError(f'Multiple training runs found; set INPUT_PATH exactly: {candidates}')
print('Full training archive:', input_path, flush=True)
ref = globals().get('SOURCE_REF', 'codex/sparse-extend-world-model')
temp = Path('/kaggle/temp'); temp.mkdir(exist_ok=True)
repo = Path(tempfile.mkdtemp(prefix='wm_retrained_feature_compare_', dir=temp))
base = f'https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{ref}/'
files = ('world_model_retrained_feature_compare.py', 'world_model_learning_curve.py',
         'offline_feature_audit.py', 'world_model_core.py', 'world_model_probe.py')
for filename in files: (repo / filename).write_bytes(urlopen(base + filename, timeout=60).read())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors', 'huggingface_hub'], check=True)
import torch
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
output = working / f'world_model_full_vs_hidden28_{stamp}'
cmd = [sys.executable, '-u', str(repo / 'world_model_retrained_feature_compare.py'),
    '--input', input_path, '--output', str(output), '--device', 'cuda:0' if torch.cuda.is_available() else 'cpu',
    '--trust-checkpoint', '--seeds', globals().get('SEEDS', '42,43,44'),
    '--updates', str(globals().get('UPDATES', 512)),
    '--validation-states-per-question', str(globals().get('VALIDATION_STATES_PER_QUESTION', 24)),
    '--validation-roots-per-question', str(globals().get('VALIDATION_ROOTS_PER_QUESTION', 4))]
embedding = globals().get('EMBEDDING_FILE', '')
if not embedding and Path('/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors').exists():
    embedding = '/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors'
if embedding: cmd += ['--embeddings', embedding]
print(f"Device: {cmd[cmd.index('--device')+1]} | LLM forwards: 0 | models: full vs hidden-28-only", flush=True)
subprocess.run(cmd, cwd=repo, check=True)
summary = json.loads((output / 'summary.json').read_text())
print('Status:', summary['status'], '| optimizer updates:', summary['optimizer_updates_per_model'],
      '×', len(summary['seeds']), 'seeds × 2 models')
print('DOWNLOAD:', output.with_suffix('.zip'))
from IPython.display import display, FileLink
display(FileLink(output.with_suffix('.zip').name))
