"""Execute with INPUT_PATH and SOURCE_REF globals; no LLM inference or training."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

working = Path('/kaggle/working'); working.mkdir(exist_ok=True)
os.chdir(working)
input_path = globals().get('INPUT_PATH', '')
if not input_path:
    candidates = list(Path('/kaggle/input').rglob('wm_token_dual_math_100q_*.zip'))
    if not candidates:
        candidates = [p.parent for p in Path('/kaggle/input').rglob('checkpoint.pt')]
    if len(candidates) != 1:
        raise RuntimeError('Set INPUT_PATH to exactly one training ZIP or extracted run directory. Found: '+str(candidates))
    input_path = str(candidates[0])
if not Path(input_path).exists():
    raise FileNotFoundError(input_path)
# Accept the dataset mount root, an extracted run folder, or the original ZIP.
if Path(input_path).is_dir():
    checkpoints = list(Path(input_path).rglob('checkpoint.pt'))
    if not checkpoints:
        archives = list(Path(input_path).rglob('*.zip'))
        if len(archives) != 1:
            raise RuntimeError('No checkpoint; specify exactly one training ZIP: '+str(archives))
        input_path = str(archives[0])
print('Input:', input_path, flush=True)
source_ref = globals().get('SOURCE_REF', 'codex/sparse-extend-world-model')
temporary = Path('/kaggle/temp'); temporary.mkdir(exist_ok=True)
repo = Path(tempfile.mkdtemp(prefix='offline_feature_audit_', dir=temporary))
base = f'https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{source_ref}/'
for name in ('offline_feature_audit.py', 'world_model_core.py', 'world_model_probe.py'):
    (repo/name).write_bytes(urlopen(base+name, timeout=60).read())
# Keep installed PyTorch; no transformers, verifier, model code, or training dependencies.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors', 'huggingface_hub'], check=True)
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
output = working / f'offline_feature_audit_{stamp}'
suite = globals().get('SUITE', 'standard')
if suite == 'followup':
    output = working / f'offline_feature_followup_{stamp}'
cmd = [sys.executable, '-u', str(repo/'offline_feature_audit.py'), '--input', input_path,
    '--output', str(output), '--device', device, '--trust-checkpoint',
    '--batch-size', str(globals().get('BATCH_SIZE', 4)), '--horizon', '3', '--suite', suite]
if globals().get('VALIDATION_ONLY', suite == 'followup'):
    cmd += ['--validation-only']
embedding_file = globals().get('EMBEDDING_FILE', '')
if not embedding_file:
    cached = Path('/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors')
    if cached.exists():
        embedding_file = str(cached)
if embedding_file:
    cmd += ['--embeddings', embedding_file]
if globals().get('ABLATE_TRAIN', False):
    cmd += ['--ablate-train']
print('Suite:', suite, '| validation-only:', '--validation-only' in cmd, flush=True)
print('NO drafter/verifier inference and NO retraining. One GPU is sufficient.', flush=True)
subprocess.run(cmd, cwd=repo, check=True)
print('DOWNLOAD:', output.with_suffix('.zip'))
summary = json.loads((output/'summary.json').read_text())
print('Baseline parity:', summary['parity_with_saved_predictions'])
from IPython.display import display, FileLink
display(FileLink(output.with_suffix('.zip').name))
