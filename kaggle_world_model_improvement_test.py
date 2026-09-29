"""Fresh Kaggle launcher: cached training ZIP -> retrained factor study -> output ZIP."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

working=Path('/kaggle/working'); working.mkdir(exist_ok=True); os.chdir(working)
root=Path(globals().get('INPUT_PATH','/kaggle/input/datasets/yumesakihikari/math100'))
if not root.exists(): raise FileNotFoundError(f'Input not mounted: {root}')
if root.is_dir():
    checkpoints=list(root.rglob('checkpoint.pt'))
    if len(checkpoints)==1: root=checkpoints[0].parent
    elif checkpoints: raise RuntimeError('More than one training archive: set INPUT_PATH exactly')
    else:
        candidates=[]
        for p in root.rglob('*.zip'):
            if not zipfile.is_zipfile(p): continue
            with zipfile.ZipFile(p) as z:
                if any(n.endswith('checkpoint.pt') for n in z.namelist()): candidates.append(p)
        if len(candidates)!=1: raise RuntimeError(f'Need the original training archive, found {candidates}')
        root=candidates[0]
ref=globals().get('SOURCE_REF','codex/sparse-extend-world-model')
temp=Path('/kaggle/temp'); temp.mkdir(exist_ok=True)
repo=Path(tempfile.mkdtemp(prefix='wm_improvement_',dir=temp))
files=('world_model_improvement_test.py','world_model_improvement_report.py',
       'world_model_probe_v2.py','world_model_core.py','world_model_probe.py','offline_feature_audit.py')
base=f'https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{ref}/'
for filename in files: (repo/filename).write_bytes(urlopen(base+filename,timeout=60).read())
subprocess.run([sys.executable,'-m','pip','install','-q','safetensors','huggingface_hub','matplotlib'],check=True)
import torch
device=globals().get('DEVICE','cuda:0' if torch.cuda.is_available() else 'cpu')
if device=='cpu': raise RuntimeError('Enable a Kaggle GPU for this full experiment')
out=working/f'world_model_improvement_{datetime.now():%Y%m%d_%H%M%S_%f}'
cmd=[sys.executable,'-u',str(repo/'world_model_improvement_test.py'),
     '--input',str(root),'--output',str(out),'--device',device,'--trust-checkpoint',
     '--variants',globals().get('VARIANTS','all'), '--seeds',globals().get('SEEDS','42,43,44'),
     '--steps',globals().get('STEPS','0,64,128,256,512'),
     '--batch-size',str(globals().get('BATCH_SIZE',8)),
     '--eval-batch-size',str(globals().get('EVAL_BATCH_SIZE',8))]
embedding=globals().get('EMBEDDING_FILE','')
if not embedding and Path('/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors').exists():
    embedding='/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors'
if embedding: cmd+=['--embeddings',str(embedding)]
print('Training archive:',root,flush=True)
print('Training small world models only; drafter/verifier forwards = 0',flush=True)
try:
    subprocess.run(cmd,cwd=repo,check=True)
finally:
    if out.with_suffix('.zip').exists():
        print('DOWNLOAD ZIP:',out.with_suffix('.zip'),flush=True)
        from IPython.display import display,FileLink
        display(FileLink(out.with_suffix('.zip').name))
if (out/'summary.json').exists():
    summary=json.loads((out/'summary.json').read_text())
    print('Status:',summary['status'],'Elapsed minutes:',round(summary['elapsed_seconds']/60,1))
    from IPython.display import Image,display
    for name in ('learning_curves.png','train_validation.png'):
        if (out/name).exists(): display(Image(filename=str(out/name)))
