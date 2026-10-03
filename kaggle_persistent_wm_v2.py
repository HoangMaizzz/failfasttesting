"""One-cell V2 entrypoint. Accept original TwoSource folder OR ZIP, never an old filename."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def locate_input(path,temp):
    path=Path(path)
    required=('summary.json','states.jsonl','edges.jsonl','labels.jsonl','teacher_targets.jsonl','config.json')
    if not path.exists():raise FileNotFoundError(f'Input not mounted: {path}')
    candidates=[path] if path.is_dir() else []
    if path.is_dir():candidates += [p.parent for p in path.rglob('summary.json')]
    matches=[]
    for candidate in dict.fromkeys(candidates):
        if not all((candidate/name).is_file() for name in required):continue
        summary=json.loads((candidate/'summary.json').read_text())
        if (summary.get('schema')=='interactive_acceptance_two_source_v1' and
            summary.get('status')=='complete' and (candidate/'experience').is_dir()):matches.append(candidate)
    if len(matches)==1:return matches[0]
    if len(matches)>1:raise ValueError(f'Multiple original TwoSource runs; set RUN_DIR exactly: {matches}')
    archives=[path] if path.is_file() else sorted(path.rglob('*.zip'))
    choices=[]
    for archive in archives:
        if not zipfile.is_zipfile(archive):continue
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.split('/')[-1]!='summary.json':continue
                try:summary=json.loads(zf.read(name))
                except (ValueError,UnicodeError):continue
                prefix=name[:-len('summary.json')]
                if (summary.get('schema')=='interactive_acceptance_two_source_v1' and
                    summary.get('status')=='complete' and
                    all(prefix+n in zf.namelist() for n in required) and
                    any(n.startswith(prefix+'experience/') and n.endswith('.npz') for n in zf.namelist())):
                    choices.append((archive,prefix))
    if len(choices)!=1:
        sample=sorted(str(p.relative_to(path)) for p in path.rglob('*') if p.is_file())[:25] if path.is_dir() else []
        raise FileNotFoundError(f'Need ORIGINAL complete TwoSource GSM8K-100 folder or ZIP, containing experience/shard*.npz. '
            f'Found {len(choices)} matching archives, files={sample}. A CV/report-only ZIP is NOT enough. No old ZIP name is required.')
    archive,prefix=choices[0];dest=temp/'two_source_input';dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as zf:
        # ZIP slip checks BEFORE writing anything.
        for info in zf.infolist():
            target=(dest/info.filename).resolve()
            if not target.is_relative_to(dest.resolve()) or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe path/symlink in input archive')
        zf.extractall(dest)
    return dest/prefix


def launch(scope):
    ref=scope.get('SOURCE_REF','codex/persistent-wm-film-v1')
    work=Path('/kaggle/working');work.mkdir(parents=True,exist_ok=True);os.chdir(work)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    temp=Path('/kaggle/temp')/f'wm_v2_{stamp}';temp.mkdir(parents=True,exist_ok=False)
    # Resolve input BEFORE installing or downloading multi-GB model weights.
    source=locate_input(scope.get('RUN_DIR','/kaggle/input/datasets/ainzkhail/2source2'),temp)
    summary=json.loads((source/'summary.json').read_text())
    config=json.loads((source/'config.json').read_text())
    if summary.get('questions_completed')!=100 or config.get('dataset','gsm8k')!='gsm8k':
        raise ValueError('This launcher needs original GSM8K experiences for exactly 100 questions')
    print('Input:',source,'questions=',summary['questions_completed'],flush=True)
    print('Reuse raw experiences; original checkpoint is NOT loaded. Train V2 from scratch, grouped 5-fold.',flush=True)
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4 x2 and enable Internet')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
        GIT_LFS_SKIP_SMUDGE='1',HF_HOME='/kaggle/temp/wm_v2_hf_cache',USE_TF='0',USE_FLAX='0',
        TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def run(command,cwd=None):
        command=list(map(str,command));print('>>>',' '.join(command),flush=True)
        subprocess.run(command,cwd=cwd,env=env,check=True)
    run([sys.executable,'-m','pip','install','-q','--no-cache-dir','transformers==4.53.1','accelerate','einops','numpy','huggingface_hub'])
    repo=temp/'repo';repo.mkdir()
    # A commit SHA AND branch ref are supported. Never remove the notebook cwd.
    run(['git','init',repo],cwd=work)
    run(['git','-C',repo,'remote','add','origin','https://github.com/HoangMaizzz/failfasttesting.git'])
    run(['git','-C',repo,'fetch','--depth','1','origin',ref])
    run(['git','-C',repo,'checkout','--detach','FETCH_HEAD'])
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],env=env,text=True).strip()
    print('Source commit:',sha,flush=True)
    run([sys.executable,'tests/test_persistent_world_model_v2.py'],cwd=repo)
    film_steps=int(scope.get('FILM_STEPS',200));dllm=temp/'Fast_dLLM_v2_1_5B'
    if film_steps:
        run([sys.executable,'-c',
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('Efficient-Large-Model/Fast_dLLM_v2_1.5B', "
            f"local_dir={str(dllm)!r}, allow_patterns=['configuration.py','*.json','*.safetensors','*.txt','*.jinja'])"],cwd=repo)
        shutil.copy2(repo/'Fast_dLLM_v2_1_5B/modeling.py',dllm/'modeling.py')
    output=work/f'wm_behavioral_v2_gsm8k100_{stamp}'
    command=[sys.executable,'-u',repo/'run_persistent_world_model_v2.py','--run_dir',source,
        '--output_dir',output,'--dllm_dir',dllm,'--num_questions','100','--device','cuda:1',
        '--source_revision',sha,
        '--target_device','0','--drafter_device','1','--target_gpu_memory_gib','8',
        '--updates_per_question',str(scope.get('UPDATES_PER_QUESTION',16)),
        '--dynamics_updates',str(scope.get('DYNAMICS_UPDATES',300)),
        '--film_steps',str(film_steps),'--film_examples_per_question',str(scope.get('FILM_EXAMPLES_PER_QUESTION',3)),
        '--real_questions_per_fold',str(scope.get('REAL_QUESTIONS_PER_FOLD',4)),
        '--batch_size','8','--seed','42']
    if 'VARIANTS' in scope:command+=['--variants',*scope['VARIANTS']]
    result=subprocess.run(list(map(str,command)),cwd=repo,env=env,check=False)
    archive=output.with_suffix('.zip')
    if archive.exists():print('DOWNLOAD ZIP (working root):',archive,flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode,command)
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:raise RuntimeError('Result ZIP failed CRC check')
        result_summary=json.loads(zf.read('summary.json'))
        if result_summary.get('status')!='complete':raise RuntimeError('Partial V2 result; inspect error.txt')
    print('Complete: offline WM CV + H1/H2/H3 + feature/dynamics/FiLM ablations.',flush=True)
    from IPython.display import display,FileLink
    display(FileLink(archive.name))


if __name__=='__main__':launch(globals())
