"""Collect with the frozen native drafter, release it, then train on held-out prompts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile


def package(output):
    output=Path(output)
    archive=output.with_suffix('.zip')
    temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as file:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not any(part in ('.git','__pycache__') for part in path.relative_to(output).parts):
                file.write(path,path.relative_to(output).as_posix())
    os.replace(temporary,archive)
    print(f'[archive] {archive} ({archive.stat().st_size/1024**2:.1f} MiB)',flush=True)
    return archive


def validate_config(cfg):
    result=dict(cfg)
    for key in ('num_questions','max_new_tokens','encoder_updates','updates','batch_size','eval_every'):
        if key in result and (type(result[key]) is not int or result[key]<1):
            raise ValueError(f'{key} must be a positive integer')
    if result.get('num_questions',100)<3:
        raise ValueError('At least three questions are needed for disjoint train/validation/test')
    if result.get('physical_block_size',32)!=32 or result.get('small_block_size',8)!=8:
        raise ValueError('V1 uses native physical block 32 and active small block 8')
    if result.get('latent_dim',128)%4:
        raise ValueError('latent_dim must be divisible by four attention heads')
    if result.get('drafter_threshold',.5)!=.5:
        raise ValueError('V1 feasibility protocol uses drafter_threshold=0.5')
    if result.get('horizon',3)!=3:
        raise ValueError('V1 evaluates horizons 1 through 3')
    return result


def run(arguments):
    cfg=validate_config(json.loads(Path(arguments.config_json).read_text()))
    output=Path(arguments.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if arguments.collect_only:
        from drafter_simulator_collect import collect
        collect(cfg,output,Path(arguments.dllm_dir))
        return
    if any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; existing results are preserved')
    started=datetime.now(timezone.utc).isoformat()
    (output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    summary=dict(status='running',started_utc=started,verifier_loaded=False,
                 source_revision=cfg.get('source_revision'),phase='collection')
    error=None
    try:
        source=Path(__file__).resolve().parent
        hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (
            'drafter_simulator_collect.py','drafter_simulator_models.py','drafter_simulator_train.py',
            'run_drafter_simulator.py','Fast_dLLM_v2_1_5B/modeling.py')}
        (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        import torch
        import numpy as np
        (output/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,
            numpy=np.__version__,platform=platform.platform(),devices=[torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())]),indent=2),encoding='utf-8')
        # Process exit releases all weights/KV tensors before the training stage.
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config_json',arguments.config_json,
                 '--output_dir',str(output),'--dllm_dir',arguments.dllm_dir,'--collect_only']
        print('[collect] native drafter only; no input ZIP or verifier required',flush=True)
        subprocess.run(command,cwd=source,check=True)
        summary['phase']='training'
        from drafter_simulator_train import train_study
        summary.update(train_study(cfg,output))
        summary.update(status='complete',phase='finished',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        error=exc;summary.update(status='partial',error_type=type(exc).__name__,error=str(exc))
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        package(output)
    if error is not None:
        raise error


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_json',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--dllm_dir',required=True)
    parser.add_argument('--collect_only',action='store_true')
    run(parser.parse_args())
