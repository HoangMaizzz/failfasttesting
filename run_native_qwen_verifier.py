"""Frozen real-Qwen raw-layer ceiling, behavioral compression, and true cuts.

No drafter dynamics, D-to-V bridge, planner or Stop/Continue optimization is
trained. Test observations are quarantined from training and layer selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import traceback
import zipfile

import numpy as np
import torch

from behavior_aware_source import sha_file
from factorized_wm_metrics import write_json as _write_json, write_jsonl as _write_jsonl, write_csv as _write_csv
from native_qwen_capture import load_verifier, depth_indices, capture_one, benchmark_partial
from native_qwen_data import (prepare_native_source, select_capture_uids, NativeHiddenStore,
    pack_native_batch, load_candidate_table, digest)
from native_qwen_metrics import (report, per_method_report, seed_summary, paired_bootstrap,
    select_layers, compression_comparison, feasibility_gate)
from native_qwen_probe import make_model, call_model, train_model
from run_latent_wm_phase0 import save_torch
from kaggle_native_qwen_verifier import archive_partial, is_result_file
from kaggle_latent_wm_phase0 import safe_extract


def write_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    return _write_json(path,value)


def write_jsonl(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    return _write_jsonl(path,value)


def write_csv(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    return _write_csv(path,value)


def log(stage, **items):
    print('[native-qwen] '+json.dumps(dict(stage=stage,**items)),flush=True)


def config_check(cfg):
    if cfg.get('schema')!='native_qwen_verifier_latent_v1' or cfg.get('num_questions')!=100:
        raise ValueError('This study requires the original 100q native-Qwen schema')
    for name in ('workers','probe_updates','latent_updates','eval_every','batch_size','bootstrap_samples'):
        if type(cfg[name]) is not int or cfg[name]<1:raise ValueError(f'{name} must be positive')
    if cfg['workers']>2 or len(cfg['devices'])<cfg['workers']:
        raise ValueError('One or two training devices are required')
    if cfg.get('tiny_config') is not None and not cfg.get('pipeline_check_only'):
        raise ValueError('A tiny random teacher is restricted to explicit offline pipeline tests')
    if cfg.get('device')=='cpu' and not cfg.get('pipeline_check_only'):
        raise ValueError('Full capture cannot use a CPU teacher override')
    if cfg['dtype']!='float16' or cfg['attn_implementation']!='sdpa':
        raise ValueError('The audited reproduction protocol uses frozen FP16 SDPA')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):
        raise ValueError('Seed list must be unique and nonempty')
    if not cfg.get('pipeline_check_only'):
        if cfg['seeds']!=[42,43,44] or cfg['benchmark_repetitions']<100 or cfg['benchmark_warmup']<10:
            raise ValueError('Full feasibility requires seeds42/43/44 and >=100/10 benchmark repetitions/warmups')
        if cfg['capture_states_per_question']!=0:
            raise ValueError('Full study must not downsample the original states')
    if cfg['selected_layers']!=2 or cfg['latent_dims']!=[64,128,256]:
        raise ValueError('Use top two validation layers and all 64/128/256 dimensions')
    if cfg['max_reproduction_mismatch_rate']!=0:
        raise ValueError('This version requires exact reproduction; no mismatch filtering is allowed')
    if not cfg['package_raw_hidden']:
        raise ValueError('Raw native representation must remain auditable in the result')
    if cfg['lambda_K_ablation']!=[0.,.1] or cfg['lambda_K']!=.1:
        raise ValueError('Primary .1 K loss and BCE-only ablation must both run')
    for device in cfg['devices'][:cfg['workers']]:
        if device=='cpu' and not cfg.get('pipeline_check_only'):
            raise ValueError('Production capture and training require two GPUs')
        if device.startswith('cuda') and (not torch.cuda.is_available() or
                int(device.split(':')[1])>=torch.cuda.device_count()):
            raise ValueError(f'Configured device unavailable: {device}')


def source_code_hashes():
    names=('native_qwen_capture.py','native_qwen_probe.py','native_qwen_data.py',
           'native_qwen_metrics.py','run_native_qwen_verifier.py','paired_latent_models.py',
           'behavior_aware_source.py','phase0_wm_models.py','phase0_wm_data.py',
           'factorized_wm_data.py','factorized_wm_metrics.py')
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}


def verifier_options(cfg):
    options=dict(cfg)
    if cfg.get('tiny_config') is not None and cfg['devices'][0]=='cpu':
        options['device']='cpu'
    return options


def find_native_result(path, cache):
    """Native capture/result ZIP or unpacked root, never an arbitrary old teacher."""
    path=Path(path).resolve()
    if not path.exists():raise FileNotFoundError(f'Native capture input missing: {path}')
    if path.is_file():path=safe_extract(path,Path(cache)/'capture_import')
    roots={p.parent for p in path.rglob('capture_manifest.json')}
    if (path/'capture_manifest.json').is_file():roots.add(path)
    if not roots:
        archives=[p for p in path.rglob('*') if p.is_file() and p.suffix.lower() not in ('.pt','.npy')
                  and zipfile.is_zipfile(p)]
        if len(archives)==1:return find_native_result(archives[0],cache)
    if len(roots)!=1:raise ValueError(f'Expected exactly one native capture root, found {len(roots)}')
    root=roots.pop()
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Native result has symlinks')
    return root


def restore_capture(path, output, cache):
    root=find_native_result(path,cache)
    required=['capture_manifest.json','capture_progress.json','candidate_embedding_ids.json',
              'candidate_embeddings.npy','candidate_embedding_manifest.json','capture_identity.json']
    for name in required:
        if not (root/name).is_file():raise ValueError(f'Native capture lacks {name}')
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        wanted=(rel.parts[0]=='native_hidden' or rel.as_posix() in required or
                rel.as_posix()=='partial_qwen_latency/latency_by_layer.json')
        if p.is_file() and wanted:
            target=Path(output)/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError('Refusing to overwrite an existing native capture')
            shutil.copy2(p,target)
    return root


def capture_audits(store, source, uids):
    selected=[source['rows'][u] for u in uids]
    progress=store.progress
    failures=[dict(state_id=u,question_id=source['rows'][u]['question'],
                   K_saved=r['saved_K'],K_rerun=r['rerun_K'],reason='greedy prefix disagrees; cause undetermined')
              for u,r in progress.items() if not r['matches']]
    reproduction=dict(selected_states=len(uids),checked_states=len(progress),mismatches=len(failures),
        exact_match_rate=None if not progress else 1-len(failures)/len(progress),
        passed=len(progress)==len(uids) and not failures,failures=failures,
        policy='fail-fast on ANY mismatch; no favorable subset is silently retained')
    alignment=dict(passed=bool(progress) and all(r['alignment']['passed'] for r in progress.values()),
        states_checked=len(progress),max_abs_error=max((r['alignment']['max_abs_error'] for r in progress.values()),default=None),
        fp16_max_abs_error=max((r['alignment']['fp16_max_abs_error'] for r in progress.values()),default=None),
        causal_position='prefix_len-1+i (zero based)',
        final_hidden_stage='post_final_norm_lm_head_input', intermediate_stage='raw_decoder_block_output',
        no_final_logits_as_features=True,per_state_audit='capture_progress.json')
    counts={name:dict(num_questions=len(ids),selected_states=sum(r['question'] in ids for r in selected),
        captured_states=sum(source['rows'][u]['question'] in ids for u in progress),
        num_positions=sum(r['length'] for r in selected if r['question'] in ids)) for name,ids in source['split'].items()}
    audit=dict(counts=counts,total_selected_states=len(uids),original_states=len(source['rows']),
        excluded_unlabeled=sum(r['accepted'] is None for r in source['rows'].values()),
        native_hidden_dtype='float16',hidden_dim=store.hidden_dim,depths=store.depths,
        no_projection=True,raw_estimated_bytes=store.total_positions*store.hidden_dim*2*len(store.depths),
        baseline_subset_identical=True,split_reused=True,old_projected32_used=False)
    return reproduction,alignment,audit


def write_capture_metadata(output, source, store):
    rows=[]
    for uid in store.uids:
        r=source['rows'][uid];start,end=store.offsets[uid]
        rows.append(dict(question_id=r['question'],state_id=uid,round_id=r['round_id'],
            action=r['action'],proposal_length=r['length'],candidate_token_ids=r['candidate'],
            prefix_length=len(r['prefix']),prefix_sha256=digest(r['prefix']),
            K_true=r['accepted'],parent_K=r['parent_K'],parent_length=r['parent_length'],
            is_parent_full_prefix=r['is_parent_full_prefix'],native_segment_start=r['segment_start'],
            position_offset=start,position_end=end,
            survival_true=[int(i<r['accepted']) for i in range(r['length'])],
            new_block_mask=[r['action']=='E' and i>=r['parent_length'] for i in range(r['length'])]))
    write_jsonl(Path(output)/'capture_rows.jsonl',rows)
    def positions():
        for r in rows:
            start=r['native_segment_start']
            for i,t in enumerate(r['candidate_token_ids']):
                yield dict(question_id=r['question_id'],state_id=r['state_id'],action=r['action'],
                    proposal_length=r['proposal_length'],position_i=i+1,candidate_token_id=t,
                    K_true=r['K_true'],survival_true_i=int(i<r['K_true']),
                    parent_K=r['parent_K'],parent_length=r['parent_length'],
                    is_parent_full_prefix=r['is_parent_full_prefix'],is_old_region=i<start,
                    is_new_block=i>=start,new_block_relative_position=i-start if i>=start else None,
                    **{'qwen_hidden_'+slot:dict(file='native_hidden/'+slot+'/hidden.npy',row=r['position_offset']+i)
                       for slot in ('layer25','layer50','layer75','layer100')})
    write_jsonl(Path(output)/'capture_positions.jsonl',positions())


def ensure_candidate_embeddings(model, source, uids, output):
    ids=sorted({t for uid in uids for t in source['rows'][uid]['candidate']})
    embedding=model.get_input_embeddings()
    if embedding.weight.requires_grad:raise ValueError('Qwen candidate embedding table is not frozen')
    if ids[-1]>=embedding.num_embeddings:raise ValueError('Saved STOP candidate is outside Qwen vocabulary')
    with torch.inference_mode():
        values=embedding.weight.index_select(0,torch.tensor(ids,device=embedding.weight.device)).half().cpu().numpy()
    if not np.isfinite(values).all():raise ValueError('Nonfinite candidate embeddings')
    np.save(Path(output)/'candidate_embeddings.npy',values,allow_pickle=False)
    write_json(Path(output)/'candidate_embedding_ids.json',ids)
    write_json(Path(output)/'candidate_embedding_manifest.json',dict(
        model_identity=source['identity'],embedding_source='Qwen own frozen get_input_embeddings().weight',
        frozen=True,projected=False,hidden_dim=values.shape[-1],tokens=len(ids),dtype='float16',
        sha256=sha_file(Path(output)/'candidate_embeddings.npy'),id_sha256=digest(ids)))


def method_name(spec):
    if spec['kind']=='direct':return 'Drafter_direct'
    if spec['kind']=='raw':
        return f"raw_L{spec['depth']}_{spec['variant']}_K{spec['lambda_K']:g}"+('_struct' if spec.get('structural_input') else '')
    return f"latent_L{spec['depth']}_dim{spec['dim']}_rec{spec.get('lambda_rec',0):g}"


def raw_specs(depths,cfg):
    specs=[dict(kind='direct',variant='HC',depth=None,dim=128,lambda_K=.1,lambda_rec=0.)]
    for depth in depths:
        for variant in ('H','HC'):
            for weight in cfg['lambda_K_ablation']:
                specs.append(dict(kind='raw',variant=variant,depth=depth,dim=128,lambda_K=weight,lambda_rec=0.))
    return specs


def latent_specs(selected,cfg):
    specs=[]
    for depth in selected:
        for dim in cfg['latent_dims']:
            specs.append(dict(kind='latent',variant='HC',depth=depth,dim=dim,lambda_K=.1,lambda_rec=0.))
        if cfg.get('structural_control'):
            specs.append(dict(kind='raw',variant='HC',depth=depth,dim=128,lambda_K=.1,
                              lambda_rec=0.,structural_input=True))
        for weight in cfg['reconstruction_weights']:
            specs.append(dict(kind='latent',variant='HC',depth=depth,dim=128,lambda_K=.1,lambda_rec=weight))
    return specs


def records_for(model,spec,uids,source,store,table,cfg,device,seed):
    model.eval();out=[]
    with torch.no_grad():
        for begin in range(0,len(uids),cfg['batch_size']):
            chosen=uids[begin:begin+cfg['batch_size']]
            b=pack_native_batch(chosen,spec,source['rows'],source['cachez'],store,table,device)
            pred=call_model(model,spec,b)
            for i,uid in enumerate(chosen):
                row=source['rows'][uid];n=row['length']
                q=pred['q'][i,:n].float().cpu().tolist()
                out.append(dict(question_id=row['question'],state_id=uid,method=method_name(spec),seed=seed,
                    split=row.get('split'),
                    action=row['action'],proposal_length=n,K_true=row['accepted'],K_pred=sum(q),q_pred=q,
                    parent_K=row['parent_K'],parent_length=row['parent_length'],
                    is_E_full_prefix=row['action']=='E' and row['is_parent_full_prefix'],
                    survival_true=[int(p<row['accepted']) for p in range(n)],
                    new_block_mask=[row['action']=='E' and p>=row['parent_length'] for p in range(n)],spec=spec))
    return out


def worker(job):
    torch.set_num_threads(1)
    source=torch.load(job['payload'],map_location='cpu',weights_only=True)
    if source['ids'].get('test') or any(r['question'] in source['split']['test'] for r in source['rows'].values()):
        raise ValueError('Held-out test states entered a native training worker')
    capture=source['capture']
    # Offsets follow the full capture manifest, but only train/validation row
    # contents and D latents are delivered to a training worker.
    all_rows=dict(source['rows'])
    for uid,geometry in source['capture_geometry'].items():
        if uid not in all_rows:all_rows[uid]=dict(length=geometry['length'])
    store=NativeHiddenStore(capture['root'],all_rows,capture['uids'],capture['depths'],
                            capture['hidden_dim'],capture['signature'],readonly=True)
    table=None
    try:
        return worker_models(job,source,store)
    finally:
        store.close()


def worker_models(job,source,store):
    table=load_candidate_table(source['capture']['root'])
    try:
        return worker_train(job,source,store,table)
    finally:
        if hasattr(table['values'],'_mmap'):table['values']._mmap.close()


def worker_train(job,source,store,table):
    allowed=set(source['ids']['train']+source['ids']['val'])
    def pack(uids,spec,device):
        if not set(uids)<=allowed:raise ValueError('A training pack tried to read held-out teacher observations')
        return pack_native_batch(uids,spec,source['rows'],source['cachez'],store,table,device)
    results=[]
    for spec in job['specs']:
        started=time.perf_counter();name=method_name(spec)
        log('train_start',seed=job['seed'],method=name,device=job['device'])
        cfg=dict(job['config'],updates=job['config']['latent_updates'] if spec['kind']=='latent' else job['config']['probe_updates'])
        def validation(model,current_spec):
            rows=records_for(model,current_spec,source['ids']['val'],source,store,table,cfg,job['device'],job['seed'])
            cohorts=report(rows)
            log('validation',seed=job['seed'],method=name,
                K_MAE=cohorts['ALL']['K_MAE_question_macro'],elapsed_seconds=time.perf_counter()-started)
            return cohorts['ALL']['K_MAE_question_macro'],cohorts
        folder=Path(job['output'])/'jobs'/f"seed{job['seed']}"/name
        model,record=train_model(spec,job['seed'],source['ids']['train'],source['rows'],pack,validation,cfg,folder,job['device'])
        vals=records_for(model,spec,source['ids']['val'],source,store,table,cfg,job['device'],job['seed'])
        write_jsonl(folder/'validation_predictions.jsonl',vals)
        record.update(method=name,elapsed_seconds=time.perf_counter()-started)
        write_json(folder/'selection.json',record);results.append(record)
        del model;gc.collect()
        if job['device'].startswith('cuda'):torch.cuda.empty_cache()
        log('train_complete',seed=job['seed'],method=name,selected_step=record['selected_step'],seconds=record['elapsed_seconds'])
    write_json(Path(job['output'])/'_cache'/f"completed_{job['stage']}_{job['seed']}.json",results)


def execute_stage(specs,source,payload,cfg,output,stage):
    devices=cfg['devices'][:cfg['workers']]
    locks={d:threading.Lock() for d in devices};active=set();active_lock=threading.Lock();cancelled=threading.Event()
    jobs=[dict(seed=seed,device=devices[i%len(devices)],specs=specs,payload=str(payload),
               config=cfg,output=str(output),stage=stage) for i,seed in enumerate(cfg['seeds'])]
    def run(job):
        with locks[job['device']]:
            if cancelled.is_set():raise RuntimeError('Training cancelled after another worker failed')
            path=Path(output)/'_cache'/f"job_{stage}_{job['seed']}.json";write_json(path,job)
            with active_lock:
                if cancelled.is_set():raise RuntimeError('Training cancelled before process launch')
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker_job',str(path)],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
                active.add(process)
            logpath=Path(output)/'jobs'/f"seed{job['seed']}"/f'{stage}_worker_log.txt'
            logpath.parent.mkdir(parents=True,exist_ok=True)
            try:
                with logpath.open('a',encoding='utf-8') as f:
                    for line in process.stdout:f.write(line);f.flush();print(line,end='',flush=True)
                code=process.wait()
                if code:raise RuntimeError(f"Native {stage} seed{job['seed']} exited {code}; see worker log")
                return json.loads((Path(output)/'_cache'/f"completed_{stage}_{job['seed']}.json").read_text())
            finally:
                with active_lock:active.discard(process)
    results=[]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        try:
            for future in as_completed(futures):results.extend(future.result())
        except BaseException:
            cancelled.set()
            for future in futures:future.cancel()
            with active_lock:
                for process in active:
                    if process.poll() is None:process.terminate()
            raise
    return results


def read_predictions(output,specs,seeds,filename):
    out=[]
    for seed in seeds:
        for spec in specs:
            path=Path(output)/'jobs'/f'seed{seed}'/method_name(spec)/filename
            with path.open() as f:out.extend(json.loads(line) for line in f if line.strip())
    return out


def test_models(specs,source,store,table,cfg,output):
    out=[];device=cfg['devices'][0]
    for seed in cfg['seeds']:
        for spec in specs:
            name=method_name(spec);folder=Path(output)/'jobs'/f'seed{seed}'/name
            model=make_model(spec,cfg,store.hidden_dim).to(device)
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            if checkpoint['seed']!=seed:raise ValueError('Selected native checkpoint seed differs')
            model.load_state_dict(checkpoint['model']);model.eval().requires_grad_(False)
            rows=records_for(model,spec,source['ids']['test'],source,store,table,cfg,device,seed)
            write_jsonl(folder/'test_predictions.jsonl',rows);out.extend(rows)
            del model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
        log('test_seed_complete',seed=seed,rows=len(out))
    return out


def write_wide_predictions(rows,output,depths,selected):
    groups=defaultdict(dict)
    for r in rows:groups[(r['seed'],r['state_id'])][r['method']]=r
    out=[]
    for (seed,uid),methods in sorted(groups.items()):
        first=next(iter(methods.values()))
        row={k:first[k] for k in ('question_id','state_id','action','proposal_length','K_true',
            'survival_true','parent_K','parent_length','is_E_full_prefix','new_block_mask')}
        row.update(seed=seed,K_direct=methods['Drafter_direct']['K_pred'],
                   survival_predictions={name:r['q_pred'] for name,r in methods.items()})
        for slot,depth in zip(('25','50','75','100'),depths):
            row['K_layer'+slot]=methods[f'raw_L{depth}_HC_K0.1']['K_pred']
        for dim in (64,128,256):
            row['K_latent'+str(dim)]={str(depth):methods[f'latent_L{depth}_dim{dim}_rec0']['K_pred'] for depth in selected}
        out.append(row)
    write_jsonl(Path(output)/'per_state_predictions.jsonl',out)


def build_outputs(output,cfg,depths,selection,specs,validation_rows,test_rows,checks,latency):
    val=per_method_report(validation_rows);test=per_method_report(test_rows)
    selected=selection['selected'];bootstrap={};comparisons=[];compression={};gates={}
    for depth in depths:
        h=f'raw_L{depth}_H_K0.1';hc=f'raw_L{depth}_HC_K0.1'
        comparisons.extend([(hc,h),(hc,'Drafter_direct'),(hc,f'raw_L{depth}_HC_K0'),
                            (h,f'raw_L{depth}_H_K0')])
    for depth in selected:
        raw=f'raw_L{depth}_HC_K0.1'
        for dim in (64,128,256):
            name=f'latent_L{depth}_dim{dim}_rec0'
            compression[name]=dict(test=compression_comparison(test,name,raw),
                                  validation=compression_comparison(val,name,raw))
            comparisons.extend([(name,raw),(name,'Drafter_direct')])
            control=raw+'_struct'
            if control in test:
                compression[name]['test_same_structural_inputs']=compression_comparison(test,name,control)
                compression[name]['validation_same_structural_inputs']=compression_comparison(val,name,control)
                comparisons.append((name,control))
        for weight in cfg['reconstruction_weights']:
            comparisons.append((f'latent_L{depth}_dim128_rec{weight:g}', f'latent_L{depth}_dim128_rec0'))
        name=f'latent_L{depth}_dim128_rec0'
        gates[str(depth)]=feasibility_gate(validation_rows,raw,name,'Drafter_direct',cfg,checks)
    for a,b in dict.fromkeys(comparisons):
        bootstrap[a+'_vs_'+b]={cohort:paired_bootstrap(test_rows,a,b,cohort,cfg['bootstrap_samples'],42)
            for cohort in ('ALL','R_ALL','R_GAIN','R_LOSS','E_ALL','E_PARENT_FULL_PREFIX')}
    primary_raw={s['depth'] for s in specs if s['kind']=='raw' and not s.get('structural_input')}
    write_json(Path(output)/'layer_probe'/'per_seed_metrics.json',{
        m:v for m,v in test.items() if m.startswith('raw_')})
    write_json(Path(output)/'layer_probe'/'layer_comparison.json',dict(selection=selection,
        validation={m:v for m,v in val.items() if m.startswith('raw_')},
        test={m:v for m,v in test.items() if m.startswith('raw_')},
        depths=sorted(primary_raw),candidate_capacity_note='H and HC share architecture; HC has a wider input and more parameters'))
    write_json(Path(output)/'direct_drafter_baseline'/'metrics.json',dict(
        validation=val['Drafter_direct'],test=test['Drafter_direct'],
        input='frozen D128 + native c20/context8; no native-Qwen observations',
        objective='existing censored hazard NLL + 0.5 survival Brier + 0.05 K Huber',
        subset='exactly the same reproduced saved states as every native method'))
    write_json(Path(output)/'verifier_latent'/'compression_comparison.json',compression)
    for spec in specs:
        name=method_name(spec)
        if spec['kind']=='raw':
            branch='hidden_candidate' if spec['variant']=='HC' else 'hidden_only'
            destination=Path(output)/'layer_probe'/branch/f"layer_{spec['depth']}"/name
        elif spec['kind']=='latent':
            destination=Path(output)/'verifier_latent'/f"layer_{spec['depth']}"/f"dim{spec['dim']}"/name
        else:continue
        write_json(destination/'metrics.json',dict(validation=val[name],test=test[name],spec=spec))
        write_json(destination/'checkpoint_index.json',{
            str(seed):dict(best=f'jobs/seed{seed}/{name}/best.pt',last=f'jobs/seed{seed}/{name}/last.pt',
                           curve=f'jobs/seed{seed}/{name}/learning_curve.json') for seed in cfg['seeds']})
    write_json(Path(output)/'bootstrap'/'question_bootstrap.json',bootstrap)
    write_json(Path(output)/'validation_feasibility_gate.json',gates)
    table=[]
    for spec in specs:
        name=method_name(spec)
        for cohort_name in ('ALL','R_ALL','R_GAIN','R_LOSS','E_ALL','E_PARENT_FULL_PREFIX','E_NEW_BLOCK'):
            summary=seed_summary(test,name,cohort_name)
            example=next(iter(test[name].values()))[cohort_name]
            table.append(dict(method=name,kind=spec['kind'],depth=spec['depth'],dim=spec['dim'],
                cohort=cohort_name,K_MAE_mean=summary['mean'],K_MAE_seed_std=summary['std'],
                num_states_per_seed=example['num_states'],num_questions=example['num_questions'],
                num_positions_per_seed=example['num_positions'],
                survival_brier_mean=np.mean([v[cohort_name]['survival_brier'] for v in test[name].values()
                    if v[cohort_name]['survival_brier'] is not None]).item()
                    if any(v[cohort_name]['survival_brier'] is not None for v in test[name].values()) else None))
    write_csv(Path(output)/'comparison.csv',table)
    quality=[]
    for depth in depths:
        for variant in ('H','HC'):
            method=f'raw_L{depth}_{variant}_K0.1'
            quality.append(dict(depth=depth,method=method,
                K_MAE_ALL=seed_summary(test,method,'ALL')['mean'],
                K_MAE_E_full=seed_summary(test,method,'E_PARENT_FULL_PREFIX')['mean'],
                latency_fraction=latency['by_depth'][str(depth)]['latency_fraction'],
                partial_forward_ms=latency['by_depth'][str(depth)]['weighted_mean_ms']))
    write_csv(Path(output)/'partial_qwen_latency'/'quality_latency.csv',quality)
    plots(output,test,depths,selected,latency)
    rawbest=selection['selected'][0];raw=f'raw_L{rawbest}_HC_K0.1';compact=f'latent_L{rawbest}_dim128_rec0'
    def number(method,cohort_name):
        value=seed_summary(test,method,cohort_name)['mean']
        return 'undefined (empty cohort)' if value is None else f'{value:.4f}'
    lines=['# Native Qwen verifier-latent ceiling and compression',
        'This uses freshly rerun uncompressed native Qwen hidden, not the previous 32D projection.',
        'Checkpoint/config, causal alignment and saved K reproduction are audited before any training.',
        'No drafter dynamics, D-to-V bridge, planner, controller or online speedup is trained/claimed.',
        f"Pipeline-only smoke: {bool(cfg.get('pipeline_check_only'))}; exact question split: 70/15/15.",
        f'Validation-selected layers: {selected}. All dimensions were fixed before accessing test.',
        f'Test K-MAE (ALL): Direct={number("Drafter_direct","ALL")}; raw layer {rawbest} HC={number(raw,"ALL")}; latent128={number(compact,"ALL")}.',
        f'Test K-MAE (E-full-prefix): Direct={number("Drafter_direct","E_PARENT_FULL_PREFIX")}; raw={number(raw,"E_PARENT_FULL_PREFIX")}; latent128={number(compact,"E_PARENT_FULL_PREFIX")}.',
        '1. Cheap decodability: inspect layer_probe/layer_comparison.json and layer_vs_K_MAE.png; layer choice uses validation, not test.',
        '2. Candidate identity: paired HC-vs-H bootstrap is saved for each depth. HC increases input parameter count; this contrast alone is not capacity-isolated.',
        '3. Compression: compare every64/128/256 latent with its native raw layer and the separate same-structural-input probe.',
        '4. Native-vs-D advantage: use validation feasibility gate and held-out paired question bootstrap, especially E-parent-full-prefix/new-block; do not rely on pooled AUC.',
        f"5. Selected raw depth costs {latency['by_depth'][str(rawbest)]['latency_fraction']:.4f} of full Qwen forward on this measured placement. This excludes the small probe and input preparation; not end-to-end latency.",
        '6. Proceed-to-bridge decision is conditional on the validation gate and independent test confirmation; a smoke cannot produce a PASS.',
        'Survival is a globally defined prefix event. Tokens after a first rejection have survival0 even when their local teacher-forced identity agrees.',
        'Raw native MLP predictions are not forced monotone; violations, Brier/AUC/bias and cohort counts are reported.',
        'Checkpoint selection uses ALL validation question-macro MAE. Native layer ranking equally weights ALL and E-full-prefix MAE across seeds.',
        'Old questions are reused, so this is a representation-feasibility pilot rather than independent benchmark generalization.',
        'Final norm is used at100% for exact LM-head alignment; intermediate depths use raw decoder-block outputs.',
        'Read per_state_predictions.jsonl and all_test_predictions.jsonl for per-state/per-position auditing.']
    write_json(Path(output)/'all_per_seed_metrics.json',test)
    (Path(output)/'FINAL_REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return gates


def plots(output,reports,depths,selected,latency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output)/'plots';root.mkdir(exist_ok=True)
    x=np.asarray(depths)
    def save(name):
        plt.tight_layout();plt.savefig(root/name,dpi=160);plt.close()
    plt.figure(figsize=(7,4))
    for variant in ('H','HC'):
        values=[seed_summary(reports,f'raw_L{d}_{variant}_K0.1') for d in depths]
        plt.errorbar(x,[v['mean'] for v in values],yerr=[v['std'] for v in values],marker='o',label=variant)
    plt.axhline(seed_summary(reports,'Drafter_direct')['mean'],color='k',linestyle='--',label='Drafter Direct')
    plt.xlabel('Qwen block depth');plt.ylabel('Test question-macro K-MAE');plt.legend();save('layer_vs_K_MAE.png')
    plt.figure(figsize=(7,4))
    for d in selected:
        plt.plot([64,128,256],[seed_summary(reports,f'latent_L{d}_dim{k}_rec0')['mean'] for k in (64,128,256)],marker='o',label=f'Layer {d}')
        plt.axhline(seed_summary(reports,f'raw_L{d}_HC_K0.1')['mean'],linestyle='--',alpha=.4)
    plt.xlabel('Verifier latent dimension');plt.ylabel('Test question-macro K-MAE');plt.legend();save('compression_vs_K_MAE.png')
    fractions=[latency['by_depth'][str(d)]['latency_fraction'] for d in depths]
    plt.figure(figsize=(7,4));plt.plot(x,fractions,marker='o');plt.xlabel('Qwen block depth');plt.ylabel('Measured fraction of full Qwen forward');save('layer_vs_latency.png')
    plt.figure(figsize=(7,4))
    for variant in ('H','HC'):
        plt.plot(fractions,[seed_summary(reports,f'raw_L{d}_{variant}_K0.1')['mean'] for d in depths],marker='o',label=variant)
    plt.xlabel('Measured partial/full forward latency');plt.ylabel('Test question-macro K-MAE');plt.legend();save('quality_vs_latency.png')


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();store=None;model=None;table=None
    summary=dict(schema='native_qwen_verifier_latent_v1',status='running',
        drafter_dynamics_trained=False,bridge_trained=False,planner_trained=False,old_projected_teacher_used=False)
    try:
        cfg=json.loads(Path(args.config).read_text());config_check(cfg)
        if args.resume and (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=cfg:
            raise ValueError('Resume configuration differs; existing result was not overwritten')
        torch.set_num_threads(4)
        cache=output/'_cache';cache.mkdir(exist_ok=True)
        log('prepare_source',input=str(args.input),phase0_input=str(args.phase0_input))
        with tempfile.TemporaryDirectory(prefix='native_qwen_source_') as source_cache:
            source=prepare_native_source(args.input,args.phase0_input,source_cache,
                cfg['encoder_verification_samples'],cfg['devices'][0])
        if {k:len(v) for k,v in source['split'].items()}!={'train':70,'val':15,'test':15}:
            raise ValueError('Do not resplit saved questions or states')
        uids=select_capture_uids(source['rows'],cfg['capture_states_per_question'],cfg['capture_seed'])
        if set(source['split']['train']+source['split']['val']+source['split']['test'])!={source['rows'][u]['question'] for u in uids}:
            raise ValueError('Capture must retain all100 questions, even in smoke')
        source_key=digest(dict(original=source['provenance']['original_data_digest'],
            phase0_artifacts=source['provenance']['artifact_hashes'],
            identity=source['identity'],uids=uids,split=source['split']))
        write_json(output/'config.json',cfg);write_json(output/'split_manifest.json',source['split'])
        write_json(output/'source_checkpoint_manifest.json',source['provenance'])
        if args.capture_input:restore_capture(args.capture_input,output,cache)
        identity_path=output/'capture_identity.json'
        if identity_path.exists():
            capture_identity=json.loads(identity_path.read_text())
            if capture_identity['source_key']!=source_key:raise ValueError('Native capture input/source/identity/split differs')
            depths=capture_identity['depths'];hidden_dim=capture_identity['hidden_dim']
            capture_signature=capture_identity['signature']
            expected_capture_code=source_code_hashes()['native_qwen_capture.py']
            if capture_identity['capture_code_sha256']!=expected_capture_code:
                raise ValueError('Cached capture implementation differs from this pinned study')
        else:
            log('load_real_verifier',**source['identity'])
            model=load_verifier(source['identity'],verifier_options(cfg))
            depths=depth_indices(model.config.num_hidden_layers);hidden_dim=model.config.hidden_size
            if len(depths)!=4:raise ValueError('Native four-depth study requires at least4 blocks')
            capture_identity=dict(source_key=source_key,model_identity=source['identity'],depths=depths,
                hidden_dim=hidden_dim,num_hidden_layers=model.config.num_hidden_layers,
                model_config=model.config.to_dict(),dtype=str(model.get_input_embeddings().weight.dtype),
                capture_code_sha256=source_code_hashes()['native_qwen_capture.py'])
            capture_signature=digest(capture_identity);capture_identity['signature']=capture_signature
            write_json(identity_path,capture_identity)
        raw_bytes=sum(source['rows'][u]['length'] for u in uids)*hidden_dim*2*len(depths)
        estimated_params=0
        for spec in raw_specs(depths,cfg)+latent_specs(depths[:2],cfg):
            estimate=make_model(spec,cfg,hidden_dim)
            estimated_params+=sum(p.numel() for p in estimate.parameters())
            del estimate
        # best.pt stores model+Adam moments; last.pt additionally retains the
        # authoritative best checkpoint. Reserve 36 bytes/parameter/job, plus
        # a worst-case ZIP copy. Training payload and reports receive 1 GiB.
        checkpoint_bytes=estimated_params*len(cfg['seeds'])*36
        existing=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        required_disk=max(0,2*(raw_bytes+checkpoint_bytes)+1024**3-existing)
        free=shutil.disk_usage(output).free
        if free < required_disk:
            raise RuntimeError(f'Insufficient disk for native hidden, resumable models and ZIP: need {required_disk/2**30:.2f} GiB additional free, have {free/2**30:.2f}; no hidden truncation allowed')
        write_json(output/'disk_preflight.json',dict(raw_bytes=raw_bytes,estimated_checkpoint_bytes=checkpoint_bytes,
            existing_output_bytes=existing,additional_peak_bytes=required_disk,free_bytes=free,
            note='Conservative uncompressed ZIP copy; model/input assets are outside Output'))
        summary['fresh_capture_calls']=0
        store=NativeHiddenStore(output,source['rows'],uids,depths,hidden_dim,capture_signature)
        store.verify_saved_rows()
        if len(store.progress)<len(uids) or not (output/'candidate_embeddings.npy').exists():
            if model is None:model=load_verifier(source['identity'],verifier_options(cfg))
            if (model.config.hidden_size,depth_indices(model.config.num_hidden_layers))!=(hidden_dim,depths):
                raise ValueError('Reloaded verifier configuration differs from captured geometry')
            ensure_candidate_embeddings(model,source,uids,output)
            for index,uid in enumerate(uids):
                if uid in store.progress:continue
                row=source['rows'][uid]
                try:
                    result=capture_one(model,row['prefix'],row['candidate'],depths)
                    summary['fresh_capture_calls']+=1
                except Exception as exc:
                    write_json(output/'alignment_unit_test.json',dict(passed=False,state_id=uid,
                        failure=str(exc),stage='capture failed before hidden was committed'))
                    raise
                store.record(uid,result)
                if result['K']!=row['accepted']:
                    store.flush_progress();reproduction,alignment,audit=capture_audits(store,source,uids)
                    write_json(output/'verifier_reproduction_check.json',reproduction)
                    write_json(output/'alignment_unit_test.json',alignment);write_json(output/'capture_audit.json',audit)
                    raise RuntimeError(f"Saved verifier K is not reproduced: {uid} saved={row['accepted']} rerun={result['K']}; training refused")
                if index%64==0 or index+1==len(uids):
                    log('capture',done=len(store.progress),total=len(uids),state_id=uid,
                        seconds=time.perf_counter()-started,raw_GiB=round(store.total_positions*hidden_dim*2*4/2**30,3))
        store.flush_progress()
        reproduction,alignment,audit=capture_audits(store,source,uids)
        write_json(output/'verifier_reproduction_check.json',reproduction)
        write_json(output/'alignment_unit_test.json',alignment);write_json(output/'capture_audit.json',audit)
        if not reproduction['passed'] or not alignment['passed']:raise ValueError('Required native capture checks failed')
        write_capture_metadata(output,source,store)
        table=load_candidate_table(output)
        emb_manifest=json.loads((output/'candidate_embedding_manifest.json').read_text())
        if emb_manifest['model_identity']!=source['identity'] or emb_manifest['sha256']!=sha_file(output/'candidate_embeddings.npy'):
            raise ValueError('Frozen candidate embedding identity/checksum differs')
        latency_path=output/'partial_qwen_latency'/'latency_by_layer.json'
        if latency_path.exists():
            latency=json.loads(latency_path.read_text())
            if latency['capture_signature']!=capture_signature:raise ValueError('Reused partial latency has a different native capture')
            if not cfg['pipeline_check_only'] and (not latency.get('full_protocol',False) or
                    latency.get('repetitions',0)<100 or latency.get('warmups',0)<10):
                raise ValueError('A smoke timing cannot be reused as a full-protocol benchmark')
            if (latency.get('repetitions')!=cfg['benchmark_repetitions'] or
                    latency.get('warmups')!=cfg['benchmark_warmup'] or latency.get('seed')!=cfg['capture_seed'] or
                    latency.get('requested_max_rows_per_bin')!=cfg['benchmark_inputs_per_bin']):
                raise ValueError('Cached partial benchmark settings differ from the requested timing protocol')
        else:
            if model is None:model=load_verifier(source['identity'],verifier_options(cfg))
            benchmark_rows=[dict(source['rows'][u],split='train') for u in uids
                            if source['rows'][u]['question'] in source['split']['train']]
            log('real_partial_benchmark',training_states=len(benchmark_rows),depths=depths)
            latency=benchmark_partial(model,benchmark_rows,depths,dict(
                repetitions=cfg['benchmark_repetitions'],warmups=cfg['benchmark_warmup'],
                max_rows_per_bin=cfg['benchmark_inputs_per_bin'],seed=cfg['capture_seed'],
                pipeline_check_only=cfg['pipeline_check_only']))
            latency['capture_signature']=capture_signature
            latency['requested_max_rows_per_bin']=cfg['benchmark_inputs_per_bin']
            write_json(latency_path,latency)
        # No simultaneous7B allocations while training the small probes.
        model=None;gc.collect()
        for device in cfg['devices']:
            if device.startswith('cuda'):
                with torch.cuda.device(device):torch.cuda.empty_cache()
        qualified=store.qualified()
        source['ids']={split:[u for u in qualified if source['rows'][u]['question'] in questions]
                       for split,questions in source['split'].items()}
        if any(not values for values in source['ids'].values()):raise ValueError('A split has no captured observations')
        for name,values in source['ids'].items():
            for uid in values:source['rows'][uid]['split']=name
        hidden_hashes={str(depth):sha_file(output/'native_hidden'/slot/'hidden.npy')
                      for depth,slot in zip(depths,('layer25','layer50','layer75','layer100'))}
        signature=digest(dict(source_key=source_key,capture_signature=capture_signature,
            cfg=cfg,code=source_code_hashes(),native_hidden_hashes=hidden_hashes,
            candidate_sha256=emb_manifest['sha256']))
        manifest_path=output/'study_manifest.json'
        if args.resume:
            old=json.loads(manifest_path.read_text())
            if old['fingerprint']!=signature:raise ValueError('Resume config/source/native capture/code fingerprint differs')
        elif manifest_path.exists():raise FileExistsError('Existing training study requires --resume')
        write_json(manifest_path,dict(schema=cfg['schema'],fingerprint=signature,status='running'))
        training_cfg=dict(cfg,study_signature=signature)
        allowed=set(source['ids']['train']+source['ids']['val'])
        train_source=dict(split=source['split'],ids={k:source['ids'][k] for k in ('train','val')},
            rows={u:{k:v for k,v in source['rows'][u].items() if k!='prefix'} for u in allowed},
            cachez={u:source['cachez'][u] for u in allowed},
            capture=dict(root=str(output),uids=uids,depths=depths,hidden_dim=hidden_dim,signature=capture_signature),
            capture_geometry={u:dict(length=source['rows'][u]['length']) for u in uids})
        payload=cache/'train_val.pt';save_torch(payload,train_source);del train_source
        log('capture_complete',qualified_states=len(qualified),split_counts={k:len(v) for k,v in source['ids'].items()})
        primary=raw_specs(depths,cfg)
        raw_training=execute_stage(primary,source,payload,training_cfg,output,'raw')
        validation_rows=read_predictions(output,primary,cfg['seeds'],'validation_predictions.jsonl')
        selection=select_layers(validation_rows,depths,{d:f'raw_L{d}_HC_K0.1' for d in depths},2)
        write_json(output/'validation_layer_selection.json',selection)
        compact=latent_specs(selection['selected'],cfg)
        latent_training=execute_stage(compact,source,payload,training_cfg,output,'latent')
        specs=primary+compact
        validation_rows+=read_predictions(output,compact,cfg['seeds'],'validation_predictions.jsonl')
        write_jsonl(output/'all_validation_predictions.jsonl',validation_rows)
        write_json(output/'test_access_manifest.json',dict(all_training_complete=True,
            layer_selection_locked=True,selection_split='val',no_test_in_training=True,
            selected_layers=selection['selected'],dimensions_fixed=cfg['latent_dims'],
            native_capture_test_inputs_used_only_for_sanity_and_capture=True))
        test_rows=test_models(specs,source,store,table,cfg,output)
        write_jsonl(output/'all_test_predictions.jsonl',test_rows)
        write_wide_predictions(test_rows,output,depths,selection['selected'])
        checks=dict(disjoint_question_splits=len(set(sum(source['split'].values(),[])))==100,
            exact_saved_K_reproduction=reproduction['passed'],causal_hidden_alignment=alignment['passed'],
            no_logits_as_features=True,no_test_layer_or_dimension_selection=True,
            identical_probe_architecture_across_depths=True,matched_state_subset_for_baseline=True,
            dimensions_use_identical_selected_layers=True,candidate_embeddings_frozen=emb_manifest['frozen'],
            cohort_counts_reported=True)
        write_json(output/'sanity_checks.json',checks)
        gates=build_outputs(output,cfg,depths,selection,specs,validation_rows,test_rows,checks,latency)
        write_json(output/'training_audit.json',dict(raw=raw_training,latent=latent_training,
            direct_baseline_retrained=True,D_encoder_retrained=False,D_dynamics_retrained=False,bridge_trained=False))
        summary.update(status='complete',pipeline_check_only=cfg['pipeline_check_only'],
            seconds=time.perf_counter()-started,captured_states=len(qualified),
            selected_layers=selection['selected'],training_jobs=len(raw_training)+len(latent_training),
            test_prediction_rows=len(test_rows),validation_gate={d:g['status'] for d,g in gates.items()},
            original_capture_states=len(uids),model_identity=source['identity'],
            real_Qwen_call_note='fresh_capture_calls excludes warmup and measured partial/full benchmark forwards')
        write_json(output/'summary.json',summary)
        write_json(manifest_path,dict(schema=cfg['schema'],fingerprint=signature,status='complete'))
        archive=archive_partial(output,force=True);log('complete',**summary,ZIP=str(archive))
        return summary
    except BaseException:
        if store is not None:store.flush_progress()
        summary.update(status='partial',seconds=time.perf_counter()-started)
        write_json(output/'summary.json',summary)
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        try:archive_partial(output,force=True)
        except Exception as error:log('packaging_failed',error=str(error))
        raise
    finally:
        if store is not None:store.close()
        if table is not None and hasattr(table['values'],'_mmap'):
            table['values']._mmap.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker_job')
    parser.add_argument('--input');parser.add_argument('--phase0_input');parser.add_argument('--output')
    parser.add_argument('--config');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--capture_input')
    args=parser.parse_args()
    if args.worker_job:worker(json.loads(Path(args.worker_job).read_text()));return
    if not all((args.input,args.phase0_input,args.output,args.config)):parser.error('Missing required study paths')
    run(args)


if __name__=='__main__':main()
