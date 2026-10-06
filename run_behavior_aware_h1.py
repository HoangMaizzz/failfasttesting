"""Paired H1 objective study on immutable Phase0 source; no LLM calls."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import torch
from torch.nn import functional as F

from behavior_aware_source import prepare_source, sha_file, weight_hash
from behavior_aware_losses import (stage_a_targets, log_survival, objective,
                                   behavior_losses, gradient_norm)
from phase0_wm_data import pack_latents, pack_native_targets
from phase0_wm_models import (DynamicsPair, DirectOutcome, VerifierReadout,
    NativeReconstruction, latent_loss, valid_positions, rollout)
from factorized_wm_metrics import write_json, write_jsonl
from run_latent_wm_phase0 import save_torch, cpu_weights, seed_all, package, release


def log(stage, **data):
    print('[behavior] '+json.dumps(dict(stage=stage,**data)),flush=True)


class BatchSampler:
    """Same seed -> same action, question and edge sequence for A/B1/B2."""
    def __init__(self, paths, rows, seed):
        self.groups=defaultdict(lambda:defaultdict(list));self.rng=random.Random(seed)
        for p in paths:self.groups[p[1][0]][rows[p[0][0]]['question']].append(p)
        self.actions=sorted(self.groups)
        if not self.actions:raise ValueError('No H1 training paths')
        self.questions={a:sorted(self.groups[a]) for a in self.actions}

    def sample(self, n):
        a=self.rng.choice(self.actions)
        return [self.rng.choice(self.groups[a][self.rng.choice(self.questions[a])]) for _ in range(n)]


def frozen_modules(payload, device):
    cfg=payload['source_config'];dim=128
    g=VerifierReadout(dim,1,cfg['dropout'])
    width=next(iter(payload['rows'].values()))['native_target'].shape[-1]
    recon=NativeReconstruction(dim,width)
    g.load_state_dict(payload['frozen']['readout']);recon.load_state_dict(payload['frozen']['reconstruction'])
    for m in (g,recon):m.to(device).eval().requires_grad_(False)
    return g,recon


@torch.no_grad()
def oracle_predictions(payload, device, batch_size):
    g,_=frozen_modules(payload,device);rows=payload['rows'];cache=payload['cachez'];result={}
    uids=sorted(rows)
    for start in range(0,len(uids),batch_size):
        group=[rows[u] for u in uids[start:start+batch_size]]
        state=pack_latents(group,cache,device)
        heads=g(state.z,state.lengths)
        q=log_survival(heads).exp().masked_fill(~valid_positions(state.lengths),0).cpu()
        hazards=heads['hazard'].sigmoid().cpu()
        for i,r in enumerate(group):
            result[r['uid']]=dict(q=q[i],hazards=hazards[i],K=float(q[i,:r['length']].sum()))
    return result


def add_oracle(payload, device, batch):
    payload['oracle']=oracle_predictions(payload,device,batch)
    release()


def pack_oracle(rows, payload, device):
    return torch.stack([payload['oracle'][r['uid']]['q'] for r in rows]).to(device)


@torch.no_grad()
def evaluate(model, method, paths, payload, g, recon, device, batch_size, seed, selection=None):
    model.eval();rows=payload['rows'];cache=payload['cachez'];out=[];state_losses=[]
    for h in (1,2,3):
        pool=[p for p in paths if len(p[1])==h]
        if method=='C' and h>1:continue
        for start in range(0,len(pool),batch_size):
            group=pool[start:start+batch_size];actions=[p[1] for p in group]
            parents=[rows[p[0][0]] for p in group];children=[rows[p[0][-1]] for p in group]
            source=pack_latents(parents,cache,device)
            if method=='C':
                heads=model(source,actions);mse=cos=None
            else:
                pred=rollout(model,source,actions)
                truth=pack_latents(children,cache,device)
                nt,nm=pack_native_targets(children,device)
                state_losses.append(float(latent_loss(pred,truth,recon,nt,nm)))
                heads=g(pred.z,pred.lengths)
                mse=(pred.z-truth.z).square().mean(-1)
                cos=F.cosine_similarity(pred.z,truth.z,dim=-1)
            q=log_survival(heads).exp().cpu();hazards=heads['hazard'].sigmoid().cpu()
            labels=stage_a_targets(children,'cpu')
            for i,((nodes,seq),a,b) in enumerate(zip(group,parents,children)):
                n=b['length'];K=float(q[i,:n].sum());ka,kb=a['accepted'],b['accepted']
                delta=None if ka is None or kb is None else kb-ka
                oracle=payload['oracle'][b['uid']]
                changed=bool((a['ids'][:,1]!=b['ids'][:a['length'],1]).any())
                out.append(dict(method=method,seed=seed,question_id=a['question'],
                    edge_id='|'.join(nodes)+'/'+''.join('E' if x else 'R' for x in seq),
                    action='E' if seq[-1] else 'R',actions=''.join('E' if x else 'R' for x in seq),horizon=h,
                    parent_length=a['length'],child_length=n,K_parent_true=ka,K_child_true=kb,
                    delta_K_true=delta,K_parent_emulator=payload['oracle'][a['uid']]['K'],
                    K_oracle_latent=oracle['K'],K_pred=K,
                    q_oracle=oracle['q'][:n].tolist(),q_pred=q[i,:n].tolist(),
                    q_qwen_truth=None if kb is None else labels['survival'][i,:n].tolist(),
                    hazards_oracle=oracle['hazards'][:n].tolist(),hazards_pred=hazards[i,:n].tolist(),
                    state_mse=None if mse is None else float(mse[i,:n].mean()),
                    state_cos=None if cos is None else float(cos[i,:n].mean()),
                    R_changed=changed if h==1 and seq[0]==0 else False,
                    R_gain=None if delta is None else delta>.5,R_loss=None if delta is None else delta<-.5,
                    E_full_prefix=None if ka is None else ka==a['length'],
                    region_mask_new_block=[j>=a['length'] for j in range(n)],
                    state_fidelity_passed=None if selection is None else selection.get('state_fidelity_passed'),
                    selection_status=None if selection is None else selection.get('status')))
    return out,(sum(state_losses)/len(state_losses) if state_losses else None)


def validation_score(predictions, direct=False):
    groups={
        'R_nonzero':[r for r in predictions if r['horizon']==1 and r['action']=='R' and r['delta_K_true'] is not None and abs(r['delta_K_true'])>.5],
        'E_full_prefix':[r for r in predictions if r['horizon']==1 and r['action']=='E' and r['E_full_prefix'] is True],
    }
    metrics={}
    for name,rows in groups.items():
        byq=defaultdict(list)
        for r in rows:
            target=r['K_child_true'] if direct else r['K_oracle_latent']
            if target is not None:byq[r['question_id']].append(abs(r['K_pred']-target))
        mean=sum(sum(v)/len(v) for v in byq.values())/len(byq) if byq else None
        metrics[name]=dict(count=len(rows),question_count=len(byq),question_macro_MAE=mean)
    values=[v['question_macro_MAE'] for v in metrics.values()]
    return (sum(values)/2 if all(v is not None for v in values) else None),metrics


def select_checkpoint(curve, method, reference, ratio):
    scored=[r for r in curve if r['validation_score'] is not None]
    if not scored:return dict(status='insufficient_validation_cohorts',checkpoint=None,state_fidelity_passed=False)
    diagnostic=min(scored,key=lambda r:(r['validation_score'],r['step']))
    eligible=scored if method in ('A','C') else [r for r in scored if r['validation_state_loss']<=ratio*reference]
    best=min(eligible,key=lambda r:(r['validation_score'],r['step'])) if eligible else diagnostic
    passed=bool(eligible)
    return dict(status='selected' if passed else 'diagnostic_only',
        checkpoint=f"checkpoints/update_{best['step']:06d}.pt",step=best['step'],
        validation_score=best['validation_score'],validation_state_loss=best['validation_state_loss'],
        state_fidelity_passed=passed,eligible_checkpoint_count=len(eligible),reference_A_min_state_loss=reference,
        min_validation_state_loss=min(r['validation_state_loss'] for r in curve if r['validation_state_loss'] is not None)
                                  if method!='C' else None)


def worker(job_path):
    job=json.loads(Path(job_path).read_text());cfg=job['config'];device=job['device'];seed=job['seed'];method=job['method']
    folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);seed_all(seed)
    payload=torch.load(job['payload'],map_location='cpu',weights_only=True)
    if any(r['question'] in set(payload['split']['test']) for r in payload['rows'].values()):
        raise ValueError('Test data entered training worker payload')
    scfg=payload['source_config']
    model=(DirectOutcome(128,scfg['dropout']) if method=='C' else
           DynamicsPair(128,scfg['dynamics_layers'],scfg['dropout'])).to(device)
    initial=weight_hash(cpu_weights(model));g,recon=frozen_modules(payload,device)
    ghash=weight_hash(cpu_weights(g));rhash=weight_hash(cpu_weights(recon))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    train=[p for p in payload['paths']['train'] if len(p[1])==1]
    val=[p for p in payload['paths']['val'] if len(p[1])==1]
    sampler=BatchSampler(train,payload['rows'],seed)
    curve=[];diagnostics=[];batch_digest='';first=1;started=time.perf_counter()
    latest=folder/'resume.pt'
    if latest.is_file():
        old=torch.load(latest,map_location='cpu',weights_only=True)
        if old['initial_hash']!=initial:raise ValueError('Changed paired initialization on resume')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        sampler.rng.setstate(old['sampler_rng']);torch.set_rng_state(old['rng_cpu'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(old['rng_cuda'],device)
        curve=old['curve'];diagnostics=old['diagnostics'];batch_digest=old['batch_digest'];first=old['step']+1
    rows=payload['rows'];cache=payload['cachez'];parameters=list(model.parameters())
    for step in range(first,job['updates']+1):
        model.train();group=sampler.sample(cfg['batch_size']);sequence=[p[1] for p in group]
        batch_digest=hashlib.sha256((batch_digest+'|'+','.join(p[0][0]+'>'+p[0][-1] for p in group)).encode()).hexdigest()
        a=[rows[p[0][0]] for p in group];b=[rows[p[0][-1]] for p in group]
        source=pack_latents(a,cache,device)
        behavior_active=method!='A' and (job['lambda_q']!=0 or job['lambda_K']!=0)
        if method=='C':
            heads=model(source,sequence);state_loss=heads['hazard'].sum()*0
        else:
            pred=model(source,torch.tensor([p[1][0] for p in group],device=device))
            target=pack_latents(b,cache,device);nt,nm=pack_native_targets(b,device)
            state_loss=latent_loss(pred,target,recon,nt,nm)
            # Frozen G keeps input gradients. NEVER no_grad this imagined path.
            heads=g(pred.z,pred.lengths) if behavior_active else None
        oracle=pack_oracle(b,payload,device).detach() if method=='B1' and behavior_active else None
        loss,qloss,kloss=objective(state_loss,heads,b,a,method,job['lambda_q'],job['lambda_K'],oracle)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite objective')
        if step<=cfg['gradient_diagnostic_updates']:
            norms={name:gradient_norm(value,parameters) for name,value in
                   (('state',state_loss),('survival',qloss),('K',kloss))}
            supported=method=='B1' or any(r['accepted'] is not None for r in b)
            if behavior_active and supported and (norms['survival'] is None or norms['survival']<=0):
                raise RuntimeError('Behavior loss does not backpropagate to trainable model')
            diagnostics.append(dict(step=step,raw_gradient_norms=norms,
                lambda_q=job['lambda_q'],lambda_K=job['lambda_K']))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None for m in (g,recon) for p in m.parameters()):
            raise RuntimeError('Gradient accumulated in frozen G/reconstruction')
        torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
        if step%cfg['eval_every']==0 or step==job['updates']:
            predictions,state_val=evaluate(model,method,val,payload,g,recon,device,cfg['batch_size'],seed)
            score,cohorts=validation_score(predictions,direct=method=='C')
            item=dict(step=step,loss=float(loss.detach()),state_loss=float(state_loss.detach()),
                survival_loss=None if qloss is None else float(qloss.detach()),
                K_loss=None if kloss is None else float(kloss.detach()),validation_score=score,
                validation_state_loss=state_val,validation_cohorts=cohorts,
                elapsed_seconds=time.perf_counter()-started,batch_digest=batch_digest)
            curve.append(item)
            save_torch(folder/f'checkpoints/update_{step:06d}.pt',dict(model=cpu_weights(model)))
            save_torch(latest,dict(model=cpu_weights(model),optimizer=optimizer.state_dict(),step=step,
                initial_hash=initial,sampler_rng=sampler.rng.getstate(),rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                curve=curve,diagnostics=diagnostics,batch_digest=batch_digest))
            write_json(folder/'learning_curve.json',curve);write_json(folder/'gradient_diagnostics.json',diagnostics)
            log('train',method=method,seed=seed,pilot=job['pilot'],**item)
    if ghash!=weight_hash(cpu_weights(g)) or rhash!=weight_hash(cpu_weights(recon)):
        raise RuntimeError('Frozen checkpoint weights changed')
    selection=select_checkpoint(curve,method,job.get('reference'),cfg['state_fidelity_ratio'])
    selection.update(method=method,seed=seed,initial_hash=initial,batch_digest=batch_digest,
        lambda_q=job['lambda_q'],lambda_K=job['lambda_K'],updates=job['updates'],
        elapsed_seconds=time.perf_counter()-started,
        frozen_verifier_hash=ghash,frozen_reconstruction_hash=rhash)
    write_json(folder/'selection.json',selection);write_json(folder/'complete.json',selection)
    return selection


def config_check(cfg):
    if cfg['latent_dim']!=128:raise ValueError('Primary source must remain 128D')
    if not cfg['seeds'] or any(s not in (42,43,44) for s in cfg['seeds']) or 42 not in cfg['seeds']:
        raise ValueError('Seeds must include42 and be subset of42/43/44')
    for k in ('max_updates','pilot_updates','eval_every','batch_size','bootstrap_samples','workers'):
        if cfg[k]<1:raise ValueError(f'{k} must be positive')
    if cfg['workers']>2:raise ValueError('At most two workers')
    if cfg['state_fidelity_ratio']<1:raise ValueError('Fidelity ratio must be >=1')
    if not cfg['lambda_grid'] or any(len(x)!=2 or min(x)<0 for x in cfg['lambda_grid']):
        raise ValueError('Invalid lambda grid')


def source_hashes():
    names=['run_behavior_aware_h1.py','behavior_aware_source.py','behavior_aware_losses.py','behavior_aware_metrics.py',
           'phase0_wm_models.py','phase0_wm_data.py','factorized_wm_data.py']
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}


def make_job(method,seed,folder,cfg,payload,pilot=False,lam=(0,0),reference=None):
    return dict(method=method,seed=seed,folder=str(folder),config=cfg,payload=str(payload),pilot=pilot,
        updates=cfg['pilot_updates'] if pilot else cfg['max_updates'],lambda_q=lam[0],lambda_K=lam[1],reference=reference)


def execute_jobs(jobs,devices,output,cache):
    slots=queue.Queue();requested=devices
    for d in requested:slots.put(d)
    cancelled=threading.Event();lock=threading.Lock();active=set()
    root=Path(output['root'])
    def execute(job):
        folder=Path(job['folder'])
        if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
        if cancelled.is_set():raise RuntimeError('Cancelled after another worker failed')
        device=slots.get()
        try:
            if cancelled.is_set():raise RuntimeError('Cancelled after another worker failed')
            job=dict(job,device=device)
            # lambda_search folders share seed; fully unique path by job hash.
            path=Path(cache)/('job_'+hashlib.sha256(str(folder).encode()).hexdigest()[:16]+'.json')
            write_json(path,job)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker_job',str(path)]
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker_log.txt').open('a',encoding='utf-8') as logfile, subprocess.Popen(
                    command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as process:
                with lock:active.add(process)
                try:
                    for line in process.stdout:
                        logfile.write(line);logfile.flush();print(line,end='',flush=True)
                    code=process.wait()
                except BaseException:
                    process.terminate();process.wait();raise
                finally:
                    with lock:active.discard(process)
            if code:raise RuntimeError(f'Worker {job["method"]} seed{job["seed"]} failed with exit {code}')
            return json.loads((folder/'complete.json').read_text())
        finally:slots.put(device)
    with ThreadPoolExecutor(max_workers=len(requested)) as executor:
        # Jobs complete in parallel; archive in parent only to avoid ZIP races.
        futures=[executor.submit(execute,j) for j in jobs]
        results={}
        try:
            for future in as_completed(futures):
                results[futures.index(future)]=future.result();package(root)
        except BaseException:
            cancelled.set()
            for future in futures:future.cancel()
            with lock:
                for process in active:
                    if process.poll() is None:process.terminate()
            raise
    return [results[i] for i in range(len(jobs))]


def study_fingerprint(cfg, provenance, code):
    # A source ZIP can move to a different mounted path between runtimes.
    # Identity is determined by hashes, not the machine's mount pathname.
    content={k:v for k,v in provenance.items() if k!='source'}
    return hashlib.sha256(json.dumps(dict(config=cfg,source=content,code=code),sort_keys=True).encode()).hexdigest()


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    summary=dict(schema='behavior_aware_h1_v1',status='running',LLM_forwards=0,encoder_retrained=False)
    try:
        cfg=json.loads(Path(args.config).read_text());config_check(cfg)
        summary['pipeline_check_only']=bool(cfg.get('pipeline_check_only',False))
        torch.set_num_threads(4)
        cache=Path(tempfile.mkdtemp(prefix='behavior_h1_'))
        devices=cfg.get('devices',['cuda:0','cuda:1'])[:cfg['workers']]
        if len(devices)!=cfg['workers']:raise ValueError('Not enough worker devices')
        if any(d.startswith('cuda') for d in devices) and torch.cuda.device_count()<len(set(devices)):
            raise ValueError('Need the requested CUDA devices')
        log('verify_frozen_source',phase0_input=str(args.phase0_input))
        source=prepare_source(args.input,args.phase0_input,cache,cfg['encoder_verification_samples'],devices[0])
        fingerprint=study_fingerprint(cfg,source['provenance'],source_hashes())
        manifest=output/'study_manifest.json'
        if args.resume:
            previous=json.loads(manifest.read_text())
            if previous['fingerprint']!=fingerprint:raise ValueError('Resume config/source/code mismatch')
        elif manifest.exists():raise FileExistsError('Choose new output or --resume')
        write_json(output/'source_checkpoint_manifest.json',source['provenance'])
        write_json(output/'split_manifest.json',source['split']);write_json(output/'config.json',cfg)
        write_json(manifest,dict(schema='behavior_aware_h1_v1',fingerprint=fingerprint,status='running'))
        write_json(output/'loss_definition.json',dict(A='unchanged Phase0 latent_loss',
            B1='state + lambda_q*stable soft BCE survival(G(pred), detached G(real)) + lambda_K*Huber(expectedK)',
            B2='state + lambda_q*stable survival BCE(Qwen) + lambda_K*Huber(observed true-parent anchored delta)',
            C='same DirectOutcome architecture; survival BCE + Huber(observed true-parent anchored delta)',
            Qwen_target='exact StageA zero-based i<K, no missing-label fill',
            delta_MAE_identity='true-parent anchored delta-MAE equals child K-MAE on delta-observed rows',
            G_parameters_frozen=True,G_imagined_input_gradients=True,G_oracle_detached=True,
            selection_uses_test=False,scope='reused questions; exploratory'))
        rows=source['rows'];trainval=set(source['split']['train']+source['split']['val'])
        training={k:v for k,v in source.items() if k not in ('rows','cachez','paths','provenance')}
        training.update(rows={u:r for u,r in rows.items() if r['question'] in trainval},
            cachez={u:z for u,z in source['cachez'].items() if rows[u]['question'] in trainval},
            paths={k:source['paths'][k] for k in ('train','val')})
        add_oracle(training,devices[0],cfg['batch_size'])
        train_file=cache/'train_val.pt';save_torch(train_file,training)
        # A single fingerprinted H1 validation rule is fixed BEFORE pilot.
        check_rows=[]
        for nodes,acts in training['paths']['val']:
            if len(acts)==1:
                a,b=(rows[nodes[0]],rows[nodes[-1]])
                check_rows.append(dict(horizon=1,action='E' if acts[0] else 'R',question_id=a['question'],
                    delta_K_true=None if a['accepted'] is None or b['accepted'] is None else b['accepted']-a['accepted'],
                    E_full_prefix=a['accepted']==a['length'] if a['accepted'] is not None else None,
                    K_pred=0,K_child_true=b['accepted'],K_oracle_latent=0))
        score,cohorts=validation_score(check_rows)
        if score is None:raise ValueError('Missing validation R-nonzero/E-full-prefix cohort; cannot silently change selection')
        write_json(output/'selection_protocol.json',dict(rule=cfg['protocol']['selection'],cohorts=cohorts,
            state_ratio=cfg['state_fidelity_ratio'],test_access=False,whole_question_macro=True))
        del training;release();package(output)
        dispatch=dict(devices=devices,root=str(output))
        ajobs=[make_job('A',s,output/'A_pure'/f'seed{s}',cfg,train_file) for s in cfg['seeds']]
        A=execute_jobs(ajobs,devices,dispatch,cache)
        refs={r['seed']:r['min_validation_state_loss'] for r in A}
        pilots=[]
        for method,directory in (('B1','B1_consistency'),('B2','B2_qwen')):
            for i,lam in enumerate(cfg['lambda_grid']):
                pilots.append(make_job(method,42,output/directory/'lambda_search'/f'candidate_{i:02d}',
                    cfg,train_file,True,lam,refs[42]))
        trial_results=execute_jobs(pilots,devices,dispatch,cache)
        chosen={}
        for method in ('B1','B2'):
            candidates=[r for r in trial_results if r['method']==method and r['status']=='selected']
            chosen[method]=min(candidates,key=lambda r:r['validation_score']) if candidates else None
        write_json(output/'lambda_selection.json',dict(status='locked',selection_split='val',pilot_seed=42,
            chosen=chosen,test_access=False,no_eligible_policy=cfg['protocol']['no_eligible_pilot']))
        finals=[]
        for method,directory in (('B1','B1_consistency'),('B2','B2_qwen')):
            if chosen[method] is not None:
                lam=(chosen[method]['lambda_q'],chosen[method]['lambda_K'])
                finals += [make_job(method,s,output/directory/f'seed{s}',cfg,train_file,False,lam,refs[s]) for s in cfg['seeds']]
        finals += [make_job('C',s,output/'C_direct'/f'seed{s}',cfg,train_file,False,(1,1)) for s in cfg['seeds']]
        trained=A+execute_jobs(finals,devices,dispatch,cache)
        # Validate paired initialization and batch prefixes at identical steps,
        # including pilots. These are audit artifacts, not merely seed promises.
        for r in trial_results+trained:
            if r['method']!='C':
                baseline=next(a for a in A if a['seed']==r['seed'])
                if r['initial_hash']!=baseline['initial_hash']:raise ValueError('Paired initialization mismatch')
        audit=[]
        for job in ajobs+pilots+finals:
            if job['method']=='C':continue
            curve=json.loads((Path(job['folder'])/'learning_curve.json').read_text())
            base=json.loads((output/'A_pure'/f"seed{job['seed']}"/'learning_curve.json').read_text())
            digest_by_step={r['step']:r['batch_digest'] for r in base}
            if any(r['step'] in digest_by_step and r['batch_digest']!=digest_by_step[r['step']] for r in curve):
                raise ValueError('Paired batch schedule mismatch')
            audit.append(dict(folder=str(Path(job['folder']).relative_to(output)),seed=job['seed'],matched_steps=len(curve)))
        write_json(output/'paired_training_audit.json',audit)
        write_json(output/'test_access_manifest.json',dict(opened_after_lambda_lock=True,
            all_final_training_complete=True,lambda_file_sha256=sha_file(output/'lambda_selection.json'),
            frozen_checkpoint_hashes=source['provenance']['artifact_hashes'],test_questions=source['split']['test']))
        # ONLY NOW run G or any candidate on held-out test states.
        testq=set(source['split']['test'])
        test_payload=dict(source_config=source['source_config'],frozen=source['frozen'],
            rows={u:r for u,r in rows.items() if r['question'] in testq},
            cachez={u:z for u,z in source['cachez'].items() if rows[u]['question'] in testq})
        add_oracle(test_payload,devices[0],cfg['batch_size'])
        g,recon=frozen_modules(test_payload,devices[0]);prediction_rows=[]
        for job in ajobs+finals:
            folder=Path(job['folder']);selection=json.loads((folder/'selection.json').read_text())
            if selection['checkpoint'] is None:continue
            scfg=source['source_config'];seed_all(job['seed'])
            model=(DirectOutcome(128,scfg['dropout']) if job['method']=='C' else
                DynamicsPair(128,scfg['dynamics_layers'],scfg['dropout'])).to(devices[0])
            model.load_state_dict(torch.load(folder/selection['checkpoint'],map_location='cpu',weights_only=True)['model'])
            pred,_=evaluate(model,job['method'],source['paths']['test'],test_payload,g,recon,
                devices[0],cfg['batch_size'],job['seed'],selection)
            write_jsonl(folder/'test_predictions.jsonl',pred);prediction_rows+=pred
            model.cpu();del model;release()
        write_jsonl(output/'all_predictions.jsonl',prediction_rows)
        from behavior_aware_metrics import write_behavior_reports
        report_cfg=dict(cfg,selection=dict(A=A,final=trained,lambda_selection=chosen),
            source_manifest=source['provenance'])
        write_behavior_reports(prediction_rows,output,report_cfg)
        summary.update(status='complete',seconds=time.perf_counter()-started,seeds=cfg['seeds'],
            trained_methods=sorted({r['method'] for r in trained}),lambda_selected={k:v is not None for k,v in chosen.items()},
            train_jobs=len(ajobs)+len(pilots)+len(finals),updates_completed=sum(r['updates'] for r in trial_results+trained),
            test_prediction_rows=len(prediction_rows),split=dict(train=70,val=15,test=15))
        write_json(output/'summary.json',summary)
        current=json.loads(manifest.read_text());current['status']='complete';write_json(manifest,current)
        package(output);log('complete',**summary)
        return summary
    except BaseException:
        summary.update(status='partial',seconds=time.perf_counter()-started)
        write_json(output/'summary.json',summary);(output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        package(output);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker_job')
    parser.add_argument('--input');parser.add_argument('--phase0_input');parser.add_argument('--output')
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/behavior_aware_h1.json'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.worker_job:worker(args.worker_job)
    elif all((args.input,args.phase0_input,args.output)):run(args)
    else:parser.error('--input --phase0_input --output are required')


if __name__=='__main__':main()
