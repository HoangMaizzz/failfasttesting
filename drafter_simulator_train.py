"""Prompt-disjoint training, free-running rollout, ablations and diagnosis."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drafter_simulator_models import (ObservationEncoder, BehaviorHeads, Transition,
                                      DirectBehaviorPredictor, behavior_loss, project_masks)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StateStore:
    """Only current/previous same-trajectory observations enter the encoder."""
    def __init__(self, output):
        self.output = Path(output)
        self.manifest = json.loads((self.output / 'capture_manifest.json').read_text())
        self.states = self.manifest['states']
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 2 * 1024**3
        self.groups = defaultdict(list)
        for i, row in enumerate(self.states):
            self.groups[row['group_id']].append(i)
        self.previous, self.next = {}, {}
        for indices in self.groups.values():
            indices.sort(key=lambda i: self.states[i]['step'])
            for a, b in zip(indices, indices[1:]):
                if self.states[b]['forward_id'] != self.states[a]['forward_id'] + 1:
                    raise ValueError('Refine edge skipped a native forward')
                self.previous[b], self.next[a] = a, b
        if not self.states:
            raise ValueError('No denoising observations; inspect collection diagnostics')
        first = self.arrays(0)
        self.hidden_dim = first['hidden'].shape[-1]
        self.content_dim = first['candidate_emb'].shape[-1]
        d, c = self.hidden_dim, self.content_dim
        if c != 64:
            raise ValueError('Expected native semantic embedding projection width 64')
        self.slices = dict(hidden=slice(0, d), content=slice(d, d + 2*c),
                           structure=slice(d + 2*c, d + 2*c + 8),
                           confidence=slice(d + 2*c + 8, d + 2*c + 11),
                           temporal=slice(d + 2*c + 11, d + 2*c + 14))
        self.input_dim = d + 2*c + 14
        self.mean, self.std = np.zeros(self.input_dim, np.float32), np.ones(self.input_dim, np.float32)

    def arrays(self, index):
        meta = self.states[index]
        name = meta['npz']
        if name not in self.cache:
            with np.load(self.output / name, allow_pickle=False) as file:
                self.cache[name] = {key: file[key] for key in file.files}
            self.cache_bytes += sum(a.nbytes for a in self.cache[name].values())
            # Avoid decompressing a question NPZ for every random mini-batch,
            # while keeping bounded CPU storage (never on the drafter GPU).
            while self.cache_bytes > self.cache_limit and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= sum(a.nbytes for a in old.values())
        self.cache.move_to_end(name)
        return {key: value[meta['row']] for key, value in self.cache[name].items()}

    def raw_features(self, index):
        a, meta = self.arrays(index), self.states[index]
        width = len(a['mask'])
        mask, eligible = a['mask'].astype(bool), a['eligible'].astype(bool)
        ratio = float(mask[eligible].mean()) if eligible.any() else 0.
        structural = np.stack([mask, eligible, np.arange(width)/max(1,width-1),
            (meta['block_start'] + np.arange(width))/4096.,
            np.full(width, meta['context_len']/4096.), np.full(width, meta['step']/8.),
            np.full(width, meta['small_block_index']/max(1,width//8-1)), np.full(width, ratio)], -1)
        confidence = np.stack([a['confidence'], a['entropy'], a['margin']], -1)
        temporal = np.zeros((width, 3), np.float32)
        if index in self.previous:
            previous = self.arrays(self.previous[index])
            temporal = np.stack([a['confidence']-previous['confidence'],
                a['entropy']-previous['entropy'], a['candidate_ids'] != previous['candidate_ids']], -1)
        return np.concatenate([a['hidden'], a['token_emb'], a['candidate_emb'],
                               structural, confidence, temporal], -1).astype(np.float32)

    def fit_normalization(self, train_ids):
        # Streaming train-only statistics; no fitting on held-out prompts.
        ids = list(train_ids)
        if not ids:
            raise ValueError('No training states')
        total = np.zeros(self.input_dim, np.float64)
        square = np.zeros_like(total)
        count = 0
        for index in ids:
            x = self.raw_features(index)
            total += x.sum(0); square += (x.astype(np.float64)**2).sum(0); count += len(x)
        normalized = self.hidden_dim + 2*self.content_dim
        self.mean[:normalized] = (total[:normalized]/count).astype(np.float32)
        variance = np.maximum(square[:normalized]/count-(total[:normalized]/count)**2, 1e-4)
        self.std[:normalized] = np.sqrt(variance).astype(np.float32)

    def features(self, index, removed=()):
        x = (self.raw_features(index)-self.mean)/self.std
        for group in removed:
            x[:, self.slices[group]] = 0.
        return x

    def truth(self, index):
        a = self.arrays(index)
        d, c = self.hidden_dim, self.content_dim
        return dict(mask=a['mask'].astype(np.float32), confidence=a['confidence'].astype(np.float32),
                    entropy=a['entropy'].astype(np.float32), margin=a['margin'].astype(np.float32),
                    content=(a['candidate_emb'].astype(np.float32)-self.mean[d+c:d+2*c])/self.std[d+c:d+2*c])

    def paths(self, question_ids, horizon=1):
        questions = set(question_ids)
        paths = []
        for index, meta in enumerate(self.states):
            if meta['question_id'] not in questions:
                continue
            path = [index]
            while len(path) <= horizon and path[-1] in self.next:
                path.append(self.next[path[-1]])
            if len(path) > 1:
                paths.append(path)
        return paths


def tensors(values, device):
    return torch.as_tensor(np.asarray(values), device=device, dtype=torch.float32)


def pack_truth(store, ids, device):
    truth = [store.truth(i) for i in ids]
    return {key: tensors([row[key] for row in truth], device) for key in truth[0]}


def sample_paths(store, paths, rng, size):
    by_question = defaultdict(list)
    for path in paths:
        by_question[store.states[path[0]]['question_id']].append(path)
    keys = sorted(by_question)
    if not keys:
        raise ValueError('No valid same-group Refine edges in this split')
    result = []
    for _ in range(size):
        pool = by_question[keys[int(rng.integers(len(keys)))]]
        result.append(pool[int(rng.integers(len(pool)))])
    return result


def fit_encoder(store, ids, val_ids, cfg, seed, removed, device):
    seed_all(seed)
    enc = ObservationEncoder(store.input_dim, cfg['latent_dim']).to(device)
    heads = BehaviorHeads(cfg['latent_dim']).to(device)
    optimizer = torch.optim.AdamW(list(enc.parameters())+list(heads.decode.parameters()), lr=3e-4)
    rng = np.random.default_rng(seed)
    by_question = defaultdict(list)
    for i in ids:
        by_question[store.states[i]['question_id']].append(i)
    keys = sorted(by_question)
    best, saved = float('inf'), None
    for step in range(1, cfg['encoder_updates']+1):
        selected = []
        for _ in range(cfg['batch_size']):
            pool = by_question[keys[int(rng.integers(len(keys)))]]
            selected.append(pool[int(rng.integers(len(pool)))])
        x = tensors([store.features(i, removed) for i in selected], device)
        z = enc(x)
        reconstruction = enc.reconstruct(z)
        losses = [F.mse_loss(reconstruction[..., sl], x[..., sl])
                  for name, sl in store.slices.items() if name not in removed]
        truth = pack_truth(store, selected, device)
        eligible = tensors([store.arrays(i)['eligible'] for i in selected], device)
        loss = sum(losses)/max(1,len(losses)) + behavior_loss(heads(z), truth, eligible)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(heads.parameters()), 1.)
        optimizer.step()
        if step % cfg['eval_every'] == 0 or step == cfg['encoder_updates']:
            with torch.no_grad():
                use = val_ids[:min(len(val_ids), 128)]
                vx = tensors([store.features(i, removed) for i in use], device)
                vz = enc(vx)
                ve = tensors([store.arrays(i)['eligible'] for i in use], device)
                value = float(behavior_loss(heads(vz), pack_truth(store,use,device), ve))
            if value < best:
                best, saved = value, (copy.deepcopy(enc.state_dict()), copy.deepcopy(heads.state_dict()))
    enc.load_state_dict(saved[0]); heads.load_state_dict(saved[1])
    enc.eval(); heads.eval()
    for p in list(enc.parameters())+list(heads.decode.parameters()):
        p.requires_grad_(False)
    return enc, heads, dict(validation_behavior_loss=best)


def fit_transition(store, train_paths, val_paths, enc, heads, cfg, seed, kind,
                   horizon, removed, device, output, name, direct=False):
    seed_all(seed)
    heads = copy.deepcopy(heads)
    model = (DirectBehaviorPredictor(store.input_dim, cfg['latent_dim']) if direct
             else Transition(cfg['latent_dim'], kind, gated=cfg.get('gated', False))).to(device)
    parameters = list(model.parameters()) + ([] if direct else list(heads.change.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    best, saved, log = float('inf'), None, []
    for step in range(1,cfg['updates']+1):
        paths = sample_paths(store, train_paths, rng, cfg['batch_size'])
        start = [p[0] for p in paths]
        x = tensors([store.features(i, removed) for i in start], device)
        with torch.no_grad(): z = enc(x)
        loss = x.sum()*0.; contributions = 0
        for h in range(1, horizon+1):
            active = [j for j,p in enumerate(paths) if len(p)>h]
            if not active:
                break
            parents = [paths[j][h-1] for j in active]
            future = [paths[j][h] for j in active]
            # The simulated state is never replaced by an observed future state.
            if direct:
                pred = model(x)
                znext = None
            else:
                znext = model(z)
                pred = heads(znext, z)
            pred = {key:value[active] for key,value in pred.items()}
            use = tensors([store.arrays(i)['eligible'] for i in parents], device)
            mask = tensors([store.arrays(i)['mask'] for i in parents], device)
            stability = tensors([store.arrays(a)['candidate_ids']==store.arrays(b)['candidate_ids']
                                 for a,b in zip(parents,future)], device)
            step_loss = behavior_loss(pred, pack_truth(store,future,device), use, mask, stability)
            if not direct:
                with torch.no_grad():
                    target = enc(tensors([store.features(i,removed) for i in future],device))
                step_loss = step_loss + F.smooth_l1_loss(znext[active], target)
                z = znext
            loss = loss + step_loss; contributions += 1
        loss = loss / max(1,contributions)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step()
        if step%cfg['eval_every']==0 or step==cfg['updates']:
            report = evaluate(store,val_paths,enc,heads,model,removed,device,1,
                              direct=direct,max_paths=cfg.get('evaluation_paths',512))
            metrics = report.get('1',{}).get('macro',{})
            value = metrics.get('commit_brier',1.)+metrics.get('confidence_mae',1.)
            log.append(dict(update=step,loss=float(loss.detach()),validation=metrics))
            print(f'[train] {name} seed={seed} update={step} loss={float(loss.detach()):.4f} val={value:.4f}',flush=True)
            if value<best:
                best,saved=value,(copy.deepcopy(model.state_dict()),copy.deepcopy(heads.state_dict()))
    model.load_state_dict(saved[0]); heads.load_state_dict(saved[1]); model.eval(); heads.eval()
    write_json(Path(output)/'training'/f'{name}_seed{seed}.json',log)
    return model,heads,dict(best_validation_score=best,updates=cfg['updates'])


def binary_metrics(p, truth):
    p,truth=np.asarray(p,np.float64),np.asarray(truth,bool)
    pred=p>=.5; tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum())
    denominator=2*tp+fp+fn
    result=dict(f1=2*tp/denominator if denominator else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                brier=float(np.mean((p-truth)**2)),prevalence=float(truth.mean()))
    bins=np.minimum((p*10).astype(int),9)
    result['ece']=sum(float((bins==b).mean())*abs(float(p[bins==b].mean())-
        float(truth[bins==b].mean())) for b in range(10) if (bins==b).any())
    # AP with tied-score groups, not a token-order-dependent rank artifact.
    if truth.any():
        order=np.argsort(-p,kind='stable'); scores=p[order]; y=truth[order]
        ends=np.r_[np.where(scores[1:]!=scores[:-1])[0]+1,len(scores)]
        cumulative=np.cumsum(y)[ends-1]; recall=cumulative/int(truth.sum())
        precision=cumulative/ends
        result['average_precision']=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    else: result['average_precision']=None
    if truth.any() and (~truth).any():
        order=np.argsort(-p,kind='stable');scores=p[order];y=truth[order]
        ends=np.r_[np.where(scores[1:]!=scores[:-1])[0]+1,len(scores)]
        tpr=np.r_[0.,np.cumsum(y)[ends-1]/truth.sum()]
        fpr=np.r_[0.,np.cumsum(~y)[ends-1]/(~truth).sum()]
        result['auroc']=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])*.5))
    else:result['auroc']=None
    return result


def summarize_records(records):
    if not records: return dict(paths=0,macro={},micro={},questions={})
    byq=defaultdict(list)
    for r in records: byq[r['question_id']].append(r)
    def aggregate(rows):
        p=np.concatenate([r['commit_p'] for r in rows]); y=np.concatenate([r['commit_truth'] for r in rows])
        binary=binary_metrics(p,y)
        result={'commit_'+k:v for k,v in binary.items()}
        for k in ('confidence_mae','entropy_mae','margin_mae','latent_rmse','delta_relative_error',
                  'content_cosine','unmask_count_mae','raw_mask_violation_rate',
                  'changed_candidate_confidence_mae','remaining_mask_confidence_mae'):
            vals=[r[k] for r in rows if r.get(k) is not None]
            result[k]=float(np.mean(vals)) if vals else None
        sp=np.concatenate([r['stability_p'] for r in rows]); sy=np.concatenate([r['stability_truth'] for r in rows])
        result.update({'stability_'+k:v for k,v in binary_metrics(sp,sy).items()})
        result.update({'candidate_change_'+k:v for k,v in binary_metrics(1-sp,~sy).items()})
        return result
    qmetrics={q:aggregate(rows) for q,rows in byq.items()}
    keys=next(iter(qmetrics.values())).keys()
    macro={k:float(np.mean([v[k] for v in qmetrics.values() if v[k] is not None]))
           if any(v[k] is not None for v in qmetrics.values()) else None for k in keys}
    intervals={}
    rng=np.random.default_rng(81)
    for key in ('commit_brier','confidence_mae','unmask_count_mae'):
        values=np.array([r[key] for r in qmetrics.values() if r[key] is not None])
        if len(values)>1:
            means=values[rng.integers(0,len(values),size=(1000,len(values)))].mean(1)
            intervals[key]=[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]
    worst=sorted(records,key=lambda r:float(np.mean((r['commit_p']-r['commit_truth'])**2)),reverse=True)[:10]
    examples=[]
    for row in worst:
        example={k:v for k,v in row.items() if k not in ('stability_p','stability_truth')}
        for key in ('commit_p','commit_truth'):
            example[key]=np.asarray(example[key]).tolist()
        examples.append(example)
    return dict(paths=len(records),macro=macro,micro=aggregate(records),questions=qmetrics,
                prompt_bootstrap_95ci=intervals,worst_commit_examples=examples)


@torch.inference_mode()
def evaluate(store, paths, enc, heads, model, removed, device, horizon=3,
             direct=False, teacher=False, oracle=False, persistence=False,
             native_heuristic=False,max_paths=None):
    paths=list(paths)
    if max_paths and len(paths)>max_paths:
        # Balanced deterministic prompt sampling; never select by target value.
        paths=sample_paths(store,paths,np.random.default_rng(909),max_paths)
    records=defaultdict(list)
    for offset in range(0,len(paths),32):
        batch=paths[offset:offset+32]
        start=[p[0] for p in batch]
        x=tensors([store.features(i,removed) for i in start],device)
        z=enc(x); initial=z.clone()
        simulated_mask=tensors([store.arrays(i)['mask'] for i in start],device).bool()
        for h in range(1,horizon+1):
            active=[j for j,p in enumerate(batch) if len(p)>h]
            if not active: break
            if teacher:
                # Teacher-forced diagnostic is explicitly labeled, never a rollout.
                real_previous=[p[min(h-1,len(p)-1)] for p in batch]
                z=enc(tensors([store.features(i,removed) for i in real_previous],device))
            if teacher or oracle:
                real_previous=[p[min(h-1,len(p)-1)] for p in batch]
                simulated_mask=tensors([store.arrays(i)['mask'] for i in real_previous],device).bool()
            future=[p[min(h,len(p)-1)] for p in batch]
            target=enc(tensors([store.features(i,removed) for i in future],device))
            nextz=(target if oracle else z if persistence or native_heuristic else model(z)) if not direct else z
            pred=(model(x) if direct else heads(nextz,z))
            if persistence or native_heuristic:
                copied=pack_truth(store,start,device)
                copied_mask=simulated_mask.clone()
                if native_heuristic:
                    active_positions=simulated_mask & tensors([store.arrays(i)['eligible'] for i in start],device).bool()
                    commit=active_positions & (copied['confidence']>.5)
                    for j in range(len(batch)):
                        if bool(active_positions[j].any()) and not bool(commit[j].any()):
                            confidence=copied['confidence'][j].masked_fill(~active_positions[j],-float('inf'))
                            commit[j,confidence.argmax()]=True
                    copied_mask=copied_mask & ~commit
                pred=dict(mask_logits=torch.where(copied_mask,20.,-20.),
                    **{k:v for k,v in copied.items() if k!='mask'},
                    stability_logits=torch.full_like(copied['mask'],20.))
            probability=pred['mask_logits'].sigmoid()
            eligible=tensors([store.arrays(i)['eligible'] for i in start],device).bool()
            projected=(simulated_mask.clone() if persistence else
                       copied_mask if native_heuristic else project_masks(probability,simulated_mask,eligible))
            for j in active:
                parent_id=batch[j][h-1]; future_id=batch[j][h]
                a,b=store.arrays(parent_id),store.arrays(future_id)
                # Cumulative unmask prediction on the fixed initial eligible mask set.
                first=store.arrays(batch[j][0])
                use=first['mask'].astype(bool)&first['eligible'].astype(bool)
                if not use.any(): continue
                truecommit=~b['mask'][use].astype(bool)
                stability_use=a['mask'].astype(bool)&a['eligible'].astype(bool)
                commit_p=(1-probability[j,use]).cpu().numpy()
                realtruth=store.truth(future_id)
                content=pred['content'][j,use]; truecontent=tensors(realtruth['content'][use],device)
                cos=F.cosine_similarity(content,truecontent,dim=-1).mean()
                delta=torch.linalg.vector_norm(target[j]-initial[j])
                err=torch.linalg.vector_norm(nextz[j]-target[j])
                record=dict(question_id=store.states[parent_id]['question_id'],state_id=store.states[parent_id]['uid'],
                    start_state_id=store.states[batch[j][0]]['uid'],
                    eligible_position_indices=np.where(use)[0].tolist(),
                    current_candidate_ids=a['candidate_ids'][use].astype(int).tolist(),
                    real_next_candidate_ids=b['candidate_ids'][use].astype(int).tolist(),
                    current_mask=a['mask'][use].astype(bool).tolist(),
                    real_next_mask=b['mask'][use].astype(bool).tolist(),
                    projected_next_mask=projected[j,use].cpu().numpy().astype(bool).tolist(),
                    current_confidence=a['confidence'][use].astype(float).tolist(),
                    real_next_confidence=b['confidence'][use].astype(float).tolist(),
                    predicted_next_confidence=pred['confidence'][j,use].cpu().numpy().astype(float).tolist(),
                    horizon=h,commit_p=commit_p,commit_truth=truecommit,
                    stability_p=(pred['stability_logits'][j,stability_use].sigmoid().cpu().numpy()
                                 if 'stability_logits' in pred else np.ones(stability_use.sum())),
                    stability_truth=a['candidate_ids'][stability_use]==b['candidate_ids'][stability_use],
                    latent_rmse=None if direct else float((nextz[j]-target[j]).square().mean().sqrt()),
                    delta_relative_error=None if direct or float(delta)<1e-6 else float(err/delta),
                    content_cosine=float(cos),
                    unmask_count_mae=float(abs((~projected[j,use]).sum().item()-truecommit.sum())),
                    raw_mask_violation_rate=float((probability[j,eligible[j]&~simulated_mask[j]]>.5).float().mean())
                        if bool((eligible[j]&~simulated_mask[j]).any()) else 0.)
                for key in ('confidence','entropy','margin'):
                    record[key+'_mae']=float(np.abs(pred[key][j,use].cpu().numpy()-realtruth[key][use]).mean())
                change=(a['candidate_ids']!=b['candidate_ids']) & use
                remain=a['mask'].astype(bool) & a['eligible'].astype(bool)
                for label,subset in (('changed_candidate',change),('remaining_mask',remain)):
                    record[label+'_confidence_mae']=(float(np.abs(
                        pred['confidence'][j,subset].cpu().numpy()-realtruth['confidence'][subset]).mean())
                        if subset.any() else None)
                records[h].append(record)
            if direct: break
            z=nextz; simulated_mask=projected
    return {str(h):summarize_records(rows) for h,rows in sorted(records.items())}


def timed(call, device, warmup, repeats):
    for _ in range(warmup): call()
    values=[]; wall=[]
    for _ in range(repeats):
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                before=time.perf_counter(); start.record();call();end.record();end.synchronize()
                values.append(start.elapsed_time(end));wall.append((time.perf_counter()-before)*1000)
        else:
            before=time.perf_counter();call();values.append((time.perf_counter()-before)*1000);wall.append(values[-1])
    return dict(gpu_or_cpu_ms={f'p{p}':float(np.percentile(values,p)) for p in (50,90,95)},
                wall_ms={f'p{p}':float(np.percentile(wall,p)) for p in (50,90,95)})


@torch.inference_mode()
def benchmark(store, enc, heads, model, removed, device,cfg,index):
    # Collection has exited: benchmark a copy on the actual drafter device.
    benchmark_device=torch.device(cfg.get('collect_device',str(device))) if device.type=='cuda' else device
    enc=copy.deepcopy(enc).to(benchmark_device).eval()
    heads=copy.deepcopy(heads).to(benchmark_device).eval()
    model=copy.deepcopy(model).to(benchmark_device).eval()
    device=benchmark_device
    x=tensors([store.features(index,removed)],device);z=enc(x)
    def rollout():
        imagined=enc(x)
        for _ in range(3):
            child=model(imagined);heads(child,imagined);imagined=child
    result=dict(device=str(device),batch_size=1,canvas=x.shape[1],dtype=str(x.dtype),
        encoder=timed(lambda:enc(x),device,cfg['benchmark_warmup'],cfg['benchmark_repetitions']),
        transition=timed(lambda:model(z),device,cfg['benchmark_warmup'],cfg['benchmark_repetitions']),
        heads=timed(lambda:heads(z,z),device,cfg['benchmark_warmup'],cfg['benchmark_repetitions']),
        complete_horizon3=timed(rollout,device,cfg['benchmark_warmup'],cfg['benchmark_repetitions']),
        note='Prepared observation tensors only: excludes feature preparation, logging and the chosen real drafter action. No end-to-end decoding speedup claimed.')
    native=[item for item in store.manifest.get('native_benchmarks',[])
            if item.get('device')==str(device) and item.get('native_ms_p50',0)>0]
    if native:
        native_p50=float(np.median([item['native_ms_p50'] for item in native]))
        result['same_device_native_comparison']=dict(
            native_forward_median_p50_ms=native_p50,
            native_dtype=native[0].get('dtype'),
            native_state_ids=[item.get('uid') for item in native],
            horizon3_core_over_one_native_forward=result['complete_horizon3']['gpu_or_cpu_ms']['p50']/native_p50,
            note='Simulator core only, not total planning overhead; native uses its collected dtype.')
    return result


DEFAULTS=dict(num_questions=100,latent_dim=128,batch_size=32,encoder_updates=400,
              updates=600,eval_every=100,seeds=[42,43,44],horizon=3,
              learning_milestones=[20,40,70],benchmark_warmup=20,
              benchmark_repetitions=100,train_device='cuda:1',evaluation_paths=512)


def train_study(config,output):
    cfg={**DEFAULTS,**config};output=Path(output);store=StateStore(output)
    audit_data(store)
    split=store.manifest['split']
    train_questions=split['train'];val_questions=split['validation'];test_questions=split['test']
    if not all((train_questions,val_questions,test_questions)):
        raise ValueError('All prompt-disjoint splits must be nonempty')
    device=torch.device(cfg['train_device'])
    train_states=[i for i,s in enumerate(store.states) if s['question_id'] in set(train_questions)]
    val_states=[i for i,s in enumerate(store.states) if s['question_id'] in set(val_questions)]
    test_states=[i for i,s in enumerate(store.states) if s['question_id'] in set(test_questions)]
    train_paths=store.paths(train_questions,3);val_paths=store.paths(val_questions,3);test_paths=store.paths(test_questions,3)
    if not train_paths or not val_paths or not test_paths:
        raise ValueError('Refine edges missing in a split; collect more prompts and inspect length histogram')
    variants=[('linear_h1','linear',1,()),('mlp_h1','mlp',1,()),
              ('transformer_h1','transformer',1,()),('transformer_h3','transformer',3,()),
              *[(f'no_{g}','transformer',3,(g,)) for g in store.slices]]
    if cfg.get('variants'):
        names=set(cfg['variants']);variants=[v for v in variants if v[0] in names]
    results=[];curves=[];checkpoint_dir=output/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        store.fit_normalization(train_states)
        encoders={}
        for name,kind,horizon,removed in variants:
            key=removed
            if key not in encoders:
                encoders[key]=fit_encoder(store,train_states,val_states,cfg,seed,removed,device)
            enc,heads,encoding=encoders[key]
            model,heads,training=fit_transition(store,train_paths,val_paths,enc,heads,cfg,seed,
                kind,horizon,removed,device,output,name)
            evaluations={mode:evaluate(store,test_paths,enc,heads,model,removed,device,3,
                teacher=mode=='teacher_forced',oracle=mode=='oracle_next_latent',
                persistence=mode=='persistence',native_heuristic=mode=='current_confidence_rule',max_paths=None)
                for mode in ('free_running','teacher_forced','oracle_next_latent','persistence','current_confidence_rule')}
            current=evaluate_current(store,test_states,enc,heads,removed,device)
            row=dict(variant=name,seed=seed,removed=list(removed),encoder=encoding,current_reconstruction=current,
                training=training,evaluation=evaluations,
                parameters=sum(p.numel() for p in enc.parameters())+sum(p.numel() for p in model.parameters())+
                           sum(p.numel() for p in heads.parameters()))
            if name=='transformer_h3':
                row['timing']=benchmark(store,enc,heads,model,removed,device,cfg,test_paths[0][0])
                torch.save(dict(encoder=enc.state_dict(),heads=heads.state_dict(),transition=model.state_dict(),
                    normalization_mean=store.mean,normalization_std=store.std,input_dim=store.input_dim,
                    config=cfg,feature_slices={k:[v.start,v.stop] for k,v in store.slices.items()}),
                    checkpoint_dir/f'transformer_h3_seed{seed}.pt')
                curves.append(dict(train_questions=len(train_questions),seed=seed,
                                    evaluation=evaluations['free_running']))
            results.append(row)
            write_json(output/'results.json',results)
            print(f'[evaluate] {name} seed={seed}: {evaluations["free_running"].get("1",{}).get("macro",{})}',flush=True)
            del model
        if not cfg.get('variants') or 'raw_direct' in cfg['variants']:
            enc,heads,_=encoders.get((),next(iter(encoders.values())))
            model,heads,training=fit_transition(store,train_paths,val_paths,enc,heads,cfg,seed,'transformer',
                1,(),device,output,'raw_direct',direct=True)
            results.append(dict(variant='raw_direct',seed=seed,training=training,
                evaluation={'free_running':evaluate(store,test_paths,enc,heads,model,(),device,1,direct=True)}))
            write_json(output/'results.json',results)
        # Independent train-subset curves: refit normalization AND encoder for each size.
        for size in cfg['learning_milestones']:
            if size>=len(train_questions) or size<1: continue
            subset=train_questions[:size];subset_states=[i for i,s in enumerate(store.states) if s['question_id'] in set(subset)]
            subset_paths=store.paths(subset,3)
            if not subset_paths: continue
            store.fit_normalization(subset_states)
            enc,heads,_=fit_encoder(store,subset_states,val_states,cfg,seed,(),device)
            model,heads,_=fit_transition(store,subset_paths,val_paths,enc,heads,cfg,seed,'transformer',3,(),
                device,output,f'learning_{size}')
            curves.append(dict(train_questions=size,seed=seed,
                evaluation=evaluate(store,test_paths,enc,heads,model,(),device,3)))
            write_json(output/'learning_curve.json',curves)
    write_json(output/'learning_curve.json',curves)
    write_json(output/'feature_schema.json',dict(groups={k:[v.start,v.stop] for k,v in store.slices.items()},
        encoder='train-only normalized behavioral autoencoder, frozen before dynamics',
        token_embeddings='native frozen drafter embeddings projected by seed-fixed linear map, not identity codes',
        observation_stage='post_native_commit, same_forward_pre_commit_hidden_and_logits',
        input_future_information=False,canvas='full physical block, native active small span indicated'))
    diagnostics=diagnose(results,store)
    write_json(output/'diagnosis.json',diagnostics)
    render_report(output,results,curves,store,diagnostics)
    return dict(training_jobs=len(results),questions=len(train_questions)+len(val_questions)+len(test_questions),
                split_counts={k:len(v) for k,v in split.items()},states=len(store.states),
                refine_edges=len(store.next),diagnosis=diagnostics)


def audit_data(store):
    split=store.manifest['split'];owners={}
    for name,ids in split.items():
        for qid in ids:
            if qid in owners:raise ValueError('Prompt appears in multiple data splits')
            owners[qid]=name
    uids=set();lengths=[];mask_counts=[];forward_times=[];capture_times=[]
    for index,state in enumerate(store.states):
        if state['uid'] in uids:raise ValueError('Duplicated state UID')
        uids.add(state['uid'])
        if state['question_id'] not in owners:raise ValueError('State has no prompt split owner')
        a=store.arrays(index)
        for field in ('hidden','candidate_emb','token_emb','confidence','entropy','margin'):
            if not np.isfinite(a[field]).all():raise ValueError(f'Nonfinite observation: {state["uid"]}/{field}')
        if len(a['mask'])!=32 or a['eligible'].sum()>8:raise ValueError('Physical canvas/native eligible span changed')
        if index in store.previous:
            previous=store.arrays(store.previous[index])
            if not np.array_equal(a['eligible'],previous['eligible']):raise ValueError('Edge changed native active span')
            if np.any(a['mask'] & ~previous['mask']):raise ValueError('Committed token was remasked')
            if 'tokens' in a and not np.array_equal(a['tokens'][~previous['mask']],previous['tokens'][~previous['mask']]):
                raise ValueError('Committed native token changed')
        mask_counts.append(int((a['mask'] & a['eligible']).sum()))
        if 'forward_ms' in a:forward_times.append(float(a['forward_ms']))
        if 'capture_ms' in a:capture_times.append(float(a['capture_ms']))
    for indices in store.groups.values():
        steps=[store.states[i]['step'] for i in indices]
        if steps!=list(range(len(steps))):raise ValueError('Group has missing native refinement indices')
        lengths.append(len(indices))
    report=dict(passed=True,states=len(uids),groups=len(lengths),refine_edges=len(store.next),
        split_counts={k:len(v) for k,v in split.items()},
        natural_trajectory_state_count_histogram={str(i):lengths.count(i) for i in sorted(set(lengths))},
        eligible_masks_histogram={str(i):mask_counts.count(i) for i in sorted(set(mask_counts))},
        horizon_support={str(h):len([p for p in store.paths(list(owners),h) if len(p)>h]) for h in (1,2,3)},
        observed_forward_ms={f'p{p}':float(np.percentile(forward_times,p)) for p in (50,90,95)} if forward_times else {},
        capture_overhead_ms={f'p{p}':float(np.percentile(capture_times,p)) for p in (50,90,95)} if capture_times else {},
        note='Instrumented forward latency is distinct from separate logging-disabled native benchmarks.')
    write_json(store.output/'data_audit.json',report)


@torch.inference_mode()
def evaluate_current(store,ids,enc,heads,removed,device):
    values=[]
    for offset in range(0,len(ids),32):
        use=ids[offset:offset+32];x=tensors([store.features(i,removed) for i in use],device)
        z=enc(x);pred=heads(z);truth=pack_truth(store,use,device)
        eligible=tensors([store.arrays(i)['eligible'] for i in use],device).bool()
        values.append(dict(current_mask_brier=float(((pred['mask_logits'].sigmoid()[eligible]-truth['mask'][eligible])**2).mean()),
            current_confidence_mae=float((pred['confidence'][eligible]-truth['confidence'][eligible]).abs().mean()),
            latent_std=float(z.std(0,unbiased=False).mean()),
            reconstruction_mse=float((enc.reconstruct(z)-x).square().mean())))
    return {k:float(np.mean([r[k] for r in values])) for k in values[0]} if values else {}


def diagnose(results,store):
    report=[]
    for row in results:
        if row['variant']!='transformer_h3': continue
        ev=row['evaluation'];one=ev['free_running'].get('1',{}).get('macro',{})
        oracle=ev['oracle_next_latent'].get('1',{}).get('macro',{})
        persistence=ev['persistence'].get('1',{}).get('macro',{})
        if one.get('commit_brier') is not None:
            if one['commit_brier']>=persistence.get('commit_brier',1.):
                report.append(dict(seed=row['seed'],priority='high',failure='not_better_than_persistence',
                    evidence=dict(model=one['commit_brier'],persistence=persistence.get('commit_brier')),
                    next_experiment='Inspect change-only cohorts, normalization and additional independent prompts.'))
            if oracle.get('commit_brier',1.)+.02<one['commit_brier']:
                report.append(dict(seed=row['seed'],priority='high',failure='transition_gap_to_oracle_latent',
                    evidence=dict(model=one['commit_brier'],oracle=oracle.get('commit_brier')),
                    next_experiment='Test transition capacity and history; the decoder can read real next state better.'))
        three=ev['free_running'].get('3',{})
        if three.get('paths',0)<100:
            report.append(dict(seed=row['seed'],priority='medium',failure='limited_horizon3_support',
                evidence=dict(paths=three.get('paths',0)),
                next_experiment='Collect more prompts without changing native threshold or fabricating terminal padding.'))
    if not report:
        report.append(dict(priority='info',failure='no_heuristic_flag',
            next_experiment='Inspect prompt-level uncertainty and raw-direct comparison before declaring feasibility.'))
    return dict(items=report,note='Heuristic hypotheses, not proven root causes; no acceptance or decoding claim.')


def render_report(output,results,curves,store,diagnostics):
    table=[]
    for row in results:
        for horizon,report in row['evaluation']['free_running'].items():
            table.append(dict(variant=row['variant'],seed=row['seed'],horizon=horizon,
                              paths=report['paths'],**report['macro']))
    if table:
        with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=['# Drafter-only native refinement study','',
          'No verifier was loaded. Terminal agreement, if inspected, is internal convergence only.',
          'Collection caps control the number of generated tokens, not refinement inside a small block.',
          'Held-out test prompts are excluded from encoder fitting, normalization and checkpoint selection.',
          'Each horizon counts one new native denoising forward within the same active small block.',
          'Raw mask probabilities and projected irreversible mask states are reported separately.',
          'Candidate-content cosine measures semantic embedding recovery, not exact token generation.',
          'Latent error is comparable within an encoder only; use behavior metrics across ablations.',
          '', 'See results.json for per-question metrics, validation training logs and oracle/teacher/persistence controls.',
          'See learning_curve.json for independently retrained train-size curves and diagnosis.json for hypotheses.',
          '', '|Variant|Seed|H|Paths|Commit Brier|Confidence MAE|Count MAE|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        text.append(f'|{r["variant"]}|{r["seed"]}|{r["horizon"]}|{r["paths"]}|{r["commit_brier"]:.4f}|{r["confidence_mae"]:.4f}|{r["unmask_count_mae"]:.3f}|')
    (output/'READ_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name in ('transformer_h1','transformer_h3','mlp_h1','linear_h1'):
            rows=[r for r in table if r['variant']==name]
            if not rows:continue
            hs=sorted({int(r['horizon']) for r in rows})
            vals=[np.mean([r['commit_brier'] for r in rows if int(r['horizon'])==h]) for h in hs]
            axes[0].plot(hs,vals,'o-',label=name)
        axes[0].set(xlabel='Free-running native forwards',ylabel='Prompt-macro commit Brier (lower is better)');axes[0].legend()
        sizes=sorted({r['train_questions'] for r in curves})
        for horizon in ('1','3'):
            values=[];valid_sizes=[]
            for size in sizes:
                vals=[r['evaluation'][horizon]['macro']['commit_brier'] for r in curves
                      if r['train_questions']==size and horizon in r['evaluation']]
                if vals: values.append(float(np.mean(vals)));valid_sizes.append(size)
            if values:axes[1].plot(valid_sizes,values,'o-',label=f'horizon {horizon}')
        axes[1].set(xlabel='Independent training prompts',ylabel='Same held-out commit Brier');axes[1].legend()
        fig.tight_layout();fig.savefig(output/'learning_and_rollout.png',dpi=150);plt.close(fig)
    except ImportError:
        pass
