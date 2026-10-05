"""Offline, restartable factorized D/V feasibility experiment. No LLM forwards.

All fitted statistics and vocabularies use training questions only. Validation
selects checkpoints and operating points; test is never used by training. D is
trained only on native D targets, V independently on saved verifier teachers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from factorized_wm_data import (load_dataset, enumerate_paths, split_questions,
                               feature_registry, assert_input_features)
from factorized_wm_models import (Layout, State, Preprocessor, pack, TokenDecoder,
    NativeTopKDecoder, StructuredDynamics, LinearDynamics, TabularVerifier,
    VerifierTransformer, DirectOutcome, mean_extension_prior, materialize,
    drafter_loss, verifier_loss, transition_history)
from factorized_wm_metrics import (write_json, write_jsonl, write_csv, verifier_report,
    dynamics_report, delta_report, threshold_sweep, select_operating_points,
    paired_question_bootstrap)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def progress(data):
    memory = {str(i): round(torch.cuda.memory_allocated(i)/2**30, 3)
              for i in range(torch.cuda.device_count())}
    print('[feasibility] ' + json.dumps(dict(data, gpu_allocated_GiB=memory)), flush=True)


def package(output):
    """Atomic ZIP next to output, never include input data or model downloads."""
    output = Path(output); final = output.with_suffix('.zip'); temp = final.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix != '.tmp' and '_cache' not in p.parts:
                archive.write(p, output.name + '/' + p.relative_to(output).as_posix())
    temp.replace(final)
    progress(dict(stage='archive', file=str(final), MiB=round(final.stat().st_size/2**20, 1)))
    return final


def save_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def balanced(items, question_fn, seed):
    """Question-uniform, then action/length-uniform, then record-uniform."""
    groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        q, cohort = question_fn(item); groups[q][cohort].append(item)
    if not groups: raise ValueError('No examples for this stage')
    rng = random.Random(seed); questions = sorted(groups)
    def sample(n):
        result = []
        for _ in range(n):
            cohorts = groups[rng.choice(questions)]
            result.append(rng.choice(cohorts[rng.choice(list(cohorts))]))
        return result
    return sample


def rollout(model, state, sequences):
    for k in range(len(sequences[0])):
        actions = torch.tensor([a[k] for a in sequences], device=state.x.device)
        state = model(state, actions)
    return state


def slice_state(state,start,count):
    return State(*(v[start:start+count] for v in (state.x,state.lengths,state.context,state.prefix)))


@torch.no_grad()
def v_rows(model, rows, device, batch=16, states=None, sequences=None):
    """Teachers are joined after prediction and are never passed to forward."""
    if isinstance(model, torch.nn.Module): model.eval()
    result = []
    for start in range(0, len(rows), batch):
        group = rows[start:start+batch]
        state = pack(group, device) if states is None else states(start, group)
        heads = (model(state) if sequences is None else
                 model(state, torch.tensor(sequences[start:start+len(group)], device=device)))
        q = heads['hazard'].sigmoid().clamp(1e-6, 1-1e-6).cpu()
        survival = q.cumprod(-1)
        for i, row in enumerate(group):
            n = row['length']; hazards = q[i, :n]
            pmf = torch.cat([1-hazards[:1], survival[i, :n-1]*(1-hazards[1:]), survival[i, n-1:n]])
            teacher = row['teacher']
            def values(name, sigmoid=False):
                value = heads.get(name)
                if value is None: return None
                value = value[i, :n].detach().cpu()
                return (value.sigmoid() if sigmoid else value).tolist()
            gates=getattr(model,'last_gates',None)
            result.append(dict(question=row['question'], uid=row['uid'], length=n,
                accepted=row['accepted'], expected_yield=float(survival[i, :n].sum()),
                mode=int(pmf.argmax()), hazards=hazards.tolist(), tf_probs=values('tf', True),
                probability_pred=values('probability'), margin_pred=values('margin'),
                tf_truth=None if teacher is None else teacher[:, 2].tolist(),
                probability_truth=None if teacher is None else teacher[:, 1].tolist(),
                margin_truth=None if teacher is None else teacher[:, 0].tolist(),
                hidden_gate_layer_mean=None if gates is None else gates[i,:n].mean(0).cpu().tolist()))
    return result


def v_score(rows):
    errors = defaultdict(list)
    for r in rows:
        if r['accepted'] is not None:
            errors[r['question']].append(abs(r['expected_yield']-r['accepted']))
    return float(np.mean([np.mean(v) for v in errors.values()])) if errors else float('inf')


def train_v(model, train, val, cfg, device, folder, direct_paths=None, prepared=None):
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=.02)
    direct = direct_paths is not None
    items = train if not direct else direct_paths
    if direct:
        items=[p for p in items if prepared[p[0][-1]]['accepted'] is not None or prepared[p[0][-1]]['teacher'] is not None]
    else: items=[r for r in items if r['accepted'] is not None or r['teacher'] is not None]
    if not any((prepared[p[0][-1]]['accepted'] is not None if direct else p['accepted'] is not None) for p in items):
        raise ValueError('No accepted-length supervision for the required hazard head')
    sample = balanced(items, (lambda r: (r['question'], r['length'])) if not direct else
        lambda p: (prepared[p[0][0]]['question'], (p[1], prepared[p[0][-1]]['length'])), cfg['seed'])
    samplers={h:balanced([p for p in items if len(p[1])==h],
        lambda p:(prepared[p[0][0]]['question'],(p[1],prepared[p[0][-1]]['length'])),cfg['seed']+h)
        for h in (1,2,3) if any(len(p[1])==h for p in items)} if direct else {}
    budget = cfg['direct_updates'] if direct else cfg['verifier_updates']
    best = float('inf'); bad = 0; curve = []
    for step in range(1, budget+1):
        model.train(); group = sample(cfg['batch_size'])
        if direct:
            actions = group[0][1]
            # Homogeneous horizon for tensor action descriptors, no truth reset.
            group = samplers[len(actions)](cfg['batch_size'])
            rows = [prepared[p[0][-1]] for p in group]
            heads = model(pack([prepared[p[0][0]] for p in group], device),
                          torch.tensor([p[1] for p in group], device=device))
        else:
            rows = group; heads = model(pack(rows, device))
        loss = verifier_loss(heads, rows, device)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite V loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step % cfg['eval_every'] == 0 or step == budget:
            if direct:
                predictions = []
                for horizon in (1, 2, 3):
                    vv = [p for p in val if len(p[1]) == horizon]
                    predictions += v_rows(model, [prepared[p[0][-1]] for p in vv], device,
                        states=lambda start, g: pack([prepared[p[0][0]] for p in vv[start:start+len(g)]], device),
                        sequences=[p[1] for p in vv])
            else: predictions = v_rows(model, val, device)
            score = v_score(predictions)
            curve.append(dict(update=step, train_loss=float(loss.detach()), val_macro_K_mae=score))
            progress(dict(stage=folder.name, **curve[-1]))
            if score < best:
                best = score; bad = 0; save_torch(folder/'best.pt', dict(weights=cpu_weights(model), update=step))
            else: bad += 1
            save_torch(folder/'last.pt', dict(weights=cpu_weights(model), optimizer=optimizer.state_dict(), update=step))
            write_json(folder/'training_curve.json', curve)
            if bad >= cfg['early_patience']: break
    model.load_state_dict(torch.load(folder/'best.pt', weights_only=True)['weights']); model.eval()
    write_json(folder/'training_complete.json',dict(updates=step,selection='validation_only'))
    return curve


@torch.no_grad()
def d_validation(model, prepared, paths, device):
    model.eval(); losses = []
    for start in range(0, len(paths), 16):
        group = paths[start:start+16]; horizon = len(group[0][1])
        # Caller passes homogeneous-horizon paths.
        if any(len(p[1]) != horizon for p in group): raise ValueError('Mixed validation horizon')
        source = pack([prepared[p[0][0]] for p in group], device)
        pred = rollout(model, source, [p[1] for p in group])
        last_source = pack([prepared[p[0][-2]] for p in group], device)
        loss, _ = drafter_loss(pred, [prepared[p[0][-1]] for p in group], last_source,
            torch.tensor([p[1][-1] for p in group], device=device), model.decoder, model.layout)
        losses.append((float(loss), len(group)))
    return sum(a*b for a, b in losses)/sum(b for _, b in losses) if losses else float('inf')


def train_d(model, prepared, train_paths, val_paths, cfg, device, folder):
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.02)
    curve = []; budgets = [cfg['h1_min_updates'], max(1, (cfg['drafter_updates']-cfg['h1_min_updates'])//2)]
    budgets += [max(1, cfg['drafter_updates']-sum(budgets))]
    update = 0; curriculum = []
    for horizon, budget in enumerate(budgets, 1):
        pool = [p for p in train_paths if len(p[1]) == horizon]
        vv = [p for p in val_paths if len(p[1]) == horizon]
        if not pool or not vv:
            curriculum.append(dict(horizon=horizon, status='insufficient_paths')); continue
        sample = balanced(pool, lambda p: (prepared[p[0][0]]['question'],
            (p[1], prepared[p[0][0]]['length'])), cfg['seed']+horizon)
        best = float('inf'); bad = 0; stage_path = folder/f'best_H{horizon}.pt'
        for step in range(1, budget+1):
            update += 1; model.train(); group = sample(cfg['batch_size'])
            source = pack([prepared[p[0][0]] for p in group], device)
            imagined = source; loss = imagined.x.sum()*0
            for depth in range(horizon):
                a = torch.tensor([p[1][depth] for p in group], device=device)
                parent = imagined; imagined = model(imagined, a)
                part, terms = drafter_loss(imagined, [prepared[p[0][depth+1]] for p in group],
                                          parent, a, model.decoder, model.layout)
                loss = loss + part/horizon
            if not torch.isfinite(loss): raise RuntimeError('Non-finite D loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            if step % cfg['eval_every'] == 0 or step == budget:
                score = d_validation(model, prepared, vv, device)
                curve.append(dict(update=update, horizon=horizon, train_loss=float(loss.detach()),
                                  val_D_only_loss=score, **terms))
                progress(dict(stage=folder.name, **curve[-1]))
                if score < best:
                    best=score; bad=0; save_torch(stage_path, dict(weights=cpu_weights(model), update=update))
                else: bad += 1
                save_torch(folder/'last.pt', dict(weights=cpu_weights(model), optimizer=optimizer.state_dict(),
                    horizon=horizon, step=step, update=update))
                write_json(folder/'training_curve.json', curve)
                # H1 minimum is a hard floor. Later horizons can stop separately.
                if bad >= cfg['early_patience'] and horizon > 1: break
        model.load_state_dict(torch.load(stage_path, weights_only=True)['weights'])
        curriculum.append(dict(horizon=horizon, updates=step, best_D_only_loss=best,
            stabilized=bad >= cfg['early_patience'], advance_reason='plateau' if bad >= cfg['early_patience'] else 'stage_budget'))
    save_torch(folder/'best.pt', dict(weights=cpu_weights(model), curriculum=curriculum))
    write_json(folder/'curriculum.json', curriculum); model.eval(); return curve


def train_native_decoder(prep, rows, cfg, device, folder):
    model = NativeTopKDecoder(prep).to(device)
    file = folder/'native_decoder.pt'
    if file.exists(): model.load_state_dict(torch.load(file, weights_only=True)['weights'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        rng = random.Random(cfg['seed']); eligible = []
        for r in rows:
            s = r['x'][:, prep.layout.structure]
            valid = torch.where((s[:,4]>0)&(s[:,6]<1e-5)&(r['topk_classes']>0).all(-1))[0]
            if len(valid): eligible.append((r, valid.tolist()))
        if not eligible: raise ValueError('No native fresh in-vocabulary top-k teachers')
        curve = []
        for step in range(1, cfg['projection_updates']+1):
            picks = [rng.choice(eligible) for _ in range(64)]
            picks = [(r, rng.choice(indices)) for r, indices in picks]
            x = torch.stack([r['x'][i] for r,i in picks]).to(device)
            target_ids = torch.stack([r['topk_classes'][i] for r,i in picks]).to(device)
            p = torch.stack([r['topk_gaps'][i] for r,i in picks]).to(device).softmax(-1)
            logp = model.logits(x).log_softmax(-1)
            loss = -(p*logp.gather(-1, target_ids)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % cfg['eval_every'] == 0 or step == cfg['projection_updates']:
                curve.append(dict(step=step, native_conditional_teacher_cross_entropy=float(loss.detach())))
                progress(dict(stage='D_only_native_decoder', **curve[-1]))
        save_torch(file, dict(weights=cpu_weights(model))); write_json(folder/'native_decoder_curve.json', curve)
    return model.eval().requires_grad_(False)


def cosine_summary(a, b):
    return float(F.cosine_similarity(a, b, dim=-1).mean()) if len(a) else None


def linear_cka(a, b):
    if len(a) < 2: return None
    a=a-a.mean(0); b=b-b.mean(0)
    denominator = (a.T@a).square().sum().sqrt()*(b.T@b).square().sum().sqrt()
    return float((a.T@b).square().sum()/denominator) if denominator > 0 else None


@torch.no_grad()
def region_stats(pred, real, mask, decoder, native_decoder, layout, prep):
    n = real['length']; pp=pred[:n][mask].cpu(); yy=real['x'][mask].cpu()
    if not len(pp): return None
    tokens=decoder.ids(pp.to(decoder.codes.device),1).flatten().cpu(); target=real['raw_stop_ids'][mask]
    correct=tokens==target
    s=yy[:,layout.structure]; hh=(s[:,3]>0)&(s[:,5]<1e-5); ll=(s[:,4]>0)&(s[:,6]<1e-5)
    result=dict(correct_tokens=int(correct.sum()), token_count=len(correct), token_agreement=float(correct.float().mean()),
        mse_hidden=float((pp[hh,:layout.h]-yy[hh,:layout.h]).square().mean()) if hh.any() else None,
        cosine_hidden=cosine_summary(pp[hh,:layout.h],yy[hh,:layout.h]),
        linear_CKA_hidden=linear_cka(pp[hh,:layout.h],yy[hh,:layout.h]),
        mse_surface=float((pp[ll,layout.surface]-yy[ll,layout.surface]).square().mean()) if ll.any() else None,
        mask_brier=float((pp[:,layout.structure][:,0]-s[:,0]).square().mean()),
        token_in_vocabulary_coverage=float((real['token_targets'][mask]>0).float().mean()),
        fresh_hidden_tokens=int(hh.sum()),fresh_logits_tokens=int(ll.sum()))
    if native_decoder is not None and ll.any():
        nd_device=native_decoder.codes.device
        ids=native_decoder.ids(pp[ll].to(nd_device),10).cpu(); truth=real['raw_topk_ids'][mask][ll]
        gaps=real['topk_gaps'][mask][ll]; classes=real['topk_classes'][mask][ll]
        result['native_top1_agreement']=float((ids[:,0]==truth[:,0]).float().mean())
        for k in (5,10):
            scores=[len(set(a[:k].tolist())&set(b[:k].tolist()))/k for a,b in zip(ids,truth)]
            result[f'top{k}_recall']=float(np.mean(scores))
            result[f'jaccard{k}']=float(np.mean([len(set(a[:k].tolist())&set(b[:k].tolist()))/len(set(a[:k].tolist())|set(b[:k].tolist())) for a,b in zip(ids,truth)]))
        valid=(classes>0).all(-1)
        result['conditional_topk_coverage']=float(valid.float().mean())
        if valid.any():
            lp=native_decoder.logits(pp[ll][valid].to(nd_device)).gather(-1,classes[valid].to(nd_device)).log_softmax(-1).cpu()
            tq=gaps[valid].softmax(-1); lq=gaps[valid].log_softmax(-1); pq=lp.exp(); mix=(tq+pq)/2
            result['conditional_topk_kl']=float((tq*(lq-lp)).sum(-1).mean())
            result['conditional_topk_js']=float((.5*tq*(lq-mix.log())+.5*pq*(lp-mix.log())).sum(-1).mean())
            result['top1_confidence_error']=float((pq[:,0]-tq[:,0]).abs().mean())
    return result


@torch.no_grad()
def evaluate_d(model, prep, prepared, paths, device, native_decoder, mode='free'):
    model.eval(); model.to(device); decoder=model.decoder.to(device)
    records=[]
    for horizon in (1,2,3):
        pool=[p for p in paths if len(p[1])==horizon]
        for start in range(0,len(pool),16):
            group=pool[start:start+16]
            source=pack([prepared[p[0][0]] for p in group],device)
            if mode=='teacher_input':
                source=pack([prepared[p[0][-2]] for p in group],device)
                pred=model(source,torch.tensor([p[1][-1] for p in group],device=device))
            else: pred=rollout(model,source,[p[1] for p in group])
            for i,(nodes,actions) in enumerate(group):
                parent=prepared[nodes[-2]]; real=prepared[nodes[-1]]; n=real['length']
                pp=pred.x[i,:n]; yy=real['x'].to(device)
                # Match device but keep raw target IDs as CPU for exact comparisons.
                target=dict(real,x=yy)
                pos=torch.arange(n,device=device); begin=round(float(parent['context'][2])*64)
                masks=dict(whole=pos>=0, immutable=pos<(parent['length'] if actions[-1] else begin),
                    frontier=pos>=(parent['length'] if actions[-1] else begin), new_block=pos>=parent['length'])
                regions={key:region_stats(pp,target,m.cpu(),decoder,native_decoder,prep.layout,prep)
                         for key,m in masks.items()}
                pred_tokens=decoder.ids(pp,1).flatten().cpu().tolist()
                changed = not torch.equal(parent['raw_stop_ids'],real['raw_stop_ids'][:parent['length']]) or n!=parent['length']
                gate=getattr(model,'last_gates',None)
                records.append(dict(question=real['question'],parent=nodes[0],child=nodes[-1],horizon=horizon,
                    actions=''.join('E' if a else 'R' for a in actions),length=n,mode=mode,
                    change='changed' if changed else 'unchanged',regions=regions,
                    pred_tokens=pred_tokens, real_tokens=real['raw_stop_ids'].tolist(),
                    hidden_gate_layer_mean=None if gate is None else gate[i,:parent['length']].mean(0).cpu().tolist()))
    return records


class PriorDynamics(torch.nn.Module):
    def __init__(self,prep,prior):
        super().__init__(); self.layout=prep.layout; self.decoder=TokenDecoder(prep); self.register_buffer('prior',prior)
        self.register_buffer('history_mean',prep.mean[-4:].clone());self.register_buffer('history_std',prep.std[-4:].clone())
    def forward(self,state,actions):
        result,new=materialize(state,actions,self.layout,self.prior)
        result.x=transition_history(result.x,state,self.layout,self.history_mean,self.history_std,self.decoder,new)
        return result


def d_score(report):
    # Select D checkpoint independently of ANY verifier outcome.
    return report['by_horizon'].get('H1',{}).get('regions',{}).get('frontier',{}).get('mse_hidden')


def release():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def response_pairs(predictions, prepared, paths, split):
    lookup={r['uid']:r for r in predictions}; result=[]
    for nodes,actions in paths:
        if len(actions)!=1: continue
        a,b=prepared[nodes[0]],prepared[nodes[1]]
        if a['accepted'] is None or b['accepted'] is None: continue
        if nodes[0] not in lookup or nodes[1] not in lookup: continue
        delta=b['accepted']-a['accepted']; prediction=lookup[nodes[1]]['expected_yield']-lookup[nodes[0]]['expected_yield']
        result.append(dict(question=a['question'],parent=nodes[0],child=nodes[1],action='E' if actions[0] else 'R',
            delta_true=delta,delta_pred=prediction,score=prediction,real_useful=delta>0,split=split))
    return result


@torch.no_grad()
def composition(d, v, direct, prior, prepared, paths, cfg, ddevice, vdevice, split):
    """Both learned modules frozen; final child labels only join AFTER forward."""
    d.eval(); v.eval(); direct.eval(); d.to(ddevice); prior.to(ddevice); v.to(vdevice); direct.to(vdevice)
    allrows=[]; decoder=d.decoder.to(ddevice)
    parent_ids=sorted({p[0][0] for p in paths})
    base={r['uid']:r for r in v_rows(v,[prepared[u] for u in parent_ids],vdevice)}
    for horizon in (1,2,3):
        pool=[p for p in paths if len(p[1])==horizon]
        for start in range(0,len(pool),cfg['batch_size']):
            group=pool[start:start+cfg['batch_size']]; sequences=[p[1] for p in group]
            source=pack([prepared[p[0][0]] for p in group],ddevice)
            imagined=rollout(d,source,sequences); copied=rollout(prior,source,sequences)
            targets=[prepared[p[0][-1]] for p in group]
            truth=v_rows(v,targets,vdevice)
            learn=v_rows(v,targets,vdevice,states=lambda start,g:slice_state(imagined,start,len(g)).to(vdevice))
            prior_pred=v_rows(v,targets,vdevice,states=lambda start,g:slice_state(copied,start,len(g)).to(vdevice))
            direct_pred=v_rows(direct,targets,vdevice,states=lambda start,g:slice_state(source,start,len(g)).to(vdevice),sequences=sequences)
            token_pred=decoder.ids(imagined.x,1).squeeze(-1).cpu()
            for i, (nodes, actions) in enumerate(group):
                a,b=prepared[nodes[0]],targets[i]; delta=None if a['accepted'] is None or b['accepted'] is None else b['accepted']-a['accepted']
                row=dict(question=a['question'],parent=nodes[0],child=nodes[-1],horizon=horizon,
                    action='E' if actions[-1] else 'R',actions=''.join('E' if x else 'R' for x in actions),
                    length=b['length'],parent_length=a['length'],accepted=b['accepted'],parent_accepted=a['accepted'],
                    real_parent_tokens=a['raw_stop_ids'].tolist(),real_child_tokens=b['raw_stop_ids'].tolist(),
                    pred_child_tokens=token_pred[i,:b['length']].tolist(),real_useful=None if delta is None else delta>0,
                    delta_true=delta,split=split)
                for name, predictions in (('oracleD',truth),('learnedD',learn),('priorD',prior_pred),('direct',direct_pred)):
                    p=predictions[i]; score=p['expected_yield']-base[nodes[0]]['expected_yield']
                    row[name+'_K']=p['expected_yield']; row[name+'_score']=score; row[name+'_verifier']=p
                allrows.append(row)
    return allrows


def composition_report(rows):
    result={}
    for name in ('oracleD','learnedD','priorD','direct'):
        byh={}
        for horizon in (1,2,3):
            subset=[r for r in rows if r['horizon']==horizon]
            delta=[dict(question=r['question'],action=r['action'],delta_true=r['delta_true'],delta_pred=r[name+'_score'])
                   for r in subset if r['delta_true'] is not None]
            byh[f'H{horizon}']=dict(verifier=verifier_report([r[name+'_verifier'] for r in subset]),
                                    response_delta=delta_report(delta))
        result[name]=byh
    return result


def apply_operating_points(validation, test, name, action):
    val=[dict(score=r[name+'_score'],real_useful=r['real_useful'],split='val')
         for r in validation if r['horizon']==1 and r['action']==action and r['real_useful'] is not None]
    test=[dict(score=r[name+'_score'],real_useful=r['real_useful'])
          for r in test if r['horizon']==1 and r['action']==action and r['real_useful'] is not None]
    chosen=select_operating_points(val); sweep=threshold_sweep(test); applied={}
    for key in ('high_precision','balanced','high_recall'):
        entry=chosen[key]
        if entry is None: applied[key]=None; continue
        threshold=float(entry['threshold'])
        tp=sum(r['score']>=threshold and r['real_useful'] for r in test)
        fp=sum(r['score']>=threshold and not r['real_useful'] for r in test)
        positives=sum(r['real_useful'] for r in test)
        applied[key]=dict(threshold=entry['threshold'],selected=tp+fp,actually_useful=tp,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/positives if positives else None,
            test_count=len(test),test_positives=positives)
    return dict(validation_selection=chosen,test_operating_points=applied),sweep


def predictability(prepared, train_paths, test_paths):
    from sklearn.neighbors import NearestNeighbors
    result={'warning':'Representation-conditioned kNN diagnostic, NOT an intrinsic predictability ceiling.'}
    for action in (0,1):
        tt=[p for p in train_paths if p[1]==(action,)]; vv=[p for p in test_paths if p[1]==(action,)]
        rows=[]
        for length in sorted({prepared[p[0][0]]['length'] for p in vv}):
            train=[p for p in tt if prepared[p[0][0]]['length']==length]
            test=[p for p in vv if prepared[p[0][0]]['length']==length]
            if len(train)<5: continue
            def vector(p):
                x=prepared[p[0][0]]; return torch.cat([x['x'][-8:].flatten(),x['x'].mean(0),x['context'],x['prefix']]).numpy()
            x=np.stack([vector(p) for p in train]); mean=x.mean(0); std=x.std(0).clip(.05)
            search=NearestNeighbors(n_neighbors=5).fit((x-mean)/std)
            distances,indices=search.kneighbors((np.stack([vector(p) for p in test])-mean)/std)
            for p,dd,neighbors in zip(test,distances,indices):
                def future(pair):
                    parent,child=prepared[pair[0][0]],prepared[pair[0][-1]]
                    begin=parent['length'] if action else round(float(parent['context'][2])*64)
                    return child['x'][begin:].flatten().numpy()
                futures=[future(train[j]) for j in neighbors]
                deltas=[]
                for j in neighbors:
                    a,b=[prepared[u] for u in train[j][0]]
                    if a['accepted'] is not None and b['accepted'] is not None: deltas.append(b['accepted']-a['accepted'])
                target=future(p)
                rows.append(dict(question=prepared[p[0][0]]['question'],length=length,distance=float(dd.mean()),
                    future_region='new block' if action else 'active frontier',
                    neighbor_future_variance=float(np.var(futures,axis=0).mean()),
                    neighbor_future_MSE=float(np.square(np.mean(futures,axis=0)-target).mean()),
                    neighbor_delta_K_variance=float(np.var(deltas)) if deltas else None))
        result['E' if action else 'R']=rows
    return result


def experiment_plan(cfg, train_count):
    """Deduplicate ablations that coincide with the final medium 32D job."""
    plan=[]; seen=set(); base=cfg['capacities']['medium']
    def add(kind,dim=32,representation='SHT',seed=42,capacity='medium',gate=True,ntrain=None):
        ntrain=train_count if ntrain is None else ntrain
        key=(kind,dim,representation,seed,capacity,gate,ntrain)
        if key in seen:return
        seen.add(key)
        plan.append(dict(kind=kind,dim=dim,representation=representation,seed=seed,capacity=capacity,
            hidden_gate=gate,ntrain=ntrain,name=f'{kind}_h{dim}_{representation}_s{seed}_{capacity}_gate{int(gate)}_q{ntrain}'))
    main_dim=max(cfg['hidden_dims']); seed=cfg['seed']
    for kind in ('D0','D1','V0','V1'): add(kind,main_dim,seed=seed)
    for representation in cfg['representations']:
        for kind in ('D2','V2'):add(kind,main_dim,representation,seed)
    for dim in cfg['hidden_dims']:
        for kind in ('D2','V2'):add(kind,dim,seed=seed)
    for kind in ('D2','V2'):add(kind,main_dim,seed=seed,gate=False)
    for capacity in cfg['capacities']:
        for kind in ('D2','V2'):add(kind,main_dim,seed=seed,capacity=capacity)
    for seed in cfg['seeds']:
        for kind in ('D2','V2'):add(kind,main_dim,seed=seed)
    for n in cfg['scaling_questions']:
        if n>train_count: raise ValueError(f'Scaling {n} exceeds fixed train split {train_count}')
        for kind in ('D2','V2'):add(kind,main_dim,seed=cfg['seed'],ntrain=n)
    reference=next(j for j in plan if j['kind']=='D2' and j['dim']==main_dim and
        j['representation']=='SHT' and j['seed']==cfg['seed'] and j['capacity']=='medium' and j['hidden_gate'] and j['ntrain']==train_count)
    plan.append(dict(reference,action_condition=False,name=reference['name']+'_no_action_embedding'))
    return plan


class Study:
    def __init__(self,dataset,split,cfg,output):
        self.dataset=dataset;self.split=split;self.cfg=cfg;self.output=output
        self.allpaths={part:enumerate_paths(dataset,ids,cfg['horizon']) for part,ids in split.items()}
        self.bad={(x['parent'],x['child']) for x in dataset.audit['immutable_prefix_K_contradictions']}
        self.prep_cache={};self.results={};self.prepared=None

    def train_paths(self,questions):
        # D fitting does NOT filter based on V labels. Behavior fitting can mask
        # demonstrably inconsistent K labels without dropping test paths.
        return [p for p in self.allpaths['train'] if self.dataset.states[p[0][0]].question in set(questions)]

    def preprocessing(self,job):
        questions=self.split['train'][:job['ntrain']];key=(job['dim'],job['seed'],job['ntrain'])
        folder=self.output/'preprocessing'/f'h{key[0]}_s{key[1]}_q{key[2]}';folder.mkdir(parents=True,exist_ok=True)
        if key in self.prep_cache:return self.prep_cache[key]
        file=folder/'frozen_preprocessor.pt'
        if file.exists():prep=Preprocessor.restore(torch.load(file,weights_only=True))
        else:
            prep=Preprocessor.fit(self.dataset.states,questions,job['dim'],self.cfg['projection_updates'],
                                  self.cfg['device_drafter'],job['seed'],progress)
            save_torch(file,prep.save_payload())
            write_json(folder/'fit_questions.json',dict(questions=questions,seed=job['seed'],hidden_dim=job['dim']))
        prepared=prep.prepare(self.dataset.states,self.cfg['device_drafter'])
        paths=self.train_paths(questions);prior=mean_extension_prior(prepared,paths,prep.layout)
        native=train_native_decoder(prep,[r for r in prepared.values() if r['question'] in set(questions)],
                                    dict(self.cfg,seed=job['seed']),self.cfg['device_drafter'],folder)
        # Keep one representation in RAM; frozen checkpoints reconstruct others.
        self.prep_cache.clear();self.prep_cache[key]=(prep,prepared,prior,native)
        return self.prep_cache[key]

    def build(self,job,prep,prior):
        cap=self.cfg['capacities'][job['capacity']]
        if job['kind']=='D0':return LinearDynamics(prep,prior)
        if job['kind'] in ('D1','D2'):
            model=StructuredDynamics(prep,prior,'mlp' if job['kind']=='D1' else 'transformer',
                job['representation'],cap['width'],cap['layers'],job['hidden_gate'])
            if not job.get('action_condition',True):
                with torch.no_grad():model.action.weight.zero_()
                model.action.requires_grad_(False)
            return model
        if job['kind'] in ('V0','V1'):
            return TabularVerifier(prep.layout,'logistic' if job['kind']=='V0' else 'hgb',
                                   job['representation'],job['seed'])
        return VerifierTransformer(prep.layout,job['representation'],cap['width'],cap['layers'],hidden_gate=job['hidden_gate'])

    def execute(self,job,return_model=False):
        folder=self.output/'jobs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
        done=folder/'result.json'
        if done.exists() and not return_model:
            result=json.loads(done.read_text());self.results[job['name']]=result
            progress(dict(stage='resume_skip',job=job['name']));return result
        seed_all(job['seed']);prep,prepared,prior,native=self.preprocessing(job)
        model=self.build(job,prep,prior);questions=self.split['train'][:job['ntrain']]
        paths=self.train_paths(questions);train=[r for r in prepared.values() if r['question'] in set(questions)]
        # Explicit inconsistent K-label mask only in V-training, with audit trail.
        bad_children={b for _,b in self.bad}
        train=[dict(r,accepted=None) if r['uid'] in bad_children else r for r in train]
        val=[r for r in prepared.values() if r['question'] in set(self.split['val'])]
        test=[r for r in prepared.values() if r['question'] in set(self.split['test'])]
        cfg=dict(self.cfg,seed=job['seed']);write_json(folder/'config.json',job)
        isd=job['kind'].startswith('D');device=cfg['device_drafter'] if isd else cfg['device_verifier']
        existing=folder/'best.pt'
        trained=done.exists() or (folder/'training_complete.json').exists() or (job['kind']=='D0' and existing.exists())
        if isinstance(model,torch.nn.Module) and existing.exists() and trained:
            saved=torch.load(existing,weights_only=True);model.load_state_dict(saved['weights'])
            if isinstance(model,LinearDynamics):model.r_weight=saved['r_weight'];model.e_weight=saved['e_weight']
            model.to(device)
        elif isd:
            if isinstance(model,LinearDynamics):
                model.fit(prepared,paths,device,job['seed'])
                save_torch(existing,dict(weights=cpu_weights(model),r_weight=model.r_weight,e_weight=model.e_weight))
            else:
                train_d(model,prepared,paths,self.allpaths['val'],cfg,device,folder)
                write_json(folder/'training_complete.json',dict(selection='D_only_validation',curriculum=[1,2,3]))
        elif isinstance(model,TabularVerifier): model.fit(train)
        else:train_v(model,train,val,cfg,device,folder)
        if done.exists():result=json.loads(done.read_text())
        elif isd:
            reports={}
            for part in ('train','val','test'):
                evaluation=paths if part=='train' else self.allpaths[part]
                records=evaluate_d(model,prep,prepared,evaluation,device,native)
                reports[part]=dynamics_report(records)
                write_jsonl(folder/f'{part}_drafter_predictions.jsonl',records)
                if part=='test':
                    teacher=evaluate_d(model,prep,prepared,evaluation,device,native,'teacher_input')
                    reports['teacher_input_test']=dynamics_report(teacher)
                    write_jsonl(folder/'teacher_input_predictions.jsonl',teacher)
            result=dict(job=job,reports=reports)
        else:
            reports={}
            for part,rows in (('train',train),('val',val),('test',test)):
                predictions=v_rows(model,rows,device);reports[part]=verifier_report(predictions)
                reports[part+'_response_delta']=delta_report(response_pairs(predictions,prepared,self.allpaths[part],part))
                write_jsonl(folder/f'{part}_verifier_predictions.jsonl',predictions)
            result=dict(job=job,reports=reports)
        write_json(done,result);self.results[job['name']]=result
        if not return_model:
            if isinstance(model,torch.nn.Module):model.cpu()
            release();self.summaries();package(self.output)
            return result
        return model,prep,prepared,prior,native,result

    def summaries(self):
        results=list(self.results.values());dim=max(self.cfg['hidden_dims']);ntrain=len(self.split['train'])
        def common(j):return j['seed']==self.cfg['seed'] and j['ntrain']==ntrain
        canonical=[r for r in results if common(r['job']) and r['job'].get('action_condition',True) and r['job']['dim']==dim and
                   r['job']['representation']=='SHT' and r['job']['capacity']=='medium' and r['job']['hidden_gate']]
        groups={kind:[r for r in canonical if r['job']['kind']==kind] for kind in ('D0','D1','D2','V0','V1','V2')}
        write_json(self.output/'drafter_model_comparison.json',{k:groups[k] for k in ('D0','D1','D2')})
        write_json(self.output/'verifier_model_comparison.json',{k:groups[k] for k in ('V0','V1','V2')})
        neural=[r for r in results if r['job']['kind'] in ('D2','V2') and common(r['job']) and r['job'].get('action_condition',True)]
        write_json(self.output/'action_condition_ablation.json',dict(warning='Removes learned action identifier only; deterministic length/commit rules still reveal R/E. Not a causal action-blind proof.',
            results=[r for r in results if not r['job'].get('action_condition',True)]+[r for r in canonical if r['job']['kind']=='D2']))
        write_json(self.output/'representation_ablation.json',[r for r in neural if r['job']['dim']==dim and
            r['job']['capacity']=='medium' and r['job']['hidden_gate']])
        write_json(self.output/'hidden_bottleneck_ablation.json',[r for r in neural if r['job']['representation']=='SHT' and
            r['job']['capacity']=='medium'])
        write_json(self.output/'capacity_ablation.json',[r for r in neural if r['job']['representation']=='SHT' and
            r['job']['dim']==dim and r['job']['hidden_gate']])
        write_json(self.output/'seed_variation.json',[r for r in results if r['job']['kind'] in ('D2','V2') and
            r['job']['dim']==dim and r['job']['representation']=='SHT' and r['job']['hidden_gate'] and
            r['job']['capacity']=='medium' and r['job']['ntrain']==ntrain and r['job'].get('action_condition',True)])
        for prefix in ('drafter','verifier'):
            kind='D2' if prefix=='drafter' else 'V2'
            write_json(self.output/(prefix+'_scaling.json'),[r for r in results if r['job']['kind']==kind and
                r['job']['representation']=='SHT' and r['job']['hidden_gate'] and r['job']['dim']==dim and
                r['job']['capacity']=='medium' and r['job']['seed']==self.cfg['seed'] and r['job'].get('action_condition',True)])

    def final(self,plan):
        main_dim=max(self.cfg['hidden_dims'])
        def canonical(kind):
            return next(j for j in plan if j['kind']==kind and j['dim']==main_dim and
                j['seed']==self.cfg['seed'] and j['representation']=='SHT' and j['hidden_gate'] and
                j['capacity']=='medium' and j['ntrain']==len(self.split['train']))
        d,prep,prepared,prior,native,dr=self.execute(canonical('D2'),True)
        v,_,_,_,_,vr=self.execute(canonical('V2'),True)
        # Primary pair is preregistered medium SHT, not chosen using test results.
        # Ablation rankings are validation-only and reported, not fed back into
        # the test to silently change this primary hypothesis.
        d.to(self.cfg['device_drafter']);v.to(self.cfg['device_verifier'])
        direct=DirectOutcome(prep.layout,self.cfg['capacities']['medium']['width'])
        folder=self.output/'direct_outcome';folder.mkdir(exist_ok=True)
        train_paths=[p for p in self.allpaths['train'] if not any((a,b) in self.bad for a,b in zip(p[0],p[0][1:]))]
        file=folder/'best.pt'
        if file.exists() and (folder/'training_complete.json').exists():direct.load_state_dict(torch.load(file,weights_only=True)['weights']);direct.to(self.cfg['device_verifier'])
        else:train_v(direct,None,self.allpaths['val'],self.cfg,self.cfg['device_verifier'],folder,train_paths,prepared)
        copied=PriorDynamics(prep,prior).to(self.cfg['device_drafter'])
        prior_records=evaluate_d(copied,prep,prepared,self.allpaths['test'],self.cfg['device_drafter'],native)
        write_json(self.output/'identity_copy_prior.json',dynamics_report(prior_records))
        write_jsonl(self.output/'identity_copy_predictions.jsonl',prior_records)
        # Decoder floor separately: native head on real child representations.
        real_readout=[]
        for uid,r in prepared.items():
            if r['question'] not in set(self.split['test']):continue
            real_readout.append(dict(question=r['question'],uid=uid,
                native_readout=region_stats(r['x'],r,torch.ones(r['length'],dtype=torch.bool),
                                            d.decoder,native,prep.layout,prep)))
        write_json(self.output/'native_decoder_real_state_floor.json',real_readout)
        validation=composition(d,v,direct,copied,prepared,self.allpaths['val'],self.cfg,
                              self.cfg['device_drafter'],self.cfg['device_verifier'],'val')
        test=composition(d,v,direct,copied,prepared,self.allpaths['test'],self.cfg,
                        self.cfg['device_drafter'],self.cfg['device_verifier'],'test')
        write_jsonl(self.output/'feasibility_validation_predictions.jsonl',validation)
        drecords=[json.loads(line) for line in (self.output/'jobs'/canonical('D2')['name']/'test_drafter_predictions.jsonl').read_text().splitlines()]
        dlookup={(r['parent'],r['child'],r['actions']):r for r in drecords}
        for row in test:
            record=dlookup[(row['parent'],row['child'],row['actions'])]
            row['drafter_regions']=record['regions'];row['hidden_gate_layer_mean']=record['hidden_gate_layer_mean']
        write_jsonl(self.output/'feasibility_final_predictions.jsonl',test)
        gates={}
        for action in ('R','E'):
            group=[r for r in test if r['horizon']==1 and r['action']==action]
            gates[action]={}
            for name in ('D','V'):
                values=[r['hidden_gate_layer_mean'] if name=='D' else r['oracleD_verifier']['hidden_gate_layer_mean'] for r in group]
                values=[v for v in values if v is not None]
                gates[action][name]=dict(count=len(values),layers=[7,14,28],mean=None if not values else np.mean(values,0),
                    std=None if not values else np.std(values,0))
        write_json(self.output/'hidden_gate_statistics.json',gates)
        write_json(self.output/'composition_oracle_vs_learned.json',composition_report(test))
        for action in ('R','E'):
            reports={};sweeps=[]
            for name in ('oracleD','learnedD','priorD','direct'):
                reports[name],sweep=apply_operating_points(validation,test,name,action)
                sweeps += [dict(model=name,**r) for r in sweep]
            write_json(self.output/f'composition_useful_action_{action}.json',reports)
            write_csv(self.output/f'useful_action_threshold_sweep_{action}.csv',sweeps)
        bs={}
        for horizon in (1,2,3):
            rows=[r for r in test if r['horizon']==horizon]
            bs[f'H{horizon}']={name:paired_question_bootstrap(rows,'learnedD_K',name+'_K',
                samples=self.cfg['bootstrap_samples'],seed=self.cfg['seed']) for name in ('oracleD','priorD','direct')}
        write_json(self.output/'question_bootstrap.json',bs)
        write_json(self.output/'predictability_knn.json',predictability(prepared,self.allpaths['train'],self.allpaths['test']))
        drtest=dr['reports']['test'];vrt=vr['reports']['test']
        for filename in ('drafter_h1_by_action','drafter_rollout_h1_h2_h3','drafter_sequence_breakdown',
                         'drafter_token_agreement','drafter_extend_new_block'):
            write_json(self.output/(filename+'.json'),drtest)
        for filename in ('verifier_token_metrics','verifier_calibration','verifier_by_length'):
            write_json(self.output/(filename+'.json'),vrt)
        write_json(self.output/'verifier_sequence_metrics.json',{sequence:verifier_report([r['oracleD_verifier'] for r in test if r['actions']==sequence])
            for sequence in sorted({r['actions'] for r in test})})
        write_json(self.output/'verifier_response_delta.json',vr['reports']['test_response_delta'])
        # Error growth is paired to SAME endpoints, not global cohorts with
        # different lengths. Pair free and real-parent final-step records.
        root=self.output/'jobs'/canonical('D2')['name']
        free=[json.loads(line) for line in (root/'test_drafter_predictions.jsonl').read_text().splitlines()]
        teacher=[json.loads(line) for line in (root/'teacher_input_predictions.jsonl').read_text().splitlines()]
        floors={(r['parent'],r['child'],r['actions']):r for r in teacher}
        growth=defaultdict(list)
        for r in free:
            base=floors[(r['parent'],r['child'],r['actions'])]
            a=r['regions']['frontier'];b=base['regions']['frontier']
            if a and b and a['mse_hidden'] is not None and b['mse_hidden'] is not None:
                growth['H'+str(r['horizon'])].append(dict(question=r['question'],
                    free=a['mse_hidden'],teacher_input=b['mse_hidden']))
        write_json(self.output/'paired_rollout_error_growth.json',{h:dict(count=len(rr),
            free_MSE=float(np.mean([r['free'] for r in rr])),teacher_input_MSE=float(np.mean([r['teacher_input'] for r in rr])),
            ratio=float(np.mean([r['free'] for r in rr])/max(np.mean([r['teacher_input'] for r in rr]),1e-8))) for h,rr in growth.items()})
        self.report(drtest,vrt,bs,test)
        self.plots()

    def report(self,dr,v,bootstrap,test):
        # Conservative automated disposition, not arbitrary AUROC pass/fail.
        evidence=[]
        for h in ('H1','H2','H3'):
            comparison=bootstrap[h]['priorD'];ci=comparison['ci95']
            evidence.append(bool(ci and ci[1]<0))
        available=all(bootstrap[h]['priorD']['ci95'] is not None for h in ('H1','H2','H3'))
        verdict=('GO—WITH REDESIGN' if any(evidence) else 'NO-GO WITH CURRENT STATE FORMULATION') if available else None
        sections=[f'# Factorized world-model feasibility\n\nProvisional verdict: {verdict if verdict else "INSUFFICIENT EVIDENCE — missing paired horizon cohorts"}.',
            'This is an offline, exploratory study on reused questions. A negative result is not an impossibility proof. '
            'No controller, FiLM, latency objective or new LLM generation was run.',
            '## 1. Does native D dynamics generalize?\n\nInspect frontier and new-block exact STOP-token agreement, native '
            'top-1 agreement, and fresh hidden error in drafter_rollout_h1_h2_h3.json. Whole-proposal identity copying '
            'can dilute errors; immutable and frontier cohorts are reported separately. Committed IDs are never edited.\n\n'+
            json.dumps({h:rr['regions'].get('frontier') for h,rr in dr['by_horizon'].items()},indent=2),
            '## 2. Can V emulate acceptance?\n\n'+json.dumps(v['all']['k'],indent=2),
            'Hazards are conditional clean-prefix acceptance; their cumulative product is actual token acceptance '
            'probability. Full-TF suffix compatibility is a separate head, not acceptance after the first rejection. '
            'Margins are tanh(signed candidate-minus-best-rival / 5), not raw logit margins.',
            '## 3. Where does composition fail?\n\n'+json.dumps(bootstrap,indent=2),
            'Negative paired macro-MAE delta favors learned D. Compare learned-D→V to real-D→V, copy/prior→V, '
            'and direct source/actions→future V. Gaps are operational decompositions, not pure additive causal errors.',
            '## 4. R and E separately\n\nSee composition_useful_action_R/E.json. Precision/recall thresholds were selected '
            'only on validation. CSV sweeps on test are descriptive, not permission to choose a new test threshold.\n\n'+
            json.dumps({a:rr['regions'].get('frontier') for a,rr in dr['by_action'].items()},indent=2),
            '## 5. Bottleneck and next decision\n\nCompare representation, capacity, hidden-gating, and 20/40/60/70 '
            'question curves using the fixed 15-question test. The primary medium SHT pair is preregistered; '
            'ablation rankings cannot silently change it. Train-versus-test reports expose overfitting. Three seeds '
            'measure training variability separately from question-bootstrap intervals. kNN neighborhoods diagnose '
            'this representation only; they do not establish intrinsic randomness or a true predictability ceiling.',
            '## Limitations\n\nNative exact full-vocabulary KL is unavailable in the original trace. Top-32 KL/JS is '
            'conditional on saved support and explicitly includes coverage. The D-only frozen native readout introduces '
            'another bottleneck: compare native_decoder_real_state_floor.json. STOP identity codes are fixed random '
            'train-vocabulary codes, not pretrained semantic embeddings. OOV tokens count as incorrect. Raw hidden '
            'vectors use frozen learned projections; cached rows retain validity/age metadata. Ablations remove learned '
            'access to information classes, but deterministic identity-copy rules remain. History is recomputed; '
            'only cached hidden and rank gaps are copied. Full-raw hidden history-cosine changes remain learned '
            'targets on the frontier, since a lossy projection cannot determine them exactly. Missing verifier '
            'teachers are never converted to negative labels. Two immutable-prefix K inconsistencies are audited; '
            'affected K labels are masked in training, while test paths are preserved.']
        (self.output/'FINAL_FEASIBILITY_REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
        write_json(self.output/'final_verdict.json',dict(verdict=verdict,status='exploratory' if available else 'insufficient_evidence',
            learned_vs_prior_significant_by_horizon=evidence,exploratory=True,
            automatic_rule='Provisional GO-with-redesign if any horizon improves composition vs prior with paired question CI95 below zero; otherwise current formulation no-go. Human review of D/V/R/E is required for unqualified GO.',
            not_a_controller_speedup_claim=True))

    def plots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,prefix in zip(axes,('drafter','verifier')):
            rows=json.loads((self.output/(prefix+'_scaling.json')).read_text())
            rows.sort(key=lambda r:r['job']['ntrain'])
            for part in ('train','val','test'):
                ys=[]
                for r in rows:
                    ys.append(r['reports'][part]['all']['k']['question_macro_mae'] if prefix=='verifier' else
                              r['reports'][part]['by_horizon']['H1']['regions']['frontier']['token_agreement'])
                ax.plot([r['job']['ntrain'] for r in rows],ys,'o-',label=part)
            ax.set(xlabel='Training questions',title=prefix,ylabel='Question-macro K MAE' if prefix=='verifier' else 'H1 frontier STOP agreement')
            ax.legend();ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(self.output/'data_scaling.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for kind,ax in zip(('D2','V2'),axes):
            for r in self.results.values():
                j=r['job']
                if j['kind']!=kind or j['representation']!='SHT' or j['dim']!=max(self.cfg['hidden_dims']) or not j['hidden_gate'] or j['capacity']!='medium' or j['seed']!=self.cfg['seed']:continue
                curve=json.loads((self.output/'jobs'/j['name']/'training_curve.json').read_text())
                key='val_D_only_loss' if kind=='D2' else 'val_macro_K_mae'
                ax.plot([x['update'] for x in curve],[x[key] for x in curve],label=f"{j['ntrain']}q")
            ax.set(xlabel='Gradient updates',title=kind+' validation learning',ylabel='D-only loss' if kind=='D2' else 'K MAE')
            ax.legend();ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(self.output/'training_curves.png',dpi=160);plt.close(fig)


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));seed_all(cfg['seed'])
    if cfg['horizon']!=3 or cfg['max_proposal_tokens']!=64 or cfg['extend_size']!=8:
        raise ValueError('This protocol requires horizon3, proposal64, extension8')
    if cfg['drafter_updates']<cfg['h1_min_updates']+2:raise ValueError('Not enough updates for curriculum')
    for module,names in (('drafter',['hidden','surface','scalars','context','prefix_ids','ids','topk_ids','actions']),
                         ('verifier',['hidden','surface','scalars','context','prefix_ids','ids','topk_ids'])):
        assert_input_features(names,module)
    dataset=load_dataset(args.input,cfg['num_questions'])
    if len(dataset.question_ids)!=cfg['num_questions']:
        raise ValueError(f"Expected {cfg['num_questions']} complete questions; got {len(dataset.question_ids)}")
    split=split_questions(dataset.question_ids,cfg['seed'])
    if not all(split.values()):raise ValueError('Need nonempty train/validation/test question splits')
    content=hashlib.sha256()
    for uid,r in sorted(dataset.states.items()):
        content.update(json.dumps([uid,r.question,r.round_id]).encode())
        for value in (r.ids,r.hidden,r.surface,r.scalars,r.context,r.prefix_ids,r.topk_ids,r.gaps,r.teacher_features,r.teacher_margin):
            if value is not None:content.update(value.numpy().tobytes())
        content.update(str(r.accepted).encode())
    content.update(json.dumps(dataset.edges,sort_keys=True).encode())
    code_source=Path(__file__).parent
    hashes={n:hashlib.sha256((code_source/n).read_bytes()).hexdigest() for n in
            ('run_factorized_wm_feasibility.py','factorized_wm_models.py','factorized_wm_data.py','factorized_wm_metrics.py')}
    signature=dict(config=cfg,content_sha256=content.hexdigest(),code_sha256=hashes,question_ids=dataset.question_ids,
                   states=len(dataset.states),edges=len(dataset.edges))
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    manifest=output/'study_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if not args.resume:raise ValueError('Output already has a study; pass --resume')
        if saved['fingerprint']!=digest:raise ValueError('Resume input/config differ from original study')
    write_json(manifest,dict(fingerprint=digest,source=dataset.source,status='running',config=cfg))
    write_json(output/'config.json',cfg);write_json(output/'split_manifest.json',split)
    write_json(output/'dataset_audit.json',dict(dataset.audit,split={k:len(v) for k,v in split.items()}))
    (output/'dataset_audit.md').write_text('# Dataset audit\n\n'+json.dumps(dataset.audit,indent=2)+'\n',encoding='utf-8')
    write_json(output/'feature_schema.json',dict(registry=feature_registry(),
        D_inputs='native parent plus known R/E actions only',V_inputs='current real/imagined native D only',
        hidden_layers=[7,14,28],hidden_compressor='D-only LN/linear autoencoder then frozen',
        native_distribution='conditional saved top32; full normalizer unavailable',
        full_vocabulary_KL=dict(status='REQUIRES_NEW_LLM_TRACE',value=None),
        STOP_materialization='same-forward top1 before E',token_codes='fixed random train-only identity codes',
        ablation_scope='Network input classes, including prefix identity, are disabled; deterministic state identity-copy rules are retained',
        hazard_targets='censored after first mismatch',margin_targets='tanh(candidate-rival / 5)'))
    # Hash all relevant code into the result for reproducibility.
    source=Path(__file__).parent
    write_json(output/'source_hashes.json',hashes)
    study=Study(dataset,split,cfg,output);plan=experiment_plan(cfg,len(split['train']))
    write_json(output/'experiment_plan.json',plan)
    progress(dict(stage='audit_complete',questions=len(dataset.question_ids),states=len(dataset.states),
                  split={k:len(v) for k,v in split.items()},jobs=len(plan)))
    for index,job in enumerate(plan,1):
        progress(dict(stage='job_start',index=index,total=len(plan),job=job['name']))
        study.execute(job)
    study.final(plan);study.summaries()
    write_json(output/'summary.json',dict(status='complete',questions=len(dataset.question_ids),
        jobs_completed=len(study.results),LLM_forwards=0,split={k:len(v) for k,v in split.items()},
        elapsed_seconds=time.monotonic()-args.started))
    write_json(manifest,dict(fingerprint=digest,source=dataset.source,status='complete',config=cfg))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.started=time.monotonic()
    try:run(args)
    except BaseException:
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(output/'summary.json',dict(status='partial',elapsed_seconds=time.monotonic()-args.started))
        raise
    finally:package(Path(args.output))


if __name__=='__main__':main()
