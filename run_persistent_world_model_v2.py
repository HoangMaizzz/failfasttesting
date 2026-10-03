"""Offline behavioral V2 on the original TwoSource experiences + native FiLM test.

All outer tests are question-grouped. Current/future verifier outputs are labels,
never pre-state inputs. Only previous actually executed STOPs enter causal memory.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import gc
import json
from pathlib import Path
import random
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from world_model_core import valid_positions, expected_acceptance, prefix_log_distribution
from persistent_world_model_v2 import (BehavioralWorldModelV2, FixedDynamicsV2,
    DirectOutcomeV2, StateV2, pack_v2, semantic_verifier, behavioral_loss,
    action_descriptor, VARIANTS)
from run_twosource_grouped_cv import load_experiences, grouped_folds, atomic_json, append_jsonl, read_jsonl
from run_persistent_world_model_v1 import _auc


def bucket(length):
    return min(40, (int(length)+7)//8*8)


class Balanced:
    def __init__(self, values, key, seed):
        self.groups = defaultdict(list)
        for value in values:
            self.groups[key(value)].append(value)
        self.keys = sorted(self.groups, key=str)
        self.rng = random.Random(seed)

    def sample(self, count):
        if not self.keys:
            raise ValueError('No eligible samples in this split')
        return [self.rng.choice(self.groups[self.rng.choice(self.keys)]) for _ in range(count)]


def causal_memory(observations):
    """Shadow labels do not count as a STOP. Within-round verifier is unavailable."""
    actual = defaultdict(dict)
    for o in observations.values():
        if o.teacher_is_actual and o.accepted is not None:
            key = (o.question, o.round_id)
            if o.round_id in actual[o.question]:
                raise ValueError(f'Multiple actual STOPs at {key}; episode provenance required')
            actual[o.question][o.round_id] = o
    memory = {}
    for o in observations.values():
        past = [v for r, v in sorted(actual[o.question].items()) if r < o.round_id]
        # Verify causality with the previously saved real-history count as well.
        expected = 0 if o.verifier_history is None else len(o.verifier_history)
        past = past[-8:]
        if len(past) > expected:
            past = past[-expected:] if expected else []
        memory[o.uid] = (torch.stack([semantic_verifier(v) for v in past])
                         if past else torch.empty(0, 40))
    return memory


def enumerate_paths(observations, edges, question_ids, horizon=3):
    allowed = set(question_ids)
    outgoing = defaultdict(list)
    for parent, child, action in edges:
        p, c = observations[parent], observations[child]
        if p.question not in allowed or c.question not in allowed:
            continue
        if p.round_id != c.round_id:
            raise ValueError('R/E edge crosses a verifier round')
        if action not in ('R', 'E'):
            continue
        expected = p.length + (8 if action == 'E' else 0)
        if c.length != expected or c.length > 64:
            raise ValueError(f'Invalid R/E length transition: {parent} -> {child}')
        outgoing[parent].append((child, int(action == 'E')))
    result = []
    def visit(nodes, actions):
        if actions and all(observations[u].accepted is not None for u in nodes):
            result.append((tuple(nodes), tuple(actions)))
        if len(actions) == horizon:
            return
        for child, action in outgoing[nodes[-1]]:
            if child in nodes:
                raise ValueError('Cyclic within-round trajectory')
            visit(nodes+[child], actions+[action])
    for parent in sorted(outgoing):
        visit([parent], [])
    return result


def target_batch(rows, device):
    """Small observable GT only. No raw hidden allocation during dynamics."""
    width = max(o.length for o in rows)
    features = torch.zeros(len(rows), width, 35, device=device)
    margins = torch.zeros(len(rows), width, device=device)
    for i, o in enumerate(rows):
        if o.teacher_features is not None:
            features[i, :o.length] = o.teacher_features.to(device)
        if o.teacher_margin is not None:
            margins[i, :o.length] = o.teacher_margin.to(device)
    return dict(lengths=torch.tensor([o.length for o in rows], device=device),
        labels=torch.tensor([o.accepted for o in rows], device=device),
        teacher_features=features, teacher_margin=margins,
        teacher_present=torch.tensor([o.teacher_features is not None for o in rows], device=device))


@torch.no_grad()
def encode_nodes(model, observations, memory, device, batch_size=8):
    model.eval()
    result = {}
    values = list(observations.values())
    for start in range(0, len(values), batch_size):
        group = values[start:start+batch_size]
        b = pack_v2(group, memory, device)
        pre = model.pre(b); post = model.post(pre, b)
        for i, o in enumerate(group):
            result[o.uid] = dict(pre=pre.z[i].cpu(), post=post.z[i].cpu(),
                                 structure=pre.structure[i].cpu())
    return result


def states_from_cache(cache, ids, device, posterior=False):
    key = 'post' if posterior else 'pre'
    return StateV2(torch.stack([cache[u][key] for u in ids]).to(device),
                   torch.stack([cache[u]['structure'] for u in ids]).to(device))


def records_from_heads(heads, lengths, rows, **extra):
    logits = heads['hazard']
    hazard = logits.sigmoid().detach().cpu()
    survival = F.logsigmoid(logits).cumsum(-1).exp().detach().cpu()
    logp = prefix_log_distribution(logits, lengths).detach().cpu()
    output = []
    for i, o in enumerate(rows):
        if o.accepted is None:
            continue
        y, length = o.accepted, o.length
        clean = min(length, y+int(y < length))
        record = dict(question=o.question, state_id=o.uid, length=length, accepted=y,
            expected_yield=float(survival[i, :length].sum()),
            mode=int(logp[i].argmax()), hazard_nll=float(-logp[i, y]),
            hazards=hazard[i, :length].tolist(), survival=survival[i, :length].tolist(), **extra)
        if 'probability' in heads and o.teacher_features is not None:
            record.update(probability_truth=o.teacher_features[:clean, 1].tolist(),
                probability_pred=heads['probability'][i, :clean].detach().cpu().tolist(),
                gap_truth=(-o.teacher_margin[:clean]).clamp_min(0).tolist(),
                gap_pred=heads['gap'][i, :clean].detach().cpu().tolist())
        output.append(record)
    return output


def ece(truth, scores, bins=10):
    truth, scores = np.asarray(truth), np.asarray(scores)
    if not len(truth):
        return None
    return float(sum((m.sum()/len(truth))*abs(truth[m].mean()-scores[m].mean())
        for k in range(bins) if (m := ((scores >= k/bins) &
        (scores < (k+1)/bins if k < bins-1 else scores <= 1))).any()))


def metrics(rows):
    if not rows:
        return {'n': 0}
    y = np.asarray([r['accepted'] for r in rows], dtype=float)
    pred = np.asarray([r['expected_yield'] for r in rows], dtype=float)
    error = pred-y
    def ranks(x):
        order = np.argsort(x, kind='stable'); r = np.empty(len(x), dtype=float)
        for value in np.unique(x):
            mask = x[order] == value; r[order[mask]] = np.flatnonzero(mask).mean()
        return r
    ry, rp = ranks(y), ranks(pred)
    rho = float(np.corrcoef(ry, rp)[0, 1]) if ry.std() and rp.std() else None
    hy, hp, sy, sp, probability_errors, gap_errors = [], [], [], [], [], []
    by_question = defaultdict(list)
    for r in rows:
        by_question[r['question']].append(abs(r['expected_yield']-r['accepted']))
        hy += [int(i < r['accepted']) for i in range(min(r['length'], r['accepted']+1))]
        hp += r['hazards'][:min(r['length'], r['accepted']+1)]
        sy += [int(i < r['accepted']) for i in range(r['length'])]; sp += r['survival']
        probability_errors += [abs(a-b) for a,b in zip(r.get('probability_truth', []), r.get('probability_pred', []))]
        gap_errors += [abs(a-b) for a,b in zip(r.get('gap_truth', []), r.get('gap_pred', []))]
    hp, hy, sp, sy = map(np.asarray, (hp, hy, sp, sy))
    result = dict(n=len(rows), questions=len(by_question), mae=float(abs(error).mean()),
        question_macro_mae=float(np.mean([np.mean(v) for v in by_question.values()])),
        rmse=float(np.sqrt((error**2).mean())), signed_bias=float(error.mean()),
        p90_absolute_error=float(np.quantile(abs(error), .9)), spearman=rho,
        hazard_nll=float(np.mean([r['hazard_nll'] for r in rows])),
        hazard_auc=_auc(hy, hp), hazard_brier=float(np.mean((hp-hy)**2)),
        hazard_ece=ece(hy,hp), survival_auc=_auc(sy,sp),
        survival_brier=float(np.mean((sp-sy)**2)), survival_ece=ece(sy,sp),
        exact_K_rate=float(np.mean([r['mode']==r['accepted'] for r in rows])),
        verifier_probability_mae=float(np.mean(probability_errors)) if probability_errors else None,
        verifier_gap_mae=float(np.mean(gap_errors)) if gap_errors else None)
    if rows[0].get('information')=='source_verifier_oracle_persistence':
        # This baseline predicts only K. Copied placeholders must NOT be reported
        # as its hazard/latent predictions.
        return {k:result[k] for k in ('n','questions','mae','question_macro_mae','rmse','signed_bias','p90_absolute_error','spearman')}
    for key in ('latent_nmse', 'latent_cosine', 'effect_abs_error'):
        v = [r[key] for r in rows if key in r]
        if v:
            result[key] = float(np.mean(v))
    return result


def report_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        groups[f"L={bucket(r['length'])}{'+' if bucket(r['length'])==40 else ''}"].append(r)
        if 'actions' in r:
            groups['actions='+r['actions']].append(r)
            groups['H='+str(r['horizon'])].append(r)
            groups['H='+str(r['horizon'])+'/actions='+r['actions']].append(r)
    return {k: metrics(v) for k,v in sorted(groups.items())}


def subset_nodes(observations, qids, limit=64):
    groups = defaultdict(list)
    for o in observations.values():
        if o.question in set(qids) and o.accepted is not None:
            groups[o.question].append(o)
    selected = []
    for q, values in sorted(groups.items()):
        # Fixed, length-stratified evaluation sample, not outcome-selected.
        bylen = defaultdict(list)
        for o in values: bylen[bucket(o.length)].append(o)
        for length, group in sorted(bylen.items()):
            selected += sorted(group, key=lambda o:o.uid)[:max(1,limit//len(bylen))]
    return selected


@torch.no_grad()
def evaluate_current(model, rows, memory, device, batch_size=8, posterior=False):
    output = []; model.eval()
    for start in range(0,len(rows),batch_size):
        group = rows[start:start+batch_size]; b = pack_v2(group,memory,device)
        state = model.pre(b)
        if posterior: state = model.post(state,b)
        output += records_from_heads(model.heads(state),state.lengths,group,
                                     information='post_privileged' if posterior else 'pre_causal')
    return output


def snapshot(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def length_mean_baseline(train_rows,test_rows):
    groups=defaultdict(list)
    for o in train_rows:
        if o.accepted is not None:groups[bucket(o.length)].append(o.accepted)
    fallback=float(np.mean([v for group in groups.values() for v in group]))
    errors=defaultdict(list)
    for o in test_rows:
        predicted=float(np.mean(groups[bucket(o.length)])) if groups[bucket(o.length)] else fallback
        errors[o.question].append(abs(predicted-o.accepted))
    return dict(question_macro_mae=float(np.mean([np.mean(v) for v in errors.values()])),
                mae=float(np.mean([v for group in errors.values() for v in group])),
                information='fit on training questions only; proposal length bucket')


def train_representation(args, out, observations, memory, train_ids, dev_ids, test_ids, variant, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    first = next(iter(observations.values()))
    model = BehavioralWorldModelV2(first.hidden.shape[-1], first.hidden.shape[1],variant).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    devrows = subset_nodes(observations,dev_ids,args.eval_states_per_question)
    testrows = subset_nodes(observations,test_ids,args.eval_states_per_question)
    train_by_question = defaultdict(list)
    for o in observations.values():
        if o.question in train_ids and o.accepted is not None:
            train_by_question[o.question].append(o)
    order = list(train_ids); rng.shuffle(order); pool=[]; curve=[]
    best, best_state, stale = float('inf'), None, 0
    milestones = sorted(set([0]+[v for v in args.milestones if v <=len(order)]+[len(order)]))
    updates=0
    for index in range(len(order)+1):
        if index:
            pool += train_by_question[order[index-1]]
            sampler = Balanced(pool,lambda o:bucket(o.length),seed+index)
            model.train()
            for _ in range(args.updates_per_question):
                batch=pack_v2(sampler.sample(args.batch_size),memory,args.device)
                loss, terms=model.representation_loss(batch)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite representation loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); updates+=1
                append_jsonl(out/'training.jsonl',[dict(stage='representation',variant=variant,
                    train_questions=index, update=updates,loss=float(loss.detach()),**terms)])
        if index not in milestones: continue
        dev=evaluate_current(model,devrows,memory,args.device,args.batch_size)
        test=evaluate_current(model,testrows,memory,args.device,args.batch_size)
        score=metrics(dev)['question_macro_mae']
        # Zero-data state is reported, but never chosen as a trained checkpoint.
        improved = index>0 and score < best
        if improved:
            best=score; best_state=snapshot(model); stale=0
            torch.save(dict(model=best_state,config=model.config,selected_questions=index,
                dev_mae=best,selection='inner_dev_only'),out/f'{variant}_representation_best.pt')
        elif index: stale+=1
        entry=dict(train_questions=index,updates=updates,dev=report_groups(dev),test=report_groups(test),selected=improved)
        curve.append(entry); atomic_json(out/f'{variant}_learning_curve.json',curve)
        print(f"[v2-repr] variant={variant} q={index} dev={score:.3f} test={metrics(test)['mae']:.3f}",flush=True)
        if stale >= args.patience: break
    if best_state is None: raise RuntimeError('No trained representation checkpoint selected')
    model.load_state_dict(best_state); model.freeze_representation()
    current=evaluate_current(model,testrows,memory,args.device,args.batch_size)
    posterior=evaluate_current(model,testrows,memory,args.device,args.batch_size,True)
    append_jsonl(out/f'{variant}_current_predictions.jsonl',current)
    append_jsonl(out/f'{variant}_posterior_privileged_predictions.jsonl',posterior)
    return model, dict(current=report_groups(current),posterior_privileged=report_groups(posterior),learning_curve=curve,
        length_mean_baseline=length_mean_baseline(pool,testrows),
        selected_dev_mae=best,selection='inner_dev_current_pre_question_macro_MAE')


def rollout_loss(model,dynamics,cache,observations,paths,device,latent_weight=.1):
    """All valid subpaths are sampled; no intermediate teacher forcing."""
    source=states_from_cache(cache,[p[0][0] for p in paths],device)
    state=source; total=state.z.sum()*0; logs=[]
    for depth in range(len(paths[0][1])):
        actions=torch.tensor([p[1][depth] for p in paths],device=device)
        previous=state
        state,effect,delta=dynamics(state,actions)
        ids=[p[0][depth+1] for p in paths]
        rows=[observations[u] for u in ids]; target=target_batch(rows,device)
        # Future posterior is a FIXED learned reference only, never an input or GT.
        teacher=states_from_cache(cache,ids,device,True)
        behavior, terms=behavioral_loss(model.heads(state),target)
        latent=((state.z-teacher.z).square().mean(-1)/teacher.z.square().mean(-1).clamp_min(.01)).mean()
        old_y=torch.tensor([observations[p[0][depth]].accepted for p in paths],device=device)
        effect_loss=F.smooth_l1_loss(effect/8,(target['labels']-old_y).float()/8)
        mask_loss=F.mse_loss(state.structure[:,2],teacher.structure[:,2])
        residual_regularizer=(delta.square().mean(-1)*(actions==0)).mean()
        total=total+behavior+latent_weight*latent+.5*effect_loss+.05*mask_loss+.001*residual_regularizer
        logs.append(dict(depth=depth+1,**terms,latent_nmse=float(latent.detach()),effect_huber=float(effect_loss.detach())))
    return total/len(paths[0][1]),logs


@torch.no_grad()
def evaluate_paths(model,dynamics,direct,cache,observations,paths,device,batch_size=16):
    output=defaultdict(list)
    for h in range(1,4):
        group=[p for p in paths if len(p[1])==h]
        for start in range(0,len(group),batch_size):
            batch=group[start:start+batch_size]
            source=states_from_cache(cache,[p[0][0] for p in batch],device)
            predicted=source; identity=source; actions_all=[]; effect=None
            for depth in range(h):
                actions=torch.tensor([p[1][depth] for p in batch],device=device); actions_all.append(actions)
                predicted,effect,_=dynamics(predicted,actions)
                _, structural=action_descriptor(identity,actions)
                identity=StateV2(source.z,structural)
            ids=[p[0][-1] for p in batch]; rows=[observations[u] for u in ids]
            teacher=states_from_cache(cache,ids,device,True)
            for name,state in [('dynamics',predicted),('no_dynamics',identity)]:
                records=records_from_heads(model.heads(state),state.lengths,rows,horizon=h)
                for i,r in enumerate(records):
                    r['actions']=''.join('E' if a else 'R' for a in batch[i][1])
                    r['source_id']=batch[i][0][0]
                    r['latent_nmse']=float((state.z[i]-teacher.z[i]).square().mean()/teacher.z[i].square().mean().clamp_min(.01))
                    r['latent_cosine']=float(F.cosine_similarity(state.z[i:i+1],teacher.z[i:i+1]))
                    if name=='dynamics':
                        delta_y=rows[i].accepted-observations[batch[i][0][-2]].accepted
                        r['effect_abs_error']=float(abs(effect[i]-delta_y))
                    output[name].append(r)
            # Persistence uses SOURCE TRUTH, so it is an explicitly privileged diagnostic.
            persistence=copy.deepcopy(output['no_dynamics'][-len(batch):])
            for p,r in zip(batch,persistence):
                r['expected_yield']=float(observations[p[0][0]].accepted)
                r['information']='source_verifier_oracle_persistence'
            output['oracle_persistence']+=persistence
            if direct is not None:
                logits,lengths=direct(source,actions_all)
                records=records_from_heads({'hazard':logits},lengths,rows,horizon=h)
                for p,r in zip(batch,records):
                    r['actions']=''.join('E' if a else 'R' for a in p[1]);r['source_id']=p[0][0]
                output['direct']+=records
    return dict(output)


def train_dynamics(args,out,model,cache,observations,train_paths,dev_paths,test_paths,name='hybrid',generic=False,latent_weight=.1):
    torch.manual_seed(args.seed+123); rng=random.Random(args.seed+123)
    dynamics=FixedDynamicsV2(generic).to(args.device)
    optimizer=torch.optim.AdamW(dynamics.parameters(),lr=args.learning_rate)
    curve=[]
    for horizon in (1,2,3):
        available=[p for p in train_paths if len(p[1])<=horizon]
        samplers={h:Balanced([p for p in available if len(p[1])==h],
            lambda p:(bucket(observations[p[0][0]].length),p[1]),args.seed+h) for h in range(1,horizon+1)}
        samplers={h:s for h,s in samplers.items() if s.keys}
        if not samplers: raise ValueError('No eligible R/E paths')
        best=float('inf'); best_state=None; stale=0
        for update in range(1,args.dynamics_updates+1):
            h=rng.choice(list(samplers)); batch=samplers[h].sample(args.batch_size)
            dynamics.train(); loss,terms=rollout_loss(model,dynamics,cache,observations,batch,args.device,latent_weight)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite dynamics loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(),1.0);optimizer.step()
            append_jsonl(out/'training.jsonl',[dict(stage='dynamics',name=name,horizon=horizon,
                update=update,loss=float(loss.detach()),terms=terms)])
            if update%args.eval_every and update!=args.dynamics_updates:continue
            dynamics.eval()
            dev=evaluate_paths(model,dynamics,None,cache,observations,
                [p for p in dev_paths if len(p[1])==horizon],args.device,args.batch_size)
            score=metrics(dev.get('dynamics',[])).get('question_macro_mae',float('inf'))
            improved=score<best
            if improved:
                best=score;best_state=snapshot(dynamics);stale=0
                torch.save(dict(dynamics=best_state,generic=generic,horizon=horizon,
                    latent_weight=latent_weight,dev_mae=best,selection='inner_dev_only'),out/f'{name}_H{horizon}_best.pt')
            else:stale+=1
            curve.append(dict(horizon=horizon,update=update,dev=report_groups(dev.get('dynamics',[])),selected=improved))
            atomic_json(out/f'{name}_dynamics_curve.json',curve)
            print(f'[v2-dyn] {name} H{horizon} update={update} dev={score:.3f}',flush=True)
            if stale>=args.patience:break
        if best_state is not None:dynamics.load_state_dict(best_state)
        # Save test scores for the best H1/H2/H3 checkpoints, not merely final.
        test=evaluate_paths(model,dynamics,None,cache,observations,test_paths,args.device,args.batch_size)
        atomic_json(out/f'{name}_checkpoint_H{horizon}_test.json',{k:report_groups(v) for k,v in test.items()})
    dynamics.eval()
    return dynamics


def train_direct(args,out,cache,observations,train_paths,dev_paths):
    torch.manual_seed(args.seed+124);rng=random.Random(args.seed+124)
    model=DirectOutcomeV2().to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    samplers={h:Balanced([p for p in train_paths if len(p[1])==h],
        lambda p:(bucket(observations[p[0][0]].length),p[1]),args.seed+h) for h in (1,2,3)}
    samplers={h:s for h,s in samplers.items() if s.keys};best=float('inf');best_state=None
    from world_model_core import acceptance_nll
    for step in range(1,args.dynamics_updates*3+1):
        h=rng.choice(list(samplers));batch=samplers[h].sample(args.batch_size)
        source=states_from_cache(cache,[p[0][0] for p in batch],args.device)
        actions=[torch.tensor([p[1][j] for p in batch],device=args.device) for j in range(h)]
        logits,lengths=model(source,actions)
        labels=torch.tensor([observations[p[0][-1]].accepted for p in batch],device=args.device)
        loss=acceptance_nll(logits,lengths,labels)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%args.eval_every and step!=args.dynamics_updates*3:continue
        # Direct has no future latent; score its endpoint behavioral readout only.
        rows=[]
        with torch.no_grad():
            for horizon in (1,2,3):
                paths=[p for p in dev_paths if len(p[1])==horizon]
                for start in range(0,len(paths),args.batch_size):
                    group=paths[start:start+args.batch_size]
                    source=states_from_cache(cache,[p[0][0] for p in group],args.device)
                    actions=[torch.tensor([p[1][j] for p in group],device=args.device) for j in range(horizon)]
                    logits,lengths=model(source,actions)
                    rows+=records_from_heads({'hazard':logits},lengths,[observations[p[0][-1]] for p in group])
        score=metrics(rows)['question_macro_mae']
        if score<best:best=score;best_state=snapshot(model)
        append_jsonl(out/'training.jsonl',[dict(stage='direct',update=step,loss=float(loss.detach()),dev_mae=score)])
    model.load_state_dict(best_state);model.eval()
    torch.save(dict(model=best_state,dev_mae=best),out/'direct_best.pt')
    return model


def package(out,summary):
    atomic_json(out/'summary.json',summary)
    temp=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zf:
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                zf.write(path,path.relative_to(out).as_posix())
    temp.replace(out.with_suffix('.zip'))
    print('[archive]',summary['status'],out.with_suffix('.zip'),flush=True)


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    summary=dict(schema='persistent_behavioral_v2_experiment',status='running',completed_folds=0,
        gt='direct real-verifier accepted prefix, survival, p(candidate), logit gap; learned latent is NOT GT',
        information='pre uses drafter + previous actual STOP only; shadow/future verifier are targets only',folds=[])
    try:
        source,observations,edges,qids=load_experiences(args.run_dir)
        cfg=json.loads((args.run_dir/'config.json').read_text(encoding='utf-8'))
        if source.get('dataset',cfg.get('dataset','gsm8k'))!='gsm8k':raise ValueError('GSM8K input required')
        if len(qids)!=args.num_questions:raise ValueError(f'Need {args.num_questions} questions; got {len(qids)}')
        if int(cfg.get('episodes_per_question',1))!=1:raise ValueError('V2 requires episode-disambiguated input; use 1 episode/question')
        memory=causal_memory(observations);folds=grouped_folds(qids,5,args.seed)
        atomic_json(out/'source_provenance.json',dict(source_summary=source,input_path=str(args.run_dir),
            source_revision=args.source_revision,nodes=len(observations),edges=len(edges),questions=qids,
            split_unit='question',current_teacher_is_target_only=True))
        for f,test_ids in enumerate(folds):
            others=sorted(set(qids)-set(test_ids));random.Random(args.seed+f).shuffle(others)
            dev_count=max(1,min(args.dev_questions,len(others)//4))
            dev_ids,train_ids=others[:dev_count],others[dev_count:]
            folder=out/f'fold_{f}';folder.mkdir()
            atomic_json(folder/'split.json',dict(train=train_ids,dev=dev_ids,test=test_ids))
            paths={k:enumerate_paths(observations,edges,ids) for k,ids in
                [('train',train_ids),('dev',dev_ids),('test',test_ids)]}
            fold_result=dict(fold=f,representation={},dynamics={},film=None)
            for variant in args.variants:
                model,representation=train_representation(args,folder,observations,memory,
                    train_ids,dev_ids,test_ids,variant,args.seed+f)
                fold_result['representation'][variant]=representation
                cache=encode_nodes(model,observations,memory,args.device,args.batch_size)
                dynamics=train_dynamics(args,folder,model,cache,observations,paths['train'],
                    paths['dev'],paths['test'],name=variant+'_hybrid')
                direct=train_direct(args,folder,cache,observations,paths['train'],paths['dev']) if variant=='full' else None
                evaluated=evaluate_paths(model,dynamics,direct,cache,observations,paths['test'],args.device,args.batch_size)
                for name,rows in evaluated.items():append_jsonl(folder/f'{variant}_{name}_predictions.jsonl',rows)
                fold_result['dynamics'][variant]={k:report_groups(v) for k,v in evaluated.items()}
                if variant=='full':
                    # Architecture ablation and objective ablation are separate comparisons.
                    for name,generic,weight in [('generic_hybrid',True,.1),('residual_latent_heavy',False,1.)]:
                        alternative=train_dynamics(args,folder,model,cache,observations,paths['train'],paths['dev'],paths['test'],name,generic,weight)
                        records=evaluate_paths(model,alternative,None,cache,observations,paths['test'],args.device,args.batch_size)['dynamics']
                        fold_result['dynamics'][name]=report_groups(records)
                        append_jsonl(folder/f'{name}_predictions.jsonl',records);del alternative
                    torch.save(dict(model=snapshot(model),config=model.config,dynamics=snapshot(dynamics)),folder/'full_for_film.pt')
                del model,cache,dynamics,direct;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                atomic_json(folder/'results.json',fold_result)
                package(out,summary)
            summary['folds'].append(fold_result);summary['completed_folds']=f+1
            package(out,summary)
        # GPU models are loaded once, AFTER offline training has released raw experiences.
        if args.film_steps:
            from world_model_film_v2 import run_film_experiment
            summary['film']=run_film_experiment(args,out,observations,edges,memory)
        else:summary['film']=dict(status='disabled_explicitly')
        summary['learning_curve_mean']=aggregate_curves(summary['folds'])
        atomic_json(out/'learning_curve_mean.json',summary['learning_curve_mean'])
        summary['pooled_oof'],summary['factor_impact']=aggregate_oof(out,args.variants,args.seed)
        atomic_json(out/'pooled_oof_metrics.json',summary['pooled_oof'])
        atomic_json(out/'factor_impact.json',summary['factor_impact'])
        summary['status']='complete'
        write_report(out,summary)
        package(out,summary)
    except BaseException:
        summary['status']='partial';(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        package(out,summary);raise


def write_report(out,summary):
    text=['# Behavioral V2 results','',
        'Ground truth = directly observed verifier behavior, not a neural latent. Primary loss: hazard + survival Brier + candidate-probability MSE + scaled-gap Huber. Dynamics adds 0.5 effect Huber and 0.1 frozen-posterior consistency. No hidden reconstruction.',
        '','Outer split: 5 question-grouped folds. Inner dev alone selects checkpoints. Test milestones are observations, never selection criteria. Do not treat sibling/path predictions as independent questions.',
        '','Rollout: all valid H1/H2/H3 within-round subsequences; pure imagined intermediate states. Runtime prior contains previous actual STOP only. Oracle persistence and current posterior are privileged diagnostics, not deployable baselines.',
        '','## Held-out diagnostics','',
        '| Fold | Current MAE | H1 MAE | H2 MAE | H3 MAE |','|---|---:|---:|---:|---:|']
    for f in summary['folds']:
        r=f['representation']['full']['current']['all'];d=f['dynamics']['full']['dynamics']
        values=[d.get('H='+str(h),{}).get('mae') for h in (1,2,3)]
        text.append('| '+str(f['fold'])+' | '+f"{r['mae']:.3f}"+' | '+' | '.join('N/A' if v is None else f'{v:.3f}' for v in values)+' |')
    text+=['','## Interpretation','',
        'Check retrained feature ablations, R/E sequences, lengths and direct/no-dynamics baselines in fold results. Low latent MSE or training loss alone is not success. If current prediction is strong but rollout loses to direct, transition representation is the bottleneck; if both fail, observations/behavior readout are the bottleneck.',
        '','FiLM must increase paired REAL verifier acceptance on held-out questions. Proxy KL/top1 improvement does not prove end-to-end benefit. Native R replay wall time is training overhead, not production action latency. EOS/unavailable native snapshots are exclusions, not negative labels.',
        '','FiLM status: '+str(summary.get('film',{}).get('status'))]
    pooled=summary.get('pooled_oof',{})
    if pooled:
        current=pooled['full_current']['all']['question_macro_mae']
        naive=float(np.mean([f['representation']['full']['length_mean_baseline']['question_macro_mae'] for f in summary['folds']]))
        text+=['','## Measured outcome (lower error is better)','',
            f'Current pre-state macro MAE: {current:.3f}; train-only length-mean baseline: {naive:.3f}. '
            +('Representation improves this naive baseline.' if current<naive else 'Representation has not beaten this naive baseline.'),
            '','| Horizon | Dynamics MAE | No-dynamics MAE | Direct MAE |','|---|---:|---:|---:|']
        for h in (1,2,3):
            key='H='+str(h)
            values=[pooled[name].get(key,{}).get('question_macro_mae') for name in ('full_dynamics','no_dynamics','direct')]
            text.append('| '+str(h)+' | '+' | '.join('N/A' if v is None else f'{v:.3f}' for v in values)+' |')
        for action in ('R','E'):
            key='H=1/actions='+action
            dyn=pooled['full_dynamics'].get(key,{}).get('question_macro_mae')
            identity=pooled['no_dynamics'].get(key,{}).get('question_macro_mae')
            if dyn is not None and identity is not None:
                text+=['',f'{action} H1: dynamics={dyn:.3f}, identity/no-dynamics={identity:.3f}. '
                    +('Learned transition beats identity on this held-out behavioral metric.' if dyn<identity else 'Transition is still a bottleneck: it does not beat identity on this metric.')]
        text+=['','Feature gains must be read from retrained `factor_impact.json` and its question-bootstrap interval. '
            'A point estimate whose interval overlaps zero is not strong evidence of a gain.']
    film=summary.get('film',{}).get('pooled_oof_real',{})
    if film:
        text+=['','## Real verifier FiLM (raw trained arms)','',
            '| Arm | Questions | Delta accepted tokens (macro) | Question-bootstrap 95% CI |',
            '|---|---:|---:|---|']
        for arm,result in film.items():
            if not result.get('n'):continue
            ci=result['question_bootstrap_ci95'];gain=result['question_macro_delta_accepted']
            text.append(f"| {arm} | {result['questions']} | {gain:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
        primary=film.get('late_acceptance',{})
        if primary.get('n'):
            lower=primary['question_bootstrap_ci95'][0]
            text+=['','Late-acceptance verdict: '+('positive held-out effect with CI above zero.' if lower>0 else
                'no conclusive positive real-verifier effect yet; proxy improvements alone are insufficient.')]
    (out/'READ_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def aggregate_curves(folds):
    groups=defaultdict(list)
    for fold in folds:
        for variant,data in fold['representation'].items():
            for row in data['learning_curve']:
                groups[(variant,row['train_questions'])].append(row['test']['all']['question_macro_mae'])
    return [dict(variant=k[0],train_questions=k[1],folds=len(values),
        mean_question_macro_mae=float(np.mean(values)),std_between_folds=float(np.std(values)),
        test_was_not_used_for_selection=True) for k,values in sorted(groups.items())]


def aggregate_oof(out,variants,seed):
    files={v+'_current':f'{v}_current_predictions.jsonl' for v in variants}
    files.update({v+'_dynamics':f'{v}_dynamics_predictions.jsonl' for v in variants})
    files.update({k+'_rollout':f'{k}_predictions.jsonl' for k in ('generic_hybrid','residual_latent_heavy')})
    files.update({k:f'full_{k}_predictions.jsonl' for k in ('no_dynamics','oracle_persistence','direct')})
    rows={key:sum((read_jsonl(out/f'fold_{f}'/name) for f in range(5)),[]) for key,name in files.items()}
    pooled={key:report_groups(value) for key,value in rows.items()}
    impacts={};rng=np.random.default_rng(seed)
    def by_question(records,h=None):
        result=defaultdict(list)
        for r in records:
            if h is None or r.get('horizon')==h:
                result[r['question']].append(abs(r['expected_yield']-r['accepted']))
        return {q:float(np.mean(v)) for q,v in result.items()}
    for key,value in rows.items():
        current=key.endswith('_current');baseline=rows['full_current' if current else 'full_dynamics']
        for h in ([None] if current else [1,2,3]):
            base,other=by_question(baseline,h),by_question(value,h);shared=sorted(base.keys()&other.keys())
            if not shared:continue
            delta=np.array([other[q]-base[q] for q in shared])
            ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)],[.025,.975])
            impacts[key+'/'+('current' if current else f'H{h}')]=dict(questions=len(shared),
                delta_question_macro_mae=float(delta.mean()),question_bootstrap_ci95=ci.tolist(),
                interpretation='positive means this ablation/baseline is worse than full hybrid',
                oracle_diagnostic=key=='oracle_persistence')
    return pooled,impacts


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--run_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--source_revision',default='unrecorded_local_run')
    p.add_argument('--dllm_dir',type=Path);p.add_argument('--num_questions',type=int,default=100)
    p.add_argument('--device',default='cuda:1');p.add_argument('--target_device',type=int,default=0)
    p.add_argument('--drafter_device',type=int,default=1);p.add_argument('--target_gpu_memory_gib',type=int,default=8)
    p.add_argument('--dev_questions',type=int,default=10);p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--learning_rate',type=float,default=2e-4);p.add_argument('--updates_per_question',type=int,default=16)
    p.add_argument('--dynamics_updates',type=int,default=300);p.add_argument('--eval_every',type=int,default=50)
    p.add_argument('--patience',type=int,default=3);p.add_argument('--milestones',type=int,nargs='+',default=[10,20,40,60,70])
    p.add_argument('--eval_states_per_question',type=int,default=64)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--film_steps',type=int,default=200);p.add_argument('--film_examples_per_question',type=int,default=3)
    p.add_argument('--real_questions_per_fold',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if 'full' not in args.variants:raise ValueError('full baseline must be included')
    if min(args.batch_size,args.eval_every,args.updates_per_question,args.dynamics_updates,args.patience)<1:raise ValueError('Positive training limits required')
    torch.set_num_threads(2)
    run(args)
