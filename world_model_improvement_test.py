"""Paired retraining ablations + validation learning curves, using cached experience.

No environment, drafter, or verifier is constructed. Validation labels never
enter the replay. Selection of changed edges is diagnostic only, not a policy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import time
import traceback
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from offline_feature_audit import Archive, load_observations, load_embeddings
from world_model_core import (ExperienceReplay, WorldModelLearner, pack_observations,
                              expected_acceptance, prefix_log_distribution, valid_positions)
from world_model_probe import ProbeWorldModel
from world_model_probe_v2 import ImprovedProbeWorldModel, experiment_variants


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def uniform(rows, n):
    return rows if len(rows) <= n else [rows[i] for i in np.linspace(0, len(rows)-1, n, dtype=int)]


def build_plan(states, observations, edges, states_per_question=24, edges_per_action=12, paths_per_action=4):
    meta = {r['state_id']: r for r in states}
    split_q = defaultdict(set)
    for r in states:
        if r['split'] not in ('train', 'validation'):
            raise ValueError('Expected explicit train/validation split')
        split_q[r['split']].add(r['question'])
    if split_q['train'] & split_q['validation']:
        raise ValueError('Question leakage across train/validation')
    if not split_q['train'] or not split_q['validation']:
        raise ValueError('Both train and validation questions are required')
    by_q = defaultdict(list)
    for r in states:
        if observations[r['state_id']].accepted is not None:
            by_q[(r['split'], r['question'])].append(r['state_id'])
    valid_edges, train_edges, outgoing = [], [], defaultdict(list)
    seen = set()
    for e in edges:
        p, c, a = e['parent'], e['child'], e['action']
        if (p,c,a) in seen: continue
        seen.add((p,c,a))
        pm, cm = meta[p], meta[c]
        if any(pm[k] != cm[k] for k in ('split', 'question', 'round_id')):
            raise ValueError('Edge crosses a split, question, or verifier round')
        if a not in ('R', 'E'): raise ValueError('Unknown action')
        edge = [p,c,a]
        if pm['split'] == 'train': train_edges.append(edge)
        elif observations[p].accepted is not None and observations[c].accepted is not None:
            valid_edges.append(edge)
            outgoing[p].append((c,a))
    buckets = defaultdict(list)
    for e in valid_edges: buckets[(meta[e[0]]['question'],e[2])].append(e)
    panel_edges = [e for key in sorted(buckets) for e in uniform(buckets[key], edges_per_action)]
    changed = [e for e in valid_edges if observations[e[0]].accepted != observations[e[1]].accepted]
    # Real paths may start at ANY state, not only the first proposal in a round.
    # Keep every outgoing branch during enumeration (depth at most three).
    path_buckets = defaultdict(list)
    for p,c,a in valid_edges:
        paths = [([p,c],[a])]
        for _ in range(2):
            extended = []
            for nodes, actions in paths:
                children = outgoing.get(nodes[-1], [])
                if children:
                    for child, action in children:
                        if child in nodes: raise ValueError('Cycle in raw trajectories')
                        extended.append((nodes+[child], actions+[action]))
                else: extended.append((nodes,actions))
            paths = extended
        for nodes, actions in paths:
            if len(actions) >= 2:
                path_buckets[(meta[p]['question'],a)].append([nodes,actions])
    plan = dict(
        train_questions=sorted(split_q['train']), validation_questions=sorted(split_q['validation']),
        train_ids=[r['state_id'] for r in states if r['split']=='train'], train_edges=train_edges,
        train_monitor=[uid for (split,q),ids in sorted(by_q.items()) if split=='train' for uid in uniform(ids,4)],
        panel_current=[uid for (split,q),ids in sorted(by_q.items()) if split=='validation' for uid in uniform(ids,states_per_question)],
        full_current=[uid for (split,q),ids in sorted(by_q.items()) if split=='validation' for uid in ids],
        panel_edges=panel_edges, full_edges=valid_edges, changed_edges=changed,
        paths=[path for key in sorted(path_buckets) for path in uniform(path_buckets[key],paths_per_action)])
    return plan


def pack(model, obs, table, device):
    return pack_observations(obs, table, device, include_candidates=getattr(model,'needs_candidates',False))


@torch.inference_mode()
def evaluate(model, obs, plan, table, device, extension, batch_size=8, full=False):
    model.eval()
    rows = []
    def score(z, actual):
        logits = model.acceptance(z)
        means = expected_acceptance(logits, z.lengths).cpu().tolist()
        probs = prefix_log_distribution(logits,z.lengths)
        return [dict(K=o.accepted, expected_K=means[i], length=o.length,
                     nll=float(-probs[i,o.accepted])) for i,o in enumerate(actual)]
    current_sets = [('panel',plan['panel_current']), ('train_monitor',plan['train_monitor'])]
    if full: current_sets.append(('full',plan['full_current']))
    for scope, ids in current_sets:
        for start in range(0,len(ids),batch_size):
            batch = [obs[i] for i in ids[start:start+batch_size]]
            z = model.encoder(pack(model,batch,table,device))
            for o,r in zip(batch,score(z,batch)):
                rows.append(dict(r, scope=scope, group='current', question=o.question,
                                 source=o.uid, state_id=o.uid, depth=0, actions=''))
    edge_sets = [('panel',plan['panel_edges']), ('changed_diagnostic',plan['changed_edges'])]
    if full: edge_sets.append(('full',plan['full_edges']))
    for scope, edges in edge_sets:
        for start in range(0,len(edges),batch_size):
            es = edges[start:start+batch_size]
            parents, children = [obs[e[0]] for e in es], [obs[e[1]] for e in es]
            z = model.encoder(pack(model,parents,table,device))
            base = expected_acceptance(model.acceptance(z),z.lengths)
            actions = torch.tensor([int(e[2]=='E') for e in es],device=device)
            predicted = model.transition(z,actions,extension)
            b = pack(model,children,table,device)
            if not torch.equal(predicted.lengths,b['lengths']): raise ValueError('Transition length mismatch')
            actual = model.encoder(b)
            actual_K = expected_acceptance(model.acceptance(actual),actual.lengths)
            valid = valid_positions(predicted.lengths,predicted.tokens.shape[1])
            active = valid & (torch.arange(valid.shape[1],device=device)[None]>=predicted.context[:,2,None]*64)
            latent_err = 1-F.cosine_similarity(predicted.tokens,actual.tokens,dim=-1)
            latent_err = (latent_err*active).sum(-1)/active.sum(-1).clamp_min(1)
            mask_err = ((predicted.mask_probs-b['scalars'][:,:,0]).square()*active).sum(-1)/active.sum(-1).clamp_min(1)
            for i,((p,c,a),r) in enumerate(zip(es,score(predicted,children))):
                delta = obs[c].accepted-obs[p].accepted
                rows.append(dict(r,scope=scope,group='h1_'+a,question=obs[p].question,
                    source=p,state_id=c,depth=1,actions=a,source_K=obs[p].accepted,
                    source_length=obs[p].length,true_delta_K=delta,
                    predicted_delta_K=r['expected_K']-float(base[i]),
                    persistence_error=abs(float(base[i])-obs[c].accepted),
                    child_observed_prediction=float(actual_K[i]),
                    latent_cosine_error=float(latent_err[i]),mask_brier=float(mask_err[i]),
                    change='gain' if delta>0 else ('loss' if delta<0 else 'same')))
    # Open-loop rollouts: the child observations are labels only; never re-encode
    # the true child to reset the imagined state between actions.
    groups = defaultdict(list)
    for nodes,actions in plan['paths']: groups[''.join(actions)].append(nodes)
    for actions, paths in sorted(groups.items()):
        for start in range(0,len(paths),batch_size):
            batch_paths = paths[start:start+batch_size]
            roots = [obs[p[0]] for p in batch_paths]
            z = model.encoder(pack(model,roots,table,device))
            for depth,action in enumerate(actions,1):
                z = model.transition(z,torch.full((len(roots),),int(action=='E'),device=device),extension)
                if depth==1: continue
                actual = [obs[p[depth]] for p in batch_paths]
                if z.lengths.tolist()!=[o.length for o in actual]: raise ValueError('Rollout length mismatch')
                for root,o,r in zip(roots,actual,score(z,actual)):
                    rows.append(dict(r,scope='rollout_panel',group=f'h{depth}',question=o.question,
                                     source=root.uid,state_id=o.uid,depth=depth,actions=actions[:depth]))
    # Branching paths can share their depth-2 prefix. Count each observed
    # source/action-prefix/target only once in a given evaluation scope.
    return list({(r['scope'],r['group'],r['question'],r['source'],r['state_id'],r['depth'],r['actions']):r
                 for r in rows}.values())


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row['scope']+'/'+row['group']
        groups[key].append(row)
        if row['depth']==1:
            groups[key+'_'+row['change']].append(row)
            groups[key+('_L8' if row['source_length']==8 else '_other_length')].append(row)
    result = {}
    for key,values in groups.items():
        errors = np.array([abs(r['expected_K']-r['K']) for r in values])
        by_q = defaultdict(list)
        for r,e in zip(values,errors): by_q[r['question']].append(e)
        s = dict(n=len(values),questions=len(by_q),mae=float(errors.mean()),
                 question_macro_mae=float(np.mean([np.mean(v) for v in by_q.values()])),
                 p90=float(np.quantile(errors,.9)),within_1=float((errors<=1).mean()),
                 within_2=float((errors<=2).mean()),bias=float(np.mean([r['expected_K']-r['K'] for r in values])),
                 nll=float(np.mean([r['nll'] for r in values])))
        if 'persistence_error' in values[0]:
            s.update(persistence_mae=float(np.mean([r['persistence_error'] for r in values])),
                observed_child_mae=float(np.mean([abs(r['child_observed_prediction']-r['K']) for r in values])),
                delta_K_mae=float(np.mean([abs(r['predicted_delta_K']-r['true_delta_K']) for r in values])),
                latent_cosine_error=float(np.mean([r['latent_cosine_error'] for r in values])),
                mask_brier=float(np.mean([r['mask_brier'] for r in values])))
        result[key]=s
    return result


def package(out, summary):
    write_json(out/'summary.json',summary)
    dest = out.with_suffix('.zip')
    temp = dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(out),compress_type=zipfile.ZIP_STORED if p.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    temp.replace(dest)
    print(f"[zip] {summary['status']}: {dest}",flush=True)


def make_learner(config, variant, table, device, seed, extension):
    torch.manual_seed(seed)
    base = ProbeWorldModel(**config)
    torch.manual_seed(seed+10000)
    model = ImprovedProbeWorldModel(**config,candidate_attention=variant['candidate_attention'],
        residual_dynamics=variant['residual_dynamics'],drop_feature=variant['drop_feature'])
    # Every shared tensor starts identically, despite different new modules.
    model.load_state_dict(base.state_dict(),strict=False)
    learner = WorldModelLearner(model,table,device,extension_size=extension,
        latent_weight=variant['latent_weight'],teacher_weight=variant['teacher_weight'],
        structure_weight=variant['structure_weight'],delta_weight=variant['delta_weight'],
        warmup_updates=16,horizon_warmup=64)
    torch.manual_seed(seed+20000)
    return learner


def run(args):
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    summary = dict(status='preparing',llm_forward_calls=0,config=vars(args),completed_runs=[],points=[])
    summary['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path(__file__).with_name('world_model_probe_v2.py'),
        Path(__file__).with_name('world_model_probe.py'),Path(__file__).with_name('world_model_core.py'),
        Path(__file__).with_name('offline_feature_audit.py'),Path(__file__).with_name('world_model_improvement_report.py'))}
    summary['runtime']=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        device=str(args.device),device_name=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'CPU')
    try:
        archive = Archive(args.input)
        checkpoint_bytes=archive.read('checkpoint.pt')
        summary['input_checkpoint_sha256']=hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
        config = checkpoint['model_config']
        summary['model_config']=config
        summary['optimizer_config']=dict(learning_rate=3e-4,weight_decay=.01,ema_decay=.99,warmup_updates=16,horizon_warmup_updates=64,horizon=3)
        states = archive.lines('states.jsonl')
        labels = {r['state_id']:r for r in archive.lines('labels.jsonl')}
        print('[data] restoring cached states and late verifier teacher targets',flush=True)
        observations = load_observations(archive,states,labels,include_teacher=True)
        plan = build_plan(states,observations,archive.lines('edges.jsonl'),args.panel_states,args.panel_edges,args.panel_paths)
        write_json(out/'evaluation_plan.json',plan)
        summary.update(train_questions=len(plan['train_questions']),validation_questions=len(plan['validation_questions']),
            teacher_train=sum(observations[i].teacher_margin is not None for i in plan['train_ids']),
            teacher_validation=sum(observations[i].teacher_margin is not None for i in plan['full_current']),
            full_validation_edges=len(plan['full_edges']),changed_validation_edges=len(plan['changed_edges']),
            plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
        variants = experiment_variants()
        names = args.variants.split(',') if args.variants!='all' else list(variants)
        if any(n not in variants for n in names): raise ValueError('Unknown experiment variant')
        if summary['teacher_train']==0 and any(variants[n]['teacher_weight'] for n in names):
            raise ValueError('No training teacher targets: cannot measure teacher effect')
        table,emb_meta = load_embeddings(args.embeddings,args.embedding_repo,args.embedding_revision,args.cache)
        if args.embedding_sha256 and emb_meta['tensor_sha256']!=args.embedding_sha256:
            raise ValueError('Embedding hash differs from verified training model')
        table = table.to(args.device)
        summary.update(embedding=emb_meta,variants={n:variants[n] for n in names},status='running')
        steps = [int(s) for s in args.steps.split(',')]
        seeds = [int(s) for s in args.seeds.split(',')]
        extension = int(checkpoint['args'].get('extend_size',8))
        # Identical chronological raw state/edge insertion and RNG seed per arm.
        for seed in seeds:
            for name in names:
                learner = make_learner(config,variants[name],table,args.device,seed,extension)
                replay = ExperienceReplay(len(plan['train_ids']),seed)
                for uid in plan['train_ids']: replay.add_node(observations[uid])
                for p,c,a in plan['train_edges']: replay.add(observations[p],observations[c],a)
                run_dir=out/name/str(seed); run_dir.mkdir(parents=True)
                train_seconds=0.
                if str(args.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
                for target in steps:
                    with (run_dir/'losses.jsonl').open('a',encoding='utf-8') as log:
                        while learner.updates<target:
                            t=time.monotonic()
                            metric=learner.update(replay,args.batch_size,3)
                            if metric is None: raise RuntimeError('No train signal; refusing stalled update loop')
                            train_seconds+=time.monotonic()-t
                            log.write(json.dumps(metric,allow_nan=False)+'\n')
                            if learner.updates%64==0:
                                print(f'[train] {name} seed={seed} update={learner.updates} loss={metric["loss"]:.3f}',flush=True)
                    t=time.monotonic()
                    rows=evaluate(learner.model,observations,plan,table,args.device,extension,args.eval_batch_size,full=target==steps[-1])
                    metrics=summarize(rows)
                    pred_file=run_dir/f'predictions_{target:05d}.jsonl.gz'
                    with gzip.open(pred_file,'wt',encoding='utf-8') as f:
                        for r in rows: f.write(json.dumps(r,allow_nan=False)+'\n')
                    point=dict(variant=name,seed=seed,update=target,train_seconds=train_seconds,
                        eval_seconds=time.monotonic()-t,wall_seconds=time.monotonic()-started,
                        parameters=sum(p.numel() for p in learner.model.parameters()),metrics=metrics,
                        predictions=str(pred_file.relative_to(out)),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None)
                    summary['points'].append(point)
                    print(f'[eval] {name} seed={seed} update={target} current_MAE={metrics["panel/current"]["mae"]:.3f}',flush=True)
                    # Durable partial ZIP after every completed evaluation.
                    summary['elapsed_seconds']=time.monotonic()-started
                    package(out,summary)
                if name in ('baseline','improved'):
                    torch.save(dict(model=learner.model.state_dict(),model_config=learner.model.config,
                        variant=variants[name],seed=seed,updates=learner.updates),run_dir/'final_model.pt')
                summary['completed_runs'].append(dict(variant=name,seed=seed))
                del learner,replay
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        summary.update(status='complete',elapsed_seconds=time.monotonic()-started)
        summary['total_optimizer_updates']=len(seeds)*len(names)*steps[-1]
        from world_model_improvement_report import create_report
        create_report(out,summary)
    except BaseException:
        summary.update(status='partial_error',elapsed_seconds=time.monotonic()-started)
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        package(out,summary)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--variants',default='all'); p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--steps',default='0,64,128,256,512')
    p.add_argument('--panel-states',type=int,default=24); p.add_argument('--panel-edges',type=int,default=12)
    p.add_argument('--panel-paths',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--embeddings'); p.add_argument('--cache',default='/kaggle/temp/wm_audit_hf')
    p.add_argument('--embedding-repo',default='Efficient-Large-Model/Fast_dLLM_v2_1.5B')
    p.add_argument('--embedding-revision',default='25093b6f63300adfd57f72145083c8a528fe4f16')
    p.add_argument('--embedding-sha256',default='a1d834b8219ee8040d8a32a019e166d10c5bb0546d6d2e2e9b2a0041b8a2e73b')
    p.add_argument('--trust-checkpoint',action='store_true')
    a=p.parse_args()
    if not a.trust_checkpoint: p.error('Only load a trusted training archive')
    steps=[int(s) for s in a.steps.split(',')]
    if steps!=sorted(set(steps)) or steps[0]!=0 or steps[-1]<1: p.error('Steps must start at 0 and increase')
    if min(a.panel_states,a.panel_edges,a.panel_paths,a.batch_size,a.eval_batch_size)<1: p.error('Counts must be positive')
    run(a)


if __name__=='__main__': main()
