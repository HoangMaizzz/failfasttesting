"""Phase 0: maximal native observation -> frozen token latent -> R/E -> V.

Offline traces only. No token-generation objective, no random identity decoder,
no verifier gradients through the dynamics. Stage gates are chosen on validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from factorized_wm_data import feature_registry
from factorized_wm_metrics import write_json, write_jsonl, verifier_report
from phase0_wm_data import (load_dataset, enumerate_paths, split_questions,
    prepare_embedding, NativeTargets, prepare_rows, pack_observations,
    pack_latents, pack_native_targets, dataset_digest)
from phase0_wm_models import (TokenEmbedding, LatentEncoder, VerifierReadout,
    RawObservationVerifier, NativeReconstruction, DynamicsPair, PriorDynamics,
    DirectOutcome, verifier_loss, native_loss, latent_loss, rollout,
    valid_positions, slice_latent)


def log(stage, **values):
    print('[phase0] ' + json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def package(output):
    """Atomic working-root ZIP, including completed checkpoints on failure."""
    output = Path(output); final = output.with_suffix('.zip'); temporary = final.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.tmp' and not path.is_symlink():
                if any(p in ('_cache', 'experience', 'input_cache', 'hf_cache', '__pycache__') for p in path.relative_to(output).parts):
                    continue
                archive.write(path, output.name + '/' + path.relative_to(output).as_posix())
    temporary.replace(final)
    log('archive', file=final.name, MiB=round(final.stat().st_size / 2**20, 2))
    return final


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sampler(items, key, seed):
    """Question-uniform sampling; preserve each question's natural state mixture."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    if not groups:
        raise ValueError('No training examples for stage')
    questions = sorted(groups); rng = random.Random(seed)
    def sample(count):
        return [rng.choice(groups[rng.choice(questions)]) for _ in range(count)]
    return sample


def heads_to_rows(heads, rows):
    """Teachers join only AFTER model forward; expected_yield retains metric API name."""
    q = heads['hazard'].detach().sigmoid().clamp(1e-6, 1-1e-6).cpu()
    survival = q.cumprod(-1)
    result = []
    for i, row in enumerate(rows):
        n = row['length']; qq = q[i, :n]; ss = survival[i, :n]
        pmf = torch.cat([1-qq[:1], ss[:-1] * (1-qq[1:]), ss[-1:]])
        teacher = row['teacher']
        result.append(dict(question=row['question'], uid=row['uid'], length=n,
            accepted=row['accepted'], expected_yield=float(ss.sum()), mode=int(pmf.argmax()),
            hazards=qq.tolist(), tf_probs=heads['tf'][i, :n].detach().sigmoid().cpu().tolist(),
            probability_pred=heads['probability'][i, :n].detach().cpu().tolist(),
            margin_pred=heads['margin'][i, :n].detach().cpu().tolist(),
            tf_truth=None if teacher is None else teacher[:, 2].tolist(),
            probability_truth=None if teacher is None else teacher[:, 1].tolist(),
            margin_truth=None if teacher is None else teacher[:, 0].tolist()))
    return result


@torch.no_grad()
def observation_predictions(rows, cfg, model=None, encoder=None, readout=None):
    device = cfg['device_encoder']; results = []
    for start in range(0, len(rows), cfg['batch_size']):
        group = rows[start:start + cfg['batch_size']]
        batch = pack_observations(group, device, cfg['prefix_max_tokens'])
        heads = model(batch) if model is not None else readout(encoder(batch).z, batch.lengths)
        results.extend(heads_to_rows(heads, group))
    return results


def macro_mae(predictions):
    groups = defaultdict(list)
    for r in predictions:
        if r['accepted'] is not None:
            groups[r['question']].append(abs(r['expected_yield'] - r['accepted']))
    return float(np.mean([np.mean(x) for x in groups.values()])) if groups else float('inf')


def validation_score(predictions):
    # Scalar K alone can select an acceptance-only shortcut; include survival.
    report = verifier_report(predictions)['all']
    brier = report['survival']['metrics']['brier']
    return macro_mae(predictions) + 4 * (0 if brier is None else brier)


def fit_observation(rows, train, val, cfg, folder, embedding, dim=None):
    folder.mkdir(parents=True, exist_ok=True); checkpoint = folder / 'best.pt'
    raw = dim is None
    seed_all(cfg['seed'] + (0 if raw else dim))
    if raw:
        model = RawObservationVerifier(TokenEmbedding(**embedding), dropout=cfg['dropout'])
        modules = {'model': model}
    else:
        encoder = LatentEncoder(TokenEmbedding(**embedding), dim, cfg['encoder_layers'], cfg['dropout'])
        readout = VerifierReadout(dim, 1, cfg['dropout'])
        recon = NativeReconstruction(dim, rows[train[0]]['native_target'].shape[1])
        modules = {'encoder': encoder, 'readout': readout, 'reconstruction': recon}
    device = cfg['device_encoder']
    for m in modules.values():
        m.to(device)
    if (folder / 'complete.json').is_file():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for name, model in modules.items():
            model.load_state_dict(saved[name]); model.eval().requires_grad_(False)
        log('resume_stage', stage_name=folder.name)
        return modules
    parameters = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=.01)
    examples = [rows[u] for u in train]
    sample = sampler(examples, lambda r: r['question'], cfg['seed'])
    budget = cfg['raw_updates'] if raw else cfg['stage_a_updates']
    best = float('inf'); bad = 0; curve = []; started = time.perf_counter()
    validation = [rows[u] for u in val]
    for step in range(1, budget + 1):
        for model in modules.values():
            model.train()
        group = sample(cfg['batch_size'])
        batch = pack_observations(group, device, cfg['prefix_max_tokens'])
        if raw:
            loss = verifier_loss(modules['model'](batch), group, device)
        else:
            state = modules['encoder'](batch)
            loss = verifier_loss(modules['readout'](state.z, state.lengths), group, device)
            target, mask = pack_native_targets(group, device)
            loss = loss + cfg['reconstruction_weight'] * native_loss(modules['reconstruction'](state), target, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite observation loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1); optimizer.step()
        if step % cfg['eval_every'] == 0 or step == budget:
            for model in modules.values():
                model.eval()
            prediction = observation_predictions(validation, cfg, **({'model': modules['model']} if raw else
                {'encoder': modules['encoder'], 'readout': modules['readout']}))
            score = validation_score(prediction)
            improved = score < best
            if improved:
                best = score; bad = 0
                save_torch(checkpoint, {name: cpu_weights(m) for name, m in modules.items()})
            else:
                bad += 1
            item = dict(step=step, loss=float(loss.detach()), validation_score=score,
                validation_K_macro_MAE=macro_mae(prediction), selected=improved,
                elapsed_seconds=time.perf_counter() - started)
            curve.append(item); write_json(folder / 'learning_curve.json', curve)
            log('raw_reference' if raw else 'stage_A', latent_dim=dim, **item)
            if step >= cfg['minimum_updates'] and bad >= cfg['patience']:
                break
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for name, model in modules.items():
        model.load_state_dict(saved[name]); model.eval().requires_grad_(False)
    write_json(folder / 'complete.json', dict(updates=step, seconds=time.perf_counter()-started,
        best_validation_score=best, frozen=True))
    return modules


@torch.no_grad()
def encode_cache(rows, encoder, cfg):
    cache = {}; device = cfg['device_encoder']; ids = sorted(rows)
    for start in range(0, len(ids), cfg['batch_size']):
        group = [rows[u] for u in ids[start:start + cfg['batch_size']]]
        state = encoder(pack_observations(group, device, cfg['prefix_max_tokens']))
        for i, row in enumerate(group):
            cache[row['uid']] = state.z[i].half().cpu()
        if start % 1600 == 0:
            log('encode_frozen_latents', completed=start + len(group), total=len(ids))
    return cache


@torch.no_grad()
def cached_predictions(rows, cache, readout, cfg):
    result = []
    for start in range(0, len(rows), cfg['batch_size']):
        group = rows[start:start + cfg['batch_size']]
        state = pack_latents(group, cache, cfg['device_encoder'])
        result.extend(heads_to_rows(readout(state.z, state.lengths), group))
    return result


def select_paths(dataset, split, include_contradictions=False):
    paths = enumerate_paths(dataset, split, horizon=3)
    if include_contradictions:
        return paths
    bad = {(r['parent'], r['child']) for r in dataset.audit['immutable_prefix_K_contradictions']}
    return [p for p in paths if not any(pair in bad for pair in zip(p[0], p[0][1:]))]


def challenge_ids(paths, rows):
    groups = {'R_K_changed': set(), 'E_full_prefix': set()}
    for nodes, actions in paths:
        if len(actions) != 1:
            continue
        a, b = rows[nodes[0]], rows[nodes[-1]]
        if a['accepted'] is None or b['accepted'] is None:
            continue
        if actions[0] == 0 and a['accepted'] != b['accepted']:
            groups['R_K_changed'].add(b['uid'])
        if actions[0] == 1 and a['accepted'] == a['length']:
            groups['E_full_prefix'].add(b['uid'])
    return groups


def stage_a_gate(raw_predictions, latent_predictions, cfg, cohorts):
    def check(raw, latent):
        a, b = verifier_report(raw)['all'], verifier_report(latent)['all']
        am, bm = a['k']['question_macro_mae'], b['k']['question_macro_mae']
        aa, ba = a['survival']['metrics']['roc_auc'], b['survival']['metrics']['roc_auc']
        ab, bb = a['survival']['metrics']['brier'], b['survival']['metrics']['brier']
        ae, be = a['survival']['metrics']['ece'], b['survival']['metrics']['ece']
        checks = dict(K_MAE=(am is not None and bm is not None and bm <= max(am, .1)*cfg['gate_mae_ratio']),
            survival_AUC=(aa is not None and ba is not None and ba >= aa-cfg['gate_auc_tolerance']),
            survival_Brier=(ab is not None and bb is not None and bb <= max(ab,.01)*cfg['gate_brier_ratio']),
            survival_ECE=(ae is not None and be is not None and be <= ae+cfg['gate_ece_tolerance']))
        return dict(pass_gate=all(checks.values()), checks=checks, raw_K_macro_MAE=am,
                    latent_K_macro_MAE=bm, count=len(latent))
    result = {'all': check(raw_predictions, latent_predictions)}
    for name, ids in cohorts.items():
        a = [r for r in raw_predictions if r['uid'] in ids]
        b = [r for r in latent_predictions if r['uid'] in ids]
        # An absent/small cohort is insufficient evidence, not an automatic pass.
        result[name] = check(a, b) if len(b) >= 5 else dict(pass_gate=False, status='insufficient_cohort', count=len(b))
    result['pass_gate'] = all(v['pass_gate'] for v in result.values())
    result['selection_split'] = 'validation'
    result['scope'] = 'Relative information retention, not proof of absolute verifier adequacy'
    return result


def extension_prior(paths, rows, cache):
    pieces = []
    for nodes, actions in paths:
        if actions == (1,):
            start = rows[nodes[0]]['length']
            pieces.append(cache[nodes[-1]][start:start + 8].float())
    if not pieces:
        raise ValueError('No train E edges for new-position prior')
    return torch.stack(pieces).mean(0)


@torch.no_grad()
def evaluate_composition(paths, rows, cache, readout, dynamics, prior, direct, source_predictions, cfg):
    device = cfg['device_dynamics']; vd = cfg['device_encoder']; result = []
    readout.eval(); dynamics.eval(); prior.eval()
    if direct is not None:
        direct.eval()
    for horizon in (1, 2, 3):
        pool = [p for p in paths if len(p[1]) == horizon]
        for start in range(0, len(pool), cfg['batch_size']):
            group = pool[start:start + cfg['batch_size']]
            sequence = [p[1] for p in group]
            source_rows = [rows[p[0][0]] for p in group]
            target_rows = [rows[p[0][-1]] for p in group]
            source = pack_latents(source_rows, cache, device)
            real = pack_latents(target_rows, cache, device)
            imagined = rollout(dynamics, source, sequence)
            copied = rollout(prior, source, sequence)
            oracle = [source_predictions[r['uid']] for r in target_rows]
            ip = heads_to_rows(readout(imagined.z.to(vd), imagined.lengths.to(vd)), target_rows)
            pp = heads_to_rows(readout(copied.z.to(vd), copied.lengths.to(vd)), target_rows)
            dp = None if direct is None else heads_to_rows(direct(source.to(vd), sequence), target_rows)
            for i, ((nodes, actions), a, b) in enumerate(zip(group, source_rows, target_rows)):
                n = b['length']
                delta = None if a['accepted'] is None or b['accepted'] is None else b['accepted'] - a['accepted']
                item = dict(question=a['question'], parent=nodes[0], child=nodes[-1], horizon=horizon,
                    actions=''.join('E' if x else 'R' for x in actions), action='E' if actions[-1] else 'R',
                    parent_length=a['length'], length=n, parent_accepted=a['accepted'], accepted=b['accepted'],
                    parent_token_changed=bool((a['ids'][:, 1] != b['ids'][:a['length'], 1]).any()),
                    changed_positions=torch.where(a['ids'][:, 1] != b['ids'][:a['length'], 1])[0].tolist(),
                    native_active_start=int(round(float(b['context'][2]) * 64)),
                    delta_true=delta,
                    latent_mse=float((imagined.z[i,:n] - real.z[i,:n]).square().mean()),
                    latent_cosine=float(F.cosine_similarity(imagined.z[i,:n], real.z[i,:n], dim=-1).mean()))
                for name, predictions in (('oracle', oracle), ('imagined', ip), ('prior', pp), ('direct', dp)):
                    if predictions is None:
                        continue
                    item[name + '_verifier'] = predictions[i]
                    item[name + '_source_K'] = source_predictions[a['uid']]['expected_yield']
                result.append(item)
    return result


def h1_gate(predictions, cfg):
    groups = {
        'R_K_changed': [r for r in predictions if r['horizon']==1 and r['action']=='R' and r['delta_true'] not in (None,0)],
        'E_full_prefix': [r for r in predictions if r['horizon']==1 and r['action']=='E' and r['parent_accepted']==r['parent_length']],
    }
    result = {}
    for name, rr in groups.items():
        rr = [r for r in rr if r['delta_true'] is not None]
        if len(rr)<5:
            result[name]=dict(pass_gate=False,status='insufficient_cohort',count=len(rr)); continue
        def error(model):
            byq=defaultdict(list)
            for r in rr:
                predicted=r[model+'_verifier']['expected_yield']-r[model+'_source_K']
                byq[r['question']].append(abs(predicted-r['delta_true']))
            return float(np.mean([np.mean(x) for x in byq.values()]))
        learned, prior_error = error('imagined'), error('prior')
        byq=defaultdict(list)
        for r in rr:
            byq[r['question']].append(abs(r['imagined_verifier']['expected_yield']-r['oracle_verifier']['expected_yield']))
        gap=float(np.mean([np.mean(x) for x in byq.values()]))
        result[name]=dict(count=len(rr),question_count=len(byq),imagined_delta_MAE=learned,
            prior_delta_MAE=prior_error,paired_oracle_K_gap=gap,
            pass_gate=learned<=prior_error*cfg['h1_delta_improvement_ratio'] and gap<=cfg['h1_oracle_gap_tokens'])
    result['pass_gate']=all(v['pass_gate'] for v in result.values())
    result['selection_split']='validation';result['scope']='Engineering gate; confirmation requires paired CI and new questions'
    return result


def fit_dynamics(rows, cache, train_paths, val_paths, cfg, folder, dim, recon, readout, prior, predictions):
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'] + dim + 100)
    model = DynamicsPair(dim, cfg['dynamics_layers'], cfg['dropout']).to(cfg['device_dynamics'])
    recon.to(cfg['device_dynamics']).eval().requires_grad_(False)
    gates = {}; completed_horizons=[]
    for horizon in (1,2,3):
        if horizon>1 and not gates['H1']['pass_gate'] and not cfg['continue_diagnostics']:
            log('skip_higher_horizons', latent_dim=dim, reason='H1 validation gate not passed')
            break
        stage = folder / f'H{horizon}'; stage.mkdir(exist_ok=True)
        if (stage/'complete.json').is_file():
            model.load_state_dict(torch.load(stage/'best.pt',map_location='cpu',weights_only=True))
            summary=json.loads((stage/'complete.json').read_text())
            if horizon==1:gates['H1']=summary['gate']
            completed_horizons.append(horizon);continue
        pool=[p for p in train_paths if len(p[1])==horizon]
        vv=[p for p in val_paths if len(p[1])==horizon]
        if not pool or not vv:
            log('skip_horizon', horizon=horizon, reason='missing train/validation paths');break
        # Each batch contains ONE action sequence. Question sampling stays uniform.
        sequences=sorted({p[1] for p in pool}); rng=random.Random(cfg['seed']+horizon)
        samples={seq:sampler([p for p in pool if p[1]==seq],lambda p:rows[p[0][0]]['question'],cfg['seed']+horizon)
                 for seq in sequences}
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
        best=float('inf');bad=0;curve=[];started=time.perf_counter()
        for step in range(1,cfg['dynamics_updates_per_horizon']+1):
            model.train(); group=samples[rng.choice(sequences)](cfg['batch_size'])
            state=pack_latents([rows[p[0][0]] for p in group],cache,cfg['device_dynamics'])
            loss=state.z.sum()*0
            for depth in range(horizon):
                action=torch.tensor([p[1][depth] for p in group],device=state.z.device)
                state=model(state,action)
                target_rows=[rows[p[0][depth+1]] for p in group]
                truth=pack_latents(target_rows,cache,state.z.device)
                native_target,native_mask=pack_native_targets(target_rows,state.z.device)
                loss=loss+latent_loss(state,truth,recon,native_target,native_mask)
            loss=loss/horizon
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite dynamics loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if step%cfg['eval_every']==0 or step==cfg['dynamics_updates_per_horizon']:
                model.eval(); scores=[]
                with torch.no_grad():
                    for start in range(0,len(vv),cfg['batch_size']):
                        g=vv[start:start+cfg['batch_size']]
                        source=pack_latents([rows[p[0][0]] for p in g],cache,cfg['device_dynamics'])
                        pred=rollout(model,source,[p[1] for p in g])
                        target=[rows[p[0][-1]] for p in g];truth=pack_latents(target,cache,pred.z.device)
                        nt,nm=pack_native_targets(target,pred.z.device)
                        scores.append(float(latent_loss(pred,truth,recon,nt,nm)))
                score=float(np.mean(scores));improved=score<best
                if improved:
                    best=score;bad=0;save_torch(stage/'best.pt',cpu_weights(model))
                else:bad+=1
                item=dict(step=step,horizon=horizon,loss=float(loss.detach()),validation_D_loss=score,
                    elapsed_seconds=time.perf_counter()-started,selected=improved)
                curve.append(item);write_json(stage/'learning_curve.json',curve);log('stage_BC',latent_dim=dim,**item)
                if step>=cfg['minimum_updates'] and bad>=cfg['patience']:break
        model.load_state_dict(torch.load(stage/'best.pt',map_location='cpu',weights_only=True));model.eval()
        gate=None
        if horizon==1:
            validation=evaluate_composition(vv,rows,cache,readout,model,prior,None,predictions,cfg)
            gate=h1_gate(validation,cfg);gates['H1']=gate
            write_jsonl(stage/'validation_predictions.jsonl',validation)
        write_json(stage/'complete.json',dict(updates=step,seconds=time.perf_counter()-started,
            best_validation_D_loss=best,gate=gate,supervision='D-only; no G gradients'))
        completed_horizons.append(horizon)
        package(folder.parent.parent)
    return model.eval().requires_grad_(False),completed_horizons,gates


def fit_direct(rows, cache, train_paths, val_paths, cfg, folder, dim):
    folder.mkdir(parents=True,exist_ok=True);model=DirectOutcome(dim,cfg['dropout']).to(cfg['device_encoder'])
    if (folder/'complete.json').is_file():
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));return model.eval().requires_grad_(False)
    groups=defaultdict(list)
    for p in train_paths:
        target=rows[p[0][-1]]
        if target['accepted'] is not None or target['teacher'] is not None:groups[p[1]].append(p)
    if not groups:raise ValueError('No direct outcome targets')
    samples={seq:sampler(pool,lambda p:rows[p[0][0]]['question'],cfg['seed']) for seq,pool in groups.items()}
    rng=random.Random(cfg['seed']);seqs=sorted(groups)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    best=float('inf');bad=0;curve=[];started=time.perf_counter()
    for step in range(1,cfg['direct_updates']+1):
        model.train();group=samples[rng.choice(seqs)](cfg['batch_size'])
        source=pack_latents([rows[p[0][0]] for p in group],cache,cfg['device_encoder'])
        target=[rows[p[0][-1]] for p in group]
        loss=verifier_loss(model(source,[p[1] for p in group]),target,source.z.device)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite direct loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%cfg['eval_every']==0 or step==cfg['direct_updates']:
            model.eval();result=[]
            with torch.no_grad():
                for h in (1,2,3):
                    vv=[p for p in val_paths if len(p[1])==h]
                    for start in range(0,len(vv),cfg['batch_size']):
                        g=vv[start:start+cfg['batch_size']]
                        source=pack_latents([rows[p[0][0]] for p in g],cache,cfg['device_encoder'])
                        result+=heads_to_rows(model(source,[p[1] for p in g]),[rows[p[0][-1]] for p in g])
            score=validation_score(result);improved=score<best
            if improved:best=score;bad=0;save_torch(folder/'best.pt',cpu_weights(model))
            else:bad+=1
            item=dict(step=step,loss=float(loss.detach()),validation_score=score,
                selected=improved,elapsed_seconds=time.perf_counter()-started)
            curve.append(item);write_json(folder/'learning_curve.json',curve);log('direct_baseline',latent_dim=dim,**item)
            if step>=cfg['minimum_updates'] and bad>=cfg['patience']:break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
    write_json(folder/'complete.json',dict(updates=step,seconds=time.perf_counter()-started))
    return model.eval().requires_grad_(False)


def validate_config(cfg):
    if cfg['num_questions']<10:raise ValueError('Need >=10 questions for grouped split')
    if not cfg['latent_dims'] or any(d%4 or d<16 for d in cfg['latent_dims']):raise ValueError('Latent dims must be >=16 and divisible by4')
    if cfg['max_proposal_tokens']!=64 or cfg['extend_size']!=8 or cfg['horizon']!=3:raise ValueError('Phase0 contract is max64, E8, H1-H3')
    for name in ('stage_a_updates','raw_updates','direct_updates','dynamics_updates_per_horizon','eval_every','batch_size'):
        if cfg[name]<1:raise ValueError(f'{name} must be positive')
    if cfg['prefix_max_tokens']<1:raise ValueError('Prefix window must be positive')
    if len(set(cfg['latent_dims']))!=len(cfg['latent_dims']):raise ValueError('Duplicate latent widths')


def source_hashes():
    root=Path(__file__).resolve().parent
    names=['run_latent_wm_phase0.py','phase0_wm_models.py','phase0_wm_data.py','phase0_wm_metrics.py',
           'factorized_wm_data.py','factorized_wm_metrics.py']
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    summary=dict(schema='latent_world_model_phase0_v1',status='running',LLM_forwards=0,widths={})
    try:
        cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));validate_config(cfg)
        torch.set_num_threads(min(4,os.cpu_count() or 1));seed_all(cfg['seed'])
        for device in (cfg['device_encoder'],cfg['device_dynamics']):
            if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA device requested but unavailable')
        cache=Path(cfg['cache_dir']) if cfg.get('cache_dir') else Path(tempfile.mkdtemp(prefix='phase0_cache_'))
        log('load_original_trace',input=str(args.input));dataset=load_dataset(args.input,max_questions=cfg['num_questions'],cache_dir=cache/'raw')
        if len(dataset.question_ids)!=cfg['num_questions']:raise ValueError('Input has fewer questions than requested')
        split=split_questions(dataset.question_ids,cfg['seed'])
        hashes=source_hashes();digest=dataset_digest(dataset)
        fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,source_hashes=hashes,data=digest),sort_keys=True).encode()).hexdigest()
        manifest_path=output/'study_manifest.json'
        if args.resume:
            old=json.loads(manifest_path.read_text())
            if old['fingerprint']!=fingerprint:raise ValueError('Resume source/config/data differs; use original pinned commit/config')
        elif manifest_path.exists():raise FileExistsError('Output already contains study; choose --resume or new output')
        write_json(manifest_path,dict(fingerprint=fingerprint,config=cfg,source=dataset.source,status='running'))
        write_json(output/'config.json',cfg);write_json(output/'source_hashes.json',hashes)
        write_json(output/'split_manifest.json',split);write_json(output/'dataset_audit.json',dataset.audit)
        write_json(output/'feature_schema.json',dict(registry=feature_registry(),
            encoder_inputs=['hidden7/14/28','native_and_STOP_ID_embeddings','ranked_top32_ID_embeddings_and_gaps',
                'scalars16','history4','context8','full_ordered_prefix_ID_embeddings'],
            verifier_inputs='Z and padding length only',token_decoder=False,random_identity_decoder=False,
            latent_state=dict(z='64 x d learned token latent', c='20 native fields, predicted on rollout',
                context='8 known structural fields advanced from source and actions'),
            dynamics_targets='frozen E(child), native reconstruction and structural targets',
            current_verifier_truth_is_input=False,full_vocab_probability_from_topK=False))
        package(output)
        if (output/'preprocessing.pt').is_file():
            preprocessing=torch.load(output/'preprocessing.pt',map_location='cpu',weights_only=True)
            embedding=preprocessing['embedding'];targets=NativeTargets(**preprocessing['native_targets'])
        else:
            log('prepare_native_embedding',mode=cfg['token_embedding'])
            embedding,embedding_report=prepare_embedding(dataset,split['train'],cfg,cache/'hf',cfg['device_encoder'])
            write_json(output/'embedding_provenance.json',embedding_report)
            targets=NativeTargets.fit(dataset,split['train'],cfg['reconstruction_hidden_dim'],cfg['reconstruction_fit_rows'],cfg['device_encoder'],cfg['seed'])
            save_torch(output/'preprocessing.pt',dict(embedding=embedding,native_targets=vars(targets)))
        rows=prepare_rows(dataset,targets)
        ids={name:sorted(u for u,r in rows.items() if r['question'] in set(q)) for name,q in split.items()}
        paths={name:select_paths(dataset,q,include_contradictions=name!='train') for name,q in split.items()}
        train_labeled=[u for u in ids['train'] if rows[u]['accepted'] is not None or rows[u]['teacher'] is not None]
        write_json(output/'path_counts.json',{name:{str(h):sum(len(p[1])==h for p in pp) for h in (1,2,3)} for name,pp in paths.items()})
        raw=fit_observation(rows,train_labeled,ids['val'],cfg,output/'raw_reference',embedding)['model']
        raw_predictions={name:observation_predictions([rows[u] for u in uu],cfg,model=raw) for name,uu in ids.items() if name!='train'}
        for name,pred in raw_predictions.items():
            write_json(output/f'raw_reference_{name}.json',verifier_report(pred));write_jsonl(output/f'raw_reference_{name}_predictions.jsonl',pred)
        raw.cpu();release();package(output)
        from phase0_wm_metrics import write_reports
        for dim in cfg['latent_dims']:
            width=output/f'latent_{dim}';width.mkdir(exist_ok=True)
            if (width/'complete.json').is_file():
                summary['widths'][str(dim)]=json.loads((width/'complete.json').read_text());continue
            modules=fit_observation(rows,train_labeled,ids['val'],cfg,width/'stage_A',embedding,dim)
            encoder,readout,recon=modules['encoder'],modules['readout'],modules['reconstruction']
            cachez=encode_cache(rows,encoder,cfg)
            # Cache is a compact derived artifact and makes the freeze auditable.
            save_torch(width/'frozen_latents.pt',cachez)
            latent_predictions=cached_predictions(list(rows.values()),cachez,readout,cfg)
            prediction_map={r['uid']:r for r in latent_predictions}
            latent_val=[prediction_map[u] for u in ids['val']]
            gate=stage_a_gate(raw_predictions['val'],latent_val,cfg,challenge_ids(paths['val'],rows))
            write_json(width/'stage_A_gate.json',gate)
            for name in ('val','test'):
                pred=[prediction_map[u] for u in ids[name]]
                write_json(width/f'real_latent_{name}.json',verifier_report(pred));write_jsonl(width/f'real_latent_{name}_predictions.jsonl',pred)
            encoder.cpu();release();package(output)
            if not gate['pass_gate'] and not cfg['continue_diagnostics']:
                item=dict(status='stage_A_gate_failed',higher_stages_skipped=True,stage_A_gate=gate)
                write_json(width/'complete.json',item);summary['widths'][str(dim)]=item
                log('skip_dynamics',latent_dim=dim,reason='Stage A validation gate failed')
                for model in (encoder,readout,recon):model.cpu()
                del cachez,prediction_map,latent_predictions,modules
                release();package(output);continue
            prior=PriorDynamics(extension_prior(paths['train'],rows,cachez)).to(cfg['device_dynamics'])
            dynamics,completed,gates=fit_dynamics(rows,cachez,paths['train'],paths['val'],cfg,width/'dynamics',dim,recon,readout,prior,prediction_map)
            if not completed:
                item=dict(status='insufficient_paths',completed_horizons=[],stage_A_gate=gate,
                    higher_stages_skipped=True)
                write_json(width/'complete.json',item);summary['widths'][str(dim)]=item
                for model in (encoder,readout,recon,dynamics,prior):model.cpu()
                del cachez,prediction_map,latent_predictions,modules
                release();package(output);continue
            allowed={h for h in completed}
            train_paths=[p for p in paths['train'] if len(p[1]) in allowed]
            val_paths=[p for p in paths['val'] if len(p[1]) in allowed]
            direct=fit_direct(rows,cachez,train_paths,val_paths,cfg,width/'direct',dim)
            for name in ('val','test'):
                pool=[p for p in paths[name] if len(p[1]) in allowed]
                results=evaluate_composition(pool,rows,cachez,readout,dynamics,prior,direct,prediction_map,cfg)
                destination=width/name;destination.mkdir(exist_ok=True)
                write_jsonl(destination/'composition_predictions.jsonl',results)
                write_reports(results,destination,bootstrap_samples=cfg['bootstrap_samples'],seed=cfg['seed'])
            item=dict(status='complete',completed_horizons=completed,stage_A_gate=gate,dynamics_gates=gates,
                continued_after_failed_gate=cfg['continue_diagnostics'] and (not gate['pass_gate'] or not gates['H1']['pass_gate']))
            write_json(width/'complete.json',item);summary['widths'][str(dim)]=item
            for model in (encoder,readout,recon,dynamics,prior,direct):model.cpu()
            del cachez,prediction_map,latent_predictions,modules;release();package(output)
        summary.update(status='complete',questions=len(dataset.question_ids),split={k:len(v) for k,v in split.items()},
            elapsed_seconds=time.perf_counter()-started)
        write_json(output/'summary.json',summary)
        manifest=json.loads(manifest_path.read_text());manifest['status']='complete';write_json(manifest_path,manifest)
        (output/'READ_RESULTS.md').write_text(
            '# Phase 0 latent world model\n\n'
            'Read summary.json, stage_A_gate.json and dynamics/H1/complete.json first. '
            'A completed study can correctly stop at a failed validation gate. '
            'raw_reference_test.json tests observation information; latent_*/real_latent_test.json '
            'tests frozen representation. latent_*/test/phase0_comparison.json and challenge_cohorts.json '
            'compare real-child, imagined, prior and direct behavior against TRUE verifier labels. '
            'K means accepted draft prefix length, excluding bonus/correction. No exact-token output is trained. '
            'Missing teacher fields remain missing. Test questions were previously reused for design; '
            'all conclusions are exploratory. Input trace/model downloads are excluded from this ZIP.\n',encoding='utf-8')
        package(output);log('complete',seconds=summary['elapsed_seconds'],widths=summary['widths'])
        return summary
    except BaseException:
        summary.update(status='partial',elapsed_seconds=time.perf_counter()-started)
        write_json(output/'summary.json',summary)
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        package(output)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/latent_wm_phase0.json'))
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
