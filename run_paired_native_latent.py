"""Saved native Qwen teacher -> bridge/direct study, with frozen H1-H3 diagnostic.

No language-model forward is run. Tests are opened after all seed selections.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
import traceback

import torch

from behavior_aware_source import weight_hash
from factorized_wm_metrics import write_json, write_jsonl
from paired_latent_data import prepare_paired_source, pack_paired_rows
from paired_latent_models import (NativeVerifier, PairedStudent, ORACLE_METHODS,
    STUDENT_METHODS, outputs, objective, parameter_counts)
from phase0_wm_data import pack_latents
from phase0_wm_models import DynamicsPair, VerifierReadout, rollout, valid_positions
from run_latent_wm_phase0 import cpu_weights, save_torch, seed_all, package


def log(stage, **values):
    print('[paired] ' + json.dumps(dict(stage=stage, **values)), flush=True)


def config_check(cfg):
    if cfg.get('schema') != 'paired_native_latent_v1' or cfg.get('latent_dim') != 128:
        raise ValueError('Requires paired_native_latent_v1 and frozen D128')
    for k in ('oracle_updates', 'max_updates', 'eval_every', 'batch_size', 'workers', 'bootstrap_samples'):
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if cfg['workers'] > 2 or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('One or two workers; unique nonempty seeds')
    for k in ('bridge_state_weight', 'bridge_behavior_weight', 'distill_weight', 'learning_rate'):
        if cfg[k] < 0 or not torch.isfinite(torch.tensor(cfg[k])):
            raise ValueError(f'{k} must be finite and nonnegative')
    if cfg['learning_rate'] == 0 or cfg['bridge_state_weight'] == 0:
        raise ValueError('Positive learning rate and state supervision required')


class QuestionSampler:
    """Action-uniform, then question-uniform, then state-uniform sampling."""
    def __init__(self, uids, rows, seed):
        self.groups = defaultdict(lambda: defaultdict(list))
        for uid in sorted(uids):
            row = rows[uid]
            self.groups[row.get('action', 'root')][row['question']].append(uid)
        self.actions = sorted(self.groups)
        self.questions = {a: sorted(self.groups[a]) for a in self.actions}
        self.rng = random.Random(seed)
        if not self.actions:
            raise ValueError('No training states with native teacher and observed K')

    def sample(self, count):
        result = []
        for _ in range(count):
            a = self.rng.choice(self.actions)
            q = self.rng.choice(self.questions[a])
            result.append(self.rng.choice(self.groups[a][q]))
        return result


def make_model(method, cfg, native=None):
    if method in ORACLE_METHODS:
        return NativeVerifier(method, dim=cfg['latent_dim'], dropout=cfg['dropout'])
    return PairedStudent(method, dim=cfg['latent_dim'], layers=cfg['student_layers'],
                         dropout=cfg['dropout'], native_head=None if native is None else native.readout)


def call_model(model, method, batch):
    if method in ORACLE_METHODS:
        return model(batch['teacher_hidden'], batch['candidate_embedding'], batch['lengths'])
    # This interface deliberately has no teacher, labels, candidate IDs or parent K.
    return model(batch['z_D'], batch['c'], batch['context'], batch['lengths'])


def scalar_macro(rows):
    byq = defaultdict(list)
    for r in rows:
        if r['accepted'] is not None:
            byq[r['question']].append(abs(r['K_pred'] - r['accepted']))
    return sum(sum(x) / len(x) for x in byq.values()) / len(byq) if byq else None


def validation_score(records):
    cohorts = dict(All=records,
        R_changed=[r for r in records if r['action'] == 'R' and r.get('parent_token_changed') is True],
        E_full_prefix=[r for r in records if r['action'] == 'E' and r['parent_accepted'] is not None
                       and r['parent_accepted'] == r['parent_length']])
    measured = {k: dict(count=len(v), question_count=len({r['question'] for r in v}),
                         K_question_macro_MAE=scalar_macro(v)) for k, v in cohorts.items()}
    values = [x['K_question_macro_MAE'] for x in measured.values() if x['K_question_macro_MAE'] is not None]
    if not values:
        raise ValueError('No observed validation labels')
    return sum(values) / len(values), measured


def prediction_record(row, seed, method, result, index, oracle=None, latent_mse=None,
                      horizon=0, actions='', uid=None, parent=None):
    n = row['length']
    parent_length = row.get('parent_length') if parent is None else parent['length']
    parent_accepted = row.get('parent_accepted') if parent is None else parent['accepted']
    return dict(uid=uid or row['uid'], state_uid=row['uid'], question=row['question'],
        seed=seed, method=method, action=(actions[-1] if actions else row.get('action', 'root')),
        length=n, parent_length=parent_length, parent_accepted=parent_accepted,
        parent_token_changed=row.get('parent_token_changed'), accepted=row['accepted'],
        K_pred=float(result['K'][index]), q_pred=result['q'][index, :n].detach().cpu().tolist(),
        hazard_pred=result['hazards'][index, :n].detach().cpu().tolist(),
        oracle_K_pred=oracle, latent_mse=latent_mse, horizon=horizon, actions=actions,
        prediction_role=('privileged_native_teacher_ceiling' if method in ORACLE_METHODS
                         else 'D_observation_only'))


@torch.no_grad()
def evaluate_current(model, method, uids, source, cfg, device, seed, native=None):
    model.eval()
    out = []
    for start in range(0, len(uids), cfg['batch_size']):
        group = [source['rows'][u] for u in uids[start:start + cfg['batch_size']]]
        b = pack_paired_rows(group, source['cachez'], device)
        pred = call_model(model, method, b)
        result = outputs(pred['hazard'], b['lengths'])
        target = None if native is None else native(b['teacher_hidden'], b['candidate_embedding'], b['lengths'])
        oracle = None if target is None else outputs(target['hazard'], b['lengths'])['K']
        mse = None if target is None else (pred['z_V'] - target['z_V']).square().mean(-1)
        for i, row in enumerate(group):
            out.append(prediction_record(row, seed, method, result, i,
                oracle=None if oracle is None else float(oracle[i]),
                latent_mse=None if mse is None else float(mse[i, :row['length']].mean())))
    return out


def checkpoint_payload(model, optimizer, sampler, step, curve, initial, digest, device, signature):
    return dict(model=cpu_weights(model), optimizer=optimizer.state_dict(), step=step,
        curve=curve, initial_hash=initial, batch_digest=digest, sampler_rng=sampler.rng.getstate(),
        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
        signature=signature)


def train_one(method, seed, source, cfg, folder, device, signature, native=None):
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / 'complete.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if (record['signature'] != signature or record.get('method') != method
                or record.get('seed') != seed):
            raise ValueError('Completed job signature differs')
        model = make_model(method, cfg, native).to(device)
        chosen = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        if chosen.get('signature') != signature:
            raise ValueError('Completed checkpoint signature differs')
        model.load_state_dict(chosen['model'])
        return model.eval(), record
    # All four students start from identical inference weights and batch stream.
    rng_seed = seed if method in ORACLE_METHODS else seed + 500
    seed_all(rng_seed)
    model = make_model(method, cfg, native).to(device)
    initial = weight_hash(cpu_weights(model))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['learning_rate'], weight_decay=.01)
    sampler = QuestionSampler(source['observation_ids']['train'], source['rows'], seed + 700)
    total = cfg['oracle_updates'] if method in ORACLE_METHODS else cfg['max_updates']
    curve, digest, first, best = [], '', 1, float('inf')
    last = folder / 'last.pt'
    if last.exists():
        old = torch.load(last, map_location='cpu', weights_only=True)
        if old['signature'] != signature or old['initial_hash'] != initial:
            raise ValueError('Resume model/source/config/initialization differs')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.rng.setstate(old['sampler_rng']); torch.set_rng_state(old['rng_cpu'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(old['rng_cuda'], device)
        curve, digest, first = old['curve'], old['batch_digest'], old['step'] + 1
        best = min((r['validation_score'] for r in curve), default=float('inf'))
    started = time.perf_counter()
    for step in range(first, total + 1):
        model.train()
        uids = sampler.sample(cfg['batch_size'])
        digest = hashlib.sha256((digest + '|'.join(uids)).encode()).hexdigest()
        rows = [source['rows'][u] for u in uids]
        batch = pack_paired_rows(rows, source['cachez'], device)
        target_z = None
        if native is not None and method != 'Direct':
            with torch.no_grad():
                target_z = native.encode(batch['teacher_hidden'], batch['candidate_embedding'], batch['lengths'])
        prediction = call_model(model, method, batch)
        loss, parts = objective(method, prediction, batch, target_z, cfg)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite loss for {method}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite gradient')
        if method.startswith('Bridge') and any(p.grad is not None for p in model.readout.parameters()):
            raise RuntimeError('Frozen verifier head accumulated parameter gradients')
        optimizer.step()
        if step % cfg['eval_every'] == 0 or step == total:
            records = evaluate_current(model, method, source['observation_ids']['val'], source,
                                       cfg, device, seed, native)
            score, cohorts = validation_score(records)
            item = dict(step=step, loss=float(loss.detach()),
                **{k: float(v.detach()) for k, v in parts.items()},
                validation_score=score, validation_cohorts=cohorts, gradient_norm=float(norm),
                elapsed_seconds=time.perf_counter() - started, batch_digest=digest)
            curve.append(item)
            if score < best:
                best = score
                save_torch(folder / 'best.pt', dict(model=cpu_weights(model), step=step,
                                                  validation_score=score, signature=signature))
            save_torch(last, checkpoint_payload(model, optimizer, sampler, step, curve,
                                                initial, digest, device, signature))
            write_json(folder / 'learning_curve.json', curve)
            log('train', method=method, seed=seed, device=device, **item)
    chosen = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(chosen['model']); model.eval()
    record = dict(method=method, seed=seed, updates=total, selected_step=chosen['step'],
        validation_score=chosen['validation_score'], signature=signature, initial_hash=initial,
        batch_digest=digest, parameter_counts=parameter_counts(model),
        elapsed_seconds=time.perf_counter() - started, test_access=False)
    write_json(folder / 'selection.json', record); write_json(complete, record)
    return model, record


def worker(job):
    torch.set_num_threads(1)
    source = torch.load(job['payload'], map_location='cpu', weights_only=True)
    test = set(source['split']['test'])
    if any(r['question'] in test for r in source['rows'].values()) or source['observation_ids'].get('test'):
        raise ValueError('Test states entered a training worker')
    cfg, seed, device = job['config'], job['seed'], job['device']
    root = Path(job['folder']); root.mkdir(parents=True, exist_ok=True)
    native = None; results = []
    for method in ('V_hidden_only', 'V_joint_probe', 'V_joint'):
        model, record = train_one(method, seed, source, cfg, root / method, device, job['signature'])
        results.append(record)
        if method == 'V_joint':
            native = model.eval().requires_grad_(False)
        else:
            del model
    native_hash = weight_hash(cpu_weights(native))
    for method in STUDENT_METHODS:
        model, record = train_one(method, seed, source, cfg, root / method, device, job['signature'], native)
        results.append(record); del model
    students = [r for r in results if r['method'] in STUDENT_METHODS]
    if len({r['initial_hash'] for r in students}) != 1 or len({r['batch_digest'] for r in students}) != 1:
        raise RuntimeError('Student initial weights or training batch schedules did not match')
    if native_hash != weight_hash(cpu_weights(native)):
        raise RuntimeError('Frozen native teacher changed during student training')
    record = dict(seed=seed, signature=job['signature'], jobs=results,
                  student_initialization_and_batches_matched=True, native_teacher_hash=native_hash)
    write_json(root / 'complete.json', record)
    return record


def execute_jobs(jobs, devices, output):
    slots = queue.Queue()
    for device in devices:
        slots.put(device)
    cancelled = threading.Event(); lock = threading.Lock(); active = set()
    def execute(job):
        if cancelled.is_set():
            raise RuntimeError('Cancelled after worker failure')
        device = slots.get()
        try:
            if cancelled.is_set():
                raise RuntimeError('Cancelled after worker failure')
            job = dict(job, device=device)
            path = Path(output) / '_cache' / f"job_seed{job['seed']}.json"
            write_json(path, job)
            root = Path(job['folder']); root.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker_job', str(path)]
            with (root / 'worker_log.txt').open('a', encoding='utf-8') as file, subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    errors='replace', bufsize=1) as process:
                with lock:
                    active.add(process)
                try:
                    for line in process.stdout:
                        file.write(line); file.flush(); print(line, end='', flush=True)
                    code = process.wait()
                finally:
                    with lock:
                        active.discard(process)
            if code:
                raise RuntimeError(f"Seed {job['seed']} worker failed with exit {code}; see worker_log.txt")
            return json.loads((root / 'complete.json').read_text())
        finally:
            slots.put(device)
    results = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(execute, job): job['seed'] for job in jobs}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                package(output)
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            with lock:
                for process in active:
                    if process.poll() is None:
                        process.terminate()
            raise
    return [results[j['seed']] for j in jobs]


def load_model(method, root, cfg, device):
    model = make_model(method, cfg).to(device)
    weights = torch.load(root / method / 'best.pt', map_location='cpu', weights_only=True)['model']
    model.load_state_dict(weights)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_rollouts(models, source, cfg, device, seed):
    """Only root D state is read before free R/E composition. End teachers score it."""
    weights = source.get('dynamics_weights')
    if weights is None:
        return [], dict(status='unavailable', reason='Source lacks frozen Phase0 H1 dynamics checkpoint')
    dcfg = source['source_config']
    dynamics = DynamicsPair(128, dcfg['dynamics_layers'], dcfg['dropout']).to(device)
    dynamics.load_state_dict(weights); dynamics.eval().requires_grad_(False)
    frozen_g = VerifierReadout(128, 1, dcfg['dropout']).to(device)
    frozen_g.load_state_dict(source['frozen']['readout']); frozen_g.eval().requires_grad_(False)
    native = models['V_joint']; out = []; counts = {}
    for horizon in (1, 2, 3):
        pool = [p for p in source['paths']['test'] if len(p[1]) == horizon
                and p[0][-1] in set(source['observation_ids']['test'])]
        counts[str(horizon)] = len(pool)
        for start in range(0, len(pool), cfg['batch_size']):
            paths = pool[start:start + cfg['batch_size']]
            parents = [source['rows'][p[0][0]] for p in paths]
            children = [source['rows'][p[0][-1]] for p in paths]
            root_state = pack_latents(parents, source['cachez'], device)
            imagined = rollout(dynamics, root_state, [p[1] for p in paths])
            child_batch = pack_paired_rows(children, source['cachez'], device)
            real_v = native(child_batch['teacher_hidden'], child_batch['candidate_embedding'], child_batch['lengths'])
            real_v_result = outputs(real_v['hazard'], child_batch['lengths'])
            if not torch.equal(imagined.lengths, child_batch['lengths']):
                raise RuntimeError('Free rollout length disagrees with observed path geometry')
            predictions = {m: model(imagined.z, imagined.c, imagined.context, imagined.lengths)
                           for m, model in models.items() if m in STUDENT_METHODS}
            predictions['Frozen_D_readout'] = frozen_g(imagined.z, imagined.lengths)
            results = {m: outputs(v['hazard'], imagined.lengths) for m, v in predictions.items()}
            latent_mse = {m: (v['z_V'] - real_v['z_V']).square().mean(-1)
                          for m, v in predictions.items() if 'z_V' in v}
            # Oracle real-V predictions are privileged ceilings, never injected.
            results['V_joint'] = real_v_result
            for i, (path, child) in enumerate(zip(paths, children)):
                nodes, seq = path
                actions = ''.join('E' if x else 'R' for x in seq)
                last_parent = source['rows'][nodes[-2]]
                for method, result in results.items():
                    out.append(prediction_record(child, seed, method, result, i,
                        oracle=float(real_v_result['K'][i]),
                        latent_mse=float(latent_mse[method][i, :child['length']].mean()) if method in latent_mse else None,
                        horizon=horizon, actions=actions,
                        uid=json.dumps(list(nodes), ensure_ascii=False, separators=(',', ':')),
                        parent=last_parent))
    return out, dict(status='ok', paths_by_horizon=counts, dynamics_retrained=False,
                    real_intermediate_state_injections=0, target_teachers='evaluation-only')


@torch.no_grad()
def benchmark_models(models, source, cfg, device):
    group = [source['rows'][u] for u in source['observation_ids']['test'][:1]]
    b = pack_paired_rows(group, source['cachez'], device)
    result = dict(device=device, batch_size=1, native_teacher_capture_included=False,
                  D_observation_encoding_included=False, real_LLM_latency_measured=False, methods={})
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
    for method, model in models.items():
        for _ in range(8):
            call_model(model, method, b)
        times = []
        for _ in range(30):
            sync(); start = time.perf_counter()
            call_model(model, method, b); sync()
            times.append(1000 * (time.perf_counter() - start))
        ordered = sorted(times)
        result['methods'][method] = dict(median_ms=ordered[len(ordered)//2],
            mean_ms=sum(times)/len(times), min_ms=min(times),
            parameter_counts=parameter_counts(model), input_length=int(b['lengths'][0]),
            role='privileged_teacher_probe' if method in ORACLE_METHODS else 'deployable_student')
    return result


@torch.no_grad()
def inference_sensitivity(models, source, cfg, device, seed):
    """Input occlusion changes distribution; it does not measure causal importance."""
    result = dict(seed=seed, scope='held-out inference occlusion; not retrained or causal importance', methods={})
    for method in ('V_joint', 'Direct', 'Bridge_behavior', 'Direct_distill'):
        model = models[method]
        fields = ('teacher_hidden', 'candidate_embedding') if method == 'V_joint' else ('z_D', 'c', 'context')
        rows_by = defaultdict(list)
        for start in range(0, len(source['observation_ids']['test']), cfg['batch_size']):
            group = [source['rows'][u] for u in source['observation_ids']['test'][start:start+cfg['batch_size']]]
            b = pack_paired_rows(group, source['cachez'], device)
            baseline = outputs(call_model(model, method, b)['hazard'], b['lengths'])['K']
            for field in fields:
                altered = dict(b); altered[field] = torch.zeros_like(b[field])
                values = outputs(call_model(model, method, altered)['hazard'], b['lengths'])['K']
                for i, row in enumerate(group):
                    rows_by[field].append(dict(question=row['question'], accepted=row['accepted'],
                        K_pred=float(values[i]), baseline=float(baseline[i])))
        result['methods'][method] = {}
        for field, rows in rows_by.items():
            base = [dict(r, K_pred=r['baseline']) for r in rows]
            result['methods'][method][field + '_zero'] = dict(count=len(rows),
                K_question_macro_MAE=scalar_macro(rows), baseline_K_question_macro_MAE=scalar_macro(base),
                delta_MAE=scalar_macro(rows)-scalar_macro(base))
    return result


def source_hashes():
    names = ('run_paired_native_latent.py', 'paired_latent_models.py', 'paired_latent_data.py',
             'paired_latent_metrics.py', 'behavior_aware_source.py', 'phase0_wm_models.py',
             'phase0_wm_data.py', 'factorized_wm_data.py', 'factorized_wm_metrics.py')
    return {n: hashlib.sha256((Path(__file__).parent / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}


def study_fingerprint(cfg, source):
    normalization = source['normalization']
    def jsonable(value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        return value
    content = dict(config={k:v for k,v in cfg.items() if k != 'devices'},
        source_phase0=source['provenance']['fingerprint'], code=source_hashes(),
        source_artifacts=source['provenance']['artifact_hashes'],
        source_raw=source['provenance']['original_data_digest'],
        teacher_hidden=source['provenance']['teacher_hidden_checksum'],
        normalization=jsonable(normalization),
        observation_ids=source['observation_ids'],
        dynamics=None if source.get('dynamics_weights') is None else weight_hash(source['dynamics_weights']))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); summary = dict(schema='paired_native_latent_v1', status='running',
        LLM_forwards=0, encoder_retrained=False, dynamics_retrained=False,
        teacher_scope='Qwen final causal hidden already projected32', full_native_ceiling=False)
    try:
        cfg = json.loads(Path(args.config).read_text()); config_check(cfg)
        summary['pipeline_check_only'] = bool(cfg.get('pipeline_check_only', False))
        torch.set_num_threads(4)
        devices = cfg.get('devices', ['cuda:0', 'cuda:1'])[:cfg['workers']]
        if len(devices) != cfg['workers']:
            raise ValueError('Insufficient configured worker devices')
        for device in devices:
            if device.startswith('cuda') and (not torch.cuda.is_available()
                    or int(device.split(':')[-1]) >= torch.cuda.device_count()):
                raise ValueError(f'Worker device unavailable: {device}')
        cache = output / '_cache'; cache.mkdir(exist_ok=True)
        log('prepare', input=str(args.input), phase0=str(args.phase0_input))
        source = prepare_paired_source(args.input, args.phase0_input, cache,
            verification_samples=cfg['encoder_verification_samples'], device=devices[0])
        signature = study_fingerprint(cfg, source)
        manifest_path = output / 'study_manifest.json'
        if args.resume:
            previous = json.loads(manifest_path.read_text())
            if previous['fingerprint'] != signature:
                raise ValueError('Resume input/config/code fingerprint differs')
        elif manifest_path.exists():
            raise FileExistsError('Output exists; use new output or --resume')
        write_json(output / 'config.json', cfg)
        write_json(output / 'split_manifest.json', source['split'])
        write_json(output / 'source_checkpoint_manifest.json', source['provenance'])
        write_json(output / 'data_audit.json', dict(
            observation_counts={k:len(v) for k,v in source['observation_ids'].items()},
            source=source.get('data_audit', source['provenance']), first3_teacher_outcomes_excluded=True,
            D_forward_uses_teacher=False, parent_K_model_input=False, normalization_fit_split='train',
            teacher_stage='final_norm_causal_hidden_projected32', compression_of_full_Qwen_tested=False))
        write_json(manifest_path, dict(schema='paired_native_latent_v1', fingerprint=signature, status='running'))
        save_torch(output / 'teacher_normalization.pt', source['normalization'])
        write_json(output / 'feature_schema.json', dict(
            D_input=['frozen_z_D128', 'native_c20', 'known_context8', 'lengths'],
            teacher_input=['Qwen_projected_hidden32', 'pretrained_STOP_embedding64'],
            forbidden_input=['teacher_margin', 'teacher_probability', 'teacher_agreement', 'true_parent_K', 'future_true_state'],
            teacher_representation='learned128 over an already projected32 source, not full-hidden compression',
            target_role='teacher-only native verifier supervision'))
        trainval = set(source['split']['train'] + source['split']['val'])
        training = {k:v for k,v in source.items() if k not in ('rows','cachez','paths','observation_ids')}
        training.update(rows={u:r for u,r in source['rows'].items() if r['question'] in trainval},
            cachez={u:z for u,z in source['cachez'].items() if source['rows'][u]['question'] in trainval},
            paths={}, observation_ids={k:source['observation_ids'][k] for k in ('train','val')})
        training_file = cache / 'train_val.pt'; save_torch(training_file, training)
        del training; gc.collect()
        jobs = [dict(seed=s, payload=str(training_file), config=cfg,
                     folder=str(output / 'jobs' / f'seed{s}'), signature=signature) for s in cfg['seeds']]
        package(output)
        completed = execute_jobs(jobs, devices, output)
        write_json(output / 'paired_training_audit.json', completed)
        write_json(output / 'test_access_manifest.json', dict(
            all_training_and_selections_complete=True, selection_split='val',
            test_questions=source['split']['test'], opened_at_seconds=time.perf_counter()-started,
            full_Qwen_teacher_capture=False, immutable_D_encoder=True))
        records = []; timings = []; sensitivity = []; rollout_status = []
        for seed in cfg['seeds']:
            device = devices[0]; root = output / 'jobs' / f'seed{seed}'
            models = {m:load_model(m, root, cfg, device) for m in ORACLE_METHODS + STUDENT_METHODS}
            seed_rows = []
            for method, model in models.items():
                seed_rows += evaluate_current(model, method, source['observation_ids']['test'],
                    source, cfg, device, seed, models['V_joint'])
            timings.append(dict(seed=seed, **benchmark_models(models, source, cfg, device)))
            sensitivity.append(inference_sensitivity(models, source, cfg, device, seed))
            if cfg.get('rollout_existing_dynamics', True):
                future, status = evaluate_rollouts(models, source, cfg, device, seed)
                seed_rows += future; rollout_status.append(dict(seed=seed, **status))
            write_jsonl(root / 'test_predictions.jsonl', seed_rows); records += seed_rows
            for model in models.values():
                model.cpu()
            del models; gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            log('test', seed=seed, prediction_rows=len(seed_rows))
            package(output)
        write_jsonl(output / 'all_predictions.jsonl', records)
        write_json(output / 'inference_latency.json', timings)
        write_json(output / 'inference_sensitivity.json', sensitivity)
        write_json(output / 'frozen_dynamics_rollout.json', rollout_status)
        from paired_latent_metrics import write_reports
        report = write_reports(records, output, bootstrap_samples=cfg['bootstrap_samples'], seed=42)
        summary.update(status='complete', seconds=time.perf_counter()-started, seeds=cfg['seeds'],
            split=dict(train=70,val=15,test=15), observation_counts={k:len(v) for k,v in source['observation_ids'].items()},
            prediction_rows=len(records), train_jobs=sum(len(c['jobs']) for c in completed),
            updates_completed=sum(r['updates'] for c in completed for r in c['jobs']),
            matched_inference_architectures=True, actual_planner_regret_tested=False)
        write_json(output / 'summary.json', summary)
        manifest = json.loads(manifest_path.read_text()); manifest['status']='complete'; write_json(manifest_path, manifest)
        text = ('# Paired native latent feasibility pilot\n\n'
            'This uses saved Qwen causal final hidden already projected to32 plus candidate embeddings. '
            'It does not establish a full native Qwen compression ceiling. Questions are reused; results are exploratory.\n\n'
            'Inspect current-state cohorts first: native joint teacher, Direct, Bridge_state, Bridge_behavior, Direct_distill. '
            'All four students have identical inference architecture and paired initial weights/batches. '
            'The bridge readout is frozen; direct readouts train. Training parameter counts are reported.\n\n'
            'H1/H2/H3 use existing frozen Phase0 H1 dynamics without real intermediate-state injection. '
            'This isolates changes to the verifier representation; it is not a new dynamics training run.\n\n'
            'Learning curves use validation only; test predictions begin after every selection is complete. '
            'Missing teacher states are excluded consistently and coverage is reported. '
            'Teacher margin/probability/agreement are excluded from teacher hidden inputs.\n\n'
            'Read comparison.csv, metrics.json, paired_question_bootstrap.json, inference_sensitivity.json, '
            'inference_latency.json and jobs/*/*/learning_curve.json. '
            'Latency excludes real D/Qwen encoding and does not establish online speedup.\n')
        (output / 'FINAL_REPORT.md').write_text(text, encoding='utf-8')
        package(output); log('complete', **summary)
        return summary
    except BaseException:
        summary.update(status='partial', seconds=time.perf_counter()-started)
        write_json(output / 'summary.json', summary)
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        package(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker_job')
    parser.add_argument('--input'); parser.add_argument('--phase0_input')
    parser.add_argument('--output'); parser.add_argument('--config'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text()))
    else:
        if not all((args.input,args.phase0_input,args.output,args.config)):
            parser.error('--input, --phase0_input, --output and --config are required')
        run(args)


if __name__ == '__main__':
    main()
