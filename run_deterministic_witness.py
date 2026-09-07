#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from oracle_run_guard import guard, preserve
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJ = ROOT / 'PRECOMPUTED_WITNESS' / 'BEST_TRAJECTORY_50.csv'
DEFAULT_SCHEDULE = ROOT / 'PRECOMPUTED_WITNESS' / 'DETERMINISTIC_PROBE_SCHEDULE.csv'


def parse_args():
    p = argparse.ArgumentParser(description='Run a real deterministic-oracle-probe U1 witness and verify positive aggregate learned-C benefit.')
    p.add_argument('--dllm_dir', default='/kaggle/working/Fast_dLLM_v2_1.5B')
    p.add_argument('--output_dir', default='/kaggle/working/outputs_math50_deterministic_witness')
    p.add_argument('--trajectory_csv', default=str(DEFAULT_TRAJ))
    p.add_argument('--probe_schedule_csv', default=str(DEFAULT_SCHEDULE))
    p.add_argument('--target_quantization', default='none')
    p.add_argument('--target_dtype', choices=['auto','fp16','bf16','fp32'], default='fp16')
    p.add_argument('--drafter_dtype', choices=['auto','fp16','bf16','fp32'], default='fp16')
    p.add_argument('--target_device', type=int, default=0)
    p.add_argument('--drafter_device', type=int, default=1)
    p.add_argument('--target_model_name', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--two_gpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--drafter_threshold', type=float, default=.50)
    p.add_argument('--lowconf_threshold', type=float, default=.70)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--min_learned_c', type=int, default=5)
    p.add_argument('--min_total_learned_benefit', type=float, default=1.0,
                   help='Require -sum(delta_J) over learned CONTINUE rows to be at least this positive margin.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log_level', default='INFO')
    return p.parse_args()


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('\n' + '='*110)
    print('RUN:', ' '.join(map(str, cmd)))
    print('LOG:', log_path)
    print('='*110, flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def bools(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes'])


def base_cmd(a, ids, dest):
    if a.target_quantization != 'none' or a.target_dtype != 'fp16' or a.drafter_dtype != 'fp16':
        raise ValueError('This experiment requires FP16 target and drafter, without quantization')
    return [
        sys.executable, '-u', 'failfast.py',
        '--dataset_name', 'math',
        '--num_questions', str(len(ids)),
        '--problem_ids', *map(str, ids),
        '--warmup_questions', '1',
        '--benchmark_modes', 'dllm_ar',
        '--dllm_variant', 'failfast',
        '--decoding_strategy', 'greedy',
        '--max_new_tokens', str(a.max_new_tokens),
        '--spec_len', '8', '--block_size', '32', '--small_block_size', '8',
        '--target_model_name', str(a.target_model_name),
        '--dllm_dir', str(a.dllm_dir),
        '--target_device', str(a.target_device),
        '--drafter_device', str(a.drafter_device),
        '--target_quantization', str(a.target_quantization),
        '--unquantized_dtype', 'float16',
        *(['--target_two_gpu_fp16'] if a.two_gpu else []),
        '--disable_reusing_drafter_kvs',
        '--drafter_thresholds', str(a.drafter_threshold),
        '--sweep_lowconf_threshold', str(a.lowconf_threshold),
        '--sweep_max_spec_len', '64', '--sweep_incr_len', '8',
        '--seed', str(a.seed), '--quiet_generation', '--disable_progress',
        '--skip_artifacts', '--skip_plots', '--overwrite',
        '--output_dir', str(dest), '--log_level', a.log_level,
    ]


def u1_common(cmd):
    cmd += [
        '--adaptive-td',
        '--adaptive-feature-schema', 'otrc_v2_2_compact_td',
        '--adaptive-credit-assignment', 'hindsight_delta_j_logistic_f2',
        '--adaptive-policy-mode', 'hindsight_delta_j_logistic_f2',
        '--adaptive-hindsight-logistic-tie-ms-per-token', '1.0',
        '--no-adaptive-hindsight-logistic-use-class-weight',
        '--no-adaptive-hindsight-logistic-use-prefix-feature',
        '--no-adaptive-hindsight-logistic-dynamic-threshold',
        '--adaptive-hindsight-logistic-utility-weighting', 'raw_abs',
        '--adaptive-hindsight-logistic-replay-stop-to-continue-ratio', '0',
        '--no-adaptive-hindsight-logistic-balance-utility-mass',
        '--adaptive-log-decisions', '--adaptive-profile-overhead',
        '--adaptive-hindsight-state-fingerprint',
    ]
    return cmd


def u1_cmd(a, ids, dest):
    cmd = base_cmd(a, ids, dest)
    u1_common(cmd)
    cmd += [
        '--adaptive-policy-ablation', 'learned',
        '--adaptive-hindsight-logistic-learning-rate', '0.05',
        '--adaptive-hindsight-logistic-continue-threshold', '0.5',
        '--adaptive-hindsight-delta-j-min-pairs', '30',
        '--adaptive-hindsight-delta-j-min-continue-pairs', '3',
        '--adaptive-hindsight-logistic-min-positive-problems', '2',
        # Keep the native nominal probe probabilities. With a deterministic
        # schedule loaded they are used for audit only; no random probe draw is made.
        '--adaptive-hindsight-delta-j-structural-probe', '0.08',
        '--adaptive-hindsight-delta-j-floor-probe', '0.02',
        '--adaptive-hindsight-logistic-replay-batch-size', '16',
        '--adaptive-hindsight-logistic-replay-buffer-size', '100',
        '--adaptive-hindsight-probe-tape', str(a.probe_schedule_csv),
        '--adaptive-hindsight-probe-trace', str(Path(dest)/'probe_tape_trace.jsonl'),
    ]
    return cmd


def always_stop_cmd(a, ids, dest):
    cmd = base_cmd(a, ids, dest)
    u1_common(cmd)
    cmd += [
        '--adaptive-policy-ablation', 'frozen_stop',
        '--adaptive-hindsight-logistic-learning-rate', '0.0',
        '--adaptive-hindsight-logistic-continue-threshold', '0.5',
        '--adaptive-hindsight-delta-j-min-pairs', '0',
        '--adaptive-hindsight-delta-j-min-continue-pairs', '0',
        '--adaptive-hindsight-logistic-min-positive-problems', '0',
        '--adaptive-hindsight-delta-j-structural-probe', '0',
        '--adaptive-hindsight-delta-j-floor-probe', '0',
        '--adaptive-hindsight-logistic-replay-batch-size', '0',
        '--adaptive-hindsight-logistic-replay-buffer-size', '100',
    ]
    return cmd


def complete(case, ids):
    p = case / 'benchmark_results.csv'
    if not p.exists(): return False
    try:
        d = pd.read_csv(p)
        if 'mode' in d.columns: d = d[d['mode'] == 'dllm_ar']
    except Exception:
        return False
    return len(d) == len(ids) and d.problem_id.astype(int).tolist() == list(ids)


def ensure(a, name, ids, cmd):
    case = Path(a.output_dir) / name
    if a.resume and complete(case, ids):
        print('[RESUME] reuse', case)
        return case
    preserve(case)
    run(cmd, Path(a.output_dir) / 'logs' / f'{name}.log')
    if not complete(case, ids):
        raise RuntimeError('Incomplete or reordered witness benchmark')
    return case


def load_bench(case):
    d = pd.read_csv(case / 'benchmark_results.csv')
    return d[d['mode'] == 'dllm_ar'].copy() if 'mode' in d.columns else d


def load_transitions(case):
    p = case / 'adaptive_full_stream_transitions.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_decisions(case):
    p = case / 'adaptive_td_decisions.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def resolved_non_tie(tr, ids):
    if tr.empty: return tr.copy()
    d = tr[tr.problem_id.astype(int).isin(set(ids))].copy()
    if 'pair_resolved' in d.columns: d = d[bools(d.pair_resolved)]
    d['binary_label_C'] = pd.to_numeric(d.binary_label_C, errors='coerce')
    d['delta_J_ms_per_token'] = pd.to_numeric(d.delta_J_ms_per_token, errors='coerce')
    d = d[d.binary_label_C.isin([0,1]) & d.delta_J_ms_per_token.notna()].copy()
    if 'update_applied' in d.columns: d = d[bools(d.update_applied)]
    return d


def pooled_ms(d):
    return float(d.actual_e2e_time.sum()*1000.0 / d.output_tokens.sum())


def summarize_u1(case, ids, scheduled_count):
    tr = resolved_non_tie(load_transitions(case), ids)
    dec = load_decisions(case)
    dec = dec[dec.problem_id.astype(int).isin(set(ids))].copy() if not dec.empty else dec
    src = tr.action_source.astype(str) if 'action_source' in tr.columns else pd.Series('', index=tr.index)
    learned = tr[src.eq('learned_continue')].copy()
    probe = tr[src.isin(['structural_probe','floor_probe'])].copy()
    b = load_bench(case)

    action_src = dec.action_source.astype(str) if not dec.empty and 'action_source' in dec.columns else pd.Series('', index=dec.index)
    oracle_probe_actions = int(action_src.isin(['structural_probe','floor_probe']).sum())
    scheduled_flags = bools(dec.deterministic_probe_scheduled) if not dec.empty and 'deterministic_probe_scheduled' in dec.columns else pd.Series(False,index=dec.index)
    selected_flags = bools(dec.deterministic_probe_selected) if not dec.empty and 'deterministic_probe_selected' in dec.columns else pd.Series(False,index=dec.index)
    state_match = bools(dec.deterministic_probe_state_match) if not dec.empty and 'deterministic_probe_state_match' in dec.columns else pd.Series(True,index=dec.index)
    shadowed = bools(dec.deterministic_probe_shadowed_by_learned) if not dec.empty and 'deterministic_probe_shadowed_by_learned' in dec.columns else pd.Series(False,index=dec.index)
    mismatch_count = int((scheduled_flags & ~state_match).sum())
    probe_p = pd.to_numeric(dec.get('probe_probability',0), errors='coerce').fillna(0.0) if not dec.empty else pd.Series(dtype=float)
    probe_eligible = probe_p.gt(0.0)
    structural_eligible = bools(dec.get('structural_probe_eligible', pd.Series(False,index=dec.index))) if not dec.empty else pd.Series(dtype=bool)
    floor_eligible = probe_eligible & ~structural_eligible
    structural_eligible = probe_eligible & structural_eligible
    nominal_expected = float(probe_p[probe_eligible].sum())
    expected_structural = float(probe_p[structural_eligible].sum())
    expected_floor = float(probe_p[floor_eligible].sum())
    variance_total = float((probe_p[probe_eligible]*(1.0-probe_p[probe_eligible])).sum())
    variance_structural = float((probe_p[structural_eligible]*(1.0-probe_p[structural_eligible])).sum())
    variance_floor = float((probe_p[floor_eligible]*(1.0-probe_p[floor_eligible])).sum())
    realized_structural = int(action_src.eq('structural_probe').sum())
    realized_floor = int(action_src.eq('floor_probe').sum())
    probe_ratio = float(oracle_probe_actions / nominal_expected) if nominal_expected > 0 else float('nan')
    def zdev(k,e,v): return (float(k)-float(e))/np.sqrt(max(float(v),1.0))
    def huber_abs(z,delta=2.0):
        aa=abs(float(z)); return 0.5*aa*aa if aa<=delta else delta*(aa-0.5*delta)
    z_total=zdev(oracle_probe_actions,nominal_expected,variance_total)
    z_struct=zdev(realized_structural,expected_structural,variance_structural)
    z_floor=zdev(realized_floor,expected_floor,variance_floor)
    probe_soft_penalty=huber_abs(z_total)+0.75*huber_abs(z_struct)+0.75*huber_abs(z_floor)
    score_max = float(pd.to_numeric(dec.get('continue_score', np.nan), errors='coerce').max()) if not dec.empty else float('nan')

    def csum(x): return float(pd.to_numeric(x.delta_J_ms_per_token, errors='coerce').sum()) if len(x) else 0.0
    tape = pd.read_json(case/'probe_tape_trace.jsonl', lines=True)
    if tape.empty:
        raise RuntimeError('Missing executed decision tape')
    mismatch_count = int((~tape.state_match | ~tape.action_match).sum())
    mismatch_count += abs(scheduled_count - len(tape))
    mismatch_count += int((tape.requested_probe & ~tape.probe_selected).sum())
    scheduled_flags = tape.scheduled
    selected_flags = tape.probe_selected
    shadowed = tape.requested_probe & tape.action_source.eq('learned_continue')
    all_pairs = load_transitions(case)
    if not all_pairs.empty:
        all_pairs = all_pairs[all_pairs.problem_id.astype(int).isin(set(ids))]
        if 'pair_resolved' in all_pairs:
            all_pairs = all_pairs[bools(all_pairs.pair_resolved)]
        all_learned = all_pairs[all_pairs.action_source.eq('learned_continue')]
    else:
        all_learned = all_pairs
    return {
        'problems': len(ids),
        'e2e_ms_per_token': pooled_ms(b),
        'output_tokens': int(b.output_tokens.sum()),
        'resolved_non_tie': int(len(tr)),
        'good_c': int((tr.binary_label_C==1).sum()),
        'good_c_rate': float((tr.binary_label_C==1).mean()) if len(tr) else float('nan'),
        'learned_continue_non_tie': int(len(learned)),
        'learned_continue_tp': int((learned.binary_label_C==1).sum()),
        'learned_continue_fp': int((learned.binary_label_C==0).sum()),
        'sum_delta_j_learned_continue': csum(all_learned),
        'sum_delta_j_learned_non_tie': csum(learned),
        'total_learned_benefit_ms_per_token': -csum(all_learned),
        'probe_continue_non_tie': int(len(probe)),
        'probe_continue_tp': int((probe.binary_label_C==1).sum()),
        'probe_continue_fp': int((probe.binary_label_C==0).sum()),
        'sum_delta_j_probe_continue': csum(probe),
        'oracle_probe_actions': oracle_probe_actions,
        'scheduled_probe_rows': int(scheduled_count),
        'scheduled_states_encountered': int(scheduled_flags.sum()),
        'scheduled_probes_selected': int(selected_flags.sum()),
        'scheduled_probes_shadowed_by_learned': int(shadowed.sum()),
        'schedule_state_mismatch_count': mismatch_count,
        'nominal_expected_random_probe_count': nominal_expected,
        'expected_random_structural_probe_count': expected_structural,
        'expected_random_floor_probe_count': expected_floor,
        'eligible_structural_probe_states': int(structural_eligible.sum()),
        'eligible_floor_probe_states': int(floor_eligible.sum()),
        'realized_structural_probe_actions': realized_structural,
        'realized_floor_probe_actions': realized_floor,
        'realized_probe_ratio_to_nominal_expectation': probe_ratio,
        'probe_total_z': float(z_total),
        'probe_structural_z': float(z_struct),
        'probe_floor_z': float(z_floor),
        'probe_soft_penalty': float(probe_soft_penalty),
        'probe_ratio_is_hard_criterion': False,
        'max_continue_score': score_max,
    }


def compare(as_case, u1_case, ids, nboot):
    a=load_bench(as_case); u=load_bench(u1_case)
    as_ms=pooled_ms(a); u_ms=pooled_ms(u); speed=as_ms/u_ms
    m=a.merge(u,on='problem_id',suffixes=('_as','_u1'))
    rng=np.random.default_rng(20260907); vals=[]
    for _ in range(nboot):
        z=m.iloc[rng.integers(0,len(m),len(m))]
        ta=z.actual_e2e_time_excluding_transfer_as.sum()*1000/z.output_tokens_as.sum()
        tu=z.actual_e2e_time_excluding_transfer_u1.sum()*1000/z.output_tokens_u1.sum()
        vals.append(float(ta/tu))
    lo,hi=np.quantile(vals,[.025,.975])
    same=m[m.output_token_hash_as.astype(str)==m.output_token_hash_u1.astype(str)] if 'output_token_hash_as' in m.columns else m.iloc[0:0]
    same_speed=float('nan')
    if len(same):
        ta=same.actual_e2e_time_excluding_transfer_as.sum()*1000/same.output_tokens_as.sum()
        tu=same.actual_e2e_time_excluding_transfer_u1.sum()*1000/same.output_tokens_u1.sum()
        same_speed=float(ta/tu)
    return {
        'always_stop_ms_per_token':as_ms,
        'u1_ms_per_token':u_ms,
        'speedup_u1_vs_always_stop':speed,
        'paired_bootstrap_ci95':[float(lo),float(hi)],
        'u1_faster_problem_count':int((m.actual_e2e_time_excluding_transfer_u1/m.output_tokens_u1 < m.actual_e2e_time_excluding_transfer_as/m.output_tokens_as).sum()),
        'exact_output_hash_count':int(len(same)),
        'exact_output_hash_rate':float(len(same)/len(m)) if len(m) else float('nan'),
        'exact_hash_subset_speedup':same_speed,
    }


def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    traj=pd.read_csv(a.trajectory_csv)
    ids=traj.problem_id.astype(int).tolist()
    if len(ids)!=50 or len(set(ids))!=50:
        raise ValueError('trajectory must contain exactly 50 unique problem IDs')
    schedule=pd.read_csv(a.probe_schedule_csv)
    if set(schedule.problem_id.astype(int))-(set(ids) | {0}):
        raise ValueError('probe schedule contains problem IDs outside the frozen trajectory')
    if schedule.empty or schedule.state_hash.isna().any():
        raise ValueError('Witness requires a full decision tape with candidate/context hashes')
    config = {**vars(a), 'ids': ids,
              'tape_sha256': hashlib.sha256(Path(a.probe_schedule_csv).read_bytes()).hexdigest()}
    guard(out, config, a.resume)
    subprocess.run([sys.executable, str(ROOT/'patch_fastdllm_frontier.py'), a.dllm_dir], check=True)

    print('[WITNESS] 50 problem IDs:', ids)
    print('[WITNESS] deterministic probe rows:', len(schedule))
    print('[WITNESS] threshold=0.5, U1 batch1x B16/K100, random probes disabled by schedule mode')

    u1_case=ensure(a,'u1_deterministic_oracle_probe',ids,u1_cmd(a,ids,out/'u1_deterministic_oracle_probe'))
    u1_summary=summarize_u1(u1_case,ids,len(schedule))
    (out/'U1_ACTUAL_SUMMARY.json').write_text(json.dumps(u1_summary,indent=2,allow_nan=True),encoding='utf-8')
    print('\n[U1 ACTUAL]')
    print(json.dumps(u1_summary,indent=2,allow_nan=True))


    min_benefit=abs(float(a.min_total_learned_benefit))
    total_benefit=float(u1_summary['total_learned_benefit_ms_per_token'])
    enough_learned=bool(u1_summary['learned_continue_non_tie']>=a.min_learned_c)
    positive_total=bool(total_benefit>=min_benefit)
    score_ok=bool(u1_summary['max_continue_score']>0.5)
    schedule_ok=bool(u1_summary['schedule_state_mismatch_count']==0)
    passed=bool(enough_learned and positive_total and score_ok and schedule_ok)
    final={
        **u1_summary,
        'criteria':{
            'learned_continue_non_tie_min':a.min_learned_c,
            'total_learned_benefit_min_ms_per_token':min_benefit,
            'equivalent_sum_delta_j_max':-min_benefit,
            'max_score_gt_0p5':True,
            'schedule_state_mismatch_zero':True,
            'always_stop_speed_required':False,
            'minimum_tp_required':False,
            'probe_ratio_required':False,
            'probe_ratio_soft_preference_only':True,
        },
        'pass_enough_learned_continue':enough_learned,
        'pass_positive_total_learned_benefit':positive_total,
        'pass_score':score_ok,
        'pass_schedule_state_match':schedule_ok,
        'PASS':passed,
    }
    (out/'FINAL_WITNESS_SUMMARY.json').write_text(json.dumps(final,indent=2,allow_nan=True),encoding='utf-8')
    verdict=(
        f"PASS={passed}\n"
        f"learned_C={final['learned_continue_non_tie']} TP={final['learned_continue_tp']} FP={final['learned_continue_fp']}\n"
        f"sum_dJ_learned={final['sum_delta_j_learned_continue']:.6f}\n"
        f"total_learned_benefit={final['total_learned_benefit_ms_per_token']:.6f} "
        f"required_min={min_benefit:.6f}\n"
        f"schedule_mismatches={final['schedule_state_mismatch_count']} max_continue_score={final['max_continue_score']:.6f}\n"
        "AlwaysSTOP speedup is NOT a criterion. Probe frequency is a SOFT naturalness preference, not a hard gate. Individual learned-C rows may be harmful; only their aggregate delta-J is constrained.\n"
    )
    (out/'VERDICT.txt').write_text(verdict,encoding='utf-8')
    print('\n'+verdict)


if __name__=='__main__': main()
