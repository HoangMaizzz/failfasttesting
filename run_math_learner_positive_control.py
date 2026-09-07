#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ADAPT_IDS = [346,245,129,122,374,56,310,22,16,300,266,152,166,237,326,312,329,3,23,117]
EVAL_IDS = [168,332,368,194,358,229,32,386,133,136,395,141,150,34,164,37,170,174,179,52,53,180,324,77,85,220,247,96,356,101]
FULL_IDS = ADAPT_IDS + EVAL_IDS
TUNE_EVAL_IDS = EVAL_IDS[:10]
TUNE_IDS = ADAPT_IDS + TUNE_EVAL_IDS


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dllm_dir',default='/kaggle/working/Fast_dLLM_v2_1.5B')
    p.add_argument('--output_dir',default='/kaggle/working/outputs_math_learner_positive_control')
    p.add_argument('--target_quantization',default='int8')
    p.add_argument('--target_device',type=int,default=0)
    p.add_argument('--drafter_device',type=int,default=0)
    p.add_argument('--drafter_threshold',type=float,default=.50)
    p.add_argument('--lowconf_threshold',type=float,default=.70)
    p.add_argument('--max_new_tokens',type=int,default=1024)
    p.add_argument('--tune_seed',type=int,default=42)
    p.add_argument('--validate_seed',type=int,default=43)
    p.add_argument('--final_seed',type=int,default=44)
    p.add_argument('--thresholds',type=float,nargs='+',default=[.25,.30,.35,.40,.45,.50])
    p.add_argument('--max_validate_thresholds',type=int,default=3)
    p.add_argument('--tune_min_learned_c',type=int,default=3)
    p.add_argument('--tune_min_tp',type=int,default=1)
    p.add_argument('--validate_min_learned_c',type=int,default=5)
    p.add_argument('--validate_min_tp',type=int,default=2)
    p.add_argument('--bootstrap_samples',type=int,default=5000)
    p.add_argument('--include_probe_control',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--log_level',default='INFO')
    a = p.parse_args()
    if len({a.tune_seed,a.validate_seed,a.final_seed}) != 3:
        p.error('Tuning, validation and final seeds must be distinct')
    if not a.thresholds or any(not math.isfinite(t) or not 0 < t < 1 for t in a.thresholds):
        p.error('Thresholds must be finite and between 0 and 1')
    if any(round(t,2) != t for t in a.thresholds) or len(set(a.thresholds)) != len(a.thresholds):
        p.error('Use unique thresholds with at most two decimal places')
    if min(a.max_validate_thresholds,a.bootstrap_samples,a.tune_min_learned_c,
           a.tune_min_tp,a.validate_min_learned_c,a.validate_min_tp) <= 0:
        p.error('Sample counts and validation limit must be positive')
    return a

def jdump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,allow_nan=True),encoding='utf-8')

def run(cmd, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print('\n'+'='*100+'\nRUN: '+' '.join(map(str,cmd))+'\nLOG: '+str(log)+'\n'+'='*100,flush=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True); f.write(line); f.flush()
        rc=p.wait()
    if rc: raise subprocess.CalledProcessError(rc,cmd)

def base_cmd(a,ids,dest,seed):
    return [sys.executable,'-u','failfast.py','--dataset_name','math','--num_questions',str(len(ids)),'--problem_ids',*map(str,ids),
            '--warmup_questions','1','--benchmark_modes','dllm_ar','--dllm_variant','failfast','--decoding_strategy','greedy',
            '--max_new_tokens',str(a.max_new_tokens),'--spec_len','8','--block_size','32','--small_block_size','8',
            '--target_model_name','Qwen/Qwen2.5-7B-Instruct','--dllm_dir',str(a.dllm_dir),'--target_device',str(a.target_device),
            '--drafter_device',str(a.drafter_device),'--target_quantization',str(a.target_quantization),
            '--unquantized_dtype','float16',
            '--drafter_thresholds',str(a.drafter_threshold),'--sweep_lowconf_threshold',str(a.lowconf_threshold),
            '--sweep_max_spec_len','64','--sweep_incr_len','8','--seed',str(seed),'--quiet_generation','--disable_progress',
            '--skip_artifacts','--skip_plots','--overwrite','--output_dir',str(dest),'--log_level',a.log_level]

def u1_common(cmd):
    cmd += ['--adaptive-td','--adaptive-feature-schema','otrc_v2_2_compact_td','--adaptive-credit-assignment','hindsight_delta_j_logistic_f2',
            '--adaptive-policy-mode','hindsight_delta_j_logistic_f2','--adaptive-hindsight-logistic-tie-ms-per-token','1.0',
            '--no-adaptive-hindsight-logistic-use-class-weight','--no-adaptive-hindsight-logistic-use-prefix-feature',
            '--no-adaptive-hindsight-logistic-dynamic-threshold','--adaptive-hindsight-logistic-utility-weighting','raw_abs',
            '--adaptive-hindsight-logistic-replay-stop-to-continue-ratio','0','--no-adaptive-hindsight-logistic-balance-utility-mass',
            '--adaptive-log-decisions','--adaptive-profile-overhead']
    return cmd

def method_cmd(a,method,ids,dest,seed,tau=.5):
    cmd=base_cmd(a,ids,dest,seed); u1_common(cmd)
    if method=='u1':
        cmd += ['--adaptive-policy-ablation','learned','--adaptive-hindsight-logistic-learning-rate','0.05',
                '--adaptive-hindsight-logistic-continue-threshold',str(tau),'--adaptive-hindsight-delta-j-min-pairs','30',
                '--adaptive-hindsight-delta-j-min-continue-pairs','3','--adaptive-hindsight-logistic-min-positive-problems','2',
                '--adaptive-hindsight-delta-j-structural-probe','0.08','--adaptive-hindsight-delta-j-floor-probe','0.02',
                '--adaptive-hindsight-logistic-replay-batch-size','16','--adaptive-hindsight-logistic-replay-buffer-size','100']
    elif method=='always_stop':
        cmd += ['--adaptive-policy-ablation','frozen_stop','--adaptive-hindsight-logistic-learning-rate','0.0',
                '--adaptive-hindsight-logistic-continue-threshold','0.999999','--adaptive-hindsight-delta-j-min-pairs','0',
                '--adaptive-hindsight-delta-j-min-continue-pairs','0','--adaptive-hindsight-logistic-min-positive-problems','0',
                '--adaptive-hindsight-delta-j-structural-probe','0','--adaptive-hindsight-delta-j-floor-probe','0',
                '--adaptive-hindsight-logistic-replay-batch-size','0','--adaptive-hindsight-logistic-replay-buffer-size','100']
    elif method=='probe_only':
        cmd += ['--adaptive-policy-ablation','learned','--adaptive-hindsight-logistic-learning-rate','0.0',
                '--adaptive-hindsight-logistic-continue-threshold','0.999999','--adaptive-hindsight-delta-j-min-pairs','0',
                '--adaptive-hindsight-delta-j-min-continue-pairs','0','--adaptive-hindsight-logistic-min-positive-problems','0',
                '--adaptive-hindsight-delta-j-structural-probe','0.08','--adaptive-hindsight-delta-j-floor-probe','0.02',
                '--adaptive-hindsight-logistic-replay-batch-size','0','--adaptive-hindsight-logistic-replay-buffer-size','100']
    else: raise ValueError(method)
    return cmd

def complete(case, ids):
    p=case/'benchmark_results.csv'
    if not p.exists(): return False
    try: d=pd.read_csv(p); d=d[d['mode']=='dllm_ar'] if 'mode' in d else d
    except: return False
    return d.problem_id.astype(int).tolist() == list(ids)

def ensure_case(a,method,ids,case,seed,tau=.5):
    if a.resume and complete(case,ids):
        print('[RESUME] reuse',case); return
    if case.exists(): case.rename(case.with_name(case.name+'_incomplete_'+str(time.time_ns())))
    run(method_cmd(a,method,ids,case,seed,tau), Path(a.output_dir)/'logs'/(case.parent.name+'_'+case.name+'.log'))
    if not complete(case,ids): raise RuntimeError(f'Incomplete or reordered output: {case}')

def bools(s):
    if s.dtype==bool:return s.fillna(False)
    return s.astype(str).str.lower().isin(['true','1','yes'])

def load_tr(case):
    p=case/'adaptive_full_stream_transitions.csv'
    try: return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError: return pd.DataFrame()

def load_bench(case):
    d=pd.read_csv(case/'benchmark_results.csv'); return d[d['mode']=='dllm_ar'].copy() if 'mode' in d else d

def auc(y,s):
    d=pd.DataFrame({'y':pd.to_numeric(y,errors='coerce'),'s':pd.to_numeric(s,errors='coerce')}).dropna()
    if d.y.nunique()<2:return float('nan')
    p=(d.y==1).sum(); n=(d.y==0).sum(); ranks=d.s.rank(method='average')
    return float((ranks[d.y==1].sum()-p*(p+1)/2)/(p*n))

def summary(case, eval_ids):
    tr=load_tr(case)
    if tr.empty:
        tr=pd.DataFrame(columns=['pair_resolved','binary_label_C','delta_J_ms_per_token',
                                 'problem_id','action_source','continue_score_before_update'])
    r=tr.copy()
    if 'pair_resolved' in r:r=r[bools(r.pair_resolved)]
    r['binary_label_C']=pd.to_numeric(r.binary_label_C,errors='coerce'); r['delta_J_ms_per_token']=pd.to_numeric(r.delta_J_ms_per_token,errors='coerce')
    all_learned=r[r.problem_id.astype(int).isin(set(eval_ids)) & r.action_source.astype(str).eq('learned_continue') & r.delta_J_ms_per_token.notna()]
    r=r[r.binary_label_C.isin([0,1]) & r.delta_J_ms_per_token.notna()]
    ev=r[r.problem_id.astype(int).isin(set(eval_ids))].copy()
    learned=ev[ev.action_source.astype(str).eq('learned_continue')]
    b=load_bench(case); be=b[b.problem_id.astype(int).isin(set(eval_ids))]
    mspt=float(be.actual_e2e_time_excluding_transfer.sum()*1000/be.output_tokens.sum())
    return {'resolved_non_tie':int(len(ev)),'good_c':int((ev.binary_label_C==1).sum()),'c_rate':float((ev.binary_label_C==1).mean()) if len(ev) else float('nan'),
            'learned_c':int(len(learned)),'learned_c_tp':int((learned.delta_J_ms_per_token<-1).sum()),'learned_c_fp':int((learned.delta_J_ms_per_token>1).sum()),
            'learned_c_ties':int((all_learned.delta_J_ms_per_token.abs()<=1).sum()),
            'sum_delta_j_learned_c':float(all_learned.delta_J_ms_per_token.sum()),'temporal_auc':auc(ev.binary_label_C,ev.continue_score_before_update),
            'e2e_ms_per_token':mspt,'output_tokens':int(be.output_tokens.sum())}

def compare(case_as,case_u1,eval_ids,bootstrap_samples=5000):
    a=load_bench(case_as); u=load_bench(case_u1); ids=set(eval_ids)
    a=a[a.problem_id.astype(int).isin(ids)].copy();u=u[u.problem_id.astype(int).isin(ids)].copy()
    def pooled(d):return float(d.actual_e2e_time_excluding_transfer.sum()*1000/d.output_tokens.sum())
    as_ms,u_ms=pooled(a),pooled(u); speed=as_ms/u_ms
    m=a.merge(u,on='problem_id',suffixes=('_as','_u1'))
    rng=np.random.default_rng(12345); vals=[]
    for _ in range(bootstrap_samples):
        z=m.iloc[rng.integers(0,len(m),len(m))]
        ta=z.actual_e2e_time_excluding_transfer_as.sum()*1000/z.output_tokens_as.sum()
        tu=z.actual_e2e_time_excluding_transfer_u1.sum()*1000/z.output_tokens_u1.sum(); vals.append(ta/tu)
    lo,hi=np.quantile(vals,[.025,.975])
    same=m[m.output_token_hash_as.astype(str)==m.output_token_hash_u1.astype(str)]
    same_speed=float('nan')
    if len(same):
        ta=same.actual_e2e_time_excluding_transfer_as.sum()*1000/same.output_tokens_as.sum();tu=same.actual_e2e_time_excluding_transfer_u1.sum()*1000/same.output_tokens_u1.sum();same_speed=ta/tu
    return {'always_stop_ms_per_token':as_ms,'u1_ms_per_token':u_ms,'speedup_as_over_u1':speed,'bootstrap_ci95':[float(lo),float(hi)],
            'u1_faster_problems':int((m.e2e_ms_per_output_token_excluding_transfer_u1<m.e2e_ms_per_output_token_excluding_transfer_as).sum()),
            'problems':int(len(m)),'exact_hash_matches':int(len(same)),'exact_hash_speedup':same_speed}

def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    config=out/'CONFIG.json'
    if config.exists():
        if not a.resume: raise ValueError('Use --resume or a new output directory')
        old=json.loads(config.read_text())
        for k,v in vars(a).items():
            if k not in {'resume','dllm_dir','output_dir','log_level'} and old.get(k)!=v:
                raise ValueError(f'Resume configuration changed: {k}')
    subprocess.run([sys.executable,'patch_fastdllm_frontier.py',a.dllm_dir],cwd=ROOT,check=True)
    pd.DataFrame([{'run_position':i,'problem_id':pid,'split':'adapt' if i<20 else 'eval'} for i,pid in enumerate(FULL_IDS)]).to_csv(out/'FIXED_SEQUENCE.csv',index=False)
    jdump(out/'CONFIG.json',vars(a))
    print('[SEQUENCE] 20 adaptation IDs:',ADAPT_IDS)
    print('[SEQUENCE] 30 evaluation IDs:',EVAL_IDS)
    print('[TUNE] thresholds:',a.thresholds)

    tune_rows=[]
    for tau in a.thresholds:
        tag=('%.2f'%tau).replace('.','p')
        case=out/'tune'/f'tau_{tag}'
        ensure_case(a,'u1',TUNE_IDS,case,a.tune_seed,tau)
        s=summary(case,TUNE_EVAL_IDS); s.update({'threshold':tau})
        eligible=s.get('learned_c',0)>=a.tune_min_learned_c and s.get('learned_c_tp',0)>=a.tune_min_tp and s.get('sum_delta_j_learned_c',1)>=-1e99 and s.get('sum_delta_j_learned_c',1)<0
        s['eligible']=bool(eligible); tune_rows.append(s)
        pd.DataFrame(tune_rows).to_csv(out/'threshold_tuning.csv',index=False)
        print(f"[TUNE tau={tau:.2f}] learned-C={s.get('learned_c')} TP/FP={s.get('learned_c_tp')}/{s.get('learned_c_fp')} sumdJ={s.get('sum_delta_j_learned_c'):.3f} ms/tok={s.get('e2e_ms_per_token'):.3f} eligible={eligible}")
    tdf=pd.DataFrame(tune_rows); tdf.to_csv(out/'threshold_tuning.csv',index=False)
    elig=tdf[tdf.eligible==True].copy()
    if elig.empty:
        (out/'TUNING_FAILED.txt').write_text('No threshold produced enough learned-C with negative learned-C utility on the 10-problem tuning suffix.\n',encoding='utf-8')
        print('TUNING FAILED: no eligible threshold.'); return
    # Selection uses local utility only; runtime remains a final outcome.
    elig=elig.sort_values(['sum_delta_j_learned_c','threshold'],ascending=[True,False])
    ranked=elig.threshold.tolist()
    print('[TUNE ranking]',ranked)

    chosen=None; val_summary=None
    for tau in ranked[:a.max_validate_thresholds]:
        tag=('%.2f'%tau).replace('.','p'); case=out/'validate'/f'tau_{tag}'
        ensure_case(a,'u1',FULL_IDS,case,a.validate_seed,tau)
        s=summary(case,EVAL_IDS); passed=s.get('learned_c',0)>=a.validate_min_learned_c and s.get('learned_c_tp',0)>=a.validate_min_tp and s.get('sum_delta_j_learned_c',1)<0
        s.update({'threshold':tau,'passed':bool(passed)}); jdump(case/'VALIDATION_SUMMARY.json',s)
        print(f"[VALIDATE tau={tau:.2f}] learned-C={s['learned_c']} TP/FP={s['learned_c_tp']}/{s['learned_c_fp']} sumdJ={s['sum_delta_j_learned_c']:.3f} ms/tok={s['e2e_ms_per_token']:.3f} PASS={passed}")
        if passed: chosen=float(tau); val_summary=s; break
    if chosen is None:
        (out/'VALIDATION_FAILED.txt').write_text('No tuned threshold validated with useful learned-C on seed 43.\n',encoding='utf-8')
        print('VALIDATION FAILED. Final AlwaysSTOP comparison was not run.'); return
    jdump(out/'FROZEN_POLICY.json',{'threshold':chosen,'adapt_ids':ADAPT_IDS,'eval_ids':EVAL_IDS,'tune_seed':a.tune_seed,'validate_seed':a.validate_seed,'final_seed':a.final_seed,'validation':val_summary})

    final=out/'final'; as_case=final/'always_stop'; u1_case=final/'u1'
    ensure_case(a,'always_stop',FULL_IDS,as_case,a.final_seed,chosen)
    ensure_case(a,'u1',FULL_IDS,u1_case,a.final_seed,chosen)
    probe_case=None
    if a.include_probe_control:
        probe_case=final/'probe_only'; ensure_case(a,'probe_only',FULL_IDS,probe_case,a.final_seed,chosen)
    us=summary(u1_case,EVAL_IDS); comp=compare(as_case,u1_case,EVAL_IDS,a.bootstrap_samples)
    result={'threshold':chosen,'u1_eval':us,'comparison':comp}
    if probe_case is not None:
        pb=summary(probe_case,EVAL_IDS); result['probe_eval']=pb
        result['u1_vs_probe_speedup']=pb['e2e_ms_per_token']/us['e2e_ms_per_token']
    pass_core=comp['speedup_as_over_u1']>1 and us['learned_c']>=a.validate_min_learned_c and us['learned_c_tp']>=a.validate_min_tp and us['sum_delta_j_learned_c']<0
    result['PASS_LEARNER_BEATS_ALWAYS_STOP']=bool(pass_core)
    result['all_outputs_match']=bool(comp['exact_hash_matches']==comp['problems'])
    result['interpretation']='Post-hoc selected-problem positive control; final seed held out, problems not held out. PASS is directional, not a significance or losslessness claim.'
    jdump(out/'FINAL_RESULT.json',result)
    lines=[f"threshold={chosen:.2f}",f"AlwaysSTOP ms/token={comp['always_stop_ms_per_token']:.6f}",f"U1 ms/token={comp['u1_ms_per_token']:.6f}",f"speedup AS/U1={comp['speedup_as_over_u1']:.6f}x",f"bootstrap95={comp['bootstrap_ci95']}",f"learned-C={us['learned_c']} TP/FP={us['learned_c_tp']}/{us['learned_c_fp']}",f"sum dJ learned-C={us['sum_delta_j_learned_c']:.6f}",f"exact hashes={comp['exact_hash_matches']}/{comp['problems']} same-hash speedup={comp['exact_hash_speedup']}",f"PASS_LEARNER_BEATS_ALWAYS_STOP={pass_core}"]
    if probe_case is not None: lines.append(f"U1 vs probe-only speedup={result['u1_vs_probe_speedup']:.6f}x")
    (out/'FINAL_VERDICT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'+'\n'.join(lines))
if __name__=='__main__': main()
