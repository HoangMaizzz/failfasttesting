#!/usr/bin/env python3
"""MATH-30 C-rich positive-control test for current U1 fixed-tau=0.5 method.

Purpose
-------
Test whether the CURRENT method can outperform Always-STOP in a deliberately
refinement-sensitive workload where beneficial CONTINUE states are much more
common than in the full benchmark.

This is a positive-control / existence test, NOT an unbiased MATH benchmark.
The 30 IDs were frozen from a PRIOR INT8/full-prefix MATH U1-fixed0.5 archive,
using only hindsight C/S labels (never U1-vs-AlwaysSTOP runtime):
  - resolved non-tie support >= 2 per problem
  - at least one Good-C (delta_J < -1 ms/output-token)
  - take top 30 by Good-C rate, tie-break by beneficial-C utility mass/support
  - restore original prior-stream order for the rerun

Frozen discovery pool statistics:
  - 114 resolved non-tie pairs
  - 39 Good-C = 34.21%
  - sum(delta_J) over all 114 = -168.43 (local diagnostic only)

Online protocol
---------------
Both methods run all 30 IDs in the same frozen order. U1 starts from zero.
The first 10 problems are an adaptation prefix; the final 20 are the primary
post-adaptation evaluation slice. Full-30 metrics are also reported.

Current/original U1 method on the frozen C-rich MATH-30 stream:
  F2 + raw |delta_J| + uniform replay B=16/K=100
  + NO utility-mass balancing + fixed tau=0.5
  + structural probe 0.08 + floor probe 0.02.

Target/verifier defaults to INT8 full-prefix; drafter is not quantized.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ID_CSV = ROOT / "math30_crich_frozen_ids.csv"
METHODS = ("always_stop", "u1_current_tau05", "failfast")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", choices=METHODS,
                   default=["always_stop", "u1_current_tau05"])
    p.add_argument("--target_quantization", default="int8")
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=0)
    p.add_argument("--drafter_threshold", type=float, default=0.50)
    p.add_argument("--lowconf_threshold", type=float, default=0.70)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--adaptation_problems", type=int, default=10)
    p.add_argument("--dllm_dir",
                   default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B")
    p.add_argument("--output_dir",
                   default="outputs_math30_crich_tau05_vs_alwaysstop")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_level", default="INFO")
    a = p.parse_args()
    if a.bootstrap_samples <= 0:
        p.error("--bootstrap_samples must be positive")
    if not 0 <= a.adaptation_problems < 30:
        p.error("--adaptation_problems must be in [0,29]")
    if a.target_quantization != "int8":
        print("WARNING: this positive-control was designed for INT8/full-prefix.", file=sys.stderr)
    return a


def load_frozen_ids() -> tuple[list[int], pd.DataFrame]:
    if not ID_CSV.exists():
        raise FileNotFoundError(ID_CSV)
    meta = pd.read_csv(ID_CSV).sort_values("run_position")
    ids = meta["problem_id"].astype(int).tolist()
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("frozen ID file must contain exactly 30 unique problems")
    return ids, meta


def run_checked(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 100, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def common_command(args: argparse.Namespace, method: str,
                   destination: Path, ids: list[int]) -> list[str]:
    cmd = [
        sys.executable, "-u", "failfast.py",
        "--dataset_name", "math",
        "--num_questions", str(len(ids)),
        "--problem_ids", *map(str, ids),
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--unquantized_dtype", "float16",
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--seed", str(args.seed),
        "--quiet_generation", "--disable_progress", "--skip_artifacts", "--skip_plots",
        "--overwrite", "--output_dir", str(destination), "--log_level", args.log_level,
    ]
    if method == "failfast":
        return cmd

    cmd.extend([
        "--adaptive-td",
        "--adaptive-feature-schema", "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment", "hindsight_delta_j_logistic_f2",
        "--adaptive-policy-mode", "hindsight_delta_j_logistic_f2",
        "--adaptive-hindsight-logistic-learning-rate", "0.05",
        "--adaptive-hindsight-logistic-tie-ms-per-token", "1.0",
        "--no-adaptive-hindsight-logistic-use-class-weight",
        "--no-adaptive-hindsight-logistic-use-prefix-feature",
        "--no-adaptive-hindsight-logistic-dynamic-threshold",
        "--adaptive-hindsight-logistic-utility-weighting", "raw_abs",
        "--adaptive-hindsight-logistic-replay-stop-to-continue-ratio", "0",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
    ])

    if method == "always_stop":
        cmd.extend([
            "--adaptive-policy-ablation", "frozen_stop",
            "--adaptive-hindsight-logistic-continue-threshold", "0.999999",
            "--adaptive-hindsight-delta-j-min-pairs", "0",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "0",
            "--adaptive-hindsight-logistic-min-positive-problems", "0",
            "--adaptive-hindsight-delta-j-structural-probe", "0",
            "--adaptive-hindsight-delta-j-floor-probe", "0",
            "--adaptive-hindsight-logistic-replay-batch-size", "0",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
            "--no-adaptive-hindsight-logistic-balance-utility-mass",
        ])
        return cmd

    if method == "u1_current_tau05":
        # Original/current U1 fixed-0.5 policy: raw |delta_J|, uniform replay,
        # no class weight, no prefix feature, no dynamic threshold, no utility-mass balance.
        cmd.extend([
            "--adaptive-policy-ablation", "learned",
            "--adaptive-hindsight-logistic-continue-threshold", "0.5",
            "--adaptive-hindsight-delta-j-min-pairs", "30",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
            "--adaptive-hindsight-logistic-min-positive-problems", "2",
            "--adaptive-hindsight-delta-j-structural-probe", "0.08",
            "--adaptive-hindsight-delta-j-floor-probe", "0.02",
            "--adaptive-hindsight-logistic-replay-batch-size", "16",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
            "--no-adaptive-hindsight-logistic-balance-utility-mass",
        ])
        return cmd
    raise ValueError(method)


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda x: str(x).strip().lower() in {"1", "true", "yes"})


def _auc(labels: pd.Series, scores: pd.Series) -> float:
    f = pd.DataFrame({"label": pd.to_numeric(labels, errors="coerce"),
                      "score": pd.to_numeric(scores, errors="coerce")}).dropna()
    if f.empty:
        return float("nan")
    p = int(f.label.sum()); n = len(f)-p
    if p == 0 or n == 0:
        return float("nan")
    ranks = f.score.rank(method="average")
    return float((ranks[f.label==1].sum()-p*(p+1)/2)/(p*n))


def load_benchmark(case: Path) -> pd.DataFrame:
    df = pd.read_csv(case / "benchmark_results.csv")
    if "mode" in df.columns:
        df = df[df["mode"] == "dllm_ar"].copy()
    return df


def case_complete(case: Path, ids: list[int]) -> bool:
    try:
        df = load_benchmark(case)
    except Exception:
        return False
    return set(df.problem_id.astype(int)) == set(ids) and len(df) == len(ids)


def subset_aggregate(method: str, case: Path, ids: list[int], label: str) -> dict:
    all_df = load_benchmark(case)
    df = all_df[all_df.problem_id.astype(int).isin(set(ids))].copy()
    tokens = float(df.output_tokens.sum())
    e2e_col = "actual_e2e_time_excluding_transfer" if "actual_e2e_time_excluding_transfer" in df.columns else "actual_algorithm_time"
    e2e_s = float(df[e2e_col].sum())
    algo_s = float(df.actual_algorithm_time.sum())
    out = {
        "slice": label, "method": method, "questions": int(df.problem_id.nunique()),
        "output_tokens": int(tokens),
        "e2e_excl_transfer_s": e2e_s,
        "e2e_excl_transfer_ms_per_output_token": 1000*e2e_s/max(tokens,1),
        "algorithm_ms_per_output_token": 1000*algo_s/max(tokens,1),
        "draft_forwards": int(df.total_num_forward_passes.sum()),
        "verifier_rounds": int(df.num_speculation_rounds.sum()),
        "acceptance_rate_percent": 100*float(df.accepted_tokens.sum())/max(1,float(df.drafted_tokens.sum())),
        "accuracy_percent": 100*float(_bool_series(df.is_correct).mean()) if "is_correct" in df.columns else float("nan"),
    }
    trp = case / "adaptive_full_stream_transitions.csv"
    if trp.exists():
        tr = pd.read_csv(trp)
        tr = tr[tr.problem_id.astype(int).isin(set(ids))].copy()
        if "update_applied" in tr.columns:
            tr = tr[_bool_series(tr.update_applied)]
        if len(tr) and "binary_label_C" in tr.columns:
            y = pd.to_numeric(tr.binary_label_C, errors="coerce")
            s = pd.to_numeric(tr.continue_score_before_update, errors="coerce")
            out["resolved_non_tie_pairs"] = int(y.notna().sum())
            out["good_C_pairs"] = int((y==1).sum())
            out["good_C_rate_percent"] = 100*float((y==1).mean())
            out["temporal_auc"] = _auc(y,s)
            src = tr.get("action_source", pd.Series("", index=tr.index)).fillna("").astype(str)
            learned = tr[src=="learned_continue"]
            out["resolved_learned_C"] = int(len(learned))
            if len(learned):
                ly = pd.to_numeric(learned.binary_label_C, errors="coerce")
                dj = pd.to_numeric(learned.delta_J_ms_per_token, errors="coerce")
                out["learned_C_TP"] = int((ly==1).sum())
                out["learned_C_FP"] = int((ly==0).sum())
                out["sum_delta_J_learned_C"] = float(dj.sum())
            else:
                out["learned_C_TP"] = out["learned_C_FP"] = 0
                out["sum_delta_J_learned_C"] = 0.0
    decp = case / "adaptive_td_decisions.csv"
    if decp.exists():
        dec = pd.read_csv(decp)
        dec = dec[dec.problem_id.astype(int).isin(set(ids))].copy()
        src = dec.get("action_source", pd.Series("", index=dec.index)).fillna("").astype(str)
        out["learned_continue_decisions"] = int((src=="learned_continue").sum())
        out["learned_stop_decisions"] = int((src=="learned_stop").sum())
        out["structural_probes"] = int((src=="structural_probe").sum())
        out["floor_probes"] = int((src=="floor_probe").sum())
    return out


def paired_comparison(always: pd.DataFrame, cand: pd.DataFrame, ids: list[int], n_boot: int, seed: int) -> dict:
    cols = ["problem_id","output_tokens","actual_algorithm_time"]
    e2e = "actual_e2e_time_excluding_transfer"
    if e2e in always.columns and e2e in cand.columns:
        cols.append(e2e)
    if "output_token_hash" in always.columns and "output_token_hash" in cand.columns:
        cols.append("output_token_hash")
    a = always[always.problem_id.astype(int).isin(set(ids))][cols].copy()
    c = cand[cand.problem_id.astype(int).isin(set(ids))][cols].copy()
    a = a.rename(columns={"output_tokens":"tok_a","actual_algorithm_time":"algo_a",e2e:"e2e_a","output_token_hash":"hash_a"})
    c = c.rename(columns={"output_tokens":"tok_c","actual_algorithm_time":"algo_c",e2e:"e2e_c","output_token_hash":"hash_c"})
    m = a.merge(c,on="problem_id",how="inner")
    time_a = "e2e_a" if "e2e_a" in m.columns else "algo_a"
    time_c = "e2e_c" if "e2e_c" in m.columns else "algo_c"
    def speed(f):
        msa = 1000*f[time_a].sum()/max(1,f.tok_a.sum())
        msc = 1000*f[time_c].sum()/max(1,f.tok_c.sum())
        return float(msa/msc)
    point = speed(m)
    rng=np.random.default_rng(seed); vals=np.empty(n_boot)
    for i in range(n_boot):
        vals[i]=speed(m.iloc[rng.integers(0,len(m),size=len(m))])
    lo,hi=np.quantile(vals,[.025,.975])
    out={"paired_problems":int(len(m)),"speedup_vs_always_stop":point,
         "bootstrap95_low":float(lo),"bootstrap95_high":float(hi),
         "u1_faster_problem_count":int(((m[time_c]/m.tok_c)<(m[time_a]/m.tok_a)).sum())}
    if "hash_a" in m.columns:
        exact=m[m.hash_a==m.hash_c]
        out["exact_hash_matches"]=int(len(exact))
        out["exact_hash_speedup"]=speed(exact) if len(exact) else float("nan")
    return out


def main() -> None:
    args=parse_args(); ids, meta=load_frozen_ids()
    subprocess.run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir],
                   cwd=ROOT, check=True)
    adapt_ids=ids[:args.adaptation_problems]; eval_ids=ids[args.adaptation_problems:]
    root=Path(args.output_dir); raw=root/"raw"/"math"; raw.mkdir(parents=True,exist_ok=True)
    manifest={
        "dataset":"math","positive_control":True,"frozen_ids":ids,
        "adaptation_ids":adapt_ids,"evaluation_ids":eval_ids,
        "selection_rule":"prior INT8 U1-fixed0.5 labels only: support>=2, >=1 Good-C; top30 by C-rate then Good-C utility mass/support; restored prior stream order",
        "discovery_pooled":{"non_tie":int(meta.resolved_non_tie.sum()),"good_C":int(meta.good_C.sum()),
                            "good_C_rate":float(meta.good_C.sum()/meta.resolved_non_tie.sum()),
                            "sum_delta_J_all":float(meta.sum_delta_J_all.sum())},
        "target_quantization":args.target_quantization,"drafter_quantized":False,
        "verifier_kv_cache":False,"methods":args.methods,
        "u1":{"state":"F2","utility":"raw_abs_delta_J","replay_B":16,"buffer_K":100,
              "utility_mass_balance":False,"tau":0.5,"structural_probe":0.08,"floor_probe":0.02},
    }
    root.mkdir(parents=True,exist_ok=True)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    meta.to_csv(root/"frozen_problem_ids_with_discovery_stats.csv",index=False)

    for method in args.methods:
        case=raw/method
        if args.resume and case_complete(case,ids):
            print("SKIP complete:",method); continue
        run_checked(common_command(args,method,case,ids),ROOT,root/f"{method}.log")

    rows=[]
    for method in args.methods:
        case=raw/method
        if case_complete(case,ids):
            rows.append(subset_aggregate(method,case,ids,"full30"))
            rows.append(subset_aggregate(method,case,eval_ids,f"eval_after_{len(adapt_ids)}_adapt"))
    summary=pd.DataFrame(rows); summary.to_csv(root/"slice_method_summary.csv",index=False)
    print("\nSUMMARY\n",summary.to_string(index=False))

    comparisons={}
    if {"always_stop","u1_current_tau05"}.issubset(set(args.methods)):
        a=load_benchmark(raw/"always_stop"); c=load_benchmark(raw/"u1_current_tau05")
        comparisons["full30"]=paired_comparison(a,c,ids,args.bootstrap_samples,args.seed)
        comparisons["post_adaptation_eval"]=paired_comparison(a,c,eval_ids,args.bootstrap_samples,args.seed+1)
        (root/"paired_comparisons.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")

        ev=comparisons["post_adaptation_eval"]; full=comparisons["full30"]
        if math.isfinite(ev["bootstrap95_low"]) and ev["bootstrap95_low"]>1:
            verdict="STRONG PASS: post-adaptation C-rich evaluation beats Always-STOP with paired 95% CI > 1."
        elif ev["speedup_vs_always_stop"]>1:
            verdict="DIRECTIONAL PASS: post-adaptation C-rich evaluation is faster than Always-STOP, but CI crosses/touches 1."
        else:
            verdict="FAIL ON THIS POSITIVE-CONTROL RUN: current U1 tau=0.5 method did not beat Always-STOP on the post-adaptation slice."
        text="\n".join([
            "MATH-30 C-RICH POSITIVE CONTROL — U1 current tau=0.5 (post-hoc label-enriched; not a standard benchmark)",
            f"Discovery Good-C prevalence: {int(meta.good_C.sum())}/{int(meta.resolved_non_tie.sum())} = {100*meta.good_C.sum()/meta.resolved_non_tie.sum():.2f}%",
            f"Adaptation prefix: {len(adapt_ids)} problems; primary evaluation: {len(eval_ids)} problems",
            "",
            f"Full-30 speedup U1 tau=.5 vs Always-STOP = {full['speedup_vs_always_stop']:.6f}x",
            f"Full-30 paired 95% CI = [{full['bootstrap95_low']:.6f}, {full['bootstrap95_high']:.6f}]",
            f"Post-adaptation speedup = {ev['speedup_vs_always_stop']:.6f}x",
            f"Post-adaptation paired 95% CI = [{ev['bootstrap95_low']:.6f}, {ev['bootstrap95_high']:.6f}]",
            f"Post-adaptation exact-hash matches = {ev.get('exact_hash_matches','NA')}/{len(eval_ids)}",
            "",
            verdict,
            "",
            "Interpretation scope: existence/positive-control only. IDs were selected by prior Good-C labels, not by runtime wins.",
            "sum(delta_J) is a local diagnostic and must not be converted into E2E milliseconds.",
        ])
        (root/"POSITIVE_CONTROL_VERDICT.txt").write_text(text,encoding="utf-8")
        print("\n"+text)

if __name__=="__main__":
    main()
