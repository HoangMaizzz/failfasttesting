#!/usr/bin/env python3
"""
Held-out matched GSM8K INT8 test for the predeclared utility-balanced OTRC variant:

    utility-balanced U1 + fixed CONTINUE threshold tau = 0.65

Primary comparison:
    Always-STOP vs U1-balanced-tau065

Optional third baseline:
    FailFast

IMPORTANT
---------
1. Target/verifier is INT8 by default. Drafter is NOT quantized.
2. Verifier uses the existing full-prefix `use_cache=False` path in failfast.py.
3. U1-balanced-tau065 keeps the current U1 ingredients:
       - F2 state
       - raw |delta J| utility weighting
       - replay buffer K=100
       - replay minibatch B=16
       - exactly one SGD minibatch update per newly resolved non-tie pair
       - structural probe 0.08, floor probe 0.02
   and adds ONLY:
       - equal C/S utility mass inside each sampled replay minibatch
       - fixed decision threshold tau=0.65
4. The default id_offset=125 is intentionally held out from the old U1-100 run
   whose runner used PROBLEM_IDS[gsm8k][25:125].  Do not change this if the
   purpose is prospective validation of tau=0.65.
5. No hindsight filtering or favorable-example selection is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ("always_stop", "u1_balanced_tau065", "failfast")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["always_stop", "u1_balanced_tau065"],
    )
    p.add_argument("--num_questions", type=int, default=100)
    p.add_argument(
        "--id_offset",
        type=int,
        default=125,
        help=(
            "Default 125 is a fresh held-out slice after the old 100-problem "
            "U1 run at offset 25. Keep 125 for prospective validation."
        ),
    )
    p.add_argument("--target_quantization", default="int8")
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=0)
    p.add_argument("--drafter_threshold", type=float, default=0.50)
    p.add_argument("--lowconf_threshold", type=float, default=0.70)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    p.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_gsm8k100_int8_u1_balanced_tau065_vs_alwaysstop"
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    if args.num_questions <= 0:
        p.error("--num_questions must be positive")
    if args.target_quantization != "int8":
        print(
            f"WARNING: predeclared experiment is INT8, got {args.target_quantization!r}",
            file=sys.stderr,
        )
    return args


def run_checked(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 100, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def selected_problem_ids(args: argparse.Namespace) -> list[int]:
    # Extend the actual old U1 ordering, not the 50-entry seed list.
    from run_u1_sgd_ablation import selected_ids
    start = int(args.id_offset) - 25
    if start < 0:
        raise ValueError("id_offset must be >= 25")
    stop = start + int(args.num_questions)
    all_ids = selected_ids(SimpleNamespace(id_offset=25, num_questions=stop), "gsm8k")
    chosen = all_ids[start:stop]
    if len(chosen) != args.num_questions:
        raise ValueError(
            f"PROBLEM_IDS['gsm8k'] has only {len(chosen)} IDs in slice "
            f"[{start}:{stop}]"
        )
    return [int(x) for x in chosen]


def common_command(
    args: argparse.Namespace,
    method: str,
    destination: Path,
    ids: list[int],
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", "gsm8k",
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
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(destination),
        "--log_level", args.log_level,
    ]

    if method == "failfast":
        return cmd

    # Shared hindsight-logistic path. Pin every relevant setting so that source
    # defaults cannot silently change the experiment.
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
        # IMPORTANT: 0 = ordinary uniform replay. This disables the old 3:1
        # replay-composition ablation if the source default was changed.
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

    if method == "u1_balanced_tau065":
        cmd.extend([
            "--adaptive-policy-ablation", "learned",
            # PREDECLARED fixed boundary from the prior logged diagnostic.
            "--adaptive-hindsight-logistic-continue-threshold", "0.65",
            "--adaptive-hindsight-delta-j-min-pairs", "30",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
            "--adaptive-hindsight-logistic-min-positive-problems", "2",
            "--adaptive-hindsight-delta-j-structural-probe", "0.08",
            "--adaptive-hindsight-delta-j-floor-probe", "0.02",
            "--adaptive-hindsight-logistic-replay-batch-size", "16",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
            # New part: after the ordinary uniform minibatch has been sampled,
            # rescale raw-|delta J| weights so C and S contribute equal TOTAL
            # utility mass, while preserving relative |delta J| within each class.
            "--adaptive-hindsight-logistic-balance-utility-mass",
            "--adaptive-hindsight-logistic-balance-min-per-class", "1",
        ])
        return cmd

    raise ValueError(method)


def load_benchmark(case_dir: Path) -> pd.DataFrame:
    p = case_dir / "benchmark_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "mode" in df.columns:
        df = df[df["mode"] == "dllm_ar"].copy()
    return df


def case_complete(case_dir: Path, n: int) -> bool:
    try:
        df = load_benchmark(case_dir)
    except Exception:
        return False
    return len(df) == n and df["problem_id"].nunique() == n


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda x: str(x).strip().lower() in {"1", "true", "yes"})


def _auc(labels: pd.Series, scores: pd.Series) -> float:
    f = pd.DataFrame({
        "label": pd.to_numeric(labels, errors="coerce"),
        "score": pd.to_numeric(scores, errors="coerce"),
    }).dropna()
    if f.empty:
        return float("nan")
    p = int(f["label"].sum())
    n = len(f) - p
    if p == 0 or n == 0:
        return float("nan")
    ranks = f["score"].rank(method="average")
    return float((ranks[f["label"] == 1].sum() - p * (p + 1) / 2) / (p * n))


def aggregate_row(method: str, case_dir: Path, ordered_ids: list[int]) -> dict:
    df = load_benchmark(case_dir)
    tokens = float(df["output_tokens"].sum())
    algo_s = float(df["actual_algorithm_time"].sum())
    e2e_s = (
        float(df["actual_e2e_time_excluding_transfer"].sum())
        if "actual_e2e_time_excluding_transfer" in df.columns
        else algo_s
    )
    out = {
        "method": method,
        "questions": int(df["problem_id"].nunique()),
        "output_tokens": int(tokens),
        "algorithm_time_s": algo_s,
        "algorithm_ms_per_output_token": 1000.0 * algo_s / max(tokens, 1.0),
        "e2e_excl_transfer_s": e2e_s,
        "e2e_excl_transfer_ms_per_output_token": 1000.0 * e2e_s / max(tokens, 1.0),
        "draft_time_s": float(df["actual_draft_time"].sum()),
        "verify_time_s": float(df["actual_verify_time"].sum()),
        "draft_forwards": int(df["total_num_forward_passes"].sum()),
        "verifier_rounds": int(df["num_speculation_rounds"].sum()),
        "acceptance_rate_percent": (
            100.0 * float(df["accepted_tokens"].sum())
            / max(1.0, float(df["drafted_tokens"].sum()))
        ),
        "accuracy_percent": (
            100.0 * float(_bool_series(df["is_correct"]).mean())
            if "is_correct" in df.columns else float("nan")
        ),
    }

    tr_path = case_dir / "adaptive_full_stream_transitions.csv"
    if tr_path.exists():
        tr = pd.read_csv(tr_path)
        if "update_applied" in tr.columns:
            tr = tr[_bool_series(tr["update_applied"])].copy()
        if not tr.empty and "binary_label_C" in tr.columns:
            y = pd.to_numeric(tr["binary_label_C"], errors="coerce")
            s = pd.to_numeric(tr["continue_score_before_update"], errors="coerce")
            out["resolved_non_tie_pairs"] = int(y.notna().sum())
            out["good_C_pairs"] = int((y == 1).sum())
            out["good_C_rate_percent"] = 100.0 * float((y == 1).mean())
            out["temporal_auc"] = _auc(y, s)

            pos = {int(pid): i for i, pid in enumerate(ordered_ids)}
            tr["_problem_pos"] = tr["problem_id"].map(pos)
            split = len(ordered_ids) / 2.0
            first = tr[tr["_problem_pos"] < split]
            second = tr[tr["_problem_pos"] >= split]
            out["temporal_auc_first_half"] = _auc(
                first["binary_label_C"], first["continue_score_before_update"]
            )
            out["temporal_auc_second_half"] = _auc(
                second["binary_label_C"], second["continue_score_before_update"]
            )

            src = tr.get("action_source", pd.Series("", index=tr.index)).astype(str)
            learned = tr[src == "learned_continue"].copy()
            out["resolved_learned_C"] = int(len(learned))
            if len(learned):
                dj = pd.to_numeric(learned["delta_J_ms_per_token"], errors="coerce")
                ly = pd.to_numeric(learned["binary_label_C"], errors="coerce")
                out["learned_C_TP"] = int((ly == 1).sum())
                out["learned_C_FP"] = int((ly == 0).sum())
                out["sum_delta_J_learned_C"] = float(dj.sum())
                out["mean_delta_J_learned_C"] = float(dj.mean())
            else:
                out["learned_C_TP"] = 0
                out["learned_C_FP"] = 0
                out["sum_delta_J_learned_C"] = 0.0

            if "utility_balance_applied" in tr.columns:
                ub = _bool_series(tr["utility_balance_applied"])
                out["utility_balance_updates"] = int(ub.sum())
                if ub.any():
                    out["mean_balance_C_scale"] = float(pd.to_numeric(
                        tr.loc[ub, "utility_balance_continue_scale"], errors="coerce"
                    ).mean())
                    out["mean_balance_S_scale"] = float(pd.to_numeric(
                        tr.loc[ub, "utility_balance_stop_scale"], errors="coerce"
                    ).mean())

    dec_path = case_dir / "adaptive_td_decisions.csv"
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        src = dec.get("action_source", pd.Series("", index=dec.index)).fillna("").astype(str)
        out["decisions"] = int(len(dec))
        out["learned_stop_decisions"] = int((src == "learned_stop").sum())
        out["learned_continue_decisions"] = int((src == "learned_continue").sum())
        out["structural_probes"] = int((src == "structural_probe").sum())
        out["floor_probes"] = int((src == "floor_probe").sum())
        out["frozen_stop_decisions"] = int((src == "frozen_stop_control").sum())
    return out


def paired_bootstrap(
    always_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    fields = ["problem_id", "output_tokens", "actual_algorithm_time"]
    if "output_token_hash" in always_df.columns and "output_token_hash" in candidate_df.columns:
        fields.append("output_token_hash")
    a = always_df[fields].copy().rename(columns={
        "output_tokens": "tokens_a",
        "actual_algorithm_time": "time_a",
        "output_token_hash": "hash_a",
    })
    c = candidate_df[fields].copy().rename(columns={
        "output_tokens": "tokens_c",
        "actual_algorithm_time": "time_c",
        "output_token_hash": "hash_c",
    })
    m = a.merge(c, on="problem_id", how="inner")
    if m.empty:
        return {}

    def speed(frame: pd.DataFrame) -> float:
        ms_a = 1000.0 * frame["time_a"].sum() / max(1.0, frame["tokens_a"].sum())
        ms_c = 1000.0 * frame["time_c"].sum() / max(1.0, frame["tokens_c"].sum())
        return float(ms_a / ms_c)

    point = speed(m)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    n = len(m)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = speed(m.iloc[idx])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    out = {
        "paired_problems": int(n),
        "speedup_u1_balanced_tau065_vs_always_stop": point,
        "bootstrap95_low": float(lo),
        "bootstrap95_high": float(hi),
        "u1_faster_problem_count": int((m["time_c"] < m["time_a"]).sum()),
    }
    if "hash_a" in m.columns and "hash_c" in m.columns:
        exact = m[m["hash_a"] == m["hash_c"]]
        out["exact_hash_matches"] = int(len(exact))
        if len(exact):
            out["exact_hash_speedup"] = speed(exact)
    return out


def main() -> None:
    args = parse_args()
    ids = selected_problem_ids(args)
    subprocess.run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir],
                   cwd=ROOT, check=True)
    root = Path(args.output_dir)
    raw = root / "raw" / "gsm8k"
    raw.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "gsm8k",
        "num_questions": args.num_questions,
        "problem_ids": ids,
        "id_offset": args.id_offset,
        "target_quantization": args.target_quantization,
        "drafter_quantized": False,
        "verifier_kv_cache": False,
        "drafter_threshold": args.drafter_threshold,
        "lowconf_threshold": args.lowconf_threshold,
        "methods": args.methods,
        "u1_balanced_tau065": {
            "state": "F2=[mask_ratio,global_proposal_position]+intercept",
            "utility_weighting": "raw_abs_delta_J",
            "replay_batch_size": 16,
            "replay_buffer_size": 100,
            "updates_per_resolved_pair": 1,
            "replay_stop_to_continue_ratio": 0,
            "utility_mass_balance": "equal C/S total mass inside sampled minibatch",
            "continue_threshold": 0.65,
            "structural_probe": 0.08,
            "floor_probe": 0.02,
            "dynamic_threshold": False,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for method in args.methods:
        case = raw / method
        if args.resume and case_complete(case, args.num_questions):
            print(f"SKIP complete: {method}")
            continue
        cmd = common_command(args, method, case, ids)
        run_checked(cmd, ROOT, root / f"{method}.log")

    rows = []
    for method in args.methods:
        case = raw / method
        if case_complete(case, args.num_questions):
            rows.append(aggregate_row(method, case, ids))
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "dataset_method_summary.csv", index=False)
    print("\nSUMMARY\n", summary.to_string(index=False))

    comparison = {}
    if {"always_stop", "u1_balanced_tau065"}.issubset(set(args.methods)):
        comparison = paired_bootstrap(
            load_benchmark(raw / "always_stop"),
            load_benchmark(raw / "u1_balanced_tau065"),
            args.bootstrap_samples,
            args.seed,
        )
        (root / "paired_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )

        speed = comparison.get("speedup_u1_balanced_tau065_vs_always_stop", float("nan"))
        lo = comparison.get("bootstrap95_low", float("nan"))
        hi = comparison.get("bootstrap95_high", float("nan"))
        row = summary[summary["method"] == "u1_balanced_tau065"]
        sum_dj = (
            float(row["sum_delta_J_learned_C"].iloc[0])
            if len(row) and "sum_delta_J_learned_C" in row.columns
            else float("nan")
        )
        learned_c = (
            int(row["learned_continue_decisions"].iloc[0])
            if len(row) and "learned_continue_decisions" in row.columns
            else 0
        )
        if math.isfinite(lo) and lo > 1.0:
            verdict = "STRONG PASS: U1-balanced-tau065 beats Always-STOP and paired 95% CI > 1."
        elif math.isfinite(speed) and speed > 1.0:
            verdict = "DIRECTIONAL PASS: U1-balanced-tau065 is faster, but paired 95% CI crosses/touches 1."
        else:
            verdict = "FAIL ON THIS RUN: U1-balanced-tau065 does not beat Always-STOP E2E."

        text = "\n".join([
            "PRIMARY TEST: GSM8K-100 held-out, INT8 verifier, full-prefix target path",
            "",
            f"speedup U1-balanced-tau065 vs Always-STOP = {speed:.6f}x",
            f"paired bootstrap 95% CI = [{lo:.6f}, {hi:.6f}]",
            f"learned CONTINUE decisions = {learned_c}",
            f"sum delta_J over resolved learned-C = {sum_dj:.6f} (negative is locally beneficial)",
            "",
            verdict,
            "",
            "Do NOT convert sum(delta_J) into E2E milliseconds; it is a local utility diagnostic.",
        ])
        (root / "VERDICT.txt").write_text(text, encoding="utf-8")
        print("\n" + text)


if __name__ == "__main__":
    main()
