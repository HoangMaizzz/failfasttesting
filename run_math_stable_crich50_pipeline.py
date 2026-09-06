#!/usr/bin/env python3
"""Stable C-rich MATH positive-control builder + final E2E test.

Goal
----
Build a 50-problem MATH sequence where the *current* U1 learner (F2, raw |dJ|,
uniform replay B16/K100, tau=.5, no utility-mass balancing) actually sees a
substantially C-rich on-policy stream, learns non-trivial CONTINUE actions, and
can then be compared fairly against matched Always-STOP.

This is deliberately a positive-control / existence test, not an unbiased MATH
benchmark. The selection stage uses hindsight/oracle-style labels. Crucially,
selection does NOT use U1-vs-AlwaysSTOP runtime wins.

Pipeline
--------
A. Oracle-style screening (cheap-ish, no learning):
   - scan fresh MATH ids in batches;
   - logistic LR=0, tau=.999999, probe=1.0 so every legal STOP opportunity is
     forced to CONTINUE often enough to resolve the exact same hindsight dJ label
     used by U1;
   - keep mixed problems with enough Good-C support and utility.

B. Build a 50-problem sequence:
   - prefer problems whose screened C/S labels are separable in current F2 state;
   - first `adaptation_problems` are chosen as training-rich prefix;
   - remaining problems form the primary evaluation slice.

C. On-policy stability confirmation:
   - reset the real U1 learner to zero;
   - run the exact 50 sequence with two different seeds;
   - require BOTH runs to be actually C-rich and to open a useful learned-C
     region (learned-C count, negative learned-C dJ, temporal AUC);
   - if a sequence fails, keep on-policy C-rich anchors, replace weak problems
     with unused screened candidates, and retry. If needed, screen more MATH ids.

D. Freeze and final held-out-seed comparison:
   - once stable, write FROZEN_CRICH50_IDS.csv;
   - run fresh matched Always-STOP and current U1 with a third seed;
   - report measured E2E speedup on the post-adaptation evaluation slice,
     bootstrap CI, exact-output-hash subset, final actual Good-C prevalence,
     and learned-C diagnostics.

No core policy changes are introduced by this runner. `adaptive_td.py` and
`failfast.py` bundled beside it are the same tau=.5/current-U1 core used in the
previous MATH-30 test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

METHOD_ALWAYS = "always_stop"
METHOD_U1 = "u1_current_tau05"
METHOD_PROBE = "probe_only_control"
METHOD_SCREEN = "oracle_screen"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Runtime/model.
    p.add_argument("--dllm_dir", default="/kaggle/working/Fast_dLLM_v2_1.5B")
    p.add_argument("--output_dir", default="/kaggle/working/outputs_math_stable_crich50")
    p.add_argument("--target_quantization", default="int8")
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=0)
    p.add_argument("--drafter_threshold", type=float, default=0.50)
    p.add_argument("--lowconf_threshold", type=float, default=0.70)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--log_level", default="INFO")

    # The existing engine uses MATH-500, not the full 5000-question MATH test set.
    p.add_argument("--screen_start_id", type=int, default=0)
    p.add_argument("--screen_end_id", type=int, default=500)  # exclusive
    p.add_argument("--screen_batch_size", type=int, default=50)
    p.add_argument(
        "--max_screen_problems", type=int, default=500,
        help=(
            "Hard cap on the total number of MATH problem IDs screened from "
            "screen_start_id. Default 500 means ids [start, start+500). "
            "Use --resume with a larger value to extend the search without "
            "re-running completed batches."
        ),
    )
    p.add_argument("--initial_candidate_target", type=int, default=65)
    p.add_argument("--candidate_increment", type=int, default=20)

    # Per-problem screen filters. These are intentionally permissive; the real
    # guarantee comes from the two on-policy confirmations of the entire 50-id sequence.
    p.add_argument("--screen_min_nontie", type=int, default=3)
    p.add_argument("--screen_min_goodc", type=int, default=1)
    p.add_argument("--screen_min_c_rate", type=float, default=0.20)
    p.add_argument("--screen_max_c_rate", type=float, default=0.80)
    p.add_argument("--screen_min_stop", type=int, default=1)
    p.add_argument("--screen_min_goodc_utility_mass", type=float, default=5.0)
    p.add_argument("--screen_min_mean_goodc_benefit", type=float, default=2.0)

    # Final sequence shape.
    p.add_argument("--final_size", type=int, default=50)
    p.add_argument("--adaptation_problems", type=int, default=20)
    p.add_argument("--target_screen_c_rate", type=float, default=0.40)

    # Stability confirmation. Defaults are intentionally strong enough to avoid
    # repeating the previous 34%->11% collapse.
    p.add_argument("--confirm_seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--final_seed", type=int, default=44)
    p.add_argument("--confirm_min_c_rate", type=float, default=0.30)
    p.add_argument("--confirm_min_goodc", type=int, default=50)
    p.add_argument("--confirm_max_c_rate_gap", type=float, default=0.07)
    p.add_argument("--confirm_min_learned_c", type=int, default=15)
    p.add_argument("--confirm_min_learned_c_tp", type=int, default=5)
    p.add_argument("--confirm_min_eval_auc", type=float, default=0.65)
    p.add_argument("--confirm_require_negative_learned_utility", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_selection_attempts", type=int, default=4)
    p.add_argument("--replace_count", type=int, default=10)

    # Final test/reporting.
    p.add_argument("--bootstrap_samples", type=int, default=5000)
    p.add_argument("--include_probe_control", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--screen_only", action="store_true")
    p.add_argument("--confirm_only", action="store_true")
    p.add_argument("--final_only", action="store_true")

    a = p.parse_args()
    if not 0 <= a.screen_start_id < a.screen_end_id <= 500:
        p.error("MATH-500 requires 0 <= start < end <= 500; this is not a fresh dataset")
    if sum((a.screen_only, a.confirm_only, a.final_only)) > 1:
        p.error("Choose only one stage-only flag")
    if len(set(a.confirm_seeds)) != len(a.confirm_seeds) or a.final_seed in a.confirm_seeds:
        p.error("Confirmation seeds must be distinct and different from final_seed")
    if min(a.bootstrap_samples, a.max_selection_attempts, a.replace_count) <= 0:
        p.error("Bootstrap samples, attempts and replacement count must be positive")
    if not (0 <= a.adaptation_problems < a.final_size):
        p.error("--adaptation_problems must be in [0, final_size-1]")
    if a.final_size != 50:
        print("WARNING: designed around a 50-problem positive control; custom final_size is allowed.", file=sys.stderr)
    if a.screen_end_id <= a.screen_start_id:
        p.error("--screen_end_id must exceed --screen_start_id")
    if a.screen_batch_size <= 0:
        p.error("--screen_batch_size must be positive")
    if a.max_screen_problems <= 0:
        p.error("--max_screen_problems must be positive")
    if len(a.confirm_seeds) < 2:
        p.error("use at least two --confirm_seeds for a real stability check")
    if a.final_seed in set(a.confirm_seeds):
        print("WARNING: final_seed overlaps a confirmation seed; a distinct held-out seed is cleaner.", file=sys.stderr)
    return a


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def run_checked(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 110)
    print("RUN:", " ".join(map(str, cmd)))
    print("LOG:", log_path)
    print("=" * 110, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda x: str(x).strip().lower() in {"1", "true", "yes"})


def auc_binary(labels: Sequence[float], scores: Sequence[float]) -> float:
    f = pd.DataFrame({"y": pd.to_numeric(pd.Series(labels), errors="coerce"),
                      "s": pd.to_numeric(pd.Series(scores), errors="coerce")}).dropna()
    if f.empty:
        return float("nan")
    p = int((f.y == 1).sum())
    n = int((f.y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    ranks = f.s.rank(method="average")
    return float((ranks[f.y == 1].sum() - p * (p + 1) / 2) / (p * n))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_weighted_f2_logistic(df: pd.DataFrame, max_iter: int = 60, ridge: float = 1e-6) -> np.ndarray:
    """Offline diagnostic only: fit the same F2 linear logit with raw |dJ| weights."""
    work = resolved_non_tie(df)
    if work.empty or work.binary_label_C.nunique() < 2:
        return np.zeros(3, dtype=float)
    X = np.column_stack([
        np.ones(len(work)),
        pd.to_numeric(work.current_mask_ratio, errors="coerce").fillna(0).to_numpy(float),
        pd.to_numeric(work.global_proposal_position, errors="coerce").fillna(0).to_numpy(float),
    ])
    y = pd.to_numeric(work.binary_label_C, errors="coerce").to_numpy(float)
    wgt = np.abs(pd.to_numeric(work.delta_J_ms_per_token, errors="coerce").fillna(0).to_numpy(float))
    wgt = np.maximum(wgt, 1e-9)
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (wgt * (p - y)) + ridge * beta
        hdiag = wgt * p * (1.0 - p)
        H = X.T @ (X * hdiag[:, None]) + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def add_f2_offline_score(df: pd.DataFrame, beta: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    X = np.column_stack([
        np.ones(len(out)),
        pd.to_numeric(out.current_mask_ratio, errors="coerce").fillna(0).to_numpy(float),
        pd.to_numeric(out.global_proposal_position, errors="coerce").fillna(0).to_numpy(float),
    ])
    out["offline_f2_score"] = sigmoid(X @ beta)
    return out


def resolved_non_tie(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "pair_resolved" in out.columns:
        out = out[bool_series(out.pair_resolved)]
    if "binary_label_C" not in out.columns:
        return out.iloc[0:0].copy()
    out["binary_label_C"] = pd.to_numeric(out.binary_label_C, errors="coerce")
    out["delta_J_ms_per_token"] = pd.to_numeric(out.delta_J_ms_per_token, errors="coerce")
    out = out[out.binary_label_C.isin([0, 1]) & out.delta_J_ms_per_token.notna()].copy()
    if "update_applied" in out.columns:
        # Non-ties are exactly the update-applied rows in current U1. Keep this
        # check to avoid accidentally counting censored/tie rows.
        out = out[bool_series(out.update_applied)].copy()
    return out


def load_transitions(case: Path) -> pd.DataFrame:
    p = case / "adaptive_full_stream_transitions.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_decisions(case: Path) -> pd.DataFrame:
    p = case / "adaptive_td_decisions.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_benchmark(case: Path) -> pd.DataFrame:
    p = case / "benchmark_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "mode" in df.columns:
        df = df[df["mode"] == "dllm_ar"].copy()
    return df


def case_complete(case: Path, ids: Sequence[int]) -> bool:
    try:
        df = load_benchmark(case)
    except Exception:
        return False
    got = df.problem_id.astype(int).tolist()
    return got == list(map(int, ids))


# -----------------------------------------------------------------------------
# failfast.py commands
# -----------------------------------------------------------------------------
def base_command(args: argparse.Namespace, ids: Sequence[int], destination: Path, seed: int) -> list[str]:
    return [
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
        "--dllm_dir", str(args.dllm_dir),
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", str(args.target_quantization),
        "--unquantized_dtype", "float16",
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--seed", str(seed),
        "--quiet_generation", "--disable_progress", "--skip_artifacts", "--skip_plots",
        "--overwrite", "--output_dir", str(destination), "--log_level", args.log_level,
    ]


def append_u1_common(cmd: list[str]) -> list[str]:
    cmd.extend([
        "--adaptive-td",
        "--adaptive-feature-schema", "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment", "hindsight_delta_j_logistic_f2",
        "--adaptive-policy-mode", "hindsight_delta_j_logistic_f2",
        "--adaptive-hindsight-logistic-tie-ms-per-token", "1.0",
        "--no-adaptive-hindsight-logistic-use-class-weight",
        "--no-adaptive-hindsight-logistic-use-prefix-feature",
        "--no-adaptive-hindsight-logistic-dynamic-threshold",
        "--adaptive-hindsight-logistic-utility-weighting", "raw_abs",
        "--adaptive-hindsight-logistic-replay-stop-to-continue-ratio", "0",
        "--no-adaptive-hindsight-logistic-balance-utility-mass",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
    ])
    return cmd


def command_for_method(args: argparse.Namespace, method: str, ids: Sequence[int], destination: Path, seed: int) -> list[str]:
    cmd = base_command(args, ids, destination, seed)
    if method == "failfast":
        return cmd
    append_u1_common(cmd)

    if method == METHOD_ALWAYS:
        cmd.extend([
            "--adaptive-policy-ablation", "frozen_stop",
            "--adaptive-hindsight-logistic-learning-rate", "0.0",
            "--adaptive-hindsight-logistic-continue-threshold", "0.999999",
            "--adaptive-hindsight-delta-j-min-pairs", "0",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "0",
            "--adaptive-hindsight-logistic-min-positive-problems", "0",
            "--adaptive-hindsight-delta-j-structural-probe", "0",
            "--adaptive-hindsight-delta-j-floor-probe", "0",
            "--adaptive-hindsight-logistic-replay-batch-size", "0",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
        ])
        return cmd

    if method == METHOD_U1:
        cmd.extend([
            "--adaptive-policy-ablation", "learned",
            "--adaptive-hindsight-logistic-learning-rate", "0.05",
            "--adaptive-hindsight-logistic-continue-threshold", "0.5",
            "--adaptive-hindsight-delta-j-min-pairs", "30",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
            "--adaptive-hindsight-logistic-min-positive-problems", "2",
            "--adaptive-hindsight-delta-j-structural-probe", "0.08",
            "--adaptive-hindsight-delta-j-floor-probe", "0.02",
            "--adaptive-hindsight-logistic-replay-batch-size", "16",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
        ])
        return cmd

    if method == METHOD_PROBE:
        # Same probe schedule as U1 but the model is frozen below threshold.
        # This is optional and isolates "learner" benefit from exploration benefit.
        cmd.extend([
            "--adaptive-policy-ablation", "learned",
            "--adaptive-hindsight-logistic-learning-rate", "0.0",
            "--adaptive-hindsight-logistic-continue-threshold", "0.999999",
            "--adaptive-hindsight-delta-j-min-pairs", "0",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "0",
            "--adaptive-hindsight-logistic-min-positive-problems", "0",
            "--adaptive-hindsight-delta-j-structural-probe", "0.08",
            "--adaptive-hindsight-delta-j-floor-probe", "0.02",
            "--adaptive-hindsight-logistic-replay-batch-size", "0",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
        ])
        return cmd

    if method == METHOD_SCREEN:
        # Oracle-style label collector. No learning; model action is STOP, then
        # exploration forces one CONTINUE counterfactual whenever legal. This
        # resolves the exact same dJ target used by current U1 while avoiding a
        # self-reinforcing learner during candidate discovery.
        cmd.extend([
            "--adaptive-policy-ablation", "learned",
            "--adaptive-hindsight-logistic-learning-rate", "0.0",
            "--adaptive-hindsight-logistic-continue-threshold", "0.999999",
            "--adaptive-hindsight-delta-j-min-pairs", "0",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "0",
            "--adaptive-hindsight-logistic-min-positive-problems", "0",
            "--adaptive-hindsight-delta-j-structural-probe", "1.0",
            "--adaptive-hindsight-delta-j-floor-probe", "1.0",
            "--adaptive-hindsight-logistic-replay-batch-size", "0",
            "--adaptive-hindsight-logistic-replay-buffer-size", "100",
        ])
        return cmd

    raise ValueError(method)


# -----------------------------------------------------------------------------
# Screening and candidate stats
# -----------------------------------------------------------------------------
@dataclass
class ProblemStats:
    problem_id: int
    non_tie: int
    good_c: int
    stop: int
    c_rate: float
    sum_delta_j_all: float
    good_c_utility_mass: float
    mean_good_c_benefit: float
    mean_stop_cost: float
    offline_f2_mean_c_score: float = float("nan")
    offline_f2_mean_s_score: float = float("nan")
    offline_f2_score_gap: float = float("nan")
    offline_f2_problem_auc: float = float("nan")
    screen_rank_score: float = float("nan")


def aggregate_problem_stats(transitions: pd.DataFrame, beta: np.ndarray | None = None) -> pd.DataFrame:
    tr = resolved_non_tie(transitions)
    if tr.empty:
        return pd.DataFrame(columns=list(ProblemStats.__annotations__))
    if beta is not None:
        tr = add_f2_offline_score(tr, beta)
    rows = []
    for pid, g in tr.groupby(tr.problem_id.astype(int), sort=False):
        y = pd.to_numeric(g.binary_label_C, errors="coerce")
        dj = pd.to_numeric(g.delta_J_ms_per_token, errors="coerce")
        gc = g[y == 1]
        st = g[y == 0]
        good_mass = float((-pd.to_numeric(gc.delta_J_ms_per_token, errors="coerce")).clip(lower=0).sum()) if len(gc) else 0.0
        mean_good_benefit = float((-pd.to_numeric(gc.delta_J_ms_per_token, errors="coerce")).mean()) if len(gc) else 0.0
        mean_stop_cost = float(pd.to_numeric(st.delta_J_ms_per_token, errors="coerce").mean()) if len(st) else 0.0
        mean_c_score = float(gc.offline_f2_score.mean()) if beta is not None and len(gc) else float("nan")
        mean_s_score = float(st.offline_f2_score.mean()) if beta is not None and len(st) else float("nan")
        gap = mean_c_score - mean_s_score if math.isfinite(mean_c_score) and math.isfinite(mean_s_score) else float("nan")
        p_auc = auc_binary(g.binary_label_C, g.offline_f2_score) if beta is not None and y.nunique() == 2 else float("nan")
        rows.append(ProblemStats(
            problem_id=int(pid), non_tie=int(len(g)), good_c=int((y == 1).sum()),
            stop=int((y == 0).sum()), c_rate=float((y == 1).mean()),
            sum_delta_j_all=float(dj.sum()), good_c_utility_mass=good_mass,
            mean_good_c_benefit=mean_good_benefit, mean_stop_cost=mean_stop_cost,
            offline_f2_mean_c_score=mean_c_score, offline_f2_mean_s_score=mean_s_score,
            offline_f2_score_gap=gap, offline_f2_problem_auc=p_auc,
        ).__dict__)
    return pd.DataFrame(rows)


def filter_screen_candidates(args: argparse.Namespace, stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return stats.copy()
    f = stats[
        (stats.non_tie >= args.screen_min_nontie)
        & (stats.good_c >= args.screen_min_goodc)
        & (stats.stop >= args.screen_min_stop)
        & (stats.c_rate >= args.screen_min_c_rate)
        & (stats.c_rate <= args.screen_max_c_rate)
        & (stats.good_c_utility_mass >= args.screen_min_goodc_utility_mass)
        & (stats.mean_good_c_benefit >= args.screen_min_mean_goodc_benefit)
    ].copy()
    if f.empty:
        return f
    # Rank: support and useful C matter; penalize pure-ish class imbalance;
    # reward F2 score separation when available.
    sep = pd.to_numeric(f.offline_f2_score_gap, errors="coerce").fillna(0).clip(lower=-1, upper=1)
    closeness = 1.0 - (f.c_rate - args.target_screen_c_rate).abs().clip(upper=1.0)
    f["screen_rank_score"] = (
        1.2 * np.log1p(f.good_c_utility_mass)
        + 0.7 * np.log1p(f.non_tie)
        + 1.5 * closeness
        + 2.0 * sep
    )
    return f.sort_values(["screen_rank_score", "good_c_utility_mass", "non_tie"], ascending=False)


def read_all_screen_transitions(screen_root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(screen_root.glob("batch_*/adaptive_full_stream_transitions.csv")):
        if not (p.parent / "SCREEN_COMPLETE.json").exists():
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print("WARNING failed reading", p, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def screened_ids_from_dirs(screen_root: Path) -> set[int]:
    out: set[int] = set()
    for case in screen_root.glob("batch_*"):
        if not (case / "SCREEN_COMPLETE.json").exists():
            continue
        try:
            df = load_benchmark(case)
        except Exception:
            continue
        out.update(df.problem_id.astype(int).tolist())
    return out


def screen_until_candidates(args: argparse.Namespace, root: Path, needed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    screen_root = root / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    screened = screened_ids_from_dirs(screen_root)

    # Hard bounded search window. This prevents a failed search from silently
    # walking all the way to MATH id 4999. Increasing --max_screen_problems with
    # --resume extends this same window without repeating completed batches.
    hard_end_id = min(args.screen_end_id, args.screen_start_id + args.max_screen_problems)
    allowed_total = max(0, hard_end_id - args.screen_start_id)

    def refresh() -> tuple[pd.DataFrame, pd.DataFrame]:
        all_tr = read_all_screen_transitions(screen_root)
        tr = resolved_non_tie(all_tr)
        beta = fit_weighted_f2_logistic(tr) if not tr.empty else np.zeros(3)
        stats = aggregate_problem_stats(tr, beta)
        cand = filter_screen_candidates(args, stats)
        if not stats.empty:
            stats.to_csv(root / "screen_problem_stats.csv", index=False)
        if not cand.empty:
            cand.to_csv(root / "screen_candidates.csv", index=False)
        json_dump(root / "screen_f2_diagnostic.json", {
            "beta": beta.tolist(),
            "pooled_auc": auc_binary(tr.binary_label_C, add_f2_offline_score(tr, beta).offline_f2_score) if (not tr.empty and tr.binary_label_C.nunique() == 2) else float("nan"),
            "screened_problem_count": len(screened),
            "candidate_count": len(cand),
            "candidate_target": int(needed),
            "screen_start_id": int(args.screen_start_id),
            "hard_end_id_exclusive": int(hard_end_id),
            "max_screen_problems": int(args.max_screen_problems),
        })
        return tr, cand

    tr, cand = refresh()
    next_id = args.screen_start_id
    screened_in_window = {pid for pid in screened if args.screen_start_id <= pid < hard_end_id}

    print("\n" + "=" * 110)
    print("SCREEN SEARCH STATUS")
    print(f"  target candidates : {needed}")
    print(f"  final set size     : {args.final_size}")
    print(f"  search id window   : [{args.screen_start_id}, {hard_end_id})")
    print(f"  hard screen cap    : {allowed_total} problems")
    print(f"  already screened   : {len(screened_in_window)}/{allowed_total}")
    print(f"  candidates found   : {len(cand)}/{needed}")
    print("  stop rule          : target reached OR hard cap exhausted")
    print("=" * 110, flush=True)

    batch_index = 0
    while len(cand) < needed and next_id < hard_end_id:
        batch = [
            i for i in range(next_id, min(hard_end_id, next_id + args.screen_batch_size))
            if i not in screened
        ]
        next_id += args.screen_batch_size
        if not batch:
            continue
        batch_index += 1
        before_ids = set(cand.problem_id.astype(int).tolist()) if not cand.empty else set()
        before_count = len(cand)
        screened_before = len({pid for pid in screened if args.screen_start_id <= pid < hard_end_id})
        print(
            f"\n[SCREEN batch {batch_index}] ids={batch[0]}..{batch[-1]} "
            f"({len(batch)} problems) | screened={screened_before}/{allowed_total} | "
            f"candidates={before_count}/{needed}",
            flush=True,
        )

        case = screen_root / f"batch_{batch[0]:04d}_{batch[-1]:04d}"
        if not (args.resume and case_complete(case, batch)):
            try:
                run_checked(
                    command_for_method(args, METHOD_SCREEN, batch, case, seed=11),
                    ROOT, root / "logs" / f"screen_{batch[0]:04d}_{batch[-1]:04d}.log",
                )
            except subprocess.CalledProcessError:
                # A single bad MATH id should not kill the whole search. Retry one-by-one
                # and record failures. This is slower only for problematic batches.
                print("[SCREEN] Batch failed; retrying ids individually to isolate bad samples.")
                if case.exists():
                    case.rename(case.with_name(f"failed_{case.name}_{time.time_ns()}"))
                failed = []
                for pid in batch:
                    one = screen_root / f"batch_{pid:04d}_{pid:04d}"
                    if args.resume and case_complete(one, [pid]):
                        continue
                    try:
                        run_checked(
                            command_for_method(args, METHOD_SCREEN, [pid], one, seed=11),
                            ROOT, root / "logs" / f"screen_{pid:04d}.log",
                        )
                        if not case_complete(one, [pid]):
                            raise RuntimeError(f"Incomplete screen output for {pid}")
                        json_dump(one / "SCREEN_COMPLETE.json", {"ids": [pid]})
                    except subprocess.CalledProcessError:
                        failed.append(pid)
                        if one.exists():
                            one.rename(one.with_name(f"failed_{one.name}_{time.time_ns()}"))
                if failed:
                    with (root / "screen_failed_ids.txt").open("a", encoding="utf-8") as h:
                        for pid in failed:
                            h.write(f"{pid}\n")
                    print(f"[SCREEN] failed ids this batch: {failed}")

        if case_complete(case, batch):
            json_dump(case / "SCREEN_COMPLETE.json", {"ids": batch})

        screened.update(batch)
        screened_in_window = {pid for pid in screened if args.screen_start_id <= pid < hard_end_id}
        tr, cand = refresh()
        after_ids = set(cand.problem_id.astype(int).tolist()) if not cand.empty else set()
        new_ids = sorted(after_ids - before_ids)
        remaining = max(0, needed - len(cand))
        yield_rate = len(cand) / max(1, len(screened_in_window))
        if yield_rate > 0 and remaining > 0:
            rough_more = int(math.ceil(remaining / yield_rate))
            rough_msg = f"~{rough_more} more screened problems at current yield"
        elif remaining == 0:
            rough_msg = "target reached"
        else:
            rough_msg = "cannot estimate yet"

        print(
            f"[SCREEN progress] screened={len(screened_in_window)}/{allowed_total} | "
            f"candidates={len(cand)}/{needed} (+{len(new_ids)} this batch) | "
            f"remaining={remaining} | yield={100*yield_rate:.1f}% | {rough_msg}",
            flush=True,
        )
        if new_ids:
            preview = new_ids[:20]
            tail = " ..." if len(new_ids) > 20 else ""
            print(f"[SCREEN new candidates] {preview}{tail}", flush=True)
        else:
            print("[SCREEN new candidates] none in this batch", flush=True)

    if len(cand) >= needed:
        msg = (
            f"SCREEN TARGET REACHED: {len(cand)} candidates after "
            f"{len(screened_in_window)} screened problems."
        )
        print("\n" + msg, flush=True)
        (root / "SCREEN_STATUS.txt").write_text(msg + "\n", encoding="utf-8")
        return tr, cand

    # Hard cap exhausted before the requested spare-candidate target. Stop cleanly
    # rather than spending confirmation compute on an under-supplied pool.
    msg = (
        f"SCREEN STOPPED AT HARD CAP: found {len(cand)}/{needed} candidates after "
        f"screening {len(screened_in_window)}/{allowed_total} problems in ids "
        f"[{args.screen_start_id}, {hard_end_id}). No confirmation/final E2E was run.\n"
        f"Resume with a larger --max_screen_problems (for example "
        f"{max(args.max_screen_problems + 250, int(args.max_screen_problems * 1.5))}) "
        f"and keep --resume; completed screen batches will be reused."
    )
    print("\n" + msg, flush=True)
    (root / "SCREEN_STOPPED_INCOMPLETE.txt").write_text(msg + "\n", encoding="utf-8")
    raise RuntimeError(msg)


# -----------------------------------------------------------------------------
# Sequence construction
# -----------------------------------------------------------------------------
def per_problem_screen_rows(screen_tr: pd.DataFrame, pid: int) -> pd.DataFrame:
    return resolved_non_tie(screen_tr[screen_tr.problem_id.astype(int) == int(pid)].copy())


def selection_metrics(screen_tr: pd.DataFrame, ids: Sequence[int]) -> dict:
    tr = resolved_non_tie(screen_tr[screen_tr.problem_id.astype(int).isin(set(map(int, ids)))].copy())
    if tr.empty:
        return {"nontie": 0, "good_c": 0, "c_rate": float("nan"), "offline_f2_auc": float("nan")}
    beta = fit_weighted_f2_logistic(tr)
    scored = add_f2_offline_score(tr, beta)
    return {
        "nontie": int(len(tr)),
        "good_c": int((tr.binary_label_C == 1).sum()),
        "c_rate": float((tr.binary_label_C == 1).mean()),
        "sum_delta_j": float(tr.delta_J_ms_per_token.sum()),
        "good_c_utility_mass": float((-tr.loc[tr.binary_label_C == 1, "delta_J_ms_per_token"]).clip(lower=0).sum()),
        "offline_f2_auc": auc_binary(scored.binary_label_C, scored.offline_f2_score),
        "offline_f2_beta": beta.tolist(),
    }


def build_sequence(
    args: argparse.Namespace,
    screen_tr: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    anchor_ids: Sequence[int] = (),
    banned_ids: Sequence[int] = (),
) -> tuple[list[int], dict]:
    banned = set(map(int, banned_ids))
    anchors = [int(x) for x in anchor_ids if int(x) not in banned]
    cand = candidates[~candidates.problem_id.astype(int).isin(banned)].copy()
    if len(cand) < args.final_size:
        raise RuntimeError("not enough unbanned candidates to build sequence")

    # Use screened global F2 model to estimate how aligned each problem is with
    # a learnable F2 boundary.
    pool_ids = cand.problem_id.astype(int).tolist()
    pool_tr = resolved_non_tie(screen_tr[screen_tr.problem_id.astype(int).isin(set(pool_ids))].copy())
    beta = fit_weighted_f2_logistic(pool_tr)
    scored = add_f2_offline_score(pool_tr, beta)
    bypid = []
    for pid in pool_ids:
        g = scored[scored.problem_id.astype(int) == pid]
        if g.empty:
            continue
        c = g[g.binary_label_C == 1]
        s = g[g.binary_label_C == 0]
        cmean = float(c.offline_f2_score.mean()) if len(c) else 0.0
        smean = float(s.offline_f2_score.mean()) if len(s) else 1.0
        sep = cmean - smean if len(c) and len(s) else 0.0
        c_mass = float((-c.delta_J_ms_per_token).clip(lower=0).sum()) if len(c) else 0.0
        rate = float((g.binary_label_C == 1).mean())
        train_value = (
            2.0 * sep
            + 1.0 * math.log1p(c_mass)
            + 0.5 * math.log1p(len(g))
            + 1.0 * (1.0 - abs(rate - args.target_screen_c_rate))
        )
        bypid.append({"problem_id": pid, "train_value": train_value, "sep": sep,
                      "c_mass": c_mass, "n": len(g), "c_rate": rate})
    rank = pd.DataFrame(bypid).sort_values(["train_value", "c_mass", "n"], ascending=False)
    rank_ids = rank.problem_id.astype(int).tolist()

    # Anchors are confirmed on-policy-rich problems from previous failed attempts.
    # Preserve them, then fill adaptation prefix with highest training-value problems.
    selected: list[int] = []
    for pid in anchors:
        if pid in rank_ids and pid not in selected:
            selected.append(pid)
    for pid in rank_ids:
        if len(selected) >= args.adaptation_problems:
            break
        if pid not in selected:
            selected.append(pid)

    # Fill the evaluation slice greedily while nudging pooled screened C-rate to
    # the target and retaining high F2 training/evaluation value.
    remaining = [pid for pid in rank_ids if pid not in selected]
    while len(selected) < args.final_size:
        best_pid = None
        best_obj = -1e18
        for pid in remaining[: min(len(remaining), 200)]:
            trial = selected + [pid]
            m = selection_metrics(screen_tr, trial)
            if not math.isfinite(m["c_rate"]):
                continue
            target_penalty = abs(m["c_rate"] - args.target_screen_c_rate)
            auc_bonus = m["offline_f2_auc"] if math.isfinite(m["offline_f2_auc"]) else 0.5
            rr = rank[rank.problem_id == pid]
            tv = float(rr.train_value.iloc[0]) if len(rr) else 0.0
            obj = 2.0 * auc_bonus - 2.5 * target_penalty + 0.15 * tv
            if obj > best_obj:
                best_obj, best_pid = obj, pid
        if best_pid is None:
            best_pid = remaining[0]
        selected.append(int(best_pid))
        remaining.remove(best_pid)

    # Reorder first adaptation problems by train value; eval stays in deterministic
    # rank order to make the exact sequence reproducible.
    ad = selected[: args.adaptation_problems]
    ev = selected[args.adaptation_problems :]
    val_map = dict(zip(rank.problem_id.astype(int), rank.train_value.astype(float)))
    ad = sorted(ad, key=lambda x: (-val_map.get(x, 0.0), x))
    ev = sorted(ev, key=lambda x: (-val_map.get(x, 0.0), x))
    seq = ad + ev
    metrics = selection_metrics(screen_tr, seq)
    metrics.update({
        "sequence": seq,
        "adaptation_ids": ad,
        "evaluation_ids": ev,
        "anchor_ids": list(map(int, anchors)),
        "screen_global_f2_beta": beta.tolist(),
    })
    return seq, metrics


# -----------------------------------------------------------------------------
# On-policy confirmation metrics
# -----------------------------------------------------------------------------
def learner_slice_metrics(case: Path, ids: Sequence[int], label: str) -> dict:
    ids_set = set(map(int, ids))
    tr = resolved_non_tie(load_transitions(case))
    if not tr.empty:
        tr = tr[tr.problem_id.astype(int).isin(ids_set)].copy()
    dec = load_decisions(case)
    if not dec.empty:
        dec = dec[dec.problem_id.astype(int).isin(ids_set)].copy()

    out = {"slice": label, "problems": len(ids), "resolved_non_tie": int(len(tr))}
    if len(tr):
        y = pd.to_numeric(tr.binary_label_C, errors="coerce")
        s = pd.to_numeric(tr.continue_score_before_update, errors="coerce")
        out.update({
            "good_c": int((y == 1).sum()),
            "stop": int((y == 0).sum()),
            "c_rate": float((y == 1).mean()),
            "temporal_auc": auc_binary(y, s),
            "sum_delta_j_all": float(pd.to_numeric(tr.delta_J_ms_per_token, errors="coerce").sum()),
        })
        src = tr.action_source.fillna("").astype(str) if "action_source" in tr.columns else pd.Series("", index=tr.index)
        learned = tr[src == "learned_continue"].copy()
        ly = pd.to_numeric(learned.binary_label_C, errors="coerce") if len(learned) else pd.Series(dtype=float)
        ldj = pd.to_numeric(learned.delta_J_ms_per_token, errors="coerce") if len(learned) else pd.Series(dtype=float)
        out.update({
            "resolved_learned_c": int(len(learned)),
            "learned_c_tp": int((ly == 1).sum()) if len(learned) else 0,
            "learned_c_fp": int((ly == 0).sum()) if len(learned) else 0,
            "sum_delta_j_learned_c": float(ldj.sum()) if len(learned) else 0.0,
        })
    else:
        out.update({
            "good_c": 0, "stop": 0, "c_rate": float("nan"), "temporal_auc": float("nan"),
            "sum_delta_j_all": 0.0, "resolved_learned_c": 0,
            "learned_c_tp": 0, "learned_c_fp": 0, "sum_delta_j_learned_c": 0.0,
        })
    if len(dec):
        src = dec.action_source.fillna("").astype(str) if "action_source" in dec.columns else pd.Series("", index=dec.index)
        out.update({
            "learned_continue_decisions": int((src == "learned_continue").sum()),
            "learned_stop_decisions": int((src == "learned_stop").sum()),
            "structural_probes": int((src == "structural_probe").sum()),
            "floor_probes": int((src == "floor_probe").sum()),
        })
    return out


def per_problem_onpolicy_stats(case: Path) -> pd.DataFrame:
    tr = resolved_non_tie(load_transitions(case))
    if tr.empty:
        return pd.DataFrame(columns=["problem_id", "non_tie", "good_c", "c_rate", "learned_c", "learned_delta"])
    rows = []
    for pid, g in tr.groupby(tr.problem_id.astype(int), sort=False):
        y = pd.to_numeric(g.binary_label_C, errors="coerce")
        src = g.action_source.fillna("").astype(str) if "action_source" in g.columns else pd.Series("", index=g.index)
        learned = g[src == "learned_continue"]
        rows.append({
            "problem_id": int(pid), "non_tie": int(len(g)), "good_c": int((y == 1).sum()),
            "stop": int((y == 0).sum()), "c_rate": float((y == 1).mean()),
            "sum_delta": float(g.delta_J_ms_per_token.sum()),
            "learned_c": int(len(learned)),
            "learned_delta": float(learned.delta_J_ms_per_token.sum()) if len(learned) else 0.0,
        })
    return pd.DataFrame(rows)


def confirmation_pass(args: argparse.Namespace, full: dict, evalm: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not math.isfinite(full.get("c_rate", float("nan"))) or full["c_rate"] < args.confirm_min_c_rate:
        reasons.append(f"full C-rate {full.get('c_rate')} < {args.confirm_min_c_rate}")
    if full.get("good_c", 0) < args.confirm_min_goodc:
        reasons.append(f"full Good-C {full.get('good_c', 0)} < {args.confirm_min_goodc}")
    if evalm.get("resolved_learned_c", 0) < args.confirm_min_learned_c:
        reasons.append(f"eval learned-C {evalm.get('resolved_learned_c', 0)} < {args.confirm_min_learned_c}")
    if evalm.get("learned_c_tp", 0) < args.confirm_min_learned_c_tp:
        reasons.append(f"eval learned-C TP {evalm.get('learned_c_tp', 0)} < {args.confirm_min_learned_c_tp}")
    auc = evalm.get("temporal_auc", float("nan"))
    if not math.isfinite(auc) or auc < args.confirm_min_eval_auc:
        reasons.append(f"eval temporal AUC {auc} < {args.confirm_min_eval_auc}")
    if args.confirm_require_negative_learned_utility and evalm.get("sum_delta_j_learned_c", 0.0) >= 0.0:
        reasons.append(f"eval sum dJ learned-C {evalm.get('sum_delta_j_learned_c')} is not negative")
    return (len(reasons) == 0), reasons


def run_confirmation(args: argparse.Namespace, root: Path, attempt: int, ids: Sequence[int], seed: int) -> dict:
    case = root / "confirm" / f"attempt_{attempt}" / f"seed_{seed}" / METHOD_U1
    if not (args.resume and case_complete(case, ids)):
        run_checked(
            command_for_method(args, METHOD_U1, ids, case, seed),
            ROOT, root / "logs" / f"confirm_attempt{attempt}_seed{seed}.log",
        )
    eval_ids = list(ids)[args.adaptation_problems :]
    full = learner_slice_metrics(case, ids, "full50")
    ev = learner_slice_metrics(case, eval_ids, "eval")
    passed, reasons = confirmation_pass(args, full, ev)
    result = {"attempt": attempt, "seed": seed, "passed_individual": passed,
              "reasons": reasons, "full": full, "eval": ev, "case": str(case)}
    json_dump(root / "confirm" / f"attempt_{attempt}" / f"seed_{seed}_summary.json", result)
    per_problem_onpolicy_stats(case).to_csv(
        root / "confirm" / f"attempt_{attempt}" / f"seed_{seed}_per_problem.csv", index=False
    )
    return result


def stable_across_confirmations(args: argparse.Namespace, results: Sequence[dict]) -> tuple[bool, list[str]]:
    reasons = []
    for r in results:
        if not r["passed_individual"]:
            reasons.append(f"seed {r['seed']} failed: " + "; ".join(r["reasons"]))
    rates = [r["full"].get("c_rate", float("nan")) for r in results]
    if all(math.isfinite(x) for x in rates):
        gap = max(rates) - min(rates)
        if gap > args.confirm_max_c_rate_gap:
            reasons.append(f"C-rate gap {gap:.4f} > {args.confirm_max_c_rate_gap}")
    else:
        reasons.append("non-finite C-rate in confirmation")
    return (len(reasons) == 0), reasons


def derive_anchor_and_banned_ids(
    args: argparse.Namespace,
    root: Path,
    attempt: int,
    ids: Sequence[int],
    confirm_results: Sequence[dict],
) -> tuple[list[int], list[int]]:
    # Merge per-problem actual stats across confirmation seeds. Problems that are
    # repeatedly C-rich are anchors; repeatedly empty/STOP-only problems are weak.
    merged = None
    for r in confirm_results:
        p = root / "confirm" / f"attempt_{attempt}" / f"seed_{r['seed']}_per_problem.csv"
        df = pd.read_csv(p)
        suffix = f"_s{r['seed']}"
        df = df.rename(columns={c: c + suffix for c in df.columns if c != "problem_id"})
        merged = df if merged is None else merged.merge(df, on="problem_id", how="outer")
    if merged is None or merged.empty:
        return [], []
    seed_cols = [f"c_rate_s{r['seed']}" for r in confirm_results]
    good_cols = [f"good_c_s{r['seed']}" for r in confirm_results]
    for c in seed_cols + good_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    merged["min_c_rate"] = merged[seed_cols].fillna(0).min(axis=1)
    merged["min_good_c"] = merged[good_cols].fillna(0).min(axis=1)
    merged["mean_c_rate"] = merged[seed_cols].fillna(0).mean(axis=1)
    merged.to_csv(root / "confirm" / f"attempt_{attempt}" / "cross_seed_per_problem.csv", index=False)

    anchors_df = merged[(merged.min_c_rate >= max(0.20, args.confirm_min_c_rate - 0.10)) & (merged.min_good_c >= 1)]
    anchors_df = anchors_df.sort_values(["min_c_rate", "min_good_c"], ascending=False)
    anchors = anchors_df.problem_id.astype(int).tolist()[: max(0, args.final_size - args.replace_count)]

    weak_df = merged[(merged.mean_c_rate < 0.10) | (merged[good_cols].fillna(0).sum(axis=1) == 0)]
    weak_df = weak_df.sort_values(["mean_c_rate", "min_good_c"], ascending=True)
    weak = weak_df.problem_id.astype(int).tolist()[: args.replace_count]
    return anchors, weak


# -----------------------------------------------------------------------------
# Final E2E comparison
# -----------------------------------------------------------------------------
def benchmark_slice(case: Path, ids: Sequence[int], method: str, label: str) -> dict:
    ids_set = set(map(int, ids))
    df = load_benchmark(case)
    df = df[df.problem_id.astype(int).isin(ids_set)].copy()
    tok = float(df.output_tokens.sum())
    e2e_col = "actual_e2e_time_excluding_transfer" if "actual_e2e_time_excluding_transfer" in df.columns else "actual_algorithm_time"
    e2e_s = float(df[e2e_col].sum())
    out = {
        "method": method, "slice": label, "questions": int(df.problem_id.nunique()),
        "output_tokens": int(tok), "e2e_s": e2e_s,
        "e2e_ms_per_output_token": 1000 * e2e_s / max(tok, 1),
        "draft_forwards": int(df.total_num_forward_passes.sum()) if "total_num_forward_passes" in df.columns else None,
        "verifier_rounds": int(df.num_speculation_rounds.sum()) if "num_speculation_rounds" in df.columns else None,
        "draft_time_s": float(df.actual_draft_time.sum()) if "actual_draft_time" in df.columns else float("nan"),
        "verify_time_s": float(df.actual_verify_time.sum()) if "actual_verify_time" in df.columns else float("nan"),
    }
    if method in {METHOD_U1, METHOD_PROBE}:
        out.update({f"learner_{k}": v for k, v in learner_slice_metrics(case, ids, label).items() if k not in {"slice", "problems"}})
    return out


def paired_comparison(always: pd.DataFrame, cand: pd.DataFrame, ids: Sequence[int], n_boot: int, seed: int) -> dict:
    ids_set = set(map(int, ids))
    def prep(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        keep = ["problem_id", "output_tokens", "actual_algorithm_time"]
        if "actual_e2e_time_excluding_transfer" in df.columns:
            keep.append("actual_e2e_time_excluding_transfer")
        if "output_token_hash" in df.columns:
            keep.append("output_token_hash")
        x = df[df.problem_id.astype(int).isin(ids_set)][keep].copy()
        ren = {"output_tokens": f"tok_{tag}", "actual_algorithm_time": f"algo_{tag}",
               "actual_e2e_time_excluding_transfer": f"e2e_{tag}", "output_token_hash": f"hash_{tag}"}
        return x.rename(columns=ren)
    a = prep(always, "a")
    c = prep(cand, "c")
    m = a.merge(c, on="problem_id", how="inner")
    ta = "e2e_a" if "e2e_a" in m.columns else "algo_a"
    tc = "e2e_c" if "e2e_c" in m.columns else "algo_c"

    def speed(f: pd.DataFrame) -> float:
        if len(f) == 0:
            return float("nan")
        msa = 1000 * float(f[ta].sum()) / max(1.0, float(f.tok_a.sum()))
        msc = 1000 * float(f[tc].sum()) / max(1.0, float(f.tok_c.sum()))
        return msa / msc

    point = speed(m)
    rng = np.random.default_rng(seed)
    vals = []
    if len(m):
        for _ in range(n_boot):
            vals.append(speed(m.iloc[rng.integers(0, len(m), size=len(m))]))
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (float("nan"), float("nan")))
    out = {
        "paired_problems": int(len(m)), "speedup_vs_always_stop": float(point),
        "bootstrap95_low": float(lo), "bootstrap95_high": float(hi),
        "candidate_faster_problem_count": int(((m[tc] / m.tok_c) < (m[ta] / m.tok_a)).sum()) if len(m) else 0,
    }
    if "hash_a" in m.columns and "hash_c" in m.columns:
        exact = m[m.hash_a == m.hash_c].copy()
        out["exact_hash_matches"] = int(len(exact))
        out["exact_hash_speedup"] = float(speed(exact)) if len(exact) else float("nan")
        if len(exact):
            vals2 = [speed(exact.iloc[rng.integers(0, len(exact), size=len(exact))]) for _ in range(n_boot)]
            elo, ehi = np.quantile(vals2, [0.025, 0.975])
            out["exact_hash_bootstrap95_low"] = float(elo)
            out["exact_hash_bootstrap95_high"] = float(ehi)
    return out


def final_run(args: argparse.Namespace, root: Path, ids: Sequence[int]) -> None:
    final_root = root / "final"
    methods = [METHOD_ALWAYS, METHOD_U1]
    if args.include_probe_control:
        methods.append(METHOD_PROBE)
    for method in methods:
        case = final_root / method
        if not (args.resume and case_complete(case, ids)):
            run_checked(
                command_for_method(args, method, ids, case, args.final_seed),
                ROOT, root / "logs" / f"final_{method}.log",
            )

    eval_ids = list(ids)[args.adaptation_problems :]
    rows = []
    for method in methods:
        rows.append(benchmark_slice(final_root / method, ids, method, "full50"))
        rows.append(benchmark_slice(final_root / method, eval_ids, method, "eval"))
    pd.DataFrame(rows).to_csv(root / "FINAL_METHOD_SUMMARY.csv", index=False)

    a = load_benchmark(final_root / METHOD_ALWAYS)
    u = load_benchmark(final_root / METHOD_U1)
    comps = {
        "full50": paired_comparison(a, u, ids, args.bootstrap_samples, args.final_seed + 100),
        "eval": paired_comparison(a, u, eval_ids, args.bootstrap_samples, args.final_seed + 101),
    }
    if args.include_probe_control:
        p = load_benchmark(final_root / METHOD_PROBE)
        comps["probe_control_full50"] = paired_comparison(a, p, ids, args.bootstrap_samples, args.final_seed + 102)
        comps["probe_control_eval"] = paired_comparison(a, p, eval_ids, args.bootstrap_samples, args.final_seed + 103)
        comps["u1_vs_probe_eval_speedup"] = {
            # Ratio computed from pooled ms/token directly below for transparency.
            "note": "See FINAL_METHOD_SUMMARY.csv; U1-vs-probe is not a paired AlwaysSTOP ratio."
        }
    json_dump(root / "FINAL_PAIRED_COMPARISONS.json", comps)

    u_full = learner_slice_metrics(final_root / METHOD_U1, ids, "full50")
    u_eval = learner_slice_metrics(final_root / METHOD_U1, eval_ids, "eval")
    final_rich = math.isfinite(u_full.get("c_rate", float("nan"))) and u_full["c_rate"] >= args.confirm_min_c_rate
    learner_ok = (
        u_eval.get("resolved_learned_c", 0) >= args.confirm_min_learned_c
        and u_eval.get("learned_c_tp", 0) >= args.confirm_min_learned_c_tp
        and u_eval.get("sum_delta_j_learned_c", 0.0) < 0.0
        and math.isfinite(u_eval.get("temporal_auc", float("nan")))
        and u_eval["temporal_auc"] >= args.confirm_min_eval_auc
    )
    ev = comps["eval"]
    e2e_win = math.isfinite(ev["speedup_vs_always_stop"]) and ev["speedup_vs_always_stop"] > 1.0
    strong = e2e_win and math.isfinite(ev["bootstrap95_low"]) and ev["bootstrap95_low"] > 1.0

    if final_rich and learner_ok and strong:
        verdict = "STRONG PASS: final held-out-seed stream stayed C-rich, learner opened a beneficial learned-C region, and eval E2E beats Always-STOP with paired 95% CI > 1."
    elif final_rich and learner_ok and e2e_win:
        verdict = "DIRECTIONAL PASS: final stream stayed C-rich and learner learned useful C; eval E2E beats Always-STOP but paired CI crosses/touches 1."
    else:
        missing = []
        if not final_rich:
            missing.append("final stream did not remain C-rich")
        if not learner_ok:
            missing.append("learner criterion failed")
        if not e2e_win:
            missing.append("U1 did not beat Always-STOP E2E")
        verdict = "FAIL FINAL REQUIREMENT: " + "; ".join(missing)

    lines = [
        "STABLE C-RICH50 FINAL VERDICT — positive control / existence test",
        "",
        f"Frozen ids: {list(map(int, ids))}",
        f"Adaptation prefix = {args.adaptation_problems}; eval = {len(eval_ids)}",
        f"Final held-out seed = {args.final_seed}",
        "",
        f"Final full C-rate = {100*u_full.get('c_rate', float('nan')):.2f}% ({u_full.get('good_c',0)}/{u_full.get('resolved_non_tie',0)})",
        f"Final eval temporal AUC = {u_eval.get('temporal_auc', float('nan')):.4f}",
        f"Final eval resolved learned-C = {u_eval.get('resolved_learned_c',0)}",
        f"Final eval learned-C TP/FP = {u_eval.get('learned_c_tp',0)}/{u_eval.get('learned_c_fp',0)}",
        f"Final eval sum dJ learned-C = {u_eval.get('sum_delta_j_learned_c', float('nan')):.6f} ms/output-token (local diagnostic; negative is beneficial)",
        "",
        f"Eval speedup U1 vs Always-STOP = {ev['speedup_vs_always_stop']:.6f}x",
        f"Eval paired 95% CI = [{ev['bootstrap95_low']:.6f}, {ev['bootstrap95_high']:.6f}]",
        f"Eval exact-hash matches = {ev.get('exact_hash_matches','NA')}/{len(eval_ids)}",
        f"Eval exact-hash speedup = {ev.get('exact_hash_speedup', float('nan')):.6f}x",
        "",
        verdict,
        "",
        "Selection used hindsight/oracle-style C/S labels and on-policy stability, never U1-vs-AlwaysSTOP runtime wins.",
        "This therefore demonstrates existence/mechanism capacity, not unbiased benchmark generalization.",
    ]
    (root / "FINAL_VERDICT.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "CONFIG.json"
    if config_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new output directory")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        flexible = {"resume", "screen_only", "confirm_only", "final_only", "max_screen_problems",
                    "max_selection_attempts", "dllm_dir", "output_dir", "log_level"}
        for key, value in vars(args).items():
            if key not in flexible and old.get(key) != value:
                raise ValueError(f"Resume configuration changed: {key}")
    json_dump(config_path, vars(args))
    subprocess.run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir], cwd=ROOT, check=True)

    frozen_csv = root / "FROZEN_CRICH50_IDS.csv"
    if args.final_only or (args.resume and frozen_csv.exists() and not args.screen_only and not args.confirm_only):
        if not frozen_csv.exists():
            raise FileNotFoundError(f"--final_only requires {frozen_csv}")
        ids = pd.read_csv(frozen_csv).sort_values("run_position").problem_id.astype(int).tolist()
        final_run(args, root, ids)
        return

    # Stage A: screen. `needed` deliberately exceeds 50 so failed confirmation
    # sequences have spare candidates to swap in without re-running immediately.
    needed = max(args.initial_candidate_target, args.final_size + args.replace_count)
    if args.confirm_only:
        # Confirm-only expects prior screen files.
        screen_tr = read_all_screen_transitions(root / "screen")
        if screen_tr.empty:
            raise FileNotFoundError("--confirm_only requires existing screen batches")
        tr = resolved_non_tie(screen_tr)
        beta = fit_weighted_f2_logistic(tr)
        stats = aggregate_problem_stats(tr, beta)
        candidates = filter_screen_candidates(args, stats)
    else:
        screen_tr, candidates = screen_until_candidates(args, root, needed)

    if args.screen_only:
        print(f"Screen-only complete: {len(candidates)} candidates. See screen_candidates.csv")
        return

    # Stage B/C: build, confirm twice, replace/refill until stable.
    anchors: list[int] = []
    banned: list[int] = []
    stable_ids: list[int] | None = None
    stable_results: list[dict] | None = None

    for attempt in range(1, args.max_selection_attempts + 1):
        # Ensure enough unused candidates. If not, expand screening automatically.
        required = args.final_size + len(set(banned)) + args.replace_count
        if len(candidates) < required and not args.confirm_only:
            needed = max(len(candidates) + args.candidate_increment, required)
            screen_tr, candidates = screen_until_candidates(args, root, needed)

        ids, sel_metrics = build_sequence(args, screen_tr, candidates, anchor_ids=anchors, banned_ids=banned)
        attempt_dir = root / "confirm" / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "run_position": range(len(ids)),
            "problem_id": ids,
            "split": ["adapt" if i < args.adaptation_problems else "eval" for i in range(len(ids))],
        }).to_csv(attempt_dir / "candidate_sequence.csv", index=False)
        json_dump(attempt_dir / "screen_selection_metrics.json", sel_metrics)
        print(f"\nATTEMPT {attempt}/{args.max_selection_attempts}: screened selection C-rate={sel_metrics.get('c_rate', float('nan')):.3f}, offline F2 AUC={sel_metrics.get('offline_f2_auc', float('nan')):.3f}")
        print(f"[CONFIRM] testing exact 50-id sequence on seeds {args.confirm_seeds}; each seed restarts U1 from zero.", flush=True)

        results = []
        for seed_idx, seed in enumerate(args.confirm_seeds, start=1):
            print(f"[CONFIRM] attempt {attempt}/{args.max_selection_attempts}, seed {seed} ({seed_idx}/{len(args.confirm_seeds)}) starting...", flush=True)
            r = run_confirmation(args, root, attempt, ids, seed)
            results.append(r)
            f = r["full"]
            e = r["eval"]
            print(
                f"[CONFIRM seed {seed}] pass={r['passed_individual']} | "
                f"full C={f.get('good_c',0)}/{f.get('resolved_non_tie',0)} "
                f"({100*f.get('c_rate', float('nan')):.1f}%) | "
                f"eval learned-C={e.get('resolved_learned_c',0)} "
                f"TP/FP={e.get('learned_c_tp',0)}/{e.get('learned_c_fp',0)} | "
                f"eval AUC={e.get('temporal_auc', float('nan')):.3f} | "
                f"sum dJ learned-C={e.get('sum_delta_j_learned_c', float('nan')):.2f}",
                flush=True,
            )
        stable, reasons = stable_across_confirmations(args, results)
        json_dump(attempt_dir / "cross_seed_verdict.json", {"stable": stable, "reasons": reasons})
        if stable:
            stable_ids = ids
            stable_results = results
            break

        print("CONFIRMATION FAILED:")
        for reason in reasons:
            print("  -", reason)
        anchors_new, weak = derive_anchor_and_banned_ids(args, root, attempt, ids, results)
        # Keep only a bounded number of anchors so the next build can actually replace weak regions.
        anchors = anchors_new[: max(0, args.final_size - args.replace_count)]
        banned = sorted(set(banned).union(weak))
        print(f"Next attempt anchors={len(anchors)}, cumulative banned={len(banned)}")

    if stable_ids is None:
        # Preserve all diagnostics rather than silently weakening criteria.
        msg = (
            "No stable C-rich50 sequence satisfied the requested criteria within "
            f"{args.max_selection_attempts} attempts. The search stops here; final Always-STOP/U1 "
            "E2E is NOT run. Inspect confirm/* and SCREEN_STATUS.txt, then resume with more "
            "screen candidates or explicitly relax criteria. The runner does NOT auto-lower them."
        )
        print("\nSELECTION STOPPED: " + msg, flush=True)
        (root / "SELECTION_STOPPED_INCOMPLETE.txt").write_text(msg + "\n", encoding="utf-8")
        raise RuntimeError(msg)

    # Freeze exact order only after BOTH seeds passed.
    frozen = pd.DataFrame({
        "run_position": range(len(stable_ids)),
        "problem_id": stable_ids,
        "split": ["adapt" if i < args.adaptation_problems else "eval" for i in range(len(stable_ids))],
    })
    frozen.to_csv(frozen_csv, index=False)
    json_dump(root / "FROZEN_CONFIRMATION_SUMMARY.json", {
        "stable": True,
        "ids": stable_ids,
        "confirm_seeds": args.confirm_seeds,
        "results": stable_results,
        "criteria": {
            "min_full_c_rate": args.confirm_min_c_rate,
            "min_full_goodc": args.confirm_min_goodc,
            "max_cross_seed_c_rate_gap": args.confirm_max_c_rate_gap,
            "min_eval_learned_c": args.confirm_min_learned_c,
            "min_eval_learned_c_tp": args.confirm_min_learned_c_tp,
            "min_eval_auc": args.confirm_min_eval_auc,
            "negative_eval_learned_c_utility_required": args.confirm_require_negative_learned_utility,
        },
    })
    print("\nFROZEN stable C-rich sequence:", stable_ids)

    # Stage D: fresh held-out seed, matched E2E.
    final_run(args, root, stable_ids)


if __name__ == "__main__":
    main()
