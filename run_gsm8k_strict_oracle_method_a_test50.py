import argparse
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_td import FEATURE_NAMES
from run_failfast_counterfactual_oracle import run_phase as run_standard_phase
from run_math_feature_ablation_benchmark import adaptive_args
from run_strict_greedy_math50 import (
    aggregate_report,
    build_verifier_profile,
    load_phase,
    run_phase as run_strict_phase,
)
from strict_greedy_oracle import build_oracle_state_key


ROOT = Path(__file__).resolve().parent
VERSION = "gsm8k_strict_oracle_method_a_test50_v1"
DATASET_SIZE = 1319


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--adaptive_max_refinement_steps", type=int, default=16)
    parser.add_argument("--adaptive_learning_rate", type=float, default=0.02)
    parser.add_argument("--adaptive_mc_learning_rate", type=float, default=0.01)
    parser.add_argument("--adaptive_mc_mix", type=float, default=0.5)
    parser.add_argument(
        "--adaptive_update_mode",
        choices=("td", "factual_return", "mixed"),
        default="mixed",
    )
    parser.add_argument("--adaptive_rho_alpha", type=float, default=0.05)
    parser.add_argument("--adaptive_risk_beta", type=float, default=1.0)
    parser.add_argument(
        "--adaptive_stop_probability_threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument("--adaptive_uncertainty_prior", type=float, default=1.0)
    parser.add_argument("--adaptive_epistemic_scale", type=float, default=0.1)
    parser.add_argument("--adaptive_q_margin", type=float, default=0.0)
    parser.add_argument("--adaptive_explore_epsilon", type=float, default=0.10)
    parser.add_argument("--adaptive_explore_min", type=float, default=0.01)
    parser.add_argument("--adaptive_explore_decay", type=float, default=0.998)
    parser.add_argument("--adaptive_warmup_rounds", type=int, default=20)
    parser.add_argument(
        "--adaptive_early_stop_min_observations",
        type=int,
        default=32,
    )
    parser.add_argument("--adaptive_min_action_probability", type=float, default=0.10)
    parser.add_argument("--adaptive_max_importance_weight", type=float, default=5.0)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_gsm8k_strict_oracle_method_a_test50",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    available = DATASET_SIZE - args.warmup_questions
    if args.num_questions <= 0 or args.num_questions > available:
        raise ValueError(f"--num_questions must be in [1, {available}]")
    if args.spec_len != 8 or args.incr_len != 8:
        raise ValueError("this matched benchmark requires --spec_len=8 and --incr_len=8")
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")


def sampled_problem_ids(args):
    population = list(range(args.warmup_questions, DATASET_SIZE))
    return sorted(random.Random(args.sample_seed).sample(
        population,
        args.num_questions,
    ))


def method_a_phase_complete(phase_dir, problem_ids):
    result_path = phase_dir / "benchmark_results.csv"
    decision_path = phase_dir / "adaptive_td_decisions.csv"
    state_path = phase_dir / "adaptive_td_runtime_state.json"
    if not all(path.exists() and path.stat().st_size > 0 for path in (
        result_path,
        decision_path,
        state_path,
    )):
        return False
    try:
        results = pd.read_csv(result_path)
        decisions = pd.read_csv(decision_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.EmptyDataError, json.JSONDecodeError):
        return False
    return (
        len(results) == len(problem_ids)
        and set(results["problem_id"].astype(int)) == set(map(int, problem_ids))
        and not decisions.empty
        and state.get("controller_name") == "avg_td"
    )


def run_method_a_fresh(args, problem_ids):
    phase_dir = Path(args.output_dir) / "raw" / "method_a_fresh"
    if args.resume and method_a_phase_complete(phase_dir, problem_ids):
        print("RESUME method_a_fresh", flush=True)
        return phase_dir
    if phase_dir.exists():
        shutil.rmtree(phase_dir)
    original_resume = args.resume
    args.resume = False
    try:
        return run_standard_phase(
            args,
            problem_ids,
            "method_a_fresh",
            adaptive_args(args, ()),
            [
                "benchmark_results.csv",
                "adaptive_td_decisions.csv",
                "adaptive_td_runtime_state.json",
            ],
        )
    finally:
        args.resume = original_resume


def source_config(args):
    return {
        "dataset": "gsm8k",
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "target_model_name": args.target_model_name,
        "drafter_threshold": args.drafter_threshold,
        "lowconf_threshold": args.lowconf_threshold,
        "max_spec_len": args.max_spec_len,
        "seed": args.seed,
    }


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def binary_auc(labels, scores):
    labels = pd.Series(labels).astype(int).reset_index(drop=True)
    scores = pd.Series(scores, dtype=float).reset_index(drop=True)
    valid = labels.notna() & scores.notna()
    labels = labels[valid]
    scores = scores[valid]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    positive_rank_sum = float(ranks[labels.eq(1)].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def normalized_proposal(value):
    if isinstance(value, str):
        value = json.loads(value)
    return [None if token is None else int(token) for token in (value or [])]


def add_method_a_state_identity(decisions):
    result = decisions.copy()
    result["state_key"] = result.apply(
        lambda row: build_oracle_state_key(
            row["problem_id"],
            row["context_len"],
            row["target_len"],
            row["step"],
            normalized_proposal(row["draft_proposal"]),
        ),
        axis=1,
    )
    result["state_occurrence"] = result.groupby("state_key").cumcount()
    return result


def add_oracle_state_identity(decisions):
    result = decisions.copy()
    if "state_key" not in result:
        result["state_key"] = result.apply(
            lambda row: build_oracle_state_key(
                row["sample_id"],
                row["context_len"],
                row["accumulated_proposal_length"],
                row["refinement_step"],
                normalized_proposal(row["draft_proposal"]),
            ),
            axis=1,
        )
    result["state_occurrence"] = result.groupby("state_key").cumcount()
    return result


def parse_feature_matrix(decisions):
    rows = [json.loads(value) if isinstance(value, str) else value
            for value in decisions["features"]]
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Method A feature log has an unexpected shape")
    return matrix


def match_decisions(method_a, oracle):
    method = add_method_a_state_identity(method_a)
    truth = add_oracle_state_identity(oracle)
    method_matrix = parse_feature_matrix(method)
    for index, name in enumerate(FEATURE_NAMES):
        method[f"feature_{name}"] = method_matrix[:, index]
    matched = method.merge(
        truth,
        on=["state_key", "state_occurrence"],
        how="inner",
        suffixes=("_method_a", "_oracle"),
        validate="one_to_one",
    )
    matched["method_a_action"] = matched["action"]
    matched["oracle_action"] = matched["chosen_action"]
    matched["oracle_stop"] = matched["oracle_action"].eq("stop").astype(int)
    matched["method_a_stop"] = matched["method_a_action"].eq("stop").astype(int)
    matched["decision_correct"] = (
        matched["method_a_action"] == matched["oracle_action"]
    ).astype(int)
    matched["confusion_class"] = np.select(
        [
            matched["oracle_stop"].eq(1) & matched["method_a_stop"].eq(1),
            matched["oracle_stop"].eq(0) & matched["method_a_stop"].eq(1),
            matched["oracle_stop"].eq(1) & matched["method_a_stop"].eq(0),
        ],
        ["true_stop", "false_stop", "false_continue"],
        default="true_continue",
    )
    return method, truth, matched


def confusion_summary(matched):
    counts = matched["confusion_class"].value_counts()
    true_stop = int(counts.get("true_stop", 0))
    false_stop = int(counts.get("false_stop", 0))
    false_continue = int(counts.get("false_continue", 0))
    true_continue = int(counts.get("true_continue", 0))
    total = len(matched)
    stop_recall = safe_divide(true_stop, true_stop + false_continue)
    continue_recall = safe_divide(true_continue, true_continue + false_stop)
    return pd.DataFrame([{
        "matched_states": total,
        "true_stop": true_stop,
        "false_stop": false_stop,
        "false_continue": false_continue,
        "true_continue": true_continue,
        "accuracy_percent": 100.0 * safe_divide(
            true_stop + true_continue,
            total,
        ),
        "balanced_accuracy_percent": 50.0 * (stop_recall + continue_recall),
        "stop_precision_percent": 100.0 * safe_divide(
            true_stop,
            true_stop + false_stop,
        ),
        "stop_recall_percent": 100.0 * stop_recall,
        "continue_precision_percent": 100.0 * safe_divide(
            true_continue,
            true_continue + false_continue,
        ),
        "continue_recall_percent": 100.0 * continue_recall,
        "oracle_stop_rate_percent": 100.0 * float(matched["oracle_stop"].mean()),
        "method_a_stop_rate_percent": 100.0 * float(
            matched["method_a_stop"].mean()
        ),
        "false_continue_rate_when_method_continues_percent": 100.0 * safe_divide(
            false_continue,
            false_continue + true_continue,
        ),
        "stop_probability_oracle_stop_mean": float(
            matched.loc[matched.oracle_stop.eq(1), "stop_probability"].mean()
        ),
        "stop_probability_oracle_continue_mean": float(
            matched.loc[matched.oracle_stop.eq(0), "stop_probability"].mean()
        ),
        "stop_probability_oracle_auc": binary_auc(
            matched["oracle_stop"],
            matched["stop_probability"],
        ),
        "advantage_mean_oracle_stop": float(
            matched.loc[matched.oracle_stop.eq(1), "advantage_mean"].mean()
        ),
        "advantage_mean_oracle_continue": float(
            matched.loc[matched.oracle_stop.eq(0), "advantage_mean"].mean()
        ),
        "advantage_mean_oracle_auc": binary_auc(
            matched["oracle_stop"],
            matched["advantage_mean"],
        ),
    }])


def feature_alignment(matched, state_path):
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    stop_theta = np.asarray(state["actions"]["stop"]["theta"], dtype=float)
    continue_theta = np.asarray(
        state["actions"]["continue"]["theta"],
        dtype=float,
    )
    records = []
    for index, name in enumerate(FEATURE_NAMES):
        values = pd.to_numeric(matched[f"feature_{name}"], errors="coerce")
        stop_values = values[matched.oracle_stop.eq(1)]
        continue_values = values[matched.oracle_stop.eq(0)]
        auc = binary_auc(matched["oracle_stop"], values)
        std = float(values.std(ddof=0))
        theta_difference = float(stop_theta[index] - continue_theta[index])
        effect = theta_difference * std
        oracle_direction = (
            "higher_means_stop" if auc > 0.52
            else "lower_means_stop" if auc < 0.48
            else "weak_or_neutral"
        )
        learned_direction = (
            "higher_means_stop" if effect > 1e-9
            else "lower_means_stop" if effect < -1e-9
            else "neutral"
        )
        if oracle_direction == "weak_or_neutral" or learned_direction == "neutral":
            alignment = "weak_or_unidentified"
        else:
            alignment = "aligned" if oracle_direction == learned_direction else "opposite"
        records.append({
            "feature": name,
            "matched_states": len(values),
            "mean": float(values.mean()),
            "std": std,
            "unique_values": int(values.nunique()),
            "oracle_stop_mean": float(stop_values.mean()),
            "oracle_continue_mean": float(continue_values.mean()),
            "oracle_stop_auc": auc,
            "oracle_direction": oracle_direction,
            "stop_theta": float(stop_theta[index]),
            "continue_theta": float(continue_theta[index]),
            "theta_stop_minus_continue": theta_difference,
            "standardized_final_effect": effect,
            "learned_direction": learned_direction,
            "alignment": alignment,
        })
    return pd.DataFrame(records)


def learning_windows(matched, problem_ids):
    order = {int(problem_id): index for index, problem_id in enumerate(problem_ids)}
    result = matched.copy()
    result["question_order"] = result["problem_id"].astype(int).map(order)
    result["learning_window"] = result["question_order"].floordiv(10).astype(int) + 1
    records = []
    for window, group in result.groupby("learning_window", sort=True):
        summary = confusion_summary(group).iloc[0].to_dict()
        summary.update({
            "learning_window": int(window),
            "question_start": int(group["question_order"].min()) + 1,
            "question_end": int(group["question_order"].max()) + 1,
            "exploration_rate_percent": 100.0 * float(
                pd.to_numeric(group["exploration_used"], errors="coerce")
                .fillna(0)
                .astype(bool)
                .mean()
            ),
        })
        records.append(summary)
    return pd.DataFrame(records)


def feature_bins(matched):
    records = []
    for name in FEATURE_NAMES:
        values = pd.to_numeric(matched[f"feature_{name}"], errors="coerce")
        if values.nunique() < 2:
            continue
        bins = pd.qcut(values, q=min(5, values.nunique()), duplicates="drop")
        frame = pd.DataFrame({
            "bin": bins.astype(str),
            "value": values,
            "oracle_stop": matched["oracle_stop"],
            "method_a_stop": matched["method_a_stop"],
            "correct": matched["decision_correct"],
        })
        for label, group in frame.groupby("bin", sort=False):
            records.append({
                "feature": name,
                "bin": label,
                "states": len(group),
                "value_mean": float(group["value"].mean()),
                "oracle_stop_rate_percent": 100.0 * float(
                    group["oracle_stop"].mean()
                ),
                "method_a_stop_rate_percent": 100.0 * float(
                    group["method_a_stop"].mean()
                ),
                "decision_accuracy_percent": 100.0 * float(
                    group["correct"].mean()
                ),
            })
    return pd.DataFrame(records)


def aggregate_method(results, method):
    output_tokens = float(results["output_tokens"].sum())
    algorithm_time = float(results["actual_algorithm_time"].sum())
    drafted = float(results["drafted_tokens"].sum())
    accepted = float(results["accepted_tokens"].sum())
    return {
        "method": method,
        "num_questions": int(len(results)),
        "output_tokens": int(output_tokens),
        "algorithm_time_s": algorithm_time,
        "ms_per_output_token": 1000.0 * algorithm_time / max(1.0, output_tokens),
        "draft_time_s": float(results["actual_draft_time"].sum()),
        "verify_time_s": float(results["actual_verify_time"].sum()),
        "post_verify_time_s": float(results["actual_post_verify_time"].sum()),
        "draft_passes": int(results["total_num_forward_passes"].sum()),
        "verifier_rounds": int(results["num_speculation_rounds"].sum()),
        "draft_passes_per_100_tokens": 100.0 * float(
            results["total_num_forward_passes"].sum()
        ) / max(1.0, output_tokens),
        "verifier_rounds_per_100_tokens": 100.0 * float(
            results["num_speculation_rounds"].sum()
        ) / max(1.0, output_tokens),
        "acceptance_rate_percent": 100.0 * accepted / max(1.0, drafted),
    }


def build_method_reports(output_dir, baseline, oracle, method_a):
    summary = pd.DataFrame([
        aggregate_method(baseline, "failfast"),
        aggregate_method(oracle, "strict_greedy_oracle"),
        aggregate_method(method_a, "method_a_fresh"),
    ])
    failfast_ms = float(
        summary.loc[summary.method.eq("failfast"), "ms_per_output_token"].iloc[0]
    )
    summary["speedup_vs_failfast"] = failfast_ms / summary["ms_per_output_token"]
    summary.to_csv(output_dir / "method_summary.csv", index=False)

    key_columns = ["problem_id", "actual_algorithm_time", "output_tokens", "output_token_hash"]
    paired = baseline[key_columns].merge(
        oracle[key_columns],
        on="problem_id",
        suffixes=("_failfast", "_oracle"),
        validate="one_to_one",
    ).merge(
        method_a[key_columns],
        on="problem_id",
        validate="one_to_one",
    ).rename(columns={
        "actual_algorithm_time": "actual_algorithm_time_method_a",
        "output_tokens": "output_tokens_method_a",
        "output_token_hash": "output_token_hash_method_a",
    })
    paired["method_a_speedup_vs_failfast"] = (
        paired["actual_algorithm_time_failfast"]
        / paired["actual_algorithm_time_method_a"]
    )
    paired["oracle_speedup_vs_failfast"] = (
        paired["actual_algorithm_time_failfast"]
        / paired["actual_algorithm_time_oracle"]
    )
    paired["all_output_hashes_match"] = (
        paired["output_token_hash_failfast"].astype(str)
        == paired["output_token_hash_oracle"].astype(str)
    ) & (
        paired["output_token_hash_failfast"].astype(str)
        == paired["output_token_hash_method_a"].astype(str)
    )
    paired.to_csv(output_dir / "paired_method_results.csv", index=False)
    return summary, paired


def main():
    args = parse_args()
    validate_args(args)
    args.dataset = "gsm8k"
    args.adaptive_state_path = None
    args.oracle_only = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = sampled_problem_ids(args)
    (output_dir / "sampled_problem_ids.json").write_text(
        json.dumps(problem_ids, indent=2),
        encoding="utf-8",
    )
    source = source_config(args)
    started = time.time()

    method_a_dir = run_method_a_fresh(args, problem_ids)

    prepass_dir = run_strict_phase(
        args,
        source,
        problem_ids,
        "verifier_profile_prepass",
        False,
    )
    profile_path = output_dir / "verifier_profile.json"
    profile = build_verifier_profile(prepass_dir, profile_path)
    search_dir = run_strict_phase(
        args,
        source,
        problem_ids,
        "oracle_search",
        True,
        profile_path,
        require_decisions=True,
    )
    policy_path = search_dir / "strict_greedy_policy.json"
    oracle_replay_dir = run_strict_phase(
        args,
        source,
        problem_ids,
        "oracle_replay",
        True,
        profile_path,
        replay_policy=policy_path,
    )

    baseline_results, baseline_calls, _ = load_phase(
        prepass_dir,
        "verifier_profile_prepass",
        "failfast",
    )
    oracle_results, oracle_calls, _ = load_phase(
        oracle_replay_dir,
        "oracle_replay",
        "oracle",
    )
    oracle_search = pd.read_csv(
        search_dir / "greedy_local_oracle_decisions.csv"
    )
    aggregate_report(
        output_dir,
        profile,
        [
            ("verifier_profile_prepass", "failfast"),
            ("oracle_replay", "oracle"),
        ],
        {
            "verifier_profile_prepass": (
                baseline_results,
                baseline_calls,
                pd.DataFrame(),
            ),
            "oracle_replay": (
                oracle_results,
                oracle_calls,
                pd.DataFrame(),
            ),
        },
        problem_ids,
        oracle_search,
        baseline_results,
    )

    method_a_results = pd.read_csv(method_a_dir / "benchmark_results.csv")
    method_a_decisions = pd.read_csv(method_a_dir / "adaptive_td_decisions.csv")
    method, oracle, matched = match_decisions(method_a_decisions, oracle_search)
    if matched.empty:
        raise RuntimeError("Method A and strict oracle produced no exact matched states")

    confusion = confusion_summary(matched)
    features = feature_alignment(
        matched,
        method_a_dir / "adaptive_td_runtime_state.json",
    )
    windows = learning_windows(matched, problem_ids)
    bins = feature_bins(matched)
    coverage = pd.DataFrame([{
        "method_a_states": len(method),
        "oracle_states": len(oracle),
        "matched_states": len(matched),
        "matched_questions": int(matched["problem_id"].nunique()),
        "total_questions": len(problem_ids),
        "method_a_state_coverage_percent": 100.0 * len(matched) / max(1, len(method)),
        "oracle_state_coverage_percent": 100.0 * len(matched) / max(1, len(oracle)),
        "method_a_duplicate_state_keys": int(method["state_key"].duplicated().sum()),
        "oracle_duplicate_state_keys": int(oracle["state_key"].duplicated().sum()),
    }])
    method_summary, paired = build_method_reports(
        output_dir,
        baseline_results,
        oracle_results,
        method_a_results,
    )

    method.to_csv(output_dir / "method_a_decisions_with_state_key.csv", index=False)
    oracle.to_csv(output_dir / "strict_oracle_decisions_with_state_key.csv", index=False)
    matched.to_csv(output_dir / "exact_matched_decisions.csv", index=False)
    confusion.to_csv(output_dir / "decision_confusion_summary.csv", index=False)
    features.to_csv(output_dir / "feature_oracle_alignment.csv", index=False)
    windows.to_csv(output_dir / "online_learning_windows.csv", index=False)
    bins.to_csv(output_dir / "feature_action_bins.csv", index=False)
    coverage.to_csv(output_dir / "state_match_coverage.csv", index=False)

    manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": "gsm8k",
        "problem_ids": problem_ids,
        "arguments": vars(args),
        "method_a_initialization": (
            "Fresh process with no --adaptive-state-path; online updates use only "
            "earlier factual observations from this ordered GSM8K run."
        ),
        "oracle_definition": (
            "Latest strict one-step greedy oracle: compare STOP@t against "
            "CONTINUE@t then forced STOP@(t+1), including predicted extra verifier "
            "calls, while preserving original FailFast outer extension behavior."
        ),
        "state_match_definition": (
            "Exact SHA-256 identity over problem_id, context_len, proposal_length, "
            "refinement_step, and every draft proposal token; repeated identical "
            "states are paired by occurrence index."
        ),
        "elapsed_runner_hours": (time.time() - started) / 3600.0,
        "method_a_state_loaded": False,
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        manifest["git_commit"] = None
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nMETHOD SUMMARY")
    print(method_summary.to_string(index=False))
    print("\nSTATE MATCH COVERAGE")
    print(coverage.to_string(index=False))
    print("\nMETHOD A VS STRICT ORACLE")
    print(confusion.to_string(index=False))
    print("\nFEATURE ALIGNMENT")
    print(features.to_string(index=False))
    print("\nONLINE LEARNING WINDOWS")
    print(windows.to_string(index=False))
    print(f"\nAll output hashes match: {100.0 * paired.all_output_hashes_match.mean():.1f}%")
    print(f"Saved report: {output_dir}")
    print(f"Saved archive: {archive}")


if __name__ == "__main__":
    main()
