import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_failfast_counterfactual_oracle import (
    DATASET_SIZES,
    add_latency_estimates,
    build_failfast_oracle_transitions,
    decision_state_key,
)


VERSION = "online_symmetric_oracle_matched_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_SIZES), default="gsm8k")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--oracle_sample_seed", type=int, default=2026)
    parser.add_argument("--order_seeds", type=int, nargs="*", default=[7, 19])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--adaptive-max-refinement-steps", type=int, default=16)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.02)
    parser.add_argument("--adaptive-mc-learning-rate", type=float, default=0.01)
    parser.add_argument("--adaptive-mc-mix", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-update-mode",
        choices=("td", "factual_return", "mixed"),
        default="mixed",
    )
    parser.add_argument("--adaptive-rho-alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-risk-beta", type=float, default=1.0)
    parser.add_argument("--adaptive-uncertainty-prior", type=float, default=1.0)
    parser.add_argument("--adaptive-epistemic-scale", type=float, default=0.1)
    parser.add_argument("--adaptive-q-margin", type=float, default=0.0)
    parser.add_argument("--adaptive-min-action-probability", type=float, default=0.10)
    parser.add_argument("--adaptive-max-importance-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_online_oracle_matched_test50",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    available = DATASET_SIZES[args.dataset] - 1
    if args.num_questions <= 0 or args.num_questions > available:
        raise ValueError(f"--num_questions must be in [1, {available}]")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("proposal lengths must be positive")
    if not 0.0 < args.adaptive_min_action_probability <= 0.5:
        raise ValueError("--adaptive-min-action-probability must be in (0, 0.5]")
    if args.adaptive_max_importance_weight < 1.0:
        raise ValueError("--adaptive-max-importance-weight must be at least 1")
    if len(set(args.order_seeds)) != len(args.order_seeds):
        raise ValueError("--order_seeds must not contain duplicates")


def oracle_problem_ids(args):
    population = list(range(1, DATASET_SIZES[args.dataset]))
    return sorted(
        random.Random(args.oracle_sample_seed).sample(
            population,
            args.num_questions,
        )
    )


def problem_orders(problem_ids, order_seeds):
    orders = {"canonical": list(problem_ids)}
    for seed in order_seeds:
        shuffled = list(problem_ids)
        random.Random(seed).shuffle(shuffled)
        orders[f"shuffle_{seed}"] = shuffled
    return orders


def run_streaming(command, cwd):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def base_command(args, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", args.dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(problem_id) for problem_id in problem_ids],
        "--warmup_questions", "0",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", str(args.spec_len),
        "--block_size", str(args.block_size),
        "--small_block_size", str(args.small_block_size),
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", str(args.max_spec_len),
        "--sweep_incr_len", str(args.incr_len),
        "--seed", str(args.seed),
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def adaptive_arguments(args):
    return [
        "--adaptive-td",
        "--adaptive-policy-mode", "symmetric",
        "--adaptive-max-refinement-steps", str(args.adaptive_max_refinement_steps),
        "--adaptive-learning-rate", str(args.adaptive_learning_rate),
        "--adaptive-mc-learning-rate", str(args.adaptive_mc_learning_rate),
        "--adaptive-mc-mix", str(args.adaptive_mc_mix),
        "--adaptive-update-mode", args.adaptive_update_mode,
        "--adaptive-rho-alpha", str(args.adaptive_rho_alpha),
        "--adaptive-risk-beta", str(args.adaptive_risk_beta),
        "--adaptive-uncertainty-prior", str(args.adaptive_uncertainty_prior),
        "--adaptive-epistemic-scale", str(args.adaptive_epistemic_scale),
        "--adaptive-q-margin", str(args.adaptive_q_margin),
        "--adaptive-min-action-probability",
        str(args.adaptive_min_action_probability),
        "--adaptive-max-importance-weight",
        str(args.adaptive_max_importance_weight),
        "--adaptive-use-step-feature",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
    ]


def run_phase(args, phase, problem_ids, extra_args, required_files):
    output_dir = Path(args.output_dir) / "raw" / phase
    complete = all((output_dir / filename).exists() for filename in required_files)
    if args.resume and complete:
        rows = pd.read_csv(output_dir / "benchmark_results.csv")
        if rows["problem_id"].astype(int).tolist() == problem_ids:
            print(f"RESUME {phase}", flush=True)
            return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = base_command(args, problem_ids, output_dir) + list(extra_args)
    print("\n" + "=" * 100, flush=True)
    print(f"RUN {phase} | samples={len(problem_ids)}", flush=True)
    print(f"problem_ids={problem_ids}", flush=True)
    print("=" * 100, flush=True)
    run_streaming(command, Path(__file__).resolve().parent)
    return output_dir


def annotate_results(rows, method, order_name, order):
    result = rows.copy()
    result["method"] = method
    result["order_name"] = order_name
    positions = {problem_id: index + 1 for index, problem_id in enumerate(order)}
    result["online_position"] = result["problem_id"].map(positions)
    result["measured_time_s"] = (
        result["actual_draft_time"]
        + result["actual_verify_time"]
        + result["actual_post_verify_time"]
    )
    result["measured_ms_per_output_token"] = (
        1000.0 * result["measured_time_s"] / result["output_tokens"]
    )
    return result


def prepare_decisions(path, order_name, order):
    decisions = pd.read_csv(path)
    positions = {problem_id: index + 1 for index, problem_id in enumerate(order)}
    decisions["order_name"] = order_name
    decisions["online_position"] = decisions["problem_id"].map(positions)
    decisions["state_key"] = decisions.apply(
        lambda row: decision_state_key(
            row["problem_id"],
            row["context_len"],
            row["target_len"],
            row["step"],
            row["draft_proposal"],
        ),
        axis=1,
    )
    decisions["state_occurrence"] = decisions.groupby("state_key").cumcount()
    return decisions


def match_oracle(decisions, oracle):
    oracle = oracle.copy()
    oracle["state_occurrence"] = oracle.groupby("state_key").cumcount()
    columns = [
        "state_key",
        "state_occurrence",
        "oracle_action",
        "stop_ms_per_output_token",
        "continue_ms_per_output_token",
        "actual_next_gain_tokens",
        "next_draft_latency_ms",
    ]
    matched = decisions.merge(
        oracle[columns],
        on=["state_key", "state_occurrence"],
        how="left",
        validate="one_to_one",
    )
    evaluated = matched.dropna(subset=["oracle_action"]).copy()
    evaluated["decision_correct"] = (
        evaluated["action"] == evaluated["oracle_action"]
    ).astype(int)
    evaluated["selected_ms_per_output_token"] = np.where(
        evaluated["action"].eq("stop"),
        evaluated["stop_ms_per_output_token"],
        evaluated["continue_ms_per_output_token"],
    )
    evaluated["oracle_ms_per_output_token"] = evaluated[
        ["stop_ms_per_output_token", "continue_ms_per_output_token"]
    ].min(axis=1)
    evaluated["regret_ms_per_output_token"] = (
        evaluated["selected_ms_per_output_token"]
        - evaluated["oracle_ms_per_output_token"]
    )
    return matched, evaluated


def safe_percent(numerator, denominator):
    return 100.0 * numerator / max(1, denominator)


def decision_summary(order_name, decisions, evaluated):
    predicted_stop = evaluated["action"].eq("stop")
    oracle_stop = evaluated["oracle_action"].eq("stop")
    true_stop = int((predicted_stop & oracle_stop).sum())
    false_stop = int((predicted_stop & ~oracle_stop).sum())
    true_continue = int((~predicted_stop & ~oracle_stop).sum())
    false_continue = int((~predicted_stop & oracle_stop).sum())
    available = decisions["stop_available"].astype(str).str.lower().eq("true").sum()
    return {
        "order_name": order_name,
        "decisions": len(decisions),
        "oracle_matched_decisions": len(evaluated),
        "oracle_match_coverage_percent": safe_percent(len(evaluated), available),
        "predicted_stop_rate_percent": safe_percent(
            decisions["action"].eq("stop").sum(),
            len(decisions),
        ),
        "oracle_stop_rate_percent": safe_percent(oracle_stop.sum(), len(evaluated)),
        "decision_accuracy_percent": safe_percent(
            evaluated["decision_correct"].sum(),
            len(evaluated),
        ),
        "stop_precision_percent": safe_percent(true_stop, true_stop + false_stop),
        "stop_recall_percent": safe_percent(true_stop, oracle_stop.sum()),
        "continue_precision_percent": safe_percent(
            true_continue,
            true_continue + false_continue,
        ),
        "continue_recall_percent": safe_percent(
            true_continue,
            (~oracle_stop).sum(),
        ),
        "false_stop_count": false_stop,
        "false_continue_count": false_continue,
        "mean_regret_ms_per_output_token": evaluated[
            "regret_ms_per_output_token"
        ].mean(),
        "p95_regret_ms_per_output_token": evaluated[
            "regret_ms_per_output_token"
        ].quantile(0.95),
    }


def performance_summary(baseline, adaptive, order_name):
    baseline_mspt = (
        1000.0 * baseline["measured_time_s"].sum() / baseline["output_tokens"].sum()
    )
    adaptive_mspt = (
        1000.0 * adaptive["measured_time_s"].sum() / adaptive["output_tokens"].sum()
    )
    paired = baseline[["problem_id", "measured_ms_per_output_token", "output_token_hash"]].merge(
        adaptive[["problem_id", "measured_ms_per_output_token", "output_token_hash"]],
        on="problem_id",
        suffixes=("_failfast", "_adaptive"),
        validate="one_to_one",
    )
    speedups = (
        paired["measured_ms_per_output_token_failfast"]
        / paired["measured_ms_per_output_token_adaptive"]
    )
    return {
        "order_name": order_name,
        "failfast_ms_per_output_token": baseline_mspt,
        "adaptive_ms_per_output_token": adaptive_mspt,
        "pooled_speedup_vs_failfast": baseline_mspt / adaptive_mspt,
        "geometric_speedup_vs_failfast": float(np.exp(np.log(speedups).mean())),
        "adaptive_win_rate_percent": safe_percent((speedups > 1.0).sum(), len(speedups)),
        "output_match_rate_percent": safe_percent(
            (
                paired["output_token_hash_failfast"]
                == paired["output_token_hash_adaptive"]
            ).sum(),
            len(paired),
        ),
        "draft_passes_per_100_tokens": (
            100.0 * adaptive["total_num_forward_passes"].sum()
            / adaptive["output_tokens"].sum()
        ),
        "verifier_rounds_per_100_tokens": (
            100.0 * adaptive["num_speculation_rounds"].sum()
            / adaptive["output_tokens"].sum()
        ),
        "acceptance_rate_percent": (
            100.0 * adaptive["accepted_tokens"].sum()
            / max(1, adaptive["drafted_tokens"].sum())
        ),
    }


def position_summary(adaptive, evaluated, order_name):
    rows = adaptive.copy()
    rows["position_bin"] = pd.cut(
        rows["online_position"],
        bins=[0, 10, 20, 30, 40, math.inf],
        labels=["1-10", "11-20", "21-30", "31-40", "41-50"],
    )
    decision_accuracy = evaluated.groupby("online_position")["decision_correct"].mean()
    records = []
    for label, group in rows.groupby("position_bin", observed=True, sort=False):
        positions = group["online_position"]
        relevant_accuracy = decision_accuracy.reindex(positions).dropna()
        records.append({
            "order_name": order_name,
            "position_bin": str(label),
            "num_questions": len(group),
            "adaptive_ms_per_output_token": (
                1000.0 * group["measured_time_s"].sum() / group["output_tokens"].sum()
            ),
            "oracle_decision_accuracy_percent": (
                100.0 * relevant_accuracy.mean()
                if len(relevant_accuracy) else np.nan
            ),
        })
    return records


def overfit_summary(performance, decisions, common_actions, initial_checks):
    speedups = performance["pooled_speedup_vs_failfast"]
    accuracies = decisions["decision_accuracy_percent"]
    stop_rates = decisions["predicted_stop_rate_percent"]
    speedup_cv = 100.0 * speedups.std(ddof=0) / max(speedups.mean(), 1e-12)
    agreement = (
        100.0 * common_actions["unanimous_action"].mean()
        if len(common_actions) else np.nan
    )
    order_sensitive = bool(
        speedups.max() - speedups.min() > 0.03
        or accuracies.max() - accuracies.min() > 10.0
        or (pd.notna(agreement) and agreement < 80.0)
    )
    return pd.DataFrame([{
        "online_runs": len(performance),
        "all_runs_zero_initialized": bool(all(initial_checks)),
        "external_state_loaded": False,
        "pooled_speedup_mean": speedups.mean(),
        "pooled_speedup_min": speedups.min(),
        "pooled_speedup_max": speedups.max(),
        "pooled_speedup_range": speedups.max() - speedups.min(),
        "pooled_speedup_cv_percent": speedup_cv,
        "oracle_accuracy_mean_percent": accuracies.mean(),
        "oracle_accuracy_range_percent": accuracies.max() - accuracies.min(),
        "stop_rate_mean_percent": stop_rates.mean(),
        "stop_rate_range_percent": stop_rates.max() - stop_rates.min(),
        "common_exact_states": len(common_actions),
        "cross_order_action_agreement_percent": agreement,
        "order_sensitivity_flag": order_sensitive,
        "interpretation": (
            "The flag detects order sensitivity, not classical parameter overfit. "
            "Every run starts from an empty controller and predicts before feedback."
        ),
    }])


def write_manifest(args, output_dir, problem_ids, orders):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "version": VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "oracle_matched_problem_ids": problem_ids,
        "online_orders": orders,
        "external_adaptive_state": None,
        "oracle_matching_rule": (
            "problem_id + context_len + target_len + refinement_step + exact proposal IDs"
        ),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = oracle_problem_ids(args)
    orders = problem_orders(problem_ids, args.order_seeds)

    baseline_dir = run_phase(
        args,
        "failfast_oracle",
        problem_ids,
        ["--collect_bucket_oracle"],
        ["benchmark_results.csv", "bucket_oracle_snapshots.csv"],
    )
    baseline = annotate_results(
        pd.read_csv(baseline_dir / "benchmark_results.csv"),
        "failfast",
        "canonical",
        problem_ids,
    )
    snapshots = add_latency_estimates(
        pd.read_csv(baseline_dir / "bucket_oracle_snapshots.csv")
    )
    oracle = build_failfast_oracle_transitions(snapshots)
    if oracle.empty:
        raise ValueError("No exact one-pass oracle transitions were collected")

    adaptive_frames = []
    decision_frames = []
    matched_frames = []
    evaluated_frames = []
    performance_records = []
    decision_records = []
    position_records = []
    initial_checks = []

    for order_name, order in orders.items():
        phase = f"adaptive_online_{order_name}"
        phase_dir = run_phase(
            args,
            phase,
            order,
            adaptive_arguments(args),
            [
                "benchmark_results.csv",
                "adaptive_td_decisions.csv",
                "adaptive_td_runtime_state.json",
            ],
        )
        adaptive = annotate_results(
            pd.read_csv(phase_dir / "benchmark_results.csv"),
            "adaptive_td",
            order_name,
            order,
        )
        decisions = prepare_decisions(
            phase_dir / "adaptive_td_decisions.csv",
            order_name,
            order,
        )
        matched, evaluated = match_oracle(decisions, oracle)
        initial = decisions.sort_values(
            ["online_position", "round_id", "decision_id"]
        ).iloc[0]
        initial_checks.append(
            abs(float(initial["q_stop_mean"])) < 1e-12
            and abs(float(initial["q_continue_mean"])) < 1e-12
            and int(initial["early_stop_observations_before"]) == 0
        )
        adaptive_frames.append(adaptive)
        decision_frames.append(decisions)
        matched_frames.append(matched)
        evaluated_frames.append(evaluated)
        performance_records.append(performance_summary(baseline, adaptive, order_name))
        decision_records.append(decision_summary(order_name, decisions, evaluated))
        position_records.extend(position_summary(adaptive, evaluated, order_name))

    performance = pd.DataFrame(performance_records)
    decision_metrics = pd.DataFrame(decision_records)
    all_decisions = pd.concat(decision_frames, ignore_index=True, sort=False)
    all_matched = pd.concat(matched_frames, ignore_index=True, sort=False)
    all_evaluated = pd.concat(evaluated_frames, ignore_index=True, sort=False)
    adaptive_results = pd.concat(adaptive_frames, ignore_index=True, sort=False)

    action_pivot = all_evaluated.drop_duplicates(
        ["order_name", "state_key"]
    ).pivot(index="state_key", columns="order_name", values="action")
    action_pivot = action_pivot.dropna()
    common_actions = pd.DataFrame({
        "state_key": action_pivot.index,
        "unanimous_action": action_pivot.nunique(axis=1).eq(1).astype(int).to_numpy(),
    })
    overfit = overfit_summary(
        performance,
        decision_metrics,
        common_actions,
        initial_checks,
    )

    baseline.to_csv(output_dir / "failfast_results.csv", index=False)
    snapshots.to_csv(output_dir / "oracle_snapshots.csv", index=False)
    oracle.to_csv(output_dir / "oracle_transitions.csv", index=False)
    adaptive_results.to_csv(output_dir / "adaptive_per_problem_results.csv", index=False)
    all_decisions.to_csv(output_dir / "adaptive_decisions.csv", index=False)
    all_matched.to_csv(output_dir / "oracle_match_all_decisions.csv", index=False)
    all_evaluated.to_csv(output_dir / "oracle_evaluated_decisions.csv", index=False)
    performance.to_csv(output_dir / "order_performance_summary.csv", index=False)
    decision_metrics.to_csv(output_dir / "oracle_decision_summary.csv", index=False)
    pd.DataFrame(position_records).to_csv(
        output_dir / "online_position_summary.csv",
        index=False,
    )
    common_actions.to_csv(output_dir / "cross_order_common_states.csv", index=False)
    overfit.to_csv(output_dir / "overfit_diagnostics.csv", index=False)
    write_manifest(args, output_dir, problem_ids, orders)

    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nORDER PERFORMANCE SUMMARY")
    print(performance.to_string(index=False))
    print("\nORACLE DECISION SUMMARY")
    print(decision_metrics.to_string(index=False))
    print("\nOVERFIT/ORDER-SENSITIVITY DIAGNOSTICS")
    print(overfit.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
