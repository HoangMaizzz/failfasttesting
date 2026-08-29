import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "compact6_shared_value_explicit_advantage_logical_frame_v4"
METHOD = "otrc_v2_2_compact_factual_no_bootstrap_shared_value_advantage"
DATASETS = ("math", "gsm8k", "aime", "humaneval")
VALUE_LEARNING_RATE = 0.015
ADVANTAGE_LEARNING_RATE = 0.02


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("none", "int8", "int4"),
        default="none",
    )
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_shared_value_advantage_active_block_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include_failfast_baseline",
        action="store_true",
        help="Run the matched original FailFast baseline and build paired reports.",
    )
    parser.add_argument(
        "--include_policy_controls",
        action="store_true",
        help="Run frozen-STOP and seeded random-STOP controls.",
    )
    parser.add_argument(
        "--greedy_policy",
        action="store_true",
        help=(
            "Use deterministic Shared V+A actions (STOP iff Q_STOP - Q_CONTINUE "
            "> q_margin) while keeping online factual learning enabled."
        ),
    )
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    available = min(len(PROBLEM_IDS[dataset]) for dataset in args.datasets)
    if args.num_questions > available:
        raise ValueError(f"--num_questions cannot exceed {available}")
    if args.warmup_questions != 1:
        raise ValueError("the matched benchmark requires one warmup question")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not 0.0 < args.drafter_threshold <= 1.0:
        raise ValueError("--drafter_threshold must be in (0, 1]")
    if not 0.0 <= args.lowconf_threshold <= 1.0:
        raise ValueError("--lowconf_threshold must be in [0, 1]")
    if args.target_device < 0 or args.drafter_device < 0:
        raise ValueError("CUDA device indices must be non-negative")
    if args.include_failfast_baseline and not set(args.datasets).issubset(
        {"math", "gsm8k"}
    ):
        raise ValueError(
            "the integrated FailFast baseline currently supports math and gsm8k"
        )


def benchmark_command(args, *, policy_ablation="learned", output_dir=None):
    command = [
        sys.executable,
        "-u",
        "run_otrc_v2_td_benchmark.py",
        "--datasets",
        *args.datasets,
        "--num_questions",
        str(args.num_questions),
        "--feature_schema",
        "otrc_v2_2_compact_td",
        "--credit_assignment",
        "verifier_boundary_factual_no_bootstrap",
        "--value_parameterization",
        "shared_value_advantage",
        "--shared_value_learning_rate",
        str(VALUE_LEARNING_RATE),
        "--shared_advantage_learning_rate",
        str(ADVANTAGE_LEARNING_RATE),
        "--adaptive_learning_rate",
        "0.02",
        "--adaptive_mc_learning_rate",
        "0.01",
        "--adaptive_mc_mix",
        "0.5",
        "--adaptive_update_mode",
        "mixed",
        "--adaptive_rho_alpha",
        "0.05",
        "--rho_warmup_boundaries",
        "0",
        "--policy_weight_ema_beta",
        "0.0",
        "--policy_weight_ema_mode",
        "global_step",
        "--adaptive_factual_ema_alpha",
        "0.2",
        "--adaptive_risk_beta",
        "1.0",
        "--adaptive_stop_probability_threshold",
        "0.75",
        "--adaptive_uncertainty_prior",
        "1.0",
        "--adaptive_epistemic_scale",
        "0.1",
        "--adaptive_q_margin",
        "0.0",
        "--adaptive_explore_epsilon",
        "0.0" if args.greedy_policy else "0.10",
        "--adaptive_explore_min",
        "0.0" if args.greedy_policy else "0.01",
        "--adaptive_explore_decay",
        "0.998",
        "--adaptive_warmup_rounds",
        "20",
        "--adaptive_early_stop_min_observations",
        "32",
        "--adaptive_policy_mode",
        "symmetric_greedy" if args.greedy_policy else "symmetric",
        "--adaptive_min_action_probability",
        "0.10",
        "--adaptive_max_importance_weight",
        "5.0",
        "--adaptive_weight_snapshot_interval",
        "100",
        "--policy_ablation",
        policy_ablation,
        "--warmup_questions",
        str(args.warmup_questions),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--spec_len",
        "8",
        "--incr_len",
        "8",
        "--max_spec_len",
        "64",
        "--block_size",
        "32",
        "--small_block_size",
        "8",
        "--target_model_name",
        args.target_model_name,
        "--dllm_dir",
        args.dllm_dir,
        "--target_device",
        str(args.target_device),
        "--drafter_device",
        str(args.drafter_device),
        "--target_quantization",
        getattr(args, "target_quantization", "none"),
        "--drafter_threshold",
        str(args.drafter_threshold),
        "--lowconf_threshold",
        str(args.lowconf_threshold),
        "--seed",
        "42",
        "--output_dir",
        str(output_dir or args.output_dir),
        "--log_level",
        args.log_level,
    ]
    if args.resume:
        command.append("--resume")
    return command


def policy_control_command(args, policy_ablation):
    if policy_ablation not in {"frozen_stop", "random_stop"}:
        raise ValueError(f"unknown policy control: {policy_ablation}")
    output_dir = Path(args.output_dir) / "policy_controls" / policy_ablation
    return benchmark_command(
        args,
        policy_ablation=policy_ablation,
        output_dir=output_dir,
    )


def failfast_baseline_command(args):
    baseline_dir = Path(args.output_dir) / "matched_failfast_baseline"
    command = [
        sys.executable,
        "-u",
        "run_matched_failfast_baseline.py",
        "--datasets",
        *args.datasets,
        "--num_questions",
        str(args.num_questions),
        "--warmup_questions",
        str(args.warmup_questions),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--drafter_threshold",
        str(args.drafter_threshold),
        "--lowconf_threshold",
        str(args.lowconf_threshold),
        "--target_device",
        str(args.target_device),
        "--drafter_device",
        str(args.drafter_device),
        "--target_quantization",
        getattr(args, "target_quantization", "none"),
        "--target_model_name",
        args.target_model_name,
        "--dllm_dir",
        args.dllm_dir,
        "--output_dir",
        str(baseline_dir),
        "--skip_archive",
        "--log_level",
        args.log_level,
    ]
    if args.resume:
        command.append("--resume")
    return command


def run_streaming(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
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


def time_bins(frame):
    if frame.empty:
        return pd.Series(dtype="object")
    ranks = frame.groupby("dataset").cumcount()
    sizes = frame.groupby("dataset")["dataset"].transform("size")
    quartiles = (4 * ranks // sizes.clip(lower=1)).clip(upper=3)
    return quartiles.map({0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"})


def shared_learning_dynamics(decisions):
    required = {
        "dataset",
        "shared_value_mean",
        "explicit_advantage_mean",
        "legacy_advantage_risk",
        "stop_probability",
        "action",
        "exploration_used",
        "controller_latency_ms",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"shared decision log is missing: {sorted(missing)}")
    frame = decisions.copy()
    frame["time_bin"] = time_bins(frame)
    result = frame.groupby(["dataset", "time_bin"], observed=True).agg(
        decisions=("action", "size"),
        value_mean=("shared_value_mean", "mean"),
        value_std=("shared_value_mean", "std"),
        advantage_mean=("explicit_advantage_mean", "mean"),
        advantage_std=("explicit_advantage_mean", "std"),
        advantage_risk_mean=("legacy_advantage_risk", "mean"),
        stop_probability_mean=("stop_probability", "mean"),
        stop_rate_percent=(
            "action",
            lambda values: 100.0 * float(values.eq("stop").mean()),
        ),
        exploration_rate_percent=(
            "exploration_used",
            lambda values: 100.0 * float(values.astype(bool).mean()),
        ),
        controller_latency_ms=("controller_latency_ms", "mean"),
    ).reset_index()
    return result


def shared_parameter_trajectory(states, feature_names):
    rows = []
    for dataset, state in states.items():
        snapshots = list(state.get("weight_snapshots") or [])
        snapshots.append({
            "decision_count": int(state.get("decision_count", 0)),
            "shared_value_theta": state["shared_value_theta"],
            "shared_advantage_theta": state["shared_advantage_theta"],
            "snapshot": "final",
        })
        for snapshot in snapshots:
            value = snapshot.get("shared_value_theta")
            advantage = snapshot.get("shared_advantage_theta")
            if value is None or advantage is None:
                continue
            for index, feature in enumerate(feature_names):
                rows.append({
                    "dataset": dataset,
                    "snapshot": snapshot.get("snapshot", "periodic"),
                    "decision_count": int(snapshot["decision_count"]),
                    "feature": feature,
                    "value_weight": float(value[index]),
                    "advantage_weight": float(advantage[index]),
                })
    return pd.DataFrame(rows)


def validate_and_report(args):
    output_dir = Path(args.output_dir)
    manifest = json.loads(
        (output_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("method") != METHOD:
        raise RuntimeError(f"unexpected benchmark method: {manifest.get('method')}")
    if set(manifest.get("datasets", [])) != set(args.datasets):
        raise RuntimeError("benchmark did not complete every requested dataset")

    decisions = []
    states = {}
    for dataset in args.datasets:
        phase_dir = output_dir / "raw" / dataset / METHOD
        state = json.loads(
            (phase_dir / "adaptive_td_runtime_state.json").read_text(
                encoding="utf-8"
            )
        )
        if state.get("value_parameterization") != "shared_value_advantage":
            raise RuntimeError(f"{dataset} did not use shared value/advantage")
        if not math.isclose(
            float(state["shared_value_learning_rate"]),
            VALUE_LEARNING_RATE,
        ):
            raise RuntimeError(f"{dataset} value learning rate mismatch")
        if not math.isclose(
            float(state["shared_advantage_learning_rate"]),
            ADVANTAGE_LEARNING_RATE,
        ):
            raise RuntimeError(f"{dataset} advantage learning rate mismatch")
        decision_path = phase_dir / "adaptive_td_decisions.csv"
        if decision_path.exists() and decision_path.stat().st_size:
            frame = pd.read_csv(decision_path)
            frame.insert(0, "dataset", dataset)
            decisions.append(frame)
        states[dataset] = state

    combined_decisions = (
        pd.concat(decisions, ignore_index=True)
        if decisions
        else pd.DataFrame()
    )
    dynamics = (
        shared_learning_dynamics(combined_decisions)
        if not combined_decisions.empty
        else pd.DataFrame()
    )
    feature_names = manifest["feature_names"]
    trajectory = shared_parameter_trajectory(states, feature_names)
    dynamics.to_csv(
        output_dir / "shared_value_advantage_learning_dynamics.csv",
        index=False,
    )
    trajectory.to_csv(
        output_dir / "shared_value_advantage_parameter_trajectory.csv",
        index=False,
    )
    summary = pd.read_csv(output_dir / "dataset_method_summary.csv")
    total_tokens = float(summary.output_tokens.sum())
    total_time = float(summary.algorithm_time_s.sum())
    pooled = pd.DataFrame([{
        "datasets": len(summary),
        "num_questions": int(summary.num_questions.sum()),
        "output_tokens": total_tokens,
        "algorithm_time_s": total_time,
        "pooled_ms_per_output_token": 1000.0 * total_time / total_tokens,
        "draft_time_s": float(summary.draft_time_s.sum()),
        "verify_time_s": float(summary.verify_time_s.sum()),
        "post_verify_time_s": float(summary.post_verify_time_s.sum()),
        "device_transfer_time_s": float(
            summary.device_transfer_time_s.sum()
        ),
        "e2e_ms_per_output_token_excluding_transfer": (
            float(
                (summary.e2e_ms_per_output_token_excluding_transfer
                 * summary.output_tokens).sum()
            )
            / max(1.0, float(summary.output_tokens.sum()))
        ),
        "draft_passes": int(summary.draft_passes.sum()),
        "verifier_rounds": int(summary.verifier_rounds.sum()),
        "controller_overhead_ms": float(summary.controller_overhead_ms.sum()),
    }])
    pooled.to_csv(
        output_dir / "shared_value_advantage_pooled_summary.csv",
        index=False,
    )
    return summary, dynamics, pooled


def _prefixed_summary(summary, prefix):
    summary = summary.copy()
    optional_defaults = {
        "device_transfer_time_s": 0.0,
    }
    for column, default in optional_defaults.items():
        if column not in summary.columns:
            summary[column] = default
    keep = [
        "dataset",
        "num_questions",
        "output_tokens",
        "algorithm_time_s",
        "ms_per_output_token",
        "draft_time_s",
        "verify_time_s",
        "post_verify_time_s",
        "device_transfer_time_s",
        "draft_passes_per_100_tokens",
        "verifier_rounds_per_100_tokens",
        "acceptance_rate_percent",
        "accuracy_percent",
    ]
    missing = set(keep).difference(summary.columns)
    if missing:
        raise ValueError(f"{prefix} summary is missing: {sorted(missing)}")
    return summary[keep].rename(
        columns={
            column: f"{prefix}_{column}"
            for column in keep
            if column != "dataset"
        }
    )


def build_integrated_comparison(args, shared_summary):
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "matched_failfast_baseline"
    baseline_summary = pd.read_csv(baseline_dir / "dataset_method_summary.csv")
    comparison = _prefixed_summary(
        shared_summary,
        "shared_va",
    ).merge(
        _prefixed_summary(baseline_summary, "failfast"),
        on="dataset",
        how="inner",
        validate="one_to_one",
    )
    if set(comparison.dataset) != set(args.datasets):
        raise RuntimeError("integrated summary did not match every requested dataset")
    comparison["shared_va_speedup_vs_failfast"] = (
        comparison["failfast_ms_per_output_token"]
        / comparison["shared_va_ms_per_output_token"]
    )
    comparison["shared_va_latency_reduction_percent"] = 100.0 * (
        1.0
        - comparison["shared_va_ms_per_output_token"]
        / comparison["failfast_ms_per_output_token"]
    )

    paired_frames = []
    for dataset in args.datasets:
        shared_path = output_dir / "raw" / dataset / METHOD / "benchmark_results.csv"
        baseline_path = (
            baseline_dir
            / "raw"
            / dataset
            / "failfast_matched"
            / "benchmark_results.csv"
        )
        shared = pd.read_csv(shared_path)
        baseline = pd.read_csv(baseline_path)
        for frame in (shared, baseline):
            if "device_transfer_time" not in frame.columns:
                frame["device_transfer_time"] = 0.0
        columns = [
            "problem_id",
            "actual_algorithm_time",
            "actual_draft_time",
            "actual_verify_time",
            "actual_post_verify_time",
            "device_transfer_time",
            "output_tokens",
            "accepted_tokens",
            "drafted_tokens",
            "num_speculation_rounds",
            "total_num_forward_passes",
            "acceptance_rate_percent",
            "is_correct",
            "output_token_hash",
        ]
        for label, frame in (("shared_va", shared), ("failfast", baseline)):
            missing = set(columns).difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{dataset} {label} results are missing: {sorted(missing)}"
                )
        paired = shared[columns].rename(
            columns={
                column: f"shared_va_{column}"
                for column in columns
                if column != "problem_id"
            }
        ).merge(
            baseline[columns].rename(
                columns={
                    column: f"failfast_{column}"
                    for column in columns
                    if column != "problem_id"
                }
            ),
            on="problem_id",
            how="inner",
            validate="one_to_one",
        )
        expected_ids = set(PROBLEM_IDS[dataset][: args.num_questions])
        if set(paired.problem_id) != expected_ids:
            raise RuntimeError(f"{dataset} paired report has mismatched problem ids")
        paired.insert(0, "dataset", dataset)
        for prefix in ("shared_va", "failfast"):
            paired[f"{prefix}_ms_per_output_token"] = (
                1000.0
                * paired[f"{prefix}_actual_algorithm_time"]
                / paired[f"{prefix}_output_tokens"].clip(lower=1)
            )
        paired["shared_va_speedup_vs_failfast"] = (
            paired["failfast_ms_per_output_token"]
            / paired["shared_va_ms_per_output_token"]
        )
        paired["shared_va_faster"] = (
            paired["shared_va_ms_per_output_token"]
            < paired["failfast_ms_per_output_token"]
        )
        paired["output_match"] = (
            paired["shared_va_output_token_hash"].astype(str)
            == paired["failfast_output_token_hash"].astype(str)
        )
        paired_frames.append(paired)

    paired = pd.concat(paired_frames, ignore_index=True)
    paired_stats = paired.groupby("dataset", as_index=False).agg(
        paired_questions=("problem_id", "size"),
        shared_va_win_rate_percent=(
            "shared_va_faster",
            lambda values: 100.0 * float(values.mean()),
        ),
        output_match_rate_percent=(
            "output_match",
            lambda values: 100.0 * float(values.mean()),
        ),
        paired_speedup_geomean=(
            "shared_va_speedup_vs_failfast",
            lambda values: float(math.exp(values.map(math.log).mean())),
        ),
    )
    comparison = comparison.merge(
        paired_stats,
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    comparison.to_csv(
        output_dir / "shared_va_vs_failfast_dataset_comparison.csv",
        index=False,
    )
    paired.to_csv(
        output_dir / "shared_va_vs_failfast_paired_comparison.csv",
        index=False,
    )
    return comparison, paired


def build_policy_control_comparison(args, shared_summary):
    output_dir = Path(args.output_dir)
    comparison = _prefixed_summary(shared_summary, "shared_va")
    paired_frames = []
    control_methods = {
        "frozen_stop": f"{METHOD}_frozen_stop",
        "random_stop": f"{METHOD}_random_stop",
    }

    for control, method in control_methods.items():
        control_dir = output_dir / "policy_controls" / control
        control_summary = pd.read_csv(control_dir / "dataset_method_summary.csv")
        comparison = comparison.merge(
            _prefixed_summary(control_summary, control),
            on="dataset",
            how="inner",
            validate="one_to_one",
        )

    for control in control_methods:
        comparison[f"shared_va_speedup_vs_{control}"] = (
            comparison[f"{control}_ms_per_output_token"]
            / comparison["shared_va_ms_per_output_token"]
        )

    for dataset in args.datasets:
        shared_path = output_dir / "raw" / dataset / METHOD / "benchmark_results.csv"
        shared = pd.read_csv(shared_path)
        columns = [
            "problem_id",
            "actual_algorithm_time",
            "actual_draft_time",
            "actual_verify_time",
            "actual_post_verify_time",
            "output_tokens",
            "accepted_tokens",
            "drafted_tokens",
            "num_speculation_rounds",
            "total_num_forward_passes",
            "acceptance_rate_percent",
            "adaptive_decisions",
            "adaptive_stop_actions",
            "is_correct",
            "output_token_hash",
        ]
        missing = set(columns).difference(shared.columns)
        if missing:
            raise ValueError(
                f"{dataset} shared_va results are missing: {sorted(missing)}"
            )
        paired = shared[columns].rename(
            columns={
                column: f"shared_va_{column}"
                for column in columns
                if column != "problem_id"
            }
        )
        for control, method in control_methods.items():
            path = (
                output_dir
                / "policy_controls"
                / control
                / "raw"
                / dataset
                / method
                / "benchmark_results.csv"
            )
            frame = pd.read_csv(path)
            missing = set(columns).difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{dataset} {control} results are missing: {sorted(missing)}"
                )
            paired = paired.merge(
                frame[columns].rename(
                    columns={
                        column: f"{control}_{column}"
                        for column in columns
                        if column != "problem_id"
                    }
                ),
                on="problem_id",
                how="inner",
                validate="one_to_one",
            )
        expected_ids = set(PROBLEM_IDS[dataset][: args.num_questions])
        if set(paired.problem_id) != expected_ids:
            raise RuntimeError(
                f"{dataset} policy-control report has mismatched problem ids"
            )
        paired.insert(0, "dataset", dataset)
        for prefix in ("shared_va", *control_methods):
            paired[f"{prefix}_ms_per_output_token"] = (
                1000.0
                * paired[f"{prefix}_actual_algorithm_time"]
                / paired[f"{prefix}_output_tokens"].clip(lower=1)
            )
        for control in control_methods:
            paired[f"shared_va_speedup_vs_{control}"] = (
                paired[f"{control}_ms_per_output_token"]
                / paired["shared_va_ms_per_output_token"]
            )
            paired[f"shared_va_faster_than_{control}"] = (
                paired["shared_va_ms_per_output_token"]
                < paired[f"{control}_ms_per_output_token"]
            )
            paired[f"shared_va_output_matches_{control}"] = (
                paired["shared_va_output_token_hash"].astype(str)
                == paired[f"{control}_output_token_hash"].astype(str)
            )
        paired_frames.append(paired)

    paired = pd.concat(paired_frames, ignore_index=True)
    statistics = []
    for dataset, frame in paired.groupby("dataset"):
        row = {"dataset": dataset, "paired_questions": len(frame)}
        for control in control_methods:
            speedups = frame[f"shared_va_speedup_vs_{control}"]
            row[f"shared_va_win_rate_vs_{control}_percent"] = 100.0 * float(
                frame[f"shared_va_faster_than_{control}"].mean()
            )
            row[f"shared_va_speedup_vs_{control}_geomean"] = float(
                math.exp(speedups.map(math.log).mean())
            )
            row[f"output_match_vs_{control}_percent"] = 100.0 * float(
                frame[f"shared_va_output_matches_{control}"].mean()
            )
        statistics.append(row)
    comparison = comparison.merge(
        pd.DataFrame(statistics),
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    comparison.to_csv(
        output_dir / "shared_va_vs_policy_controls_dataset_comparison.csv",
        index=False,
    )
    paired.to_csv(
        output_dir / "shared_va_vs_policy_controls_paired_comparison.csv",
        index=False,
    )
    return comparison, paired


def build_all_method_summary(args, shared_summary):
    output_dir = Path(args.output_dir)
    frames = []

    def add(frame, label):
        item = frame.copy()
        item["method"] = label
        frames.append(item)

    add(shared_summary, "shared_va_learned")
    if args.include_failfast_baseline:
        add(
            pd.read_csv(
                output_dir
                / "matched_failfast_baseline"
                / "dataset_method_summary.csv"
            ),
            "failfast_original",
        )
    if args.include_policy_controls:
        for control in ("frozen_stop", "random_stop"):
            add(
                pd.read_csv(
                    output_dir
                    / "policy_controls"
                    / control
                    / "dataset_method_summary.csv"
                ),
                control,
            )

    summary = pd.concat(frames, ignore_index=True)
    learned_latency = shared_summary[["dataset", "ms_per_output_token"]].rename(
        columns={"ms_per_output_token": "shared_va_ms_per_output_token"}
    )
    summary = summary.merge(
        learned_latency,
        on="dataset",
        how="left",
        validate="many_to_one",
    )
    summary["shared_va_speedup_vs_method"] = (
        summary["ms_per_output_token"]
        / summary["shared_va_ms_per_output_token"]
    )
    summary.to_csv(
        output_dir / "all_methods_dataset_summary.csv",
        index=False,
    )
    return summary


def main():
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_streaming(benchmark_command(args))
    summary, dynamics, pooled = validate_and_report(args)

    control_comparison = pd.DataFrame()
    control_paired = pd.DataFrame()
    if args.include_policy_controls:
        for policy_ablation in ("frozen_stop", "random_stop"):
            run_streaming(policy_control_command(args, policy_ablation))
        control_comparison, control_paired = build_policy_control_comparison(
            args,
            summary,
        )

    comparison = pd.DataFrame()
    paired = pd.DataFrame()
    if args.include_failfast_baseline:
        run_streaming(failfast_baseline_command(args))
        comparison, paired = build_integrated_comparison(args, summary)

    all_method_summary = build_all_method_summary(args, summary)

    output_dir = Path(args.output_dir)
    wrapper_manifest = {
        "version": VERSION,
        "method": METHOD,
        "datasets": list(args.datasets),
        "num_questions_per_dataset": args.num_questions,
        "value_learning_rate": VALUE_LEARNING_RATE,
        "advantage_learning_rate": ADVANTAGE_LEARNING_RATE,
        "selected_q_effective_learning_rate": (
            VALUE_LEARNING_RATE + 0.25 * ADVANTAGE_LEARNING_RATE
        ),
        "unselected_q_shared_learning_rate": (
            VALUE_LEARNING_RATE - 0.25 * ADVANTAGE_LEARNING_RATE
        ),
        "uncertainty": "legacy_per_action_covariance",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "arguments": vars(args),
        "executed_policy": (
            "deterministic_greedy"
            if args.greedy_policy
            else "symmetric_sampling"
        ),
        "includes_matched_failfast_baseline": args.include_failfast_baseline,
        "includes_policy_controls": args.include_policy_controls,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "shared_value_advantage_manifest.json").write_text(
        json.dumps(wrapper_manifest, indent=2),
        encoding="utf-8",
    )

    archive_path = None
    if not args.skip_archive:
        archive_path = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    print("\nSHARED VALUE + EXPLICIT ADVANTAGE DATASET SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSHARED LEARNING DYNAMICS", flush=True)
    print(dynamics.to_string(index=False), flush=True)
    print("\nPOOLED SUMMARY", flush=True)
    print(pooled.to_string(index=False), flush=True)
    if not comparison.empty:
        print("\nSHARED V+A VS MATCHED FAILFAST", flush=True)
        print(comparison.to_string(index=False), flush=True)
        print(
            "\nPaired rows: "
            f"{len(paired)}; output match: {100.0 * paired.output_match.mean():.2f}%",
            flush=True,
        )
    if not control_comparison.empty:
        print("\nSHARED V+A VS FROZEN/RANDOM STOP CONTROLS", flush=True)
        print(control_comparison.to_string(index=False), flush=True)
        print(f"\nPolicy-control paired rows: {len(control_paired)}", flush=True)
    print("\nALL METHODS DATASET SUMMARY", flush=True)
    print(all_method_summary.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    if archive_path:
        print(f"Archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
