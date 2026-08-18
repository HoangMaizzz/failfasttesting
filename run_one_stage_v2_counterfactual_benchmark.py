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

import pandas as pd

from run_cost_aware_v2_benchmark import (
    build_comparison_summary,
    build_dataset_summary,
    build_paired_observations,
    collect_frontier_diagnostics,
    run_streaming,
    safe_ratio,
)


DATASET_SIZES = {
    "math": 500,
    "aime": 30,
    "gsm8k": 1319,
    "humaneval": 164,
}
BENCHMARK_VERSION = "one_stage_v2_counterfactual_v1"
METHODS = {
    "failfast": {
        "frontier_mode": "disabled",
        "spec_len": 10,
        "incr_len": 10,
        "lowconf_threshold": 0.45,
        "counterfactual": False,
    },
    "one_stage_v2": {
        "frontier_mode": "cost_aware_v2",
        "spec_len": 8,
        "incr_len": 8,
        "lowconf_threshold": 0.45,
        "counterfactual": False,
    },
    "one_stage_v2_counterfactual": {
        "frontier_mode": "cost_aware_v2",
        "spec_len": 8,
        "incr_len": 8,
        "lowconf_threshold": 0.45,
        "counterfactual": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SIZES),
        default=list(DATASET_SIZES),
    )
    parser.add_argument("--num_questions", type=int, default=15)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_v2_hysteresis", type=float, default=0.03)
    parser.add_argument("--frontier_v2_extension_cost_margin", type=float, default=0.05)
    parser.add_argument("--frontier_v2_gain_calibration_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_min_gain_calibration_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_gain_ucb_beta", type=float, default=1.0)
    parser.add_argument("--frontier_v2_gain_prior_std", type=float, default=2.0)
    parser.add_argument("--frontier_v2_hazard_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_extension_prior_strength", type=float, default=2.0)
    parser.add_argument("--frontier_v2_min_hazard_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_min_calibration_tokens", type=int, default=64)
    parser.add_argument("--counterfactual_rate", type=float, default=0.5)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_one_stage_v2_counterfactual_test15",
    )
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1")
    if not 0.0 <= args.counterfactual_rate <= 1.0:
        raise ValueError("--counterfactual_rate must be in [0, 1]")
    if not -0.99 < args.frontier_v2_extension_cost_margin <= 1.0:
        raise ValueError("--frontier_v2_extension_cost_margin must be in (-0.99, 1]")
    if args.frontier_v2_gain_calibration_prior_strength <= 0:
        raise ValueError("--frontier_v2_gain_calibration_prior_strength must be positive")
    if args.frontier_v2_min_gain_calibration_observations <= 0:
        raise ValueError("--frontier_v2_min_gain_calibration_observations must be positive")
    if args.frontier_v2_gain_ucb_beta < 0:
        raise ValueError("--frontier_v2_gain_ucb_beta must be non-negative")
    if args.frontier_v2_gain_prior_std <= 0:
        raise ValueError("--frontier_v2_gain_prior_std must be positive")
    for dataset in args.datasets:
        available = DATASET_SIZES[dataset] - args.warmup_questions
        if args.num_questions > available:
            raise ValueError(f"{dataset} has only {available} non-warmup samples")


def sampled_problem_ids(args):
    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        population = list(range(args.warmup_questions, DATASET_SIZES[dataset]))
        rng = random.Random(args.sample_seed + 1009 * dataset_index)
        result[dataset] = sorted(rng.sample(population, args.num_questions))
    return result


def run_metadata(args, dataset, method, problem_ids):
    config = METHODS[method]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "method": method,
        "problem_ids": problem_ids,
        "warmup_questions": args.warmup_questions,
        "max_new_tokens": args.max_new_tokens,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "drafter_threshold": args.drafter_threshold,
        "max_spec_len": args.max_spec_len,
        "frontier_min_steps": args.frontier_min_steps,
        "frontier_patience": args.frontier_patience,
        "frontier_v2_hysteresis": args.frontier_v2_hysteresis,
        "frontier_v2_extension_cost_margin": args.frontier_v2_extension_cost_margin,
        "frontier_v2_gain_calibration_prior_strength": args.frontier_v2_gain_calibration_prior_strength,
        "frontier_v2_min_gain_calibration_observations": args.frontier_v2_min_gain_calibration_observations,
        "frontier_v2_gain_ucb_beta": args.frontier_v2_gain_ucb_beta,
        "frontier_v2_gain_prior_std": args.frontier_v2_gain_prior_std,
        "frontier_v2_hazard_prior_strength": args.frontier_v2_hazard_prior_strength,
        "frontier_v2_extension_prior_strength": args.frontier_v2_extension_prior_strength,
        "frontier_v2_min_hazard_observations": args.frontier_v2_min_hazard_observations,
        "frontier_v2_min_calibration_tokens": args.frontier_v2_min_calibration_tokens,
        "counterfactual_rate": args.counterfactual_rate if config["counterfactual"] else 0.0,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "method_config": config,
    }


def results_complete(result_path, metadata_path, expected_metadata):
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        rows = pd.read_csv(result_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    expected_ids = expected_metadata["problem_ids"]
    return (
        metadata == expected_metadata
        and len(rows) == len(expected_ids)
        and sorted(rows["problem_id"].astype(int).tolist()) == expected_ids
    )


def run_method(args, dataset, method, problem_ids):
    config = METHODS[method]
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata(args, dataset, method, problem_ids)
    if args.resume and results_complete(result_path, metadata_path, metadata):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
            "frontier_v2_runtime_state.json",
            "run_metadata.json",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()
        counterfactual_rate = args.counterfactual_rate if config["counterfactual"] else 0.0
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(len(problem_ids)),
            "--problem_ids", *[str(problem_id) for problem_id in problem_ids],
            "--warmup_questions", str(args.warmup_questions),
            "--benchmark_modes", "dllm_ar",
            "--dllm_variant", "failfast",
            "--decoding_strategy", "greedy",
            "--max_new_tokens", str(args.max_new_tokens),
            "--spec_len", str(config["spec_len"]),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(config["lowconf_threshold"]),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(config["incr_len"]),
            "--frontier_stop_mode", config["frontier_mode"],
            "--frontier_min_steps", str(args.frontier_min_steps),
            "--frontier_patience", str(args.frontier_patience),
            "--frontier_v2_hysteresis", str(args.frontier_v2_hysteresis),
            "--frontier_v2_extension_cost_margin", str(args.frontier_v2_extension_cost_margin),
            "--frontier_v2_gain_calibration_prior_strength", str(args.frontier_v2_gain_calibration_prior_strength),
            "--frontier_v2_min_gain_calibration_observations", str(args.frontier_v2_min_gain_calibration_observations),
            "--frontier_v2_gain_ucb_beta", str(args.frontier_v2_gain_ucb_beta),
            "--frontier_v2_gain_prior_std", str(args.frontier_v2_gain_prior_std),
            "--frontier_v2_hazard_prior_strength", str(args.frontier_v2_hazard_prior_strength),
            "--frontier_v2_extension_prior_strength", str(args.frontier_v2_extension_prior_strength),
            "--frontier_v2_min_hazard_observations", str(args.frontier_v2_min_hazard_observations),
            "--frontier_v2_min_calibration_tokens", str(args.frontier_v2_min_calibration_tokens),
            "--frontier_v2_counterfactual_rate", str(counterfactual_rate),
            "--frontier_v2_counterfactual_seed", str(args.sample_seed),
            "--seed", str(args.seed),
            "--quiet_generation",
            "--disable_progress",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]
        print("\n" + "=" * 100, flush=True)
        print(
            f"RUN {dataset} | {method} | samples={len(problem_ids)} | "
            f"counterfactual_rate={counterfactual_rate}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = pd.read_csv(result_path)
    rows["dataset"] = dataset
    rows["method"] = method
    rows["actual_measured_time"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["actual_measured_ms_per_output_token"] = (
        1000.0 * rows["actual_measured_time"]
        / pd.to_numeric(rows["output_tokens"], errors="coerce")
    )
    return rows


def counterfactual_summary(events, margin):
    events = events[
        events["trigger"] == "cost_aware_v2_counterfactual_extend"
    ].copy()
    if events.empty:
        return events, pd.DataFrame()
    numeric_columns = (
        "actual_extension_accepted_tokens",
        "predicted_extension_gain",
        "decision_expected_output",
        "stop_ms_per_output",
        "predicted_extend_ms_per_output",
        "estimated_extension_total_ms",
        "extension_size",
    )
    for column in numeric_columns:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events["actual_extend_ms_per_output"] = (
        events["estimated_extension_total_ms"]
        / (
            events["decision_expected_output"]
            + events["actual_extension_accepted_tokens"]
        ).clip(lower=1e-6)
    )
    events["strictly_beneficial"] = (
        events["actual_extend_ms_per_output"] < events["stop_ms_per_output"]
    )
    events["beneficial_under_policy_margin"] = (
        events["actual_extend_ms_per_output"]
        <= events["stop_ms_per_output"] * (1.0 + margin)
    )
    records = []
    for dataset, group in events.groupby("dataset", sort=False):
        records.append({
            "dataset": dataset,
            "num_counterfactual_events": len(group),
            "num_problems": group["problem_id"].nunique(),
            "mean_predicted_gain": group["predicted_extension_gain"].mean(),
            "mean_actual_gain": group["actual_extension_accepted_tokens"].mean(),
            "positive_gain_rate_percent": 100.0 * (group["actual_extension_accepted_tokens"] > 0).mean(),
            "full_extension_accept_rate_percent": 100.0 * (group["actual_extension_accepted_tokens"] == group["extension_size"]).mean(),
            "strictly_beneficial_rate_percent": 100.0 * group["strictly_beneficial"].mean(),
            "beneficial_under_policy_margin_rate_percent": 100.0 * group["beneficial_under_policy_margin"].mean(),
            "mean_actual_to_stop_cost_ratio": (
                group["actual_extend_ms_per_output"] / group["stop_ms_per_output"]
            ).mean(),
        })
    overall = events.copy()
    records.append({
        "dataset": "overall",
        "num_counterfactual_events": len(overall),
        "num_problems": overall[["dataset", "problem_id"]].drop_duplicates().shape[0],
        "mean_predicted_gain": overall["predicted_extension_gain"].mean(),
        "mean_actual_gain": overall["actual_extension_accepted_tokens"].mean(),
        "positive_gain_rate_percent": 100.0 * (overall["actual_extension_accepted_tokens"] > 0).mean(),
        "full_extension_accept_rate_percent": 100.0 * (overall["actual_extension_accepted_tokens"] == overall["extension_size"]).mean(),
        "strictly_beneficial_rate_percent": 100.0 * overall["strictly_beneficial"].mean(),
        "beneficial_under_policy_margin_rate_percent": 100.0 * overall["beneficial_under_policy_margin"].mean(),
        "mean_actual_to_stop_cost_ratio": (
            overall["actual_extend_ms_per_output"] / overall["stop_ms_per_output"]
        ).mean(),
    })
    return events, pd.DataFrame(records)


def missed_extension_summary(rounds):
    rounds = rounds[rounds["method"] == "one_stage_v2"].copy()
    rounds["eligible_full_accept"] = (
        (pd.to_numeric(rounds["full_accept"], errors="coerce") == 1)
        & (pd.to_numeric(rounds["extension_capacity"], errors="coerce") == 1)
    )
    rounds["missed_extension"] = (
        rounds["eligible_full_accept"]
        & (pd.to_numeric(rounds["extension_count"], errors="coerce") == 0)
    )
    records = []
    for dataset, group in rounds.groupby("dataset", sort=False):
        eligible = int(group["eligible_full_accept"].sum())
        missed = int(group["missed_extension"].sum())
        records.append({
            "dataset": dataset,
            "eligible_full_accept_rounds": eligible,
            "missed_extension_rounds": missed,
            "missed_extension_rate_percent": 100.0 * safe_ratio(missed, eligible),
            "missed_extension_rate_all_rounds_percent": 100.0 * safe_ratio(missed, len(group)),
        })
    return pd.DataFrame(records)


def write_manifest(args, output_dir, problem_ids):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "problem_ids": problem_ids,
        "methods": METHODS,
        "primary_metric": "actual draft + verify + post-verify controller time per output token",
        "counterfactual_policy": "randomly force a rejected extension decision without changing its logged original action",
        "counterfactual_limit": "on-policy exploration changes later decoding state; event outcomes are locally randomized, not full-trajectory paired",
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = sampled_problem_ids(args)
    frames = []
    methods = tuple(METHODS)
    for dataset_index, dataset in enumerate(args.datasets):
        rotation = dataset_index % len(methods)
        method_order = methods[rotation:] + methods[:rotation]
        for method in method_order:
            frames.append(run_method(args, dataset, method, problem_ids[dataset]))
    rows = pd.concat(frames, ignore_index=True, sort=False)
    dataset_summary = build_dataset_summary(rows)
    paired_frames = []
    comparison_frames = []
    overall_frames = []
    for candidate in ("one_stage_v2", "one_stage_v2_counterfactual"):
        paired = build_paired_observations(rows, candidate)
        comparison, overall = build_comparison_summary(
            dataset_summary, paired, candidate
        )
        paired_frames.append(paired)
        comparison_frames.append(comparison)
        overall_frames.append(overall)
    paired = pd.concat(paired_frames, ignore_index=True, sort=False)
    comparison = pd.concat(comparison_frames, ignore_index=True, sort=False)
    overall = pd.concat(overall_frames, ignore_index=True, sort=False)
    round_diagnostics = collect_frontier_diagnostics(
        output_dir, args.datasets, methods[1:], "frontier_round_diagnostics.csv"
    )
    extension_diagnostics = collect_frontier_diagnostics(
        output_dir, args.datasets, methods[1:], "frontier_extension_diagnostics.csv"
    )
    counterfactual_events, counterfactual = counterfactual_summary(
        extension_diagnostics, args.frontier_v2_extension_cost_margin
    )
    missed = missed_extension_summary(round_diagnostics)
    rows.to_csv(output_dir / "per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_observations.csv", index=False)
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    round_diagnostics.to_csv(output_dir / "all_frontier_round_diagnostics.csv", index=False)
    extension_diagnostics.to_csv(output_dir / "all_frontier_extension_diagnostics.csv", index=False)
    counterfactual_events.to_csv(output_dir / "counterfactual_events.csv", index=False)
    counterfactual.to_csv(output_dir / "counterfactual_summary.csv", index=False)
    missed.to_csv(output_dir / "missed_extension_summary.csv", index=False)
    write_manifest(args, output_dir, problem_ids)
    archive_path = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print("\nDATASET COMPARISON")
    print(comparison.to_string(index=False))
    print("\nOVERALL COMPARISON")
    print(overall.to_string(index=False))
    print("\nCOUNTERFACTUAL SUMMARY")
    print(counterfactual.to_string(index=False))
    print("\nMISSED EXTENSION SUMMARY")
    print(missed.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
