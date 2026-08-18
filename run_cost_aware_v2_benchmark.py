import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATASETS = ("math", "aime", "gsm8k", "gpqa", "humaneval")
DATASET_LIMITS = {"aime": 30}
BENCHMARK_VERSION = "conditional_hazard_cost_aware_v2_v1"
METHODS = {
    "failfast": {
        "frontier_mode": "disabled",
        "spec_len": 10,
        "lowconf_threshold": 0.45,
        "incr_len": 10,
    },
    "cost_aware_v2_lowconf_0p45": {
        "frontier_mode": "cost_aware_v2",
        "spec_len": 8,
        "lowconf_threshold": 0.45,
        "incr_len": 8,
    },
    "cost_aware_v2_lowconf_0p60": {
        "frontier_mode": "cost_aware_v2",
        "spec_len": 8,
        "lowconf_threshold": 0.60,
        "incr_len": 8,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    candidate_choices = tuple(method for method in METHODS if method != "failfast")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["math", "aime", "gsm8k", "humaneval"])
    parser.add_argument("--candidate", choices=candidate_choices)
    parser.add_argument("--candidates", nargs="+", choices=candidate_choices)
    parser.add_argument("--num_questions", type=int, default=10)
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
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--frontier_v2_hysteresis", type=float, default=0.03)
    parser.add_argument("--frontier_v2_extension_cost_margin", type=float, default=-0.03)
    parser.add_argument("--frontier_v2_hazard_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_extension_prior_strength", type=float, default=2.0)
    parser.add_argument("--frontier_v2_min_hazard_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_min_calibration_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.candidate and args.candidates:
        parser.error("use either --candidate or --candidates, not both")
    args.candidates = args.candidates or (
        [args.candidate]
        if args.candidate
        else ["cost_aware_v2_lowconf_0p45", "cost_aware_v2_lowconf_0p60"]
    )
    args.candidate = None
    if args.output_dir is None:
        candidate_label = "_vs_".join(args.candidates)
        args.output_dir = f"/content/failfasttesting/outputs_{candidate_label}_test{args.num_questions}"
    return args


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1 for latency calibration")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not 0 <= args.frontier_v2_hysteresis < 1:
        raise ValueError("--frontier_v2_hysteresis must be in [0, 1)")
    if not -0.99 < args.frontier_v2_extension_cost_margin <= 1.0:
        raise ValueError("--frontier_v2_extension_cost_margin must be in (-0.99, 1.0]")
    if args.frontier_v2_hazard_prior_strength <= 0:
        raise ValueError("--frontier_v2_hazard_prior_strength must be positive")
    if args.frontier_v2_extension_prior_strength <= 0:
        raise ValueError("--frontier_v2_extension_prior_strength must be positive")
    if args.frontier_v2_min_hazard_observations <= 0:
        raise ValueError("--frontier_v2_min_hazard_observations must be positive")
    if args.frontier_v2_min_calibration_tokens <= 0:
        raise ValueError("--frontier_v2_min_calibration_tokens must be positive")
    if len(args.candidates) != len(set(args.candidates)):
        raise ValueError("--candidates must not contain duplicates")


def measured_questions(args, dataset):
    return min(args.num_questions, DATASET_LIMITS.get(dataset, args.num_questions))


def run_metadata(args, dataset, method):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "method": method,
        "num_questions": measured_questions(args, dataset),
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
        "frontier_cost_token_equiv": args.frontier_cost_token_equiv,
        "frontier_v2_hysteresis": args.frontier_v2_hysteresis,
        "frontier_v2_extension_cost_margin": args.frontier_v2_extension_cost_margin,
        "frontier_v2_hazard_prior_strength": args.frontier_v2_hazard_prior_strength,
        "frontier_v2_extension_prior_strength": args.frontier_v2_extension_prior_strength,
        "frontier_v2_min_hazard_observations": args.frontier_v2_min_hazard_observations,
        "frontier_v2_min_calibration_tokens": args.frontier_v2_min_calibration_tokens,
        "seed": args.seed,
        "method_config": METHODS[method],
    }


def results_complete(result_path, metadata_path, expected_rows, expected_metadata):
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        rows = pd.read_csv(result_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    return (
        metadata == expected_metadata
        and len(rows) == expected_rows
        and rows["problem_id"].nunique() == expected_rows
    )


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
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_method(args, dataset, method):
    config = METHODS[method]
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    expected_rows = measured_questions(args, dataset)
    expected_metadata = run_metadata(args, dataset, method)

    if args.resume and results_complete(
        result_path,
        metadata_path,
        expected_rows,
        expected_metadata,
    ):
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
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(expected_rows),
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
            "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
            "--frontier_v2_hysteresis", str(args.frontier_v2_hysteresis),
            "--frontier_v2_extension_cost_margin", str(args.frontier_v2_extension_cost_margin),
            "--frontier_v2_hazard_prior_strength", str(args.frontier_v2_hazard_prior_strength),
            "--frontier_v2_extension_prior_strength", str(args.frontier_v2_extension_prior_strength),
            "--frontier_v2_min_hazard_observations", str(args.frontier_v2_min_hazard_observations),
            "--frontier_v2_min_calibration_tokens", str(args.frontier_v2_min_calibration_tokens),
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
            f"RUN {dataset} | {method} | samples={expected_rows} | "
            f"spec_len={config['spec_len']} | lowconf={config['lowconf_threshold']}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(expected_metadata, handle, indent=2)

    rows = pd.read_csv(result_path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {result_path}, found {len(rows)}")
    rows["dataset"] = dataset
    rows["method"] = method
    rows["actual_measured_time"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["actual_measured_ms_per_output_token"] = (
        1000.0 * rows["actual_measured_time"] / pd.to_numeric(rows["output_tokens"], errors="coerce")
    )
    return rows


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def aggregate_method(group):
    def numeric_sum(column):
        if column not in group:
            return 0.0
        return pd.to_numeric(group[column], errors="coerce").sum()

    output_tokens = pd.to_numeric(group["output_tokens"], errors="coerce").sum()
    draft_time = numeric_sum("actual_draft_time")
    verify_time = numeric_sum("actual_verify_time")
    controller_time = numeric_sum("actual_post_verify_time")
    total_time = draft_time + verify_time + controller_time
    theoretical_time = numeric_sum("theo_total_time")
    drafted_tokens = pd.to_numeric(group["drafted_tokens"], errors="coerce").sum()
    accepted_tokens = pd.to_numeric(group["accepted_tokens"], errors="coerce").sum()
    rounds = pd.to_numeric(group["num_speculation_rounds"], errors="coerce").sum()
    passes = pd.to_numeric(group["total_num_forward_passes"], errors="coerce").sum()
    extend_actions = pd.to_numeric(group["frontier_v2_extend_actions"], errors="coerce").sum()
    verify_actions = pd.to_numeric(group["frontier_v2_verify_actions"], errors="coerce").sum()
    refinement_stops = numeric_sum("frontier_v2_refinement_stop_actions")
    extension_stops = numeric_sum("frontier_v2_extension_stop_actions")
    fallback_steps = numeric_sum("frontier_v2_fallback_steps")
    hazard_ready_steps = numeric_sum("frontier_v2_hazard_ready_steps")
    extension_ready_steps = numeric_sum("frontier_v2_extension_history_ready_steps")
    fill_passes = pd.to_numeric(group["frontier_fill_forward_passes"], errors="coerce").sum()
    denoising_passes = pd.to_numeric(group["frontier_denoising_forward_passes"], errors="coerce").sum()
    return {
        "num_samples": len(group),
        "output_tokens": output_tokens,
        "actual_measured_time_s": total_time,
        "actual_draft_time_s": draft_time,
        "actual_verify_time_s": verify_time,
        "actual_controller_time_s": controller_time,
        "actual_measured_ms_per_output_token": safe_ratio(1000.0 * total_time, output_tokens),
        "output_tokens_per_second": safe_ratio(output_tokens, total_time),
        "actual_draft_ms_per_output_token": safe_ratio(1000.0 * draft_time, output_tokens),
        "actual_verify_ms_per_output_token": safe_ratio(1000.0 * verify_time, output_tokens),
        "actual_controller_ms_per_output_token": safe_ratio(1000.0 * controller_time, output_tokens),
        "theoretical_time_ms": theoretical_time,
        "theoretical_ms_per_output_token": safe_ratio(theoretical_time, output_tokens),
        "acceptance_rate_percent": safe_ratio(100.0 * accepted_tokens, drafted_tokens),
        "drafted_tokens_per_round": safe_ratio(drafted_tokens, rounds),
        "accepted_tokens_per_round": safe_ratio(accepted_tokens, rounds),
        "output_tokens_per_round": safe_ratio(output_tokens, rounds),
        "draft_forward_passes_per_100_output_tokens": safe_ratio(100.0 * passes, output_tokens),
        "verifier_rounds_per_100_output_tokens": safe_ratio(100.0 * rounds, output_tokens),
        "frontier_v2_extend_actions_per_100_rounds": safe_ratio(100.0 * extend_actions, rounds),
        "frontier_v2_verify_actions_per_100_rounds": safe_ratio(100.0 * verify_actions, rounds),
        "frontier_v2_refinement_stops_per_100_rounds": safe_ratio(100.0 * refinement_stops, rounds),
        "frontier_v2_extension_stops_per_100_rounds": safe_ratio(100.0 * extension_stops, rounds),
        "frontier_v2_fallback_steps": fallback_steps,
        "frontier_v2_hazard_ready_steps": hazard_ready_steps,
        "frontier_v2_extension_history_ready_steps": extension_ready_steps,
        "frontier_v2_predicted_extension_gain_mean": pd.to_numeric(
            group.get("frontier_v2_predicted_extension_gain_mean"), errors="coerce"
        ).replace(0.0, math.nan).mean(),
        "frontier_fill_passes_per_100_output_tokens": safe_ratio(100.0 * fill_passes, output_tokens),
        "frontier_denoising_passes_per_100_output_tokens": safe_ratio(100.0 * denoising_passes, output_tokens),
        "frontier_expected_output_mean": pd.to_numeric(group["frontier_expected_output_mean"], errors="coerce").mean(),
        "parsed_accuracy_percent": 100.0 * pd.to_numeric(group["is_correct"], errors="coerce").mean(),
    }


def build_dataset_summary(rows):
    records = []
    for (dataset, method), group in rows.groupby(["dataset", "method"], sort=False):
        record = {"dataset": dataset, "method": method}
        record.update(aggregate_method(group))
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "method"])


def build_paired_observations(rows, candidate_method):
    columns = [
        "dataset",
        "problem_id",
        "output_tokens",
        "actual_measured_time",
        "actual_measured_ms_per_output_token",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
        "accepted_tokens",
        "drafted_tokens",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "frontier_v2_extend_actions",
        "frontier_v2_verify_actions",
        "frontier_v2_refinement_stop_actions",
        "frontier_v2_extension_stop_actions",
        "frontier_v2_fallback_steps",
        "frontier_v2_hazard_ready_steps",
        "frontier_v2_extension_history_ready_steps",
        "frontier_v2_predicted_extension_gain_mean",
        "frontier_fill_forward_passes",
        "frontier_denoising_forward_passes",
        "frontier_expected_output_mean",
        "output_token_hash",
        "is_correct",
    ]
    baseline = rows[rows["method"] == "failfast"][columns].copy()
    candidate = rows[rows["method"] == candidate_method][columns].copy()
    baseline = baseline.rename(columns={column: f"failfast_{column}" for column in columns if column not in ("dataset", "problem_id")})
    candidate = candidate.rename(columns={column: f"candidate_{column}" for column in columns if column not in ("dataset", "problem_id")})
    paired = baseline.merge(candidate, on=["dataset", "problem_id"], how="inner", validate="one_to_one")
    paired["candidate_speedup_vs_failfast"] = (
        paired["failfast_actual_measured_ms_per_output_token"]
        / paired["candidate_actual_measured_ms_per_output_token"]
    )
    paired["candidate_ms_per_output_token_delta"] = (
        paired["candidate_actual_measured_ms_per_output_token"]
        - paired["failfast_actual_measured_ms_per_output_token"]
    )
    paired["verifier_round_delta"] = paired["candidate_num_speculation_rounds"] - paired["failfast_num_speculation_rounds"]
    paired["draft_forward_pass_delta"] = paired["candidate_total_num_forward_passes"] - paired["failfast_total_num_forward_passes"]
    paired["fill_forward_pass_delta"] = paired["candidate_frontier_fill_forward_passes"] - paired["failfast_frontier_fill_forward_passes"]
    paired["denoising_forward_pass_delta"] = paired["candidate_frontier_denoising_forward_passes"] - paired["failfast_frontier_denoising_forward_passes"]
    paired["output_length_delta"] = paired["candidate_output_tokens"] - paired["failfast_output_tokens"]
    paired["output_matches_failfast"] = paired["candidate_output_token_hash"] == paired["failfast_output_token_hash"]
    paired["candidate_method"] = candidate_method
    return paired


def build_comparison_summary(dataset_summary, paired, candidate_method):
    records = []
    for dataset in dataset_summary["dataset"].unique():
        baseline = dataset_summary[(dataset_summary["dataset"] == dataset) & (dataset_summary["method"] == "failfast")].iloc[0]
        candidate = dataset_summary[(dataset_summary["dataset"] == dataset) & (dataset_summary["method"] == candidate_method)].iloc[0]
        observations = paired[paired["dataset"] == dataset]
        records.append({
            "dataset": dataset,
            "candidate_method": candidate_method,
            "num_samples": len(observations),
            "candidate_speedup_vs_failfast": safe_ratio(
                baseline["actual_measured_ms_per_output_token"],
                candidate["actual_measured_ms_per_output_token"],
            ),
            "candidate_theoretical_speedup_vs_failfast": safe_ratio(
                baseline["theoretical_ms_per_output_token"],
                candidate["theoretical_ms_per_output_token"],
            ),
            "candidate_win_rate_percent": 100.0 * (observations["candidate_ms_per_output_token_delta"] < 0).mean(),
            "output_match_rate_percent": 100.0 * observations["output_matches_failfast"].mean(),
            "failfast_ms_per_output_token": baseline["actual_measured_ms_per_output_token"],
            "candidate_ms_per_output_token": candidate["actual_measured_ms_per_output_token"],
            "failfast_output_tokens_per_second": baseline["output_tokens_per_second"],
            "candidate_output_tokens_per_second": candidate["output_tokens_per_second"],
            "failfast_draft_ms_per_output_token": baseline["actual_draft_ms_per_output_token"],
            "candidate_draft_ms_per_output_token": candidate["actual_draft_ms_per_output_token"],
            "failfast_verify_ms_per_output_token": baseline["actual_verify_ms_per_output_token"],
            "candidate_verify_ms_per_output_token": candidate["actual_verify_ms_per_output_token"],
            "failfast_controller_ms_per_output_token": baseline["actual_controller_ms_per_output_token"],
            "candidate_controller_ms_per_output_token": candidate["actual_controller_ms_per_output_token"],
            "failfast_verifier_rounds_per_100_tokens": baseline["verifier_rounds_per_100_output_tokens"],
            "candidate_verifier_rounds_per_100_tokens": candidate["verifier_rounds_per_100_output_tokens"],
            "failfast_output_tokens_per_round": baseline["output_tokens_per_round"],
            "candidate_output_tokens_per_round": candidate["output_tokens_per_round"],
            "failfast_acceptance_rate_percent": baseline["acceptance_rate_percent"],
            "candidate_acceptance_rate_percent": candidate["acceptance_rate_percent"],
            "failfast_draft_passes_per_100_tokens": baseline["draft_forward_passes_per_100_output_tokens"],
            "candidate_draft_passes_per_100_tokens": candidate["draft_forward_passes_per_100_output_tokens"],
            "candidate_v2_extend_actions_per_100_rounds": candidate["frontier_v2_extend_actions_per_100_rounds"],
            "candidate_v2_refinement_stops_per_100_rounds": candidate["frontier_v2_refinement_stops_per_100_rounds"],
            "candidate_v2_extension_stops_per_100_rounds": candidate["frontier_v2_extension_stops_per_100_rounds"],
            "failfast_accuracy_percent": baseline["parsed_accuracy_percent"],
            "candidate_accuracy_percent": candidate["parsed_accuracy_percent"],
        })
    summary = pd.DataFrame(records)
    failfast = dataset_summary[dataset_summary["method"] == "failfast"]
    candidate = dataset_summary[dataset_summary["method"] == candidate_method]
    failfast_pooled_ms = safe_ratio(
        1000.0 * failfast["actual_measured_time_s"].sum(),
        failfast["output_tokens"].sum(),
    )
    candidate_pooled_ms = safe_ratio(
        1000.0 * candidate["actual_measured_time_s"].sum(),
        candidate["output_tokens"].sum(),
    )
    overall = pd.DataFrame([{
        "candidate_method": candidate_method,
        "datasets_completed": summary["dataset"].nunique(),
        "num_samples": len(paired),
        "macro_speedup_candidate_vs_failfast": summary["candidate_speedup_vs_failfast"].mean(),
        "pooled_speedup_candidate_vs_failfast": safe_ratio(failfast_pooled_ms, candidate_pooled_ms),
        "paired_candidate_win_rate_percent": 100.0 * (paired["candidate_ms_per_output_token_delta"] < 0).mean(),
        "output_match_rate_percent": 100.0 * paired["output_matches_failfast"].mean(),
        "failfast_pooled_ms_per_output_token": failfast_pooled_ms,
        "candidate_pooled_ms_per_output_token": candidate_pooled_ms,
    }])
    return summary, overall


def build_threshold_comparison(rows):
    low = rows[rows["method"] == "cost_aware_v2_lowconf_0p45"].copy()
    high = rows[rows["method"] == "cost_aware_v2_lowconf_0p60"].copy()
    columns = [
        "dataset",
        "problem_id",
        "actual_measured_time",
        "actual_measured_ms_per_output_token",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
        "output_tokens",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "accepted_tokens",
        "drafted_tokens",
        "output_token_hash",
    ]
    low = low[columns].rename(columns={
        column: f"lowconf_0p45_{column}"
        for column in columns
        if column not in ("dataset", "problem_id")
    })
    high = high[columns].rename(columns={
        column: f"lowconf_0p60_{column}"
        for column in columns
        if column not in ("dataset", "problem_id")
    })
    paired = low.merge(high, on=["dataset", "problem_id"], validate="one_to_one")
    paired["speedup_0p60_vs_0p45"] = (
        paired["lowconf_0p45_actual_measured_ms_per_output_token"]
        / paired["lowconf_0p60_actual_measured_ms_per_output_token"]
    )
    paired["lowconf_0p60_faster"] = (
        paired["lowconf_0p60_actual_measured_ms_per_output_token"]
        < paired["lowconf_0p45_actual_measured_ms_per_output_token"]
    )
    paired["output_matches"] = (
        paired["lowconf_0p45_output_token_hash"]
        == paired["lowconf_0p60_output_token_hash"]
    )
    records = []
    for dataset, group in paired.groupby("dataset", sort=False):
        low_time = pd.to_numeric(group["lowconf_0p45_actual_measured_time"], errors="coerce").sum()
        high_time = pd.to_numeric(group["lowconf_0p60_actual_measured_time"], errors="coerce").sum()
        low_tokens = pd.to_numeric(group["lowconf_0p45_output_tokens"], errors="coerce").sum()
        high_tokens = pd.to_numeric(group["lowconf_0p60_output_tokens"], errors="coerce").sum()
        records.append({
            "dataset": dataset,
            "num_samples": len(group),
            "pooled_speedup_0p60_vs_0p45": safe_ratio(
                safe_ratio(1000.0 * low_time, low_tokens),
                safe_ratio(1000.0 * high_time, high_tokens),
            ),
            "paired_0p60_win_rate_percent": 100.0 * group["lowconf_0p60_faster"].mean(),
            "output_match_rate_percent": 100.0 * group["output_matches"].mean(),
        })
    return paired, pd.DataFrame(records)


def collect_frontier_diagnostics(output_dir, datasets, methods, filename):
    frames = []
    for dataset in datasets:
        for method in methods:
            path = output_dir / "raw" / dataset / method / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["dataset"] = dataset
            frame["method"] = method
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_manifest(args, output_dir):
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
        "methods": {method: METHODS[method] for method in ("failfast", *args.candidates)},
        "primary_metric": "measured milliseconds per output token",
        "time_formula": "actual_draft_time + actual_verify_time + actual_post_verify_time",
        "comparison": f"{', '.join(args.candidates)} versus FailFast",
        "v2_controller": "conditional token hazards plus extension-offset hazards",
        "v2_latency_model": "measured EMA lookup by context-length and proposal-length bins",
        "v2_fallback": "FailFast until online hazard history is sufficient",
        "target_decoding": "greedy",
        "macro_speedup": "arithmetic mean of per-dataset candidate speedups versus FailFast",
        "pooled_speedup": "ratio of total FailFast milliseconds/output-token to total candidate milliseconds/output-token",
        "dataset_order_policy": "method order rotates by dataset",
    }
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    methods = ("failfast", *args.candidates)
    for dataset_index, dataset in enumerate(args.datasets):
        rotation = dataset_index % len(methods)
        method_order = methods[rotation:] + methods[:rotation]
        for method in method_order:
            frames.append(run_method(args, dataset, method))
    rows = pd.concat(frames, ignore_index=True, sort=False)
    dataset_summary = build_dataset_summary(rows)
    paired_frames = []
    comparison_frames = []
    overall_frames = []
    for candidate in args.candidates:
        candidate_paired = build_paired_observations(rows, candidate)
        candidate_comparison, candidate_overall = build_comparison_summary(
            dataset_summary,
            candidate_paired,
            candidate,
        )
        paired_frames.append(candidate_paired)
        comparison_frames.append(candidate_comparison)
        overall_frames.append(candidate_overall)
    paired = pd.concat(paired_frames, ignore_index=True, sort=False)
    comparison = pd.concat(comparison_frames, ignore_index=True, sort=False)
    overall = pd.concat(overall_frames, ignore_index=True, sort=False)
    threshold_paired, threshold_summary = build_threshold_comparison(rows)
    round_diagnostics = collect_frontier_diagnostics(
        output_dir,
        args.datasets,
        args.candidates,
        "frontier_round_diagnostics.csv",
    )
    extension_diagnostics = collect_frontier_diagnostics(
        output_dir,
        args.datasets,
        args.candidates,
        "frontier_extension_diagnostics.csv",
    )
    gain_diagnostics = collect_frontier_diagnostics(
        output_dir,
        args.datasets,
        args.candidates,
        "frontier_gain_diagnostics.csv",
    )
    rows.to_csv(output_dir / "per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_observations.csv", index=False)
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    threshold_paired.to_csv(output_dir / "paired_v2_threshold_observations.csv", index=False)
    threshold_summary.to_csv(output_dir / "v2_threshold_comparison.csv", index=False)
    round_diagnostics.to_csv(output_dir / "all_frontier_round_diagnostics.csv", index=False)
    extension_diagnostics.to_csv(output_dir / "all_frontier_extension_diagnostics.csv", index=False)
    gain_diagnostics.to_csv(output_dir / "all_frontier_gain_diagnostics.csv", index=False)
    write_manifest(args, output_dir)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nDATASET COMPARISON")
    print(comparison.to_string(index=False))
    print("\nOVERALL COMPARISON")
    print(overall.to_string(index=False))
    print("\nV2 THRESHOLD COMPARISON")
    print(threshold_summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
