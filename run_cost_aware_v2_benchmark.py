import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATASETS = ("math", "aime", "gsm8k", "gpqa", "humaneval")
DATASET_LIMITS = {"aime": 30}
BENCHMARK_VERSION = "paired_frontier_candidate_v2"
METHODS = {
    "failfast": {
        "frontier_mode": "disabled",
        "spec_len": 10,
        "lowconf_threshold": 0.45,
        "incr_len": 10,
    },
    "cost_aware_v2": {
        "frontier_mode": "cost_aware_v2",
        "spec_len": 10,
        "lowconf_threshold": 0.60,
        "incr_len": 4,
    },
    "cost_aware_v1_spec8": {
        "frontier_mode": "cost_aware",
        "spec_len": 8,
        "lowconf_threshold": 0.60,
        "incr_len": 10,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["math", "aime", "gsm8k", "humaneval"])
    parser.add_argument("--candidate", choices=("cost_aware_v2", "cost_aware_v1_spec8"), default="cost_aware_v2")
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
    parser.add_argument("--frontier_v2_min_expected_output", type=float, default=7.0)
    parser.add_argument("--frontier_v2_hysteresis", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"/content/failfasttesting/outputs_{args.candidate}_test{args.num_questions}"
    return args


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1 for latency calibration")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.frontier_v2_min_expected_output <= 1:
        raise ValueError("--frontier_v2_min_expected_output must be greater than 1")
    if not 0 <= args.frontier_v2_hysteresis < 1:
        raise ValueError("--frontier_v2_hysteresis must be in [0, 1)")


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
        "frontier_v2_min_expected_output": args.frontier_v2_min_expected_output,
        "frontier_v2_hysteresis": args.frontier_v2_hysteresis,
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
        if result_path.exists():
            result_path.unlink()
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
            "--frontier_v2_min_expected_output", str(args.frontier_v2_min_expected_output),
            "--frontier_v2_hysteresis", str(args.frontier_v2_hysteresis),
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
    output_tokens = pd.to_numeric(group["output_tokens"], errors="coerce").sum()
    draft_time = pd.to_numeric(group["actual_draft_time"], errors="coerce").sum()
    verify_time = pd.to_numeric(group["actual_verify_time"], errors="coerce").sum()
    controller_time = pd.to_numeric(group["actual_post_verify_time"], errors="coerce").sum()
    total_time = draft_time + verify_time + controller_time
    drafted_tokens = pd.to_numeric(group["drafted_tokens"], errors="coerce").sum()
    accepted_tokens = pd.to_numeric(group["accepted_tokens"], errors="coerce").sum()
    rounds = pd.to_numeric(group["num_speculation_rounds"], errors="coerce").sum()
    passes = pd.to_numeric(group["total_num_forward_passes"], errors="coerce").sum()
    extend_actions = pd.to_numeric(group["frontier_v2_extend_actions"], errors="coerce").sum()
    verify_actions = pd.to_numeric(group["frontier_v2_verify_actions"], errors="coerce").sum()
    fill_passes = pd.to_numeric(group["frontier_fill_forward_passes"], errors="coerce").sum()
    denoising_passes = pd.to_numeric(group["frontier_denoising_forward_passes"], errors="coerce").sum()
    return {
        "num_samples": len(group),
        "output_tokens": output_tokens,
        "actual_measured_time_s": total_time,
        "actual_measured_ms_per_output_token": safe_ratio(1000.0 * total_time, output_tokens),
        "actual_draft_ms_per_output_token": safe_ratio(1000.0 * draft_time, output_tokens),
        "actual_verify_ms_per_output_token": safe_ratio(1000.0 * verify_time, output_tokens),
        "actual_controller_ms_per_output_token": safe_ratio(1000.0 * controller_time, output_tokens),
        "acceptance_rate_percent": safe_ratio(100.0 * accepted_tokens, drafted_tokens),
        "drafted_tokens_per_round": safe_ratio(drafted_tokens, rounds),
        "accepted_tokens_per_round": safe_ratio(accepted_tokens, rounds),
        "output_tokens_per_round": safe_ratio(output_tokens, rounds),
        "draft_forward_passes_per_100_output_tokens": safe_ratio(100.0 * passes, output_tokens),
        "verifier_rounds_per_100_output_tokens": safe_ratio(100.0 * rounds, output_tokens),
        "frontier_v2_extend_actions_per_100_rounds": safe_ratio(100.0 * extend_actions, rounds),
        "frontier_v2_verify_actions_per_100_rounds": safe_ratio(100.0 * verify_actions, rounds),
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
            "candidate_win_rate_percent": 100.0 * (observations["candidate_ms_per_output_token_delta"] < 0).mean(),
            "output_match_rate_percent": 100.0 * observations["output_matches_failfast"].mean(),
            "failfast_ms_per_output_token": baseline["actual_measured_ms_per_output_token"],
            "candidate_ms_per_output_token": candidate["actual_measured_ms_per_output_token"],
            "failfast_verifier_rounds_per_100_tokens": baseline["verifier_rounds_per_100_output_tokens"],
            "candidate_verifier_rounds_per_100_tokens": candidate["verifier_rounds_per_100_output_tokens"],
            "failfast_output_tokens_per_round": baseline["output_tokens_per_round"],
            "candidate_output_tokens_per_round": candidate["output_tokens_per_round"],
            "failfast_acceptance_rate_percent": baseline["acceptance_rate_percent"],
            "candidate_acceptance_rate_percent": candidate["acceptance_rate_percent"],
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
        "methods": {
            "failfast": METHODS["failfast"],
            args.candidate: METHODS[args.candidate],
        },
        "primary_metric": "measured milliseconds per output token",
        "time_formula": "actual_draft_time + actual_verify_time + actual_post_verify_time",
        "comparison": f"{args.candidate} versus FailFast",
        "target_decoding": "greedy",
        "macro_speedup": "arithmetic mean of per-dataset candidate speedups versus FailFast",
        "pooled_speedup": "ratio of total FailFast milliseconds/output-token to total candidate milliseconds/output-token",
        "dataset_order_policy": "method order alternates by dataset",
    }
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset_index, dataset in enumerate(args.datasets):
        method_order = ("failfast", args.candidate) if dataset_index % 2 == 0 else (args.candidate, "failfast")
        for method in method_order:
            frames.append(run_method(args, dataset, method))
    rows = pd.concat(frames, ignore_index=True, sort=False)
    dataset_summary = build_dataset_summary(rows)
    paired = build_paired_observations(rows, args.candidate)
    comparison, overall = build_comparison_summary(dataset_summary, paired, args.candidate)
    rows.to_csv(output_dir / "per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_observations.csv", index=False)
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    write_manifest(args, output_dir)
    print("\nDATASET COMPARISON")
    print(comparison.to_string(index=False))
    print("\nOVERALL COMPARISON")
    print(overall.to_string(index=False))
    print(f"\nSaved report: {output_dir}")


if __name__ == "__main__":
    main()
