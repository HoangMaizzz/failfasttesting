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
METHOD_ORDER = (
    "ar_only",
    "ar_draft",
    "fast_dllm",
    "failfast",
    "cost_aware_no_extend",
    "cost_aware_extend",
)
LOCAL_METHODS = {
    "ar_only": {
        "mode": "verifier_ar",
        "spec_len": 1,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "ar_draft": {
        "mode": "ar_ar",
        "spec_len": 10,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "fast_dllm": {
        "mode": "dllm_ar",
        "spec_len": 10,
        "dllm_variant": "fixed",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "failfast": {
        "mode": "dllm_ar",
        "spec_len": 10,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "cost_aware_no_extend": {
        "mode": "dllm_ar",
        "spec_len": 5,
        "dllm_variant": "failfast",
        "frontier_mode": "cost_aware_no_extend",
        "lowconf_threshold": 0.45,
    },
    "cost_aware_extend": {
        "mode": "dllm_ar",
        "spec_len": 5,
        "dllm_variant": "failfast",
        "frontier_mode": "cost_aware",
        "lowconf_threshold": 0.60,
    },
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--drafter_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--incr_len", type=int, default=10)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/content/failfasttesting/outputs_actual_latency")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 0:
        raise ValueError("--warmup_questions must be non-negative")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")


def run_streaming(command, cwd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
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


def local_output_dir(args, dataset, method):
    return Path(args.output_dir) / "raw" / dataset / method


def measured_questions(args, dataset):
    return min(args.num_questions, DATASET_LIMITS.get(dataset, args.num_questions))


def local_results_complete(path, expected_rows):
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    return len(rows) == expected_rows and rows["problem_id"].nunique() == expected_rows


def run_local_method(args, dataset, method):
    config = LOCAL_METHODS[method]
    output_dir = local_output_dir(args, dataset, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_results.csv"
    expected_rows = measured_questions(args, dataset)
    if args.resume and local_results_complete(benchmark_path, expected_rows):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        if benchmark_path.exists():
            benchmark_path.unlink()
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(expected_rows),
            "--warmup_questions", str(args.warmup_questions),
            "--benchmark_modes", config["mode"],
            "--decoding_strategy", "greedy",
            "--max_new_tokens", str(args.max_new_tokens),
            "--spec_len", str(config["spec_len"]),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--drafter_model_name", args.drafter_model_name,
            "--dllm_dir", args.dllm_dir,
            "--dllm_variant", config["dllm_variant"],
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(config["lowconf_threshold"]),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(args.incr_len),
            "--frontier_stop_mode", config["frontier_mode"],
            "--frontier_min_steps", str(args.frontier_min_steps),
            "--frontier_patience", str(args.frontier_patience),
            "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
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
        print(f"RUN {dataset} | {method} | measured={expected_rows} | max_tokens={args.max_new_tokens}", flush=True)
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
    rows = pd.read_csv(benchmark_path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {benchmark_path}, found {len(rows)}")
    rows["source_problem_id"] = rows["problem_id"]
    rows["dataset"] = dataset
    rows["method"] = method
    rows["backend"] = "transformers_hf"
    rows["runtime_comparable_to_ar"] = True
    rows["measurement_note"] = "measured draft + verify + decision/update; greedy decoding"
    if dataset != "gsm8k":
        rows["is_correct"] = math.nan
    for key, value in config.items():
        rows[key] = value
    return rows


def read_existing_rows(args):
    frames = []
    for dataset in args.datasets:
        for method in args.methods:
            path = local_output_dir(args, dataset, method) / "benchmark_results.csv"
            if not path.exists():
                continue
            rows = pd.read_csv(path)
            rows["source_problem_id"] = rows["problem_id"]
            rows["dataset"] = dataset
            rows["method"] = method
            rows["backend"] = "transformers_hf"
            rows["runtime_comparable_to_ar"] = True
            rows["measurement_note"] = "measured draft + verify + decision/update; greedy decoding"
            if dataset != "gsm8k":
                rows["is_correct"] = math.nan
            for key, value in LOCAL_METHODS[method].items():
                rows[key] = value
            frames.append(rows)
    if not frames:
        raise RuntimeError("No benchmark results were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_sum(group, column):
    if column not in group:
        return math.nan
    return pd.to_numeric(group[column], errors="coerce").sum(min_count=1)


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def add_paired_ar_metrics(rows):
    rows = rows.copy()
    rows["actual_measured_total_time"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["actual_measured_ms_per_output_token"] = (
        1000.0 * rows["actual_measured_total_time"] / pd.to_numeric(rows["output_tokens"], errors="coerce")
    )
    rows["actual_draft_ms_per_output_token"] = 1000.0 * pd.to_numeric(rows["actual_draft_time"], errors="coerce") / pd.to_numeric(rows["output_tokens"], errors="coerce")
    rows["actual_verify_ms_per_output_token"] = 1000.0 * pd.to_numeric(rows["actual_verify_time"], errors="coerce") / pd.to_numeric(rows["output_tokens"], errors="coerce")
    rows["actual_computation_ms_per_output_token"] = 1000.0 * pd.to_numeric(rows["actual_post_verify_time"], errors="coerce") / pd.to_numeric(rows["output_tokens"], errors="coerce")
    rows["draft_time_percent"] = 100.0 * pd.to_numeric(rows["actual_draft_time"], errors="coerce") / rows["actual_measured_total_time"]
    rows["verify_time_percent"] = 100.0 * pd.to_numeric(rows["actual_verify_time"], errors="coerce") / rows["actual_measured_total_time"]
    rows["computation_time_percent"] = 100.0 * pd.to_numeric(rows["actual_post_verify_time"], errors="coerce") / rows["actual_measured_total_time"]
    rows["output_tokens_per_ms"] = 1.0 / rows["actual_measured_ms_per_output_token"]
    ar = rows[rows["method"] == "ar_only"][["dataset", "problem_id", "actual_measured_ms_per_output_token", "output_token_hash"]].rename(
        columns={"actual_measured_ms_per_output_token": "ar_measured_ms_per_output_token", "output_token_hash": "ar_output_token_hash"}
    )
    rows = rows.merge(ar, on=["dataset", "problem_id"], how="left")
    rows["actual_speedup_vs_ar"] = rows["ar_measured_ms_per_output_token"] / rows["actual_measured_ms_per_output_token"]
    rows["output_matches_ar"] = (rows["output_token_hash"] == rows["ar_output_token_hash"]).astype("boolean")
    rows.loc[rows["method"] == "ar_only", "actual_speedup_vs_ar"] = 1.0
    rows.loc[rows["method"] == "ar_only", "output_matches_ar"] = True
    return rows


def aggregate_group(group):
    output_tokens = numeric_sum(group, "output_tokens")
    drafted_tokens = numeric_sum(group, "drafted_tokens")
    accepted_tokens = numeric_sum(group, "accepted_tokens")
    rounds = numeric_sum(group, "num_speculation_rounds")
    passes = numeric_sum(group, "total_num_forward_passes")
    draft_time = numeric_sum(group, "actual_draft_time")
    verify_time = numeric_sum(group, "actual_verify_time")
    computation_time = numeric_sum(group, "actual_post_verify_time")
    measured_total_time = draft_time + verify_time + computation_time
    e2e_time = numeric_sum(group, "actual_e2e_time")
    row = {
        "num_samples": len(group),
        "output_tokens": output_tokens,
        "actual_measured_total_time_s": measured_total_time,
        "actual_measured_total_time_mean_s": pd.to_numeric(group["actual_measured_total_time"], errors="coerce").mean(),
        "actual_measured_ms_per_output_token": safe_ratio(1000.0 * measured_total_time, output_tokens),
        "output_tokens_per_ms": safe_ratio(output_tokens, 1000.0 * measured_total_time),
        "actual_draft_time_s": draft_time,
        "actual_draft_time_mean_s": pd.to_numeric(group["actual_draft_time"], errors="coerce").mean(),
        "actual_draft_ms_per_output_token": safe_ratio(1000.0 * draft_time, output_tokens),
        "actual_verify_time_s": verify_time,
        "actual_verify_time_mean_s": pd.to_numeric(group["actual_verify_time"], errors="coerce").mean(),
        "actual_verify_ms_per_output_token": safe_ratio(1000.0 * verify_time, output_tokens),
        "actual_computation_time_s": computation_time,
        "actual_computation_time_mean_s": pd.to_numeric(group["actual_post_verify_time"], errors="coerce").mean(),
        "actual_computation_ms_per_output_token": safe_ratio(1000.0 * computation_time, output_tokens),
        "draft_time_percent": safe_ratio(100.0 * draft_time, measured_total_time),
        "verify_time_percent": safe_ratio(100.0 * verify_time, measured_total_time),
        "computation_time_percent": safe_ratio(100.0 * computation_time, measured_total_time),
        "diagnostic_e2e_time_s": e2e_time,
        "diagnostic_e2e_ms_per_output_token": safe_ratio(1000.0 * e2e_time, output_tokens),
        "actual_unattributed_core_time_mean_s": pd.to_numeric(group["actual_unattributed_core_time"], errors="coerce").mean(),
        "acceptance_rate_percent": safe_ratio(100.0 * accepted_tokens, drafted_tokens),
        "drafted_tokens_per_round": safe_ratio(drafted_tokens, rounds),
        "accepted_tokens_per_round": safe_ratio(accepted_tokens, rounds),
        "output_tokens_per_round": safe_ratio(output_tokens, rounds),
        "draft_forward_passes_per_100_output_tokens": safe_ratio(100.0 * passes, output_tokens),
        "verifier_rounds_per_100_output_tokens": safe_ratio(100.0 * rounds, output_tokens),
        "output_match_rate_vs_ar_percent": 100.0 * group["output_matches_ar"].mean(),
        "parsed_accuracy_percent": 100.0 * pd.to_numeric(group["is_correct"], errors="coerce").mean(),
        "runtime_comparable_to_ar": bool(group["runtime_comparable_to_ar"].all()),
        "backend": ",".join(sorted(group["backend"].dropna().unique())),
    }
    if group["method"].iloc[0] == "ar_only":
        row["acceptance_rate_percent"] = math.nan
        row["drafted_tokens_per_round"] = math.nan
        row["accepted_tokens_per_round"] = math.nan
    return row


def build_dataset_summary(rows):
    records = []
    for (dataset, method), group in rows.groupby(["dataset", "method"], sort=False):
        record = {"dataset": dataset, "method": method}
        record.update(aggregate_group(group))
        records.append(record)
    summary = pd.DataFrame(records)
    ar_tpt = summary[summary["method"] == "ar_only"].set_index("dataset")["actual_measured_ms_per_output_token"]
    summary["actual_speedup_vs_ar"] = summary.apply(
        lambda row: safe_ratio(ar_tpt.get(row["dataset"], math.nan), row["actual_measured_ms_per_output_token"]), axis=1
    )
    summary.loc[summary["method"] == "ar_only", "actual_speedup_vs_ar"] = 1.0
    summary["method_order"] = summary["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return summary.sort_values(["dataset", "method_order"]).drop(columns="method_order")


def build_average_summary(dataset_summary):
    numeric_columns = [
        column for column in dataset_summary.select_dtypes(include="number").columns
        if column != "num_samples"
    ]
    average = dataset_summary.groupby("method", as_index=False)[numeric_columns].mean()
    average["datasets_completed"] = dataset_summary.groupby("method")["dataset"].nunique().values
    average["method_order"] = average["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return average.sort_values("method_order").drop(columns="method_order")


def write_manifest(args, output_dir):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "methods": LOCAL_METHODS,
        "primary_metric": "actual_measured_ms_per_output_token",
        "primary_time_formula": "actual_draft_time + actual_verify_time + actual_post_verify_time",
        "component_definitions": {
            "actual_draft_time": "drafter call including dLLM denoising and internal stop/extend controller",
            "actual_verify_time": "target-model verification forward or AR-only target generation",
            "actual_post_verify_time": "accept/reject, replacement/bonus selection, latency EMA, and acceptance-calibration updates",
        },
        "speedup_formula": "AR aggregate measured ms/output-token divided by method aggregate measured ms/output-token",
        "timing_scope": "measured drafter execution, target verification, and required post-verification decision/controller updates; excludes model loading, dataset loading, prompt tokenization, output decoding, reporting, plots, file writes, and unattributed loop overhead",
        "decoding": "greedy",
        "warmup_policy": "warmup samples excluded; adaptive controller state reset immediately before the first measured sample",
        "hardware_latency_constants_used": False,
        "eagle3_included": False,
    }
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def save_reports(args, rows):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = add_paired_ar_metrics(rows)
    dataset_summary = build_dataset_summary(rows)
    average_summary = build_average_summary(dataset_summary)
    speedup_table = dataset_summary.pivot(index="method", columns="dataset", values="actual_speedup_vs_ar")
    speedup_table = speedup_table.reindex(index=METHOD_ORDER, columns=[dataset for dataset in DATASETS if dataset in speedup_table.columns])
    speedup_table["average"] = speedup_table.mean(axis=1)
    obsolete_columns = [
        column for column in rows.columns
        if column.startswith("theo_")
        or column.startswith("modeled_")
        or column in ("actual_speedup_vs_AR", "actual_e2e_speedup_vs_AR")
    ]
    rows.drop(columns=obsolete_columns, errors="ignore").to_csv(output_dir / "actual_per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "actual_dataset_summary.csv", index=False)
    average_summary.to_csv(output_dir / "actual_average_summary.csv", index=False)
    speedup_table.to_csv(output_dir / "actual_speedup_table.csv")
    write_manifest(args, output_dir)
    display_columns = [
        "dataset",
        "method",
        "num_samples",
        "actual_speedup_vs_ar",
        "output_tokens_per_ms",
        "actual_measured_ms_per_output_token",
        "actual_measured_total_time_mean_s",
        "actual_draft_time_mean_s",
        "actual_verify_time_mean_s",
        "actual_computation_time_mean_s",
        "draft_time_percent",
        "verify_time_percent",
        "computation_time_percent",
        "acceptance_rate_percent",
        "accepted_tokens_per_round",
        "output_tokens_per_round",
        "draft_forward_passes_per_100_output_tokens",
        "verifier_rounds_per_100_output_tokens",
        "output_match_rate_vs_ar_percent",
    ]
    print("\nACTUAL DRAFT + VERIFY + COMPUTATION SUMMARY", flush=True)
    print(dataset_summary[display_columns].to_string(index=False), flush=True)
    print("\nACTUAL SPEEDUP VS AR", flush=True)
    print(speedup_table.to_string(), flush=True)
    print(f"\nSaved reports: {output_dir}", flush=True)


def main():
    args = parse_args()
    validate_args(args)
    if args.aggregate_only:
        save_reports(args, read_existing_rows(args))
        return
    frames = []
    for dataset in args.datasets:
        for method in args.methods:
            frames.append(run_local_method(args, dataset, method))
    save_reports(args, pd.concat(frames, ignore_index=True, sort=False))


if __name__ == "__main__":
    main()
