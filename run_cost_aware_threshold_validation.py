import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATASETS = ("math", "aime", "gsm8k", "humaneval")
METHODS = {
    "cost_aware_lowconf_0p45": 0.45,
    "cost_aware_lowconf_0p60": 0.60,
}
BENCHMARK_VERSION = "cost_aware_v1_threshold_validation_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_cost_aware_threshold_validation_test20",
    )
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_questions <= 0:
        parser.error("--num_questions must be positive")
    if args.warmup_questions < 1:
        parser.error("--warmup_questions must be at least 1")
    for name in ("spec_len", "incr_len", "max_spec_len", "block_size", "small_block_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_spec_len < args.spec_len:
        parser.error("--max_spec_len must be at least --spec_len")
    return args


def run_streaming(command, cwd):
    process = subprocess.Popen(
        command,
        cwd=cwd,
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


def expected_questions(args, dataset):
    return min(args.num_questions, 30) if dataset == "aime" else args.num_questions


def metadata_for(args, dataset, method):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "method": method,
        "lowconf_threshold": METHODS[method],
        "num_questions": expected_questions(args, dataset),
        "warmup_questions": args.warmup_questions,
        "max_new_tokens": args.max_new_tokens,
        "spec_len": args.spec_len,
        "incr_len": args.incr_len,
        "max_spec_len": args.max_spec_len,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "drafter_threshold": args.drafter_threshold,
        "frontier_min_steps": args.frontier_min_steps,
        "frontier_patience": args.frontier_patience,
        "frontier_cost_token_equiv": args.frontier_cost_token_equiv,
        "seed": args.seed,
    }


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_complete(output_dir, metadata, expected_rows):
    result_path = output_dir / "benchmark_results.csv"
    round_path = output_dir / "frontier_round_diagnostics.csv"
    metadata_path = output_dir / "run_metadata.json"
    if not all(path.exists() for path in (result_path, round_path, metadata_path)):
        return False
    if load_json(metadata_path) != metadata:
        return False
    return len(pd.read_csv(result_path)) == expected_rows


def run_method(args, dataset, method):
    repo_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = expected_questions(args, dataset)
    metadata = metadata_for(args, dataset, method)
    metadata_path = output_dir / "run_metadata.json"

    if args.resume and run_complete(output_dir, metadata, expected_rows):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
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
            "--spec_len", str(args.spec_len),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(METHODS[method]),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(args.incr_len),
            "--frontier_stop_mode", "cost_aware",
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
        print(
            f"RUN {dataset} | {method} | samples={expected_rows} | "
            f"spec_len={args.spec_len} | incr_len={args.incr_len} | "
            f"lowconf={METHODS[method]}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, repo_dir)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    benchmark = pd.read_csv(output_dir / "benchmark_results.csv")
    rounds = pd.read_csv(output_dir / "frontier_round_diagnostics.csv")
    extension_path = output_dir / "frontier_extension_diagnostics.csv"
    extensions = pd.read_csv(extension_path) if extension_path.exists() else pd.DataFrame()
    gain_path = output_dir / "frontier_gain_diagnostics.csv"
    gains = pd.read_csv(gain_path) if gain_path.exists() else pd.DataFrame()
    for frame in (benchmark, rounds, extensions, gains):
        if not frame.empty:
            frame["dataset"] = dataset
            frame["method"] = method
    benchmark["actual_measured_time"] = (
        pd.to_numeric(benchmark["actual_draft_time"], errors="coerce")
        + pd.to_numeric(benchmark["actual_verify_time"], errors="coerce")
        + pd.to_numeric(benchmark["actual_post_verify_time"], errors="coerce")
    )
    benchmark["actual_measured_ms_per_output_token"] = (
        1000.0
        * benchmark["actual_measured_time"]
        / pd.to_numeric(benchmark["output_tokens"], errors="coerce")
    )
    return benchmark, rounds, extensions, gains


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def summarize_timing(benchmark):
    records = []
    for (dataset, method), group in benchmark.groupby(["dataset", "method"], sort=False):
        output_tokens = pd.to_numeric(group["output_tokens"], errors="coerce").sum()
        rounds = pd.to_numeric(group["num_speculation_rounds"], errors="coerce").sum()
        drafted = pd.to_numeric(group["drafted_tokens"], errors="coerce").sum()
        accepted = pd.to_numeric(group["accepted_tokens"], errors="coerce").sum()
        draft_time = pd.to_numeric(group["actual_draft_time"], errors="coerce").sum()
        verify_time = pd.to_numeric(group["actual_verify_time"], errors="coerce").sum()
        controller_time = pd.to_numeric(group["actual_post_verify_time"], errors="coerce").sum()
        total_time = draft_time + verify_time + controller_time
        records.append({
            "dataset": dataset,
            "method": method,
            "lowconf_threshold": METHODS[method],
            "num_samples": len(group),
            "output_tokens": output_tokens,
            "actual_measured_time_s": total_time,
            "actual_measured_ms_per_output_token": safe_ratio(1000.0 * total_time, output_tokens),
            "draft_ms_per_output_token": safe_ratio(1000.0 * draft_time, output_tokens),
            "verify_ms_per_output_token": safe_ratio(1000.0 * verify_time, output_tokens),
            "controller_ms_per_output_token": safe_ratio(1000.0 * controller_time, output_tokens),
            "acceptance_rate_percent": safe_ratio(100.0 * accepted, drafted),
            "drafted_tokens_per_round": safe_ratio(drafted, rounds),
            "accepted_tokens_per_round": safe_ratio(accepted, rounds),
            "output_tokens_per_round": safe_ratio(output_tokens, rounds),
            "verifier_rounds_per_100_output_tokens": safe_ratio(100.0 * rounds, output_tokens),
        })
    return pd.DataFrame(records).sort_values(["dataset", "lowconf_threshold"])


def summarize_rounds(rounds):
    records = []
    for (dataset, method), group in rounds.groupby(["dataset", "method"], sort=False):
        full_accept = pd.to_numeric(group["full_accept"], errors="coerce").sum()
        full_accept_capacity = pd.to_numeric(
            group["full_accept_with_extension_capacity"], errors="coerce"
        ).sum()
        zero_extension = group[
            (pd.to_numeric(group["full_accept_with_extension_capacity"], errors="coerce") == 1)
            & (pd.to_numeric(group["extension_count"], errors="coerce") == 0)
        ]
        full_accept_capacity_group = group[
            pd.to_numeric(group["full_accept_with_extension_capacity"], errors="coerce") == 1
        ]
        full_accept_capacity_cost_stop = pd.to_numeric(
            full_accept_capacity_group["cost_stop_requested"], errors="coerce"
        ).sum()
        full_accept_capacity_lowconf_stop = full_accept_capacity_group[
            full_accept_capacity_group["stop_reason"].eq("failfast_low_confidence")
        ]
        cost_stops = pd.to_numeric(group["cost_stop_requested"], errors="coerce").sum()
        records.append({
            "dataset": dataset,
            "method": method,
            "lowconf_threshold": METHODS[method],
            "num_rounds": len(group),
            "full_accept_rounds": full_accept,
            "full_accept_rate_percent": safe_ratio(100.0 * full_accept, len(group)),
            "full_accept_with_extension_capacity_rounds": full_accept_capacity,
            "full_accept_with_capacity_rate_all_rounds_percent": safe_ratio(
                100.0 * full_accept_capacity, len(group)
            ),
            "full_accept_with_capacity_rate_given_full_accept_percent": safe_ratio(
                100.0 * full_accept_capacity, full_accept
            ),
            "full_accept_with_capacity_zero_extension_rounds": len(zero_extension),
            "full_accept_with_capacity_zero_extension_rate_percent": safe_ratio(
                100.0 * len(zero_extension), full_accept_capacity
            ),
            "full_accept_with_capacity_cost_stop_requested_rounds": full_accept_capacity_cost_stop,
            "full_accept_with_capacity_cost_stop_requested_rate_percent": safe_ratio(
                100.0 * full_accept_capacity_cost_stop, full_accept_capacity
            ),
            "full_accept_with_capacity_lowconf_stop_rounds": len(full_accept_capacity_lowconf_stop),
            "full_accept_with_capacity_lowconf_stop_rate_percent": safe_ratio(
                100.0 * len(full_accept_capacity_lowconf_stop), full_accept_capacity
            ),
            "cost_stop_requested_rounds": cost_stops,
            "mean_extension_count": pd.to_numeric(group["extension_count"], errors="coerce").mean(),
            "mean_draft_len": pd.to_numeric(group["draft_len"], errors="coerce").mean(),
            "mean_accepted_len": pd.to_numeric(group["accepted_len"], errors="coerce").mean(),
        })
    return pd.DataFrame(records).sort_values(["dataset", "lowconf_threshold"])


def prediction_metrics(group, predicted_column, actual_column):
    predicted = pd.to_numeric(group[predicted_column], errors="coerce")
    actual = pd.to_numeric(group[actual_column], errors="coerce")
    valid = predicted.notna() & actual.notna()
    predicted = predicted[valid]
    actual = actual[valid]
    if predicted.empty:
        return {
            "num_observations": 0,
            "predicted_mean": math.nan,
            "actual_mean": math.nan,
            "bias_predicted_minus_actual": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "pearson_correlation": math.nan,
            "predicted_to_actual_ratio": math.nan,
        }
    error = predicted - actual
    return {
        "num_observations": len(predicted),
        "predicted_mean": predicted.mean(),
        "actual_mean": actual.mean(),
        "bias_predicted_minus_actual": error.mean(),
        "mae": error.abs().mean(),
        "rmse": math.sqrt((error ** 2).mean()),
        "pearson_correlation": (
            predicted.corr(actual)
            if len(predicted) > 1 and predicted.std() > 0 and actual.std() > 0
            else math.nan
        ),
        "predicted_to_actual_ratio": safe_ratio(predicted.sum(), actual.sum()),
    }


def summarize_expected_prefix(rounds):
    records = []
    for (dataset, method), group in rounds.groupby(["dataset", "method"], sort=False):
        record = {
            "dataset": dataset,
            "method": method,
            "lowconf_threshold": METHODS[method],
        }
        record.update(prediction_metrics(
            group,
            "predicted_accepted_tokens",
            "actual_accepted_tokens",
        ))
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "lowconf_threshold"])


def summarize_extension_gain(extensions):
    if extensions.empty:
        return pd.DataFrame()
    valid = extensions[
        extensions["trigger"].eq("high_confidence_extend")
        & pd.to_numeric(extensions["predicted_extension_gain"], errors="coerce").notna()
    ].copy()
    records = []
    for (dataset, method), group in valid.groupby(["dataset", "method"], sort=False):
        record = {
            "dataset": dataset,
            "method": method,
            "lowconf_threshold": METHODS[method],
            "extension_reached_rate_percent": 100.0 * (
                pd.to_numeric(group["actual_extension_accepted_tokens"], errors="coerce") > 0
            ).mean(),
            "original_prefix_full_accept_rate_percent": 100.0 * pd.to_numeric(
                group["original_prefix_fully_accepted"], errors="coerce"
            ).mean(),
        }
        record.update(prediction_metrics(
            group,
            "predicted_extension_gain",
            "actual_extension_accepted_tokens",
        ))
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "lowconf_threshold"])


def summarize_next_step_gain(gains):
    if gains.empty:
        return pd.DataFrame()
    valid = gains[pd.to_numeric(gains["same_target_len"], errors="coerce") == 1].copy()
    records = []
    for (dataset, method), group in valid.groupby(["dataset", "method"], sort=False):
        record = {
            "dataset": dataset,
            "method": method,
            "lowconf_threshold": METHODS[method],
        }
        record.update(prediction_metrics(
            group,
            "predicted_next_gain",
            "actual_next_gain",
        ))
        records.append(record)
    return pd.DataFrame(records).sort_values(["dataset", "lowconf_threshold"])


def build_calibration_table(frame, predicted_column, actual_column, bins):
    if frame.empty or predicted_column not in frame or actual_column not in frame:
        return pd.DataFrame()
    data = frame.copy()
    data[predicted_column] = pd.to_numeric(data[predicted_column], errors="coerce")
    data[actual_column] = pd.to_numeric(data[actual_column], errors="coerce")
    data = data.dropna(subset=[predicted_column, actual_column])
    if data.empty:
        return pd.DataFrame()
    data["prediction_bin"] = pd.cut(data[predicted_column], bins=bins, right=False)
    result = data.groupby(
        ["dataset", "method", "prediction_bin"],
        observed=True,
        sort=False,
    ).agg(
        num_observations=(actual_column, "size"),
        predicted_mean=(predicted_column, "mean"),
        actual_mean=(actual_column, "mean"),
    ).reset_index()
    result["bias_predicted_minus_actual"] = result["predicted_mean"] - result["actual_mean"]
    result["lowconf_threshold"] = result["method"].map(METHODS)
    return result


def build_paired_comparison(benchmark):
    columns = [
        "dataset",
        "problem_id",
        "actual_measured_ms_per_output_token",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
        "output_tokens",
        "num_speculation_rounds",
        "accepted_tokens",
        "drafted_tokens",
        "output_token_hash",
    ]
    left = benchmark[benchmark["method"] == "cost_aware_lowconf_0p45"][columns].copy()
    right = benchmark[benchmark["method"] == "cost_aware_lowconf_0p60"][columns].copy()
    left = left.rename(columns={column: f"lowconf_0p45_{column}" for column in columns[2:]})
    right = right.rename(columns={column: f"lowconf_0p60_{column}" for column in columns[2:]})
    paired = left.merge(right, on=["dataset", "problem_id"], validate="one_to_one")
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
    return paired


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
        "methods": METHODS,
        "controlled_variables": [
            "dataset",
            "problem_id",
            "prompt",
            "seed",
            "model",
            "spec_len",
            "incr_len",
            "max_spec_len",
            "drafter_threshold",
            "max_new_tokens",
        ],
        "only_changed_variable": "lowconf_threshold",
        "primary_timing_metric": "actual_draft_time + actual_verify_time + actual_post_verify_time per output token",
        "expected_prefix_error": "predicted expected accepted prefix minus actual verifier accepted prefix",
        "extension_gain_error": "predicted extension gain minus accepted tokens beyond the pre-extension boundary",
    }
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_frames = []
    round_frames = []
    extension_frames = []
    gain_frames = []

    for dataset_index, dataset in enumerate(args.datasets):
        method_order = list(METHODS)
        if dataset_index % 2:
            method_order.reverse()
        for method in method_order:
            benchmark, rounds, extensions, gains = run_method(args, dataset, method)
            benchmark_frames.append(benchmark)
            round_frames.append(rounds)
            if not extensions.empty:
                extension_frames.append(extensions)
            if not gains.empty:
                gain_frames.append(gains)

    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    rounds = pd.concat(round_frames, ignore_index=True)
    extensions = (
        pd.concat(extension_frames, ignore_index=True)
        if extension_frames
        else pd.DataFrame()
    )
    gains = pd.concat(gain_frames, ignore_index=True) if gain_frames else pd.DataFrame()

    timing = summarize_timing(benchmark)
    full_accept = summarize_rounds(rounds)
    expected_prefix = summarize_expected_prefix(rounds)
    extension_gain = summarize_extension_gain(extensions)
    next_step_gain = summarize_next_step_gain(gains)
    overall_timing = summarize_timing(benchmark.assign(dataset="ALL"))
    overall_full_accept = summarize_rounds(rounds.assign(dataset="ALL"))
    overall_expected_prefix = summarize_expected_prefix(rounds.assign(dataset="ALL"))
    overall_extension_gain = summarize_extension_gain(
        extensions.assign(dataset="ALL") if not extensions.empty else extensions
    )
    overall_next_step_gain = summarize_next_step_gain(
        gains.assign(dataset="ALL") if not gains.empty else gains
    )
    prefix_calibration = build_calibration_table(
        rounds,
        "predicted_accepted_tokens",
        "actual_accepted_tokens",
        [-math.inf, 1, 2, 3, 4, 5, 6, 8, 10, 15, math.inf],
    )
    extension_calibration = build_calibration_table(
        extensions[extensions["trigger"].eq("high_confidence_extend")]
        if not extensions.empty
        else extensions,
        "predicted_extension_gain",
        "actual_extension_accepted_tokens",
        [-math.inf, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, math.inf],
    )
    next_step_gain_calibration = build_calibration_table(
        gains[pd.to_numeric(gains["same_target_len"], errors="coerce") == 1]
        if not gains.empty
        else gains,
        "predicted_next_gain",
        "actual_next_gain",
        [-math.inf, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, math.inf],
    )
    paired = build_paired_comparison(benchmark)
    lowconf_0p45 = overall_timing[
        overall_timing["method"] == "cost_aware_lowconf_0p45"
    ].iloc[0]
    lowconf_0p60 = overall_timing[
        overall_timing["method"] == "cost_aware_lowconf_0p60"
    ].iloc[0]
    overall_comparison = pd.DataFrame([{
        "num_paired_samples": len(paired),
        "pooled_speedup_0p60_vs_0p45": safe_ratio(
            lowconf_0p45["actual_measured_ms_per_output_token"],
            lowconf_0p60["actual_measured_ms_per_output_token"],
        ),
        "paired_0p60_win_rate_percent": 100.0 * paired["lowconf_0p60_faster"].mean(),
        "output_match_rate_percent": 100.0 * paired["output_matches"].mean(),
        "lowconf_0p45_ms_per_output_token": lowconf_0p45["actual_measured_ms_per_output_token"],
        "lowconf_0p60_ms_per_output_token": lowconf_0p60["actual_measured_ms_per_output_token"],
    }])

    benchmark.to_csv(output_dir / "all_benchmark_results.csv", index=False)
    rounds.to_csv(output_dir / "all_round_diagnostics.csv", index=False)
    extensions.to_csv(output_dir / "all_extension_diagnostics.csv", index=False)
    gains.to_csv(output_dir / "all_next_step_gain_diagnostics.csv", index=False)
    timing.to_csv(output_dir / "timing_summary.csv", index=False)
    overall_timing.to_csv(output_dir / "overall_timing_summary.csv", index=False)
    full_accept.to_csv(output_dir / "full_accept_no_extend_summary.csv", index=False)
    overall_full_accept.to_csv(output_dir / "overall_full_accept_no_extend_summary.csv", index=False)
    expected_prefix.to_csv(output_dir / "expected_prefix_validation_summary.csv", index=False)
    overall_expected_prefix.to_csv(output_dir / "overall_expected_prefix_validation_summary.csv", index=False)
    extension_gain.to_csv(output_dir / "extension_gain_validation_summary.csv", index=False)
    overall_extension_gain.to_csv(output_dir / "overall_extension_gain_validation_summary.csv", index=False)
    next_step_gain.to_csv(output_dir / "next_step_gain_validation_summary.csv", index=False)
    overall_next_step_gain.to_csv(output_dir / "overall_next_step_gain_validation_summary.csv", index=False)
    prefix_calibration.to_csv(output_dir / "expected_prefix_calibration.csv", index=False)
    extension_calibration.to_csv(output_dir / "extension_gain_calibration.csv", index=False)
    next_step_gain_calibration.to_csv(output_dir / "next_step_gain_calibration.csv", index=False)
    paired.to_csv(output_dir / "paired_threshold_comparison.csv", index=False)
    overall_comparison.to_csv(output_dir / "overall_threshold_comparison.csv", index=False)
    write_manifest(args, output_dir)

    print("\nTIMING SUMMARY")
    print(timing.to_string(index=False))
    print("\nOVERALL THRESHOLD COMPARISON")
    print(overall_comparison.to_string(index=False))
    print("\nFULL ACCEPT WITHOUT FURTHER EXTENSION")
    print(full_accept.to_string(index=False))
    print("\nEXPECTED PREFIX VALIDATION")
    print(expected_prefix.to_string(index=False))
    print("\nEXTENSION GAIN VALIDATION")
    print(extension_gain.to_string(index=False) if not extension_gain.empty else "No extension events")
    print("\nNEXT DENOISING STEP GAIN VALIDATION")
    print(next_step_gain.to_string(index=False) if not next_step_gain.empty else "No comparable step transitions")
    print(f"\nSaved report: {output_dir}")


if __name__ == "__main__":
    main()
