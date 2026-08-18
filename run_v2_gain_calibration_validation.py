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


DATASETS = ("math", "aime", "gsm8k", "humaneval")
BENCHMARK_VERSION = "cost_aware_v2_gain_calibration_validation_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_v2_hysteresis", type=float, default=0.03)
    parser.add_argument("--frontier_v2_extension_cost_margin", type=float, default=-0.03)
    parser.add_argument("--frontier_v2_hazard_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_extension_prior_strength", type=float, default=2.0)
    parser.add_argument("--frontier_v2_min_hazard_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_min_calibration_tokens", type=int, default=64)
    parser.add_argument("--frontier_v2_gain_calibration_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_min_gain_calibration_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_prefix_calibration_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_min_prefix_calibration_observations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_v2_gain_calibration_validation_test10",
    )
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not -0.99 < args.frontier_v2_extension_cost_margin <= 1.0:
        raise ValueError("--frontier_v2_extension_cost_margin must be in (-0.99, 1.0]")
    if args.frontier_v2_gain_calibration_prior_strength <= 0:
        raise ValueError("--frontier_v2_gain_calibration_prior_strength must be positive")
    if args.frontier_v2_min_gain_calibration_observations <= 0:
        raise ValueError("--frontier_v2_min_gain_calibration_observations must be positive")
    if args.frontier_v2_prefix_calibration_prior_strength <= 0:
        raise ValueError("--frontier_v2_prefix_calibration_prior_strength must be positive")
    if args.frontier_v2_min_prefix_calibration_observations <= 0:
        raise ValueError("--frontier_v2_min_prefix_calibration_observations must be positive")


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


def run_metadata(args, dataset):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "num_questions": args.num_questions,
        "warmup_questions": args.warmup_questions,
        "max_new_tokens": args.max_new_tokens,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "spec_len": args.spec_len,
        "incr_len": args.incr_len,
        "lowconf_threshold": args.lowconf_threshold,
        "drafter_threshold": args.drafter_threshold,
        "max_spec_len": args.max_spec_len,
        "frontier_min_steps": args.frontier_min_steps,
        "frontier_patience": args.frontier_patience,
        "frontier_v2_hysteresis": args.frontier_v2_hysteresis,
        "frontier_v2_extension_cost_margin": args.frontier_v2_extension_cost_margin,
        "frontier_v2_hazard_prior_strength": args.frontier_v2_hazard_prior_strength,
        "frontier_v2_extension_prior_strength": args.frontier_v2_extension_prior_strength,
        "frontier_v2_min_hazard_observations": args.frontier_v2_min_hazard_observations,
        "frontier_v2_min_calibration_tokens": args.frontier_v2_min_calibration_tokens,
        "frontier_v2_gain_calibration_prior_strength": args.frontier_v2_gain_calibration_prior_strength,
        "frontier_v2_min_gain_calibration_observations": args.frontier_v2_min_gain_calibration_observations,
        "frontier_v2_prefix_calibration_prior_strength": args.frontier_v2_prefix_calibration_prior_strength,
        "frontier_v2_min_prefix_calibration_observations": args.frontier_v2_min_prefix_calibration_observations,
        "seed": args.seed,
    }


def results_complete(result_path, diagnostics_path, metadata_path, metadata):
    if not result_path.exists() or not diagnostics_path.exists() or not metadata_path.exists():
        return False
    try:
        results = pd.read_csv(result_path)
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return (
        saved_metadata == metadata
        and len(results) == metadata["num_questions"]
        and results["problem_id"].nunique() == metadata["num_questions"]
    )


def run_dataset(args, dataset):
    output_dir = Path(args.output_dir) / "raw" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    diagnostics_path = output_dir / "frontier_extension_diagnostics.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata(args, dataset)
    if not (
        args.resume
        and results_complete(result_path, diagnostics_path, metadata_path, metadata)
    ):
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
            "frontier_v2_runtime_state.json",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(args.num_questions),
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
            "--sweep_lowconf_threshold", str(args.lowconf_threshold),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(args.incr_len),
            "--frontier_stop_mode", "cost_aware_v2",
            "--frontier_min_steps", str(args.frontier_min_steps),
            "--frontier_patience", str(args.frontier_patience),
            "--frontier_v2_hysteresis", str(args.frontier_v2_hysteresis),
            "--frontier_v2_extension_cost_margin", str(args.frontier_v2_extension_cost_margin),
            "--frontier_v2_hazard_prior_strength", str(args.frontier_v2_hazard_prior_strength),
            "--frontier_v2_extension_prior_strength", str(args.frontier_v2_extension_prior_strength),
            "--frontier_v2_min_hazard_observations", str(args.frontier_v2_min_hazard_observations),
            "--frontier_v2_min_calibration_tokens", str(args.frontier_v2_min_calibration_tokens),
            "--frontier_v2_gain_calibration_prior_strength", str(args.frontier_v2_gain_calibration_prior_strength),
            "--frontier_v2_min_gain_calibration_observations", str(args.frontier_v2_min_gain_calibration_observations),
            "--frontier_v2_prefix_calibration_prior_strength", str(args.frontier_v2_prefix_calibration_prior_strength),
            "--frontier_v2_min_prefix_calibration_observations", str(args.frontier_v2_min_prefix_calibration_observations),
            "--seed", str(args.seed),
            "--quiet_generation",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]
        print("\n" + "=" * 100, flush=True)
        print(f"RUN {dataset} | samples={args.num_questions} | gain calibration validation", flush=True)
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    events = pd.read_csv(diagnostics_path)
    events = events[
        (events["trigger"] == "cost_aware_v2_extend")
        & events["raw_predicted_extension_gain"].notna()
        & events["predicted_extension_gain"].notna()
    ].copy()
    events["dataset"] = dataset
    return events


def safe_ratio(numerator, denominator):
    if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
        return math.nan
    return numerator / denominator


def correlation(left, right):
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return math.nan
    return left.corr(right)


def summarize_events(group):
    raw = pd.to_numeric(group["raw_predicted_extension_gain"], errors="coerce")
    calibrated = pd.to_numeric(group["predicted_extension_gain"], errors="coerce")
    actual = pd.to_numeric(group["actual_extension_accepted_tokens"], errors="coerce")
    raw_error = raw - actual
    calibrated_error = calibrated - actual
    raw_mae = raw_error.abs().mean()
    calibrated_mae = calibrated_error.abs().mean()
    raw_prefix = pd.to_numeric(
        group["raw_prefix_survival_probability"], errors="coerce"
    )
    calibrated_prefix = pd.to_numeric(
        group["calibrated_prefix_survival_probability"], errors="coerce"
    )
    actual_prefix = pd.to_numeric(
        group["original_prefix_fully_accepted"], errors="coerce"
    )
    prefix_full = actual_prefix == 1
    raw_conditional = pd.to_numeric(
        group["raw_conditional_extension_gain"], errors="coerce"
    )[prefix_full]
    calibrated_conditional = pd.to_numeric(
        group["calibrated_conditional_extension_gain"], errors="coerce"
    )[prefix_full]
    actual_conditional = actual[prefix_full]
    if "dataset" in group:
        num_problems = group[["dataset", "problem_id"]].drop_duplicates().shape[0]
    else:
        num_problems = group["problem_id"].nunique()
    return {
        "num_events": len(group),
        "num_problems": num_problems,
        "mean_actual_gain": actual.mean(),
        "mean_raw_expected_gain": raw.mean(),
        "mean_calibrated_expected_gain": calibrated.mean(),
        "raw_bias_pred_minus_actual": raw_error.mean(),
        "calibrated_bias_pred_minus_actual": calibrated_error.mean(),
        "raw_mae": raw_mae,
        "calibrated_mae": calibrated_mae,
        "mae_reduction_percent": 100.0 * safe_ratio(raw_mae - calibrated_mae, raw_mae),
        "raw_rmse": math.sqrt((raw_error.pow(2)).mean()),
        "calibrated_rmse": math.sqrt((calibrated_error.pow(2)).mean()),
        "raw_underprediction_rate_percent": 100.0 * (raw < actual).mean(),
        "calibrated_underprediction_rate_percent": 100.0 * (calibrated < actual).mean(),
        "actual_to_raw_total_ratio": safe_ratio(actual.sum(), raw.sum()),
        "actual_to_calibrated_total_ratio": safe_ratio(actual.sum(), calibrated.sum()),
        "raw_pearson_correlation": correlation(raw, actual),
        "calibrated_pearson_correlation": correlation(calibrated, actual),
        "mean_gain_correction": pd.to_numeric(
            group["extension_gain_correction"], errors="coerce"
        ).mean(),
        "prefix_fully_accepted_rate_percent": 100.0 * actual_prefix.mean(),
        "mean_raw_prefix_survival": raw_prefix.mean(),
        "mean_calibrated_prefix_survival": calibrated_prefix.mean(),
        "raw_prefix_bias": (raw_prefix - actual_prefix).mean(),
        "calibrated_prefix_bias": (calibrated_prefix - actual_prefix).mean(),
        "raw_prefix_brier_score": ((raw_prefix - actual_prefix) ** 2).mean(),
        "calibrated_prefix_brier_score": (
            (calibrated_prefix - actual_prefix) ** 2
        ).mean(),
        "mean_actual_gain_given_prefix_full": actual_conditional.mean(),
        "mean_raw_conditional_gain": raw_conditional.mean(),
        "mean_calibrated_conditional_gain": calibrated_conditional.mean(),
        "raw_conditional_bias": (raw_conditional - actual_conditional).mean(),
        "calibrated_conditional_bias": (
            calibrated_conditional - actual_conditional
        ).mean(),
        "raw_conditional_mae": (raw_conditional - actual_conditional).abs().mean(),
        "calibrated_conditional_mae": (
            calibrated_conditional - actual_conditional
        ).abs().mean(),
    }


def grouped_summary(events, group_columns):
    records = []
    for keys, group in events.groupby(group_columns, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record.update(summarize_events(group))
        records.append(record)
    return pd.DataFrame(records)


def add_analysis_columns(events):
    events = events.copy()
    events["raw_error"] = (
        events["raw_predicted_extension_gain"] - events["actual_extension_accepted_tokens"]
    )
    events["calibrated_error"] = (
        events["predicted_extension_gain"] - events["actual_extension_accepted_tokens"]
    )
    events["raw_absolute_error"] = events["raw_error"].abs()
    events["calibrated_absolute_error"] = events["calibrated_error"].abs()
    events["proposal_length_bucket"] = events["from_len"].map(
        lambda length: f"{8 * ((int(length) - 1) // 8) + 1}-{8 * math.ceil(int(length) / 8)}"
    )
    count = pd.to_numeric(events["gain_calibration_count"], errors="coerce").fillna(0)
    source = events["gain_calibration_source"].fillna("uncalibrated")
    events["calibration_stage"] = source + ":" + pd.cut(
        count,
        bins=[-1, 0, 31, float("inf")],
        labels=["0", "1-31", "32+"],
    ).astype(str)
    prefix_count = pd.to_numeric(
        events["prefix_calibration_count"], errors="coerce"
    ).fillna(0)
    prefix_source = events["prefix_calibration_source"].fillna("uncalibrated")
    events["prefix_calibration_stage"] = prefix_source + ":" + pd.cut(
        prefix_count,
        bins=[-1, 0, 31, float("inf")],
        labels=["0", "1-31", "32+"],
    ).astype(str)
    return events


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
        "evaluation_unit": "executed cost_aware_v2 extension event",
        "raw_prediction": "raw prefix survival multiplied by raw conditional extension gain",
        "calibrated_prediction": "calibrated prefix survival multiplied by calibrated conditional extension gain",
        "actual_gain": "draft extension tokens accepted by greedy verifier",
        "leakage_policy": "each event prediction uses only outcomes from earlier verifier rounds",
        "limitation": "on-policy evaluation does not observe rejected extension counterfactuals",
    }
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [run_dataset(args, dataset) for dataset in args.datasets]
    events = pd.concat(frames, ignore_index=True, sort=False)
    if events.empty:
        raise RuntimeError("No calibrated Cost-Aware v2 extension events were recorded")
    events = add_analysis_columns(events)
    dataset_summary = grouped_summary(events, ["dataset"])
    overall_summary = pd.DataFrame([summarize_events(events)])
    length_summary = grouped_summary(events, ["dataset", "proposal_length_bucket"])
    calibration_stage_summary = grouped_summary(events, ["dataset", "calibration_stage"])
    prefix_calibration_stage_summary = grouped_summary(
        events, ["dataset", "prefix_calibration_stage"]
    )
    events.to_csv(output_dir / "per_extension_event.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_gain_calibration_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "overall_gain_calibration_summary.csv", index=False)
    length_summary.to_csv(output_dir / "proposal_length_gain_summary.csv", index=False)
    calibration_stage_summary.to_csv(output_dir / "calibration_stage_summary.csv", index=False)
    prefix_calibration_stage_summary.to_csv(
        output_dir / "prefix_calibration_stage_summary.csv", index=False
    )
    write_manifest(args, output_dir)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nDATASET GAIN CALIBRATION SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nOVERALL GAIN CALIBRATION SUMMARY")
    print(overall_summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
