import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SIZES = {
    "math": 500,
    "aime": 30,
    "gsm8k": 1319,
    "humaneval": 164,
}
METHODS = {
    "bucket_renewal_spec8": {
        "frontier_mode": "bucket_renewal",
        "spec_len": 8,
        "incr_len": 8,
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
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--bucket_renewal_min_steps", type=int, default=1)
    parser.add_argument("--bucket_renewal_hysteresis", type=float, default=0.0)
    parser.add_argument("--bucket_prior_strength", type=float, default=8.0)
    parser.add_argument("--bucket_min_observations", type=int, default=8)
    parser.add_argument("--bucket_latency_ema_alpha", type=float, default=0.2)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference_csv")
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_bucket_renewal_test15",
    )
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1")
    if args.bucket_renewal_min_steps < 1:
        raise ValueError("--bucket_renewal_min_steps must be at least 1")
    if not 0.0 <= args.bucket_renewal_hysteresis < 1.0:
        raise ValueError("--bucket_renewal_hysteresis must be in [0, 1)")
    for dataset in args.datasets:
        available = DATASET_SIZES[dataset] - args.warmup_questions
        if args.num_questions > available:
            raise ValueError(f"{dataset} has only {available} non-warmup samples")


def sampled_problem_ids(args):
    sampled = {}
    for dataset_index, dataset in enumerate(args.datasets):
        population = list(range(args.warmup_questions, DATASET_SIZES[dataset]))
        rng = random.Random(args.sample_seed + dataset_index * 1009)
        sampled[dataset] = sorted(rng.sample(population, args.num_questions))
    return sampled


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


def metadata_for(args, dataset, method, problem_ids):
    return {
        "version": "bucket_renewal_v3_step_buckets",
        "dataset": dataset,
        "method": method,
        "problem_ids": problem_ids,
        "max_new_tokens": args.max_new_tokens,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "drafter_threshold": args.drafter_threshold,
        "lowconf_threshold": args.lowconf_threshold,
        "max_spec_len": args.max_spec_len,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "bucket_renewal_min_steps": args.bucket_renewal_min_steps,
        "bucket_renewal_hysteresis": args.bucket_renewal_hysteresis,
        "bucket_prior_strength": args.bucket_prior_strength,
        "bucket_min_observations": args.bucket_min_observations,
        "bucket_latency_ema_alpha": args.bucket_latency_ema_alpha,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "method_config": METHODS[method],
    }


def result_is_complete(result_path, metadata_path, expected_metadata):
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
        and sorted(rows["problem_id"].astype(int).tolist()) == expected_ids
    )


def run_method(args, dataset, method, problem_ids):
    config = METHODS[method]
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = metadata_for(args, dataset, method, problem_ids)
    if args.resume and result_is_complete(result_path, metadata_path, metadata):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
            "bucket_renewal_runtime_state.json",
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
            "--sweep_lowconf_threshold", str(args.lowconf_threshold),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(config["incr_len"]),
            "--frontier_stop_mode", config["frontier_mode"],
            "--bucket_renewal_min_steps", str(args.bucket_renewal_min_steps),
            "--bucket_renewal_hysteresis", str(args.bucket_renewal_hysteresis),
            "--bucket_prior_strength", str(args.bucket_prior_strength),
            "--bucket_min_observations", str(args.bucket_min_observations),
            "--bucket_latency_ema_alpha", str(args.bucket_latency_ema_alpha),
            "--seed", str(args.seed),
            "--quiet_generation",
            "--disable_progress",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]
        if config["frontier_mode"] == "bucket_renewal":
            command.append("--collect_draft_diagnostics")
        print("\n" + "=" * 100, flush=True)
        print(
            f"RUN {dataset} | {method} | samples={len(problem_ids)} | "
            f"problem_ids={problem_ids}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = pd.read_csv(result_path)
    rows["dataset"] = dataset
    rows["method"] = method
    rows["measured_time_s"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["measured_ms_per_output_token"] = (
        1000.0 * rows["measured_time_s"] / rows["output_tokens"]
    )
    rows["verifier_rounds_per_100_tokens"] = (
        100.0 * rows["num_speculation_rounds"] / rows["output_tokens"]
    )
    rows["draft_passes_per_100_tokens"] = (
        100.0 * rows["total_num_forward_passes"] / rows["output_tokens"]
    )
    rows["output_tokens_per_round"] = (
        rows["output_tokens"] / rows["num_speculation_rounds"]
    )
    return rows


def aggregate(rows):
    records = []
    for (dataset, method), group in rows.groupby(["dataset", "method"], sort=False):
        output_tokens = group["output_tokens"].sum()
        drafted_tokens = group["drafted_tokens"].sum()
        records.append({
            "dataset": dataset,
            "method": method,
            "num_samples": len(group),
            "measured_ms_per_output_token": 1000.0 * group["measured_time_s"].sum() / output_tokens,
            "draft_ms_per_output_token": 1000.0 * group["actual_draft_time"].sum() / output_tokens,
            "verify_ms_per_output_token": 1000.0 * group["actual_verify_time"].sum() / output_tokens,
            "post_verify_ms_per_output_token": 1000.0 * group["actual_post_verify_time"].sum() / output_tokens,
            "acceptance_rate_percent": 100.0 * group["accepted_tokens"].sum() / drafted_tokens,
            "verifier_rounds_per_100_tokens": 100.0 * group["num_speculation_rounds"].sum() / output_tokens,
            "draft_passes_per_100_tokens": 100.0 * group["total_num_forward_passes"].sum() / output_tokens,
            "output_tokens_per_round": output_tokens / group["num_speculation_rounds"].sum(),
            "bucket_stop_actions": group.get(
                "bucket_stop_actions", pd.Series(0, index=group.index)
            ).sum(),
        })
    return pd.DataFrame(records)


def load_reference(args, problem_ids):
    if not args.reference_csv:
        return pd.DataFrame()
    reference = pd.read_csv(args.reference_csv)
    required = {
        "dataset",
        "problem_id",
        "output_tokens",
        "accepted_tokens",
        "drafted_tokens",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "output_token_hash",
    }
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"Reference CSV is missing columns: {missing}")
    reference = reference[reference["dataset"].isin(args.datasets)].copy()
    for dataset in args.datasets:
        actual_ids = sorted(
            reference.loc[reference["dataset"] == dataset, "problem_id"]
            .astype(int)
            .tolist()
        )
        if actual_ids != problem_ids[dataset]:
            raise ValueError(
                f"Reference problem IDs do not match {dataset}: "
                f"expected={problem_ids[dataset]} actual={actual_ids}"
            )
    if "actual_measured_time" in reference:
        reference["measured_time_s"] = pd.to_numeric(
            reference["actual_measured_time"], errors="coerce"
        )
    else:
        reference["measured_time_s"] = (
            pd.to_numeric(reference["actual_draft_time"], errors="coerce")
            + pd.to_numeric(reference["actual_verify_time"], errors="coerce")
            + pd.to_numeric(reference["actual_post_verify_time"], errors="coerce")
        )
    reference["method"] = "failfast_spec8_reference"
    reference["measured_ms_per_output_token"] = (
        1000.0 * reference["measured_time_s"] / reference["output_tokens"]
    )
    reference["bucket_stop_actions"] = 0
    return reference


def paired_comparison(rows, candidate, baseline):
    columns = [
        "dataset",
        "problem_id",
        "method",
        "measured_ms_per_output_token",
        "output_token_hash",
    ]
    pivot = rows[columns].pivot(index=["dataset", "problem_id"], columns="method")
    candidate_ms = pivot[("measured_ms_per_output_token", candidate)]
    baseline_ms = pivot[("measured_ms_per_output_token", baseline)]
    result = pd.DataFrame({
        "dataset": pivot.index.get_level_values("dataset"),
        "problem_id": pivot.index.get_level_values("problem_id"),
        "candidate": candidate,
        "baseline": baseline,
        "speedup": baseline_ms.to_numpy() / candidate_ms.to_numpy(),
        "candidate_wins": (candidate_ms < baseline_ms).astype(int).to_numpy(),
        "output_match": (
            pivot[("output_token_hash", candidate)]
            == pivot[("output_token_hash", baseline)]
        ).astype(int).to_numpy(),
    })
    return result


def calibration_summary(output_dir, datasets):
    frames = []
    for dataset in datasets:
        path = (
            output_dir / "raw" / dataset / "bucket_renewal_spec8"
            / "frontier_round_diagnostics.csv"
        )
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset"] = dataset
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    observations = pd.concat(frames, ignore_index=True, sort=False)
    observations["predicted_accepted_tokens"] = pd.to_numeric(
        observations["predicted_accepted_tokens"], errors="coerce"
    )
    observations["actual_accepted_tokens"] = pd.to_numeric(
        observations["actual_accepted_tokens"], errors="coerce"
    )
    observations = observations.dropna(
        subset=["predicted_accepted_tokens", "actual_accepted_tokens"]
    )
    records = []
    for dataset, group in observations.groupby("dataset", sort=False):
        error = group["predicted_accepted_tokens"] - group["actual_accepted_tokens"]
        correlation = group[["predicted_accepted_tokens", "actual_accepted_tokens"]].corr().iloc[0, 1]
        records.append({
            "dataset": dataset,
            "num_rounds": len(group),
            "predicted_mean": group["predicted_accepted_tokens"].mean(),
            "actual_mean": group["actual_accepted_tokens"].mean(),
            "bias": error.mean(),
            "mae": error.abs().mean(),
            "rmse": np.sqrt(np.mean(np.square(error))),
            "pearson_correlation": correlation,
        })
    return observations, pd.DataFrame(records)


def gain_calibration_summary(output_dir, datasets):
    frames = []
    for dataset in datasets:
        path = (
            output_dir / "raw" / dataset / "bucket_renewal_spec8"
            / "frontier_gain_diagnostics.csv"
        )
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset"] = dataset
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    observations = pd.concat(frames, ignore_index=True, sort=False)
    observations = observations[observations["same_target_len"].astype(int) == 1].copy()
    observations["predicted_next_gain"] = pd.to_numeric(
        observations["predicted_next_gain"], errors="coerce"
    )
    observations["actual_next_gain"] = pd.to_numeric(
        observations["actual_next_gain"], errors="coerce"
    )
    observations = observations.dropna(subset=["predicted_next_gain", "actual_next_gain"])
    records = []
    for (dataset, from_step), group in observations.groupby(
        ["dataset", "from_step"], sort=False
    ):
        error = group["predicted_next_gain"] - group["actual_next_gain"]
        correlation = group[["predicted_next_gain", "actual_next_gain"]].corr().iloc[0, 1]
        records.append({
            "dataset": dataset,
            "from_step": int(from_step),
            "num_transitions": len(group),
            "predicted_gain_mean": group["predicted_next_gain"].mean(),
            "actual_gain_mean": group["actual_next_gain"].mean(),
            "gain_bias": error.mean(),
            "gain_mae": error.abs().mean(),
            "gain_rmse": np.sqrt(np.mean(np.square(error))),
            "gain_pearson_correlation": correlation,
        })
    return observations, pd.DataFrame(records)


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
        "benchmark_version": "bucket_renewal_v3_step_buckets",
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "problem_ids": problem_ids,
        "methods": METHODS,
        "reference_csv": args.reference_csv,
        "primary_metric": "draft + verify + post-verify latency per output token",
        "decoding": "greedy",
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
    for dataset in args.datasets:
        for method in METHODS:
            frames.append(run_method(args, dataset, method, problem_ids[dataset]))
    candidate_rows = pd.concat(frames, ignore_index=True, sort=False)
    reference_rows = load_reference(args, problem_ids)
    rows = (
        pd.concat([candidate_rows, reference_rows], ignore_index=True, sort=False)
        if not reference_rows.empty
        else candidate_rows
    )
    dataset_summary = aggregate(rows)
    if reference_rows.empty:
        paired = pd.DataFrame()
        paired_summary = pd.DataFrame()
    else:
        paired = paired_comparison(
            rows,
            "bucket_renewal_spec8",
            "failfast_spec8_reference",
        )
        paired_summary = paired.groupby(["candidate", "baseline"], as_index=False).agg(
            num_pairs=("speedup", "size"),
            geometric_mean_speedup=("speedup", lambda values: float(np.exp(np.log(values).mean()))),
            win_rate_percent=("candidate_wins", lambda values: 100.0 * values.mean()),
            output_match_rate_percent=("output_match", lambda values: 100.0 * values.mean()),
        )
    calibration_rows, calibration = calibration_summary(output_dir, args.datasets)
    gain_rows, gain_calibration = gain_calibration_summary(output_dir, args.datasets)

    candidate_rows.to_csv(output_dir / "per_observation.csv", index=False)
    if not reference_rows.empty:
        rows.to_csv(output_dir / "combined_per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_observations.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)
    calibration_rows.to_csv(output_dir / "acceptance_calibration_observations.csv", index=False)
    calibration.to_csv(output_dir / "acceptance_calibration_summary.csv", index=False)
    gain_rows.to_csv(output_dir / "gain_calibration_observations.csv", index=False)
    gain_calibration.to_csv(output_dir / "gain_calibration_summary.csv", index=False)
    write_manifest(args, output_dir, problem_ids)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )

    print("\nDATASET SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nPAIRED SUMMARY")
    print(paired_summary.to_string(index=False))
    print("\nACCEPTANCE CALIBRATION")
    print(calibration.to_string(index=False))
    print("\nREFINEMENT GAIN CALIBRATION")
    print(gain_calibration.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
