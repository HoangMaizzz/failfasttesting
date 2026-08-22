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


DATASET_SIZES = {
    "math": 500,
    "aime": 30,
    "gsm8k": 1319,
    "humaneval": 164,
}
BENCHMARK_VERSION = "bucket_renewal_oracle_validation_v1"
METHOD = "bucket_renewal_shadow_oracle"


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
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--bucket_prior_strength", type=float, default=8.0)
    parser.add_argument("--bucket_min_observations", type=int, default=8)
    parser.add_argument("--bucket_latency_ema_alpha", type=float, default=0.2)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_bucket_renewal_oracle_test15",
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
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_metadata(args, dataset, problem_ids):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "problem_ids": problem_ids,
        "method": METHOD,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "resume", "datasets"}
        },
    }


def result_complete(result_path, snapshot_path, metadata_path, expected_metadata):
    if not result_path.exists() or not snapshot_path.exists() or not metadata_path.exists():
        return False
    try:
        results = pd.read_csv(result_path)
        snapshots = pd.read_csv(snapshot_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    expected_ids = expected_metadata["problem_ids"]
    return (
        metadata == expected_metadata
        and sorted(results["problem_id"].astype(int).tolist()) == expected_ids
        and set(snapshots["problem_id"].astype(int)).issubset(set(expected_ids))
    )


def run_dataset(args, dataset, problem_ids):
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) / "raw" / dataset / METHOD
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    snapshot_path = output_dir / "bucket_oracle_snapshots.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata(args, dataset, problem_ids)
    if args.resume and result_complete(
        result_path,
        snapshot_path,
        metadata_path,
        metadata,
    ):
        print(f"RESUME {dataset} | {METHOD}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "bucket_oracle_snapshots.csv",
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
            "--spec_len", str(args.spec_len),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(args.lowconf_threshold),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(args.incr_len),
            "--frontier_stop_mode", "bucket_renewal",
            "--bucket_renewal_min_steps", "1",
            "--bucket_prior_strength", str(args.bucket_prior_strength),
            "--bucket_min_observations", str(args.bucket_min_observations),
            "--bucket_latency_ema_alpha", str(args.bucket_latency_ema_alpha),
            "--collect_bucket_oracle",
            "--bucket_oracle_force_continue",
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
            f"RUN {dataset} | samples={len(problem_ids)} | ids={problem_ids}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, root)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = pd.read_csv(result_path)
    snapshots = pd.read_csv(snapshot_path)
    results["dataset"] = dataset
    snapshots["dataset"] = dataset
    return results, snapshots


def to_optional_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def build_oracle_transitions(snapshots):
    records = []
    group_columns = ["dataset", "problem_id", "round_id", "target_len"]
    for keys, group in snapshots.groupby(group_columns, sort=False):
        group = group.sort_values(["step", "draft_passes_elapsed"]).reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]
            if int(following["step"]) != int(current["step"]) + 1:
                continue
            draft_pass_delta = int(following["draft_passes_elapsed"]) - int(
                current["draft_passes_elapsed"]
            )
            if draft_pass_delta != 1:
                continue
            predicted_gain = current["predicted_next_gain"]
            predicted_stop = current["predicted_stop_ms_per_output"]
            predicted_continue = current["predicted_continue_ms_per_output"]
            predicted_action = to_optional_bool(current["predicted_should_continue"])
            actual_current_output = float(current["emitted_len_if_stop"])
            actual_next_output = float(following["emitted_len_if_stop"])
            actual_gain = actual_next_output - actual_current_output
            actual_stop = (
                float(current["draft_latency_elapsed_ms"])
                + float(current["actual_verify_latency_ms"])
                + float(current["actual_post_verify_latency_ms"])
            ) / max(actual_current_output, 1e-9)
            actual_continue = (
                float(following["draft_latency_elapsed_ms"])
                + float(following["actual_verify_latency_ms"])
                + float(following["actual_post_verify_latency_ms"])
            ) / max(actual_next_output, 1e-9)
            oracle_continue = actual_continue < actual_stop
            selected_cost = (
                actual_continue if predicted_action else actual_stop
                if predicted_action is not None else np.nan
            )
            record = dict(zip(group_columns, keys))
            record.update({
                "from_step": int(current["step"]),
                "to_step": int(following["step"]),
                "draft_pass_delta": draft_pass_delta,
                "predicted_current_output": current["predicted_expected_output"],
                "actual_current_output": actual_current_output,
                "predicted_next_gain": predicted_gain,
                "actual_next_gain": actual_gain,
                "gain_error": (
                    float(predicted_gain) - actual_gain
                    if pd.notna(predicted_gain)
                    else np.nan
                ),
                "predicted_next_output": (
                    float(current["predicted_expected_output"]) + float(predicted_gain)
                    if pd.notna(current["predicted_expected_output"])
                    and pd.notna(predicted_gain)
                    else np.nan
                ),
                "actual_next_output": actual_next_output,
                "predicted_stop_ms_per_output": predicted_stop,
                "actual_stop_ms_per_output": actual_stop,
                "stop_cost_error": (
                    float(predicted_stop) - actual_stop
                    if pd.notna(predicted_stop)
                    else np.nan
                ),
                "predicted_continue_ms_per_output": predicted_continue,
                "actual_continue_ms_per_output": actual_continue,
                "continue_cost_error": (
                    float(predicted_continue) - actual_continue
                    if pd.notna(predicted_continue)
                    else np.nan
                ),
                "predicted_action": (
                    "continue" if predicted_action else "stop"
                    if predicted_action is not None else "bootstrap"
                ),
                "oracle_action": "continue" if oracle_continue else "stop",
                "decision_correct": (
                    int(predicted_action == oracle_continue)
                    if predicted_action is not None else np.nan
                ),
                "decision_regret_ms_per_output": (
                    selected_cost - min(actual_stop, actual_continue)
                    if predicted_action is not None else np.nan
                ),
                "predicted_gain_source": current["predicted_gain_source"],
                "gain_bucket_count": current["gain_bucket_count"],
                "gain_bucket_weight": current["gain_bucket_weight"],
                "calibration_tokens": current["calibration_tokens"],
            })
            records.append(record)
    return pd.DataFrame(records)


def safe_correlation(left, right):
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2 or frame["left"].std() == 0 or frame["right"].std() == 0:
        return np.nan
    return float(frame["left"].corr(frame["right"]))


def summarize_transitions(transitions, group_columns):
    records = []
    for keys, group in transitions.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        gain = group.dropna(subset=["gain_error"])
        cost = group.dropna(
            subset=[
                "stop_cost_error",
                "continue_cost_error",
                "decision_correct",
            ]
        )
        predicted_continue = cost["predicted_action"].eq("continue")
        oracle_continue = cost["oracle_action"].eq("continue")
        tp = int((predicted_continue & oracle_continue).sum())
        fp = int((predicted_continue & ~oracle_continue).sum())
        fn = int((~predicted_continue & oracle_continue).sum())
        tn = int((~predicted_continue & ~oracle_continue).sum())
        record = dict(zip(group_columns, keys))
        record.update({
            "transitions": len(group),
            "gain_evaluated": len(gain),
            "decision_evaluated": len(cost),
            "predicted_gain_mean": gain["predicted_next_gain"].mean(),
            "actual_gain_mean": gain["actual_next_gain"].mean(),
            "gain_bias": gain["gain_error"].mean(),
            "gain_mae": gain["gain_error"].abs().mean(),
            "gain_rmse": math.sqrt((gain["gain_error"] ** 2).mean()) if len(gain) else np.nan,
            "gain_pearson": safe_correlation(
                gain["predicted_next_gain"],
                gain["actual_next_gain"],
            ),
            "current_output_mae": (
                gain["predicted_current_output"] - gain["actual_current_output"]
            ).abs().mean(),
            "next_output_mae": (
                gain["predicted_next_output"] - gain["actual_next_output"]
            ).abs().mean(),
            "stop_cost_bias_ms_per_output": cost["stop_cost_error"].mean(),
            "stop_cost_mae_ms_per_output": cost["stop_cost_error"].abs().mean(),
            "continue_cost_bias_ms_per_output": cost["continue_cost_error"].mean(),
            "continue_cost_mae_ms_per_output": cost["continue_cost_error"].abs().mean(),
            "predicted_continue_rate_percent": 100.0 * predicted_continue.mean(),
            "oracle_continue_rate_percent": 100.0 * oracle_continue.mean(),
            "decision_accuracy_percent": 100.0 * cost["decision_correct"].mean(),
            "continue_true_positive": tp,
            "continue_false_positive": fp,
            "continue_false_negative": fn,
            "stop_true_positive": tn,
            "continue_precision_percent": 100.0 * tp / (tp + fp) if tp + fp else np.nan,
            "continue_recall_percent": 100.0 * tp / (tp + fn) if tp + fn else np.nan,
            "stop_recall_percent": 100.0 * tn / (tn + fp) if tn + fp else np.nan,
            "mean_regret_ms_per_output": cost["decision_regret_ms_per_output"].mean(),
            "p95_regret_ms_per_output": cost["decision_regret_ms_per_output"].quantile(0.95),
        })
        records.append(record)
    return pd.DataFrame(records)


def gain_calibration_table(transitions):
    evaluated = transitions.dropna(subset=["predicted_next_gain"]).copy()
    bins = [-np.inf, 0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
    labels = ["<=0", "0-0.25", "0.25-0.5", "0.5-1", "1-2", ">2"]
    evaluated["predicted_gain_bin"] = pd.cut(
        evaluated["predicted_next_gain"], bins=bins, labels=labels
    )
    return evaluated.groupby(
        ["dataset", "predicted_gain_bin"],
        observed=True,
        dropna=False,
    ).agg(
        transitions=("actual_next_gain", "size"),
        predicted_gain_mean=("predicted_next_gain", "mean"),
        actual_gain_mean=("actual_next_gain", "mean"),
        gain_mae=("gain_error", lambda values: values.abs().mean()),
        positive_actual_gain_percent=("actual_next_gain", lambda values: 100.0 * (values > 0).mean()),
    ).reset_index()


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
        "timing_scope": (
            "Oracle verifier calls are diagnostic only and are subtracted from "
            "the method's reported post-verify and end-to-end latency."
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
    problem_ids = sampled_problem_ids(args)
    result_frames = []
    snapshot_frames = []
    for dataset in args.datasets:
        results, snapshots = run_dataset(args, dataset, problem_ids[dataset])
        result_frames.append(results)
        snapshot_frames.append(snapshots)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    snapshots = pd.concat(snapshot_frames, ignore_index=True, sort=False)
    transitions = build_oracle_transitions(snapshots)
    if transitions.empty:
        raise ValueError("No exact one-pass, same-length oracle transitions were collected")
    dataset_summary = summarize_transitions(transitions, ["dataset"])
    step_summary = summarize_transitions(transitions, ["dataset", "from_step"])
    overall_summary = summarize_transitions(
        transitions.assign(scope="all_datasets"),
        ["scope"],
    )
    gain_calibration = gain_calibration_table(transitions)

    results.to_csv(output_dir / "all_benchmark_results.csv", index=False)
    snapshots.to_csv(output_dir / "oracle_snapshots.csv", index=False)
    transitions.to_csv(output_dir / "oracle_transitions.csv", index=False)
    dataset_summary.to_csv(output_dir / "oracle_dataset_summary.csv", index=False)
    step_summary.to_csv(output_dir / "oracle_step_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "oracle_overall_summary.csv", index=False)
    gain_calibration.to_csv(output_dir / "gain_calibration.csv", index=False)
    write_manifest(args, output_dir, problem_ids)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )

    print("\nBUCKET RENEWAL ORACLE DATASET SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nBUCKET RENEWAL ORACLE OVERALL SUMMARY")
    print(overall_summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
