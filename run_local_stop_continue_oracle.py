import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS, aggregate_method


ROOT = Path(__file__).resolve().parent
DATASET_COUNTS = {
    "math": 50,
    "gsm8k": 50,
    "humaneval": 50,
    "aime": 25,
}
RESULTS_FILE = "benchmark_results.csv"
CALLS_FILE = "verifier_calls.csv"
DECISIONS_FILE = "greedy_local_oracle_decisions.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Factual greedy local STOP/CONTINUE oracle. At each decision it "
            "measures both current actions through the next verifier boundary."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_COUNTS, default=list(DATASET_COUNTS))
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--drafter_threshold", type=float, default=0.30)
    parser.add_argument("--lowconf_threshold", type=float, default=0.50)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("none", "int8", "int4"),
        default="int8",
    )
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_local_stop_continue_oracle_175"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")
    if args.target_device < 0 or args.drafter_device < 0:
        raise ValueError("CUDA device indices must be non-negative")
    for dataset in args.datasets:
        if len(PROBLEM_IDS[dataset]) < DATASET_COUNTS[dataset]:
            raise ValueError(f"{dataset} does not provide enough matched IDs")


def selected_problem_ids(dataset):
    return [int(value) for value in PROBLEM_IDS[dataset][:DATASET_COUNTS[dataset]]]


def chunks(values, size):
    for start in range(0, len(values), size):
        yield start // size, values[start:start + size]


def common_command(args, dataset, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(value) for value in problem_ids],
        "--warmup_questions", "0",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--frontier_stop_mode", "disabled",
        "--disable_reusing_drafter_kvs",
        "--seed", "42",
        "--log_verifier_calls",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


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


def csv_has_ids(path, expected_ids):
    if not path.exists() or not path.stat().st_size:
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return False
    return set(pd.to_numeric(frame["problem_id"]).astype(int)) == set(expected_ids)


def run_baseline_profile(args, dataset, problem_ids):
    output_dir = Path(args.output_dir) / "raw" / dataset / "failfast_profile"
    if args.resume and csv_has_ids(output_dir / RESULTS_FILE, problem_ids):
        print(f"SKIP completed FailFast profile: {dataset}", flush=True)
        return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print("\n" + "=" * 96, flush=True)
    print(f"PROFILE + BASELINE {dataset.upper()} | questions={len(problem_ids)}", flush=True)
    print("=" * 96, flush=True)
    run_streaming(common_command(args, dataset, problem_ids, output_dir))
    if not csv_has_ids(output_dir / RESULTS_FILE, problem_ids):
        raise RuntimeError(f"incomplete FailFast profile for {dataset}")
    return output_dir


def build_verifier_profile(profile_dir, destination):
    calls = pd.read_csv(Path(profile_dir) / CALLS_FILE)
    if calls.empty:
        raise ValueError("FailFast profile has no verifier calls")
    calls = calls.copy()
    calls["context_bucket"] = (
        pd.to_numeric(calls["context_length"]).astype(int) // 256
    )
    calls["proposal_bucket"] = (
        pd.to_numeric(calls["proposal_length"]).astype(int).map(
            lambda value: max(1, math.ceil(value / 8))
        )
    )
    bins = (
        calls.groupby(["context_bucket", "proposal_bucket"], as_index=False)
        .agg(
            observations=("verify_latency_ms", "size"),
            mean_verify_latency_ms=("verify_latency_ms", "mean"),
        )
        .to_dict("records")
    )
    profile = {
        "version": "frozen_failfast_hardware_profile_v2",
        "mean_verify_latency_ms": float(calls["verify_latency_ms"].mean()),
        "mean_tokens_per_verify": float(calls["emitted_tokens"].mean()),
        "verifier_calls": int(len(calls)),
        "context_bucket_size": 256,
        "proposal_bucket_size": 8,
        "latency_bins": bins,
        "use": (
            "Current STOP/CONTINUE branches use measured verifier latency. "
            "This frozen profile is used only to price future verifier calls "
            "caused by a branch emitting fewer tokens."
        ),
    }
    destination.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def run_oracle_batches(args, dataset, problem_ids, profile_path):
    batch_dirs = []
    for batch_index, batch_ids in chunks(problem_ids, args.batch_size):
        output_dir = (
            Path(args.output_dir)
            / "raw"
            / dataset
            / "local_oracle"
            / f"batch_{batch_index:03d}"
        )
        complete = (
            csv_has_ids(output_dir / RESULTS_FILE, batch_ids)
            and (output_dir / DECISIONS_FILE).exists()
        )
        if args.resume and complete:
            print(f"SKIP completed oracle batch: {dataset} {batch_index}", flush=True)
            batch_dirs.append(output_dir)
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        command = common_command(args, dataset, batch_ids, output_dir)
        command.extend([
            "--strict_greedy_local_oracle",
            "--strict_greedy_verifier_profile", str(profile_path),
            "--strict_greedy_epsilon_ms", str(args.epsilon_ms),
        ])
        print("\n" + "=" * 96, flush=True)
        print(
            f"LOCAL ORACLE {dataset.upper()} | batch={batch_index} | "
            f"problem_ids={batch_ids}",
            flush=True,
        )
        print("=" * 96, flush=True)
        run_streaming(command)
        if not csv_has_ids(output_dir / RESULTS_FILE, batch_ids):
            raise RuntimeError(f"incomplete local oracle batch: {output_dir}")
        batch_dirs.append(output_dir)
    return batch_dirs


def combine_csv(paths, destination):
    frames = [pd.read_csv(path) for path in paths if path.exists() and path.stat().st_size]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(destination, index=False)
    return combined


def summarize_dataset(args, dataset, problem_ids, baseline_dir, oracle_dirs, profile):
    dataset_dir = Path(args.output_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(Path(baseline_dir) / RESULTS_FILE)
    oracle = combine_csv(
        [path / RESULTS_FILE for path in oracle_dirs],
        dataset_dir / "local_oracle_results.csv",
    )
    decisions = combine_csv(
        [path / DECISIONS_FILE for path in oracle_dirs],
        dataset_dir / "local_oracle_decisions.csv",
    )
    calls = combine_csv(
        [path / CALLS_FILE for path in oracle_dirs],
        dataset_dir / "local_oracle_verifier_calls.csv",
    )
    if set(oracle["problem_id"].astype(int)) != set(problem_ids):
        raise RuntimeError(f"combined local oracle IDs do not match {dataset}")
    baseline_agg = aggregate_method(baseline, "failfast")
    oracle_agg = aggregate_method(oracle, "local_oracle")
    report = {
        "dataset": dataset,
        "num_questions": len(problem_ids),
        "failfast_ms_per_output_token": baseline_agg["ms_per_output_token"],
        "local_oracle_ms_per_output_token": oracle_agg["ms_per_output_token"],
        "local_oracle_speedup_vs_failfast": (
            baseline_agg["ms_per_output_token"] / oracle_agg["ms_per_output_token"]
        ),
        "failfast_algorithm_time_s": baseline_agg["algorithm_time_s"],
        "local_oracle_algorithm_time_s": oracle_agg["algorithm_time_s"],
        "failfast_draft_passes": baseline_agg["draft_passes"],
        "local_oracle_draft_passes": oracle_agg["draft_passes"],
        "failfast_verifier_rounds": baseline_agg["verifier_rounds"],
        "local_oracle_verifier_rounds": oracle_agg["verifier_rounds"],
        "oracle_decisions": int(len(decisions)),
        "oracle_stop_rate_percent": (
            100.0 * decisions["chosen_action"].eq("stop").mean()
            if not decisions.empty else 0.0
        ),
        "near_tie_rate_percent": (
            100.0 * (decisions["DeltaJ_ms"].abs() <= args.epsilon_ms).mean()
            if not decisions.empty else 0.0
        ),
        "output_match_rate_percent": 100.0 * oracle.merge(
            baseline[["problem_id", "output_token_hash"]],
            on="problem_id",
            suffixes=("_oracle", "_failfast"),
        ).eval("output_token_hash_oracle == output_token_hash_failfast").mean(),
        "profile_mean_verify_latency_ms": profile["mean_verify_latency_ms"],
        "profile_mean_tokens_per_verify": profile["mean_tokens_per_verify"],
        "oracle_measured_verify_latency_ms": float(calls["verify_latency_ms"].mean()),
    }
    pd.DataFrame([baseline_agg, oracle_agg]).to_csv(
        dataset_dir / "method_summary.csv", index=False
    )
    pd.DataFrame([report]).to_csv(dataset_dir / "dataset_comparison.csv", index=False)
    paired = baseline[["problem_id", "actual_algorithm_time", "output_token_hash"]].merge(
        oracle[["problem_id", "actual_algorithm_time", "output_token_hash"]],
        on="problem_id",
        suffixes=("_failfast", "_oracle"),
    )
    paired["speedup_oracle_vs_failfast"] = (
        paired["actual_algorithm_time_failfast"]
        / paired["actual_algorithm_time_oracle"]
    )
    paired["output_match"] = (
        paired["output_token_hash_failfast"] == paired["output_token_hash_oracle"]
    )
    paired.to_csv(dataset_dir / "paired_problem_comparison.csv", index=False)
    return report


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reports = []
    manifest_ids = {}
    for dataset in args.datasets:
        problem_ids = selected_problem_ids(dataset)
        manifest_ids[dataset] = problem_ids
        baseline_dir = run_baseline_profile(args, dataset, problem_ids)
        profile_path = output_dir / dataset / "verifier_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile = build_verifier_profile(baseline_dir, profile_path)
        oracle_dirs = run_oracle_batches(args, dataset, problem_ids, profile_path)
        reports.append(
            summarize_dataset(
                args, dataset, problem_ids, baseline_dir, oracle_dirs, profile
            )
        )
    combined = pd.DataFrame(reports)
    combined.to_csv(output_dir / "local_oracle_dataset_summary.csv", index=False)
    manifest = {
        "version": "factual_greedy_local_stop_continue_oracle_v1",
        "problem_ids": manifest_ids,
        "arguments": vars(args),
        "oracle_definition": (
            "At every legal refinement decision, measure STOP and CONTINUE "
            "through the next real greedy verifier boundary. Later decisions "
            "inside each probe follow baseline FailFast CONTINUE. Choose the "
            "lower measured local compute cost after adding the frozen-profile "
            "cost of future verifier calls implied by emitted-token deficit."
        ),
        "is_global_oracle": False,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nLOCAL ORACLE DATASET SUMMARY", flush=True)
    print(combined.to_string(index=False), flush=True)
    if not args.skip_archive:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
