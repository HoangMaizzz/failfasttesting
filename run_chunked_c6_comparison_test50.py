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
DATASETS = ("math", "gsm8k")
METHODS = ("c6_annealed", "always_stop", "failfast")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=2)
    parser.add_argument("--target_quantization", choices=("int8", "int4"), default="int8")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument("--dllm_dir", default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--output_dir", default="/home/maihoang/failfasttesting/outputs_chunked_c6_annealed_failfast_stop_int8_test50")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def chunks(values, size):
    for start in range(0, len(values), size):
        yield start // size, values[start:start + size]


def run_streaming(command):
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def complete(directory, problem_ids, adaptive):
    result = directory / "benchmark_results.csv"
    if not result.exists() or not result.stat().st_size:
        return False
    try:
        frame = pd.read_csv(result)
    except (OSError, pd.errors.EmptyDataError):
        return False
    if set(frame.problem_id.astype(int)) != set(problem_ids):
        return False
    return not adaptive or (directory / "adaptive_td_runtime_state.json").exists()


def base_command(args, dataset, problem_ids, output_dir, warmup):
    return [
        sys.executable, "-u", "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *map(str, problem_ids),
        "--warmup_questions", str(warmup),
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", "1024",
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--drafter_thresholds", "0.3",
        "--sweep_lowconf_threshold", "0.5",
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--seed", "42",
        "--log_verifier_calls",
        "--quiet_generation", "--disable_progress", "--skip_artifacts",
        "--skip_plots", "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def adaptive_flags(method, state_path=None):
    flags = [
        "--adaptive-td", "--adaptive-controller", "avg_td",
        "--adaptive-feature-schema", "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment", "verifier_boundary_factual_no_bootstrap",
        "--adaptive-learning-rate", "0.02",
        "--adaptive-value-parameterization", "shared_value_advantage",
        "--adaptive-shared-value-learning-rate", "0.015",
        "--adaptive-shared-advantage-learning-rate", "0.02",
        "--adaptive-mc-learning-rate", "0.01", "--adaptive-mc-mix", "0.5",
        "--adaptive-update-mode", "mixed", "--adaptive-rho-alpha", "0.05",
        "--adaptive-rho-warmup-boundaries", "0",
        "--adaptive-policy-weight-ema-beta", "0.0",
        "--adaptive-policy-weight-ema-mode", "global_step",
        "--adaptive-factual-ema-alpha", "0.2", "--adaptive-risk-beta", "1.0",
        "--adaptive-stop-probability-threshold", "0.75",
        "--adaptive-uncertainty-prior", "1.0", "--adaptive-epistemic-scale", "0.1",
        "--adaptive-q-margin", "0.0", "--adaptive-explore-epsilon", "0.1",
        "--adaptive-explore-min", "0.02", "--adaptive-explore-decay", "0.998",
        "--adaptive-warmup-rounds", "20",
        "--adaptive-early-stop-min-observations", "32",
        "--adaptive-policy-mode", "symmetric_annealed",
        "--adaptive-policy-ablation", "frozen_stop" if method == "always_stop" else "learned",
        "--adaptive-min-action-probability", "0.1",
        "--adaptive-max-importance-weight", "5.0",
        "--adaptive-weight-snapshot-interval", "100",
        "--adaptive-log-decisions", "--adaptive-profile-overhead",
    ]
    if method == "always_stop":
        flags.append("--adaptive-freeze")
    if state_path is not None:
        flags.extend(["--adaptive-state-path", str(state_path)])
    return flags


def run_method(args, dataset, method):
    ids = [int(value) for value in PROBLEM_IDS[dataset][:50]]
    adaptive = method != "failfast"
    chunk_dirs = []
    prior_state = None
    for index, problem_ids in chunks(ids, args.chunk_size):
        directory = Path(args.output_dir) / "chunks" / dataset / method / f"chunk_{index:03d}"
        if args.resume and complete(directory, problem_ids, adaptive):
            print(f"SKIP {dataset} {method} chunk={index}", flush=True)
        else:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
            command = base_command(args, dataset, problem_ids, directory, 1 if index == 0 else 0)
            if adaptive:
                command.extend(adaptive_flags(method, prior_state))
            print("\n" + "=" * 96, flush=True)
            print(f"RUN {dataset.upper()} | {method} | chunk={index} | ids={problem_ids}", flush=True)
            print("=" * 96, flush=True)
            run_streaming(command)
            if not complete(directory, problem_ids, adaptive):
                raise RuntimeError(f"incomplete chunk: {directory}")
        chunk_dirs.append(directory)
        if adaptive:
            prior_state = directory / "adaptive_td_runtime_state.json"
    return chunk_dirs


def combine(paths, destination):
    frames = [pd.read_csv(path) for path in paths if path.exists() and path.stat().st_size]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    return frame


def summarize(args, runs):
    reports = []
    paired_frames = []
    for dataset in DATASETS:
        method_results = {}
        for method in METHODS:
            dirs = runs[(dataset, method)]
            method_results[method] = combine(
                [directory / "benchmark_results.csv" for directory in dirs],
                Path(args.output_dir) / "combined" / dataset / f"{method}_benchmark_results.csv",
            )
            combine(
                [directory / "verifier_calls.csv" for directory in dirs],
                Path(args.output_dir) / "combined" / dataset / f"{method}_verifier_calls.csv",
            )
            if method != "failfast":
                combine(
                    [directory / "adaptive_td_decisions.csv" for directory in dirs],
                    Path(args.output_dir) / "combined" / dataset / f"{method}_decisions.csv",
                )
        aggregates = {method: aggregate_method(frame, method) for method, frame in method_results.items()}
        for method, row in aggregates.items():
            row = dict(row)
            row["dataset"] = dataset
            row["speedup_vs_failfast"] = aggregates["failfast"]["ms_per_output_token"] / row["ms_per_output_token"]
            reports.append(row)

        columns = ["problem_id", "actual_algorithm_time", "output_tokens", "output_token_hash"]
        paired = None
        for method, frame in method_results.items():
            item = frame[columns].copy().rename(columns={c: f"{method}_{c}" for c in columns if c != "problem_id"})
            paired = item if paired is None else paired.merge(item, on="problem_id", validate="one_to_one")
        paired.insert(0, "dataset", dataset)
        for method in METHODS:
            paired[f"{method}_ms_per_output_token"] = 1000.0 * paired[f"{method}_actual_algorithm_time"] / paired[f"{method}_output_tokens"].clip(lower=1)
            paired[f"{method}_speedup_vs_failfast"] = paired["failfast_ms_per_output_token"] / paired[f"{method}_ms_per_output_token"]
            paired[f"{method}_output_matches_failfast"] = paired[f"{method}_output_token_hash"].astype(str) == paired["failfast_output_token_hash"].astype(str)
        paired_frames.append(paired)
    summary = pd.DataFrame(reports)
    paired = pd.concat(paired_frames, ignore_index=True)
    summary.to_csv(Path(args.output_dir) / "all_methods_dataset_summary.csv", index=False)
    paired.to_csv(Path(args.output_dir) / "paired_problem_comparison.csv", index=False)
    return summary


def main():
    args = parse_args()
    if args.chunk_size <= 0 or args.chunk_size > 25:
        raise ValueError("--chunk_size must be in [1, 25]")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = {}
    for method in METHODS:
        for dataset in DATASETS:
            runs[(dataset, method)] = run_method(args, dataset, method)
    summary = summarize(args, runs)
    manifest = {
        "version": "chunked_c6_annealed_controls_test50_v1",
        "datasets": list(DATASETS), "methods": list(METHODS),
        "problem_ids": {dataset: PROBLEM_IDS[dataset][:50] for dataset in DATASETS},
        "chunk_size": args.chunk_size,
        "state_continuity": "weights, uncertainty, EMA, rho, counters, and RNG state",
        "elapsed_hours": (time.time() - started) / 3600.0,
        "arguments": vars(args),
    }
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nALL METHODS DATASET SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    if not args.skip_archive:
        archive = shutil.make_archive(str(Path(args.output_dir)), "zip", root_dir=Path(args.output_dir))
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
