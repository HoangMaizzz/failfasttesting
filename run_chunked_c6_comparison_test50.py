import argparse
from collections import deque
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
DATASET_SIZES = {"math": 500, "gsm8k": 1319}


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


class RunFailure(subprocess.CalledProcessError):
    def __init__(self, returncode, command, output_tail):
        super().__init__(returncode, command, output="".join(output_tail))
        self.output_tail = list(output_tail)


def run_streaming(command):
    output_tail = deque(maxlen=200)
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line)
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RunFailure(code, command, output_tail)


def failure_category(error):
    text = "".join(error.output_tail).lower()
    if "outofmemoryerror" in text or "cuda out of memory" in text:
        return "cuda_oom"
    return "runtime_error"


def candidate_pool(dataset, target_count=50):
    preferred = [int(value) for value in PROBLEM_IDS[dataset][:target_count]]
    start = 1 if dataset == "gsm8k" else 0
    backups = [value for value in range(start, DATASET_SIZES[dataset]) if value not in set(preferred)]
    return preferred + backups


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


def run_fixed_method(args, dataset, method, ids):
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


def _write_selection_progress(path, dataset, candidates, next_index, successful, skipped):
    payload = {
        "dataset": dataset,
        "target_count": 50,
        "candidate_order": candidates,
        "next_candidate_index": next_index,
        "successful": successful,
        "selected_problem_ids": [row["problem_id"] for row in successful],
        "skipped": skipped,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_c6_selector(args, dataset, target_count=50):
    """Run C6 one question per process and freeze the first target_count successes."""
    root = Path(args.output_dir) / "selector" / dataset
    progress_path = root / "selection_progress.json"
    candidates = candidate_pool(dataset, target_count)
    successful = []
    skipped = []
    next_index = 0
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        successful = progress.get("successful", [])
        skipped = progress.get("skipped", [])
        next_index = int(progress.get("next_candidate_index", 0))
        valid = []
        for row in successful:
            directory = Path(row["directory"])
            if complete(directory, [int(row["problem_id"])], True):
                valid.append(row)
            else:
                break
        if len(valid) != len(successful):
            successful = valid
            attempted = {row["problem_id"] for row in successful + skipped}
            next_index = next((i for i, value in enumerate(candidates) if value not in attempted), len(candidates))

    while len(successful) < target_count:
        if next_index >= len(candidates):
            raise RuntimeError(f"candidate pool exhausted for {dataset}")
        problem_id = candidates[next_index]
        attempt = next_index
        next_index += 1
        directory = root / f"attempt_{attempt:04d}_id_{problem_id}"
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        prior_state = None
        if successful:
            prior_state = Path(successful[-1]["directory"]) / "adaptive_td_runtime_state.json"
        command = base_command(
            args, dataset, [problem_id], directory, 1 if not successful else 0,
        )
        command.extend(adaptive_flags("c6_annealed", prior_state))
        print("\n" + "=" * 96, flush=True)
        print(
            f"SELECT {dataset.upper()} | C6 | success={len(successful)}/{target_count} "
            f"| candidate={problem_id}", flush=True,
        )
        print("=" * 96, flush=True)
        try:
            run_streaming(command)
            if not complete(directory, [problem_id], True):
                raise RuntimeError(f"incomplete selector result: {directory}")
        except RunFailure as error:
            category = failure_category(error)
            failure_log = directory / "failure.log"
            failure_log.write_text("".join(error.output_tail), encoding="utf-8")
            skipped.append({
                "problem_id": problem_id,
                "candidate_index": attempt,
                "category": category,
                "returncode": error.returncode,
                "failure_log": str(failure_log),
            })
            print(f"SKIP {dataset} id={problem_id}: {category}; trying next candidate", flush=True)
            _write_selection_progress(
                progress_path, dataset, candidates, next_index, successful, skipped,
            )
            continue
        successful.append({
            "problem_id": problem_id,
            "candidate_index": attempt,
            "directory": str(directory.resolve()),
        })
        _write_selection_progress(
            progress_path, dataset, candidates, next_index, successful, skipped,
        )

    selected_ids = [int(row["problem_id"]) for row in successful]
    selected_dirs = [Path(row["directory"]) for row in successful]
    (root / "selected_problem_ids.json").write_text(
        json.dumps(selected_ids, indent=2), encoding="utf-8",
    )
    return selected_dirs, selected_ids, skipped


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
    selected = {}
    skipped = {}
    # C6 is the hardware-feasibility selector. Its successful IDs are frozen
    # before either control method starts, preserving paired comparisons.
    for dataset in DATASETS:
        runs[(dataset, "c6_annealed")], selected[dataset], skipped[dataset] = run_c6_selector(args, dataset)
    selected_path = Path(args.output_dir) / "selected_problem_ids.json"
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    skipped_rows = [dict(dataset=dataset, **row) for dataset, rows in skipped.items() for row in rows]
    pd.DataFrame(skipped_rows).to_csv(Path(args.output_dir) / "skipped_selector_questions.csv", index=False)
    for method in ("always_stop", "failfast"):
        for dataset in DATASETS:
            runs[(dataset, method)] = run_fixed_method(args, dataset, method, selected[dataset])
    summary = summarize(args, runs)
    manifest = {
        "version": "resilient_paired_c6_annealed_controls_test50_v2",
        "datasets": list(DATASETS), "methods": list(METHODS),
        "problem_ids": selected,
        "skipped_selector_questions": skipped,
        "selection_protocol": (
            "C6 runs one candidate per fresh process; failed candidates are logged and replaced "
            "deterministically, then the successful ID set is frozen for both controls"
        ),
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
