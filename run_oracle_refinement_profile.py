import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_adaptive_unmask_only_benchmark import DATASET_SIZES
from run_cost_aware_v2_benchmark import run_streaming


BENCHMARK_VERSION = "oracle_refinement_profile_v4"
METHOD_NAME = "failfast_oracle_profile"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SIZES),
        default=list(DATASET_SIZES),
    )
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_oracle_refinement_profile_test10",
    )
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
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
    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        population = list(range(args.warmup_questions, DATASET_SIZES[dataset]))
        rng = random.Random(args.sample_seed + 1009 * dataset_index)
        result[dataset] = sorted(rng.sample(population, args.num_questions))
    return result


def run_metadata(args, dataset, problem_ids):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": dataset,
        "problem_ids": problem_ids,
        "warmup_questions": args.warmup_questions,
        "max_new_tokens": args.max_new_tokens,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "spec_len": args.spec_len,
        "incr_len": args.incr_len,
        "drafter_threshold": args.drafter_threshold,
        "lowconf_threshold": args.lowconf_threshold,
        "max_spec_len": args.max_spec_len,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "method": METHOD_NAME,
    }


def results_complete(result_path, oracle_path, metadata_path, expected_metadata):
    if not result_path.exists() or not oracle_path.exists() or not metadata_path.exists():
        return False
    try:
        rows = pd.read_csv(result_path)
        oracle = pd.read_csv(oracle_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    expected_ids = expected_metadata["problem_ids"]
    required_oracle_columns = {
        "token_confidences",
        "token_margins",
        "token_forced",
        "token_recoverable",
        "actual_verify_latency_ms",
        "draft_passes_elapsed",
        "oracle_current_fill_tokens",
        "oracle_cached_fill_tokens",
        "oracle_missing_fill_tokens",
    }
    return (
        metadata == expected_metadata
        and len(rows) == len(expected_ids)
        and not oracle.empty
        and required_oracle_columns.issubset(oracle.columns)
        and sorted(rows["problem_id"].astype(int).tolist()) == expected_ids
    )


def run_dataset(args, dataset, problem_ids):
    output_dir = Path(args.output_dir) / "raw" / dataset / METHOD_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    oracle_path = output_dir / "frontier_oracle_refinement_diagnostics.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata(args, dataset, problem_ids)
    if args.resume and results_complete(result_path, oracle_path, metadata_path, metadata):
        print(f"RESUME {dataset} | {METHOD_NAME}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
            "frontier_oracle_refinement_diagnostics.csv",
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
            "--frontier_stop_mode", "disabled",
            "--collect_oracle_refinement",
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
            f"RUN ORACLE {dataset} | samples={len(problem_ids)} | "
            f"problem_ids={problem_ids}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = pd.read_csv(result_path)
    rows["dataset"] = dataset
    rows["method"] = METHOD_NAME
    oracle = pd.read_csv(oracle_path)
    oracle["dataset"] = dataset
    oracle["method"] = METHOD_NAME
    return rows, oracle


def summarize_oracle_steps(oracle):
    records = []
    for (dataset, step), group in oracle.groupby(["dataset", "step"], sort=True):
        records.append({
            "dataset": dataset,
            "step": int(step),
            "snapshots": len(group),
            "mean_masks_remaining": group["masks_remaining"].mean(),
            "mean_committed_tokens": group["committed_tokens"].mean(),
            "mean_filled_tokens": group["filled_tokens"].mean(),
            "mean_accepted_len_if_stop": group["accepted_len_if_stop"].mean(),
            "mean_emitted_len_if_stop": group["emitted_len_if_stop"].mean(),
            "mean_delta_accepted_len": group["delta_accepted_len"].mean(),
            "mean_delta_emitted_len": group["delta_emitted_len"].mean(),
            "positive_delta_accepted_rate_percent": 100.0
            * (group["delta_accepted_len"] > 0).mean(),
        })
    return pd.DataFrame(records)


def summarize_oracle_rounds(oracle):
    records = []
    for keys, group in oracle.groupby(["dataset", "problem_id", "round_id"], sort=False):
        dataset, problem_id, round_id = keys
        group = group.sort_values("step")
        final_row = group.iloc[-1]
        best_accept = group["accepted_len_if_stop"].max()
        best_accept_step = int(group.loc[group["accepted_len_if_stop"].idxmax(), "step"])
        efficiency = group["emitted_len_if_stop"] / group["draft_latency_elapsed_ms"].clip(lower=1e-9)
        best_efficiency_step = int(group.loc[efficiency.idxmax(), "step"])
        final_step = int(final_row["step"])
        records.append({
            "dataset": dataset,
            "problem_id": int(problem_id),
            "round_id": int(round_id),
            "final_step": final_step,
            "best_accept_step": best_accept_step,
            "best_efficiency_step": best_efficiency_step,
            "final_accepted_len": int(final_row["accepted_len_if_stop"]),
            "best_accepted_len": int(best_accept),
            "accept_saturates_before_final": int(best_accept_step < final_step),
            "efficiency_best_before_final": int(best_efficiency_step < final_step),
            "wasted_steps_by_accept": max(0, final_step - best_accept_step),
            "wasted_steps_by_efficiency": max(0, final_step - best_efficiency_step),
            "final_masks_remaining": int(final_row["masks_remaining"]),
            "final_draft_latency_elapsed_ms": float(final_row["draft_latency_elapsed_ms"]),
        })
    return pd.DataFrame(records)


def summarize_datasets(round_summary):
    records = []
    for dataset, group in round_summary.groupby("dataset", sort=False):
        records.append({
            "dataset": dataset,
            "rounds": len(group),
            "mean_final_step": group["final_step"].mean(),
            "mean_best_accept_step": group["best_accept_step"].mean(),
            "mean_best_efficiency_step": group["best_efficiency_step"].mean(),
            "accept_saturates_before_final_percent": 100.0
            * group["accept_saturates_before_final"].mean(),
            "efficiency_best_before_final_percent": 100.0
            * group["efficiency_best_before_final"].mean(),
            "mean_wasted_steps_by_accept": group["wasted_steps_by_accept"].mean(),
            "mean_wasted_steps_by_efficiency": group["wasted_steps_by_efficiency"].mean(),
            "mean_final_accepted_len": group["final_accepted_len"].mean(),
            "mean_best_accepted_len": group["best_accepted_len"].mean(),
        })
    return pd.DataFrame(records)


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
        "method": METHOD_NAME,
        "purpose": (
            "Offline oracle profiling for Fast-dLLM refinement depth. "
            "Oracle verifier passes are diagnostics only and are excluded from "
            "benchmark_results timing."
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

    benchmark_frames = []
    oracle_frames = []
    for dataset in args.datasets:
        benchmark_rows, oracle_rows = run_dataset(args, dataset, problem_ids[dataset])
        benchmark_frames.append(benchmark_rows)
        oracle_frames.append(oracle_rows)

    benchmark = pd.concat(benchmark_frames, ignore_index=True, sort=False)
    oracle = pd.concat(oracle_frames, ignore_index=True, sort=False)
    step_summary = summarize_oracle_steps(oracle)
    round_summary = summarize_oracle_rounds(oracle)
    dataset_summary = summarize_datasets(round_summary)

    benchmark.to_csv(output_dir / "per_observation.csv", index=False)
    oracle.to_csv(output_dir / "oracle_refinement_snapshots.csv", index=False)
    step_summary.to_csv(output_dir / "oracle_step_summary.csv", index=False)
    round_summary.to_csv(output_dir / "oracle_round_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "oracle_dataset_summary.csv", index=False)
    write_manifest(args, output_dir, problem_ids)

    archive_path = None
    if not args.skip_archive:
        archive_path = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    print("\nORACLE REFINEMENT DATASET SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nORACLE REFINEMENT STEP SUMMARY")
    print(step_summary.head(40).to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    if archive_path:
        print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
