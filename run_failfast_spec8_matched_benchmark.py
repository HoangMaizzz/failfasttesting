import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_adaptive_unmask_only_benchmark import DATASET_SIZES, sampled_problem_ids
from run_cost_aware_v2_benchmark import (
    aggregate_method,
    build_comparison_summary,
    build_dataset_summary,
    build_paired_observations,
    run_streaming,
)


BENCHMARK_VERSION = "failfast_spec8_matched_v1"
METHOD = {
    "name": "failfast_spec8",
    "frontier_mode": "disabled",
    "spec_len": 8,
    "incr_len": 8,
    "lowconf_threshold": 0.45,
}
REFERENCE_METHOD = "adaptive_unmask_only"


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
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adaptive_results_dir",
        default="/content/failfasttesting/outputs_adaptive_unmask_only_test15",
    )
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_failfast_spec8_matched_test15",
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
        "drafter_threshold": args.drafter_threshold,
        "max_spec_len": args.max_spec_len,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "method": METHOD,
    }


def results_complete(result_path, metadata_path, expected_metadata):
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
        and len(rows) == len(expected_ids)
        and sorted(rows["problem_id"].astype(int).tolist()) == expected_ids
    )


def run_dataset(args, dataset, problem_ids):
    output_dir = Path(args.output_dir) / "raw" / dataset / METHOD["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = run_metadata(args, dataset, problem_ids)
    if args.resume and results_complete(result_path, metadata_path, metadata):
        print(f"RESUME {dataset} | {METHOD['name']}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "frontier_round_diagnostics.csv",
            "frontier_extension_diagnostics.csv",
            "frontier_gain_diagnostics.csv",
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
            "--spec_len", str(METHOD["spec_len"]),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(METHOD["lowconf_threshold"]),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(METHOD["incr_len"]),
            "--frontier_stop_mode", METHOD["frontier_mode"],
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
            f"RUN {dataset} | {METHOD['name']} | samples={len(problem_ids)} | "
            f"problem_ids={problem_ids}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = pd.read_csv(result_path)
    rows["dataset"] = dataset
    rows["method"] = "failfast"
    rows["actual_measured_time"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["actual_measured_ms_per_output_token"] = (
        1000.0
        * rows["actual_measured_time"]
        / pd.to_numeric(rows["output_tokens"], errors="coerce")
    )
    return rows


def build_failfast_summary(rows):
    records = []
    for dataset, group in rows.groupby("dataset", sort=False):
        record = {"dataset": dataset, "method": METHOD["name"]}
        record.update(aggregate_method(group))
        records.append(record)
    return pd.DataFrame(records)


def load_matched_reference(reference_dir, expected_problem_ids):
    path = Path(reference_dir) / "per_observation.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    required = {"dataset", "problem_id", "method"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Adaptive reference is missing columns: {sorted(missing)}")
    rows = rows[rows["method"] == REFERENCE_METHOD].copy()
    expected = {
        (dataset, int(problem_id))
        for dataset, problem_ids in expected_problem_ids.items()
        for problem_id in problem_ids
    }
    observed = set(zip(rows["dataset"], rows["problem_id"].astype(int)))
    if observed != expected:
        missing_keys = sorted(expected - observed)
        extra_keys = sorted(observed - expected)
        raise ValueError(
            "Adaptive reference problem IDs do not match this benchmark: "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    return rows


def write_comparison(output_dir, failfast_rows, adaptive_rows):
    combined = pd.concat([failfast_rows, adaptive_rows], ignore_index=True, sort=False)
    dataset_summary = build_dataset_summary(combined)
    paired = build_paired_observations(combined, REFERENCE_METHOD)
    dataset_comparison, overall_comparison = build_comparison_summary(
        dataset_summary,
        paired,
        REFERENCE_METHOD,
    )
    combined.to_csv(output_dir / "combined_per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "combined_dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_observations.csv", index=False)
    dataset_comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    overall_comparison.to_csv(output_dir / "overall_comparison.csv", index=False)
    print("\nMATCHED ADAPTIVE-ONLY VS FAILFAST SPEC8")
    print(dataset_comparison.to_string(index=False))
    print("\nOVERALL COMPARISON")
    print(overall_comparison.to_string(index=False))


def write_manifest(args, output_dir, problem_ids, reference_loaded):
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
        "method": METHOD,
        "adaptive_reference_loaded": reference_loaded,
        "comparison_contract": "both methods use spec_len=8, incr_len=8, lowconf_threshold=0.45",
        "primary_metric": "actual_draft_time + actual_verify_time + actual_post_verify_time per output token",
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
    frames = [
        run_dataset(args, dataset, problem_ids[dataset])
        for dataset in args.datasets
    ]
    rows = pd.concat(frames, ignore_index=True, sort=False)
    summary = build_failfast_summary(rows)
    rows.to_csv(output_dir / "per_observation.csv", index=False)
    summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    adaptive_rows = load_matched_reference(args.adaptive_results_dir, problem_ids)
    if adaptive_rows is None:
        print(
            f"\nAdaptive reference not found at {args.adaptive_results_dir}; "
            "saved FailFast results without paired comparison.",
            flush=True,
        )
    else:
        write_comparison(output_dir, rows, adaptive_rows)
    write_manifest(args, output_dir, problem_ids, adaptive_rows is not None)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nFAILFAST SPEC8 SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
