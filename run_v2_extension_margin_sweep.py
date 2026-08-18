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

from run_cost_aware_v2_benchmark import aggregate_method, safe_ratio


DATASETS = ("math", "aime", "gsm8k", "humaneval")
BENCHMARK_VERSION = "cost_aware_v2_extension_margin_sweep_v2"
DEFAULT_MARGINS = (-0.05, -0.03, 0.0, 0.03, 0.05, 0.10)


def margin_label(value):
    value = float(value)
    magnitude = round(abs(value) * 100)
    if value < 0:
        return f"require_gain_{magnitude}pct"
    if value > 0:
        return f"allow_loss_{magnitude}pct"
    return "break_even"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--extension_cost_margins", nargs="+", type=float, default=list(DEFAULT_MARGINS))
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
    parser.add_argument("--frontier_v2_gain_calibration_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_min_gain_calibration_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_prefix_calibration_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_min_prefix_calibration_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_hazard_prior_strength", type=float, default=8.0)
    parser.add_argument("--frontier_v2_extension_prior_strength", type=float, default=2.0)
    parser.add_argument("--frontier_v2_min_hazard_observations", type=int, default=8)
    parser.add_argument("--frontier_v2_min_calibration_tokens", type=int, default=64)
    parser.add_argument("--reference_margin", type=float, default=-0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_v2_extension_margin_sweep_test20",
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
    if len(args.extension_cost_margins) != len(set(args.extension_cost_margins)):
        raise ValueError("--extension_cost_margins must not contain duplicates")
    if any(not -0.99 < margin <= 1.0 for margin in args.extension_cost_margins):
        raise ValueError("extension cost margins must be in (-0.99, 1.0]")
    if args.frontier_v2_gain_calibration_prior_strength <= 0:
        raise ValueError("--frontier_v2_gain_calibration_prior_strength must be positive")
    if args.frontier_v2_min_gain_calibration_observations <= 0:
        raise ValueError("--frontier_v2_min_gain_calibration_observations must be positive")
    if args.frontier_v2_prefix_calibration_prior_strength <= 0:
        raise ValueError("--frontier_v2_prefix_calibration_prior_strength must be positive")
    if args.frontier_v2_min_prefix_calibration_observations <= 0:
        raise ValueError("--frontier_v2_min_prefix_calibration_observations must be positive")
    matching_reference = next(
        (
            margin
            for margin in args.extension_cost_margins
            if math.isclose(margin, args.reference_margin, rel_tol=0.0, abs_tol=1e-9)
        ),
        None,
    )
    if matching_reference is None:
        raise ValueError("--reference_margin must be included in --extension_cost_margins")
    args.reference_margin = matching_reference


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


def expected_metadata(args, dataset, margin):
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
        "frontier_v2_extension_cost_margin": margin,
        "frontier_v2_gain_calibration_prior_strength": args.frontier_v2_gain_calibration_prior_strength,
        "frontier_v2_min_gain_calibration_observations": args.frontier_v2_min_gain_calibration_observations,
        "frontier_v2_prefix_calibration_prior_strength": args.frontier_v2_prefix_calibration_prior_strength,
        "frontier_v2_min_prefix_calibration_observations": args.frontier_v2_min_prefix_calibration_observations,
        "frontier_v2_hazard_prior_strength": args.frontier_v2_hazard_prior_strength,
        "frontier_v2_extension_prior_strength": args.frontier_v2_extension_prior_strength,
        "frontier_v2_min_hazard_observations": args.frontier_v2_min_hazard_observations,
        "frontier_v2_min_calibration_tokens": args.frontier_v2_min_calibration_tokens,
        "seed": args.seed,
    }


def results_complete(result_path, metadata_path, metadata):
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        rows = pd.read_csv(result_path)
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return (
        saved_metadata == metadata
        and len(rows) == metadata["num_questions"]
        and rows["problem_id"].nunique() == metadata["num_questions"]
    )


def run_case(args, dataset, margin):
    label = margin_label(margin)
    output_dir = Path(args.output_dir) / "raw" / dataset / label
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata = expected_metadata(args, dataset, margin)
    if not (args.resume and results_complete(result_path, metadata_path, metadata)):
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
            "--frontier_v2_extension_cost_margin", str(margin),
            "--frontier_v2_gain_calibration_prior_strength", str(args.frontier_v2_gain_calibration_prior_strength),
            "--frontier_v2_min_gain_calibration_observations", str(args.frontier_v2_min_gain_calibration_observations),
            "--frontier_v2_prefix_calibration_prior_strength", str(args.frontier_v2_prefix_calibration_prior_strength),
            "--frontier_v2_min_prefix_calibration_observations", str(args.frontier_v2_min_prefix_calibration_observations),
            "--frontier_v2_hazard_prior_strength", str(args.frontier_v2_hazard_prior_strength),
            "--frontier_v2_extension_prior_strength", str(args.frontier_v2_extension_prior_strength),
            "--frontier_v2_min_hazard_observations", str(args.frontier_v2_min_hazard_observations),
            "--frontier_v2_min_calibration_tokens", str(args.frontier_v2_min_calibration_tokens),
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
            f"RUN {dataset} | {label} | margin={margin:+.2%} | samples={args.num_questions}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = pd.read_csv(result_path)
    if len(rows) != args.num_questions:
        raise RuntimeError(f"Expected {args.num_questions} rows in {result_path}, found {len(rows)}")
    rows["dataset"] = dataset
    rows["extension_policy"] = label
    rows["extension_cost_margin"] = float(margin)
    rows["actual_measured_time"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["actual_measured_ms_per_output_token"] = (
        1000.0 * rows["actual_measured_time"]
        / pd.to_numeric(rows["output_tokens"], errors="coerce")
    )
    return rows


def build_summary(rows, group_columns):
    records = []
    for keys, group in rows.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record.update(aggregate_method(group))
        records.append(record)
    return pd.DataFrame(records)


def build_reference_comparison(rows, reference_margin):
    metrics = [
        "actual_measured_time",
        "actual_measured_ms_per_output_token",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
        "output_tokens",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "accepted_tokens",
        "drafted_tokens",
        "frontier_v2_extend_actions",
        "frontier_v2_extension_stop_actions",
        "output_token_hash",
    ]
    reference_mask = rows["extension_cost_margin"].map(
        lambda margin: math.isclose(
            float(margin), float(reference_margin), rel_tol=0.0, abs_tol=1e-9
        )
    )
    reference = rows[reference_mask][
        ["dataset", "problem_id", *metrics]
    ].copy()
    reference = reference.rename(columns={metric: f"reference_{metric}" for metric in metrics})
    frames = []
    for margin in sorted(rows["extension_cost_margin"].unique()):
        if math.isclose(float(margin), float(reference_margin)):
            continue
        candidate = rows[rows["extension_cost_margin"] == margin][
            ["dataset", "problem_id", *metrics]
        ].copy()
        candidate = candidate.rename(columns={metric: f"candidate_{metric}" for metric in metrics})
        paired = reference.merge(candidate, on=["dataset", "problem_id"], validate="one_to_one")
        paired["reference_margin"] = reference_margin
        paired["candidate_margin"] = margin
        paired["candidate_policy"] = margin_label(margin)
        paired["candidate_speedup_vs_reference"] = (
            paired["reference_actual_measured_ms_per_output_token"]
            / paired["candidate_actual_measured_ms_per_output_token"]
        )
        paired["candidate_faster"] = paired["candidate_speedup_vs_reference"] > 1.0
        paired["output_matches_reference"] = (
            paired["reference_output_token_hash"] == paired["candidate_output_token_hash"]
        )
        frames.append(paired)
    paired = pd.concat(frames, ignore_index=True, sort=False)
    records = []
    for (dataset, margin), group in paired.groupby(["dataset", "candidate_margin"], sort=False):
        reference_ms = safe_ratio(
            1000.0 * group["reference_actual_measured_time"].sum(),
            group["reference_output_tokens"].sum(),
        )
        candidate_ms = safe_ratio(
            1000.0 * group["candidate_actual_measured_time"].sum(),
            group["candidate_output_tokens"].sum(),
        )
        records.append({
            "dataset": dataset,
            "candidate_margin": margin,
            "candidate_policy": margin_label(margin),
            "num_samples": len(group),
            "pooled_speedup_vs_reference": safe_ratio(reference_ms, candidate_ms),
            "paired_win_rate_percent": 100.0 * group["candidate_faster"].mean(),
            "output_match_rate_percent": 100.0 * group["output_matches_reference"].mean(),
            "reference_ms_per_output_token": reference_ms,
            "candidate_ms_per_output_token": candidate_ms,
            "verifier_round_delta": (
                group["candidate_num_speculation_rounds"]
                - group["reference_num_speculation_rounds"]
            ).mean(),
            "extend_action_delta": (
                group["candidate_frontier_v2_extend_actions"]
                - group["reference_frontier_v2_extend_actions"]
            ).mean(),
        })
    return paired, pd.DataFrame(records)


def collect_diagnostics(output_dir, datasets, margins, filename):
    frames = []
    for dataset in datasets:
        for margin in margins:
            label = margin_label(margin)
            path = output_dir / "raw" / dataset / label / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["dataset"] = dataset
            frame["extension_policy"] = label
            frame["extension_cost_margin"] = margin
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


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
        "primary_metric": "measured milliseconds per output token",
        "time_formula": "actual_draft_time + actual_verify_time + actual_post_verify_time",
        "margin_semantics": "negative requires predicted gain; positive permits predicted loss",
        "target_decoding": "greedy",
        "comparison_policy": "paired by dataset and problem_id against reference_margin",
        "run_order_policy": "margin order rotates by dataset",
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
    frames = []
    margins = tuple(args.extension_cost_margins)
    for dataset_index, dataset in enumerate(args.datasets):
        rotation = dataset_index % len(margins)
        margin_order = margins[rotation:] + margins[:rotation]
        for margin in margin_order:
            frames.append(run_case(args, dataset, margin))
    rows = pd.concat(frames, ignore_index=True, sort=False)
    dataset_summary = build_summary(rows, ["dataset", "extension_policy", "extension_cost_margin"])
    overall_summary = build_summary(rows, ["extension_policy", "extension_cost_margin"])
    paired, comparison = build_reference_comparison(rows, args.reference_margin)
    round_diagnostics = collect_diagnostics(
        output_dir, args.datasets, margins, "frontier_round_diagnostics.csv"
    )
    extension_diagnostics = collect_diagnostics(
        output_dir, args.datasets, margins, "frontier_extension_diagnostics.csv"
    )
    gain_diagnostics = collect_diagnostics(
        output_dir, args.datasets, margins, "frontier_gain_diagnostics.csv"
    )
    rows.to_csv(output_dir / "per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_margin_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "overall_margin_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_vs_current_observations.csv", index=False)
    comparison.to_csv(output_dir / "margin_comparison_vs_current.csv", index=False)
    round_diagnostics.to_csv(output_dir / "all_frontier_round_diagnostics.csv", index=False)
    extension_diagnostics.to_csv(output_dir / "all_frontier_extension_diagnostics.csv", index=False)
    gain_diagnostics.to_csv(output_dir / "all_frontier_gain_diagnostics.csv", index=False)
    write_manifest(args, output_dir)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nDATASET MARGIN SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nOVERALL MARGIN SUMMARY")
    print(overall_summary.to_string(index=False))
    print("\nPAIRED COMPARISON VS CURRENT -3%")
    print(comparison.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
