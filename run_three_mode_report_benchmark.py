import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import transformers


TARGET_TPT_MS = {
    "Qwen2.5-3B-Instruct": 7.2,
    "Qwen2.5-7B-Instruct": 13.5,
    "Qwen2.5-14B-Instruct": 24.7,
    "Qwen2.5-32B-Instruct": 52.6,
}

REQUIRED_BENCHMARK_COLUMNS = {
    "problem_id",
    "actual_total_time",
    "actual_draft_time",
    "actual_verify_time",
    "actual_e2e_time",
    "output_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "num_speculation_rounds",
    "total_num_forward_passes",
    "output_token_hash",
    "is_correct",
}

CASES = (
    {
        "case": "disabled_spec10_lowconf0p45",
        "frontier_mode": "disabled",
        "spec_len": 10,
        "lowconf_threshold": 0.45,
        "max_spec_len": 60,
        "incr_len": 10,
    },
    {
        "case": "cost_aware_no_extend_spec5_lowconf0p45",
        "frontier_mode": "cost_aware_no_extend",
        "spec_len": 5,
        "lowconf_threshold": 0.45,
        "max_spec_len": 60,
        "incr_len": 10,
    },
    {
        "case": "cost_aware_extend_spec5_lowconf0p60",
        "frontier_mode": "cost_aware",
        "spec_len": 5,
        "lowconf_threshold": 0.60,
        "max_spec_len": 60,
        "incr_len": 10,
    },
)

COMPARISONS = (
    {
        "comparison": "no_extend_vs_failfast",
        "baseline": "disabled_spec10_lowconf0p45",
        "candidate": "cost_aware_no_extend_spec5_lowconf0p45",
    },
    {
        "comparison": "extend_vs_failfast",
        "baseline": "disabled_spec10_lowconf0p45",
        "candidate": "cost_aware_extend_spec5_lowconf0p60",
    },
    {
        "comparison": "extend_vs_no_extend",
        "baseline": "cost_aware_no_extend_spec5_lowconf0p45",
        "candidate": "cost_aware_extend_spec5_lowconf0p60",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_tpt_ms", type=float)
    parser.add_argument("--draft_fwd_pass_ms", type=float, default=6.1)
    parser.add_argument("--output_dir", default="/content/failfasttesting/outputs_three_mode_report")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 0:
        raise ValueError("--warmup_questions must be non-negative")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    target_name = args.target_model_name.rsplit("/", 1)[-1]
    if target_name not in TARGET_TPT_MS:
        raise ValueError(f"Unsupported verifier model for failfast latency table: {target_name}")
    if args.target_tpt_ms is None:
        args.target_tpt_ms = TARGET_TPT_MS[target_name]
    if args.target_tpt_ms <= 0 or args.draft_fwd_pass_ms <= 0:
        raise ValueError("Modeled latency constants must be positive")


def as_bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def run_streaming(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_case(args, repetition, case):
    output_dir = Path(args.output_dir) / f"repetition_{repetition + 1}" / case["case"]
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_results.csv"
    expected_total_rows = args.num_questions + args.warmup_questions
    if args.resume and benchmark_path.exists():
        existing_rows = pd.read_csv(benchmark_path)
        complete = len(existing_rows) == expected_total_rows
        compatible = REQUIRED_BENCHMARK_COLUMNS.issubset(existing_rows.columns)
        if complete and compatible:
            print(f"RESUMING completed case: {case['case']} repetition {repetition + 1}", flush=True)
            rows = existing_rows
        else:
            benchmark_path.unlink()
            rows = None
    else:
        rows = None
    if rows is None and benchmark_path.exists():
        benchmark_path.unlink()

    if rows is None:
        cmd = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", "gsm8k",
            "--num_questions", str(expected_total_rows),
            "--benchmark_modes", "dllm_ar",
            "--max_new_tokens", str(args.max_new_tokens),
            "--spec_len", str(case["spec_len"]),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(case["lowconf_threshold"]),
            "--sweep_max_spec_len", str(case["max_spec_len"]),
            "--sweep_incr_len", str(case["incr_len"]),
            "--frontier_stop_mode", case["frontier_mode"],
            "--frontier_min_steps", str(args.frontier_min_steps),
            "--frontier_patience", str(args.frontier_patience),
            "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
            "--seed", str(args.seed),
            "--quiet_generation",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]

        print("\n" + "=" * 100, flush=True)
        print(
            f"REPETITION {repetition + 1}/{args.repetitions} | {case['case']} | "
            f"spec_len={case['spec_len']} | lowconf={case['lowconf_threshold']} | "
            f"max_spec_len={case['max_spec_len']} | incr_len={case['incr_len']}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(cmd)
        rows = pd.read_csv(benchmark_path)

    rows = rows[rows["problem_id"] >= args.warmup_questions].copy()
    expected_rows = args.num_questions
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} measured rows, found {len(rows)} in {benchmark_path}")
    rows["measured_problem_id"] = rows["problem_id"] - args.warmup_questions
    rows["repetition"] = repetition + 1
    for key, value in case.items():
        rows[key] = value
    return rows


def aggregate_group(group, args):
    output_tokens = group["output_tokens"].sum()
    drafted_tokens = group["drafted_tokens"].sum()
    accepted_tokens = group["accepted_tokens"].sum()
    rounds = group["num_speculation_rounds"].sum()
    passes = group["total_num_forward_passes"].sum()
    modeled_total_ms = passes * args.draft_fwd_pass_ms + rounds * args.target_tpt_ms
    actual_e2e_total_ms = 1000.0 * group["actual_e2e_time"].sum()
    modeled_ms_per_output_token = modeled_total_ms / output_tokens
    return {
        "num_observations": len(group),
        "num_unique_questions": group["measured_problem_id"].nunique(),
        "output_tokens": output_tokens,
        "modeled_total_time_ms": modeled_total_ms,
        "modeled_ms_per_output_token": modeled_ms_per_output_token,
        "modeled_speedup_vs_verifier_ar": args.target_tpt_ms / modeled_ms_per_output_token,
        "actual_e2e_time_mean_s": group["actual_e2e_time"].mean(),
        "actual_e2e_ms_per_output_token": actual_e2e_total_ms / output_tokens,
        "actual_compute_time_mean_s": group["actual_total_time"].mean(),
        "actual_draft_time_mean_s": group["actual_draft_time"].mean(),
        "actual_verify_time_mean_s": group["actual_verify_time"].mean(),
        "acceptance_rate_percent": 100.0 * accepted_tokens / drafted_tokens,
        "drafted_tokens_per_round": drafted_tokens / rounds,
        "accepted_tokens_per_round": accepted_tokens / rounds,
        "output_tokens_per_round": output_tokens / rounds,
        "draft_forward_passes_per_100_output_tokens": 100.0 * passes / output_tokens,
        "verifier_rounds_per_100_output_tokens": 100.0 * rounds / output_tokens,
        "gsm8k_accuracy_percent": 100.0 * as_bool_series(group["is_correct"]).mean(),
    }


def aggregate_cases(raw_rows, args):
    rows = []
    for case_name, group in raw_rows.groupby("case", sort=False):
        row = {"case": case_name}
        row.update(aggregate_group(group, args))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("modeled_ms_per_output_token")


def aggregate_repetitions(raw_rows, args):
    rows = []
    for (repetition, case_name), group in raw_rows.groupby(["repetition", "case"], sort=False):
        row = {"repetition": repetition, "case": case_name}
        row.update(aggregate_group(group, args))
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_rows(raw_rows, args):
    rows = []
    index_columns = ["repetition", "measured_problem_id"]
    for comparison in COMPARISONS:
        baseline = raw_rows[raw_rows["case"] == comparison["baseline"]].set_index(index_columns)
        candidate = raw_rows[raw_rows["case"] == comparison["candidate"]].set_index(index_columns)
        for index in baseline.index.intersection(candidate.index):
            base = baseline.loc[index]
            current = candidate.loc[index]
            base_modeled_tpt = (
                base["total_num_forward_passes"] * args.draft_fwd_pass_ms
                + base["num_speculation_rounds"] * args.target_tpt_ms
            ) / base["output_tokens"]
            current_modeled_tpt = (
                current["total_num_forward_passes"] * args.draft_fwd_pass_ms
                + current["num_speculation_rounds"] * args.target_tpt_ms
            ) / current["output_tokens"]
            base_actual_tpt = 1000.0 * base["actual_e2e_time"] / base["output_tokens"]
            current_actual_tpt = 1000.0 * current["actual_e2e_time"] / current["output_tokens"]
            rows.append({
                "comparison": comparison["comparison"],
                "baseline": comparison["baseline"],
                "candidate": comparison["candidate"],
                "repetition": index[0],
                "measured_problem_id": index[1],
                "modeled_speedup_candidate_vs_baseline": base_modeled_tpt / current_modeled_tpt,
                "modeled_ms_per_output_token_delta": current_modeled_tpt - base_modeled_tpt,
                "actual_e2e_speedup_candidate_vs_baseline": base_actual_tpt / current_actual_tpt,
                "actual_e2e_ms_per_output_token_delta": current_actual_tpt - base_actual_tpt,
                "acceptance_rate_delta_percent": current["acceptance_rate_percent"] - base["acceptance_rate_percent"],
                "verifier_round_delta": current["num_speculation_rounds"] - base["num_speculation_rounds"],
                "draft_forward_pass_delta": current["total_num_forward_passes"] - base["total_num_forward_passes"],
                "output_length_delta": current["output_tokens"] - base["output_tokens"],
                "output_matches_baseline": current["output_token_hash"] == base["output_token_hash"],
                "correctness_delta": int(as_bool(current["is_correct"])) - int(as_bool(base["is_correct"])),
            })
    return pd.DataFrame(rows)


def summarize_paired(paired):
    metric_columns = [
        "modeled_speedup_candidate_vs_baseline",
        "modeled_ms_per_output_token_delta",
        "actual_e2e_speedup_candidate_vs_baseline",
        "actual_e2e_ms_per_output_token_delta",
        "acceptance_rate_delta_percent",
        "verifier_round_delta",
        "draft_forward_pass_delta",
        "output_length_delta",
        "correctness_delta",
    ]
    rows = []
    for comparison_name, observations in paired.groupby("comparison", sort=False):
        first = observations.iloc[0]
        by_problem = observations.groupby("measured_problem_id")[metric_columns].mean()
        row = {
            "comparison": comparison_name,
            "baseline": first["baseline"],
            "candidate": first["candidate"],
            "num_questions": len(by_problem),
            "num_observations": len(observations),
            "candidate_modeled_win_rate_percent": 100.0 * (by_problem["modeled_ms_per_output_token_delta"] < 0).mean(),
            "candidate_actual_win_rate_percent": 100.0 * (by_problem["actual_e2e_ms_per_output_token_delta"] < 0).mean(),
            "output_match_rate_percent": 100.0 * observations["output_matches_baseline"].mean(),
        }
        for column in metric_columns:
            values = by_problem[column]
            mean = values.mean()
            std = values.std(ddof=1)
            half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            row[f"{column}_mean"] = mean
            row[f"{column}_median"] = values.median()
            row[f"{column}_std"] = std
            row[f"{column}_ci95_low"] = mean - half_width
            row[f"{column}_ci95_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(args, output_dir):
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        git_commit = None
    manifest = {
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arguments": vars(args),
        "cases": CASES,
        "primary_metric": "modeled_ms_per_output_token",
        "primary_metric_formula": (
            f"(draft_forward_passes * {args.draft_fwd_pass_ms} + "
            f"verifier_rounds * {args.target_tpt_ms}) / output_tokens"
        ),
        "secondary_metric": "synchronized_actual_e2e_ms_per_output_token",
        "timing_scope": "prompt/model inputs prepared; synchronized immediately before generation and after final token; excludes model loading, prompt tokenization, output decoding, plotting, file output, and console generation output",
        "case_order_policy": "cyclic rotation across repetitions",
        "sampling_temperature": 0.6,
    }
    with (output_dir / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, output_dir)

    all_rows = []
    for repetition in range(args.repetitions):
        offset = repetition % len(CASES)
        ordered_cases = CASES[offset:] + CASES[:offset]
        for case in ordered_cases:
            all_rows.append(run_case(args, repetition, case))

    raw_rows = pd.concat(all_rows, ignore_index=True)
    aggregate = aggregate_cases(raw_rows, args)
    repetition_summary = aggregate_repetitions(raw_rows, args)
    paired = build_paired_rows(raw_rows, args)
    paired_summary = summarize_paired(paired)

    raw_rows.to_csv(output_dir / "report_per_observation.csv", index=False)
    aggregate.to_csv(output_dir / "report_aggregate.csv", index=False)
    repetition_summary.to_csv(output_dir / "report_per_repetition.csv", index=False)
    paired.to_csv(output_dir / "report_paired_observations.csv", index=False)
    paired_summary.to_csv(output_dir / "report_paired_summary.csv", index=False)

    display_columns = [
        "case",
        "num_unique_questions",
        "modeled_ms_per_output_token",
        "modeled_speedup_vs_verifier_ar",
        "actual_e2e_ms_per_output_token",
        "actual_e2e_time_mean_s",
        "acceptance_rate_percent",
        "accepted_tokens_per_round",
        "output_tokens_per_round",
        "draft_forward_passes_per_100_output_tokens",
        "verifier_rounds_per_100_output_tokens",
        "gsm8k_accuracy_percent",
    ]
    paired_columns = [
        "comparison",
        "num_questions",
        "candidate_modeled_win_rate_percent",
        "candidate_actual_win_rate_percent",
        "output_match_rate_percent",
        "modeled_speedup_candidate_vs_baseline_mean",
        "modeled_speedup_candidate_vs_baseline_ci95_low",
        "modeled_speedup_candidate_vs_baseline_ci95_high",
        "actual_e2e_speedup_candidate_vs_baseline_mean",
        "actual_e2e_speedup_candidate_vs_baseline_ci95_low",
        "actual_e2e_speedup_candidate_vs_baseline_ci95_high",
    ]
    print("\nREPORT AGGREGATE (PRIMARY SORT: FAILFAST A6000 MODELED LATENCY)")
    print(aggregate[display_columns].to_string(index=False))
    print("\nPAIRED REPORT")
    print(paired_summary[paired_columns].to_string(index=False))
    print(f"\nSaved report: {output_dir}")


if __name__ == "__main__":
    main()
