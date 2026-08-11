import argparse
import hashlib
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd


CASES = (
    {
        "case": "paper_failfast_dynamic_spec10_tau0p50",
        "mode": "disabled",
        "spec_len": 10,
        "lowconf_threshold": 0.50,
        "max_spec_len": 60,
    },
    {
        "case": "disabled_fixed_spec5_tau0p50",
        "mode": "disabled",
        "spec_len": 5,
        "lowconf_threshold": 0.50,
        "max_spec_len": 5,
    },
    {
        "case": "cost_aware_fixed_spec5_tau0p50",
        "mode": "cost_aware_no_extend",
        "spec_len": 5,
        "lowconf_threshold": 0.50,
        "max_spec_len": 5,
    },
    {
        "case": "disabled_dynamic_spec5_tau0p50",
        "mode": "disabled",
        "spec_len": 5,
        "lowconf_threshold": 0.50,
        "max_spec_len": 60,
    },
    {
        "case": "cost_aware_dynamic_spec5_tau0p50",
        "mode": "cost_aware",
        "spec_len": 5,
        "lowconf_threshold": 0.50,
        "max_spec_len": 60,
    },
)

COMPARISONS = (
    {
        "comparison": "fixed5_controller_effect",
        "baseline": "disabled_fixed_spec5_tau0p50",
        "candidate": "cost_aware_fixed_spec5_tau0p50",
    },
    {
        "comparison": "dynamic5_controller_effect",
        "baseline": "disabled_dynamic_spec5_tau0p50",
        "candidate": "cost_aware_dynamic_spec5_tau0p50",
    },
    {
        "comparison": "initial_length_effect_without_controller",
        "baseline": "paper_failfast_dynamic_spec10_tau0p50",
        "candidate": "disabled_dynamic_spec5_tau0p50",
    },
    {
        "comparison": "full_method_vs_paper_failfast",
        "baseline": "paper_failfast_dynamic_spec10_tau0p50",
        "candidate": "cost_aware_dynamic_spec5_tau0p50",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--output_dir", default="/content/failfasttesting/outputs_controlled_denoising_ablation")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def get_output_tokens(rounds):
    output_tokens = []
    for round_data in rounds:
        draft = round_data.get("~draft_proposal", [])
        accepted_len = int(round_data.get("accepted_len", 0))
        output_tokens.extend(draft[:accepted_len])
        final_token = round_data.get("final_token")
        if final_token is not None:
            output_tokens.append(int(final_token))
    return output_tokens


def token_sequence_hash(token_ids):
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


def run_case(args, case):
    output_dir = Path(args.output_dir) / case["case"]
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_results.csv"
    if benchmark_path.exists():
        benchmark_path.unlink()

    cmd = [
        sys.executable,
        "failfast.py",
        "--dataset_name", "gsm8k",
        "--num_questions", str(args.num_questions),
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
        "--sweep_incr_len", "10",
        "--frontier_stop_mode", case["mode"],
        "--frontier_min_steps", str(args.frontier_min_steps),
        "--frontier_patience", str(args.frontier_patience),
        "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
        "--collect_draft_diagnostics",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]

    print("\n" + "=" * 100)
    print(
        f"RUNNING {case['case']} | mode={case['mode']} | "
        f"spec_len={case['spec_len']} | lowconf={case['lowconf_threshold']} | "
        f"max_spec_len={case['max_spec_len']}"
    )
    print("=" * 100, flush=True)
    run_streaming(cmd)
    return output_dir


def load_case_rows(case, output_dir):
    benchmark = pd.read_csv(output_dir / "benchmark_results.csv")
    benchmark = benchmark[benchmark["mode"] == "dllm_ar"].set_index("problem_id")
    sample_rows = []
    round_rows = []

    for pickle_path in sorted(output_dir.glob("pickles/**/*.pickle")):
        with pickle_path.open("rb") as handle:
            data = pickle.load(handle)

        problem_id = int(pickle_path.parents[1].name)
        if problem_id not in benchmark.index:
            continue
        rounds = data.get("stats_each_round", [])
        totals = {
            "total": 0,
            "prefill": 0,
            "cache_update": 0,
            "denoising": 0,
            "fill": 0,
        }

        for round_id, round_data in enumerate(rounds):
            stats = round_data.get("frontier_stats") or {}
            breakdown = stats.get("forward_pass_breakdown") or {}
            draft_len = len(round_data.get("~draft_proposal", []))
            accepted_len = int(round_data.get("accepted_len", 0))
            output_len = accepted_len + 1
            for key in totals:
                totals[key] += int(breakdown.get(key, 0))
            round_rows.append({
                "case": case["case"],
                "problem_id": problem_id,
                "round": round_id,
                "draft_len": draft_len,
                "accepted_len": accepted_len,
                "output_len": output_len,
                "total_forward_passes": int(breakdown.get("total", round_data.get("num_forward_passes", 0))),
                "denoising_forward_passes": int(breakdown.get("denoising", 0)),
                "fill_forward_passes": int(breakdown.get("fill", 0)),
                "prefill_forward_passes": int(breakdown.get("prefill", 0)),
                "cache_update_forward_passes": int(breakdown.get("cache_update", 0)),
                "stop_reason": stats.get("stop_reason"),
            })

        drafted_tokens = sum(len(round_data.get("~draft_proposal", [])) for round_data in rounds)
        accepted_tokens = sum(int(round_data.get("accepted_len", 0)) for round_data in rounds)
        output_token_ids = get_output_tokens(rounds)
        output_tokens = len(output_token_ids)
        benchmark_row = benchmark.loc[problem_id]
        sample_rows.append({
            "case": case["case"],
            "mode": case["mode"],
            "spec_len": case["spec_len"],
            "lowconf_threshold": case["lowconf_threshold"],
            "max_spec_len": case["max_spec_len"],
            "problem_id": problem_id,
            "num_rounds": len(rounds),
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "output_tokens": output_tokens,
            "output_token_hash": token_sequence_hash(output_token_ids),
            "acceptance_rate_percent": 100.0 * safe_div(accepted_tokens, drafted_tokens),
            "drafted_tokens_per_round": safe_div(drafted_tokens, len(rounds)),
            "accepted_tokens_per_round": safe_div(accepted_tokens, len(rounds)),
            "output_tokens_per_round": safe_div(output_tokens, len(rounds)),
            "total_forward_passes": totals["total"],
            "denoising_forward_passes": totals["denoising"],
            "fill_forward_passes": totals["fill"],
            "prefill_forward_passes": totals["prefill"],
            "cache_update_forward_passes": totals["cache_update"],
            "denoising_steps_per_10_drafted_tokens": 10.0 * safe_div(totals["denoising"], drafted_tokens),
            "total_passes_per_10_drafted_tokens": 10.0 * safe_div(totals["total"], drafted_tokens),
            "denoising_steps_per_10_output_tokens": 10.0 * safe_div(totals["denoising"], output_tokens),
            "total_passes_per_10_output_tokens": 10.0 * safe_div(totals["total"], output_tokens),
            "fill_passes_per_100_output_tokens": 100.0 * safe_div(totals["fill"], output_tokens),
            "rounds_per_100_output_tokens": 100.0 * safe_div(len(rounds), output_tokens),
            "actual_total_time": float(benchmark_row["actual_total_time"]),
            "actual_draft_time": float(benchmark_row["actual_draft_time"]),
            "actual_verify_time": float(benchmark_row["actual_verify_time"]),
        })

    return sample_rows, round_rows


def aggregate_samples(samples):
    rows = []
    for case_name, group in samples.groupby("case", sort=False):
        first = group.iloc[0]
        drafted = group["drafted_tokens"].sum()
        accepted = group["accepted_tokens"].sum()
        output = group["output_tokens"].sum()
        rounds = group["num_rounds"].sum()
        denoising = group["denoising_forward_passes"].sum()
        total_passes = group["total_forward_passes"].sum()
        rows.append({
            "case": case_name,
            "mode": first["mode"],
            "spec_len": int(first["spec_len"]),
            "lowconf_threshold": first["lowconf_threshold"],
            "max_spec_len": int(first["max_spec_len"]),
            "num_samples": len(group),
            "actual_total_time_mean": group["actual_total_time"].mean(),
            "actual_draft_time_mean": group["actual_draft_time"].mean(),
            "actual_verify_time_mean": group["actual_verify_time"].mean(),
            "acceptance_rate_percent": 100.0 * safe_div(accepted, drafted),
            "drafted_tokens_per_round": safe_div(drafted, rounds),
            "accepted_tokens_per_round": safe_div(accepted, rounds),
            "output_tokens_per_round": safe_div(output, rounds),
            "denoising_steps_per_round": safe_div(denoising, rounds),
            "denoising_steps_per_10_drafted_tokens": 10.0 * safe_div(denoising, drafted),
            "total_passes_per_10_drafted_tokens": 10.0 * safe_div(total_passes, drafted),
            "denoising_steps_per_10_output_tokens": 10.0 * safe_div(denoising, output),
            "total_passes_per_10_output_tokens": 10.0 * safe_div(total_passes, output),
            "fill_passes_per_100_output_tokens": 100.0 * safe_div(group["fill_forward_passes"].sum(), output),
            "rounds_per_100_output_tokens": 100.0 * safe_div(rounds, output),
            "fill_passes_total": group["fill_forward_passes"].sum(),
            "prefill_passes_total": group["prefill_forward_passes"].sum(),
            "cache_update_passes_total": group["cache_update_forward_passes"].sum(),
            "drafted_tokens_total": drafted,
            "output_tokens_total": output,
        })
    return pd.DataFrame(rows).sort_values("actual_total_time_mean")


def build_paired_comparison(samples):
    rows = []
    for comparison in COMPARISONS:
        baseline = samples[samples["case"] == comparison["baseline"]].set_index("problem_id")
        candidate = samples[samples["case"] == comparison["candidate"]].set_index("problem_id")
        for problem_id in baseline.index.intersection(candidate.index):
            base = baseline.loc[problem_id]
            current = candidate.loc[problem_id]
            base_steps = base["denoising_steps_per_10_drafted_tokens"]
            current_steps = current["denoising_steps_per_10_drafted_tokens"]
            rows.append({
                "problem_id": problem_id,
                "comparison": comparison["comparison"],
                "baseline": comparison["baseline"],
                "candidate": comparison["candidate"],
                "actual_speedup_vs_baseline": safe_div(base["actual_total_time"], current["actual_total_time"]),
                "actual_total_time_delta": current["actual_total_time"] - base["actual_total_time"],
                "actual_draft_time_delta": current["actual_draft_time"] - base["actual_draft_time"],
                "actual_verify_time_delta": current["actual_verify_time"] - base["actual_verify_time"],
                "acceptance_rate_delta_percent": current["acceptance_rate_percent"] - base["acceptance_rate_percent"],
                "denoising_steps_per_10_drafted_tokens_delta": current_steps - base_steps,
                "denoising_step_reduction_percent": 100.0 * safe_div(base_steps - current_steps, base_steps),
                "total_pass_reduction_per_10_drafted_tokens_percent": 100.0 * safe_div(
                    base["total_passes_per_10_drafted_tokens"]
                    - current["total_passes_per_10_drafted_tokens"],
                    base["total_passes_per_10_drafted_tokens"],
                ),
                "denoising_steps_per_10_output_tokens_delta": (
                    current["denoising_steps_per_10_output_tokens"]
                    - base["denoising_steps_per_10_output_tokens"]
                ),
                "total_passes_per_10_output_tokens_delta": (
                    current["total_passes_per_10_output_tokens"]
                    - base["total_passes_per_10_output_tokens"]
                ),
                "accepted_tokens_per_round_delta": (
                    current["accepted_tokens_per_round"] - base["accepted_tokens_per_round"]
                ),
                "output_matches_baseline": current["output_token_hash"] == base["output_token_hash"],
                "output_length_delta": current["output_tokens"] - base["output_tokens"],
                "rounds_per_100_output_tokens_delta": (
                    current["rounds_per_100_output_tokens"]
                    - base["rounds_per_100_output_tokens"]
                ),
            })
    return pd.DataFrame(rows)


def aggregate_paired_comparisons(paired):
    rows = []
    numeric_columns = [
        "actual_speedup_vs_baseline",
        "actual_total_time_delta",
        "actual_draft_time_delta",
        "actual_verify_time_delta",
        "acceptance_rate_delta_percent",
        "denoising_step_reduction_percent",
        "total_pass_reduction_per_10_drafted_tokens_percent",
        "denoising_steps_per_10_output_tokens_delta",
        "total_passes_per_10_output_tokens_delta",
        "accepted_tokens_per_round_delta",
        "output_length_delta",
        "rounds_per_100_output_tokens_delta",
    ]
    for comparison_name, group in paired.groupby("comparison", sort=False):
        first = group.iloc[0]
        row = {
            "comparison": comparison_name,
            "baseline": first["baseline"],
            "candidate": first["candidate"],
            "num_samples": len(group),
            "candidate_win_rate_percent": 100.0 * (group["actual_total_time_delta"] < 0).mean(),
            "output_match_rate_percent": 100.0 * group["output_matches_baseline"].mean(),
        }
        for column in numeric_columns:
            mean = group[column].mean()
            std = group[column].std(ddof=1)
            ci95_half_width = 1.96 * std / math.sqrt(len(group)) if len(group) > 1 else float("nan")
            row[f"{column}_mean"] = mean
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_std"] = std
            row[f"{column}_ci95_low"] = mean - ci95_half_width
            row[f"{column}_ci95_high"] = mean + ci95_half_width
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    all_samples = []
    all_rounds = []
    for case in CASES:
        output_dir = run_case(args, case)
        sample_rows, round_rows = load_case_rows(case, output_dir)
        all_samples.extend(sample_rows)
        all_rounds.extend(round_rows)

    output_root = Path(args.output_dir)
    samples = pd.DataFrame(all_samples)
    rounds = pd.DataFrame(all_rounds)
    summary = aggregate_samples(samples)
    paired = build_paired_comparison(samples)
    paired_summary = aggregate_paired_comparisons(paired)
    samples.to_csv(output_root / "denoising_per_sample.csv", index=False)
    rounds.to_csv(output_root / "denoising_per_round.csv", index=False)
    summary.to_csv(output_root / "denoising_summary.csv", index=False)
    paired.to_csv(output_root / "paired_controlled_comparisons.csv", index=False)
    paired_summary.to_csv(output_root / "paired_controlled_summary.csv", index=False)

    columns = [
        "case",
        "num_samples",
        "actual_total_time_mean",
        "actual_draft_time_mean",
        "actual_verify_time_mean",
        "acceptance_rate_percent",
        "drafted_tokens_per_round",
        "accepted_tokens_per_round",
        "output_tokens_per_round",
        "denoising_steps_per_round",
        "denoising_steps_per_10_drafted_tokens",
        "total_passes_per_10_drafted_tokens",
        "denoising_steps_per_10_output_tokens",
        "total_passes_per_10_output_tokens",
        "fill_passes_per_100_output_tokens",
        "rounds_per_100_output_tokens",
    ]
    print("\nDENOISING STEP ABLATION SUMMARY")
    print(summary[columns].to_string(index=False))
    paired_metric_columns = [
        "actual_speedup_vs_baseline_mean",
        "actual_speedup_vs_baseline_ci95_low",
        "actual_speedup_vs_baseline_ci95_high",
        "actual_total_time_delta_mean",
        "actual_total_time_delta_ci95_low",
        "actual_total_time_delta_ci95_high",
        "actual_draft_time_delta_mean",
        "actual_verify_time_delta_mean",
        "acceptance_rate_delta_percent_mean",
        "denoising_step_reduction_percent_mean",
        "total_pass_reduction_per_10_drafted_tokens_percent_mean",
        "denoising_steps_per_10_output_tokens_delta_mean",
        "total_passes_per_10_output_tokens_delta_mean",
        "accepted_tokens_per_round_delta_mean",
        "rounds_per_100_output_tokens_delta_mean",
    ]
    paired_mean_columns = [
        "comparison",
        "num_samples",
        "candidate_win_rate_percent",
        "output_match_rate_percent",
        *paired_metric_columns,
    ]
    print("\nCONTROLLED PAIRED COMPARISONS")
    print(paired_summary[paired_mean_columns].to_string(index=False))
    print(f"\nSaved: {output_root / 'denoising_summary.csv'}")


if __name__ == "__main__":
    main()
