import argparse
import math
from pathlib import Path

import pandas as pd

from run_denoising_step_ablation import aggregate_samples, load_case_rows, run_case


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--output_dir", default="/content/failfasttesting/outputs_disabled_spec_len_ablation")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def build_cases(args):
    threshold_name = f"{args.lowconf_threshold:.2f}".replace(".", "p")
    return (
        {
            "case": f"disabled_dynamic_spec10_tau{threshold_name}",
            "mode": "disabled",
            "spec_len": 10,
            "lowconf_threshold": args.lowconf_threshold,
            "max_spec_len": args.max_spec_len,
        },
        {
            "case": f"disabled_dynamic_spec5_tau{threshold_name}",
            "mode": "disabled",
            "spec_len": 5,
            "lowconf_threshold": args.lowconf_threshold,
            "max_spec_len": args.max_spec_len,
        },
    )


def paired_comparison(samples, baseline_name, candidate_name):
    metrics = [
        "actual_total_time",
        "actual_draft_time",
        "actual_verify_time",
        "num_rounds",
        "drafted_tokens",
        "accepted_tokens",
        "acceptance_rate_percent",
        "denoising_forward_passes",
        "total_forward_passes",
        "denoising_steps_per_10_output_tokens",
        "total_passes_per_10_output_tokens",
        "rounds_per_100_output_tokens",
    ]
    baseline = samples[samples["case"] == baseline_name].set_index("problem_id")
    candidate = samples[samples["case"] == candidate_name].set_index("problem_id")
    rows = []
    for problem_id in baseline.index.intersection(candidate.index):
        base = baseline.loc[problem_id]
        current = candidate.loc[problem_id]
        row = {
            "problem_id": problem_id,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "actual_speedup_spec5_vs_spec10": (
                base["actual_total_time"] / current["actual_total_time"]
                if current["actual_total_time"]
                else 0.0
            ),
            "spec5_faster": current["actual_total_time"] < base["actual_total_time"],
            "output_matches_spec10": current["output_token_hash"] == base["output_token_hash"],
            "output_length_delta": current["output_tokens"] - base["output_tokens"],
        }
        for metric in metrics:
            row[f"{metric}_spec10"] = base[metric]
            row[f"{metric}_spec5"] = current[metric]
            row[f"{metric}_delta_spec5_minus_spec10"] = current[metric] - base[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired(paired):
    metric_columns = [
        "actual_speedup_spec5_vs_spec10",
        *[
            column
            for column in paired.columns
            if column.endswith("_delta_spec5_minus_spec10")
        ],
    ]
    row = {
        "num_samples": len(paired),
        "spec5_win_rate_percent": 100.0 * paired["spec5_faster"].mean(),
        "output_match_rate_percent": 100.0 * paired["output_matches_spec10"].mean(),
    }
    for column in metric_columns:
        mean = paired[column].mean()
        std = paired[column].std(ddof=1)
        half_width = 1.96 * std / math.sqrt(len(paired)) if len(paired) > 1 else float("nan")
        row[f"{column}_mean"] = mean
        row[f"{column}_median"] = paired[column].median()
        row[f"{column}_std"] = std
        row[f"{column}_ci95_low"] = mean - half_width
        row[f"{column}_ci95_high"] = mean + half_width
    return pd.DataFrame([row])


def main():
    args = parse_args()
    cases = build_cases(args)
    sample_rows = []
    round_rows = []
    for case in cases:
        case_output_dir = run_case(args, case)
        case_samples, case_rounds = load_case_rows(case, case_output_dir)
        sample_rows.extend(case_samples)
        round_rows.extend(case_rounds)

    output_dir = Path(args.output_dir)
    samples = pd.DataFrame(sample_rows)
    rounds = pd.DataFrame(round_rows)
    aggregate = aggregate_samples(samples)
    paired = paired_comparison(samples, cases[0]["case"], cases[1]["case"])
    paired_summary = summarize_paired(paired)

    samples.to_csv(output_dir / "disabled_spec_len_per_sample.csv", index=False)
    rounds.to_csv(output_dir / "disabled_spec_len_per_round.csv", index=False)
    aggregate.to_csv(output_dir / "disabled_spec_len_aggregate.csv", index=False)
    paired.to_csv(output_dir / "disabled_spec_len_paired.csv", index=False)
    paired_summary.to_csv(output_dir / "disabled_spec_len_paired_summary.csv", index=False)

    aggregate_columns = [
        "case",
        "num_samples",
        "actual_total_time_mean",
        "actual_draft_time_mean",
        "actual_verify_time_mean",
        "acceptance_rate_percent",
        "drafted_tokens_per_round",
        "accepted_tokens_per_round",
        "output_tokens_per_round",
        "denoising_steps_per_10_output_tokens",
        "total_passes_per_10_output_tokens",
        "rounds_per_100_output_tokens",
    ]
    summary_columns = [
        "num_samples",
        "spec5_win_rate_percent",
        "output_match_rate_percent",
        "actual_speedup_spec5_vs_spec10_mean",
        "actual_speedup_spec5_vs_spec10_ci95_low",
        "actual_speedup_spec5_vs_spec10_ci95_high",
        "actual_total_time_delta_spec5_minus_spec10_mean",
        "actual_total_time_delta_spec5_minus_spec10_ci95_low",
        "actual_total_time_delta_spec5_minus_spec10_ci95_high",
        "actual_draft_time_delta_spec5_minus_spec10_mean",
        "actual_verify_time_delta_spec5_minus_spec10_mean",
        "num_rounds_delta_spec5_minus_spec10_mean",
        "total_forward_passes_delta_spec5_minus_spec10_mean",
    ]
    print("\nDISABLED SPEC_LEN AGGREGATE")
    print(aggregate[aggregate_columns].to_string(index=False))
    print("\nPAIRED SPEC_LEN=5 VS SPEC_LEN=10")
    print(paired_summary[summary_columns].to_string(index=False))
    print(f"\nSaved: {output_dir / 'disabled_spec_len_paired_summary.csv'}")


if __name__ == "__main__":
    main()
