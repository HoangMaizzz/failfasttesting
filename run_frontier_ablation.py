import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


FRONTIER_MODES = ("disabled", "mask_efficiency", "frontier", "cost_aware")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--spec_len", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dllm_dir", type=str, default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--output_dir", type=str, default="/content/failfasttesting/outputs_frontier_ablation")
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_mode(args, mode):
    output_dir = Path(args.output_dir) / f"problem_{args.problem_id}" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    if csv_path.exists():
        csv_path.unlink()

    cmd = [
        sys.executable,
        "failfast.py",
        "--dataset_name", "gsm8k",
        "--num_questions", str(args.problem_id + 1),
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", str(args.spec_len),
        "--block_size", str(args.block_size),
        "--small_block_size", str(args.small_block_size),
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--frontier_stop_mode", mode,
        "--frontier_min_steps", str(args.frontier_min_steps),
        "--frontier_patience", str(args.frontier_patience),
        "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]

    print(f"\n{'=' * 90}")
    print(f"FRONTIER MODE: {mode} | PROBLEM ID: {args.problem_id}")
    print(f"{'=' * 90}", flush=True)
    subprocess.run(cmd, check=True)

    df = pd.read_csv(csv_path)
    row = df[(df["problem_id"] == args.problem_id) & (df["mode"] == "dllm_ar")].iloc[-1].to_dict()
    row["frontier_stop_mode"] = mode
    return row


def main():
    args = parse_args()
    rows = [run_mode(args, mode) for mode in FRONTIER_MODES]
    summary = pd.DataFrame(rows)
    columns = [
        "frontier_stop_mode",
        "problem_id",
        "actual_total_time",
        "actual_draft_time",
        "actual_verify_time",
        "actual_draft_verify_ratio",
        "acceptance_rate_percent",
        "actual_speedup_vs_AR",
        "theo_total_time",
        "theo_speedup_vs_AR",
    ]
    summary = summary[columns].sort_values("actual_total_time")
    summary_path = Path(args.output_dir) / f"problem_{args.problem_id}" / "frontier_4mode_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFINAL FRONTIER ABLATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
