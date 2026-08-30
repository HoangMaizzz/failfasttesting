import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from run_chunked_c6_comparison_test50 import (
    adaptive_flags,
    base_command,
    complete,
    run_streaming,
)
from run_otrc_v2_td_benchmark import PROBLEM_IDS, aggregate_method


ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_QUESTIONS = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/home/maihoang/failfasttesting/outputs_deterministic_int8_gsm8k_smoke5",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument("--num_questions", type=int, default=DEFAULT_SMOKE_QUESTIONS)
    parser.add_argument(
        "--target_quantization",
        choices=("int8_deterministic", "torchao_int8_weight_only"),
        default="int8_deterministic",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def command_args(args):
    return SimpleNamespace(
        dllm_dir=args.dllm_dir,
        target_device=args.target_device,
        drafter_device=args.drafter_device,
        target_quantization=args.target_quantization,
        log_level=args.log_level,
    )


def run_method(args, method, problem_ids):
    directory = Path(args.output_dir) / "raw" / method
    adaptive = method == "c6_annealed"
    if args.resume and complete(directory, problem_ids, adaptive):
        print(f"SKIP completed method={method}", flush=True)
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    command = base_command(
        command_args(args), "gsm8k", problem_ids, directory, 1,
    )
    if adaptive:
        command.extend(adaptive_flags(method))
    print("\n" + "=" * 96, flush=True)
    print(
        f"RUN GSM8K {args.target_quantization} | {method} | ids={problem_ids}",
        flush=True,
    )
    print("=" * 96, flush=True)
    run_streaming(command)
    if not complete(directory, problem_ids, adaptive):
        raise RuntimeError(f"incomplete smoke result: {directory}")
    return directory


def main():
    args = parse_args()
    if args.num_questions <= 0 or args.num_questions > len(PROBLEM_IDS["gsm8k"]):
        raise ValueError("--num_questions is outside the fixed GSM8K ID pool")
    problem_ids = [
        int(value) for value in PROBLEM_IDS["gsm8k"][:args.num_questions]
    ]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    directories = {
        method: run_method(args, method, problem_ids)
        for method in ("c6_annealed", "failfast")
    }
    frames = {
        method: pd.read_csv(directory / "benchmark_results.csv")
        for method, directory in directories.items()
    }
    left = frames["c6_annealed"][
        ["problem_id", "actual_algorithm_time", "output_tokens", "output_token_hash"]
    ].rename(columns={
        "actual_algorithm_time": "c6_algorithm_time_s",
        "output_tokens": "c6_output_tokens",
        "output_token_hash": "c6_output_hash",
    })
    right = frames["failfast"][
        ["problem_id", "actual_algorithm_time", "output_tokens", "output_token_hash"]
    ].rename(columns={
        "actual_algorithm_time": "failfast_algorithm_time_s",
        "output_tokens": "failfast_output_tokens",
        "output_token_hash": "failfast_output_hash",
    })
    paired = left.merge(right, on="problem_id", validate="one_to_one")
    paired["output_match"] = (
        paired["c6_output_hash"].astype(str)
        == paired["failfast_output_hash"].astype(str)
    )
    paired["c6_ms_per_output_token"] = (
        1000.0 * paired["c6_algorithm_time_s"] / paired["c6_output_tokens"].clip(lower=1)
    )
    paired["failfast_ms_per_output_token"] = (
        1000.0 * paired["failfast_algorithm_time_s"]
        / paired["failfast_output_tokens"].clip(lower=1)
    )
    paired["c6_speedup_vs_failfast"] = (
        paired["failfast_ms_per_output_token"] / paired["c6_ms_per_output_token"]
    )
    paired.to_csv(root / "paired_output_comparison.csv", index=False)

    rows = []
    for method, frame in frames.items():
        row = aggregate_method(frame, method)
        row["output_matches"] = int(paired["output_match"].sum())
        row["output_match_rate_percent"] = 100.0 * paired["output_match"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({
        "backend": args.target_quantization,
        "dataset": "gsm8k",
        "problem_ids": problem_ids,
        "decoding": "greedy",
        "timing": "measured draft + verifier + post-verifier/controller; no audit replay",
        "arguments": vars(args),
    }, indent=2), encoding="utf-8")
    print("\nPAIRED OUTPUT COMPARISON", flush=True)
    print(paired.to_string(index=False), flush=True)
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    archive = shutil.make_archive(str(root), "zip", root_dir=root)
    print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
