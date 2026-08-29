import argparse
import json
import shutil
import subprocess
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS
from run_shared_value_advantage_benchmark import (
    benchmark_command,
    failfast_baseline_command,
    run_streaming,
    shared_method_name,
)


ROOT = Path(__file__).resolve().parent
DATASETS = ("aime", "humaneval")
FEATURE_SCHEMA = "otrc_v2_2_compact_td"
POLICY_CONFIGS = (
    ("always_stop", "fixed", "frozen_stop"),
    ("compact6_fixed_stochastic", "fixed", "learned"),
    ("compact6_annealed", "annealed", "learned"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run matched FailFast, Always-STOP, Compact6 fixed stochastic, "
            "and Compact6 annealed on AIME and HumanEval."
        )
    )
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--drafter_threshold", type=float, default=0.30)
    parser.add_argument("--lowconf_threshold", type=float, default=0.50)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("none", "int8", "int4"),
        default="int8",
    )
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_aime_humaneval_c6_policy_compare_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0 < args.num_questions <= min(len(PROBLEM_IDS[d]) for d in DATASETS):
        raise ValueError("--num_questions exceeds the matched AIME/HumanEval pool")
    if args.warmup_questions != 1:
        raise ValueError("the matched benchmark requires one warmup question")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.target_device < 0 or args.drafter_device < 0:
        raise ValueError("CUDA device indices must be non-negative")
    if args.drafter_threshold != 0.30 or args.lowconf_threshold != 0.50:
        raise ValueError("this comparison fixes tau_D=0.30 and tau_F=0.50")


def shared_args(args, output_dir, exploration_policy):
    return Namespace(
        datasets=list(DATASETS),
        num_questions=args.num_questions,
        warmup_questions=args.warmup_questions,
        max_new_tokens=args.max_new_tokens,
        drafter_threshold=args.drafter_threshold,
        lowconf_threshold=args.lowconf_threshold,
        feature_schema=FEATURE_SCHEMA,
        exploration_policy=exploration_policy,
        greedy_policy=False,
        target_device=args.target_device,
        drafter_device=args.drafter_device,
        target_quantization=args.target_quantization,
        target_model_name=args.target_model_name,
        dllm_dir=args.dllm_dir,
        output_dir=str(output_dir),
        resume=args.resume,
        log_level=args.log_level,
    )


def experiment_commands(args):
    output_dir = Path(args.output_dir)
    baseline_args = shared_args(args, output_dir / "failfast", "fixed")
    commands = [("failfast", failfast_baseline_command(baseline_args))]
    for label, exploration_policy, policy_ablation in POLICY_CONFIGS:
        phase_args = shared_args(
            args,
            output_dir / label,
            exploration_policy,
        )
        commands.append(
            (
                label,
                benchmark_command(
                    phase_args,
                    policy_ablation=policy_ablation,
                    output_dir=phase_args.output_dir,
                ),
            )
        )
    return commands


def output_spec(args):
    root = Path(args.output_dir)
    specs = {
        "failfast": (
            root / "failfast" / "matched_failfast_baseline",
            "failfast_matched",
        )
    }
    for label, exploration_policy, policy_ablation in POLICY_CONFIGS:
        phase_args = shared_args(args, root / label, exploration_policy)
        specs[label] = (
            root / label,
            shared_method_name(phase_args, policy_ablation),
        )
    return specs


def build_reports(args):
    summary_frames = []
    result_frames = []
    for label, (phase_dir, method) in output_spec(args).items():
        summary = pd.read_csv(phase_dir / "dataset_method_summary.csv")
        summary["case"] = label
        summary_frames.append(summary)
        for dataset in DATASETS:
            results = pd.read_csv(
                phase_dir / "raw" / dataset / method / "benchmark_results.csv"
            )
            results["dataset"] = dataset
            results["case"] = label
            results["method"] = method
            result_frames.append(results)

    summary = pd.concat(summary_frames, ignore_index=True)
    baseline = (
        summary.loc[summary.case.eq("failfast"), ["dataset", "ms_per_output_token"]]
        .rename(columns={"ms_per_output_token": "failfast_ms_per_output_token"})
    )
    summary = summary.merge(baseline, on="dataset", how="left")
    summary["speedup_vs_failfast"] = (
        summary["failfast_ms_per_output_token"] / summary["ms_per_output_token"]
    )

    results = pd.concat(result_frames, ignore_index=True)
    results["measured_ms_per_output_token"] = (
        1000.0 * results["actual_algorithm_time"] / results["output_tokens"].clip(lower=1)
    )
    paired_baseline = (
        results.loc[
            results.case.eq("failfast"),
            ["dataset", "problem_id", "measured_ms_per_output_token", "output_token_hash"],
        ]
        .rename(
            columns={
                "measured_ms_per_output_token": "failfast_ms_per_output_token",
                "output_token_hash": "failfast_output_token_hash",
            }
        )
    )
    results = results.merge(paired_baseline, on=["dataset", "problem_id"], how="left")
    results["paired_speedup_vs_failfast"] = (
        results["failfast_ms_per_output_token"]
        / results["measured_ms_per_output_token"]
    )
    results["output_matches_failfast"] = (
        results["output_token_hash"] == results["failfast_output_token_hash"]
    )

    root = Path(args.output_dir)
    summary.to_csv(root / "four_method_dataset_summary.csv", index=False)
    results.to_csv(root / "four_method_per_question_results.csv", index=False)
    return summary


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for index, (label, command) in enumerate(experiment_commands(args), start=1):
        print("\n" + "=" * 96, flush=True)
        print(f"RUN {index}/4 | {label} | datasets={','.join(DATASETS)}", flush=True)
        print("=" * 96, flush=True)
        run_streaming(command)

    summary = build_reports(args)
    manifest = {
        "datasets": list(DATASETS),
        "num_questions": args.num_questions,
        "selected_problem_ids": {
            dataset: PROBLEM_IDS[dataset][: args.num_questions]
            for dataset in DATASETS
        },
        "cases": ["failfast", *[item[0] for item in POLICY_CONFIGS]],
        "arguments": vars(args),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nFOUR-METHOD DATASET SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
