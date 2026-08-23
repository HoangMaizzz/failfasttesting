import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_adaptive_td_benchmark import (
    DATASET_SIZES,
    aggregate,
    paired_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SIZES),
        default=["gsm8k"],
    )
    parser.add_argument("--train_questions", type=int, default=50)
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--adaptive-max-refinement-steps", type=int, default=16)
    parser.add_argument("--adaptive-learning-rate", type=float, default=0.02)
    parser.add_argument("--adaptive-mc-learning-rate", type=float, default=0.01)
    parser.add_argument("--adaptive-mc-mix", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-update-mode",
        choices=("td", "factual_return", "mixed"),
        default="mixed",
    )
    parser.add_argument("--adaptive-rho-alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-risk-beta", type=float, default=1.0)
    parser.add_argument("--adaptive-uncertainty-prior", type=float, default=1.0)
    parser.add_argument("--adaptive-epistemic-scale", type=float, default=0.1)
    parser.add_argument("--adaptive-q-margin", type=float, default=0.0)
    parser.add_argument("--adaptive-min-action-probability", type=float, default=0.10)
    parser.add_argument("--adaptive-max-importance-weight", type=float, default=5.0)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_symmetric_td_test50",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.train_questions <= 0 or args.num_questions <= 0:
        raise ValueError("train and evaluation sample counts must be positive")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("--spec_len and --incr_len must be positive")
    if not 0.0 < args.adaptive_min_action_probability <= 0.5:
        raise ValueError("--adaptive-min-action-probability must be in (0, 0.5]")
    if args.adaptive_max_importance_weight < 1.0:
        raise ValueError("--adaptive-max-importance-weight must be at least 1")
    required = args.train_questions + args.num_questions
    for dataset in args.datasets:
        if required > DATASET_SIZES[dataset]:
            raise ValueError(f"{dataset} has fewer than {required} samples")


def sampled_problem_ids(args):
    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        population = list(range(DATASET_SIZES[dataset]))
        rng = random.Random(args.sample_seed + 1009 * dataset_index)
        selected = rng.sample(
            population,
            args.train_questions + args.num_questions,
        )
        result[dataset] = {
            "train": sorted(selected[:args.train_questions]),
            "evaluation": sorted(selected[args.train_questions:]),
        }
    return result


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


def common_command(args, dataset, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(problem_id) for problem_id in problem_ids],
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
        "--seed", str(args.seed),
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def adaptive_arguments(args):
    return [
        "--adaptive-td",
        "--adaptive-policy-mode", "symmetric",
        "--adaptive-max-refinement-steps", str(args.adaptive_max_refinement_steps),
        "--adaptive-learning-rate", str(args.adaptive_learning_rate),
        "--adaptive-mc-learning-rate", str(args.adaptive_mc_learning_rate),
        "--adaptive-mc-mix", str(args.adaptive_mc_mix),
        "--adaptive-update-mode", args.adaptive_update_mode,
        "--adaptive-rho-alpha", str(args.adaptive_rho_alpha),
        "--adaptive-risk-beta", str(args.adaptive_risk_beta),
        "--adaptive-uncertainty-prior", str(args.adaptive_uncertainty_prior),
        "--adaptive-epistemic-scale", str(args.adaptive_epistemic_scale),
        "--adaptive-q-margin", str(args.adaptive_q_margin),
        "--adaptive-min-action-probability",
        str(args.adaptive_min_action_probability),
        "--adaptive-max-importance-weight",
        str(args.adaptive_max_importance_weight),
        "--adaptive-use-step-feature",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
    ]


def run_phase(args, dataset, phase, method, problem_ids, state_path=None):
    output_dir = Path(args.output_dir) / phase / dataset / method
    result_path = output_dir / "benchmark_results.csv"
    expected_ids = sorted(problem_ids)
    if args.resume and result_path.exists():
        rows = pd.read_csv(result_path)
        if sorted(rows["problem_id"].astype(int).tolist()) == expected_ids:
            print(f"RESUME {phase} | {dataset} | {method}", flush=True)
            return rows
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = common_command(args, dataset, problem_ids, output_dir)
    if method != "failfast":
        command.extend(adaptive_arguments(args))
    if state_path is not None:
        command.extend([
            "--adaptive-state-path", str(state_path),
            "--adaptive-freeze",
        ])
    print("\n" + "=" * 100, flush=True)
    print(
        f"RUN {phase} | {dataset} | {method} | samples={len(problem_ids)}",
        flush=True,
    )
    print("=" * 100, flush=True)
    run_streaming(command, Path(__file__).resolve().parent)
    return pd.read_csv(result_path)


def annotate(rows, dataset, method):
    rows = rows.copy()
    rows["dataset"] = dataset
    rows["method"] = method
    rows["measured_time_s"] = (
        pd.to_numeric(rows["actual_draft_time"], errors="coerce")
        + pd.to_numeric(rows["actual_verify_time"], errors="coerce")
        + pd.to_numeric(rows["actual_post_verify_time"], errors="coerce")
    )
    rows["measured_ms_per_output_token"] = (
        1000.0 * rows["measured_time_s"] / rows["output_tokens"]
    )
    rows["e2e_ms_per_output_token"] = (
        1000.0 * rows["actual_e2e_time"] / rows["output_tokens"]
    )
    return rows


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = sampled_problem_ids(args)
    (output_dir / "sampled_problem_ids.json").write_text(
        json.dumps(problem_ids, indent=2),
        encoding="utf-8",
    )

    training_rows = []
    evaluation_rows = []
    for dataset in args.datasets:
        train_ids = problem_ids[dataset]["train"]
        evaluation_ids = problem_ids[dataset]["evaluation"]
        trained = run_phase(
            args,
            dataset,
            "training",
            "adaptive_td",
            train_ids,
        )
        training_rows.append(annotate(trained, dataset, "adaptive_td_training"))
        state_path = (
            output_dir
            / "training"
            / dataset
            / "adaptive_td"
            / "adaptive_td_runtime_state.json"
        )
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        failfast_rows = run_phase(
            args,
            dataset,
            "evaluation",
            "failfast",
            evaluation_ids,
        )
        adaptive_rows = run_phase(
            args,
            dataset,
            "evaluation",
            "adaptive_td",
            evaluation_ids,
            state_path=state_path,
        )
        evaluation_rows.extend([
            annotate(failfast_rows, dataset, "failfast"),
            annotate(adaptive_rows, dataset, "adaptive_td"),
        ])

    training = pd.concat(training_rows, ignore_index=True, sort=False)
    evaluation = pd.concat(evaluation_rows, ignore_index=True, sort=False)
    training_summary = aggregate(training)
    evaluation_summary = aggregate(evaluation)
    paired = paired_comparison(evaluation)
    paired_summary = paired.groupby("dataset", sort=False).agg(
        num_samples=("problem_id", "size"),
        geometric_speedup=(
            "measured_speedup_vs_failfast",
            lambda values: float(np.exp(np.log(values).mean())),
        ),
        adaptive_win_rate_percent=("adaptive_wins", lambda values: 100.0 * values.mean()),
        output_match_rate_percent=("output_match", lambda values: 100.0 * values.mean()),
    ).reset_index()

    training.to_csv(output_dir / "training_per_problem_results.csv", index=False)
    training_summary.to_csv(output_dir / "training_dataset_summary.csv", index=False)
    evaluation.to_csv(output_dir / "evaluation_per_problem_results.csv", index=False)
    evaluation_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)
    print("\nFROZEN EVALUATION SUMMARY")
    print(evaluation_summary.to_string(index=False))
    print("\nPAIRED SUMMARY")
    print(paired_summary.to_string(index=False))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
