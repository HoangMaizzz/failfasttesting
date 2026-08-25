import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_math_feature_ablation_benchmark import adaptive_args


ROOT = Path(__file__).resolve().parent
GSM8K_SIZE = 1319


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=int, default=1202)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_greedy_losslessness_audit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--matched_num_questions", type=int, default=50)
    parser.add_argument("--log_level", default="INFO")

    # Method A settings used by the matched GSM8K report.
    parser.add_argument("--adaptive_max_refinement_steps", type=int, default=16)
    parser.add_argument("--adaptive_learning_rate", type=float, default=0.02)
    parser.add_argument("--adaptive_mc_learning_rate", type=float, default=0.01)
    parser.add_argument("--adaptive_mc_mix", type=float, default=0.5)
    parser.add_argument("--adaptive_update_mode", default="mixed")
    parser.add_argument("--adaptive_rho_alpha", type=float, default=0.05)
    parser.add_argument("--adaptive_risk_beta", type=float, default=1.0)
    parser.add_argument("--adaptive_stop_probability_threshold", type=float, default=0.75)
    parser.add_argument("--adaptive_uncertainty_prior", type=float, default=1.0)
    parser.add_argument("--adaptive_epistemic_scale", type=float, default=0.1)
    parser.add_argument("--adaptive_q_margin", type=float, default=0.0)
    parser.add_argument("--adaptive_explore_epsilon", type=float, default=0.10)
    parser.add_argument("--adaptive_explore_min", type=float, default=0.01)
    parser.add_argument("--adaptive_explore_decay", type=float, default=0.998)
    parser.add_argument("--adaptive_warmup_rounds", type=int, default=20)
    parser.add_argument("--adaptive_early_stop_min_observations", type=int, default=32)
    parser.add_argument("--adaptive_min_action_probability", type=float, default=0.10)
    parser.add_argument("--adaptive_max_importance_weight", type=float, default=5.0)
    return parser.parse_args()


def run_streaming(command):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def matched_problem_ids(args):
    population = list(range(1, GSM8K_SIZE))
    return sorted(random.Random(args.sample_seed).sample(
        population,
        args.matched_num_questions,
    ))


def base_command(args, output_dir, problem_ids):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", "gsm8k",
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(problem_id) for problem_id in problem_ids],
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--drafter_thresholds", "0.05",
        "--sweep_lowconf_threshold", "0.45",
        "--sweep_max_spec_len", "60",
        "--sweep_incr_len", "8",
        "--seed", str(args.seed),
        "--audit_greedy_consistency",
        "--audit_greedy_problem_ids", str(args.problem_id),
        "--log_verifier_calls",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def compare_token_traces(failfast, method_a):
    left = failfast.rename(columns={"token_id": "failfast_token"})
    right = method_a.rename(columns={"token_id": "method_a_token"})
    merged = left[["output_position", "failfast_token"]].merge(
        right[["output_position", "method_a_token"]],
        on="output_position",
        how="outer",
    ).sort_values("output_position")
    merged["matches"] = merged["failfast_token"].eq(merged["method_a_token"])
    differences = merged[~merged["matches"]].copy()
    if differences.empty:
        return pd.DataFrame([{
            "first_different_position": None,
            "failfast_token": None,
            "method_a_token": None,
            "failfast_output_length": len(failfast),
            "method_a_output_length": len(method_a),
        }])
    first = differences.iloc[0]
    return pd.DataFrame([{
        "first_different_position": int(first["output_position"]),
        "failfast_token": first["failfast_token"],
        "method_a_token": first["method_a_token"],
        "failfast_output_length": len(failfast),
        "method_a_output_length": len(method_a),
    }])


def summarize_audit(method, rows):
    internal = rows[rows["emitted_matches_batched"].eq(0)]
    causal = rows[rows["batched_matches_prefix"].eq(0)]
    return {
        "method": method,
        "audited_tokens": len(rows),
        "internal_token_mismatches": len(internal),
        "batched_prefix_argmax_mismatches": len(causal),
        "first_internal_mismatch_position": (
            internal["absolute_output_position"].min() if not internal.empty else None
        ),
        "first_batched_prefix_mismatch_position": (
            causal["absolute_output_position"].min() if not causal.empty else None
        ),
        "minimum_batched_margin": rows["batched_margin"].min(),
        "minimum_prefix_margin": rows["prefix_margin"].min(),
    }


def main():
    args = parse_args()
    method_a_problem_ids = matched_problem_ids(args)
    if args.problem_id not in method_a_problem_ids:
        raise ValueError(
            f"problem_id={args.problem_id} is not in the matched "
            f"{args.matched_num_questions}-question sample"
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    phases = {
        "failfast": ([args.problem_id], []),
        "method_a": (method_a_problem_ids, adaptive_args(args, ())),
    }
    summaries = []
    traces = {}
    all_mismatches = []
    for method, (problem_ids, extra_args) in phases.items():
        phase_dir = output_dir / "raw" / method
        phase_dir.mkdir(parents=True)
        print(f"\nRUN {method} | problem_id={args.problem_id}", flush=True)
        run_streaming(
            base_command(args, phase_dir, problem_ids) + list(extra_args)
        )

        audit = pd.read_csv(phase_dir / "greedy_consistency_audit.csv")
        trace = pd.read_csv(phase_dir / "output_token_trace.csv")
        benchmark = pd.read_csv(phase_dir / "benchmark_results.csv")
        benchmark = benchmark[benchmark["problem_id"].eq(args.problem_id)]
        if len(benchmark) != 1:
            raise RuntimeError(
                f"{method} produced {len(benchmark)} rows for "
                f"problem_id={args.problem_id}"
            )
        traces[method] = trace
        summary = summarize_audit(method, audit)
        summary.update({
            "output_tokens": int(benchmark.iloc[0]["output_tokens"]),
            "output_token_hash": benchmark.iloc[0]["output_token_hash"],
            "predicted_answer": benchmark.iloc[0]["predicted_answer"],
        })
        summaries.append(summary)
        mismatches = audit[
            audit["emitted_matches_batched"].eq(0)
            | audit["batched_matches_prefix"].eq(0)
        ].copy()
        mismatches.insert(0, "method", method)
        all_mismatches.append(mismatches)

    summary_frame = pd.DataFrame(summaries)
    comparison = compare_token_traces(traces["failfast"], traces["method_a"])
    mismatch_frame = pd.concat(all_mismatches, ignore_index=True)
    summary_frame.to_csv(output_dir / "greedy_consistency_summary.csv", index=False)
    comparison.to_csv(output_dir / "cross_method_first_difference.csv", index=False)
    mismatch_frame.to_csv(output_dir / "greedy_consistency_mismatches.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({
        "problem_id": args.problem_id,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "method_a_problem_ids": method_a_problem_ids,
        "purpose": "diagnose greedy losslessness independently of benchmark timing",
    }, indent=2), encoding="utf-8")

    print("\nGREEDY CONSISTENCY SUMMARY", flush=True)
    print(summary_frame.to_string(index=False), flush=True)
    print("\nCROSS-METHOD FIRST DIFFERENCE", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
