import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "compact6_no_weight_ema_vs_failfast_aime29_humaneval29_v2"
METHOD = "otrc_v2_2_compact_factual_no_bootstrap"
DATASET_COUNTS = {"aime": 29, "humaneval": 29}
REPORT_FILES = (
    "dataset_method_summary.csv",
    "feature_statistics.csv",
    "feature_conditioning.csv",
    "learning_dynamics.csv",
    "weight_trajectory.csv",
    "policy_ema_summary.csv",
    "policy_ema_learning_dynamics.csv",
    "snapshot_invariants.csv",
    "confidence_diagnostics.csv",
    "factual_target_summary.csv",
    "factual_target_learning_dynamics.csv",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_no_weight_ema_aime29_humaneval29"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def validate_args(args):
    if args.warmup_questions != 1:
        raise ValueError("this matched benchmark requires one warmup question")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    for dataset, count in DATASET_COUNTS.items():
        if len(PROBLEM_IDS[dataset]) < count:
            raise ValueError(f"not enough fixed problem IDs for {dataset}")


def benchmark_command(args, dataset, count, output_dir):
    command = [
        sys.executable,
        "-u",
        "run_otrc_v2_td_benchmark.py",
        "--datasets",
        dataset,
        "--num_questions",
        str(count),
        "--feature_schema",
        "otrc_v2_2_compact_td",
        "--credit_assignment",
        "verifier_boundary_factual_no_bootstrap",
        "--rho_warmup_boundaries",
        "0",
        "--policy_weight_ema_beta",
        "0.0",
        "--policy_weight_ema_mode",
        "global_step",
        "--warmup_questions",
        str(args.warmup_questions),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--spec_len",
        "8",
        "--incr_len",
        "8",
        "--max_spec_len",
        "60",
        "--block_size",
        "32",
        "--small_block_size",
        "8",
        "--target_model_name",
        args.target_model_name,
        "--dllm_dir",
        args.dllm_dir,
        "--drafter_threshold",
        "0.05",
        "--lowconf_threshold",
        "0.45",
        "--adaptive_learning_rate",
        "0.02",
        "--adaptive_mc_learning_rate",
        "0.01",
        "--adaptive_mc_mix",
        "0.5",
        "--adaptive_update_mode",
        "mixed",
        "--adaptive_rho_alpha",
        "0.05",
        "--adaptive_factual_ema_alpha",
        "0.2",
        "--adaptive_risk_beta",
        "1.0",
        "--adaptive_stop_probability_threshold",
        "0.75",
        "--adaptive_uncertainty_prior",
        "1.0",
        "--adaptive_epistemic_scale",
        "0.1",
        "--adaptive_explore_epsilon",
        "0.10",
        "--adaptive_explore_min",
        "0.01",
        "--adaptive_explore_decay",
        "0.998",
        "--adaptive_warmup_rounds",
        "20",
        "--adaptive_early_stop_min_observations",
        "32",
        "--adaptive_min_action_probability",
        "0.10",
        "--adaptive_max_importance_weight",
        "5.0",
        "--adaptive_weight_snapshot_interval",
        "100",
        "--seed",
        "42",
        "--output_dir",
        str(output_dir),
        "--log_level",
        args.log_level,
    ]
    if args.resume:
        command.append("--resume")
    return command


def failfast_command(args, dataset, count, output_dir):
    problem_ids = PROBLEM_IDS[dataset][:count]
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name",
        dataset,
        "--num_questions",
        str(count),
        "--problem_ids",
        *[str(problem_id) for problem_id in problem_ids],
        "--warmup_questions",
        str(args.warmup_questions),
        "--benchmark_modes",
        "dllm_ar",
        "--dllm_variant",
        "failfast",
        "--decoding_strategy",
        "greedy",
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--spec_len",
        "8",
        "--block_size",
        "32",
        "--small_block_size",
        "8",
        "--target_model_name",
        args.target_model_name,
        "--dllm_dir",
        args.dllm_dir,
        "--drafter_thresholds",
        "0.05",
        "--sweep_lowconf_threshold",
        "0.45",
        "--sweep_max_spec_len",
        "60",
        "--sweep_incr_len",
        "8",
        "--seed",
        "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir",
        str(output_dir),
        "--log_level",
        args.log_level,
    ]


def run_streaming(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
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


def combine_reports(output_dir):
    combined = {}
    for filename in REPORT_FILES:
        frames = []
        for dataset in DATASET_COUNTS:
            path = output_dir / dataset / filename
            if path.exists() and path.stat().st_size:
                frames.append(pd.read_csv(path))
        if frames:
            combined_path = output_dir / filename
            pd.concat(frames, ignore_index=True).to_csv(
                combined_path,
                index=False,
            )
            combined[filename] = len(pd.concat(frames, ignore_index=True))
    return combined


def phase_complete(directory, expected_ids):
    result_path = directory / "benchmark_results.csv"
    if not result_path.exists() or not result_path.stat().st_size:
        return False
    try:
        results = pd.read_csv(result_path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return set(results.problem_id.astype(int)) == set(expected_ids)


def paired_comparison(output_dir):
    per_problem_frames = []
    dataset_rows = []
    for dataset, count in DATASET_COUNTS.items():
        no_weight = pd.read_csv(
            output_dir
            / dataset
            / "raw"
            / dataset
            / METHOD
            / "benchmark_results.csv"
        )
        failfast = pd.read_csv(
            output_dir / "failfast" / dataset / "benchmark_results.csv"
        )
        columns = [
            "problem_id",
            "actual_algorithm_time",
            "actual_draft_time",
            "actual_verify_time",
            "actual_post_verify_time",
            "output_tokens",
            "accepted_tokens",
            "drafted_tokens",
            "num_speculation_rounds",
            "total_num_forward_passes",
            "acceptance_rate_percent",
            "output_token_hash",
            "is_correct",
        ]
        paired = failfast[columns].merge(
            no_weight[columns],
            on="problem_id",
            suffixes=("_failfast", "_no_weight_ema"),
            validate="one_to_one",
        )
        if len(paired) != count:
            raise RuntimeError(f"paired coverage mismatch for {dataset}")
        paired.insert(0, "dataset", dataset)
        paired["failfast_ms_per_output_token"] = (
            1000.0
            * paired.actual_algorithm_time_failfast
            / paired.output_tokens_failfast.clip(lower=1)
        )
        paired["no_weight_ema_ms_per_output_token"] = (
            1000.0
            * paired.actual_algorithm_time_no_weight_ema
            / paired.output_tokens_no_weight_ema.clip(lower=1)
        )
        paired["no_weight_ema_speedup_vs_failfast"] = (
            paired.failfast_ms_per_output_token
            / paired.no_weight_ema_ms_per_output_token
        )
        paired["no_weight_ema_wins"] = (
            paired.no_weight_ema_ms_per_output_token
            < paired.failfast_ms_per_output_token
        )
        paired["output_match"] = (
            paired.output_token_hash_failfast
            == paired.output_token_hash_no_weight_ema
        )
        per_problem_frames.append(paired)

        ff_tokens = float(paired.output_tokens_failfast.sum())
        nw_tokens = float(paired.output_tokens_no_weight_ema.sum())
        ff_mspt = (
            1000.0 * paired.actual_algorithm_time_failfast.sum()
            / max(1.0, ff_tokens)
        )
        nw_mspt = (
            1000.0 * paired.actual_algorithm_time_no_weight_ema.sum()
            / max(1.0, nw_tokens)
        )
        speedups = paired.no_weight_ema_speedup_vs_failfast.clip(lower=1e-12)
        dataset_rows.append({
            "dataset": dataset,
            "num_questions": len(paired),
            "failfast_ms_per_output_token": ff_mspt,
            "no_weight_ema_ms_per_output_token": nw_mspt,
            "pooled_speedup_vs_failfast": ff_mspt / nw_mspt,
            "geometric_mean_speedup_vs_failfast": float(
                math.exp(float(speedups.map(math.log).mean()))
            ),
            "no_weight_ema_win_rate_percent": (
                100.0 * float(paired.no_weight_ema_wins.mean())
            ),
            "output_match_rate_percent": 100.0 * float(paired.output_match.mean()),
            "failfast_draft_time_s": float(paired.actual_draft_time_failfast.sum()),
            "no_weight_ema_draft_time_s": float(
                paired.actual_draft_time_no_weight_ema.sum()
            ),
            "failfast_verify_time_s": float(paired.actual_verify_time_failfast.sum()),
            "no_weight_ema_verify_time_s": float(
                paired.actual_verify_time_no_weight_ema.sum()
            ),
            "failfast_acceptance_rate_percent": (
                100.0 * paired.accepted_tokens_failfast.sum()
                / max(1.0, float(paired.drafted_tokens_failfast.sum()))
            ),
            "no_weight_ema_acceptance_rate_percent": (
                100.0 * paired.accepted_tokens_no_weight_ema.sum()
                / max(1.0, float(paired.drafted_tokens_no_weight_ema.sum()))
            ),
            "failfast_verifier_rounds_per_100_tokens": (
                100.0 * paired.num_speculation_rounds_failfast.sum()
                / max(1.0, ff_tokens)
            ),
            "no_weight_ema_verifier_rounds_per_100_tokens": (
                100.0 * paired.num_speculation_rounds_no_weight_ema.sum()
                / max(1.0, nw_tokens)
            ),
            "failfast_draft_passes_per_100_tokens": (
                100.0 * paired.total_num_forward_passes_failfast.sum()
                / max(1.0, ff_tokens)
            ),
            "no_weight_ema_draft_passes_per_100_tokens": (
                100.0 * paired.total_num_forward_passes_no_weight_ema.sum()
                / max(1.0, nw_tokens)
            ),
        })

    per_problem = pd.concat(per_problem_frames, ignore_index=True)
    dataset_summary = pd.DataFrame(dataset_rows)
    ff_tokens = float(per_problem.output_tokens_failfast.sum())
    nw_tokens = float(per_problem.output_tokens_no_weight_ema.sum())
    ff_mspt = (
        1000.0 * per_problem.actual_algorithm_time_failfast.sum()
        / max(1.0, ff_tokens)
    )
    nw_mspt = (
        1000.0 * per_problem.actual_algorithm_time_no_weight_ema.sum()
        / max(1.0, nw_tokens)
    )
    overall = pd.DataFrame([{
        "datasets": len(DATASET_COUNTS),
        "num_questions": len(per_problem),
        "failfast_ms_per_output_token": ff_mspt,
        "no_weight_ema_ms_per_output_token": nw_mspt,
        "pooled_speedup_vs_failfast": ff_mspt / nw_mspt,
        "macro_speedup_vs_failfast": float(
            dataset_summary.pooled_speedup_vs_failfast.mean()
        ),
        "no_weight_ema_win_rate_percent": (
            100.0 * float(per_problem.no_weight_ema_wins.mean())
        ),
        "output_match_rate_percent": 100.0 * float(per_problem.output_match.mean()),
    }])
    per_problem.to_csv(output_dir / "paired_per_problem.csv", index=False)
    dataset_summary.to_csv(
        output_dir / "paired_dataset_comparison.csv",
        index=False,
    )
    overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    return dataset_summary, overall


def validate_outputs(output_dir):
    states = {}
    for dataset, count in DATASET_COUNTS.items():
        dataset_dir = output_dir / dataset
        result_path = (
            dataset_dir / "raw" / dataset / METHOD / "benchmark_results.csv"
        )
        state_path = (
            dataset_dir
            / "raw"
            / dataset
            / METHOD
            / "adaptive_td_runtime_state.json"
        )
        results = pd.read_csv(result_path)
        expected_ids = set(PROBLEM_IDS[dataset][:count])
        if set(results.problem_id.astype(int)) != expected_ids:
            raise RuntimeError(f"measured problem IDs do not match {dataset}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if float(state.get("policy_weight_ema_beta", -1.0)) != 0.0:
            raise RuntimeError(f"Weight EMA was not disabled for {dataset}")
        failfast_path = output_dir / "failfast" / dataset
        if not phase_complete(failfast_path, expected_ids):
            raise RuntimeError(f"FailFast output is incomplete for {dataset}")
        states[dataset] = {
            "num_questions": count,
            "problem_ids": sorted(expected_ids),
            "decision_count": int(state.get("decision_count", 0)),
        }
    return states


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for dataset, count in DATASET_COUNTS.items():
        print("\n" + "=" * 100, flush=True)
        print(
            f"RUN NO-WEIGHT-EMA | {dataset} | samples={count}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(
            benchmark_command(args, dataset, count, output_dir / dataset)
        )
        failfast_dir = output_dir / "failfast" / dataset
        expected_ids = PROBLEM_IDS[dataset][:count]
        if args.resume and phase_complete(failfast_dir, expected_ids):
            print(f"RESUME FAILFAST | {dataset}", flush=True)
        else:
            failfast_dir.mkdir(parents=True, exist_ok=True)
            print(f"RUN FAILFAST | {dataset} | samples={count}", flush=True)
            run_streaming(
                failfast_command(args, dataset, count, failfast_dir)
            )

    states = validate_outputs(output_dir)
    reports = combine_reports(output_dir)
    comparison, overall = paired_comparison(output_dir)
    manifest = {
        "version": VERSION,
        "arguments": vars(args),
        "method": METHOD,
        "weight_ema_enabled": False,
        "failfast_baseline_executed": True,
        "oracle_executed": False,
        "datasets": states,
        "combined_report_rows": reports,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "no_weight_ema_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    summary = pd.read_csv(output_dir / "dataset_method_summary.csv")
    print("\nNO-WEIGHT-EMA DATASET SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nPAIRED DATASET COMPARISON", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("\nOVERALL COMPARISON", flush=True)
    print(overall.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
