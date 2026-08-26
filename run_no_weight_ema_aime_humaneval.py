import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "compact6_no_weight_ema_aime29_humaneval29_v1"
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

    states = validate_outputs(output_dir)
    reports = combine_reports(output_dir)
    manifest = {
        "version": VERSION,
        "arguments": vars(args),
        "method": METHOD,
        "weight_ema_enabled": False,
        "baseline_or_oracle_executed": False,
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
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
