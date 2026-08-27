import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "compact6_shared_value_explicit_advantage_active_block_v2"
METHOD = "otrc_v2_2_compact_factual_no_bootstrap_shared_value_advantage"
DATASETS = ("math", "gsm8k", "aime", "humaneval")
VALUE_LEARNING_RATE = 0.015
ADVANTAGE_LEARNING_RATE = 0.02


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument("--num_questions", type=int, default=25)
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
            "outputs_shared_value_advantage_active_block_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    available = min(len(PROBLEM_IDS[dataset]) for dataset in args.datasets)
    if args.num_questions > available:
        raise ValueError(f"--num_questions cannot exceed {available}")
    if args.warmup_questions != 1:
        raise ValueError("the matched benchmark requires one warmup question")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")


def benchmark_command(args):
    command = [
        sys.executable,
        "-u",
        "run_otrc_v2_td_benchmark.py",
        "--datasets",
        *args.datasets,
        "--num_questions",
        str(args.num_questions),
        "--feature_schema",
        "otrc_v2_2_compact_td",
        "--credit_assignment",
        "verifier_boundary_factual_no_bootstrap",
        "--value_parameterization",
        "shared_value_advantage",
        "--shared_value_learning_rate",
        str(VALUE_LEARNING_RATE),
        "--shared_advantage_learning_rate",
        str(ADVANTAGE_LEARNING_RATE),
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
        "--rho_warmup_boundaries",
        "0",
        "--policy_weight_ema_beta",
        "0.0",
        "--policy_weight_ema_mode",
        "global_step",
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
        "--adaptive_q_margin",
        "0.0",
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
        "--seed",
        "42",
        "--output_dir",
        args.output_dir,
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


def time_bins(frame):
    if frame.empty:
        return pd.Series(dtype="object")
    ranks = frame.groupby("dataset").cumcount()
    sizes = frame.groupby("dataset")["dataset"].transform("size")
    quartiles = (4 * ranks // sizes.clip(lower=1)).clip(upper=3)
    return quartiles.map({0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"})


def shared_learning_dynamics(decisions):
    required = {
        "dataset",
        "shared_value_mean",
        "explicit_advantage_mean",
        "legacy_advantage_risk",
        "stop_probability",
        "action",
        "exploration_used",
        "controller_latency_ms",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"shared decision log is missing: {sorted(missing)}")
    frame = decisions.copy()
    frame["time_bin"] = time_bins(frame)
    result = frame.groupby(["dataset", "time_bin"], observed=True).agg(
        decisions=("action", "size"),
        value_mean=("shared_value_mean", "mean"),
        value_std=("shared_value_mean", "std"),
        advantage_mean=("explicit_advantage_mean", "mean"),
        advantage_std=("explicit_advantage_mean", "std"),
        advantage_risk_mean=("legacy_advantage_risk", "mean"),
        stop_probability_mean=("stop_probability", "mean"),
        stop_rate_percent=(
            "action",
            lambda values: 100.0 * float(values.eq("stop").mean()),
        ),
        exploration_rate_percent=(
            "exploration_used",
            lambda values: 100.0 * float(values.astype(bool).mean()),
        ),
        controller_latency_ms=("controller_latency_ms", "mean"),
    ).reset_index()
    return result


def shared_parameter_trajectory(states, feature_names):
    rows = []
    for dataset, state in states.items():
        snapshots = list(state.get("weight_snapshots") or [])
        snapshots.append({
            "decision_count": int(state.get("decision_count", 0)),
            "shared_value_theta": state["shared_value_theta"],
            "shared_advantage_theta": state["shared_advantage_theta"],
            "snapshot": "final",
        })
        for snapshot in snapshots:
            value = snapshot.get("shared_value_theta")
            advantage = snapshot.get("shared_advantage_theta")
            if value is None or advantage is None:
                continue
            for index, feature in enumerate(feature_names):
                rows.append({
                    "dataset": dataset,
                    "snapshot": snapshot.get("snapshot", "periodic"),
                    "decision_count": int(snapshot["decision_count"]),
                    "feature": feature,
                    "value_weight": float(value[index]),
                    "advantage_weight": float(advantage[index]),
                })
    return pd.DataFrame(rows)


def validate_and_report(args):
    output_dir = Path(args.output_dir)
    manifest = json.loads(
        (output_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("method") != METHOD:
        raise RuntimeError(f"unexpected benchmark method: {manifest.get('method')}")
    if set(manifest.get("datasets", [])) != set(args.datasets):
        raise RuntimeError("benchmark did not complete every requested dataset")

    decisions = []
    states = {}
    for dataset in args.datasets:
        phase_dir = output_dir / "raw" / dataset / METHOD
        state = json.loads(
            (phase_dir / "adaptive_td_runtime_state.json").read_text(
                encoding="utf-8"
            )
        )
        if state.get("value_parameterization") != "shared_value_advantage":
            raise RuntimeError(f"{dataset} did not use shared value/advantage")
        if not math.isclose(
            float(state["shared_value_learning_rate"]),
            VALUE_LEARNING_RATE,
        ):
            raise RuntimeError(f"{dataset} value learning rate mismatch")
        if not math.isclose(
            float(state["shared_advantage_learning_rate"]),
            ADVANTAGE_LEARNING_RATE,
        ):
            raise RuntimeError(f"{dataset} advantage learning rate mismatch")
        decision_path = phase_dir / "adaptive_td_decisions.csv"
        if decision_path.exists() and decision_path.stat().st_size:
            frame = pd.read_csv(decision_path)
            frame.insert(0, "dataset", dataset)
            decisions.append(frame)
        states[dataset] = state

    combined_decisions = (
        pd.concat(decisions, ignore_index=True)
        if decisions
        else pd.DataFrame()
    )
    dynamics = (
        shared_learning_dynamics(combined_decisions)
        if not combined_decisions.empty
        else pd.DataFrame()
    )
    feature_names = manifest["feature_names"]
    trajectory = shared_parameter_trajectory(states, feature_names)
    dynamics.to_csv(
        output_dir / "shared_value_advantage_learning_dynamics.csv",
        index=False,
    )
    trajectory.to_csv(
        output_dir / "shared_value_advantage_parameter_trajectory.csv",
        index=False,
    )
    summary = pd.read_csv(output_dir / "dataset_method_summary.csv")
    total_tokens = float(summary.output_tokens.sum())
    total_time = float(summary.algorithm_time_s.sum())
    pooled = pd.DataFrame([{
        "datasets": len(summary),
        "num_questions": int(summary.num_questions.sum()),
        "output_tokens": total_tokens,
        "algorithm_time_s": total_time,
        "pooled_ms_per_output_token": 1000.0 * total_time / total_tokens,
        "draft_time_s": float(summary.draft_time_s.sum()),
        "verify_time_s": float(summary.verify_time_s.sum()),
        "post_verify_time_s": float(summary.post_verify_time_s.sum()),
        "draft_passes": int(summary.draft_passes.sum()),
        "verifier_rounds": int(summary.verifier_rounds.sum()),
        "controller_overhead_ms": float(summary.controller_overhead_ms.sum()),
    }])
    pooled.to_csv(
        output_dir / "shared_value_advantage_pooled_summary.csv",
        index=False,
    )
    return summary, dynamics, pooled


def main():
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_streaming(benchmark_command(args))
    summary, dynamics, pooled = validate_and_report(args)

    output_dir = Path(args.output_dir)
    wrapper_manifest = {
        "version": VERSION,
        "method": METHOD,
        "datasets": list(args.datasets),
        "num_questions_per_dataset": args.num_questions,
        "value_learning_rate": VALUE_LEARNING_RATE,
        "advantage_learning_rate": ADVANTAGE_LEARNING_RATE,
        "selected_q_effective_learning_rate": (
            VALUE_LEARNING_RATE + 0.25 * ADVANTAGE_LEARNING_RATE
        ),
        "unselected_q_shared_learning_rate": (
            VALUE_LEARNING_RATE - 0.25 * ADVANTAGE_LEARNING_RATE
        ),
        "uncertainty": "legacy_per_action_covariance",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "arguments": vars(args),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "shared_value_advantage_manifest.json").write_text(
        json.dumps(wrapper_manifest, indent=2),
        encoding="utf-8",
    )

    archive_path = None
    if not args.skip_archive:
        archive_path = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    print("\nSHARED VALUE + EXPLICIT ADVANTAGE DATASET SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSHARED LEARNING DYNAMICS", flush=True)
    print(dynamics.to_string(index=False), flush=True)
    print("\nPOOLED SUMMARY", flush=True)
    print(pooled.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    if archive_path:
        print(f"Archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
