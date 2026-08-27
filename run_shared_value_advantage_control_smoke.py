import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from adaptive_td import (
    CONTINUE,
    STOP,
    AdaptiveTDConfig,
    OnlineTDRefinementController,
)
from run_otrc_v2_td_benchmark import PROBLEM_IDS, command_for


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")
INDEPENDENT_METHOD = "independent_q_eta0p02"
CONTROL_METHOD = "shared_value_eta0p01_advantage_eta0p04"
INDEPENDENT_LEARNING_RATE = 0.02
CONTROL_VALUE_LEARNING_RATE = 0.01
CONTROL_ADVANTAGE_LEARNING_RATE = 0.04
TOLERANCE = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Small control experiment showing that Shared V+A with "
            "eta_V=0.01 and eta_A=0.04 reproduces an independent Q-head "
            "update with eta=0.02 on the same factual observations."
        )
    )
    parser.add_argument("--num_questions", type=int, default=3)
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
            "outputs_shared_value_advantage_control_smoke"
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
    available = min(len(PROBLEM_IDS[dataset]) for dataset in DATASETS)
    if args.num_questions <= 0 or args.num_questions > available:
        raise ValueError(f"--num_questions must be in [1, {available}]")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")


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


def controller_args(args, value_parameterization):
    return SimpleNamespace(
        warmup_questions=1,
        max_new_tokens=args.max_new_tokens,
        spec_len=8,
        block_size=32,
        small_block_size=8,
        target_model_name=args.target_model_name,
        dllm_dir=args.dllm_dir,
        drafter_threshold=0.05,
        lowconf_threshold=0.45,
        max_spec_len=60,
        incr_len=8,
        feature_schema="otrc_v2_2_compact_td",
        credit_assignment="verifier_boundary_factual_no_bootstrap",
        adaptive_learning_rate=INDEPENDENT_LEARNING_RATE,
        value_parameterization=value_parameterization,
        shared_value_learning_rate=CONTROL_VALUE_LEARNING_RATE,
        shared_advantage_learning_rate=CONTROL_ADVANTAGE_LEARNING_RATE,
        adaptive_mc_learning_rate=0.01,
        adaptive_mc_mix=0.5,
        adaptive_update_mode="mixed",
        adaptive_rho_alpha=0.05,
        rho_warmup_boundaries=0,
        policy_weight_ema_beta=0.0,
        policy_weight_ema_mode="global_step",
        adaptive_factual_ema_alpha=0.2,
        adaptive_risk_beta=1.0,
        adaptive_stop_probability_threshold=0.75,
        adaptive_uncertainty_prior=1.0,
        adaptive_epistemic_scale=0.1,
        adaptive_q_margin=0.0,
        adaptive_explore_epsilon=0.10,
        adaptive_explore_min=0.01,
        adaptive_explore_decay=0.998,
        adaptive_warmup_rounds=20,
        adaptive_early_stop_min_observations=32,
        adaptive_min_action_probability=0.10,
        adaptive_max_importance_weight=5.0,
        adaptive_weight_snapshot_interval=100,
        seed=42,
        log_level=args.log_level,
    )


def smoke_command(args, dataset, problem_ids, output_dir, parameterization):
    return command_for(
        controller_args(args, parameterization),
        dataset,
        problem_ids,
        output_dir,
    )


def phase_complete(directory, problem_ids, parameterization):
    required = (
        directory / "benchmark_results.csv",
        directory / "adaptive_td_decisions.csv",
        directory / "adaptive_td_runtime_state.json",
    )
    if not all(path.exists() and path.stat().st_size for path in required):
        return False
    try:
        benchmark = pd.read_csv(required[0])
        state = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.EmptyDataError):
        return False
    return (
        set(benchmark["problem_id"].astype(int)) == set(problem_ids)
        and state.get("value_parameterization", "independent_q")
        == parameterization
        and math.isclose(
            float(state.get("shared_value_learning_rate", 0.0)),
            CONTROL_VALUE_LEARNING_RATE,
        )
        and math.isclose(
            float(state.get("shared_advantage_learning_rate", 0.0)),
            CONTROL_ADVANTAGE_LEARNING_RATE,
        )
    )


def run_phase(args, dataset, method, parameterization, problem_ids):
    directory = Path(args.output_dir) / "raw" / dataset / method
    if args.resume and phase_complete(directory, problem_ids, parameterization):
        print(f"RESUME {dataset} | {method}", flush=True)
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100, flush=True)
    print(
        f"RUN {dataset} | {method} | IDs={problem_ids}",
        flush=True,
    )
    print("=" * 100, flush=True)
    run_streaming(smoke_command(
        args,
        dataset,
        problem_ids,
        directory,
        parameterization,
    ))
    if not phase_complete(directory, problem_ids, parameterization):
        raise RuntimeError(f"incomplete smoke phase: {dataset}/{method}")
    return directory


def replay_config(parameterization):
    return AdaptiveTDConfig(
        feature_dim=6,
        feature_schema="otrc_v2_2_compact_td",
        feature_version=226,
        credit_assignment="verifier_boundary_factual_no_bootstrap",
        value_parameterization=parameterization,
        learning_rate=INDEPENDENT_LEARNING_RATE,
        shared_value_learning_rate=CONTROL_VALUE_LEARNING_RATE,
        shared_advantage_learning_rate=CONTROL_ADVANTAGE_LEARNING_RATE,
        policy_mode="symmetric",
    )


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def replay_control_equivalence(decisions):
    independent = OnlineTDRefinementController(replay_config("independent_q"))
    control = OnlineTDRefinementController(
        replay_config("shared_value_advantage")
    )
    records = []
    for source_index, row in decisions.reset_index(drop=True).iterrows():
        if not as_bool(row.get("factual_update_applied", False)):
            continue
        target = pd.to_numeric(row.get("factual_target"), errors="coerce")
        weight = pd.to_numeric(row.get("importance_weight"), errors="coerce")
        if pd.isna(target) or pd.isna(weight):
            continue
        features = json.loads(row["features"])
        action = str(row.get("executed_action", row["action"])).lower()
        if action not in {STOP, CONTINUE}:
            raise ValueError(f"unknown factual action: {action}")
        independent_residual = independent._update_factual_action_value(
            action,
            features,
            float(target),
            observation_weight=float(weight),
        )
        control_residual = control._update_factual_action_value(
            action,
            features,
            float(target),
            observation_weight=float(weight),
        )
        stop_error = max(
            abs(left - right)
            for left, right in zip(
                independent.values[STOP].theta,
                control.values[STOP].theta,
            )
        )
        continue_error = max(
            abs(left - right)
            for left, right in zip(
                independent.values[CONTINUE].theta,
                control.values[CONTINUE].theta,
            )
        )
        records.append({
            "source_decision_index": int(source_index),
            "problem_id": int(row["problem_id"]),
            "round_id": int(row["round_id"]),
            "decision_id": int(row["decision_id"]),
            "action": action,
            "target": float(target),
            "importance_weight": float(weight),
            "independent_residual": float(independent_residual),
            "control_residual": float(control_residual),
            "residual_absolute_error": abs(
                float(independent_residual) - float(control_residual)
            ),
            "stop_head_max_absolute_error": stop_error,
            "continue_head_max_absolute_error": continue_error,
        })
    return pd.DataFrame(records)


def aggregate_benchmark(dataset, method, frame):
    output_tokens = float(frame["output_tokens"].sum())
    algorithm_time = float(frame["actual_algorithm_time"].sum())
    return {
        "dataset": dataset,
        "method": method,
        "questions": int(len(frame)),
        "output_tokens": int(output_tokens),
        "algorithm_time_s": algorithm_time,
        "algorithm_ms_per_output_token": (
            1000.0 * algorithm_time / output_tokens
        ),
        "draft_time_s": float(frame["actual_draft_time"].sum()),
        "verify_time_s": float(frame["actual_verify_time"].sum()),
        "post_verify_time_s": float(frame["actual_post_verify_time"].sum()),
        "draft_forward_passes": int(frame["total_num_forward_passes"].sum()),
        "verifier_rounds": int(frame["num_speculation_rounds"].sum()),
        "acceptance_rate_percent": (
            100.0 * float(frame["accepted_tokens"].sum())
            / max(float(frame["drafted_tokens"].sum()), 1.0)
        ),
    }


def compare_methods(dataset, directories):
    independent = pd.read_csv(
        directories[INDEPENDENT_METHOD] / "benchmark_results.csv"
    )
    control = pd.read_csv(
        directories[CONTROL_METHOD] / "benchmark_results.csv"
    )
    columns = [
        "problem_id",
        "output_token_hash",
        "output_tokens",
        "total_num_forward_passes",
        "num_speculation_rounds",
        "actual_algorithm_time",
    ]
    paired = independent[columns].merge(
        control[columns],
        on="problem_id",
        suffixes=("_independent", "_control"),
        validate="one_to_one",
    )
    paired.insert(0, "dataset", dataset)
    paired["output_hash_match"] = paired[
        "output_token_hash_independent"
    ].eq(paired["output_token_hash_control"])
    paired["draft_pass_delta_control_minus_independent"] = (
        paired["total_num_forward_passes_control"]
        - paired["total_num_forward_passes_independent"]
    )
    paired["verifier_round_delta_control_minus_independent"] = (
        paired["num_speculation_rounds_control"]
        - paired["num_speculation_rounds_independent"]
    )

    independent_decisions = pd.read_csv(
        directories[INDEPENDENT_METHOD] / "adaptive_td_decisions.csv"
    )
    control_decisions = pd.read_csv(
        directories[CONTROL_METHOD] / "adaptive_td_decisions.csv"
    )
    replay = replay_control_equivalence(independent_decisions)
    replay.insert(0, "dataset", dataset)

    keys = ["problem_id", "round_id", "decision_id"]
    decision_columns = keys + [
        "features",
        "action",
        "executed_action",
        "q_stop_mean",
        "q_continue_mean",
        "stop_probability",
    ]
    behavior = independent_decisions[decision_columns].merge(
        control_decisions[decision_columns],
        on=keys,
        how="outer",
        suffixes=("_independent", "_control"),
        indicator=True,
    )
    behavior.insert(0, "dataset", dataset)
    matched = behavior["_merge"].eq("both")
    behavior["features_match"] = (
        matched
        & behavior["features_independent"].eq(behavior["features_control"])
    )
    behavior["action_match"] = (
        matched
        & behavior["action_independent"].eq(behavior["action_control"])
    )
    return independent, control, paired, behavior, replay


def main():
    args = parse_args()
    validate_args(args)
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_aggregates = []
    all_paired = []
    all_behavior = []
    all_replay = []
    selected_ids = {}
    for dataset in DATASETS:
        problem_ids = PROBLEM_IDS[dataset][:args.num_questions]
        selected_ids[dataset] = problem_ids
        directories = {
            INDEPENDENT_METHOD: run_phase(
                args,
                dataset,
                INDEPENDENT_METHOD,
                "independent_q",
                problem_ids,
            ),
            CONTROL_METHOD: run_phase(
                args,
                dataset,
                CONTROL_METHOD,
                "shared_value_advantage",
                problem_ids,
            ),
        }
        independent, control, paired, behavior, replay = compare_methods(
            dataset,
            directories,
        )
        all_aggregates.extend([
            aggregate_benchmark(dataset, INDEPENDENT_METHOD, independent),
            aggregate_benchmark(dataset, CONTROL_METHOD, control),
        ])
        all_paired.append(paired)
        all_behavior.append(behavior)
        all_replay.append(replay)

    aggregate = pd.DataFrame(all_aggregates)
    paired = pd.concat(all_paired, ignore_index=True)
    behavior = pd.concat(all_behavior, ignore_index=True)
    replay = pd.concat(all_replay, ignore_index=True)
    max_replay_error = float(replay[[
        "residual_absolute_error",
        "stop_head_max_absolute_error",
        "continue_head_max_absolute_error",
    ]].max().max()) if not replay.empty else math.nan
    control_passed = bool(not replay.empty and max_replay_error <= TOLERANCE)

    aggregate.to_csv(output_dir / "smoke_dataset_summary.csv", index=False)
    paired.to_csv(output_dir / "smoke_paired_questions.csv", index=False)
    behavior.to_csv(output_dir / "smoke_behavior_alignment.csv", index=False)
    replay.to_csv(output_dir / "same_stream_numerical_equivalence.csv", index=False)
    report = {
        "purpose": (
            "control test, not hyperparameter tuning or a speed benchmark"
        ),
        "datasets": list(DATASETS),
        "problem_ids": selected_ids,
        "independent_learning_rate": INDEPENDENT_LEARNING_RATE,
        "shared_value_learning_rate": CONTROL_VALUE_LEARNING_RATE,
        "shared_advantage_learning_rate": CONTROL_ADVANTAGE_LEARNING_RATE,
        "selected_q_effective_learning_rate": (
            CONTROL_VALUE_LEARNING_RATE
            + 0.25 * CONTROL_ADVANTAGE_LEARNING_RATE
        ),
        "unselected_q_effective_learning_rate": (
            CONTROL_VALUE_LEARNING_RATE
            - 0.25 * CONTROL_ADVANTAGE_LEARNING_RATE
        ),
        "same_stream_updates": int(len(replay)),
        "same_stream_max_absolute_error": max_replay_error,
        "same_stream_tolerance": TOLERANCE,
        "same_stream_equivalence_passed": control_passed,
        "integration_output_match_rate_percent": (
            100.0 * float(paired["output_hash_match"].mean())
        ),
        "integration_matched_decision_states": int(
            behavior["_merge"].eq("both").sum()
        ),
        "integration_action_match_rate_percent": (
            100.0
            * float(
                behavior.loc[
                    behavior["_merge"].eq("both"), "action_match"
                ].mean()
            )
        ),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "interpretation": (
            "Only same_stream_equivalence_passed tests mathematical update "
            "equivalence. Separate GPU runs can differ because measured latency "
            "enters features and factual targets."
        ),
    }
    (output_dir / "smoke_control_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSMOKE DATASET SUMMARY", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nCONTROL RESULT", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if not control_passed:
        raise RuntimeError(
            "Shared V+A control did not reproduce independent-Q updates on "
            "the same factual stream"
        )

    if not args.skip_archive:
        archive = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir,
        )
        print(f"Archive: {archive}", flush=True)
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
