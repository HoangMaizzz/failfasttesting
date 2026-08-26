import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_td import FEATURE_SCHEMAS
from run_math_feature_ablation_benchmark import aggregate_method


ROOT = Path(__file__).resolve().parent
VERSION = "otrc_td_representation_benchmark_v5"
FEATURE_VARIANCE_EPS = 1e-8
FACTUAL_CREDIT_ASSIGNMENTS = {
    "verifier_boundary_factual",
    "verifier_boundary_factual_no_bootstrap",
}
PROBLEM_IDS = {
    "math": [
        2, 6, 42, 51, 53, 57, 61, 108, 115, 123, 129, 148, 161,
        164, 179, 183, 193, 204, 216, 226, 231, 252, 258, 263, 281,
    ],
    "gsm8k": [
        6, 24, 51, 157, 166, 184, 201, 211, 227, 244, 289, 431, 458,
        492, 516, 589, 590, 599, 633, 644, 655, 698, 713, 731, 745,
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(PROBLEM_IDS),
        default=["math", "gsm8k"],
    )
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument(
        "--feature_schema",
        choices=(
            "otrc_v2_td",
            "otrc_v2_1_td",
            "otrc_v2_2_td",
            "otrc_v2_2_compact_td",
        ),
        default="otrc_v2_2_td",
    )
    parser.add_argument(
        "--credit_assignment",
        choices=(
            "per_step_td",
            "verifier_boundary_factual",
            "verifier_boundary_factual_no_bootstrap",
        ),
        default="per_step_td",
    )
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--adaptive_learning_rate", type=float, default=0.02)
    parser.add_argument("--adaptive_mc_learning_rate", type=float, default=0.01)
    parser.add_argument("--adaptive_mc_mix", type=float, default=0.5)
    parser.add_argument(
        "--adaptive_update_mode",
        choices=("td", "factual_return", "mixed"),
        default="mixed",
    )
    parser.add_argument("--adaptive_rho_alpha", type=float, default=0.05)
    parser.add_argument("--rho_warmup_boundaries", type=int, default=0)
    parser.add_argument("--policy_weight_ema_beta", type=float, default=0.0)
    parser.add_argument("--adaptive_factual_ema_alpha", type=float, default=0.2)
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
    parser.add_argument("--adaptive_weight_snapshot_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_otrc_v2_2_td_test25",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0 or args.num_questions > 25:
        raise ValueError("--num_questions must be in [1, 25]")
    if args.spec_len != 8 or args.incr_len != 8:
        raise ValueError("the matched benchmark requires --spec_len=8 and --incr_len=8")
    if (
        args.credit_assignment in FACTUAL_CREDIT_ASSIGNMENTS
        and args.feature_schema not in {
            "otrc_v2_2_td",
            "otrc_v2_2_compact_td",
        }
    ):
        raise ValueError(
            "verifier-boundary factual credit requires --feature_schema "
            "otrc_v2_2_td or otrc_v2_2_compact_td"
        )
    if args.rho_warmup_boundaries < 0:
        raise ValueError("--rho_warmup_boundaries must be non-negative")
    if (
        args.rho_warmup_boundaries
        and args.credit_assignment
        != "verifier_boundary_factual_no_bootstrap"
    ):
        raise ValueError(
            "rho warmup is only supported by factual no-bootstrap credit"
        )
    if not 0.0 <= args.policy_weight_ema_beta < 1.0:
        raise ValueError("--policy_weight_ema_beta must be in [0, 1)")
    if (
        args.policy_weight_ema_beta
        and args.credit_assignment
        != "verifier_boundary_factual_no_bootstrap"
    ):
        raise ValueError(
            "policy weight EMA is only supported by factual no-bootstrap credit"
        )


def method_name(args):
    if args.credit_assignment == "verifier_boundary_factual_no_bootstrap":
        base = (
            "otrc_v2_2_compact_factual_no_bootstrap"
            if args.feature_schema == "otrc_v2_2_compact_td"
            else "otrc_v2_2_factual_no_bootstrap"
        )
        warmup = int(getattr(args, "rho_warmup_boundaries", 0))
        if warmup:
            base = f"{base}_rho_warmup{warmup}"
        policy_beta = float(getattr(args, "policy_weight_ema_beta", 0.0))
        if policy_beta:
            beta_label = f"{policy_beta:.6f}".rstrip("0").rstrip(".")
            base = f"{base}_policy_ema{beta_label.replace('.', 'p')}"
        return base
    if args.credit_assignment == "verifier_boundary_factual":
        return (
            "otrc_v2_2_compact_factual"
            if args.feature_schema == "otrc_v2_2_compact_td"
            else "otrc_v2_2_factual"
        )
    return args.feature_schema


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


def command_for(args, dataset, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(value) for value in problem_ids],
        "--warmup_questions", str(args.warmup_questions),
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
        "--adaptive-td",
        "--adaptive-controller", "avg_td",
        "--adaptive-feature-schema", args.feature_schema,
        "--adaptive-credit-assignment", args.credit_assignment,
        "--adaptive-learning-rate", str(args.adaptive_learning_rate),
        "--adaptive-mc-learning-rate", str(args.adaptive_mc_learning_rate),
        "--adaptive-mc-mix", str(args.adaptive_mc_mix),
        "--adaptive-update-mode", args.adaptive_update_mode,
        "--adaptive-rho-alpha", str(args.adaptive_rho_alpha),
        "--adaptive-rho-warmup-boundaries",
        str(args.rho_warmup_boundaries),
        "--adaptive-policy-weight-ema-beta",
        str(args.policy_weight_ema_beta),
        "--adaptive-factual-ema-alpha", str(args.adaptive_factual_ema_alpha),
        "--adaptive-risk-beta", str(args.adaptive_risk_beta),
        "--adaptive-stop-probability-threshold",
        str(args.adaptive_stop_probability_threshold),
        "--adaptive-uncertainty-prior", str(args.adaptive_uncertainty_prior),
        "--adaptive-epistemic-scale", str(args.adaptive_epistemic_scale),
        "--adaptive-q-margin", str(args.adaptive_q_margin),
        "--adaptive-explore-epsilon", str(args.adaptive_explore_epsilon),
        "--adaptive-explore-min", str(args.adaptive_explore_min),
        "--adaptive-explore-decay", str(args.adaptive_explore_decay),
        "--adaptive-warmup-rounds", str(args.adaptive_warmup_rounds),
        "--adaptive-early-stop-min-observations",
        str(args.adaptive_early_stop_min_observations),
        "--adaptive-policy-mode", "symmetric",
        "--adaptive-min-action-probability",
        str(args.adaptive_min_action_probability),
        "--adaptive-max-importance-weight",
        str(args.adaptive_max_importance_weight),
        "--adaptive-weight-snapshot-interval",
        str(args.adaptive_weight_snapshot_interval),
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
        "--seed", str(args.seed),
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def phase_complete(
    directory,
    problem_ids,
    feature_schema,
    credit_assignment,
    rho_warmup_boundaries,
    policy_weight_ema_beta,
):
    required = [
        directory / "benchmark_results.csv",
        directory / "adaptive_td_decisions.csv",
        directory / "adaptive_td_runtime_state.json",
    ]
    if not all(path.exists() and path.stat().st_size for path in required):
        return False
    try:
        results = pd.read_csv(required[0])
        state = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.EmptyDataError):
        return False
    return (
        set(results.problem_id.astype(int)) == set(problem_ids)
        and state.get("feature_schema") == feature_schema
        and state.get("credit_assignment", "per_step_td") == credit_assignment
        and int(state.get("rho_warmup_boundaries", 0))
        == int(rho_warmup_boundaries)
        and abs(
            float(state.get("policy_weight_ema_beta", 0.0))
            - float(policy_weight_ema_beta)
        )
        <= 1e-12
    )


def run_dataset(args, dataset, problem_ids):
    method = method_name(args)
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    if args.resume and phase_complete(
        output_dir,
        problem_ids,
        args.feature_schema,
        args.credit_assignment,
        args.rho_warmup_boundaries,
        args.policy_weight_ema_beta,
    ):
        print(f"RESUME {dataset} {method}", flush=True)
        return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100, flush=True)
    print(f"RUN {dataset} | {method} only | samples={len(problem_ids)}", flush=True)
    print("=" * 100, flush=True)
    run_streaming(command_for(args, dataset, problem_ids, output_dir))
    return output_dir


def factual_target_diagnostics(dataset, transitions):
    if transitions.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {
        "action",
        "boundary_id",
        "decisions_in_boundary",
        "verifier_boundaries_spanned",
        "delta_time_ms",
        "post_boundary_latency_ms",
        "emitted_tokens",
        "rho_tokens_per_ms",
        "bootstrap_value",
        "td_target",
        "td_error",
        "terminal",
    }
    missing = required.difference(transitions.columns)
    if missing:
        raise ValueError(
            f"factual transition log is missing fields: {sorted(missing)}"
        )
    numeric = [
        "decisions_in_boundary",
        "verifier_boundaries_spanned",
        "delta_time_ms",
        "post_boundary_latency_ms",
        "emitted_tokens",
        "rho_tokens_per_ms",
        "bootstrap_value",
        "td_target",
        "td_error",
    ]
    transitions = transitions.copy()
    if "update_applied" not in transitions:
        transitions["update_applied"] = True
    if "rho_warmup_boundaries" not in transitions:
        transitions["rho_warmup_boundaries"] = 0
    if "verifier_boundary_index" not in transitions:
        transitions["verifier_boundary_index"] = transitions["boundary_id"]
    for column in numeric:
        transitions[column] = pd.to_numeric(
            transitions[column],
            errors="coerce",
        )
    target_scale = max(float(transitions["td_target"].std(ddof=0)), 1e-9)
    transitions["absolute_td_error"] = transitions["td_error"].abs()
    transitions["normalized_absolute_td_error"] = (
        transitions["absolute_td_error"] / target_scale
    )
    learning_rows = transitions.loc[
        transitions["update_applied"].astype(bool)
    ]
    summary = pd.DataFrame([{
        "dataset": dataset,
        "transitions": int(len(transitions)),
        "boundaries": int(transitions["boundary_id"].nunique()),
        "stop_updates": int(transitions["action"].eq("stop").sum()),
        "continue_updates": int(transitions["action"].eq("continue").sum()),
        "terminal_updates": int(transitions["terminal"].astype(bool).sum()),
        "learning_updates": int(
            transitions["update_applied"].astype(bool).sum()
        ),
        "warmup_transitions": int(
            (~transitions["update_applied"].astype(bool)).sum()
        ),
        "rho_warmup_boundaries": int(
            pd.to_numeric(
                transitions["rho_warmup_boundaries"],
                errors="coerce",
            ).fillna(0).max()
        ),
        "first_learning_update_boundary": (
            None
            if learning_rows.empty
            else int(learning_rows["verifier_boundary_index"].iloc[0])
        ),
        "rho_at_first_learning_update": (
            None
            if learning_rows.empty
            else float(learning_rows["rho_tokens_per_ms"].iloc[0])
        ),
        "mean_decisions_per_boundary": float(
            transitions.groupby("boundary_id").size().mean()
        ),
        "mean_verifier_boundaries_spanned": float(
            transitions["verifier_boundaries_spanned"].mean()
        ),
        "mean_delta_time_ms": float(transitions["delta_time_ms"].mean()),
        "mean_post_boundary_latency_ms": float(
            transitions["post_boundary_latency_ms"].mean()
        ),
        "mean_emitted_tokens": float(transitions["emitted_tokens"].mean()),
        "mean_rho_tokens_per_ms": float(
            transitions["rho_tokens_per_ms"].mean()
        ),
        "mean_bootstrap_value": float(transitions["bootstrap_value"].mean()),
        "mean_td_target": float(transitions["td_target"].mean()),
        "td_target_std": target_scale,
        "mean_absolute_td_error": float(
            transitions["absolute_td_error"].mean()
        ),
        "normalized_mean_absolute_td_error": float(
            transitions["normalized_absolute_td_error"].mean()
        ),
    }])
    ordered = transitions.reset_index(drop=True)
    bins = min(4, len(ordered))
    ordered["time_bin"] = pd.qcut(
        np.arange(len(ordered)),
        bins,
        labels=[f"Q{index + 1}" for index in range(bins)],
    )
    dynamics = ordered.groupby("time_bin", observed=True).agg(
        transitions=("action", "size"),
        boundaries=("boundary_id", "nunique"),
        stop_rate_percent=(
            "action",
            lambda values: 100.0 * values.eq("stop").mean(),
        ),
        delta_time_ms_mean=("delta_time_ms", "mean"),
        post_boundary_latency_ms_mean=("post_boundary_latency_ms", "mean"),
        emitted_tokens_mean=("emitted_tokens", "mean"),
        rho_tokens_per_ms_mean=("rho_tokens_per_ms", "mean"),
        learning_update_rate_percent=(
            "update_applied",
            lambda values: 100.0 * values.astype(bool).mean(),
        ),
        bootstrap_value_mean=("bootstrap_value", "mean"),
        td_target_mean=("td_target", "mean"),
        absolute_td_error_mean=("absolute_td_error", "mean"),
        normalized_absolute_td_error_mean=(
            "normalized_absolute_td_error",
            "mean",
        ),
    ).reset_index()
    dynamics.insert(0, "dataset", dataset)
    return summary, dynamics


def parse_feature_matrix(decisions, feature_names):
    matrix = np.asarray([
        json.loads(value) if isinstance(value, str) else value
        for value in decisions["features"]
    ], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("OTRC decision log has an unexpected feature shape")
    return pd.DataFrame(matrix, columns=feature_names)


def feature_diagnostics(dataset, decisions, feature_names):
    features = parse_feature_matrix(decisions, feature_names)
    records = []
    for name in feature_names:
        series = features[name]
        records.append({
            "dataset": dataset,
            "feature": name,
            "count": len(series),
            "mean": series.mean(),
            "std": series.std(ddof=0),
            "variance": series.var(ddof=0),
            "min": series.min(),
            "max": series.max(),
            "approximately_constant": int(
                series.var(ddof=0) <= FEATURE_VARIANCE_EPS
            ),
        })
    nonconstant = [
        name for name in feature_names
        if name != "bias" and features[name].var(ddof=0) > FEATURE_VARIANCE_EPS
    ]
    if nonconstant:
        standardized = features[nonconstant].copy()
        standardized = (
            standardized - standardized.mean()
        ) / standardized.std(ddof=0)
        gram = standardized.to_numpy().T @ standardized.to_numpy()
        condition_number = float(np.linalg.cond(gram))
        pearson = features[nonconstant].corr(method="pearson")
        spearman = features[nonconstant].corr(method="spearman")
    else:
        condition_number = float("nan")
        pearson = pd.DataFrame()
        spearman = pd.DataFrame()
    conditioning = pd.DataFrame([{
        "dataset": dataset,
        "decisions": len(features),
        "feature_dim": len(feature_names),
        "nonconstant_feature_dim": len(nonconstant),
        "standardized_gram_condition_number": condition_number,
        "high_correlation_pairs": sum(
            abs(float(pearson.loc[left, right])) > 0.9
            for index, left in enumerate(nonconstant)
            for right in nonconstant[index + 1:]
        ),
    }])
    return pd.DataFrame(records), pearson, spearman, conditioning


def learning_dynamics(dataset, decisions):
    ordered = decisions.sort_values("decision_monotonic_s").reset_index(drop=True)
    bins = min(4, len(ordered))
    ordered["time_bin"] = pd.qcut(
        np.arange(len(ordered)),
        bins,
        labels=[f"Q{index + 1}" for index in range(bins)],
    )
    result = ordered.groupby("time_bin", observed=True).agg(
        decisions=("action", "size"),
        q_stop_mean=("q_stop_mean", "mean"),
        q_continue_mean=("q_continue_mean", "mean"),
        advantage_mean=("advantage_mean", "mean"),
        stop_probability_mean=("stop_probability", "mean"),
        stop_rate_percent=("action", lambda values: 100.0 * values.eq("stop").mean()),
        exploration_rate_percent=(
            "exploration_used",
            lambda values: 100.0 * values.astype(bool).mean(),
        ),
        controller_latency_ms=("controller_latency_ms", "mean"),
    ).reset_index()
    result.insert(0, "dataset", dataset)
    return result


def snapshot_diagnostics(dataset, decisions):
    required = {
        "proposal_remaining_masks",
        "remaining_masks",
        "proposal_remaining_confidence_count",
        "proposal_remaining_confidence_coverage",
        "proposal_snapshot_valid",
        "proposal_snapshot_phase",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decision log is missing snapshot fields: {sorted(missing)}")
    valid = decisions["proposal_snapshot_valid"].astype(str).str.lower().eq("true")
    post_commit = decisions["proposal_snapshot_phase"].eq(
        "post_commit_pre_decision"
    )
    mask_match = pd.to_numeric(
        decisions["proposal_remaining_masks"], errors="coerce"
    ).eq(pd.to_numeric(decisions["remaining_masks"], errors="coerce"))
    coverage = pd.to_numeric(
        decisions["proposal_remaining_confidence_coverage"], errors="coerce"
    )
    return {
        "dataset": dataset,
        "decisions": len(decisions),
        "valid_snapshot_percent": 100.0 * float(valid.mean()),
        "post_commit_snapshot_percent": 100.0 * float(post_commit.mean()),
        "mask_count_match_percent": 100.0 * float(mask_match.mean()),
        "confidence_coverage_mean": float(coverage.mean()),
        "confidence_coverage_min": float(coverage.min()),
        "zero_confidence_coverage_percent": 100.0 * float(coverage.eq(0.0).mean()),
    }


def confidence_diagnostics(dataset, decisions, drafter_threshold):
    required = {
        "proposal_remaining_masks",
        "proposal_remaining_min_confidence",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decision log is missing confidence fields: {sorted(missing)}")
    remaining_masks = pd.to_numeric(
        decisions["proposal_remaining_masks"], errors="coerce"
    )
    unresolved = remaining_masks.gt(0)
    values = pd.to_numeric(
        decisions.loc[unresolved, "proposal_remaining_min_confidence"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return {
            "dataset": dataset,
            "unresolved_decisions": int(unresolved.sum()),
            "confidence_observations": 0,
            "min_confidence_mean": np.nan,
            "min_confidence_std": np.nan,
            "min_confidence_min": np.nan,
            "min_confidence_max": np.nan,
            "below_drafter_threshold_percent": np.nan,
            "at_zero_percent": np.nan,
            "at_one_percent": np.nan,
        }
    return {
        "dataset": dataset,
        "unresolved_decisions": int(unresolved.sum()),
        "confidence_observations": int(len(values)),
        "min_confidence_mean": float(values.mean()),
        "min_confidence_std": float(values.std(ddof=0)),
        "min_confidence_min": float(values.min()),
        "min_confidence_max": float(values.max()),
        "below_drafter_threshold_percent": 100.0 * float(
            values.lt(float(drafter_threshold)).mean()
        ),
        "at_zero_percent": 100.0 * float(values.eq(0.0).mean()),
        "at_one_percent": 100.0 * float(values.eq(1.0).mean()),
    }


def weight_rows(dataset, state, feature_names):
    records = []
    stop = state["actions"]["stop"]["theta"]
    continue_ = state["actions"]["continue"]["theta"]
    policy_state = state.get("policy_weight_ema") or {}
    stop_policy_state = policy_state.get("stop") or {}
    continue_policy_state = policy_state.get("continue") or {}
    policy_stop = (
        stop_policy_state.get("theta", stop)
        if stop_policy_state.get("initialized", False)
        else stop
    )
    policy_continue = (
        continue_policy_state.get("theta", continue_)
        if continue_policy_state.get("initialized", False)
        else continue_
    )
    for index, name in enumerate(feature_names):
        records.append({
            "dataset": dataset,
            "snapshot": "final",
            "decision_count": state["decision_count"],
            "feature": name,
            "theta_stop": stop[index],
            "theta_continue": continue_[index],
            "theta_diff": stop[index] - continue_[index],
            "policy_theta_stop": policy_stop[index],
            "policy_theta_continue": policy_continue[index],
            "policy_theta_diff": (
                policy_stop[index] - policy_continue[index]
            ),
        })
    for snapshot in state.get("weight_snapshots") or []:
        for index, name in enumerate(feature_names):
            records.append({
                "dataset": dataset,
                "snapshot": "periodic",
                "decision_count": snapshot["decision_count"],
                "feature": name,
                "theta_stop": snapshot["theta_stop"][index],
                "theta_continue": snapshot["theta_continue"][index],
                "theta_diff": snapshot["theta_diff"][index],
                "policy_theta_stop": snapshot.get(
                    "policy_theta_stop",
                    snapshot["theta_stop"],
                )[index],
                "policy_theta_continue": snapshot.get(
                    "policy_theta_continue",
                    snapshot["theta_continue"],
                )[index],
                "policy_theta_diff": snapshot.get(
                    "policy_theta_diff",
                    snapshot["theta_diff"],
                )[index],
            })
    return records


def policy_ema_diagnostics(dataset, decisions):
    required = {
        "raw_stop_probability",
        "ema_stop_probability",
        "raw_greedy_action",
        "ema_greedy_action",
        "raw_ema_greedy_disagreement",
        "raw_advantage",
        "ema_advantage",
        "raw_bias_difference",
        "ema_bias_difference",
        "stop_weight_ema_distance_l2",
        "continue_weight_ema_distance_l2",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(
            f"decision log is missing policy EMA fields: {sorted(missing)}"
        )
    frame = decisions.copy().reset_index(drop=True)
    numeric = required.difference({"raw_greedy_action", "ema_greedy_action"})
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    disagreement = frame["raw_greedy_action"].ne(frame["ema_greedy_action"])
    summary = {
        "dataset": dataset,
        "decisions": int(len(frame)),
        "policy_weight_ema_enabled": bool(
            frame.get("policy_weight_ema_enabled", pd.Series([False]))
            .astype(str).str.lower().eq("true").any()
        ),
        "raw_greedy_stop_rate_percent": 100.0 * float(
            frame["raw_greedy_action"].eq("stop").mean()
        ),
        "ema_greedy_stop_rate_percent": 100.0 * float(
            frame["ema_greedy_action"].eq("stop").mean()
        ),
        "raw_ema_disagreement_percent": 100.0 * float(disagreement.mean()),
        "raw_stop_probability_mean": float(
            frame["raw_stop_probability"].mean()
        ),
        "ema_stop_probability_mean": float(
            frame["ema_stop_probability"].mean()
        ),
        "raw_advantage_std": float(frame["raw_advantage"].std(ddof=0)),
        "ema_advantage_std": float(frame["ema_advantage"].std(ddof=0)),
        "raw_bias_difference_final": float(
            frame["raw_bias_difference"].iloc[-1]
        ),
        "ema_bias_difference_final": float(
            frame["ema_bias_difference"].iloc[-1]
        ),
        "stop_weight_ema_distance_l2_mean": float(
            frame["stop_weight_ema_distance_l2"].mean()
        ),
        "continue_weight_ema_distance_l2_mean": float(
            frame["continue_weight_ema_distance_l2"].mean()
        ),
    }
    frame["time_bin"] = pd.qcut(
        frame.index,
        q=min(4, len(frame)),
        labels=[f"Q{index + 1}" for index in range(min(4, len(frame)))],
    )
    dynamics = frame.groupby("time_bin", observed=True).agg(
        decisions=("ema_advantage", "size"),
        raw_advantage_mean=("raw_advantage", "mean"),
        ema_advantage_mean=("ema_advantage", "mean"),
        raw_stop_probability_mean=("raw_stop_probability", "mean"),
        ema_stop_probability_mean=("ema_stop_probability", "mean"),
        raw_greedy_stop_rate_percent=(
            "raw_greedy_action",
            lambda values: 100.0 * values.eq("stop").mean(),
        ),
        ema_greedy_stop_rate_percent=(
            "ema_greedy_action",
            lambda values: 100.0 * values.eq("stop").mean(),
        ),
        raw_ema_disagreement_percent=(
            "raw_ema_greedy_disagreement",
            lambda values: 100.0 * values.astype(str).str.lower().eq("true").mean(),
        ),
        raw_bias_difference_mean=("raw_bias_difference", "mean"),
        ema_bias_difference_mean=("ema_bias_difference", "mean"),
        stop_weight_ema_distance_l2_mean=(
            "stop_weight_ema_distance_l2",
            "mean",
        ),
        continue_weight_ema_distance_l2_mean=(
            "continue_weight_ema_distance_l2",
            "mean",
        ),
    ).reset_index()
    dynamics.insert(0, "dataset", dataset)
    return summary, dynamics


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries = []
    feature_rows = []
    conditioning_frames = []
    dynamics_frames = []
    weights = []
    snapshot_rows = []
    confidence_rows = []
    factual_summary_frames = []
    factual_dynamics_frames = []
    policy_ema_rows = []
    policy_ema_dynamics_frames = []
    selected_ids = {}
    feature_names = FEATURE_SCHEMAS[args.feature_schema]

    for dataset in args.datasets:
        problem_ids = PROBLEM_IDS[dataset][:args.num_questions]
        selected_ids[dataset] = problem_ids
        phase_dir = run_dataset(args, dataset, problem_ids)
        results = pd.read_csv(phase_dir / "benchmark_results.csv")
        decisions = pd.read_csv(phase_dir / "adaptive_td_decisions.csv")
        state = json.loads(
            (phase_dir / "adaptive_td_runtime_state.json").read_text(
                encoding="utf-8"
            )
        )

        method = method_name(args)
        summary = aggregate_method(results, method)
        summary["dataset"] = dataset
        summary["controller_overhead_ms"] = float(
            pd.to_numeric(results["adaptive_controller_ms"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        summary["exploration_rate_percent"] = 100.0 * float(
            decisions["exploration_used"].astype(bool).mean()
        )
        summary["output_hash_unique"] = int(results.output_token_hash.nunique())
        summary["policy_weight_ema_beta"] = args.policy_weight_ema_beta
        summaries.append(summary)

        stats, pearson, spearman, conditioning = feature_diagnostics(
            dataset,
            decisions,
            feature_names,
        )
        feature_rows.extend(stats.to_dict("records"))
        conditioning_frames.append(conditioning)
        dynamics_frames.append(learning_dynamics(dataset, decisions))
        weights.extend(weight_rows(dataset, state, feature_names))
        ema_summary, ema_dynamics = policy_ema_diagnostics(dataset, decisions)
        policy_ema_rows.append(ema_summary)
        policy_ema_dynamics_frames.append(ema_dynamics)
        snapshot_rows.append(snapshot_diagnostics(dataset, decisions))
        confidence_rows.append(confidence_diagnostics(
            dataset,
            decisions,
            args.drafter_threshold,
        ))
        pearson.to_csv(output_dir / f"{dataset}_feature_correlation_pearson.csv")
        spearman.to_csv(output_dir / f"{dataset}_feature_correlation_spearman.csv")
        transition_path = phase_dir / "adaptive_full_stream_transitions.csv"
        if args.credit_assignment in FACTUAL_CREDIT_ASSIGNMENTS:
            if not transition_path.exists():
                raise FileNotFoundError(
                    f"missing factual transition report: {transition_path}"
                )
            factual_summary, factual_dynamics = factual_target_diagnostics(
                dataset,
                pd.read_csv(transition_path),
            )
            factual_summary_frames.append(factual_summary)
            factual_dynamics_frames.append(factual_dynamics)

    pd.DataFrame(summaries).to_csv(
        output_dir / "dataset_method_summary.csv",
        index=False,
    )
    pd.DataFrame(feature_rows).to_csv(
        output_dir / "feature_statistics.csv",
        index=False,
    )
    pd.concat(conditioning_frames, ignore_index=True).to_csv(
        output_dir / "feature_conditioning.csv",
        index=False,
    )
    pd.concat(dynamics_frames, ignore_index=True).to_csv(
        output_dir / "learning_dynamics.csv",
        index=False,
    )
    pd.DataFrame(weights).to_csv(
        output_dir / "weight_trajectory.csv",
        index=False,
    )
    pd.DataFrame(policy_ema_rows).to_csv(
        output_dir / "policy_ema_summary.csv",
        index=False,
    )
    pd.concat(policy_ema_dynamics_frames, ignore_index=True).to_csv(
        output_dir / "policy_ema_learning_dynamics.csv",
        index=False,
    )
    pd.DataFrame(snapshot_rows).to_csv(
        output_dir / "snapshot_invariants.csv",
        index=False,
    )
    pd.DataFrame(confidence_rows).to_csv(
        output_dir / "confidence_diagnostics.csv",
        index=False,
    )
    if factual_summary_frames:
        pd.concat(factual_summary_frames, ignore_index=True).to_csv(
            output_dir / "factual_target_summary.csv",
            index=False,
        )
        pd.concat(factual_dynamics_frames, ignore_index=True).to_csv(
            output_dir / "factual_target_learning_dynamics.csv",
            index=False,
        )
    manifest = {
        "version": VERSION,
        "feature_schema": args.feature_schema,
        "feature_names": list(feature_names),
        "credit_assignment": args.credit_assignment,
        "method": method_name(args),
        "datasets": list(args.datasets),
        "problem_ids": selected_ids,
        "arguments": vars(args),
        "baseline_or_oracle_executed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"\n{method_name(args)} DATASET SUMMARY", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
