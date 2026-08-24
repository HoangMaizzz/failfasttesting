import argparse
import json
import math
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_td import FEATURE_NAMES
from run_failfast_counterfactual_oracle import (
    DATASET_SIZES,
    decision_state_key,
    run_phase,
)
from run_gsm8k_balanced_oracle_dataset import (
    causal_oracle_comparison,
    paired_causal_comparison,
)


VERSION = "math_avg_td_feature_ablation_v1"
FEATURE_GROUPS = {
    "full": (),
    "drop_mask_progress": (
        "remaining_mask_ratio",
        "newly_unmasked_ratio",
        "first_mask_ratio",
    ),
    "drop_confidence": (
        "mean_confidence",
        "min_confidence",
        "max_confidence",
        "confidence_std",
    ),
    "drop_margin": ("mean_margin",),
    "drop_frontier": ("frontier_ratio",),
    "drop_stability": (
        "proposal_change_ratio",
        "recoverable_change_ratio",
    ),
    "drop_step": ("refinement_step",),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50)
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
    parser.add_argument("--adaptive_max_refinement_steps", type=int, default=16)
    parser.add_argument("--adaptive_learning_rate", type=float, default=0.02)
    parser.add_argument("--adaptive_mc_learning_rate", type=float, default=0.01)
    parser.add_argument("--adaptive_mc_mix", type=float, default=0.5)
    parser.add_argument(
        "--adaptive_update_mode",
        choices=("td", "factual_return", "mixed"),
        default="mixed",
    )
    parser.add_argument("--adaptive_rho_alpha", type=float, default=0.05)
    parser.add_argument("--adaptive_risk_beta", type=float, default=1.0)
    parser.add_argument(
        "--adaptive_stop_probability_threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument("--adaptive_uncertainty_prior", type=float, default=1.0)
    parser.add_argument("--adaptive_epistemic_scale", type=float, default=0.1)
    parser.add_argument("--adaptive_q_margin", type=float, default=0.0)
    parser.add_argument("--adaptive_explore_epsilon", type=float, default=0.10)
    parser.add_argument("--adaptive_explore_min", type=float, default=0.01)
    parser.add_argument("--adaptive_explore_decay", type=float, default=0.998)
    parser.add_argument("--adaptive_warmup_rounds", type=int, default=20)
    parser.add_argument(
        "--adaptive_early_stop_min_observations",
        type=int,
        default=32,
    )
    parser.add_argument("--adaptive_min_action_probability", type=float, default=0.10)
    parser.add_argument("--adaptive_max_importance_weight", type=float, default=5.0)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=tuple(FEATURE_GROUPS),
        default=list(FEATURE_GROUPS),
    )
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_math_feature_ablation_test50",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    available = DATASET_SIZES["math"] - args.warmup_questions
    if args.num_questions <= 0 or args.num_questions > available:
        raise ValueError(f"--num_questions must be in [1, {available}]")
    if "full" not in args.ablations:
        raise ValueError("--ablations must include full")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("proposal lengths must be positive")


def sampled_problem_ids(args):
    population = list(range(args.warmup_questions, DATASET_SIZES["math"]))
    return sorted(random.Random(args.sample_seed).sample(population, args.num_questions))


def adaptive_args(args, disabled_features):
    values = [
        "--adaptive-td",
        "--adaptive-controller", "avg_td",
        "--adaptive-max-refinement-steps", str(args.adaptive_max_refinement_steps),
        "--adaptive-learning-rate", str(args.adaptive_learning_rate),
        "--adaptive-mc-learning-rate", str(args.adaptive_mc_learning_rate),
        "--adaptive-mc-mix", str(args.adaptive_mc_mix),
        "--adaptive-update-mode", args.adaptive_update_mode,
        "--adaptive-rho-alpha", str(args.adaptive_rho_alpha),
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
        "--adaptive-use-step-feature",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
    ]
    if disabled_features:
        values.extend(["--adaptive-disable-features", *disabled_features])
    return values


def aggregate_method(results, method):
    output_tokens = float(results["output_tokens"].sum())
    drafted_tokens = float(results["drafted_tokens"].sum())
    algorithm_time = float(results["actual_algorithm_time"].sum())
    decisions = pd.to_numeric(
        results.get("adaptive_decisions", pd.Series(0, index=results.index)),
        errors="coerce",
    ).fillna(0)
    stops = pd.to_numeric(
        results.get("adaptive_stop_actions", pd.Series(0, index=results.index)),
        errors="coerce",
    ).fillna(0)
    return {
        "method": method,
        "num_questions": int(results["problem_id"].nunique()),
        "output_tokens": int(output_tokens),
        "algorithm_time_s": algorithm_time,
        "ms_per_output_token": 1000.0 * algorithm_time / max(1.0, output_tokens),
        "draft_time_s": float(results["actual_draft_time"].sum()),
        "verify_time_s": float(results["actual_verify_time"].sum()),
        "post_verify_time_s": float(results["actual_post_verify_time"].sum()),
        "draft_passes": int(results["total_num_forward_passes"].sum()),
        "verifier_rounds": int(results["num_speculation_rounds"].sum()),
        "draft_passes_per_100_tokens": 100.0
        * results["total_num_forward_passes"].sum()
        / max(1.0, output_tokens),
        "verifier_rounds_per_100_tokens": 100.0
        * results["num_speculation_rounds"].sum()
        / max(1.0, output_tokens),
        "acceptance_rate_percent": 100.0
        * results["accepted_tokens"].sum()
        / max(1.0, drafted_tokens),
        "adaptive_decisions": int(decisions.sum()),
        "adaptive_stop_rate_percent": 100.0
        * stops.sum()
        / max(1.0, decisions.sum()),
        "accuracy_percent": 100.0 * results["is_correct"].mean(),
    }


def method_summary(result_frames):
    records = [aggregate_method(frame, method) for method, frame in result_frames.items()]
    summary = pd.DataFrame(records)
    baseline = summary.loc[summary.method.eq("failfast")].iloc[0]
    oracle = summary.loc[summary.method.eq("causal_oracle")].iloc[0]
    summary["speedup_vs_failfast"] = (
        baseline.ms_per_output_token / summary.ms_per_output_token
    )
    denominator = baseline.ms_per_output_token - oracle.ms_per_output_token
    summary["oracle_latency_reduction_recovered_percent"] = (
        100.0
        * (baseline.ms_per_output_token - summary.ms_per_output_token)
        / denominator
        if denominator > 0.0
        else np.nan
    )
    return summary.sort_values("ms_per_output_token").reset_index(drop=True)


def paired_ablation_summary(result_frames):
    full = result_frames["avg_td_full"]
    records = []
    for method, frame in result_frames.items():
        if not method.startswith("avg_td_") or method == "avg_td_full":
            continue
        paired = full[[
            "problem_id",
            "actual_algorithm_time",
            "output_tokens",
            "output_token_hash",
        ]].merge(
            frame[[
                "problem_id",
                "actual_algorithm_time",
                "output_tokens",
                "output_token_hash",
            ]],
            on="problem_id",
            suffixes=("_full", "_ablated"),
            validate="one_to_one",
        )
        full_mspt = 1000.0 * paired.actual_algorithm_time_full / paired.output_tokens_full
        ablated_mspt = (
            1000.0 * paired.actual_algorithm_time_ablated / paired.output_tokens_ablated
        )
        speedups = ablated_mspt / full_mspt
        rng = np.random.default_rng(2026)
        bootstrap = []
        log_speedups = np.log(speedups.clip(lower=1e-12).to_numpy())
        for _ in range(2000):
            sample = rng.choice(log_speedups, size=len(log_speedups), replace=True)
            bootstrap.append(float(np.exp(sample.mean())))
        records.append({
            "ablation": method.removeprefix("avg_td_"),
            "disabled_features": json.dumps(
                FEATURE_GROUPS[method.removeprefix("avg_td_")]
            ),
            "num_questions": len(paired),
            "pooled_full_ms_per_output_token": 1000.0
            * paired.actual_algorithm_time_full.sum()
            / paired.output_tokens_full.sum(),
            "pooled_ablated_ms_per_output_token": 1000.0
            * paired.actual_algorithm_time_ablated.sum()
            / paired.output_tokens_ablated.sum(),
            "full_speedup_vs_ablated": float(
                paired.actual_algorithm_time_ablated.sum()
                / paired.output_tokens_ablated.sum()
                / (
                    paired.actual_algorithm_time_full.sum()
                    / paired.output_tokens_full.sum()
                )
            ),
            "full_win_rate_percent": 100.0 * float(full_mspt.lt(ablated_mspt).mean()),
            "geometric_mean_full_speedup_vs_ablated": float(
                np.exp(np.log(speedups.clip(lower=1e-12)).mean())
            ),
            "geometric_speedup_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "geometric_speedup_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "output_hash_match_percent": 100.0
            * float(
                paired.output_token_hash_full.eq(
                    paired.output_token_hash_ablated
                ).mean()
            ),
        })
    return pd.DataFrame(records).sort_values(
        "full_speedup_vs_ablated",
        ascending=False,
    )


def load_feature_matrix(decisions):
    rows = [json.loads(value) for value in decisions["features"]]
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("adaptive decision features have an unexpected shape")
    return matrix


def feature_state_summary(output_dir, variants):
    records = []
    for variant in variants:
        method = f"avg_td_{variant}"
        raw_dir = output_dir / "raw" / method
        decisions = pd.read_csv(raw_dir / "adaptive_td_decisions.csv")
        state = json.loads(
            (raw_dir / "adaptive_td_runtime_state.json").read_text(encoding="utf-8")
        )
        matrix = load_feature_matrix(decisions)
        stop_theta = np.asarray(state["actions"]["stop"]["theta"], dtype=float)
        continue_theta = np.asarray(
            state["actions"]["continue"]["theta"],
            dtype=float,
        )
        for index, name in enumerate(FEATURE_NAMES):
            std = float(matrix[:, index].std())
            theta_difference = float(stop_theta[index] - continue_theta[index])
            records.append({
                "method": method,
                "feature": name,
                "disabled": int(name in FEATURE_GROUPS[variant]),
                "observations": len(matrix),
                "mean": float(matrix[:, index].mean()),
                "std": std,
                "minimum": float(matrix[:, index].min()),
                "maximum": float(matrix[:, index].max()),
                "unique_values": int(np.unique(matrix[:, index]).size),
                "stop_theta": float(stop_theta[index]),
                "continue_theta": float(continue_theta[index]),
                "theta_stop_minus_continue": theta_difference,
                "standardized_final_effect": theta_difference * std,
            })
    return pd.DataFrame(records)


def add_state_occurrence(frame):
    result = frame.copy()
    result["state_key"] = result.apply(
        lambda row: decision_state_key(
            row.problem_id,
            row.context_len,
            row.target_len,
            row.step,
            row.draft_proposal,
        ),
        axis=1,
    )
    result["state_occurrence"] = result.groupby("state_key").cumcount()
    return result


def auc_score(values, labels):
    values = pd.Series(values)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if values.nunique() < 2 or positives == 0 or negatives == 0:
        return np.nan
    ranks = values.rank().to_numpy()
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def feature_oracle_alignment(output_dir):
    decisions = add_state_occurrence(
        pd.read_csv(output_dir / "raw" / "avg_td_full" / "adaptive_td_decisions.csv")
    )
    candidates = pd.read_csv(
        output_dir / "raw" / "causal_oracle" / "causal_oracle_candidates.csv"
    )
    candidates = candidates[
        candidates["oracle_action"].isin(["stop", "continue"])
    ].copy()
    candidates = add_state_occurrence(candidates)
    matched = decisions.merge(
        candidates[[
            "state_key",
            "state_occurrence",
            "oracle_action",
            "counterfactual_ms_per_output_token",
            "selected",
        ]],
        on=["state_key", "state_occurrence"],
        how="left",
        validate="one_to_one",
    )
    evaluated = matched.dropna(subset=["oracle_action"]).copy()
    if evaluated.empty:
        return (
            evaluated,
            pd.DataFrame(columns=[
                "feature",
                "matched_states",
                "match_rate_percent",
                "mean",
                "std",
                "unique_values",
                "spearman_with_predicted_stop_advantage",
                "oracle_stop_auc",
                "oracle_stop_mean",
                "oracle_continue_mean",
            ]),
            pd.DataFrame([{
                "all_adaptive_decisions": len(decisions),
                "oracle_matched_decisions": 0,
                "match_rate_percent": 0.0,
            }]),
        )
    evaluated["decision_correct"] = evaluated.action.eq(evaluated.oracle_action)
    evaluated["oracle_stop"] = evaluated.oracle_action.eq("stop").astype(int)
    evaluated["predicted_stop"] = evaluated.action.eq("stop").astype(int)
    evaluated["predicted_advantage"] = (
        evaluated.q_stop_mean - evaluated.q_continue_mean
    )
    matrix = load_feature_matrix(evaluated)
    records = []
    for index, name in enumerate(FEATURE_NAMES):
        values = pd.Series(matrix[:, index])
        records.append({
            "feature": name,
            "matched_states": len(evaluated),
            "match_rate_percent": 100.0 * len(evaluated) / max(1, len(decisions)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "unique_values": int(values.nunique()),
            "spearman_with_predicted_stop_advantage": values.corr(
                evaluated["predicted_advantage"].reset_index(drop=True),
                method="spearman",
            ),
            "oracle_stop_auc": auc_score(values, evaluated.oracle_stop),
            "oracle_stop_mean": float(values[evaluated.oracle_stop.to_numpy() == 1].mean()),
            "oracle_continue_mean": float(
                values[evaluated.oracle_stop.to_numpy() == 0].mean()
            ),
        })
    predicted_stop = evaluated.predicted_stop.astype(bool)
    oracle_stop = evaluated.oracle_stop.astype(bool)
    true_stop = int((predicted_stop & oracle_stop).sum())
    false_stop = int((predicted_stop & ~oracle_stop).sum())
    false_continue = int((~predicted_stop & oracle_stop).sum())
    true_continue = int((~predicted_stop & ~oracle_stop).sum())
    decision_summary = pd.DataFrame([{
        "all_adaptive_decisions": len(decisions),
        "oracle_matched_decisions": len(evaluated),
        "match_rate_percent": 100.0 * len(evaluated) / max(1, len(decisions)),
        "decision_accuracy_percent": 100.0 * evaluated.decision_correct.mean(),
        "true_stop": true_stop,
        "false_stop": false_stop,
        "false_continue": false_continue,
        "true_continue": true_continue,
        "stop_precision_percent": 100.0 * true_stop / max(1, true_stop + false_stop),
        "stop_recall_percent": 100.0 * true_stop / max(1, true_stop + false_continue),
        "continue_precision_percent": 100.0
        * true_continue
        / max(1, true_continue + false_continue),
        "continue_recall_percent": 100.0
        * true_continue
        / max(1, true_continue + false_stop),
        "spearman_predicted_advantage_vs_oracle_stop": pd.Series(
            evaluated.predicted_advantage
        ).corr(pd.Series(evaluated.oracle_stop), method="spearman"),
    }])
    return evaluated, pd.DataFrame(records), decision_summary


def learning_windows(evaluated):
    if evaluated.empty:
        return pd.DataFrame()
    ordered = evaluated.reset_index(drop=True).copy()
    ordered["decision_position"] = np.arange(len(ordered))
    ordered["learning_window"] = pd.cut(
        ordered.decision_position,
        bins=[-1, 99, 499, 999, math.inf],
        labels=["0-99", "100-499", "500-999", "1000+"],
    )
    records = []
    for window, group in ordered.groupby("learning_window", observed=True):
        records.append({
            "learning_window": str(window),
            "matched_decisions": len(group),
            "decision_accuracy_percent": 100.0 * group.decision_correct.mean(),
            "policy_stop_rate_percent": 100.0 * group.action.eq("stop").mean(),
            "oracle_stop_rate_percent": 100.0 * group.oracle_action.eq("stop").mean(),
            "mean_predicted_stop_probability": group.stop_probability.mean(),
        })
    return pd.DataFrame(records)


def write_manifest(args, output_dir, problem_ids):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    payload = {
        "version": VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": "math",
        "arguments": vars(args),
        "problem_ids": problem_ids,
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": {
            name: list(features) for name, features in FEATURE_GROUPS.items()
        },
        "interpretation": {
            "ablation": (
                "Each avg_td variant learns online from zero on the same ordered "
                "questions. A positive full_speedup_vs_ablated means the removed "
                "feature group helped the full controller."
            ),
            "causal_oracle": (
                "The causal oracle executes the best observed refinement snapshot "
                "at each round. Diagnostic probes and discarded draft work are excluded "
                "from actual_algorithm_time."
            ),
            "limitations": (
                "One online order and stochastic symmetric action sampling do not prove "
                "feature causality. Correlated features are ablated as groups."
            ),
        },
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.dataset = "math"
    args.adaptive_state_path = None
    args.oracle_only = True
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = sampled_problem_ids(args)
    (output_dir / "sampled_problem_ids.json").write_text(
        json.dumps(problem_ids, indent=2),
        encoding="utf-8",
    )

    baseline_dir = run_phase(
        args,
        problem_ids,
        "failfast",
        [],
        ["benchmark_results.csv"],
    )
    causal_dir = run_phase(
        args,
        problem_ids,
        "causal_oracle",
        ["--collect_bucket_oracle", "--causal_oracle"],
        [
            "benchmark_results.csv",
            "causal_oracle_decisions.csv",
            "causal_oracle_candidates.csv",
        ],
    )

    result_frames = {
        "failfast": pd.read_csv(baseline_dir / "benchmark_results.csv"),
        "causal_oracle": pd.read_csv(causal_dir / "benchmark_results.csv"),
    }
    execution_variants = list(args.ablations)
    random.Random(args.sample_seed + 7919).shuffle(execution_variants)
    args.execution_order = execution_variants
    for variant in execution_variants:
        method = f"avg_td_{variant}"
        variant_dir = run_phase(
            args,
            problem_ids,
            method,
            adaptive_args(args, FEATURE_GROUPS[variant]),
            [
                "benchmark_results.csv",
                "adaptive_td_decisions.csv",
                "adaptive_td_runtime_state.json",
            ],
        )
        result_frames[method] = pd.read_csv(variant_dir / "benchmark_results.csv")

    summary = method_summary(result_frames)
    ablations = paired_ablation_summary(result_frames)
    feature_states = feature_state_summary(output_dir, args.ablations)
    evaluated, oracle_features, oracle_decisions = feature_oracle_alignment(output_dir)
    windows = learning_windows(evaluated)
    causal_decisions = pd.read_csv(causal_dir / "causal_oracle_decisions.csv")
    causal_summary = causal_oracle_comparison(
        result_frames["failfast"],
        result_frames["causal_oracle"],
        causal_decisions,
    )
    causal_paired = paired_causal_comparison(
        result_frames["failfast"],
        result_frames["causal_oracle"],
    )

    all_results = []
    for method, frame in result_frames.items():
        item = frame.copy()
        item["method"] = method
        all_results.append(item)
    pd.concat(all_results, ignore_index=True, sort=False).to_csv(
        output_dir / "per_problem_results.csv",
        index=False,
    )
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    ablations.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    feature_states.to_csv(output_dir / "feature_state_summary.csv", index=False)
    evaluated.to_csv(output_dir / "full_oracle_matched_decisions.csv", index=False)
    oracle_features.to_csv(output_dir / "feature_oracle_alignment.csv", index=False)
    oracle_decisions.to_csv(output_dir / "full_oracle_decision_summary.csv", index=False)
    windows.to_csv(output_dir / "full_oracle_learning_windows.csv", index=False)
    causal_summary.to_csv(output_dir / "causal_oracle_summary.csv", index=False)
    causal_paired.to_csv(output_dir / "causal_oracle_paired_results.csv", index=False)
    write_manifest(args, output_dir, problem_ids)

    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nMETHOD SUMMARY")
    print(summary.to_string(index=False))
    print("\nFEATURE GROUP ABLATION")
    print(ablations.to_string(index=False))
    print("\nCAUSAL ORACLE UPPER BOUND")
    print(causal_summary.to_string(index=False))
    print("\nFULL CONTROLLER ORACLE ALIGNMENT")
    print(oracle_decisions.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
