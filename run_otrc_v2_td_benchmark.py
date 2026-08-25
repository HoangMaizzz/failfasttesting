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
VERSION = "otrc_td_representation_benchmark_v2"
FEATURE_VARIANCE_EPS = 1e-8
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
        choices=("otrc_v2_td", "otrc_v2_1_td"),
        default="otrc_v2_1_td",
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
        default="/content/failfasttesting/outputs_otrc_v2_1_td_test25",
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
        "--adaptive-learning-rate", str(args.adaptive_learning_rate),
        "--adaptive-mc-learning-rate", str(args.adaptive_mc_learning_rate),
        "--adaptive-mc-mix", str(args.adaptive_mc_mix),
        "--adaptive-update-mode", args.adaptive_update_mode,
        "--adaptive-rho-alpha", str(args.adaptive_rho_alpha),
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


def phase_complete(directory, problem_ids, feature_schema):
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
    )


def run_dataset(args, dataset, problem_ids):
    method = args.feature_schema
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    if args.resume and phase_complete(output_dir, problem_ids, args.feature_schema):
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


def weight_rows(dataset, state, feature_names):
    records = []
    stop = state["actions"]["stop"]["theta"]
    continue_ = state["actions"]["continue"]["theta"]
    for index, name in enumerate(feature_names):
        records.append({
            "dataset": dataset,
            "snapshot": "final",
            "decision_count": state["decision_count"],
            "feature": name,
            "theta_stop": stop[index],
            "theta_continue": continue_[index],
            "theta_diff": stop[index] - continue_[index],
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
            })
    return records


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

        summary = aggregate_method(results, args.feature_schema)
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
        snapshot_rows.append(snapshot_diagnostics(dataset, decisions))
        pearson.to_csv(output_dir / f"{dataset}_feature_correlation_pearson.csv")
        spearman.to_csv(output_dir / f"{dataset}_feature_correlation_spearman.csv")

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
    pd.DataFrame(snapshot_rows).to_csv(
        output_dir / "snapshot_invariants.csv",
        index=False,
    )
    manifest = {
        "version": VERSION,
        "feature_schema": args.feature_schema,
        "feature_names": list(feature_names),
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
    print(f"\n{args.feature_schema} DATASET SUMMARY", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
