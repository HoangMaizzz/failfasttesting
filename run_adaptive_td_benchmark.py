import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SIZES = {
    "math": 500,
    "aime": 30,
    "gsm8k": 1319,
    "humaneval": 164,
}
METHODS = (
    "failfast",
    "adaptive_td",
    "adaptive_force_continue",
    "fixed_r1",
    "fixed_r2",
    "fixed_r3",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SIZES),
        default=list(DATASET_SIZES),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["failfast", "adaptive_td"],
    )
    parser.add_argument("--num_questions", type=int, default=15)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
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
    parser.add_argument("--adaptive-explore-epsilon", type=float, default=0.03)
    parser.add_argument("--adaptive-explore-min", type=float, default=0.005)
    parser.add_argument("--adaptive-explore-decay", type=float, default=0.999)
    parser.add_argument("--adaptive-warmup-rounds", type=int, default=20)
    parser.add_argument("--adaptive-use-step-feature", action="store_true")
    parser.add_argument(
        "--adaptive-use-margin-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-use-stability-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adaptive-log-decisions", action="store_true")
    parser.add_argument("--adaptive-profile-overhead", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_adaptive_td_test15",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 0:
        raise ValueError("--warmup_questions must be non-negative")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("--spec_len and --incr_len must be positive")
    for dataset in args.datasets:
        available = DATASET_SIZES[dataset] - args.warmup_questions
        if args.num_questions > available:
            raise ValueError(f"{dataset} has only {available} measured samples")


def sampled_problem_ids(args):
    result = {}
    for dataset_index, dataset in enumerate(args.datasets):
        population = list(range(args.warmup_questions, DATASET_SIZES[dataset]))
        rng = random.Random(args.sample_seed + 1009 * dataset_index)
        result[dataset] = sorted(rng.sample(population, args.num_questions))
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


def metadata(args, dataset, method, problem_ids):
    return {
        "version": "online_td_v1_mean_uncertainty_v5",
        "dataset": dataset,
        "method": method,
        "problem_ids": problem_ids,
        "max_new_tokens": args.max_new_tokens,
        "spec_len": args.spec_len,
        "incr_len": args.incr_len,
        "max_spec_len": args.max_spec_len,
        "target_model_name": args.target_model_name,
        "dllm_dir": args.dllm_dir,
        "drafter_threshold": args.drafter_threshold,
        "lowconf_threshold": args.lowconf_threshold,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "adaptive": {
            "max_refinement_steps": args.adaptive_max_refinement_steps,
            "learning_rate": args.adaptive_learning_rate,
            "mc_learning_rate": args.adaptive_mc_learning_rate,
            "mc_mix": args.adaptive_mc_mix,
            "update_mode": args.adaptive_update_mode,
            "rho_alpha": args.adaptive_rho_alpha,
            "risk_beta": args.adaptive_risk_beta,
            "uncertainty_prior": args.adaptive_uncertainty_prior,
            "epistemic_scale": args.adaptive_epistemic_scale,
            "q_margin": args.adaptive_q_margin,
            "explore_epsilon": args.adaptive_explore_epsilon,
            "explore_min": args.adaptive_explore_min,
            "explore_decay": args.adaptive_explore_decay,
            "warmup_rounds": args.adaptive_warmup_rounds,
            "use_step_feature": args.adaptive_use_step_feature,
            "use_margin_feature": args.adaptive_use_margin_feature,
            "use_stability_feature": args.adaptive_use_stability_feature,
            "log_decisions": args.adaptive_log_decisions,
            "profile_overhead": args.adaptive_profile_overhead,
        },
    }


def complete_run(result_path, metadata_path, expected):
    if not result_path.exists() or not metadata_path.exists():
        return False
    try:
        rows = pd.read_csv(result_path)
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return (
        actual == expected
        and sorted(rows["problem_id"].astype(int).tolist()) == expected["problem_ids"]
    )


def run_method(args, dataset, method, problem_ids):
    output_dir = Path(args.output_dir) / "raw" / dataset / method
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    expected_metadata = metadata(args, dataset, method, problem_ids)
    if args.resume and complete_run(result_path, metadata_path, expected_metadata):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        for filename in (
            "benchmark_results.csv",
            "adaptive_td_decisions.csv",
            "adaptive_td_runtime_state.json",
            "run_metadata.json",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(len(problem_ids)),
            "--problem_ids", *[str(problem_id) for problem_id in problem_ids],
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
            "--seed", str(args.seed),
            "--quiet_generation",
            "--disable_progress",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]
        if method != "failfast":
            command.extend([
                "--adaptive-td",
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
                "--adaptive-explore-epsilon", str(args.adaptive_explore_epsilon),
                "--adaptive-explore-min", str(args.adaptive_explore_min),
                "--adaptive-explore-decay", str(args.adaptive_explore_decay),
                "--adaptive-warmup-rounds", str(args.adaptive_warmup_rounds),
            ])
            if args.adaptive_log_decisions:
                command.append("--adaptive-log-decisions")
            if args.adaptive_profile_overhead:
                command.append("--adaptive-profile-overhead")
            if method.startswith("fixed_r"):
                command.extend([
                    "--adaptive-fixed-refinement-steps",
                    method.removeprefix("fixed_r"),
                ])
            if method == "adaptive_force_continue":
                command.append("--adaptive-force-continue")
            if args.adaptive_use_step_feature:
                command.append("--adaptive-use-step-feature")
            if not args.adaptive_use_margin_feature:
                command.append("--no-adaptive-use-margin-feature")
            if not args.adaptive_use_stability_feature:
                command.append("--no-adaptive-use-stability-feature")
        print("\n" + "=" * 100, flush=True)
        print(
            f"RUN {dataset} | {method} | samples={len(problem_ids)} | "
            f"problem_ids={problem_ids}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
        metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")

    rows = pd.read_csv(result_path)
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


def aggregate(rows):
    records = []
    for (dataset, method), group in rows.groupby(["dataset", "method"], sort=False):
        output_tokens = group["output_tokens"].sum()
        drafted_tokens = group["drafted_tokens"].sum()
        measured_time_s = group["measured_time_s"].sum()
        e2e_time_s = group["actual_e2e_time"].sum()
        verifier_rounds = group["num_speculation_rounds"].sum()
        adaptive_decisions = group.get(
            "adaptive_decisions",
            pd.Series(0, index=group.index),
        )
        adaptive_stop_actions = group.get(
            "adaptive_stop_actions",
            pd.Series(0, index=group.index),
        )
        adaptive_exploration_actions = group.get(
            "adaptive_exploration_actions",
            pd.Series(0, index=group.index),
        )
        stop_available_decisions = group.get(
            "adaptive_stop_available_decisions",
            pd.Series(0, index=group.index),
        )
        candidate_coverage_decisions = group.get(
            "adaptive_candidate_coverage_decisions",
            pd.Series(0, index=group.index),
        )
        outer_verify_eligible_decisions = group.get(
            "adaptive_outer_verify_eligible_decisions",
            pd.Series(0, index=group.index),
        )
        stop_then_extend = group.get(
            "adaptive_stop_then_extend_actions",
            pd.Series(0, index=group.index),
        )
        stop_then_verify = group.get(
            "adaptive_stop_then_verify_actions",
            pd.Series(0, index=group.index),
        )
        outer_action_matches = group.get(
            "adaptive_outer_action_matches",
            pd.Series(0, index=group.index),
        )
        outer_action_mismatches = group.get(
            "adaptive_outer_action_mismatches",
            pd.Series(0, index=group.index),
        )
        decision_count = adaptive_decisions.sum()
        weighted_step = (
            group.get(
                "adaptive_mean_refinement_step",
                pd.Series(0.0, index=group.index),
            )
            * adaptive_decisions
        ).sum()
        records.append({
            "dataset": dataset,
            "method": method,
            "num_samples": len(group),
            "output_tokens": output_tokens,
            "total_measured_latency_s": measured_time_s,
            "total_e2e_latency_s": e2e_time_s,
            "measured_tokens_per_second": output_tokens / measured_time_s,
            "e2e_tokens_per_second": output_tokens / e2e_time_s,
            "measured_ms_per_output_token": 1000.0 * measured_time_s / output_tokens,
            "e2e_ms_per_output_token": 1000.0 * e2e_time_s / output_tokens,
            "draft_ms_per_output_token": 1000.0 * group["actual_draft_time"].sum() / output_tokens,
            "verify_ms_per_output_token": 1000.0 * group["actual_verify_time"].sum() / output_tokens,
            "post_verify_ms_per_output_token": 1000.0 * group["actual_post_verify_time"].sum() / output_tokens,
            "acceptance_rate_percent": 100.0 * group["accepted_tokens"].sum() / max(1, drafted_tokens),
            "emitted_tokens_per_verifier_call": output_tokens / verifier_rounds,
            "verifier_rounds_per_100_tokens": 100.0 * verifier_rounds / output_tokens,
            "draft_passes_per_100_tokens": 100.0 * group["total_num_forward_passes"].sum() / output_tokens,
            "adaptive_decisions": decision_count,
            "adaptive_stop_actions": adaptive_stop_actions.sum(),
            "adaptive_stop_rate_percent": 100.0 * adaptive_stop_actions.sum() / max(1, decision_count),
            "adaptive_exploration_rate_percent": 100.0 * adaptive_exploration_actions.sum() / max(1, decision_count),
            "adaptive_stop_available_rate_percent": 100.0 * stop_available_decisions.sum() / max(1, decision_count),
            "adaptive_candidate_coverage_rate_percent": 100.0 * candidate_coverage_decisions.sum() / max(1, decision_count),
            "adaptive_outer_verify_eligible_rate_percent": 100.0 * outer_verify_eligible_decisions.sum() / max(1, decision_count),
            "adaptive_stop_then_extend_actions": stop_then_extend.sum(),
            "adaptive_stop_then_verify_actions": stop_then_verify.sum(),
            "adaptive_outer_action_match_rate_percent": 100.0 * outer_action_matches.sum() / max(1, outer_action_matches.sum() + outer_action_mismatches.sum()),
            "adaptive_mean_decision_step": weighted_step / max(1, decision_count),
            "adaptive_controller_ms": group.get("adaptive_controller_ms", pd.Series(0, index=group.index)).sum(),
            "adaptive_controller_ms_per_output_token": group.get("adaptive_controller_ms", pd.Series(0, index=group.index)).sum() / output_tokens,
        })
    return pd.DataFrame(records)


def paired_comparison(rows):
    if not {"failfast", "adaptive_td"}.issubset(set(rows["method"])):
        return pd.DataFrame()
    columns = [
        "dataset",
        "problem_id",
        "method",
        "measured_ms_per_output_token",
        "e2e_ms_per_output_token",
        "output_token_hash",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "output_tokens",
    ]
    pivot = rows[columns].pivot(index=["dataset", "problem_id"], columns="method")
    result = pd.DataFrame({
        "dataset": pivot.index.get_level_values("dataset"),
        "problem_id": pivot.index.get_level_values("problem_id"),
        "measured_speedup_vs_failfast": (
            pivot[("measured_ms_per_output_token", "failfast")]
            / pivot[("measured_ms_per_output_token", "adaptive_td")]
        ).to_numpy(),
        "e2e_speedup_vs_failfast": (
            pivot[("e2e_ms_per_output_token", "failfast")]
            / pivot[("e2e_ms_per_output_token", "adaptive_td")]
        ).to_numpy(),
        "adaptive_wins": (
            pivot[("measured_ms_per_output_token", "adaptive_td")]
            < pivot[("measured_ms_per_output_token", "failfast")]
        ).astype(int).to_numpy(),
        "output_match": (
            pivot[("output_token_hash", "adaptive_td")]
            == pivot[("output_token_hash", "failfast")]
        ).astype(int).to_numpy(),
        "adaptive_minus_failfast_verifier_rounds_per_100_tokens": (
            100.0
            * pivot[("num_speculation_rounds", "adaptive_td")]
            / pivot[("output_tokens", "adaptive_td")]
            - 100.0
            * pivot[("num_speculation_rounds", "failfast")]
            / pivot[("output_tokens", "failfast")]
        ).to_numpy(),
        "adaptive_minus_failfast_draft_passes_per_100_tokens": (
            100.0
            * pivot[("total_num_forward_passes", "adaptive_td")]
            / pivot[("output_tokens", "adaptive_td")]
            - 100.0
            * pivot[("total_num_forward_passes", "failfast")]
            / pivot[("output_tokens", "failfast")]
        ).to_numpy(),
    })
    return result


def learning_curves(output_dir, datasets):
    frames = []
    for dataset in datasets:
        path = output_dir / "raw" / dataset / "adaptive_td" / "adaptive_td_decisions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset"] = dataset
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    decisions = pd.concat(frames, ignore_index=True, sort=False)
    rounds = pd.to_numeric(decisions["completed_rounds_before"], errors="coerce")
    decisions["learning_window"] = pd.cut(
        rounds,
        bins=[-1, 99, 499, 999, np.inf],
        labels=["0-99", "100-499", "500-999", "1000+"],
    )
    records = []
    for (dataset, window), group in decisions.groupby(
        ["dataset", "learning_window"], observed=True, sort=False
    ):
        records.append({
            "dataset": dataset,
            "learning_window": str(window),
            "decisions": len(group),
            "stop_rate_percent": 100.0 * group["action"].eq("stop").mean(),
            "exploration_rate_percent": 100.0 * group[
                "exploration_used"
            ].astype(str).str.lower().eq("true").mean(),
            "mean_refinement_step": group["step"].mean(),
            "mean_q_stop": group["q_stop_mean"].mean(),
            "mean_q_continue": group["q_continue_mean"].mean(),
            "mean_controller_latency_ms": group["controller_latency_ms"].mean(),
            "mean_rho_tokens_per_ms": group["rho_tokens_per_ms"].mean(),
            "stop_available_rate_percent": 100.0 * group[
                "stop_available"
            ].astype(str).str.lower().eq("true").mean(),
            "candidate_coverage_rate_percent": 100.0 * group[
                "candidate_coverage_available"
            ].astype(str).str.lower().eq("true").mean(),
            "outer_verify_eligible_rate_percent": 100.0 * group[
                "outer_failfast_verify_eligible"
            ].astype(str).str.lower().eq("true").mean(),
        })
    return decisions, pd.DataFrame(records)


def controller_state_reports(output_dir, datasets, methods):
    state_rows = []
    overhead_rows = []
    for dataset in datasets:
        for method in methods:
            path = output_dir / "raw" / dataset / method / "adaptive_td_runtime_state.json"
            if not path.exists():
                continue
            state = json.loads(path.read_text(encoding="utf-8"))
            for action, values in state.get("actions", {}).items():
                state_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "action": action,
                    "sample_count": values.get("sample_count", 0),
                    "residual_mean": values.get("residual_mean"),
                    "residual_variance": values.get("residual_variance"),
                    "rho_tokens_per_ms": state.get("rho_tokens_per_ms"),
                    "completed_rounds": state.get("completed_rounds", 0),
                })
            for component, values in state.get("overhead", {}).items():
                overhead_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "component": component,
                    **values,
                    "forward_latency_ema_ms": state.get("forward_latency_ema_ms"),
                })
    return pd.DataFrame(state_rows), pd.DataFrame(overhead_rows)


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
    frames = []
    for dataset in args.datasets:
        for method in args.methods:
            frames.append(run_method(args, dataset, method, problem_ids[dataset]))
    rows = pd.concat(frames, ignore_index=True, sort=False)
    summary = aggregate(rows)
    paired = paired_comparison(rows)
    decisions, curves = learning_curves(output_dir, args.datasets)
    controller_states, overhead = controller_state_reports(
        output_dir,
        args.datasets,
        args.methods,
    )
    rows.to_csv(output_dir / "per_problem_results.csv", index=False)
    summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    if not paired.empty:
        paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    if not decisions.empty:
        decisions.to_csv(output_dir / "adaptive_decisions.csv", index=False)
        curves.to_csv(output_dir / "online_learning_curves.csv", index=False)
    if not controller_states.empty:
        controller_states.to_csv(output_dir / "controller_state_summary.csv", index=False)
    if not overhead.empty:
        overhead.to_csv(output_dir / "controller_overhead_summary.csv", index=False)
    print("\nDATASET METHOD SUMMARY")
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\nPAIRED SUMMARY")
        print(
            paired.groupby("dataset", sort=False).agg(
                num_samples=("problem_id", "size"),
                geometric_speedup=(
                    "measured_speedup_vs_failfast",
                    lambda values: float(np.exp(np.log(values).mean())),
                ),
                adaptive_win_rate_percent=("adaptive_wins", lambda values: 100.0 * values.mean()),
                output_match_rate_percent=("output_match", lambda values: 100.0 * values.mean()),
            ).reset_index().to_string(index=False)
        )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
