import argparse
import json
import math
import os
import platform
import random
import shutil
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
VERSION = "failfast_counterfactual_oracle_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_SIZES), default="gsm8k")
    parser.add_argument("--num_questions", type=int, default=50)
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
    parser.add_argument("--adaptive_state_path")
    parser.add_argument("--oracle_only", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_failfast_counterfactual_oracle_test50",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    available = DATASET_SIZES[args.dataset] - args.warmup_questions
    if args.num_questions <= 0 or args.num_questions > available:
        raise ValueError(f"--num_questions must be in [1, {available}]")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("proposal lengths must be positive")


def sampled_problem_ids(args):
    population = list(range(args.warmup_questions, DATASET_SIZES[args.dataset]))
    return sorted(random.Random(args.sample_seed).sample(population, args.num_questions))


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
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def base_command(args, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", args.dataset,
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


def run_phase(args, problem_ids, phase, extra_args, required_files):
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) / "raw" / phase
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = all((output_dir / filename).exists() for filename in required_files)
    if args.resume and complete:
        print(f"RESUME {phase}", flush=True)
        return output_dir
    for filename in required_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    command = base_command(args, problem_ids, output_dir) + list(extra_args)
    print("\n" + "=" * 100, flush=True)
    print(f"RUN {phase} | samples={len(problem_ids)}", flush=True)
    print("=" * 100, flush=True)
    run_streaming(command, root)
    return output_dir


def train_or_locate_state(args, problem_ids):
    if args.adaptive_state_path:
        state_path = Path(args.adaptive_state_path)
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        return state_path
    training_dir = run_phase(
        args,
        problem_ids,
        "adaptive_training",
        ["--adaptive-td", "--adaptive-log-decisions"],
        ["benchmark_results.csv", "adaptive_td_runtime_state.json"],
    )
    return training_dir / "adaptive_td_runtime_state.json"


def add_latency_estimates(snapshots):
    result = snapshots.copy()
    result["context_bucket"] = (
        pd.to_numeric(result["context_len"], errors="coerce").fillna(0).astype(int) // 256
    )
    result["proposal_bucket"] = np.ceil(
        pd.to_numeric(result["target_len"], errors="coerce").fillna(1) / 8.0
    ).astype(int)
    keys = ["context_bucket", "proposal_bucket"]
    result["estimated_verify_latency_ms"] = result.groupby(keys)[
        "actual_verify_latency_ms"
    ].transform("median")
    result["estimated_post_verify_latency_ms"] = result.groupby(keys)[
        "actual_post_verify_latency_ms"
    ].transform("median")
    return result


def optional_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def build_transitions(snapshots, fallback_rho):
    records = []
    group_columns = ["problem_id", "round_id", "target_len"]
    for keys, group in snapshots.groupby(group_columns, sort=False):
        group = group.sort_values(["step", "draft_passes_elapsed"]).reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]
            pass_delta = int(following["draft_passes_elapsed"]) - int(
                current["draft_passes_elapsed"]
            )
            if int(following["step"]) != int(current["step"]) + 1 or pass_delta != 1:
                continue
            if not optional_bool(current.get("adaptive_stop_available")):
                continue
            predicted_action = str(current.get("adaptive_policy_action", "")).lower()
            if predicted_action not in {"stop", "continue"}:
                continue
            rho = pd.to_numeric(
                pd.Series([current.get("adaptive_rho_tokens_per_ms")]),
                errors="coerce",
            ).iloc[0]
            rho = float(rho) if pd.notna(rho) and rho > 0 else float(fallback_rho)
            draft_delta_ms = max(
                0.0,
                float(following["draft_latency_elapsed_ms"])
                - float(current["draft_latency_elapsed_ms"]),
            )
            stop_future_ms = (
                float(current["estimated_verify_latency_ms"])
                + float(current["estimated_post_verify_latency_ms"])
            )
            continue_future_ms = (
                draft_delta_ms
                + float(following["estimated_verify_latency_ms"])
                + float(following["estimated_post_verify_latency_ms"])
            )
            stop_output = float(current["emitted_len_if_stop"])
            continue_output = float(following["emitted_len_if_stop"])
            stop_utility = stop_output - rho * stop_future_ms
            continue_utility = continue_output - rho * continue_future_ms
            stop_total_ms = float(current["draft_latency_elapsed_ms"]) + stop_future_ms
            continue_total_ms = (
                float(following["draft_latency_elapsed_ms"])
                + float(following["estimated_verify_latency_ms"])
                + float(following["estimated_post_verify_latency_ms"])
            )
            stop_ms_per_output = stop_total_ms / max(1.0, stop_output)
            continue_ms_per_output = continue_total_ms / max(1.0, continue_output)
            oracle_action = (
                "continue" if continue_ms_per_output < stop_ms_per_output else "stop"
            )
            oracle_utility_action = (
                "continue" if continue_utility > stop_utility else "stop"
            )
            selected_utility = stop_utility if predicted_action == "stop" else continue_utility
            selected_ms_per_output = (
                stop_ms_per_output
                if predicted_action == "stop"
                else continue_ms_per_output
            )
            record = dict(zip(group_columns, keys))
            record.update({
                "from_step": int(current["step"]),
                "to_step": int(following["step"]),
                "predicted_action": predicted_action,
                "predicted_reason": current.get("adaptive_policy_reason"),
                "stop_probability": current.get("adaptive_stop_probability"),
                "q_stop_mean": current.get("adaptive_q_stop_mean"),
                "q_continue_mean": current.get("adaptive_q_continue_mean"),
                "rho_tokens_per_ms": rho,
                "stop_output_tokens": stop_output,
                "continue_output_tokens": continue_output,
                "actual_next_gain_tokens": continue_output - stop_output,
                "next_draft_latency_ms": draft_delta_ms,
                "stop_future_latency_ms": stop_future_ms,
                "continue_future_latency_ms": continue_future_ms,
                "stop_ms_per_output_token": stop_ms_per_output,
                "continue_ms_per_output_token": continue_ms_per_output,
                "stop_utility_tokens": stop_utility,
                "continue_utility_tokens": continue_utility,
                "oracle_action": oracle_action,
                "oracle_utility_action": oracle_utility_action,
                "decision_correct": int(predicted_action == oracle_action),
                "false_stop": int(predicted_action == "stop" and oracle_action == "continue"),
                "missed_stop": int(predicted_action == "continue" and oracle_action == "stop"),
                "regret_tokens": max(stop_utility, continue_utility) - selected_utility,
                "regret_ms_per_output_token": (
                    selected_ms_per_output
                    - min(stop_ms_per_output, continue_ms_per_output)
                ),
                "regret_ms_equivalent": (
                    (max(stop_utility, continue_utility) - selected_utility) / rho
                    if rho > 0 else np.nan
                ),
            })
            records.append(record)
    return pd.DataFrame(records)


def build_failfast_oracle_transitions(snapshots):
    records = []
    group_columns = ["problem_id", "round_id", "target_len"]
    for keys, group in snapshots.groupby(group_columns, sort=False):
        group = group.sort_values(["step", "draft_passes_elapsed"]).reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            following = group.iloc[index + 1]
            pass_delta = int(following["draft_passes_elapsed"]) - int(
                current["draft_passes_elapsed"]
            )
            if int(following["step"]) != int(current["step"]) + 1 or pass_delta != 1:
                continue
            stop_output = float(current["emitted_len_if_stop"])
            continue_output = float(following["emitted_len_if_stop"])
            stop_total_ms = (
                float(current["draft_latency_elapsed_ms"])
                + float(current["estimated_verify_latency_ms"])
                + float(current["estimated_post_verify_latency_ms"])
            )
            continue_total_ms = (
                float(following["draft_latency_elapsed_ms"])
                + float(following["estimated_verify_latency_ms"])
                + float(following["estimated_post_verify_latency_ms"])
            )
            stop_ms_per_output = stop_total_ms / max(1.0, stop_output)
            continue_ms_per_output = continue_total_ms / max(1.0, continue_output)
            record = dict(zip(group_columns, keys))
            record.update({
                "from_step": int(current["step"]),
                "to_step": int(following["step"]),
                "draft_pass_delta": pass_delta,
                "stop_output_tokens": stop_output,
                "continue_output_tokens": continue_output,
                "actual_next_gain_tokens": continue_output - stop_output,
                "next_draft_latency_ms": max(
                    0.0,
                    float(following["draft_latency_elapsed_ms"])
                    - float(current["draft_latency_elapsed_ms"]),
                ),
                "stop_ms_per_output_token": stop_ms_per_output,
                "continue_ms_per_output_token": continue_ms_per_output,
                "oracle_action": (
                    "continue"
                    if continue_ms_per_output < stop_ms_per_output
                    else "stop"
                ),
                "oracle_advantage_ms_per_output_token": (
                    stop_ms_per_output - continue_ms_per_output
                ),
            })
            records.append(record)
    return pd.DataFrame(records)


def policy_summary(transitions):
    if transitions.empty:
        raise ValueError("No exact adjacent one-pass counterfactual transitions were found")
    predicted_stop = transitions["predicted_action"].eq("stop")
    oracle_stop = transitions["oracle_action"].eq("stop")
    true_stop = int((predicted_stop & oracle_stop).sum())
    false_stop = int((predicted_stop & ~oracle_stop).sum())
    missed_stop = int((~predicted_stop & oracle_stop).sum())
    return pd.DataFrame([{
        "transitions": len(transitions),
        "predicted_stop_count": int(predicted_stop.sum()),
        "oracle_stop_count": int(oracle_stop.sum()),
        "decision_accuracy_percent": 100.0 * transitions["decision_correct"].mean(),
        "stop_precision_percent": 100.0 * true_stop / max(1, true_stop + false_stop),
        "false_stop_rate_percent": 100.0 * false_stop / max(1, predicted_stop.sum()),
        "stop_recall_percent": 100.0 * true_stop / max(1, oracle_stop.sum()),
        "missed_stop_rate_percent": 100.0 * missed_stop / max(1, oracle_stop.sum()),
        "mean_regret_tokens": transitions["regret_tokens"].mean(),
        "mean_regret_ms_per_output_token": transitions[
            "regret_ms_per_output_token"
        ].mean(),
        "p95_regret_ms_per_output_token": transitions[
            "regret_ms_per_output_token"
        ].quantile(0.95),
        "mean_regret_ms_equivalent": transitions["regret_ms_equivalent"].mean(),
        "p95_regret_ms_equivalent": transitions["regret_ms_equivalent"].quantile(0.95),
    }])


def select_round_candidates(snapshots):
    records = []
    for keys, group in snapshots.groupby(["problem_id", "round_id"], sort=False):
        group = group.sort_values(["draft_passes_elapsed", "target_len", "step"])
        group = group.assign(
            replay_total_latency_ms=(
                group["draft_latency_elapsed_ms"]
                + group["estimated_verify_latency_ms"]
                + group["estimated_post_verify_latency_ms"]
            ),
        )
        group["replay_ms_per_output_token"] = (
            group["replay_total_latency_ms"]
            / group["emitted_len_if_stop"].clip(lower=1)
        )
        factual = group.iloc[-1]
        oracle = group.loc[group["replay_ms_per_output_token"].idxmin()]
        stop_rows = (
            group[group["adaptive_policy_action"].astype(str).str.lower().eq("stop")]
            if "adaptive_policy_action" in group
            else group.iloc[0:0]
        )
        policy = stop_rows.iloc[0] if len(stop_rows) else factual
        records.append({
            "problem_id": keys[0],
            "round_id": keys[1],
            "factual_step": factual["step"],
            "policy_step": policy["step"],
            "oracle_step": oracle["step"],
            "factual_output_tokens": factual["emitted_len_if_stop"],
            "policy_output_tokens": policy["emitted_len_if_stop"],
            "oracle_output_tokens": oracle["emitted_len_if_stop"],
            "factual_latency_ms": factual["replay_total_latency_ms"],
            "policy_latency_ms": policy["replay_total_latency_ms"],
            "oracle_latency_ms": oracle["replay_total_latency_ms"],
            "policy_matches_oracle_step": int(policy.name == oracle.name),
        })
    return pd.DataFrame(records)


def pooled_ms_per_token(rounds, prefix):
    return rounds[f"{prefix}_latency_ms"].sum() / max(
        1.0,
        rounds[f"{prefix}_output_tokens"].sum(),
    )


def speed_summary(baseline_results, replay_results, rounds):
    baseline_time_s = (
        baseline_results["actual_draft_time"].sum()
        + baseline_results["actual_verify_time"].sum()
        + baseline_results["actual_post_verify_time"].sum()
    )
    baseline_mspt = 1000.0 * baseline_time_s / baseline_results["output_tokens"].sum()
    factual_mspt = pooled_ms_per_token(rounds, "factual")
    policy_mspt = pooled_ms_per_token(rounds, "policy")
    oracle_mspt = pooled_ms_per_token(rounds, "oracle")
    hashes_match = baseline_results[["problem_id", "output_token_hash"]].merge(
        replay_results[["problem_id", "output_token_hash"]],
        on="problem_id",
        suffixes=("_baseline", "_replay"),
    )
    return pd.DataFrame([{
        "num_questions": baseline_results["problem_id"].nunique(),
        "num_rounds": len(rounds),
        "output_hash_match_percent": 100.0 * (
            hashes_match["output_token_hash_baseline"]
            == hashes_match["output_token_hash_replay"]
        ).mean(),
        "measured_failfast_ms_per_output_token": baseline_mspt,
        "replay_factual_ms_per_output_token": factual_mspt,
        "replay_controller_ms_per_output_token": policy_mspt,
        "replay_oracle_ms_per_output_token": oracle_mspt,
        "oracle_replay_speedup_vs_failfast_replay": factual_mspt / oracle_mspt,
        "controller_replay_speedup_vs_failfast_replay": factual_mspt / policy_mspt,
        "oracle_replay_speedup_vs_controller_replay": policy_mspt / oracle_mspt,
        "measured_failfast_to_local_oracle_upper_bound": baseline_mspt / oracle_mspt,
        "controller_matches_oracle_step_percent": 100.0 * rounds[
            "policy_matches_oracle_step"
        ].mean(),
    }])


def oracle_only_speed_summary(results, rounds, transitions):
    measured_time_s = (
        results["actual_draft_time"].sum()
        + results["actual_verify_time"].sum()
        + results["actual_post_verify_time"].sum()
    )
    measured_mspt = 1000.0 * measured_time_s / results["output_tokens"].sum()
    factual_mspt = pooled_ms_per_token(rounds, "factual")
    oracle_mspt = pooled_ms_per_token(rounds, "oracle")
    return pd.DataFrame([{
        "num_questions": results["problem_id"].nunique(),
        "num_rounds": len(rounds),
        "counterfactual_transitions": len(transitions),
        "measured_instrumented_failfast_ms_per_output_token": measured_mspt,
        "replay_factual_failfast_ms_per_output_token": factual_mspt,
        "replay_oracle_ms_per_output_token": oracle_mspt,
        "oracle_replay_speedup_vs_failfast_replay": factual_mspt / oracle_mspt,
        "oracle_continue_rate_percent": 100.0
        * transitions["oracle_action"].eq("continue").mean(),
        "oracle_stop_rate_percent": 100.0
        * transitions["oracle_action"].eq("stop").mean(),
        "mean_oracle_advantage_ms_per_output_token": transitions[
            "oracle_advantage_ms_per_output_token"
        ].abs().mean(),
    }])


def calibration_table(transitions):
    evaluated = transitions.dropna(subset=["stop_probability"]).copy()
    evaluated["probability_bin"] = pd.cut(
        evaluated["stop_probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
    )
    return evaluated.groupby("probability_bin", observed=True).agg(
        decisions=("decision_correct", "size"),
        predicted_stop_probability_mean=("stop_probability", "mean"),
        oracle_stop_rate=("oracle_action", lambda values: values.eq("stop").mean()),
        decision_accuracy=("decision_correct", "mean"),
        regret_ms_equivalent=("regret_ms_equivalent", "mean"),
    ).reset_index()


def write_manifest(args, output_dir, problem_ids, state_path):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "version": VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "problem_ids": problem_ids,
        "adaptive_state_path": str(state_path),
        "interpretation": (
            "Oracle speedup is a local replay upper bound. Counterfactual verifier "
            "calls are diagnostic, excluded from measured FailFast latency, and never "
            "used to update the frozen controller."
        ),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = sampled_problem_ids(args)

    if args.oracle_only:
        oracle_dir = run_phase(
            args,
            problem_ids,
            "failfast_oracle_only",
            ["--collect_bucket_oracle"],
            ["benchmark_results.csv", "bucket_oracle_snapshots.csv"],
        )
        results = pd.read_csv(oracle_dir / "benchmark_results.csv")
        snapshots = add_latency_estimates(
            pd.read_csv(oracle_dir / "bucket_oracle_snapshots.csv")
        )
        transitions = build_failfast_oracle_transitions(snapshots)
        if transitions.empty:
            raise ValueError("No adjacent one-pass FailFast transitions were collected")
        rounds = select_round_candidates(snapshots)
        speed = oracle_only_speed_summary(results, rounds, transitions)

        results.to_csv(output_dir / "failfast_results.csv", index=False)
        snapshots.to_csv(output_dir / "counterfactual_snapshots.csv", index=False)
        transitions.to_csv(output_dir / "counterfactual_transitions.csv", index=False)
        rounds.to_csv(output_dir / "round_oracle_choices.csv", index=False)
        speed.to_csv(output_dir / "oracle_speed_summary.csv", index=False)
        write_manifest(args, output_dir, problem_ids, "not_used_oracle_only")
        archive_path = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
        print("\nFAILFAST LOCAL ORACLE SPEED SUMMARY")
        print(speed.to_string(index=False))
        print(f"\nSaved report: {output_dir}")
        print(f"Saved archive: {archive_path}")
        return

    baseline_dir = run_phase(
        args,
        problem_ids,
        "failfast_baseline",
        [],
        ["benchmark_results.csv"],
    )
    state_path = train_or_locate_state(args, problem_ids)
    replay_dir = run_phase(
        args,
        problem_ids,
        "failfast_counterfactual_replay",
        [
            "--adaptive-td",
            "--adaptive-state-path", str(state_path),
            "--adaptive-freeze",
            "--adaptive-counterfactual-replay",
            "--collect_bucket_oracle",
        ],
        ["benchmark_results.csv", "bucket_oracle_snapshots.csv"],
    )

    baseline = pd.read_csv(baseline_dir / "benchmark_results.csv")
    replay = pd.read_csv(replay_dir / "benchmark_results.csv")
    snapshots = add_latency_estimates(
        pd.read_csv(replay_dir / "bucket_oracle_snapshots.csv")
    )
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    transitions = build_transitions(snapshots, state.get("rho_tokens_per_ms", 0.0))
    decisions = policy_summary(transitions)
    rounds = select_round_candidates(snapshots)
    speed = speed_summary(baseline, replay, rounds)
    calibration = calibration_table(transitions)

    baseline.to_csv(output_dir / "failfast_baseline_results.csv", index=False)
    replay.to_csv(output_dir / "failfast_replay_results.csv", index=False)
    snapshots.to_csv(output_dir / "counterfactual_snapshots.csv", index=False)
    transitions.to_csv(output_dir / "counterfactual_transitions.csv", index=False)
    rounds.to_csv(output_dir / "round_oracle_choices.csv", index=False)
    decisions.to_csv(output_dir / "policy_decision_summary.csv", index=False)
    speed.to_csv(output_dir / "oracle_speed_summary.csv", index=False)
    calibration.to_csv(output_dir / "stop_probability_calibration.csv", index=False)
    write_manifest(args, output_dir, problem_ids, state_path)
    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )

    print("\nFAILFAST LOCAL ORACLE SPEED SUMMARY")
    print(speed.to_string(index=False))
    print("\nFROZEN CONTROLLER DECISION SUMMARY")
    print(decisions.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
