import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "benchmark_references" / "math_failfast8_test50"
VERSION = "strict_greedy_local_refinement_oracle_math50_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_strict_greedy_math50",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--paired_repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def run_streaming(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
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


def load_reference():
    manifest = json.loads(
        (REFERENCE_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    problem_ids = [int(value) for value in manifest["problem_ids"]]
    if len(problem_ids) != 50 or len(set(problem_ids)) != 50:
        raise ValueError("the bundled MATH reference must contain exactly 50 IDs")
    reference = pd.read_csv(
        REFERENCE_DIR / "raw" / "failfast" / "benchmark_results.csv"
    )
    if set(reference["problem_id"].astype(int)) != set(problem_ids):
        raise ValueError("the bundled baseline rows do not match the 50 MATH IDs")
    return manifest, reference, problem_ids


def phase_complete(phase_dir, expected_rows, require_decisions=False):
    result_path = phase_dir / "benchmark_results.csv"
    call_path = phase_dir / "verifier_calls.csv"
    if not result_path.exists() or not call_path.exists():
        return False
    try:
        results = pd.read_csv(result_path)
        calls = pd.read_csv(call_path)
    except (pd.errors.EmptyDataError, OSError):
        return False
    if len(results) != expected_rows or calls.empty:
        return False
    if require_decisions:
        decision_path = phase_dir / "greedy_local_oracle_decisions.csv"
        policy_path = phase_dir / "strict_greedy_policy.json"
        return (
            decision_path.exists()
            and decision_path.stat().st_size > 0
            and policy_path.exists()
            and policy_path.stat().st_size > 0
        )
    return True


def common_command(source, problem_ids, dllm_dir, output_dir, log_level):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", "math",
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(value) for value in problem_ids],
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(source["max_new_tokens"]),
        "--spec_len", "8",
        "--block_size", str(source["block_size"]),
        "--small_block_size", str(source["small_block_size"]),
        "--target_model_name", source["target_model_name"],
        "--dllm_dir", str(dllm_dir),
        "--drafter_thresholds", str(source["drafter_threshold"]),
        "--sweep_lowconf_threshold", str(source["lowconf_threshold"]),
        "--sweep_max_spec_len", str(source["max_spec_len"]),
        "--sweep_incr_len", "8",
        "--seed", str(source["seed"]),
        "--log_verifier_calls",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", log_level,
    ]


def run_phase(
    args,
    source,
    problem_ids,
    label,
    oracle,
    profile_path=None,
    replay_policy=None,
    require_decisions=False,
):
    phase_dir = Path(args.output_dir) / "raw" / label
    if args.resume and phase_complete(
        phase_dir, len(problem_ids), require_decisions
    ):
        print(f"SKIP completed phase: {label}", flush=True)
        return phase_dir
    if phase_dir.exists():
        shutil.rmtree(phase_dir)
    phase_dir.mkdir(parents=True)
    command = common_command(
        source, problem_ids, args.dllm_dir, phase_dir, args.log_level
    )
    if oracle:
        command.extend([
            "--strict_greedy_local_oracle",
            "--strict_greedy_verifier_profile", str(profile_path),
            "--strict_greedy_epsilon_ms", str(args.epsilon_ms),
        ])
        if replay_policy is not None:
            command.extend([
                "--strict_greedy_replay_policy", str(replay_policy),
            ])
    print("\n" + "=" * 100, flush=True)
    print(f"RUN {label} | method={'oracle' if oracle else 'failfast'}", flush=True)
    print("=" * 100, flush=True)
    run_streaming(command)
    if not phase_complete(phase_dir, len(problem_ids), require_decisions):
        raise RuntimeError(f"phase did not produce complete outputs: {label}")
    return phase_dir


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p5": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def build_verifier_profile(prepass_dir, profile_path):
    calls = pd.read_csv(prepass_dir / "verifier_calls.csv")
    latencies = calls["verify_latency_ms"].astype(float)
    emitted = calls["emitted_tokens"].astype(float)
    latency_stats = describe(latencies)
    emitted_stats = describe(emitted)
    profile = {
        "mean_verify_latency_ms": latency_stats["mean"],
        "median_verify_latency_ms": latency_stats["median"],
        "std_verify_latency_ms": latency_stats["std"],
        "mean_tokens_per_verify": float(emitted.sum() / len(emitted)),
        "arithmetic_mean_emitted_tokens": emitted_stats["mean"],
        "median_emitted_tokens": emitted_stats["median"],
        "std_emitted_tokens": emitted_stats["std"],
        "verifier_calls": int(len(calls)),
        "total_emitted_tokens": int(emitted.sum()),
        "source": "real warmed FailFast-8 prepass verifier calls",
    }
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_phase(phase_dir, label, method):
    results = pd.read_csv(phase_dir / "benchmark_results.csv").copy()
    calls = pd.read_csv(phase_dir / "verifier_calls.csv").copy()
    results["phase"] = label
    results["method"] = method
    calls["phase"] = label
    calls["method"] = method
    decision_path = phase_dir / "greedy_local_oracle_decisions.csv"
    decisions = (
        pd.read_csv(decision_path)
        if decision_path.exists()
        else pd.DataFrame()
    )
    if not decisions.empty:
        decisions["phase"] = label
    return results, calls, decisions


def per_problem_phase(results, calls, incr_len=8):
    call_summary = calls.groupby("problem_id", as_index=False).agg(
        verifier_calls=("round_id", "count"),
        actual_total_verify_latency_ms=("verify_latency_ms", "sum"),
        mean_actual_verify_latency_ms=("verify_latency_ms", "mean"),
        emitted_tokens_from_calls=("emitted_tokens", "sum"),
        blocks=(
            "proposal_length",
            lambda values: int(sum(math.ceil(float(value) / incr_len) for value in values)),
        ),
    )
    merged = results.merge(call_summary, on="problem_id", validate="one_to_one")
    merged["real_latency_ms"] = merged["actual_algorithm_time"] * 1000.0
    merged["ms_per_output_token"] = (
        merged["real_latency_ms"] / merged["output_tokens"].clip(lower=1)
    )
    return merged


def average_method(frames, calls, method):
    phase_rows = [per_problem_phase(frame, call) for frame, call in zip(frames, calls)]
    combined = pd.concat(phase_rows, ignore_index=True)
    numeric = [
        "real_latency_ms",
        "ms_per_output_token",
        "output_tokens",
        "verifier_calls",
        "actual_total_verify_latency_ms",
        "mean_actual_verify_latency_ms",
        "total_num_forward_passes",
        "blocks",
        "num_speculation_rounds",
    ]
    averaged = combined.groupby("problem_id", as_index=False)[numeric].mean()
    hashes = combined.groupby("problem_id")["output_token_hash"].agg(
        lambda values: "|".join(sorted(set(map(str, values))))
    )
    averaged["output_token_hash"] = averaged["problem_id"].map(hashes)
    averaged["method"] = method
    return averaged, combined


def decision_summary(decisions):
    if decisions.empty:
        return pd.DataFrame(columns=["problem_id"])
    decisions = decisions.copy()
    decisions["predicted_signed_call_change"] = np.where(
        decisions["chosen_action"].eq("stop"),
        decisions["predicted_extra_calls_stop"]
        - decisions["predicted_extra_calls_continue"],
        0.0,
    )
    decisions["both_extend"] = (
        decisions["stop_outer_path"].str.startswith("EXTEND")
        & decisions["continue_outer_path"].str.startswith("EXTEND")
    ).astype(int)
    per_phase = decisions.groupby(["phase", "sample_id"], as_index=False).agg(
        predicted_net_verifier_call_change=(
            "predicted_signed_call_change", "sum"
        ),
        total_greedy_decisions=("decision_id", "count"),
        stop_count=("chosen_action", lambda values: int((values == "stop").sum())),
        continue_count=(
            "chosen_action", lambda values: int((values == "continue").sum())
        ),
        decisions_different_from_baseline=("differs_from_baseline", "sum"),
        decisions_changed_by_verify_penalty=("changed_by_verifier_penalty", "sum"),
        verify_to_extend_flips=("verify_to_extend_flip", "sum"),
        both_extend_count=("both_extend", "sum"),
    )
    return per_phase.groupby("sample_id", as_index=False).agg({
        "predicted_net_verifier_call_change": "mean",
        "total_greedy_decisions": "mean",
        "stop_count": "mean",
        "continue_count": "mean",
        "decisions_different_from_baseline": "mean",
        "decisions_changed_by_verify_penalty": "mean",
        "verify_to_extend_flips": "mean",
        "both_extend_count": "mean",
    }).rename(columns={"sample_id": "problem_id"})


def safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def aggregate_report(
    output_dir,
    profile,
    phase_specs,
    loaded,
    problem_ids,
    search_decisions,
    reference,
):
    baseline_frames, baseline_calls = [], []
    oracle_frames, oracle_calls = [], []
    all_results, all_calls = [], []
    for label, method in phase_specs:
        results, calls, decisions = loaded[label]
        all_results.append(results)
        all_calls.append(calls)
        if method == "failfast":
            baseline_frames.append(results)
            baseline_calls.append(calls)
        else:
            oracle_frames.append(results)
            oracle_calls.append(calls)

    baseline, baseline_raw = average_method(
        baseline_frames, baseline_calls, "failfast"
    )
    oracle, oracle_raw = average_method(oracle_frames, oracle_calls, "oracle")
    decision_rows = search_decisions.copy()
    decision_rows["phase"] = "oracle_search"
    decisions = decision_summary(decision_rows)
    paired = baseline.merge(
        oracle,
        on="problem_id",
        suffixes=("_baseline", "_greedy"),
        validate="one_to_one",
    ).merge(decisions, on="problem_id", how="left", validate="one_to_one")
    count_columns = [
        "predicted_net_verifier_call_change",
        "total_greedy_decisions",
        "stop_count",
        "continue_count",
        "decisions_different_from_baseline",
        "decisions_changed_by_verify_penalty",
        "verify_to_extend_flips",
        "both_extend_count",
    ]
    paired[count_columns] = paired[count_columns].fillna(0.0)
    paired["baseline_real_latency_ms"] = paired["real_latency_ms_baseline"]
    paired["greedy_real_latency_ms"] = paired["real_latency_ms_greedy"]
    paired["paired_delta_ms"] = (
        paired["greedy_real_latency_ms"] - paired["baseline_real_latency_ms"]
    )
    paired["real_speedup"] = (
        paired["baseline_real_latency_ms"] / paired["greedy_real_latency_ms"]
    )
    paired["baseline_ms_per_token"] = paired["ms_per_output_token_baseline"]
    paired["greedy_ms_per_token"] = paired["ms_per_output_token_greedy"]
    paired["generated_tokens"] = paired["output_tokens_greedy"]
    paired["baseline_verifier_calls"] = paired["verifier_calls_baseline"]
    paired["greedy_verifier_calls"] = paired["verifier_calls_greedy"]
    paired["actual_net_verifier_call_change"] = (
        paired["greedy_verifier_calls"] - paired["baseline_verifier_calls"]
    )
    paired["verifier_call_prediction_error"] = (
        paired["predicted_net_verifier_call_change"]
        - paired["actual_net_verifier_call_change"]
    )
    paired["baseline_dLLM_forwards"] = paired["total_num_forward_passes_baseline"]
    paired["greedy_dLLM_forwards"] = paired["total_num_forward_passes_greedy"]
    paired["baseline_blocks"] = paired["blocks_baseline"]
    paired["greedy_blocks"] = paired["blocks_greedy"]
    paired["baseline_rounds"] = paired["num_speculation_rounds_baseline"]
    paired["greedy_rounds"] = paired["num_speculation_rounds_greedy"]
    paired["mean_verify_latency_ms"] = profile["mean_verify_latency_ms"]
    paired["mean_tokens_per_verify"] = profile["mean_tokens_per_verify"]
    paired["predicted_total_verify_latency_ms"] = (
        paired["greedy_verifier_calls"] * profile["mean_verify_latency_ms"]
    )
    paired["actual_total_verify_latency_ms"] = (
        paired["actual_total_verify_latency_ms_greedy"]
    )
    paired["execution_order"] = "B-O-O-B" if len(phase_specs) == 4 else "B-O"
    paired["output_hash_match"] = (
        paired["output_token_hash_baseline"] == paired["output_token_hash_greedy"]
    )
    reference_columns = [
        "problem_id",
        "output_token_hash",
        "num_speculation_rounds",
        "total_num_forward_passes",
    ]
    paired = paired.merge(
        reference[reference_columns],
        on="problem_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    paired["baseline_reference_output_match"] = (
        paired["output_token_hash_baseline"].astype(str)
        == paired["output_token_hash"].astype(str)
    )
    paired["baseline_reference_round_match"] = np.isclose(
        paired["baseline_rounds"], paired["num_speculation_rounds"]
    )
    paired["baseline_reference_forward_match"] = np.isclose(
        paired["baseline_dLLM_forwards"], paired["total_num_forward_passes"]
    )

    summary_columns = [
        "problem_id", "baseline_real_latency_ms", "greedy_real_latency_ms",
        "paired_delta_ms", "real_speedup", "generated_tokens",
        "baseline_ms_per_token", "greedy_ms_per_token",
        "baseline_verifier_calls", "greedy_verifier_calls",
        "predicted_net_verifier_call_change", "actual_net_verifier_call_change",
        "verifier_call_prediction_error", "baseline_dLLM_forwards",
        "greedy_dLLM_forwards", "baseline_blocks", "greedy_blocks",
        "baseline_rounds", "greedy_rounds", "total_greedy_decisions",
        "stop_count", "continue_count", "decisions_different_from_baseline",
        "decisions_changed_by_verify_penalty", "verify_to_extend_flips",
        "both_extend_count", "mean_verify_latency_ms", "mean_tokens_per_verify",
        "predicted_total_verify_latency_ms", "actual_total_verify_latency_ms",
        "execution_order", "output_hash_match",
        "baseline_reference_output_match", "baseline_reference_round_match",
        "baseline_reference_forward_match",
    ]
    paired[summary_columns].to_csv(
        output_dir / "greedy_local_oracle_summary.csv", index=False
    )
    decision_rows.to_csv(
        output_dir / "greedy_local_oracle_decisions.csv", index=False
    )
    pd.concat(all_results, ignore_index=True).to_csv(
        output_dir / "all_benchmark_results.csv", index=False
    )
    combined_calls = pd.concat(all_calls, ignore_index=True)
    combined_calls.to_csv(output_dir / "all_verifier_calls.csv", index=False)

    baseline_total = float(paired["baseline_real_latency_ms"].sum())
    greedy_total = float(paired["greedy_real_latency_ms"].sum())
    generated = float(paired["generated_tokens"].sum())
    call_errors = paired["verifier_call_prediction_error"].astype(float)
    actual_call_delta = paired["actual_net_verifier_call_change"].astype(float)
    predicted_call_delta = paired["predicted_net_verifier_call_change"].astype(float)
    oracle_call_rows = pd.concat(oracle_calls, ignore_index=True)
    actual_verify = oracle_call_rows["verify_latency_ms"].astype(float)
    verify_error = profile["mean_verify_latency_ms"] - actual_verify
    decisions_total = float(paired["total_greedy_decisions"].sum())
    identical = paired["decisions_different_from_baseline"].eq(0)
    flip_rows = decision_rows[decision_rows["verify_to_extend_flip"].eq(1)]
    both_extend_rows = decision_rows[
        decision_rows["stop_outer_path"].str.startswith("EXTEND")
        & decision_rows["continue_outer_path"].str.startswith("EXTEND")
    ]
    delta_stats = describe(paired["paired_delta_ms"])
    speed_stats = describe(paired["real_speedup"])
    report = {
        "num_samples": len(problem_ids),
        "total_generated_tokens": generated,
        "baseline_real_latency_ms": baseline_total,
        "greedy_real_latency_ms": greedy_total,
        "pooled_real_speedup": baseline_total / greedy_total,
        "latency_reduction_percent": 100.0 * (1.0 - greedy_total / baseline_total),
        "baseline_ms_per_token": baseline_total / generated,
        "greedy_ms_per_token": greedy_total / generated,
        "baseline_tokens_per_second": 1000.0 * generated / baseline_total,
        "greedy_tokens_per_second": 1000.0 * generated / greedy_total,
        "paired_speedup_mean": speed_stats["mean"],
        "paired_speedup_median": speed_stats["median"],
        "paired_speedup_std": speed_stats["std"],
        "paired_delta_ms_mean": delta_stats["mean"],
        "paired_delta_ms_median": delta_stats["median"],
        "paired_delta_ms_std": delta_stats["std"],
        "wins": int((paired["paired_delta_ms"] < -1.0).sum()),
        "losses": int((paired["paired_delta_ms"] > 1.0).sum()),
        "ties": int((paired["paired_delta_ms"].abs() <= 1.0).sum()),
        "baseline_verifier_calls": float(paired["baseline_verifier_calls"].sum()),
        "greedy_verifier_calls": float(paired["greedy_verifier_calls"].sum()),
        "verifier_call_delta": float(actual_call_delta.sum()),
        "verifier_call_delta_percent": 100.0 * float(actual_call_delta.sum())
        / max(1.0, float(paired["baseline_verifier_calls"].sum())),
        "baseline_dLLM_forwards": float(paired["baseline_dLLM_forwards"].sum()),
        "greedy_dLLM_forwards": float(paired["greedy_dLLM_forwards"].sum()),
        "dLLM_forward_delta": float(
            (paired["greedy_dLLM_forwards"] - paired["baseline_dLLM_forwards"]).sum()
        ),
        "total_greedy_decisions": decisions_total,
        "stop_percent": 100.0 * float(paired["stop_count"].sum()) / max(1.0, decisions_total),
        "continue_percent": 100.0 * float(paired["continue_count"].sum()) / max(1.0, decisions_total),
        "decision_disagreement_percent": 100.0
        * float(paired["decisions_different_from_baseline"].sum())
        / max(1.0, decisions_total),
        "changed_by_verify_penalty_percent": 100.0
        * float(paired["decisions_changed_by_verify_penalty"].sum())
        / max(1.0, decisions_total),
        "verify_to_extend_flip_count": float(paired["verify_to_extend_flips"].sum()),
        "verify_to_extend_continue_win_percent": 100.0
        * float(flip_rows["chosen_action"].eq("continue").mean())
        if len(flip_rows)
        else float("nan"),
        "both_extend_count": int(len(both_extend_rows)),
        "both_extend_continue_win_percent": 100.0
        * float(both_extend_rows["chosen_action"].eq("continue").mean())
        if len(both_extend_rows)
        else float("nan"),
        "mean_real_verifier_latency_ms": float(actual_verify.mean()),
        "mean_emitted_tokens_per_verifier_call": float(
            oracle_call_rows["emitted_tokens"].mean()
        ),
        "verifier_call_prediction_bias": float(call_errors.mean()),
        "verifier_call_prediction_mae": float(call_errors.abs().mean()),
        "verifier_call_prediction_rmse": float(np.sqrt(np.mean(call_errors ** 2))),
        "verifier_call_prediction_correlation": safe_correlation(
            predicted_call_delta, actual_call_delta
        ),
        "mean_verify_latency_prediction_bias_ms": float(verify_error.mean()),
        "mean_verify_latency_prediction_mae_ms": float(verify_error.abs().mean()),
        "mean_verify_latency_prediction_mape_percent": 100.0
        * float((verify_error.abs() / actual_verify.clip(lower=1e-9)).mean()),
        "mean_verify_latency_prediction_rmse_ms": float(
            np.sqrt(np.mean(verify_error ** 2))
        ),
        "actual_verify_latency_std_ms": float(actual_verify.std(ddof=1)),
        "actual_verify_latency_p5_ms": float(actual_verify.quantile(0.05)),
        "actual_verify_latency_p95_ms": float(actual_verify.quantile(0.95)),
        "output_hash_match_percent": 100.0 * float(paired["output_hash_match"].mean()),
        "baseline_reference_output_match_percent": 100.0
        * float(paired["baseline_reference_output_match"].mean()),
        "baseline_reference_round_match_percent": 100.0
        * float(paired["baseline_reference_round_match"].mean()),
        "baseline_reference_forward_match_percent": 100.0
        * float(paired["baseline_reference_forward_match"].mean()),
        "identical_action_samples": int(identical.sum()),
        "identical_action_mean_timing_delta_ms": float(
            paired.loc[identical, "paired_delta_ms"].mean()
        ) if identical.any() else float("nan"),
        "identical_action_pooled_speedup": float(
            paired.loc[identical, "baseline_real_latency_ms"].sum()
            / paired.loc[identical, "greedy_real_latency_ms"].sum()
        ) if identical.any() else float("nan"),
        "lcp_used": False,
        "lcp_candidates_checked": 0,
        "lcp_semantic_mismatch_count": 0,
    }
    report_frame = pd.DataFrame([report])
    report_frame.to_csv(
        output_dir / "greedy_local_oracle_dataset_report.csv", index=False
    )
    return report_frame


def main():
    args = parse_args()
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")
    manifest, reference, problem_ids = load_reference()
    source = manifest["arguments"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    prepass_dir = run_phase(
        args, source, problem_ids, "verifier_profile_prepass", False
    )
    profile_path = output_dir / "verifier_profile.json"
    profile = build_verifier_profile(prepass_dir, profile_path)

    search_dir = run_phase(
        args,
        source,
        problem_ids,
        "oracle_search",
        True,
        profile_path,
        require_decisions=True,
    )
    policy_path = search_dir / "strict_greedy_policy.json"
    search_decisions = pd.read_csv(
        search_dir / "greedy_local_oracle_decisions.csv"
    )

    phase_specs = [("baseline_1", "failfast"), ("oracle_1", "oracle")]
    if args.paired_repetitions == 2:
        phase_specs.extend([("oracle_2", "oracle"), ("baseline_2", "failfast")])
    loaded = {}
    for label, method in phase_specs:
        phase_dir = run_phase(
            args,
            source,
            problem_ids,
            label,
            method == "oracle",
            profile_path,
            replay_policy=policy_path if method == "oracle" else None,
        )
        loaded[label] = load_phase(phase_dir, label, method)

    report = aggregate_report(
        output_dir,
        profile,
        phase_specs,
        loaded,
        problem_ids,
        search_decisions,
        reference,
    )
    report_manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "problem_ids": problem_ids,
        "source_reference": str(REFERENCE_DIR),
        "source_arguments": source,
        "paired_repetitions": args.paired_repetitions,
        "execution_order": [label for label, _ in phase_specs],
        "epsilon_ms": args.epsilon_ms,
        "oracle_definition": (
            "Rolling strict one-step greedy comparison of STOP@t against "
            "CONTINUE@t then forced STOP@(t+1). Both branches execute original "
            "FailFast outer EXTEND behavior to the next real verifier boundary."
        ),
        "primary_timing": (
            "Real selected-path draft wall latency plus verifier and post-verifier "
            "latency. Counterfactual search replay time is excluded."
        ),
        "elapsed_runner_hours": (time.time() - started) / 3600.0,
    }
    try:
        report_manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        report_manifest["git_commit"] = None
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(report_manifest, indent=2), encoding="utf-8"
    )
    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print("\nSTRICT GREEDY LOCAL ORACLE DATASET REPORT")
    print(report.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive}")


if __name__ == "__main__":
    main()
