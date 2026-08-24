import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_failfast_counterfactual_oracle import run_phase
from run_gsm8k_balanced_oracle_dataset import (
    causal_oracle_comparison,
    paired_causal_comparison,
)


VERSION = "math_failfast_future_adjusted_greedy_oracle_v1"
DEFAULT_BASELINE_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "benchmark_references"
    / "math_failfast8_test50"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_output_dir",
        default=str(DEFAULT_BASELINE_OUTPUT_DIR),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_math_future_adjusted_oracle_test50"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def build_future_cost_profile(baseline):
    required = {
        "problem_id",
        "output_tokens",
        "num_speculation_rounds",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
    }
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(f"baseline results missing columns: {sorted(missing)}")
    if baseline["problem_id"].duplicated().any():
        raise ValueError("baseline results must contain one row per problem")

    def stats(frame):
        rounds = float(frame["num_speculation_rounds"].sum())
        if rounds <= 0.0:
            raise ValueError("baseline must contain verifier rounds")
        return {
            "observed_rounds": int(rounds),
            "tokens_per_round": float(frame["output_tokens"].sum() / rounds),
            "draft_ms_per_round": float(
                1000.0 * frame["actual_draft_time"].sum() / rounds
            ),
            "verify_ms_per_round": float(
                1000.0 * frame["actual_verify_time"].sum() / rounds
            ),
            "post_verify_ms_per_round": float(
                1000.0 * frame["actual_post_verify_time"].sum() / rounds
            ),
        }

    return {
        "version": "failfast_per_problem_future_round_profile_v1",
        "source": "measured_unmodified_failfast",
        "global": stats(baseline),
        "per_problem": {
            str(int(row.problem_id)): stats(pd.DataFrame([row]))
            for _, row in baseline.iterrows()
        },
    }


def configure_from_manifest(args, manifest):
    source = manifest["arguments"]
    args.dataset = "math"
    args.num_questions = len(manifest["problem_ids"])
    args.warmup_questions = int(source["warmup_questions"])
    args.max_new_tokens = int(source["max_new_tokens"])
    args.spec_len = int(source["spec_len"])
    args.block_size = int(source["block_size"])
    args.small_block_size = int(source["small_block_size"])
    args.target_model_name = source["target_model_name"]
    args.dllm_dir = source["dllm_dir"]
    args.drafter_threshold = float(source["drafter_threshold"])
    args.lowconf_threshold = float(source["lowconf_threshold"])
    args.max_spec_len = int(source["max_spec_len"])
    args.incr_len = int(source["incr_len"])
    args.seed = int(source["seed"])
    args.adaptive_state_path = None
    return [int(value) for value in manifest["problem_ids"]]


def posthoc_policy_upper_bound(baseline, oracle):
    columns = ["problem_id", "actual_algorithm_time", "output_tokens"]
    paired = baseline[columns].merge(
        oracle[columns],
        on="problem_id",
        suffixes=("_failfast", "_oracle"),
        validate="one_to_one",
    )
    paired["failfast_ms_per_output_token"] = 1000.0 * (
        paired["actual_algorithm_time_failfast"]
        / paired["output_tokens_failfast"].clip(lower=1)
    )
    paired["oracle_ms_per_output_token"] = 1000.0 * (
        paired["actual_algorithm_time_oracle"]
        / paired["output_tokens_oracle"].clip(lower=1)
    )
    use_oracle = paired["oracle_ms_per_output_token"].lt(
        paired["failfast_ms_per_output_token"]
    )
    paired["selected_policy"] = np.where(
        use_oracle,
        "future_adjusted_oracle",
        "failfast",
    )
    paired["selected_time_s"] = np.where(
        use_oracle,
        paired["actual_algorithm_time_oracle"],
        paired["actual_algorithm_time_failfast"],
    )
    paired["selected_output_tokens"] = np.where(
        use_oracle,
        paired["output_tokens_oracle"],
        paired["output_tokens_failfast"],
    )
    baseline_mspt = 1000.0 * (
        paired["actual_algorithm_time_failfast"].sum()
        / paired["output_tokens_failfast"].sum()
    )
    selected_mspt = 1000.0 * (
        paired["selected_time_s"].sum()
        / paired["selected_output_tokens"].sum()
    )
    summary = pd.DataFrame([{
        "num_questions": len(paired),
        "oracle_selected_questions": int(use_oracle.sum()),
        "failfast_selected_questions": int((~use_oracle).sum()),
        "failfast_ms_per_output_token": baseline_mspt,
        "posthoc_best_ms_per_output_token": selected_mspt,
        "posthoc_best_speedup_vs_failfast": baseline_mspt / selected_mspt,
        "posthoc_best_latency_reduction_percent": 100.0
        * (1.0 - selected_mspt / baseline_mspt),
    }])
    return summary, paired


def action_summary(decisions):
    traces = []
    for value in decisions["oracle_action_trace"]:
        traces.extend(json.loads(value))
    trace_frame = pd.DataFrame(traces)
    records = []
    for step, count in decisions["selected_step"].value_counts().sort_index().items():
        records.append({
            "record_type": "selected_step",
            "value": str(int(step)),
            "count": int(count),
            "percent": 100.0 * count / len(decisions),
        })
    if not trace_frame.empty:
        for action, count in trace_frame["action"].value_counts().items():
            records.append({
                "record_type": "greedy_action",
                "value": str(action),
                "count": int(count),
                "percent": 100.0 * count / len(trace_frame),
            })
    return pd.DataFrame(records)


def future_penalty_summary(decisions):
    return pd.DataFrame([{
        "oracle_rounds": len(decisions),
        "mean_profile_tokens_per_round": decisions[
            "profile_tokens_per_round"
        ].mean(),
        "mean_profile_draft_ms_per_round": decisions[
            "profile_draft_ms_per_round"
        ].mean(),
        "mean_profile_verify_ms_per_round": decisions[
            "profile_verify_ms_per_round"
        ].mean(),
        "mean_selected_extra_verifier_rounds": decisions[
            "selected_expected_extra_verifier_rounds"
        ].mean(),
        "sum_selected_extra_verifier_rounds": decisions[
            "selected_expected_extra_verifier_rounds"
        ].sum(),
        "mean_selected_future_draft_penalty_ms": decisions[
            "selected_future_draft_penalty_ms"
        ].mean(),
        "mean_selected_future_verify_penalty_ms": decisions[
            "selected_future_verify_penalty_ms"
        ].mean(),
        "mean_selected_future_round_penalty_ms": decisions[
            "selected_future_round_penalty_ms"
        ].mean(),
        "sum_selected_future_round_penalty_ms": decisions[
            "selected_future_round_penalty_ms"
        ].sum(),
    }])


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_output_dir)
    manifest_path = baseline_dir / "benchmark_manifest.json"
    baseline_path = baseline_dir / "raw" / "failfast" / "benchmark_results.csv"
    if not manifest_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(
            "The preliminary MATH report must contain benchmark_manifest.json "
            "and raw/failfast/benchmark_results.csv"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "math":
        raise ValueError("baseline report must use the MATH dataset")
    problem_ids = configure_from_manifest(args, manifest)
    baseline = pd.read_csv(baseline_path)
    if set(map(int, baseline["problem_id"])) != set(problem_ids):
        raise ValueError("baseline problem IDs do not match sampled_problem_ids")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_future_cost_profile(baseline)
    profile_path = output_dir / "failfast_future_cost_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    oracle_dir = run_phase(
        args,
        problem_ids,
        "future_adjusted_causal_oracle",
        [
            "--collect_bucket_oracle",
            "--causal_oracle",
            "--causal_oracle_future_cost_profile",
            str(profile_path.resolve()),
        ],
        [
            "benchmark_results.csv",
            "causal_oracle_decisions.csv",
            "causal_oracle_candidates.csv",
        ],
    )
    oracle = pd.read_csv(oracle_dir / "benchmark_results.csv")
    decisions = pd.read_csv(oracle_dir / "causal_oracle_decisions.csv")
    summary = causal_oracle_comparison(baseline, oracle, decisions)
    paired = paired_causal_comparison(baseline, oracle)
    upper_summary, upper_paired = posthoc_policy_upper_bound(baseline, oracle)
    actions = action_summary(decisions)
    penalties = future_penalty_summary(decisions)

    baseline.to_csv(output_dir / "baseline_failfast_results.csv", index=False)
    oracle.to_csv(output_dir / "future_adjusted_oracle_results.csv", index=False)
    summary.to_csv(output_dir / "future_adjusted_oracle_summary.csv", index=False)
    paired.to_csv(output_dir / "future_adjusted_oracle_paired.csv", index=False)
    upper_summary.to_csv(output_dir / "posthoc_upper_bound_summary.csv", index=False)
    upper_paired.to_csv(output_dir / "posthoc_upper_bound_paired.csv", index=False)
    actions.to_csv(output_dir / "future_adjusted_action_summary.csv", index=False)
    penalties.to_csv(output_dir / "future_penalty_summary.csv", index=False)
    (output_dir / "source_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sampled_problem_ids.json").write_text(
        json.dumps(problem_ids, indent=2),
        encoding="utf-8",
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    report_manifest = {
        "version": VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_output_dir": str(baseline_dir),
        "output_dir": str(output_dir),
        "problem_ids": problem_ids,
        "oracle_definition": (
            "Sequential greedy comparison of adjacent FailFast refinement snapshots. "
            "Each candidate receives a future-round penalty estimated from measured "
            "per-problem FailFast token progress and draft/verify/post latency."
        ),
        "upper_bound_definition": (
            "Post-hoc per-question best of measured FailFast and the executed "
            "future-adjusted greedy oracle. It is an upper bound over these two "
            "executed policies, not a global decoding oracle."
        ),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(report_manifest, indent=2),
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nFUTURE-ADJUSTED GREEDY ORACLE")
    print(summary.to_string(index=False))
    print("\nPOST-HOC EXECUTED-POLICY UPPER BOUND")
    print(upper_summary.to_string(index=False))
    print("\nACTION SUMMARY")
    print(actions.to_string(index=False))
    print("\nFUTURE-ROUND PENALTY SUMMARY")
    print(penalties.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive}")


if __name__ == "__main__":
    main()
