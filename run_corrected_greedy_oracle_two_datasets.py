import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from run_strict_greedy_math50 import (
    REFERENCE_DIR,
    aggregate_report,
    build_verifier_profile,
    load_phase,
    run_phase,
)


ROOT = Path(__file__).resolve().parent
VERSION = "corrected_one_action_greedy_oracle_two_datasets_v1"
GSM8K_SIZE = 1319


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--paired_repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_corrected_greedy_oracle_math_gsm8k",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def math_configuration(num_questions):
    manifest = json.loads(
        (REFERENCE_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    problem_ids = [int(value) for value in manifest["problem_ids"]]
    if num_questions > len(problem_ids):
        raise ValueError(
            f"bundled MATH reference contains only {len(problem_ids)} IDs"
        )
    source = dict(manifest["arguments"])
    source["dataset"] = "math"
    return source, problem_ids[:num_questions]


def gsm8k_configuration(num_questions, sample_seed):
    if num_questions <= 0 or num_questions > GSM8K_SIZE - 1:
        raise ValueError(f"GSM8K num_questions must be in [1, {GSM8K_SIZE - 1}]")
    problem_ids = sorted(random.Random(sample_seed).sample(
        list(range(1, GSM8K_SIZE)),
        num_questions,
    ))
    source = {
        "dataset": "gsm8k",
        "max_new_tokens": 1024,
        "block_size": 32,
        "small_block_size": 8,
        "target_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "drafter_threshold": 0.05,
        "lowconf_threshold": 0.45,
        "max_spec_len": 60,
        "seed": 42,
    }
    return source, problem_ids


def dataset_runner_args(args, dataset_dir):
    return SimpleNamespace(
        output_dir=str(dataset_dir),
        dllm_dir=args.dllm_dir,
        paired_repetitions=args.paired_repetitions,
        epsilon_ms=args.epsilon_ms,
        resume=args.resume,
        log_level=args.log_level,
    )


def run_dataset(args, dataset, source, problem_ids):
    dataset_dir = Path(args.output_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_args = dataset_runner_args(args, dataset_dir)
    started = time.time()

    prepass_dir = run_phase(
        local_args,
        source,
        problem_ids,
        "verifier_profile_prepass",
        False,
    )
    profile_path = dataset_dir / "verifier_profile.json"
    profile = build_verifier_profile(prepass_dir, profile_path)
    reference = pd.read_csv(prepass_dir / "benchmark_results.csv")

    search_dir = run_phase(
        local_args,
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
    search_decisions["oracle_stop"] = (
        search_decisions["chosen_action"].eq("stop").astype(int)
    )
    search_decisions["score_margin_ms"] = search_decisions["DeltaJ_ms"]
    search_decisions["absolute_score_margin_ms"] = (
        search_decisions["DeltaJ_ms"].abs()
    )
    search_decisions["near_tie"] = (
        search_decisions["absolute_score_margin_ms"] <= args.epsilon_ms
    ).astype(int)

    phase_specs = [("baseline_1", "failfast"), ("oracle_1", "oracle")]
    if args.paired_repetitions == 2:
        phase_specs.extend([("oracle_2", "oracle"), ("baseline_2", "failfast")])
    loaded = {}
    for label, method in phase_specs:
        phase_dir = run_phase(
            local_args,
            source,
            problem_ids,
            label,
            method == "oracle",
            profile_path,
            replay_policy=policy_path if method == "oracle" else None,
        )
        loaded[label] = load_phase(phase_dir, label, method)

    report = aggregate_report(
        dataset_dir,
        profile,
        phase_specs,
        loaded,
        problem_ids,
        search_decisions,
        reference,
    )
    search_decisions.insert(0, "dataset", dataset)
    search_decisions.to_csv(
        dataset_dir / "corrected_oracle_feature_labels.csv",
        index=False,
    )
    manifest = {
        "version": VERSION,
        "dataset": dataset,
        "problem_ids": problem_ids,
        "source_arguments": source,
        "paired_repetitions": args.paired_repetitions,
        "execution_order": [label for label, _ in phase_specs],
        "epsilon_ms": args.epsilon_ms,
        "oracle_definition": (
            "At each real refinement state, compare STOP@t and CONTINUE@t. "
            "Only the current action is oracle-controlled; all subsequent "
            "counterfactual refinement actions use baseline FailFast-8 until "
            "the next verifier boundary."
        ),
        "feature_label_definition": (
            "chosen_action minimizes local branch cost plus the empirical "
            "fractional future-verifier-call penalty"
        ),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (dataset_dir / "corrected_oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    result = report.copy()
    result.insert(0, "dataset", dataset)
    return result, search_decisions


def main():
    args = parse_args()
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    configurations = {
        "math": math_configuration(args.num_questions),
        "gsm8k": gsm8k_configuration(args.num_questions, args.sample_seed),
    }
    reports = []
    decisions = []
    for dataset, (source, problem_ids) in configurations.items():
        print("\n" + "=" * 100, flush=True)
        print(
            f"DATASET {dataset.upper()} | questions={len(problem_ids)} | "
            "oracle=corrected_one_action_baseline_rollout",
            flush=True,
        )
        print("=" * 100, flush=True)
        report, decision_rows = run_dataset(
            args,
            dataset,
            source,
            problem_ids,
        )
        reports.append(report)
        decisions.append(decision_rows)

    combined_report = pd.concat(reports, ignore_index=True)
    combined_decisions = pd.concat(decisions, ignore_index=True)
    combined_report.to_csv(
        output_dir / "two_dataset_corrected_oracle_summary.csv",
        index=False,
    )
    combined_decisions.to_csv(
        output_dir / "two_dataset_corrected_oracle_feature_labels.csv",
        index=False,
    )
    (output_dir / "benchmark_manifest.json").write_text(json.dumps({
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "datasets": list(configurations),
        "num_questions_per_dataset": args.num_questions,
        "sample_seed": args.sample_seed,
        "paired_repetitions": args.paired_repetitions,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }, indent=2), encoding="utf-8")

    print("\nTWO-DATASET CORRECTED ORACLE SUMMARY", flush=True)
    print(combined_report.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
