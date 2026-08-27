import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from capacity_audit import COMPACT6, run_capacity_audit
from run_corrected_greedy_oracle_two_datasets import (
    gsm8k_configuration,
    math_configuration,
)
from run_strict_greedy_math50 import (
    build_verifier_profile,
    common_command,
    phase_complete,
    run_phase,
    run_streaming,
)


ROOT = Path(__file__).resolve().parent
VERSION = "failfast_support_compact6_capacity_audit_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=30)
    parser.add_argument("--reference_pool_size", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--bootstrap_repetitions", type=int, default=500)
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_compact6_capacity_math30_gsm8k30",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _local_args(args, dataset_dir):
    return SimpleNamespace(
        output_dir=str(dataset_dir),
        dllm_dir=args.dllm_dir,
        paired_repetitions=1,
        epsilon_ms=args.epsilon_ms,
        resume=args.resume,
        log_level=args.log_level,
    )


def run_collector(args, source, problem_ids, dataset_dir, profile_path):
    phase_dir = dataset_dir / "raw" / "failfast_support_capacity_collector"
    if args.resume and phase_complete(
        phase_dir, len(problem_ids), require_decisions=True
    ):
        print("SKIP completed capacity collector", flush=True)
        return phase_dir
    if phase_dir.exists():
        shutil.rmtree(phase_dir)
    phase_dir.mkdir(parents=True)
    command = common_command(
        source, problem_ids, args.dllm_dir, phase_dir, args.log_level
    )
    command.extend((
        "--strict_greedy_local_oracle",
        "--strict_greedy_capacity_collector",
        "--strict_greedy_verifier_profile", str(profile_path),
        "--strict_greedy_epsilon_ms", str(args.epsilon_ms),
        "--adaptive-factual-ema-alpha", "0.2",
    ))
    print("\n" + "=" * 100, flush=True)
    print(
        f"RUN FAILFAST-SUPPORT CAPACITY COLLECTOR | questions={len(problem_ids)}",
        flush=True,
    )
    print("=" * 100, flush=True)
    run_streaming(command)
    if not phase_complete(phase_dir, len(problem_ids), require_decisions=True):
        raise RuntimeError("capacity collector did not produce complete outputs")
    return phase_dir


def run_dataset(args, dataset, source, problem_ids):
    dataset_dir = Path(args.output_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_args = _local_args(args, dataset_dir)
    prepass = run_phase(
        local_args,
        source,
        problem_ids,
        "independent_frozen_profile_prepass",
        False,
    )
    profile_path = dataset_dir / "frozen_verifier_profile.json"
    profile = build_verifier_profile(prepass, profile_path)
    collector = run_collector(
        args, source, problem_ids, dataset_dir, profile_path
    )
    decisions = pd.read_csv(collector / "greedy_local_oracle_decisions.csv")
    decisions.insert(0, "dataset", dataset)
    if not decisions["executed_action"].eq("continue").all():
        raise RuntimeError("collector left FailFast CONTINUE support")
    expected_features = {f"feature_{name}" for name in COMPACT6}
    missing = expected_features.difference(decisions.columns)
    if missing:
        raise RuntimeError(f"collector omitted Compact6 features: {sorted(missing)}")
    decisions.to_csv(dataset_dir / "capacity_states.csv", index=False)
    metadata = {
        "dataset": dataset,
        "problem_ids": [int(value) for value in problem_ids],
        "source_arguments": source,
        "profile": profile,
        "collector_support": "unmodified FailFast-8 (CONTINUE at every adaptive hook)",
    }
    (dataset_dir / "collector_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return decisions


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
        "gsm8k": gsm8k_configuration(
            args.num_questions, args.sample_seed, args.reference_pool_size
        ),
    }
    frames = []
    for dataset, (source, problem_ids) in configurations.items():
        print("\n" + "#" * 100, flush=True)
        print(f"DATASET {dataset.upper()} | questions={len(problem_ids)}", flush=True)
        print("#" * 100, flush=True)
        frames.append(run_dataset(args, dataset, source, problem_ids))
    decisions = pd.concat(frames, ignore_index=True)
    decisions.to_csv(output_dir / "compact6_capacity_states.csv", index=False)
    metrics = run_capacity_audit(
        decisions,
        output_dir / "audit",
        epsilon_ms=args.epsilon_ms,
        seed=42,
        bootstrap=args.bootstrap_repetitions,
    )
    manifest = {
        "version": VERSION,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "datasets": list(configurations),
        "num_questions_per_dataset": args.num_questions,
        "epsilon_ms_native_delta_j": args.epsilon_ms,
        "feature_schema": list(COMPACT6),
        "profile_policy": "independent FailFast prepass, frozen before labels",
        "label_policy": "STOP if DeltaJ>1ms; CONTINUE if DeltaJ<-1ms; else TIE",
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nCAPACITY PROBE SUMMARY", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
