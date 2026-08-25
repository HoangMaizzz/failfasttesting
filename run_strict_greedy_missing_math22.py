import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_strict_greedy_math50 import common_command, load_reference, run_streaming


ROOT = Path(__file__).resolve().parent
ORACLE_REFERENCE_DIR = (
    ROOT / "benchmark_references" / "math_strict_greedy_oracle_test50"
)
MISSING_PROBLEM_IDS = (
    6, 51, 57, 108, 115, 123, 129, 161, 164, 193, 204,
    216, 281, 301, 308, 375, 394, 398, 402, 413, 419, 441,
)
VERSION = "strict_greedy_oracle_missing_math22_replay_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_strict_greedy_missing_math22",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def replace_option(command, option, value):
    index = command.index(option)
    command[index + 1] = str(value)


def validate_replay(results, decisions):
    expected_ids = set(MISSING_PROBLEM_IDS)
    actual_ids = set(results["problem_id"].astype(int))
    if actual_ids != expected_ids or len(results) != len(expected_ids):
        raise RuntimeError(
            f"oracle replay IDs differ: expected={sorted(expected_ids)}, "
            f"actual={sorted(actual_ids)}"
        )
    required = {
        "sample_id",
        "context_len",
        "draft_proposal",
        "state_key",
        "chosen_action",
        "accumulated_proposal_length",
        "refinement_step",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise RuntimeError(f"decision log is missing columns: {sorted(missing)}")
    if decisions.empty or decisions["state_key"].isna().any():
        raise RuntimeError("oracle replay did not save complete decision states")
    duplicates = decisions.duplicated(
        [
            "sample_id",
            "context_len",
            "accumulated_proposal_length",
            "refinement_step",
        ]
    )
    if duplicates.any():
        raise RuntimeError(
            f"oracle replay produced {int(duplicates.sum())} duplicate states"
        )

    reference = pd.read_csv(
        ORACLE_REFERENCE_DIR / "oracle_replay_reference.csv"
    )
    reference = reference[reference["problem_id"].isin(expected_ids)].copy()
    paired = results.merge(
        reference[
            [
                "problem_id",
                "output_token_hash",
                "output_tokens",
                "num_speculation_rounds",
                "total_num_forward_passes",
            ]
        ],
        on="problem_id",
        suffixes=("_replay", "_reference"),
        validate="one_to_one",
    )
    checks = {
        "output_hash_match": (
            paired["output_token_hash_replay"]
            == paired["output_token_hash_reference"]
        ),
        "output_tokens_match": (
            paired["output_tokens_replay"]
            == paired["output_tokens_reference"]
        ),
        "verifier_rounds_match": (
            paired["num_speculation_rounds_replay"]
            == paired["num_speculation_rounds_reference"]
        ),
        "draft_forwards_match": (
            paired["total_num_forward_passes_replay"]
            == paired["total_num_forward_passes_reference"]
        ),
    }
    for name, values in checks.items():
        paired[name] = values
        if not bool(values.all()):
            failed = paired.loc[~values, "problem_id"].astype(int).tolist()
            raise RuntimeError(f"{name} failed for problem IDs: {failed}")
    return paired


def main():
    args = parse_args()
    source_manifest, _, source_ids = load_reference()
    if not set(MISSING_PROBLEM_IDS).issubset(source_ids):
        raise RuntimeError("missing-22 IDs are not contained in MATH-50 reference")

    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw" / "oracle_replay_missing22"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir.mkdir(parents=True)

    policy_path = ORACLE_REFERENCE_DIR / "strict_greedy_policy.json"
    profile_path = ORACLE_REFERENCE_DIR / "verifier_profile.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    missing_policy_ids = set(MISSING_PROBLEM_IDS).difference(
        int(value) for value in policy.get("policies", {})
    )
    if missing_policy_ids:
        raise RuntimeError(
            f"bundled oracle policy misses IDs: {sorted(missing_policy_ids)}"
        )

    command = common_command(
        source_manifest["arguments"],
        MISSING_PROBLEM_IDS,
        args.dllm_dir,
        raw_dir,
        args.log_level,
    )
    replace_option(command, "--warmup_questions", 0)
    command.extend([
        "--strict_greedy_local_oracle",
        "--strict_greedy_verifier_profile", str(profile_path),
        "--strict_greedy_epsilon_ms", "1.0",
        "--strict_greedy_replay_policy", str(policy_path),
    ])

    started = time.time()
    print("=" * 100, flush=True)
    print(
        "REPLAY latest strict oracle policy for 22 MATH questions; "
        "counterfactual search and FailFast baseline are skipped",
        flush=True,
    )
    print("=" * 100, flush=True)
    run_streaming(command)

    results = pd.read_csv(raw_dir / "benchmark_results.csv")
    calls = pd.read_csv(raw_dir / "verifier_calls.csv")
    decisions = pd.read_csv(raw_dir / "greedy_local_oracle_decisions.csv")
    paired = validate_replay(results, decisions)

    results.to_csv(output_dir / "oracle_missing22_results.csv", index=False)
    calls.to_csv(output_dir / "oracle_missing22_verifier_calls.csv", index=False)
    decisions.to_csv(output_dir / "oracle_missing22_decisions.csv", index=False)
    paired.to_csv(output_dir / "oracle_missing22_replay_checks.csv", index=False)

    report = pd.DataFrame([{
        "num_questions": len(results),
        "num_decisions": len(decisions),
        "output_hash_match_percent": 100.0,
        "round_match_percent": 100.0,
        "forward_match_percent": 100.0,
        "algorithm_time_s": float(results["actual_algorithm_time"].sum()),
        "elapsed_runner_minutes": (time.time() - started) / 60.0,
    }])
    report.to_csv(output_dir / "oracle_missing22_summary.csv", index=False)

    manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "problem_ids": list(MISSING_PROBLEM_IDS),
        "source_policy": str(policy_path),
        "search_reexecuted": False,
        "baseline_reexecuted": False,
        "state_columns": [
            "context_len",
            "draft_proposal",
            "state_key",
            "accumulated_proposal_length",
            "refinement_step",
        ],
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        manifest["git_commit"] = None
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print("\nSTRICT ORACLE MISSING-22 REPLAY SUMMARY")
    print(report.to_string(index=False))
    print(f"\nSaved: {output_dir}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
