import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from run_greedy_losslessness_audit import compare_token_traces, summarize_audit
from run_otrc_v2_td_benchmark import PROBLEM_IDS, command_for


ROOT = Path(__file__).resolve().parent
PROBLEM_ID = 31
SHARED_HISTORY_IDS = PROBLEM_IDS["humaneval"][:11]
SHARED_METHOD = (
    "otrc_v2_2_compact_factual_no_bootstrap_shared_value_advantage"
)
AUDIT_REQUIRED_FILES = (
    "benchmark_results.csv",
    "greedy_consistency_audit.csv",
    "output_token_trace.csv",
    "verifier_calls.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce and diagnose the HumanEval problem 31 greedy-output "
            "mismatch between FailFast-8 and Shared Value + Advantage."
        )
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_humaneval31_losslessness_audit"
        ),
    )
    parser.add_argument("--near_tie_margin", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.near_tie_margin < 0.0:
        raise ValueError("--near_tie_margin must be non-negative")
    if SHARED_HISTORY_IDS[-1] != PROBLEM_ID:
        raise RuntimeError("Shared history no longer ends at HumanEval problem 31")


def run_streaming(command):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
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


def shared_args(args):
    return SimpleNamespace(
        warmup_questions=1,
        max_new_tokens=args.max_new_tokens,
        spec_len=8,
        block_size=32,
        small_block_size=8,
        target_model_name=args.target_model_name,
        dllm_dir=args.dllm_dir,
        drafter_threshold=0.05,
        lowconf_threshold=0.45,
        max_spec_len=60,
        incr_len=8,
        feature_schema="otrc_v2_2_compact_td",
        credit_assignment="verifier_boundary_factual_no_bootstrap",
        adaptive_learning_rate=0.02,
        value_parameterization="shared_value_advantage",
        shared_value_learning_rate=0.015,
        shared_advantage_learning_rate=0.02,
        adaptive_mc_learning_rate=0.01,
        adaptive_mc_mix=0.5,
        adaptive_update_mode="mixed",
        adaptive_rho_alpha=0.05,
        rho_warmup_boundaries=0,
        policy_weight_ema_beta=0.0,
        policy_weight_ema_mode="global_step",
        adaptive_factual_ema_alpha=0.2,
        adaptive_risk_beta=1.0,
        adaptive_stop_probability_threshold=0.75,
        adaptive_uncertainty_prior=1.0,
        adaptive_epistemic_scale=0.1,
        adaptive_q_margin=0.0,
        adaptive_explore_epsilon=0.10,
        adaptive_explore_min=0.01,
        adaptive_explore_decay=0.998,
        adaptive_warmup_rounds=20,
        adaptive_early_stop_min_observations=32,
        adaptive_min_action_probability=0.10,
        adaptive_max_importance_weight=5.0,
        adaptive_weight_snapshot_interval=100,
        seed=42,
        log_level=args.log_level,
    )


def failfast_command(args, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", "humaneval",
        "--num_questions", "1",
        "--problem_ids", str(PROBLEM_ID),
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--drafter_thresholds", "0.05",
        "--sweep_lowconf_threshold", "0.45",
        "--sweep_max_spec_len", "60",
        "--sweep_incr_len", "8",
        "--seed", "42",
        "--audit_greedy_consistency",
        "--audit_greedy_problem_ids", str(PROBLEM_ID),
        "--log_verifier_calls",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


def shared_command(args, output_dir):
    command = command_for(
        shared_args(args),
        "humaneval",
        SHARED_HISTORY_IDS,
        output_dir,
    )
    command.extend([
        "--audit_greedy_consistency",
        "--audit_greedy_problem_ids", str(PROBLEM_ID),
        "--log_verifier_calls",
    ])
    return command


def phase_complete(directory, expected_problem_ids, adaptive):
    required = [directory / name for name in AUDIT_REQUIRED_FILES]
    if adaptive:
        required.extend([
            directory / "adaptive_td_decisions.csv",
            directory / "adaptive_td_runtime_state.json",
        ])
    if not all(path.exists() and path.stat().st_size for path in required):
        return False
    try:
        results = pd.read_csv(directory / "benchmark_results.csv")
        audit = pd.read_csv(directory / "greedy_consistency_audit.csv")
        trace = pd.read_csv(directory / "output_token_trace.csv")
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return (
        set(results["problem_id"].astype(int)) == set(expected_problem_ids)
        and set(audit["problem_id"].astype(int)) == {PROBLEM_ID}
        and set(trace["problem_id"].astype(int)) == {PROBLEM_ID}
    )


def run_phase(args, name, problem_ids, command_builder, adaptive):
    directory = Path(args.output_dir) / "raw" / name
    if args.resume and phase_complete(directory, problem_ids, adaptive):
        print(f"RESUME {name}", flush=True)
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100, flush=True)
    print(
        f"RUN {name} | HumanEval IDs={list(problem_ids)} | "
        f"audit_problem={PROBLEM_ID}",
        flush=True,
    )
    print("=" * 100, flush=True)
    run_streaming(command_builder(args, directory))
    if not phase_complete(directory, problem_ids, adaptive):
        raise RuntimeError(f"{name} did not produce a complete audit")
    return directory


def audit_row_at(audit, position):
    if position is None or (isinstance(position, float) and math.isnan(position)):
        return None
    rows = audit[
        pd.to_numeric(
            audit["absolute_output_position"], errors="coerce"
        ).eq(int(position))
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def numeric(row, key):
    if row is None:
        return None
    value = row.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def classify_divergence(comparison_row, audit_rows, near_tie_margin):
    position = comparison_row.get("first_different_position")
    if position is None or pd.isna(position):
        return {
            "classification": "lossless_match",
            "severity": "none",
            "explanation": "Both methods emitted the same token sequence.",
        }

    available = [row for row in audit_rows.values() if row is not None]
    if not available:
        return {
            "classification": "missing_audit_at_divergence",
            "severity": "invalid_audit",
            "explanation": "No exact-prefix audit row exists at the first divergence.",
        }

    if any(numeric(row, "emitted_matches_batched") == 0 for row in available):
        return {
            "classification": "commit_or_indexing_mismatch",
            "severity": "implementation_bug",
            "explanation": (
                "An emitted token is not the verifier batched argmax selected "
                "for that position. Inspect correction/bonus indexing and EOS truncation."
            ),
        }

    if any(numeric(row, "batched_matches_prefix") == 0 for row in available):
        margins = [
            value
            for row in available
            for value in (
                numeric(row, "batched_margin"),
                numeric(row, "prefix_margin"),
            )
            if value is not None
        ]
        near_tie = bool(margins) and min(margins) <= near_tie_margin
        return {
            "classification": (
                "near_tie_batched_prefix_mismatch"
                if near_tie
                else "large_margin_batched_prefix_mismatch"
            ),
            "severity": "numerical_instability" if near_tie else "implementation_bug",
            "explanation": (
                "The verifier batched argmax differs from exact-prefix greedy "
                "decoding at the first divergent position."
            ),
        }

    if any(numeric(row, "emitted_is_eos") == 1 for row in available):
        return {
            "classification": "eos_boundary_mismatch",
            "severity": "implementation_bug",
            "explanation": "The first divergence occurs at an EOS boundary.",
        }

    prefix_tokens = {
        int(value)
        for value in (
            numeric(row, "prefix_argmax_token") for row in available
        )
        if value is not None
    }
    if len(prefix_tokens) > 1:
        margins = [
            numeric(row, "prefix_margin") for row in available
        ]
        margins = [value for value in margins if value is not None]
        near_tie = bool(margins) and min(margins) <= near_tie_margin
        return {
            "classification": (
                "near_tie_exact_prefix_nondeterminism"
                if near_tie
                else "exact_prefix_nondeterminism"
            ),
            "severity": "numerical_instability",
            "explanation": (
                "The common prefix produced different exact-prefix verifier "
                "argmax tokens in the two process runs."
            ),
        }

    return {
        "classification": "cross_path_divergence_unclassified",
        "severity": "requires_trace_review",
        "explanation": (
            "Both methods are internally consistent at the recorded token, "
            "but the available audit fields do not explain the divergence."
        ),
    }


def trace_context(trace, position, radius=8):
    if position is None or pd.isna(position):
        return trace.iloc[0:0].copy()
    position = int(position)
    numeric_positions = pd.to_numeric(trace["output_position"], errors="coerce")
    return trace[
        numeric_positions.between(position - radius, position + radius)
    ].copy()


def rows_at_round(path, round_id):
    if not path.exists() or round_id is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "problem_id" in frame:
        frame = frame[pd.to_numeric(frame["problem_id"], errors="coerce").eq(PROBLEM_ID)]
    if "round_id" in frame:
        frame = frame[pd.to_numeric(frame["round_id"], errors="coerce").eq(round_id)]
    return frame


def analyze(output_dir, phase_dirs, near_tie_margin):
    traces = {}
    audits = {}
    summaries = []
    benchmark_rows = []
    for name, directory in phase_dirs.items():
        trace = pd.read_csv(directory / "output_token_trace.csv")
        audit = pd.read_csv(directory / "greedy_consistency_audit.csv")
        benchmark = pd.read_csv(directory / "benchmark_results.csv")
        trace = trace[pd.to_numeric(trace["problem_id"], errors="coerce").eq(PROBLEM_ID)]
        audit = audit[pd.to_numeric(audit["problem_id"], errors="coerce").eq(PROBLEM_ID)]
        benchmark = benchmark[
            pd.to_numeric(benchmark["problem_id"], errors="coerce").eq(PROBLEM_ID)
        ].copy()
        if len(benchmark) != 1:
            raise RuntimeError(f"{name} has {len(benchmark)} benchmark rows for problem 31")
        traces[name] = trace
        audits[name] = audit
        summary = summarize_audit(name, audit)
        summary.update({
            "output_tokens": int(benchmark.iloc[0]["output_tokens"]),
            "output_token_hash": benchmark.iloc[0]["output_token_hash"],
            "verifier_rounds": int(benchmark.iloc[0]["num_speculation_rounds"]),
            "draft_forward_passes": int(
                benchmark.iloc[0]["total_num_forward_passes"]
            ),
        })
        summaries.append(summary)
        benchmark.insert(0, "audit_method", name)
        benchmark_rows.append(benchmark)

    comparison = compare_token_traces(traces["failfast"], traces["shared_value_advantage"])
    comparison = comparison.rename(columns={"method_a_token": "shared_value_advantage_token"})
    comparison_row = comparison.iloc[0].to_dict()
    position = comparison_row["first_different_position"]
    divergence_audits = {
        name: audit_row_at(audit, position) for name, audit in audits.items()
    }
    diagnosis = classify_divergence(
        comparison_row,
        divergence_audits,
        near_tie_margin,
    )

    detail_rows = []
    for name, row in divergence_audits.items():
        if row is not None:
            row = dict(row)
            row["audit_method"] = name
            detail_rows.append(row)
    details = pd.DataFrame(detail_rows)

    pd.DataFrame(summaries).to_csv(
        output_dir / "greedy_consistency_summary.csv", index=False
    )
    comparison.to_csv(output_dir / "cross_method_first_difference.csv", index=False)
    details.to_csv(output_dir / "first_divergence_audit_rows.csv", index=False)
    pd.concat(benchmark_rows, ignore_index=True).to_csv(
        output_dir / "problem31_benchmark_comparison.csv", index=False
    )

    contexts = []
    for name, trace in traces.items():
        context = trace_context(trace, position)
        context.insert(0, "audit_method", name)
        contexts.append(context)
    pd.concat(contexts, ignore_index=True).to_csv(
        output_dir / "first_divergence_token_context.csv", index=False
    )

    round_ids = {
        name: int(row["round_id"])
        for name, row in divergence_audits.items()
        if row is not None and not pd.isna(row.get("round_id"))
    }
    verifier_contexts = []
    for name, directory in phase_dirs.items():
        frame = rows_at_round(directory / "verifier_calls.csv", round_ids.get(name))
        if not frame.empty:
            frame.insert(0, "audit_method", name)
            verifier_contexts.append(frame)
    if verifier_contexts:
        pd.concat(verifier_contexts, ignore_index=True).to_csv(
            output_dir / "verifier_calls_at_divergence.csv", index=False
        )

    shared_round = round_ids.get("shared_value_advantage")
    shared_decisions = rows_at_round(
        phase_dirs["shared_value_advantage"] / "adaptive_td_decisions.csv",
        shared_round,
    )
    if not shared_decisions.empty:
        shared_decisions.to_csv(
            output_dir / "shared_decisions_at_divergence.csv", index=False
        )

    report = {
        "problem_id": PROBLEM_ID,
        "shared_history_problem_ids": SHARED_HISTORY_IDS,
        "near_tie_margin": near_tie_margin,
        "first_difference": comparison_row,
        "diagnosis": diagnosis,
        "divergent_round_ids": round_ids,
        "interpretation_order": [
            "emitted_matches_batched tests commit/correction/bonus indexing",
            "batched_matches_prefix tests batched verifier causality",
            "prefix margins distinguish near ties from large-margin errors",
            "EOS flags test termination-boundary handling",
        ],
    }
    (output_dir / "diagnosis.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return pd.DataFrame(summaries), comparison, diagnosis


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_dirs = {
        "failfast": run_phase(
            args,
            "failfast",
            [PROBLEM_ID],
            failfast_command,
            adaptive=False,
        ),
        "shared_value_advantage": run_phase(
            args,
            "shared_value_advantage",
            SHARED_HISTORY_IDS,
            shared_command,
            adaptive=True,
        ),
    }
    summary, comparison, diagnosis = analyze(
        output_dir,
        phase_dirs,
        args.near_tie_margin,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "dataset": "humaneval",
            "problem_id": PROBLEM_ID,
            "shared_history_problem_ids": SHARED_HISTORY_IDS,
            "seed": 42,
            "max_new_tokens": args.max_new_tokens,
            "target_model_name": args.target_model_name,
            "shared_method": SHARED_METHOD,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nGREEDY CONSISTENCY SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nFIRST CROSS-METHOD DIFFERENCE", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("\nDIAGNOSIS", flush=True)
    print(json.dumps(diagnosis, indent=2), flush=True)

    if not args.skip_archive:
        archive = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir,
        )
        print(f"\nArchive: {archive}", flush=True)
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
