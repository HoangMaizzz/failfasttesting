import argparse
import json
import platform
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_failfast_counterfactual_oracle import DATASET_SIZES
from run_gsm8k_balanced_oracle_dataset import (
    PASS_CLASSES,
    causal_oracle_comparison,
    causal_selection_rounds,
    causal_stop_distribution,
    paired_causal_comparison,
    problem_profiles,
    round_distribution_summary,
    run_causal_oracle,
    run_failfast_pool,
)


VERSION = "gsm8k_round_balanced_question_benchmark_v1"
COUNT_COLUMNS = [f"oracle_{label}_rounds" for label in PASS_CLASSES]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--selected_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--selection_restarts", type=int, default=32)
    parser.add_argument("--selection_max_iterations", type=int, default=200)
    parser.add_argument("--balance_strength", type=float, default=0.25)
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
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--selection_seed", type=int, default=2027)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_gsm8k_round_balanced_questions100"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    available = DATASET_SIZES["gsm8k"] - args.warmup_questions
    if args.pool_size <= 0 or args.pool_size > available:
        raise ValueError(f"--pool_size must be in [1, {available}]")
    if args.selected_size <= 0 or args.selected_size > args.pool_size:
        raise ValueError("--selected_size must be in [1, pool_size]")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.selection_restarts <= 0 or args.selection_max_iterations <= 0:
        raise ValueError("selection search limits must be positive")
    if not 0.0 <= args.balance_strength <= 1.0:
        raise ValueError("--balance_strength must be in [0, 1]")
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("proposal lengths must be positive")


def nested_pool_problem_ids(args):
    population = list(
        range(args.warmup_questions, DATASET_SIZES["gsm8k"])
    )
    random.Random(args.sample_seed).shuffle(population)
    return population[:args.pool_size]


def round_balance_score(counts, target_shares=None):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return float("inf")
    shares = counts / total
    if target_shares is None:
        target_shares = np.full(len(PASS_CLASSES), 1.0 / len(PASS_CLASSES))
    target_shares = np.asarray(target_shares, dtype=np.float64)
    deviations = np.abs(shares - target_shares)
    return float(deviations.max() + 0.1 * np.sqrt(np.mean(deviations ** 2)))


def round_balance_metrics(counts, target_shares=None):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    shares = counts / max(1.0, total)
    mean = float(counts.mean())
    metrics = {
        "total_rounds": int(total),
        "step1_rounds": int(counts[0]),
        "step2_rounds": int(counts[1]),
        "step3plus_rounds": int(counts[2]),
        "step1_percent": 100.0 * float(shares[0]),
        "step2_percent": 100.0 * float(shares[1]),
        "step3plus_percent": 100.0 * float(shares[2]),
        "round_count_cv_percent": (
            100.0 * float(counts.std()) / mean if mean > 0 else float("inf")
        ),
        "max_to_min_round_ratio": (
            float(counts.max() / counts.min())
            if counts.min() > 0
            else float("inf")
        ),
        "selection_objective": round_balance_score(counts, target_shares),
    }
    if target_shares is not None:
        target_shares = np.asarray(target_shares, dtype=np.float64)
        for index, label in enumerate(PASS_CLASSES):
            metrics[f"target_{label}_percent"] = (
                100.0 * float(target_shares[index])
            )
    return metrics


def select_round_balanced_questions(
    profiles,
    selected_size,
    selection_seed,
    restarts=32,
    max_iterations=200,
    balance_strength=0.25,
):
    ordered = profiles.sort_values("problem_id").reset_index(drop=True).copy()
    if len(ordered) < selected_size:
        raise ValueError("The oracle pool contains fewer questions than requested")
    vectors = ordered[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    pool_counts = vectors.sum(axis=0)
    natural_shares = pool_counts / max(1.0, float(pool_counts.sum()))
    uniform_shares = np.full(len(PASS_CLASSES), 1.0 / len(PASS_CLASSES))
    target_shares = (
        (1.0 - balance_strength) * natural_shares
        + balance_strength * uniform_shares
    )
    rng = np.random.default_rng(selection_seed)
    all_indices = np.arange(len(ordered), dtype=np.int64)

    rare_share = vectors[:, 2] / np.maximum(1.0, vectors.sum(axis=1))
    initial_sets = [
        np.argsort(-rare_share, kind="stable")[:selected_size]
    ]
    for _ in range(max(0, restarts - 1)):
        initial_sets.append(
            rng.choice(all_indices, size=selected_size, replace=False)
        )

    random_reference = np.sort(initial_sets[1] if len(initial_sets) > 1 else initial_sets[0])
    best_indices = None
    best_score = float("inf")
    best_counts = None

    for initial in initial_sets:
        selected = np.zeros(len(ordered), dtype=bool)
        selected[np.asarray(initial, dtype=np.int64)] = True
        counts = vectors[selected].sum(axis=0)
        score = round_balance_score(counts, target_shares)

        for _ in range(max_iterations):
            selected_indices = all_indices[selected]
            unselected_indices = all_indices[~selected]
            candidate_counts = (
                counts[None, None, :]
                - vectors[selected_indices][:, None, :]
                + vectors[unselected_indices][None, :, :]
            )
            totals = candidate_counts.sum(axis=2)
            shares = candidate_counts / np.maximum(1.0, totals[:, :, None])
            deviations = np.abs(shares - target_shares[None, None, :])
            scores = deviations.max(axis=2) + 0.1 * np.sqrt(
                np.mean(deviations ** 2, axis=2)
            )
            flat_index = int(np.argmin(scores))
            candidate_score = float(scores.flat[flat_index])
            if candidate_score >= score - 1e-12:
                break
            remove_offset, add_offset = np.unravel_index(
                flat_index, scores.shape
            )
            remove_index = selected_indices[remove_offset]
            add_index = unselected_indices[add_offset]
            selected[remove_index] = False
            selected[add_index] = True
            counts = candidate_counts[remove_offset, add_offset]
            score = candidate_score

        if score < best_score - 1e-12:
            best_score = score
            best_indices = all_indices[selected].copy()
            best_counts = counts.copy()

    if best_indices is None or best_counts is None:
        raise RuntimeError("Round-balance subset search did not produce a result")

    selected_profiles = ordered.iloc[np.sort(best_indices)].copy()
    selected_profiles["selection_objective"] = best_score
    random_counts = vectors[random_reference].sum(axis=0)
    diagnostics = {
        "pool_distribution": round_balance_metrics(
            pool_counts, natural_shares
        ),
        "target_distribution_percent": {
            label: 100.0 * float(target_shares[index])
            for index, label in enumerate(PASS_CLASSES)
        },
        "balance_strength": float(balance_strength),
        "optimized": round_balance_metrics(best_counts, target_shares),
        "random_reference": round_balance_metrics(
            random_counts, target_shares
        ),
        "num_pool_questions": int(len(ordered)),
        "num_selected_questions": int(selected_size),
        "selection_seed": int(selection_seed),
        "selection_restarts": int(restarts),
        "selection_max_iterations": int(max_iterations),
    }
    return selected_profiles.reset_index(drop=True), diagnostics


def make_compact_report_archive(output_dir):
    archive_path = output_dir.with_name(f"{output_dir.name}_report.zip")
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{output_dir.name}/{path.name}")
    return archive_path


def benchmark_manifest(args, candidate_ids, evaluated_ids, selected_ids, diagnostics):
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    return {
        "version": VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "candidate_problem_ids": list(map(int, candidate_ids)),
        "evaluated_problem_ids": list(map(int, evaluated_ids)),
        "selected_problem_ids": list(map(int, selected_ids)),
        "selection_diagnostics": diagnostics,
        "selection_definition": (
            "Each question contributes all of its causal-oracle speculation rounds. "
            "A fixed-size subset of unique problem_ids is selected toward a soft "
            "target interpolated between the candidate pool's natural round "
            "distribution and a uniform step1/step2/step3plus distribution. No "
            "question is assigned a single class and no round is discarded after "
            "question selection."
        ),
        "timing_definition": (
            "FailFast and the causal oracle are compared on the same selected questions "
            "using actual_algorithm_time = draft + verify + post-verify algorithm time. "
            "Counterfactual probe time used only to reveal oracle choices is excluded "
            "from deployable oracle latency and reported separately."
        ),
    }


def main():
    args = parse_args()
    validate_args(args)
    args.dataset = "gsm8k"
    args.num_questions = args.pool_size
    args.adaptive_state_path = None
    args.oracle_only = True
    args.collect_until_balanced = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = nested_pool_problem_ids(args)
    causal_results, causal_decisions, causal_candidates = run_causal_oracle(
        args,
        candidate_ids,
    )
    evaluated_ids = sorted(causal_results["problem_id"].astype(int).unique())
    rounds = causal_selection_rounds(causal_decisions)
    profiles = problem_profiles(rounds)
    selected_profiles, diagnostics = select_round_balanced_questions(
        profiles,
        args.selected_size,
        args.selection_seed,
        args.selection_restarts,
        args.selection_max_iterations,
        args.balance_strength,
    )
    selected_ids = sorted(selected_profiles["problem_id"].astype(int).tolist())
    selected_id_set = set(selected_ids)

    failfast_results = run_failfast_pool(args, selected_ids)
    selected_causal_results = causal_results[
        causal_results["problem_id"].astype(int).isin(selected_id_set)
    ].copy()
    selected_decisions = causal_decisions[
        causal_decisions["problem_id"].astype(int).isin(selected_id_set)
    ].copy()
    selected_candidates = causal_candidates[
        causal_candidates["problem_id"].astype(int).isin(selected_id_set)
    ].copy()
    selected_rounds = rounds[
        rounds["problem_id"].astype(int).isin(selected_id_set)
    ].copy()

    if len(failfast_results) != args.selected_size:
        raise RuntimeError("FailFast did not finish every selected question")
    if len(selected_causal_results) != args.selected_size:
        raise RuntimeError("Causal oracle did not finish every selected question")

    causal_results.to_csv(
        output_dir / "pool_causal_oracle_results.csv", index=False
    )
    causal_decisions.to_csv(
        output_dir / "pool_causal_oracle_decisions.csv", index=False
    )
    causal_candidates.to_csv(
        output_dir / "pool_causal_oracle_candidates.csv", index=False
    )
    rounds.to_csv(output_dir / "pool_causal_oracle_rounds.csv", index=False)
    profiles.to_csv(output_dir / "pool_problem_profiles.csv", index=False)
    failfast_results.to_csv(
        output_dir / "selected_failfast_results.csv", index=False
    )
    selected_causal_results.to_csv(
        output_dir / "selected_causal_oracle_results.csv", index=False
    )
    selected_decisions.to_csv(
        output_dir / "selected_causal_oracle_decisions.csv", index=False
    )
    selected_candidates.to_csv(
        output_dir / "selected_causal_oracle_candidates.csv", index=False
    )
    selected_rounds.to_csv(
        output_dir / "selected_all_rounds.csv", index=False
    )
    selected_profiles.to_csv(
        output_dir / "selected_problem_profiles.csv", index=False
    )
    round_distribution_summary(rounds, "candidate_pool").to_csv(
        output_dir / "pool_round_distribution.csv", index=False
    )
    selected_distribution = round_distribution_summary(
        selected_rounds,
        "selected_100_questions_all_rounds",
    )
    selected_distribution.to_csv(
        output_dir / "selected_round_distribution.csv", index=False
    )
    causal_stop_distribution(selected_decisions).to_csv(
        output_dir / "selected_oracle_stop_distribution.csv", index=False
    )

    summary = causal_oracle_comparison(
        failfast_results,
        selected_causal_results,
        selected_decisions,
    )
    for key, value in diagnostics["optimized"].items():
        summary[f"selected_{key}"] = value
    summary.to_csv(output_dir / "oracle_vs_failfast_summary.csv", index=False)
    paired_causal_comparison(
        failfast_results,
        selected_causal_results,
    ).to_csv(output_dir / "oracle_vs_failfast_paired_results.csv", index=False)

    (output_dir / "selected_problem_ids.json").write_text(
        json.dumps(selected_ids, indent=2), encoding="utf-8"
    )
    (output_dir / "all_evaluated_problem_ids.json").write_text(
        json.dumps(evaluated_ids, indent=2), encoding="utf-8"
    )
    (output_dir / "selection_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(
            benchmark_manifest(
                args,
                candidate_ids,
                evaluated_ids,
                selected_ids,
                diagnostics,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    archive_path = make_compact_report_archive(output_dir)
    print("\nROUND BALANCE: RANDOM REFERENCE")
    print(json.dumps(diagnostics["random_reference"], indent=2))
    print("\nROUND BALANCE: OPTIMIZED 100-QUESTION SUBSET")
    print(json.dumps(diagnostics["optimized"], indent=2))
    print("\nSELECTED ROUND DISTRIBUTION")
    print(selected_distribution.to_string(index=False))
    print("\nCAUSAL ORACLE VS FAILFAST")
    print(summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved compact archive: {archive_path}")


if __name__ == "__main__":
    main()
