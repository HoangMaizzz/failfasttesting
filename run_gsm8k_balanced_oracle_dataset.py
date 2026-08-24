import argparse
import itertools
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_failfast_counterfactual_oracle import (
    DATASET_SIZES,
    run_phase,
)


VERSION = "gsm8k_balanced_failfast_causal_oracle_v6"
PASS_CLASSES = ("step1", "step2", "step3plus")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--selected_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
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
            "outputs_gsm8k_causal_oracle_pool300_test50"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--collect_until_balanced", action="store_true")
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
    if args.spec_len <= 0 or args.incr_len <= 0:
        raise ValueError("proposal lengths must be positive")


def balanced_quotas(total):
    base, remainder = divmod(total, len(PASS_CLASSES))
    return {
        label: base + int(index < remainder)
        for index, label in enumerate(PASS_CLASSES)
    }


def pass_class(value):
    value = float(value)
    if value <= 1.0:
        return "step1"
    if value <= 2.0:
        return "step2"
    return "step3plus"


def pool_problem_ids(args):
    population = list(range(args.warmup_questions, DATASET_SIZES["gsm8k"]))
    sampled = random.Random(args.sample_seed).sample(population, args.pool_size)
    return sampled if args.collect_until_balanced else sorted(sampled)


def partition_problem_ids(problem_ids, batch_size):
    return [
        problem_ids[start:start + batch_size]
        for start in range(0, len(problem_ids), batch_size)
    ]


def completed_failfast_batch(batch_dir, expected_problem_ids):
    results_path = batch_dir / "benchmark_results.csv"
    if not results_path.exists():
        return False
    try:
        results = pd.read_csv(results_path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    expected = set(map(int, expected_problem_ids))
    result_ids = set(pd.to_numeric(results["problem_id"], errors="coerce").dropna().astype(int))
    return len(results) == len(expected) and result_ids == expected


def run_failfast_pool(args, pool_ids):
    result_frames = []
    batches = partition_problem_ids(pool_ids, args.batch_size)
    for batch_index, batch_ids in enumerate(batches, start=1):
        phase = f"failfast_pool_batch_{batch_index:03d}"
        batch_dir = Path(args.output_dir) / "raw" / phase
        if args.resume and completed_failfast_batch(batch_dir, batch_ids):
            print(
                f"RESUME {phase} | batch={batch_index}/{len(batches)} | "
                f"samples={len(batch_ids)}",
                flush=True,
            )
        else:
            results_path = batch_dir / "benchmark_results.csv"
            if results_path.exists():
                results_path.unlink()
            run_phase(
                args,
                batch_ids,
                phase,
                [],
                ["benchmark_results.csv"],
            )
            if not completed_failfast_batch(batch_dir, batch_ids):
                raise RuntimeError(f"Incomplete FailFast batch: {phase}")
        result_frames.append(pd.read_csv(batch_dir / "benchmark_results.csv"))
    return pd.concat(result_frames, ignore_index=True)


def completed_causal_batch(batch_dir, expected_problem_ids):
    results_path = batch_dir / "benchmark_results.csv"
    decisions_path = batch_dir / "causal_oracle_decisions.csv"
    candidates_path = batch_dir / "causal_oracle_candidates.csv"
    if not all(path.exists() for path in (results_path, decisions_path, candidates_path)):
        return False
    try:
        results = pd.read_csv(results_path)
        decisions = pd.read_csv(decisions_path, usecols=["problem_id"])
        candidates = pd.read_csv(candidates_path, usecols=["problem_id"])
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    expected = set(map(int, expected_problem_ids))
    result_ids = set(pd.to_numeric(results["problem_id"], errors="coerce").dropna().astype(int))
    decision_ids = set(
        pd.to_numeric(decisions["problem_id"], errors="coerce").dropna().astype(int)
    )
    candidate_ids = set(
        pd.to_numeric(candidates["problem_id"], errors="coerce").dropna().astype(int)
    )
    return (
        len(results) == len(expected)
        and result_ids == expected
        and decision_ids == expected
        and candidate_ids == expected
    )


def run_causal_oracle(args, problem_ids):
    result_frames = []
    decision_frames = []
    candidate_frames = []
    batches = partition_problem_ids(problem_ids, args.batch_size)
    required_files = (
        "benchmark_results.csv",
        "causal_oracle_decisions.csv",
        "causal_oracle_candidates.csv",
    )
    for batch_index, batch_ids in enumerate(batches, start=1):
        phase = f"causal_oracle_pool_batch_{batch_index:03d}"
        batch_dir = Path(args.output_dir) / "raw" / phase
        if args.resume and completed_causal_batch(batch_dir, batch_ids):
            print(
                f"RESUME {phase} | batch={batch_index}/{len(batches)} | "
                f"samples={len(batch_ids)}",
                flush=True,
            )
        else:
            for filename in required_files:
                path = batch_dir / filename
                if path.exists():
                    path.unlink()
            run_phase(
                args,
                batch_ids,
                phase,
                ["--collect_bucket_oracle", "--causal_oracle"],
                required_files,
            )
            if not completed_causal_batch(batch_dir, batch_ids):
                raise RuntimeError(f"Incomplete causal oracle batch: {phase}")
        result_frames.append(pd.read_csv(batch_dir / "benchmark_results.csv"))
        decision_frames.append(pd.read_csv(batch_dir / "causal_oracle_decisions.csv"))
        candidate_frames.append(pd.read_csv(batch_dir / "causal_oracle_candidates.csv"))
        if args.collect_until_balanced:
            partial_decisions = pd.concat(decision_frames, ignore_index=True)
            partial_rounds = causal_selection_rounds(partial_decisions)
            partial_profiles = problem_profiles(partial_rounds)
            eligible = {
                label: int(partial_profiles[f"oracle_{label}_rounds"].gt(0).sum())
                for label in PASS_CLASSES
            }
            try:
                select_balanced_problems(
                    partial_profiles,
                    partial_rounds,
                    args.selected_size,
                    args.selection_seed,
                )
            except ValueError:
                print(
                    f"COLLECTION PROGRESS | evaluated={len(partial_profiles)} | "
                    f"eligible={eligible}",
                    flush=True,
                )
            else:
                print(
                    f"BALANCED TARGET REACHED | evaluated={len(partial_profiles)} | "
                    f"eligible={eligible}",
                    flush=True,
                )
                break
    return (
        pd.concat(result_frames, ignore_index=True),
        pd.concat(decision_frames, ignore_index=True),
        pd.concat(candidate_frames, ignore_index=True),
    )


def causal_selection_rounds(decisions):
    rounds = decisions.copy()
    rounds["oracle_refinement_steps"] = pd.to_numeric(
        rounds["selected_step"], errors="raise"
    ).astype(int)
    rounds["oracle_draft_passes"] = pd.to_numeric(
        rounds["selected_draft_passes"], errors="raise"
    ).astype(int)
    rounds["factual_draft_passes"] = pd.to_numeric(
        rounds["physical_draft_passes"], errors="raise"
    ).astype(int)
    rounds["oracle_pass_class"] = rounds["oracle_refinement_steps"].map(pass_class)
    return rounds


def problem_profiles(rounds):
    profiled = rounds.copy()
    profiled["oracle_pass_class"] = profiled["oracle_refinement_steps"].map(
        pass_class
    )
    records = []
    for problem_id, group in profiled.groupby("problem_id", sort=True):
        counts = group["oracle_pass_class"].value_counts()
        median_steps = float(group["oracle_refinement_steps"].median())
        problem_class = pass_class(median_steps)
        record = {
            "problem_id": int(problem_id),
            "problem_oracle_class": problem_class,
            "num_rounds": int(len(group)),
            "median_oracle_refinement_steps": median_steps,
            "mean_oracle_refinement_steps": float(
                group["oracle_refinement_steps"].mean()
            ),
            "mean_oracle_draft_passes": float(
                group["oracle_draft_passes"].mean()
            ),
            "mean_factual_draft_passes": float(
                group["factual_draft_passes"].mean()
            ),
            "local_oracle_round_match_percent": float(
                100.0
                * (group["oracle_draft_passes"] == group["factual_draft_passes"]).mean()
            ),
        }
        for label in PASS_CLASSES:
            count = int(counts.get(label, 0))
            record[f"oracle_{label}_rounds"] = count
            record[f"oracle_{label}_round_percent"] = 100.0 * count / len(group)
        record["class_purity_percent"] = record[
            f"oracle_{problem_class}_round_percent"
        ]
        records.append(record)
    return pd.DataFrame(records)


def select_balanced_problems(profiles, rounds, selected_size, selection_seed):
    quotas = balanced_quotas(selected_size)
    eligible = {
        label: set(
            profiles.loc[
                profiles[f"oracle_{label}_rounds"].gt(0), "problem_id"
            ].astype(int)
        )
        for label in PASS_CLASSES
    }
    base_order = tuple(
        sorted(PASS_CLASSES, key=lambda label: len(eligible[label]) / quotas[label])
    )
    orders = [base_order] + [
        order for order in itertools.permutations(PASS_CLASSES) if order != base_order
    ]
    assignments = None
    for order in orders:
        rng = random.Random(f"{selection_seed}:{','.join(order)}")
        available_ids = set(profiles["problem_id"].astype(int))
        candidate_assignments = {}
        for label in order:
            candidates = sorted(eligible[label] & available_ids)
            rng.shuffle(candidates)
            chosen = candidates[:quotas[label]]
            if len(chosen) != quotas[label]:
                break
            candidate_assignments[label] = chosen
            available_ids.difference_update(chosen)
        if len(candidate_assignments) == len(PASS_CLASSES):
            assignments = candidate_assignments
            break
    if assignments is None:
        available = {label: len(eligible[label]) for label in PASS_CLASSES}
        raise ValueError(
            "The oracle pool cannot provide unique questions for all anchor-round "
            f"quotas. quotas={quotas}, eligible_questions={available}. Increase "
            "--pool_size or change --sample_seed."
        )

    assigned_class = {
        problem_id: label
        for label, problem_ids in assignments.items()
        for problem_id in problem_ids
    }
    selected_ids = sorted(assigned_class)
    selected = profiles[profiles["problem_id"].isin(selected_ids)].copy()
    selected["selection_stratum"] = selected["problem_id"].map(assigned_class)
    selected["selection_stratum_round_percent"] = selected.apply(
        lambda row: row[f"oracle_{row['selection_stratum']}_round_percent"],
        axis=1,
    )
    selected["selection_seed"] = int(selection_seed)
    selected["class_quota"] = selected["selection_stratum"].map(quotas)

    anchor_rng = random.Random(f"{selection_seed}:anchor-rounds")
    anchor_records = []
    for row in selected.itertuples(index=False):
        candidates = rounds[
            rounds["problem_id"].eq(row.problem_id)
            & rounds["oracle_pass_class"].eq(row.selection_stratum)
        ]
        anchor = candidates.iloc[anchor_rng.randrange(len(candidates))].copy()
        anchor["selection_stratum"] = row.selection_stratum
        anchor_records.append(anchor)
    anchors = pd.DataFrame(anchor_records).sort_values(
        ["selection_stratum", "problem_id", "round_id"]
    )
    return (
        selected.sort_values(["selection_stratum", "problem_id"]),
        anchors,
        quotas,
    )


def distribution_summary(profiles, scope):
    class_column = (
        "selection_stratum"
        if "selection_stratum" in profiles.columns
        else "problem_oracle_class"
    )
    purity_column = (
        "selection_stratum_round_percent"
        if "selection_stratum_round_percent" in profiles.columns
        else "class_purity_percent"
    )
    records = []
    for label in PASS_CLASSES:
        group = profiles[profiles[class_column].eq(label)]
        records.append({
            "scope": scope,
            "stratification_class": label,
            "num_questions": int(len(group)),
            "question_percent": 100.0 * len(group) / max(1, len(profiles)),
            "mean_class_purity_percent": (
                float(group[purity_column].mean())
                if len(group)
                else np.nan
            ),
            "mean_oracle_draft_passes": (
                float(group["mean_oracle_draft_passes"].mean())
                if len(group)
                else np.nan
            ),
            "mean_oracle_refinement_steps": (
                float(group["mean_oracle_refinement_steps"].mean())
                if len(group)
                else np.nan
            ),
        })
    return pd.DataFrame(records)


def round_distribution_summary(rounds, scope):
    records = []
    for label in PASS_CLASSES:
        group = rounds[rounds["oracle_pass_class"].eq(label)]
        records.append({
            "scope": scope,
            "oracle_pass_class": label,
            "num_rounds": int(len(group)),
            "round_percent": 100.0 * len(group) / max(1, len(rounds)),
            "mean_factual_draft_passes": (
                float(group["factual_draft_passes"].mean())
                if len(group)
                else np.nan
            ),
            "mean_oracle_draft_passes": (
                float(group["oracle_draft_passes"].mean())
                if len(group)
                else np.nan
            ),
            "mean_oracle_refinement_steps": (
                float(group["oracle_refinement_steps"].mean())
                if len(group)
                else np.nan
            ),
        })
    return pd.DataFrame(records)


def algorithm_ms_per_output_token(results):
    total_ms = 1000.0 * results["actual_algorithm_time"].sum()
    return total_ms / max(1.0, results["output_tokens"].sum())


def causal_oracle_comparison(failfast, causal, decisions):
    failfast_mspt = algorithm_ms_per_output_token(failfast)
    causal_mspt = algorithm_ms_per_output_token(causal)
    modeled_failfast_mspt = (
        failfast["theo_total_time"].sum()
        / max(1.0, failfast["output_tokens"].sum())
    )
    modeled_causal_mspt = (
        causal["theo_total_time"].sum()
        / max(1.0, causal["output_tokens"].sum())
    )
    hashes = failfast[["problem_id", "output_token_hash"]].merge(
        causal[["problem_id", "output_token_hash"]],
        on="problem_id",
        suffixes=("_failfast", "_causal"),
        validate="one_to_one",
    )
    paired_times = failfast[
        ["problem_id", "actual_algorithm_time", "output_tokens"]
    ].merge(
        causal[["problem_id", "actual_algorithm_time", "output_tokens"]],
        on="problem_id",
        suffixes=("_failfast", "_causal"),
        validate="one_to_one",
    )
    paired_speedups = (
        paired_times["actual_algorithm_time_failfast"]
        / paired_times["output_tokens_failfast"].clip(lower=1)
    ) / (
        paired_times["actual_algorithm_time_causal"]
        / paired_times["output_tokens_causal"].clip(lower=1)
    )
    diagnostic_time_s = (
        decisions["counterfactual_probe_wall_time_ms"].sum()
        + decisions["excluded_extra_draft_latency_ms"].sum()
    ) / 1000.0
    fallback_rounds = int(
        pd.to_numeric(
            decisions.get("snapshot_fallback_used", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()
    )
    return pd.DataFrame([{
        "num_questions": int(failfast["problem_id"].nunique()),
        "failfast_algorithm_time_s": float(failfast["actual_algorithm_time"].sum()),
        "causal_oracle_algorithm_time_s": float(causal["actual_algorithm_time"].sum()),
        "failfast_output_tokens": int(failfast["output_tokens"].sum()),
        "causal_oracle_output_tokens": int(causal["output_tokens"].sum()),
        "failfast_ms_per_output_token": failfast_mspt,
        "causal_oracle_ms_per_output_token": causal_mspt,
        "causal_oracle_speedup_vs_failfast": failfast_mspt / causal_mspt,
        "causal_oracle_geometric_mean_speedup_vs_failfast": float(
            np.exp(np.log(paired_speedups.clip(lower=1e-12)).mean())
        ),
        "causal_oracle_win_rate_percent": 100.0 * float(
            paired_speedups.gt(1.0).mean()
        ),
        "causal_oracle_latency_reduction_percent": 100.0 * (
            1.0 - causal_mspt / failfast_mspt
        ),
        "modeled_failfast_ms_per_output_token": modeled_failfast_mspt,
        "modeled_causal_oracle_ms_per_output_token": modeled_causal_mspt,
        "modeled_causal_oracle_speedup_vs_failfast": (
            modeled_failfast_mspt / modeled_causal_mspt
        ),
        "failfast_draft_time_s": float(failfast["actual_draft_time"].sum()),
        "causal_oracle_draft_time_s": float(causal["actual_draft_time"].sum()),
        "failfast_verify_time_s": float(failfast["actual_verify_time"].sum()),
        "causal_oracle_verify_time_s": float(causal["actual_verify_time"].sum()),
        "failfast_post_verify_time_s": float(
            failfast["actual_post_verify_time"].sum()
        ),
        "causal_oracle_post_verify_time_s": float(
            causal["actual_post_verify_time"].sum()
        ),
        "failfast_draft_passes": int(failfast["total_num_forward_passes"].sum()),
        "causal_oracle_draft_passes": int(
            causal["total_num_forward_passes"].sum()
        ),
        "failfast_verifier_rounds": int(failfast["num_speculation_rounds"].sum()),
        "causal_oracle_verifier_rounds": int(
            causal["num_speculation_rounds"].sum()
        ),
        "failfast_acceptance_rate_percent": 100.0 * (
            failfast["accepted_tokens"].sum()
            / max(1.0, failfast["drafted_tokens"].sum())
        ),
        "causal_oracle_acceptance_rate_percent": 100.0 * (
            causal["accepted_tokens"].sum()
            / max(1.0, causal["drafted_tokens"].sum())
        ),
        "output_hash_match_percent": 100.0 * (
            hashes["output_token_hash_failfast"]
            == hashes["output_token_hash_causal"]
        ).mean(),
        "failfast_accuracy_percent": 100.0 * failfast["is_correct"].mean(),
        "causal_oracle_accuracy_percent": 100.0 * causal["is_correct"].mean(),
        "counterfactual_execution_match_percent": 100.0 * decisions[
            "counterfactual_matches_execution"
        ].mean(),
        "excluded_oracle_diagnostic_time_s": diagnostic_time_s,
        "causal_oracle_snapshot_fallback_rounds": fallback_rounds,
        "causal_oracle_snapshot_fallback_rate_percent": (
            100.0 * fallback_rounds / max(1, len(decisions))
        ),
    }])


def paired_causal_comparison(failfast, causal):
    columns = [
        "problem_id",
        "actual_algorithm_time",
        "output_tokens",
        "actual_draft_time",
        "actual_verify_time",
        "actual_post_verify_time",
        "num_speculation_rounds",
        "total_num_forward_passes",
        "acceptance_rate_percent",
        "output_token_hash",
        "is_correct",
    ]
    paired = failfast[columns].merge(
        causal[columns],
        on="problem_id",
        suffixes=("_failfast", "_causal"),
        validate="one_to_one",
    )
    paired["failfast_ms_per_output_token"] = 1000.0 * (
        paired["actual_algorithm_time_failfast"]
        / paired["output_tokens_failfast"].clip(lower=1)
    )
    paired["causal_oracle_ms_per_output_token"] = 1000.0 * (
        paired["actual_algorithm_time_causal"]
        / paired["output_tokens_causal"].clip(lower=1)
    )
    paired["causal_oracle_speedup_vs_failfast"] = (
        paired["failfast_ms_per_output_token"]
        / paired["causal_oracle_ms_per_output_token"]
    )
    paired["causal_oracle_wins"] = (
        paired["causal_oracle_ms_per_output_token"]
        < paired["failfast_ms_per_output_token"]
    )
    paired["output_hash_match"] = (
        paired["output_token_hash_failfast"]
        == paired["output_token_hash_causal"]
    )
    return paired.sort_values("problem_id")


def causal_stop_distribution(decisions):
    profiled = decisions.copy()
    profiled["selected_refinement_class"] = profiled["selected_step"].map(
        pass_class
    )
    return profiled.groupby("selected_refinement_class", sort=False).agg(
        rounds=("round_id", "size"),
        mean_selected_refinement_steps=("selected_step", "mean"),
        mean_selected_draft_passes=("selected_draft_passes", "mean"),
        mean_candidates=("num_candidates", "mean"),
        mean_expected_emitted_tokens=("selected_expected_emitted_len", "mean"),
        mean_executed_emitted_tokens=("executed_emitted_len", "mean"),
        mean_effective_draft_latency_ms=("selected_draft_latency_ms", "mean"),
        mean_executed_verify_latency_ms=("executed_verify_latency_ms", "mean"),
        execution_match_rate=("counterfactual_matches_execution", "mean"),
    ).reset_index()


def manifest(args, candidate_ids, evaluated_ids, selected, quotas):
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
        "candidate_problem_ids": candidate_ids,
        "evaluated_problem_ids": evaluated_ids,
        "num_evaluated_questions": len(evaluated_ids),
        "selected_problem_ids": selected["problem_id"].astype(int).tolist(),
        "balanced_class_quotas": quotas,
        "selection_definition": (
            "Each selected question contributes one randomly selected anchor round. "
            "Anchor-round quotas are balanced by selected_step, the local refinement "
            "depth at the proposal length chosen and executed by the causal oracle over "
            "the evaluated question pool: 1 is step1, 2 is step2, and >=3 is step3plus. "
            "Total draft forward passes remain a separate latency metric. Questions are "
            "unique across strata, and the complete causal-round distribution is retained."
        ),
        "causal_oracle_interpretation": (
            "The causal oracle chooses an observed refinement snapshot, executes that "
            "proposal, commits its verified output, and generates the next round from "
            "the resulting context. Counterfactual probes and draft work beyond the "
            "chosen snapshot are excluded and reported separately as oracle-only "
            "diagnostic cost. No local replay speedup is reported."
        ),
    }


def main():
    args = parse_args()
    args.dataset = "gsm8k"
    args.num_questions = args.pool_size
    args.adaptive_state_path = None
    args.oracle_only = True
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = pool_problem_ids(args)
    causal_results, causal_decisions, causal_candidates = run_causal_oracle(
        args,
        candidate_ids,
    )
    evaluated_ids = causal_results["problem_id"].astype(int).tolist()
    failfast_results = run_failfast_pool(args, evaluated_ids)
    rounds = causal_selection_rounds(causal_decisions)
    profiles = problem_profiles(rounds)

    failfast_results.to_csv(output_dir / "pool_failfast_results.csv", index=False)
    causal_results.to_csv(output_dir / "pool_causal_oracle_results.csv", index=False)
    causal_decisions.to_csv(
        output_dir / "pool_causal_oracle_decisions.csv", index=False
    )
    causal_candidates.to_csv(
        output_dir / "pool_causal_oracle_candidates.csv", index=False
    )
    failfast_results.to_csv(
        output_dir / "all_evaluated_failfast_results.csv", index=False
    )
    causal_results.to_csv(
        output_dir / "all_evaluated_causal_oracle_results.csv", index=False
    )
    causal_decisions.to_csv(
        output_dir / "all_evaluated_causal_oracle_decisions.csv", index=False
    )
    causal_candidates.to_csv(
        output_dir / "all_evaluated_causal_oracle_candidates.csv", index=False
    )
    rounds.to_csv(
        output_dir / "all_evaluated_causal_oracle_rounds.csv", index=False
    )
    profiles.to_csv(
        output_dir / "all_evaluated_problem_profiles.csv", index=False
    )
    profiles.to_csv(output_dir / "pool_problem_profiles.csv", index=False)
    distribution_summary(profiles, "pool").to_csv(
        output_dir / "pool_class_distribution.csv", index=False
    )
    round_distribution_summary(rounds, "pool_all_rounds").to_csv(
        output_dir / "pool_round_class_distribution.csv", index=False
    )

    selected, anchors, quotas = select_balanced_problems(
        profiles,
        rounds,
        args.selected_size,
        args.selection_seed,
    )
    selected_ids = sorted(set(selected["problem_id"].astype(int)))
    selected_failfast = failfast_results[
        failfast_results["problem_id"].isin(selected_ids)
    ].copy()
    selected_causal = causal_results[
        causal_results["problem_id"].isin(selected_ids)
    ].copy()
    selected_decisions = causal_decisions[
        causal_decisions["problem_id"].isin(selected_ids)
    ].copy()
    selected_candidates = causal_candidates[
        causal_candidates["problem_id"].isin(selected_ids)
    ].copy()
    selected_rounds = rounds[rounds["problem_id"].isin(selected_ids)].copy()

    selected.to_csv(output_dir / "selected_problem_profiles.csv", index=False)
    anchors.to_csv(output_dir / "selected_anchor_rounds.csv", index=False)
    selected_failfast.to_csv(
        output_dir / "selected_failfast_results.csv", index=False
    )
    selected_causal.to_csv(
        output_dir / "selected_causal_oracle_results.csv", index=False
    )
    selected_decisions.to_csv(
        output_dir / "selected_causal_oracle_decisions.csv", index=False
    )
    selected_candidates.to_csv(
        output_dir / "selected_causal_oracle_candidates.csv", index=False
    )
    selected_rounds.to_csv(
        output_dir / "selected_causal_oracle_rounds.csv", index=False
    )
    causal_stop_distribution(causal_decisions).to_csv(
        output_dir / "pool_causal_oracle_stop_distribution.csv",
        index=False,
    )
    causal_stop_distribution(selected_decisions).to_csv(
        output_dir / "causal_oracle_stop_distribution.csv",
        index=False,
    )
    distribution_summary(selected, "selected").to_csv(
        output_dir / "selected_class_distribution.csv", index=False
    )
    pd.concat(
        [
            round_distribution_summary(
                selected_rounds,
                "selected_all_rounds",
            ),
            round_distribution_summary(
                anchors,
                "selected_anchor_rounds",
            ),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "selected_round_class_distribution.csv", index=False)

    pool_summary = causal_oracle_comparison(
        failfast_results,
        causal_results,
        causal_decisions,
    )
    pool_summary.insert(0, "scope", f"all_evaluated_{len(evaluated_ids)}")
    selected_summary = causal_oracle_comparison(
        selected_failfast,
        selected_causal,
        selected_decisions,
    )
    selected_summary.insert(0, "scope", "balanced_50")
    causal_summary = pd.concat(
        [pool_summary, selected_summary], ignore_index=True
    )
    causal_summary.to_csv(output_dir / "causal_oracle_summary.csv", index=False)
    paired_causal_comparison(failfast_results, causal_results).to_csv(
        output_dir / "pool_causal_oracle_paired_results.csv",
        index=False,
    )
    paired_causal_comparison(selected_failfast, selected_causal).to_csv(
        output_dir / "causal_oracle_paired_results.csv",
        index=False,
    )
    (output_dir / "selected_problem_ids.json").write_text(
        json.dumps(selected_ids, indent=2), encoding="utf-8"
    )
    (output_dir / "all_evaluated_problem_ids.json").write_text(
        json.dumps(evaluated_ids, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(
            manifest(args, candidate_ids, evaluated_ids, selected, quotas),
            indent=2,
        ),
        encoding="utf-8",
    )

    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nBALANCED 50-QUESTION CLASS DISTRIBUTION")
    print(distribution_summary(selected, "selected").to_string(index=False))
    print("\nEXECUTED CAUSAL ORACLE SUMMARY")
    print(causal_summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
