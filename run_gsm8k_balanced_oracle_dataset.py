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
    add_latency_estimates,
    build_failfast_oracle_transitions,
    run_phase,
    select_round_candidates,
)


VERSION = "gsm8k_balanced_failfast_oracle_v1"
PASS_CLASSES = ("step1", "step2", "step3plus")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--selected_size", type=int, default=50)
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
    parser.add_argument("--theoretical_draft_forward_ms", type=float, default=6.1)
    parser.add_argument("--theoretical_verify_round_ms", type=float, default=13.5)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--selection_seed", type=int, default=2027)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_gsm8k_balanced_oracle_pool300_test50"
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
    return sorted(random.Random(args.sample_seed).sample(population, args.pool_size))


def problem_profiles(rounds):
    profiled = rounds.copy()
    profiled["oracle_pass_class"] = profiled["oracle_draft_passes"].map(pass_class)
    records = []
    for problem_id, group in profiled.groupby("problem_id", sort=True):
        counts = group["oracle_pass_class"].value_counts()
        median_passes = float(group["oracle_draft_passes"].median())
        problem_class = pass_class(median_passes)
        record = {
            "problem_id": int(problem_id),
            "problem_oracle_class": problem_class,
            "num_rounds": int(len(group)),
            "median_oracle_draft_passes": median_passes,
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
        })
    return pd.DataFrame(records)


def local_oracle_upper_bound(
    results,
    rounds,
    scope,
    theoretical_draft_forward_ms=6.1,
    theoretical_verify_round_ms=13.5,
):
    measured_total_ms = 1000.0 * (
        results["actual_draft_time"].sum()
        + results["actual_verify_time"].sum()
        + results["actual_post_verify_time"].sum()
    )
    measured_output_tokens = float(results["output_tokens"].sum())
    factual_latency_ms = float(rounds["factual_latency_ms"].sum())
    factual_output_tokens = float(rounds["factual_output_tokens"].sum())
    oracle_latency_ms = float(rounds["oracle_latency_ms"].sum())
    oracle_output_tokens = float(rounds["oracle_output_tokens"].sum())
    measured_mspt = measured_total_ms / max(1.0, measured_output_tokens)
    factual_mspt = factual_latency_ms / max(1.0, factual_output_tokens)
    oracle_mspt = oracle_latency_ms / max(1.0, oracle_output_tokens)
    speedup = factual_mspt / oracle_mspt
    theoretical_factual_latency_ms = float(
        rounds["factual_draft_passes"].sum() * theoretical_draft_forward_ms
        + len(rounds) * theoretical_verify_round_ms
    )
    theoretical_oracle_latency_ms = float(
        rounds["oracle_draft_passes"].sum() * theoretical_draft_forward_ms
        + len(rounds) * theoretical_verify_round_ms
    )
    theoretical_factual_mspt = (
        theoretical_factual_latency_ms / max(1.0, factual_output_tokens)
    )
    theoretical_oracle_mspt = (
        theoretical_oracle_latency_ms / max(1.0, oracle_output_tokens)
    )
    theoretical_speedup = theoretical_factual_mspt / theoretical_oracle_mspt
    return pd.DataFrame([{
        "scope": scope,
        "num_questions": int(results["problem_id"].nunique()),
        "num_rounds": int(len(rounds)),
        "measured_failfast_total_time_s": measured_total_ms / 1000.0,
        "measured_failfast_output_tokens": int(measured_output_tokens),
        "measured_failfast_ms_per_output_token": measured_mspt,
        "replay_failfast_total_time_s": factual_latency_ms / 1000.0,
        "replay_failfast_output_tokens": int(factual_output_tokens),
        "replay_failfast_ms_per_output_token": factual_mspt,
        "local_oracle_total_time_s": oracle_latency_ms / 1000.0,
        "local_oracle_output_tokens": int(oracle_output_tokens),
        "local_oracle_ms_per_output_token": oracle_mspt,
        "local_oracle_upper_bound_speedup_vs_failfast_replay": speedup,
        "local_oracle_upper_bound_latency_reduction_percent": 100.0 * (1.0 - 1.0 / speedup),
        "measured_to_local_oracle_speedup": measured_mspt / oracle_mspt,
        "theoretical_draft_forward_ms": theoretical_draft_forward_ms,
        "theoretical_verify_round_ms": theoretical_verify_round_ms,
        "theoretical_failfast_ms_per_output_token": theoretical_factual_mspt,
        "theoretical_local_oracle_ms_per_output_token": theoretical_oracle_mspt,
        "theoretical_local_oracle_upper_bound_speedup_vs_failfast": theoretical_speedup,
        "theoretical_local_oracle_latency_reduction_percent": 100.0 * (
            1.0 - 1.0 / theoretical_speedup
        ),
        "factual_draft_passes": int(rounds["factual_draft_passes"].sum()),
        "local_oracle_draft_passes": int(rounds["oracle_draft_passes"].sum()),
        "draft_pass_reduction_percent": 100.0 * (
            1.0
            - rounds["oracle_draft_passes"].sum()
            / max(1.0, rounds["factual_draft_passes"].sum())
        ),
        "factual_draft_latency_s": rounds["factual_draft_latency_ms"].sum() / 1000.0,
        "local_oracle_draft_latency_s": rounds["oracle_draft_latency_ms"].sum() / 1000.0,
        "factual_verify_latency_s": rounds["factual_verify_latency_ms"].sum() / 1000.0,
        "local_oracle_verify_latency_s": rounds["oracle_verify_latency_ms"].sum() / 1000.0,
        "factual_post_verify_latency_s": rounds[
            "factual_post_verify_latency_ms"
        ].sum() / 1000.0,
        "local_oracle_post_verify_latency_s": rounds[
            "oracle_post_verify_latency_ms"
        ].sum() / 1000.0,
        "round_oracle_choice_accuracy_percent": 100.0,
    }])


def manifest(args, pool_ids, selected, quotas):
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
        "pool_problem_ids": pool_ids,
        "selected_problem_ids": selected["problem_id"].astype(int).tolist(),
        "balanced_class_quotas": quotas,
        "selection_definition": (
            "Each selected question contributes one randomly selected anchor round. "
            "Anchor-round quotas are balanced across local-oracle total draft forward "
            "passes: 1 is step1, 2 is step2, and >=3 is step3plus. Questions are "
            "unique across strata. The median class and complete round distribution "
            "for every question remain available as diagnostics. draft_passes_elapsed "
            "is used because the local step counter resets after proposal extension."
        ),
        "upper_bound_interpretation": (
            "The reported maximum is a local one-round replay upper bound that picks "
            "the lowest measured latency per emitted token among observed snapshots. "
            "It assumes every local choice is correct but does not regenerate the "
            "causal future trajectory, so it is not a deployable end-to-end speedup."
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
    pool_ids = pool_problem_ids(args)
    raw_dir = run_phase(
        args,
        pool_ids,
        "failfast_oracle_pool",
        ["--collect_bucket_oracle"],
        ["benchmark_results.csv", "bucket_oracle_snapshots.csv"],
    )

    results = pd.read_csv(raw_dir / "benchmark_results.csv")
    snapshots = add_latency_estimates(
        pd.read_csv(raw_dir / "bucket_oracle_snapshots.csv")
    )
    transitions = build_failfast_oracle_transitions(snapshots)
    rounds = select_round_candidates(snapshots)
    rounds["oracle_pass_class"] = rounds["oracle_draft_passes"].map(pass_class)
    profiles = problem_profiles(rounds)

    results.to_csv(output_dir / "pool_failfast_results.csv", index=False)
    snapshots.to_csv(output_dir / "pool_oracle_snapshots.csv", index=False)
    transitions.to_csv(output_dir / "pool_oracle_transitions.csv", index=False)
    rounds.to_csv(output_dir / "pool_round_oracle_choices.csv", index=False)
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
    selected_ids = set(selected["problem_id"].astype(int))
    selected_results = results[results["problem_id"].isin(selected_ids)].copy()
    selected_snapshots = snapshots[
        snapshots["problem_id"].isin(selected_ids)
    ].copy()
    selected_transitions = transitions[
        transitions["problem_id"].isin(selected_ids)
    ].copy()
    selected_rounds = rounds[rounds["problem_id"].isin(selected_ids)].copy()

    selected.to_csv(output_dir / "selected_problem_profiles.csv", index=False)
    anchors.to_csv(output_dir / "selected_anchor_rounds.csv", index=False)
    selected_results.to_csv(output_dir / "selected_failfast_results.csv", index=False)
    selected_snapshots.to_csv(output_dir / "selected_oracle_snapshots.csv", index=False)
    selected_transitions.to_csv(
        output_dir / "selected_oracle_transitions.csv", index=False
    )
    selected_rounds.to_csv(
        output_dir / "selected_round_oracle_choices.csv", index=False
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

    upper_bounds = pd.concat(
        [
            local_oracle_upper_bound(
                results,
                rounds,
                "pool",
                args.theoretical_draft_forward_ms,
                args.theoretical_verify_round_ms,
            ),
            local_oracle_upper_bound(
                selected_results,
                selected_rounds,
                "balanced_selected",
                args.theoretical_draft_forward_ms,
                args.theoretical_verify_round_ms,
            ),
        ],
        ignore_index=True,
    )
    upper_bounds.to_csv(output_dir / "local_oracle_upper_bound.csv", index=False)
    (output_dir / "selected_problem_ids.json").write_text(
        json.dumps(sorted(selected_ids), indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest(args, pool_ids, selected, quotas), indent=2),
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
    print("\nLOCAL ORACLE UPPER BOUND")
    print(upper_bounds.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
