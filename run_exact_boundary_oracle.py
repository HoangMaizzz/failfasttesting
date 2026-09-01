import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from adaptive_td import CONTINUE, STOP
from run_chunked_c6_comparison_test50 import adaptive_flags, base_command, complete
from run_local_stop_continue_oracle import build_verifier_profile
from run_otrc_v2_td_benchmark import PROBLEM_IDS
from raw_state_experiment import parse_nested


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ("math", "gsm8k")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exact STOP/CONTINUE dynamic programming to the next verifier "
            "boundary on five C6 behavior rounds per problem."
        )
    )
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(PROBLEM_IDS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--pilot_questions", type=int, default=2)
    parser.add_argument("--boundaries_per_problem", type=int, default=5)
    parser.add_argument("--max_states_per_boundary", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("int8", "int8_deterministic", "none"),
        default="int8",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_exact_boundary_oracle_math_gsm8k_test10"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if not 0 < args.pilot_questions <= args.num_questions:
        raise ValueError("--pilot_questions must be in [1, num_questions]")
    if args.boundaries_per_problem <= 0:
        raise ValueError("--boundaries_per_problem must be positive")
    if args.max_states_per_boundary <= 1:
        raise ValueError("--max_states_per_boundary must exceed one")
    if len(args.datasets) != 2 or len(set(args.datasets)) != 2:
        raise ValueError("--datasets requires exactly two distinct datasets")
    for dataset in args.datasets:
        if args.num_questions > len(PROBLEM_IDS[dataset]):
            raise ValueError(f"not enough fixed IDs for {dataset}")


def command_args(args):
    return SimpleNamespace(
        dllm_dir=args.dllm_dir,
        target_device=args.target_device,
        drafter_device=args.drafter_device,
        target_quantization=args.target_quantization,
        log_level=args.log_level,
    )


def run_streaming(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def fixed_ids(dataset, count):
    return [int(value) for value in PROBLEM_IDS[dataset][:count]]


def run_method(args, dataset, problem_ids, directory, *, adaptive):
    expected_adaptive = bool(adaptive)
    if args.resume and complete(directory, problem_ids, expected_adaptive):
        extra = ["adaptive_td_decisions.csv"] if adaptive else ["verifier_calls.csv"]
        if all((directory / name).exists() for name in extra):
            print(f"SKIP completed {'C6' if adaptive else 'profile'}: {dataset}")
            return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    command = base_command(
        command_args(args), dataset, problem_ids, directory, 1
    )
    command[command.index("--max_new_tokens") + 1] = str(args.max_new_tokens)
    if adaptive:
        command.extend(adaptive_flags("c6_annealed"))
        # Exact branches disable KV reuse to keep memory bounded.  Collect the
        # behavior trajectory through the identical drafter path so its action
        # script remains replayable at exact-boundary time.
        command.append("--disable_reusing_drafter_kvs")
    print("\n" + "=" * 96, flush=True)
    print(
        f"RUN {'C6 BEHAVIOR' if adaptive else 'FROZEN PROFILE'} "
        f"{dataset.upper()} | questions={len(problem_ids)}",
        flush=True,
    )
    print("=" * 96, flush=True)
    run_streaming(command)
    if not complete(directory, problem_ids, expected_adaptive):
        raise RuntimeError(f"incomplete output: {directory}")
    return directory


def augment_frozen_profile(profile_dir, destination):
    profile = build_verifier_profile(profile_dir, destination)
    results = pd.read_csv(profile_dir / "benchmark_results.csv")
    passes = max(1.0, float(results.total_num_forward_passes.sum()))
    rounds = max(1.0, float(results.num_speculation_rounds.sum()))
    algorithm_ms = max(
        1e-9, 1000.0 * float(results.actual_algorithm_time.sum())
    )
    profile.update({
        "version": "exact_boundary_frozen_hardware_profile_v1",
        "mean_draft_forward_latency_ms": (
            1000.0 * float(results.actual_draft_time.sum()) / passes
        ),
        "mean_post_verify_latency_ms": (
            1000.0 * float(results.actual_post_verify_time.sum()) / rounds
        ),
        "rho_tokens_per_ms": (
            float(results.output_tokens.sum()) / algorithm_ms
        ),
        "profile_problem_ids": [
            int(value) for value in results.problem_id.tolist()
        ],
    })
    destination.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def legal_decisions(frame):
    frame = frame.copy()
    values = frame["stop_available"]
    if values.dtype == bool:
        mask = values
    else:
        mask = values.astype(str).str.lower().isin({"1", "true", "yes"})
    frame = frame.loc[mask].sort_values(
        ["problem_id", "round_id", "decision_id"]
    )
    frame["logged_decision_id"] = pd.to_numeric(frame.decision_id).astype(int)
    frame["decision_id"] = (
        frame.groupby(["problem_id", "round_id"], sort=False)
        .cumcount()
        .astype(int)
    )
    return frame


def evenly_spaced(values, limit):
    values = list(values)
    count = min(len(values), int(limit))
    if count <= 0:
        return []
    positions = sorted(
        set(np.linspace(0, len(values) - 1, count).round().astype(int))
    )
    return [values[position] for position in positions]


def build_exact_behavior_policy(decisions, results, boundaries_per_problem):
    replay_decisions = decisions.copy().sort_values(
        ["problem_id", "round_id", "decision_id"]
    )
    decisions = legal_decisions(decisions)
    policies = {}
    selected_rows = []
    selected_round_rows = []
    for result in results.itertuples(index=False):
        problem_id = int(result.problem_id)
        problem = decisions[decisions.problem_id.astype(int) == problem_id]
        eligible_rounds = sorted(problem.round_id.astype(int).unique().tolist())
        selected_rounds = set(evenly_spaced(eligible_rounds, boundaries_per_problem))
        selected_rows.append(problem[problem.round_id.astype(int).isin(selected_rounds)])
        for round_id in selected_rounds:
            selected_round_rows.append({
                "problem_id": problem_id,
                "round_id": int(round_id),
            })
        rounds = []
        for round_id in range(int(result.num_speculation_rounds)):
            replay_problem = replay_decisions[
                replay_decisions.problem_id.astype(int) == problem_id
            ]
            current = replay_problem[
                replay_problem.round_id.astype(int) == round_id
            ]
            action_column = (
                "executed_action"
                if "executed_action" in current.columns
                else "action"
            )
            actions = [str(value) for value in current[action_column].tolist()]
            if any(value not in {STOP, CONTINUE} for value in actions):
                raise ValueError(f"invalid action for problem {problem_id}")
            rounds.append({
                "round_id": int(round_id),
                "actions": actions,
                "exact_boundary_probe": bool(round_id in selected_rounds),
            })
        policies[str(problem_id)] = rounds
    selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else decisions.iloc[0:0].copy()
    )
    selected_rounds = pd.DataFrame(selected_round_rows)
    return {
        "version": "exact_boundary_c6_behavior_v1",
        "policies": policies,
    }, selected, selected_rounds


def write_problem_policy(policy, problem_id, destination):
    destination.write_text(json.dumps({
        "version": policy["version"],
        "policies": {str(problem_id): policy["policies"][str(problem_id)]},
    }, indent=2), encoding="utf-8")


def exact_command(args, dataset, problem_id, directory, profile, policy):
    command = base_command(
        command_args(args), dataset, [problem_id], directory, 0
    )
    command[command.index("--max_new_tokens") + 1] = str(args.max_new_tokens)
    command.extend([
        # Exact replay repeatedly forks drafter trajectories.  Keeping a
        # selected-path KV cache alive while the int8 verifier runs leaves too
        # little headroom on 16 GB cards; exact scoring already uses the frozen
        # latency profile, so recomputing the cache does not bias the objective.
        "--disable_reusing_drafter_kvs",
        "--strict_greedy_local_oracle",
        "--strict_greedy_capacity_collector",
        "--strict_greedy_exact_boundary",
        "--strict_greedy_exact_max_states", str(args.max_states_per_boundary),
        "--strict_greedy_verifier_profile", str(profile),
        "--strict_greedy_behavior_policy", str(policy),
        "--strict_greedy_epsilon_ms", str(args.epsilon_ms),
    ])
    return command


def exact_complete(directory, problem_id):
    required = [
        directory / "benchmark_results.csv",
        directory / "exact_boundary_tree_summary.csv",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        results = pd.read_csv(required[0])
        summary = pd.read_csv(required[1])
    except (OSError, pd.errors.EmptyDataError):
        return False
    return (
        set(results.problem_id.astype(int)) == {int(problem_id)}
        and not summary.empty
    )


def run_exact_problem(args, dataset, problem_id, profile, policy):
    directory = (
        Path(args.output_dir) / "raw" / dataset / "exact" / f"id_{problem_id}"
    )
    if args.resume and exact_complete(directory, problem_id):
        print(f"SKIP exact {dataset} problem={problem_id}", flush=True)
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    print("\n" + "#" * 96, flush=True)
    print(f"EXACT BOUNDARY {dataset.upper()} | problem={problem_id}", flush=True)
    print("#" * 96, flush=True)
    run_streaming(exact_command(
        args, dataset, problem_id, directory, profile, policy
    ))
    if not exact_complete(directory, problem_id):
        raise RuntimeError(f"incomplete exact output: {directory}")
    gc.collect()
    return directory


def concat_existing(paths):
    frames = []
    for path in paths:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_exact_raw_states(rows):
    required = {
        "raw_previous_state",
        "raw_current_state",
        "has_previous_state",
        "active_block_start",
        "active_block_end",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise RuntimeError(f"exact raw-state log is missing: {sorted(missing)}")
    for row in rows.itertuples(index=False):
        previous = parse_nested(row.raw_previous_state)
        current = parse_nested(row.raw_current_state)
        if len(previous) != 8 or len(current) != 8:
            raise RuntimeError("exact raw-state snapshot must contain 8 positions")
        if any(len(values) != 6 for values in previous + current):
            raise RuntimeError("exact raw-state positions must contain 6 values")
        if any(abs(float(values[1]) - 1.0) > 1e-8 for values in previous + current):
            raise RuntimeError("exact raw-state decision contains an unobserved token")
        if not bool(int(row.has_previous_state)) and previous != current:
            raise RuntimeError("first exact raw state did not copy current to previous")
        if int(row.active_block_end) - int(row.active_block_start) != 8:
            raise RuntimeError("exact raw state crossed an active-block boundary")


def summarize_subset(
    root, behavior_dirs, selected_by_dataset, ids_by_dataset, datasets, label
):
    report_dir = root / label
    report_dir.mkdir(parents=True, exist_ok=True)
    all_alignment = []
    all_trees = []
    output_checks = []
    for dataset in datasets:
        allowed = set(ids_by_dataset[dataset])
        exact_dirs = [
            root / "raw" / dataset / "exact" / f"id_{problem_id}"
            for problem_id in ids_by_dataset[dataset]
        ]
        trees = concat_existing([
            directory / "exact_boundary_tree_summary.csv"
            for directory in exact_dirs
        ])
        if not trees.empty:
            trees.insert(0, "dataset", dataset)
            all_trees.append(trees)
        behavior_results = pd.read_csv(
            behavior_dirs[dataset] / "benchmark_results.csv"
        )
        exact_results = concat_existing([
            directory / "benchmark_results.csv" for directory in exact_dirs
        ])
        if not exact_results.empty:
            check = behavior_results[
                behavior_results.problem_id.astype(int).isin(allowed)
            ][["problem_id", "output_token_hash"]].merge(
                exact_results[["problem_id", "output_token_hash"]],
                on="problem_id",
                suffixes=("_behavior", "_replay"),
                validate="one_to_one",
            )
            check["output_match"] = (
                check.output_token_hash_behavior == check.output_token_hash_replay
            )
            check.insert(0, "dataset", dataset)
            output_checks.append(check)
        rows = concat_existing([
            directory / "greedy_local_oracle_decisions.csv"
            for directory in exact_dirs
        ])
        if rows.empty:
            continue
        validate_exact_raw_states(rows)
        rows = rows[rows.on_behavior_path.astype(int) == 1].copy()
        rows = rows.rename(columns={"sample_id": "problem_id"})
        selected = selected_by_dataset[dataset]
        selected = selected[selected.problem_id.astype(int).isin(allowed)].copy()
        key = ["problem_id", "round_id", "decision_id"]
        alignment = selected.merge(
            rows,
            on=key,
            how="inner",
            suffixes=("_c6", "_oracle"),
            validate="one_to_one",
        )
        alignment["c6_advantage"] = (
            pd.to_numeric(alignment.q_stop_mean)
            - pd.to_numeric(alignment.q_continue_mean)
        )
        alignment["c6_action"] = np.where(
            alignment.c6_advantage > 0.0, STOP, CONTINUE
        )
        alignment["sign_correct"] = (
            alignment.c6_action == alignment.oracle_action
        )
        alignment.insert(0, "dataset", dataset)
        alignment.to_csv(report_dir / f"{dataset}_exact_alignment.csv", index=False)
        all_alignment.append(alignment)

    alignment = (
        pd.concat(all_alignment, ignore_index=True)
        if all_alignment
        else pd.DataFrame()
    )
    if alignment.empty:
        alignment.to_csv(report_dir / "all_exact_alignment.csv", index=False)
        pd.DataFrame().to_csv(
            report_dir / "exact_alignment_summary.csv", index=False
        )
        if all_trees:
            pd.concat(all_trees, ignore_index=True).to_csv(
                report_dir / "exact_tree_summary.csv", index=False
            )
        checks = (
            pd.concat(output_checks, ignore_index=True)
            if output_checks else pd.DataFrame()
        )
        checks.to_csv(report_dir / "output_replay_check.csv", index=False)
        if not checks.empty and not bool(checks.output_match.all()):
            raise RuntimeError(f"exact replay changed target output in {label}")
        return report_dir

    non_tie = alignment[alignment.oracle_label != "tie"].copy()
    summaries = []
    for dataset, frame in non_tie.groupby("dataset"):
        truth_stop = frame.oracle_action.eq(STOP)
        predicted_stop = frame.c6_action.eq(STOP)
        summaries.append({
            "dataset": dataset,
            "problems": int(frame.problem_id.nunique()),
            "states": int(len(frame)),
            "stop_fraction_percent": 100.0 * float(truth_stop.mean()),
            "sign_accuracy_percent": 100.0 * float(frame.sign_correct.mean()),
            "stop_recall_percent": 100.0 * float(
                predicted_stop[truth_stop].mean()
            ) if truth_stop.any() else float("nan"),
            "continue_recall_percent": 100.0 * float(
                (~predicted_stop[~truth_stop]).mean()
            ) if (~truth_stop).any() else float("nan"),
            "advantage_spearman": float(
                frame.c6_advantage.rank().corr(
                    frame.oracle_advantage_tokens.rank()
                )
            ),
        })
    pd.DataFrame(summaries).to_csv(
        report_dir / "exact_alignment_summary.csv", index=False
    )
    alignment.to_csv(report_dir / "all_exact_alignment.csv", index=False)
    if all_trees:
        pd.concat(all_trees, ignore_index=True).to_csv(
            report_dir / "exact_tree_summary.csv", index=False
        )
    checks = (
        pd.concat(output_checks, ignore_index=True)
        if output_checks else pd.DataFrame()
    )
    checks.to_csv(report_dir / "output_replay_check.csv", index=False)
    if not checks.empty and not bool(checks.output_match.all()):
        raise RuntimeError(f"exact replay changed target output in {label}")
    return report_dir


def archive(root, suffix):
    destination = root.parent / f"{root.name}_{suffix}"
    archive_path = shutil.make_archive(
        str(destination), "zip", root.parent, root.name
    )
    print(f"\nARCHIVE READY: {archive_path}", flush=True)
    return archive_path


def main():
    args = parse_args()
    validate_args(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    datasets = tuple(args.datasets)
    ids = {dataset: fixed_ids(dataset, args.num_questions) for dataset in datasets}
    behavior_dirs = {}
    selected = {}
    policies = {}
    profile_paths = {}

    for dataset in datasets:
        profile_dir = run_method(
            args,
            dataset,
            ids[dataset],
            root / "raw" / dataset / "frozen_profile_prepass",
            adaptive=False,
        )
        profile_path = root / dataset / "frozen_hardware_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        augment_frozen_profile(profile_dir, profile_path)
        profile_paths[dataset] = profile_path

        behavior_dir = run_method(
            args,
            dataset,
            ids[dataset],
            root / "raw" / dataset / "c6_behavior",
            adaptive=True,
        )
        behavior_dirs[dataset] = behavior_dir
        policy, selected_states, selected_rounds = build_exact_behavior_policy(
            pd.read_csv(behavior_dir / "adaptive_td_decisions.csv"),
            pd.read_csv(behavior_dir / "benchmark_results.csv"),
            args.boundaries_per_problem,
        )
        policy_path = root / dataset / "behavior_policy.json"
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        selected_states.to_csv(root / dataset / "selected_c6_states.csv", index=False)
        selected_rounds.to_csv(root / dataset / "selected_boundary_rounds.csv", index=False)
        policies[dataset] = policy
        selected[dataset] = selected_states

    per_problem_policies = {}
    for dataset in datasets:
        directory = root / dataset / "behavior_policies"
        directory.mkdir(parents=True, exist_ok=True)
        for problem_id in ids[dataset]:
            path = directory / f"id_{problem_id}.json"
            write_problem_policy(policies[dataset], problem_id, path)
            per_problem_policies[(dataset, problem_id)] = path

    # Complete both datasets' pilot before moving to the remaining questions.
    for problem_index in range(args.pilot_questions):
        for dataset in datasets:
            problem_id = ids[dataset][problem_index]
            run_exact_problem(
                args,
                dataset,
                problem_id,
                profile_paths[dataset],
                per_problem_policies[(dataset, problem_id)],
            )
    pilot_ids = {
        dataset: values[: args.pilot_questions]
        for dataset, values in ids.items()
    }
    summarize_subset(
        root, behavior_dirs, selected, pilot_ids, datasets, "pilot4_report"
    )
    if not args.skip_archive:
        archive(root, "pilot4")

    for problem_index in range(args.pilot_questions, args.num_questions):
        for dataset in datasets:
            problem_id = ids[dataset][problem_index]
            run_exact_problem(
                args,
                dataset,
                problem_id,
                profile_paths[dataset],
                per_problem_policies[(dataset, problem_id)],
            )
    summarize_subset(root, behavior_dirs, selected, ids, datasets, "final_report")
    manifest = {
        "version": "exact_verifier_boundary_oracle_v1",
        "datasets": list(datasets),
        "problem_ids": ids,
        "boundaries_per_problem": args.boundaries_per_problem,
        "max_states_per_boundary": args.max_states_per_boundary,
        "objective": "Y - frozen_rho * frozen_profile_latency",
        "measured_verifier_latency_used_for_winner": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if not args.skip_archive:
        archive(root, "final")


if __name__ == "__main__":
    main()
