import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from adaptive_td import CONTINUE, STOP, V22_COMPACT_FEATURE_NAMES
from run_chunked_c6_comparison_test50 import adaptive_flags, base_command, complete
from run_local_stop_continue_oracle import build_verifier_profile
from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure factual STOP/CONTINUE boundary advantage on states visited "
            "by Compact6 Shared V+A, then audit current-Q and feature capacity."
        )
    )
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--max_states_per_problem", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("none", "int8", "int8_deterministic"),
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
            "outputs_paired_boundary_feature_probe_math_gsm8k_20"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 1:
        raise ValueError("--num_questions must exceed one for grouped evaluation")
    if args.max_states_per_problem <= 0:
        raise ValueError("--max_states_per_problem must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")
    for dataset in DATASETS:
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


def behavior_complete(directory, problem_ids):
    return (
        complete(directory, problem_ids, True)
        and (directory / "adaptive_td_decisions.csv").exists()
        and (directory / "verifier_calls.csv").exists()
    )


def run_behavior(args, dataset, problem_ids):
    directory = Path(args.output_dir) / "raw" / dataset / "c6_behavior"
    if args.resume and behavior_complete(directory, problem_ids):
        print(f"SKIP completed C6 behavior: {dataset}", flush=True)
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    command = base_command(command_args(args), dataset, problem_ids, directory, 1)
    command[command.index("--max_new_tokens") + 1] = str(args.max_new_tokens)
    command.extend(adaptive_flags("c6_annealed"))
    print("\n" + "=" * 96, flush=True)
    print(f"C6 BEHAVIOR {dataset.upper()} | questions={len(problem_ids)}", flush=True)
    print("=" * 96, flush=True)
    run_streaming(command)
    if not behavior_complete(directory, problem_ids):
        raise RuntimeError(f"incomplete C6 behavior output: {directory}")
    return directory


def evenly_spaced_positions(length, limit):
    count = min(int(length), int(limit))
    if count <= 0:
        return []
    if count == length:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, count).round().astype(int).tolist()))


def build_behavior_policy(decisions, results, max_states_per_problem):
    policies = {}
    selected_rows = []
    decisions = decisions.sort_values(
        ["problem_id", "round_id", "decision_id"]
    ).copy()
    for problem_id, result in results.groupby("problem_id", sort=False):
        problem_id = int(problem_id)
        problem_rows = decisions[decisions.problem_id.astype(int) == problem_id]
        selected_positions = set(
            evenly_spaced_positions(len(problem_rows), max_states_per_problem)
        )
        selected_index = {
            (int(row.round_id), int(row.decision_id))
            for position, row in enumerate(problem_rows.itertuples(index=False))
            if position in selected_positions
        }
        selected_rows.append(
            problem_rows[
                [
                    (int(row.round_id), int(row.decision_id)) in selected_index
                    for row in problem_rows.itertuples(index=False)
                ]
            ]
        )
        num_rounds = int(result.iloc[0]["num_speculation_rounds"])
        rounds = []
        for round_id in range(num_rounds):
            round_rows = problem_rows[
                problem_rows.round_id.astype(int) == round_id
            ].sort_values("decision_id")
            actions = [str(value) for value in round_rows["action"].tolist()]
            if any(action not in {STOP, CONTINUE} for action in actions):
                raise ValueError(f"invalid C6 action for problem {problem_id}")
            rounds.append({
                "round_id": round_id,
                "actions": actions,
                "probe_decision_ids": [
                    int(value)
                    for value in round_rows["decision_id"].tolist()
                    if (round_id, int(value)) in selected_index
                ],
            })
        policies[str(problem_id)] = rounds
    selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else pd.DataFrame(columns=decisions.columns)
    )
    return {"version": "c6_behavior_path_probe_v1", "policies": policies}, selected


def write_single_problem_policy(policy, problem_id, destination):
    payload = {
        "version": policy["version"],
        "policies": {str(problem_id): policy["policies"][str(problem_id)]},
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def probe_command(args, dataset, problem_id, output_dir, profile_path, policy_path):
    command = base_command(
        command_args(args), dataset, [problem_id], output_dir, 0
    )
    command[command.index("--max_new_tokens") + 1] = str(args.max_new_tokens)
    command.extend([
        "--strict_greedy_local_oracle",
        "--strict_greedy_capacity_collector",
        "--strict_greedy_verifier_profile", str(profile_path),
        "--strict_greedy_behavior_policy", str(policy_path),
        "--strict_greedy_epsilon_ms", str(args.epsilon_ms),
    ])
    return command


def probe_complete(directory, problem_id):
    result_path = directory / "benchmark_results.csv"
    decisions_path = directory / "greedy_local_oracle_decisions.csv"
    if not result_path.exists() or not decisions_path.exists():
        return False
    try:
        results = pd.read_csv(result_path)
        decisions = pd.read_csv(decisions_path)
    except (OSError, pd.errors.EmptyDataError):
        return False
    return (
        set(results.problem_id.astype(int)) == {int(problem_id)}
        and not decisions.empty
    )


def run_probes(args, dataset, problem_ids, policy, profile_path):
    directories = []
    policy_root = Path(args.output_dir) / dataset / "behavior_policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    for index, problem_id in enumerate(problem_ids):
        directory = (
            Path(args.output_dir)
            / "raw" / dataset / "paired_probe" / f"id_{problem_id}"
        )
        policy_path = policy_root / f"id_{problem_id}.json"
        write_single_problem_policy(policy, problem_id, policy_path)
        if args.resume and probe_complete(directory, problem_id):
            print(f"SKIP completed probe: {dataset} id={problem_id}", flush=True)
            directories.append(directory)
            continue
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        print("\n" + "=" * 96, flush=True)
        print(
            f"PAIRED PROBE {dataset.upper()} | {index + 1}/{len(problem_ids)} "
            f"| problem_id={problem_id}",
            flush=True,
        )
        print("=" * 96, flush=True)
        run_streaming(
            probe_command(
                args, dataset, problem_id, directory, profile_path, policy_path
            )
        )
        if not probe_complete(directory, problem_id):
            raise RuntimeError(f"incomplete paired probe: {directory}")
        directories.append(directory)
    return directories


def concat_csv(paths):
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = pd.Series(np.asarray(scores, dtype=float))
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if not n_pos or not n_neg:
        return math.nan
    ranks = scores.rank(method="average").to_numpy()
    return float(
        (ranks[positives].sum() - n_pos * (n_pos + 1) / 2)
        / (n_pos * n_neg)
    )


def classification_metrics(labels, scores, advantage):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    predicted = scores > 0.0
    actual = labels == 1
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    stop_recall = tp / max(1, tp + fn)
    continue_recall = tn / max(1, tn + fp)
    wrong = predicted != actual
    return {
        "states": int(len(labels)),
        "stop_states": int(actual.sum()),
        "continue_states": int((~actual).sum()),
        "tp_stop": tp,
        "fp_stop": fp,
        "fn_stop": fn,
        "tn_continue": tn,
        "sign_accuracy": float((predicted == actual).mean()),
        "balanced_accuracy": 0.5 * (stop_recall + continue_recall),
        "stop_recall": stop_recall,
        "continue_recall": continue_recall,
        "auc": rank_auc(labels, scores),
        "pearson_advantage": float(
            pd.Series(scores).corr(pd.Series(advantage), method="pearson")
        ),
        "spearman_advantage": float(
            pd.Series(scores).corr(pd.Series(advantage), method="spearman")
        ),
        "mean_regret_tokens": float(np.where(wrong, np.abs(advantage), 0.0).mean()),
    }


def fit_ridge(train, feature_names, ridge=1.0):
    x = train[list(feature_names)].to_numpy(dtype=float)
    y = train["oracle_advantage_tokens"].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    bias_index = feature_names.index("bias") if "bias" in feature_names else None
    scales[scales < 1e-12] = 1.0
    if bias_index is not None:
        means[bias_index] = 0.0
        scales[bias_index] = 1.0
    z = (x - means) / scales
    penalty = np.eye(z.shape[1]) * float(ridge)
    if bias_index is not None:
        penalty[bias_index, bias_index] = 0.0
    weights = np.linalg.pinv(z.T @ z + penalty) @ z.T @ y
    return means, scales, weights


def predict_ridge(frame, feature_names, model):
    means, scales, weights = model
    x = frame[list(feature_names)].to_numpy(dtype=float)
    return ((x - means) / scales) @ weights


def grouped_predictions(frame, feature_names):
    problem_ids = sorted(frame.problem_id.astype(int).unique())
    folds = min(5, len(problem_ids))
    predictions = np.full(len(frame), np.nan)
    problem_to_fold = {
        problem_id: index % folds for index, problem_id in enumerate(problem_ids)
    }
    for fold in range(folds):
        test_mask = frame.problem_id.astype(int).map(problem_to_fold).eq(fold).to_numpy()
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        model = fit_ridge(train, feature_names)
        predictions[test_mask] = predict_ridge(test, feature_names, model)
    return predictions


def evaluate_capacity(frame, feature_names, label):
    predictions = grouped_predictions(frame, list(feature_names))
    metrics = classification_metrics(
        frame["oracle_stop"].to_numpy(),
        predictions,
        frame["oracle_advantage_tokens"].to_numpy(),
    )
    return {"model": label, "features": json.dumps(list(feature_names)), **metrics}


def prepare_paired_states(args, dataset, behavior_dir, selected, probe_dirs):
    probe = concat_csv(
        [directory / "greedy_local_oracle_decisions.csv" for directory in probe_dirs]
    ).rename(columns={"sample_id": "problem_id"})
    key = ["problem_id", "round_id", "decision_id"]
    selected = selected.copy()
    for column in key:
        selected[column] = pd.to_numeric(selected[column]).astype(int)
        probe[column] = pd.to_numeric(probe[column]).astype(int)
    paired = selected.merge(
        probe,
        on=key,
        how="inner",
        suffixes=("_c6", "_probe"),
        validate="one_to_one",
    )
    if len(paired) != len(selected):
        raise RuntimeError(
            f"paired state coverage mismatch for {dataset}: "
            f"{len(paired)}/{len(selected)}"
        )
    for left, right in (
        ("draft_length", "accumulated_proposal_length"),
        ("step", "refinement_step"),
    ):
        if not np.array_equal(
            pd.to_numeric(paired[left]).astype(int).to_numpy(),
            pd.to_numeric(paired[right]).astype(int).to_numpy(),
        ):
            raise RuntimeError(f"state replay mismatch: {left} != {right}")
    rho = pd.to_numeric(paired["rho_tokens_per_ms"]).astype(float)
    fallback_rho = (
        pd.to_numeric(paired["rho_profile_tokens_per_ms"]).astype(float)
    )
    paired["rho_used_tokens_per_ms"] = rho.where(rho > 0.0, fallback_rho)
    paired["G_stop_tokens"] = (
        pd.to_numeric(paired["stop_Y"]).astype(float)
        - paired["rho_used_tokens_per_ms"]
        * pd.to_numeric(paired["stop_local_cost_ms"]).astype(float)
    )
    paired["G_continue_tokens"] = (
        pd.to_numeric(paired["continue_Y"]).astype(float)
        - paired["rho_used_tokens_per_ms"]
        * pd.to_numeric(paired["continue_local_cost_ms"]).astype(float)
    )
    paired["oracle_advantage_tokens"] = (
        paired["G_stop_tokens"] - paired["G_continue_tokens"]
    )
    paired["tie"] = (
        paired["oracle_advantage_tokens"].abs()
        <= paired["rho_used_tokens_per_ms"] * float(args.epsilon_ms)
    )
    paired["oracle_stop"] = paired["oracle_advantage_tokens"] > 0.0
    paired["c6_advantage"] = (
        pd.to_numeric(paired["q_stop_mean"]).astype(float)
        - pd.to_numeric(paired["q_continue_mean"]).astype(float)
    )
    paired["c6_sign_correct"] = (
        (paired["c6_advantage"] > 0.0) == paired["oracle_stop"]
    )
    paired["dataset"] = dataset

    behavior_results = pd.read_csv(Path(behavior_dir) / "benchmark_results.csv")
    probe_results = concat_csv(
        [directory / "benchmark_results.csv" for directory in probe_dirs]
    )
    output_check = behavior_results[["problem_id", "output_token_hash"]].merge(
        probe_results[["problem_id", "output_token_hash"]],
        on="problem_id",
        suffixes=("_behavior", "_probe"),
        validate="one_to_one",
    )
    output_check["output_match"] = (
        output_check.output_token_hash_behavior
        == output_check.output_token_hash_probe
    )
    output_check.to_csv(
        Path(args.output_dir) / dataset / "behavior_replay_output_check.csv",
        index=False,
    )
    if not bool(output_check.output_match.all()):
        raise RuntimeError(f"behavior replay changed target output for {dataset}")
    return paired


def summarize_dataset(args, dataset, paired):
    directory = Path(args.output_dir) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    paired.to_csv(directory / "paired_boundary_states.csv", index=False)
    evaluated = paired.loc[~paired.tie].reset_index(drop=True)
    if evaluated.empty:
        raise RuntimeError(f"all paired states are ties for {dataset}")
    policy = {
        "dataset": dataset,
        "model": "current_c6_shared_va",
        **classification_metrics(
            evaluated.oracle_stop.to_numpy(),
            evaluated.c6_advantage.to_numpy(),
            evaluated.oracle_advantage_tokens.to_numpy(),
        ),
        "sampled_states": int(len(paired)),
        "ties": int(paired.tie.sum()),
        "tie_rate_percent": 100.0 * float(paired.tie.mean()),
    }
    feature_names = list(V22_COMPACT_FEATURE_NAMES)
    capacity_rows = [evaluate_capacity(evaluated, feature_names, "compact6")]
    for removed in feature_names[1:]:
        kept = [name for name in feature_names if name != removed]
        row = evaluate_capacity(evaluated, kept, f"compact6_minus_{removed}")
        row["removed_feature"] = removed
        capacity_rows.append(row)
    capacity = pd.DataFrame(capacity_rows)
    capacity.insert(0, "dataset", dataset)
    capacity.to_csv(directory / "feature_capacity_ablation.csv", index=False)
    diagnostics = pd.DataFrame({
        "feature": feature_names,
        "mean": [float(evaluated[name].mean()) for name in feature_names],
        "std": [float(evaluated[name].std(ddof=0)) for name in feature_names],
        "correlation_with_oracle_advantage": [
            float(evaluated[name].corr(evaluated.oracle_advantage_tokens))
            for name in feature_names
        ],
    })
    diagnostics.insert(0, "dataset", dataset)
    diagnostics.to_csv(directory / "feature_diagnostics.csv", index=False)
    pd.DataFrame([policy]).to_csv(directory / "current_policy_alignment.csv", index=False)
    return policy, capacity, evaluated


def cross_dataset_capacity(evaluated_by_dataset):
    rows = []
    feature_names = list(V22_COMPACT_FEATURE_NAMES)
    for train_name, test_name in (("math", "gsm8k"), ("gsm8k", "math")):
        train = evaluated_by_dataset[train_name]
        test = evaluated_by_dataset[test_name]
        model = fit_ridge(train, feature_names)
        predictions = predict_ridge(test, feature_names, model)
        rows.append({
            "train_dataset": train_name,
            "test_dataset": test_name,
            **classification_metrics(
                test.oracle_stop.to_numpy(),
                predictions,
                test.oracle_advantage_tokens.to_numpy(),
            ),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    validate_args(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    policy_rows = []
    capacity_frames = []
    evaluated_by_dataset = {}
    manifest_ids = {}
    for dataset in DATASETS:
        problem_ids = fixed_ids(dataset, args.num_questions)
        manifest_ids[dataset] = problem_ids
        behavior_dir = run_behavior(args, dataset, problem_ids)
        decisions = pd.read_csv(behavior_dir / "adaptive_td_decisions.csv")
        results = pd.read_csv(behavior_dir / "benchmark_results.csv")
        policy, selected = build_behavior_policy(
            decisions, results, args.max_states_per_problem
        )
        dataset_dir = root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "behavior_policy.json").write_text(
            json.dumps(policy, indent=2), encoding="utf-8"
        )
        selected.to_csv(dataset_dir / "selected_c6_states.csv", index=False)
        profile_path = dataset_dir / "verifier_profile.json"
        build_verifier_profile(behavior_dir, profile_path)
        probe_dirs = run_probes(
            args, dataset, problem_ids, policy, profile_path
        )
        paired = prepare_paired_states(
            args, dataset, behavior_dir, selected, probe_dirs
        )
        policy_row, capacity, evaluated = summarize_dataset(args, dataset, paired)
        policy_rows.append(policy_row)
        capacity_frames.append(capacity)
        evaluated_by_dataset[dataset] = evaluated

    pd.DataFrame(policy_rows).to_csv(
        root / "current_policy_alignment_summary.csv", index=False
    )
    pd.concat(capacity_frames, ignore_index=True).to_csv(
        root / "feature_capacity_ablation_summary.csv", index=False
    )
    cross = cross_dataset_capacity(evaluated_by_dataset)
    cross.to_csv(root / "cross_dataset_feature_capacity.csv", index=False)
    manifest = {
        "version": "paired_boundary_compact6_feature_probe_v1",
        "problem_ids": manifest_ids,
        "arguments": vars(args),
        "label": "A*=G_stop-G_continue; G=emitted_tokens-rho_at_state*measured_boundary_latency",
        "tie": "abs(A*) <= rho_at_state * epsilon_ms",
        "support": (
            "States visited by the actual annealed Compact6 Shared V+A behavior; "
            "both actions replay exact drafting/outer-FailFast flow to the next "
            "real greedy verifier boundary."
        ),
        "scope": "local verifier-boundary advantage, not an end-to-end oracle",
        "quantization_note": (
            "The default server run uses the existing BitsAndBytes INT8 backend. "
            "Both paired branches use the same verifier, but labels are specific "
            "to this quantized backend and are not claimed to equal FP16 labels."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nCURRENT POLICY ALIGNMENT", flush=True)
    print(pd.DataFrame(policy_rows).to_string(index=False), flush=True)
    print("\nCROSS-DATASET FEATURE CAPACITY", flush=True)
    print(cross.to_string(index=False), flush=True)
    if not args.skip_archive:
        archive = shutil.make_archive(str(root), "zip", root_dir=root)
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
