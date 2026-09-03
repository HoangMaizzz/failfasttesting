import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k", "humaneval")
METHODS = ("failfast", "logistic_f2_e2", "u1_single", "u1_batch1x")
KNOWN_UNRUNNABLE_IDS = {"math": {301}, "gsm8k": set(), "humaneval": set()}
REPLACEMENT_IDS = {"math": (489,), "gsm8k": (), "humaneval": ()}
DATASET_SIZES = {"math": 500, "gsm8k": 1319, "humaneval": 164}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["failfast", "u1_batch1x"])
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--id_offset", type=int, default=25)
    parser.add_argument("--target_quantization", default="int8")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument("--drafter_threshold", type=float, default=0.50)
    parser.add_argument("--lowconf_threshold", type=float, default=0.70)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_u1_batch1x_fixed05_tauD0p50_tauF0p70"
        ),
    )
    return parser.parse_args()


def run(command):
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


def selected_ids(args, dataset):
    start = args.id_offset
    candidates = list(PROBLEM_IDS[dataset][start:]) + list(REPLACEMENT_IDS[dataset])
    used = set(candidates)
    supplemental = [
        problem_id for problem_id in range(DATASET_SIZES[dataset])
        if problem_id not in used
        and problem_id not in KNOWN_UNRUNNABLE_IDS[dataset]
    ]
    random.Random(42000 + sum(map(ord, dataset))).shuffle(supplemental)
    candidates.extend(supplemental)
    ids = [
        problem_id for problem_id in candidates
        if problem_id not in KNOWN_UNRUNNABLE_IDS[dataset]
    ][:args.num_questions]
    if len(ids) != args.num_questions:
        raise ValueError(f"{dataset} provides only {len(ids)} usable new IDs")
    return ids


def command(args, dataset, method, destination):
    ids = selected_ids(args, dataset)
    result = [
        sys.executable, "-u", "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(ids)),
        "--problem_ids", *map(str, ids),
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--seed", "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(destination),
        "--log_level", "INFO",
    ]
    if method != "failfast":
        result.extend([
            "--adaptive-td",
            "--adaptive-feature-schema", "otrc_v2_2_compact_td",
            "--adaptive-credit-assignment", "hindsight_delta_j_logistic_f2",
            "--adaptive-policy-mode", "hindsight_delta_j_logistic_f2",
            "--adaptive-hindsight-logistic-learning-rate", "0.05",
            "--adaptive-hindsight-logistic-continue-threshold", "0.5",
            "--adaptive-hindsight-logistic-tie-ms-per-token", "1.0",
            "--adaptive-hindsight-logistic-min-positive-problems", "2",
            "--adaptive-hindsight-delta-j-class-balance-alpha", "5.0",
            "--adaptive-hindsight-delta-j-max-continue-weight", "3.0",
            "--adaptive-hindsight-delta-j-min-pairs", "30",
            "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
            "--adaptive-hindsight-delta-j-structural-probe", "0.08",
            "--adaptive-hindsight-delta-j-floor-probe", "0.02",
            "--adaptive-log-decisions",
            "--adaptive-profile-overhead",
        ])
        if method == "logistic_f2_e2":
            result.extend([
                "--adaptive-hindsight-logistic-utility-weighting", "legacy",
                "--adaptive-hindsight-logistic-replay-batch-size", "0",
                "--adaptive-hindsight-logistic-use-class-weight",
            ])
        elif method == "u1_single":
            result.extend([
                "--adaptive-hindsight-logistic-utility-weighting", "raw_abs",
                "--adaptive-hindsight-logistic-replay-batch-size", "0",
                "--no-adaptive-hindsight-logistic-use-class-weight",
            ])
        elif method == "u1_batch1x":
            result.extend([
                "--adaptive-hindsight-logistic-utility-weighting", "raw_abs",
                "--adaptive-hindsight-logistic-replay-batch-size", "16",
                "--adaptive-hindsight-logistic-replay-buffer-size", "100",
                "--no-adaptive-hindsight-logistic-use-class-weight",
            ])
    return result


def auc(labels, scores):
    frame = pd.DataFrame({"label": labels, "score": scores}).dropna()
    positives = int(frame.label.sum())
    negatives = len(frame) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = frame.score.rank(method="average")
    return float((ranks[frame.label == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def summarize(dataset, method, destination):
    benchmark = pd.read_csv(destination / "benchmark_results.csv")
    tokens = float(benchmark.output_tokens.sum())
    row = {
        "dataset": dataset,
        "method": method,
        "questions": int(benchmark.problem_id.nunique()),
        "output_tokens": int(tokens),
        "algorithm_time_s": float(benchmark.actual_algorithm_time.sum()),
        "ms_per_output_token": 1000.0 * benchmark.actual_algorithm_time.sum() / max(1.0, tokens),
        "draft_time_s": float(benchmark.actual_draft_time.sum()),
        "verify_time_s": float(benchmark.actual_verify_time.sum()),
        "draft_forwards": int(benchmark.total_num_forward_passes.sum()),
        "verifier_rounds": int(benchmark.num_speculation_rounds.sum()),
    }
    if method == "failfast":
        return row
    transitions_path = destination / "adaptive_full_stream_transitions.csv"
    transitions = pd.read_csv(transitions_path) if transitions_path.exists() else pd.DataFrame()
    decisions = pd.read_csv(destination / "adaptive_td_decisions.csv")
    source = decisions.get("action_source", pd.Series(dtype=str)).fillna("")
    runtime = json.loads(
        (destination / "adaptive_td_runtime_state.json").read_text(encoding="utf-8")
    )["hindsight_block_gain"]
    row.update({
        "decisions": len(decisions),
        "learned_stop": int((source == "learned_stop").sum()),
        "learned_continue": int((source == "learned_continue").sum()),
        "structural_probes": int((source == "structural_probe").sum()),
        "floor_probes": int((source == "floor_probe").sum()),
        "resolved_pairs": int(runtime["resolved_count"]),
        "tie_pairs": int(runtime["tie_count"]),
        "censored_pairs": int(runtime["censored_count"]),
    })
    if not transitions.empty:
        update_applied = transitions.update_applied.map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
        labeled = transitions[update_applied].copy()
        y = pd.to_numeric(labeled.binary_label_C, errors="coerce")
        score = pd.to_numeric(labeled.continue_score_before_update, errors="coerce")
        predicted = score > pd.to_numeric(labeled.continue_threshold, errors="coerce")
        truth = y.astype(bool)
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        tn = int((~predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        continue_recall = tp / max(1, tp + fn)
        stop_recall = tn / max(1, tn + fp)
        propensity = pd.to_numeric(
            labeled.behavior_continue_probability, errors="coerce"
        ).clip(lower=1e-6)
        ips_weight = 1.0 / propensity
        correct = (predicted == truth).astype(float)
        row.update({
            "temporal_auc": auc(y, score),
            "accuracy_percent": 100.0 * float(correct.mean()),
            "balanced_accuracy_percent": 50.0 * (continue_recall + stop_recall),
            "continue_precision_percent": 100.0 * tp / max(1, tp + fp),
            "continue_recall_percent": 100.0 * continue_recall,
            "stop_recall_percent": 100.0 * stop_recall,
            "TP_continue": tp,
            "FP_continue": fp,
            "TN_stop": tn,
            "FN_continue": fn,
            "snips_accuracy_percent": 100.0 * float(
                (correct * ips_weight).sum() / max(1e-9, ips_weight.sum())
            ),
            "max_inverse_propensity": float(ips_weight.max()),
        })
        learned_c = labeled[
            (labeled.model_action == "continue")
            & (labeled.action_source == "learned_continue")
        ]
        row["sum_delta_j_learned_continue"] = float(learned_c.delta_J_ms_per_token.sum())
        row["mean_delta_j_learned_continue"] = (
            float(learned_c.delta_J_ms_per_token.mean()) if len(learned_c) else float("nan")
        )
    return row


def main():
    args = parse_args()
    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    (root / "raw").mkdir(parents=True)
    run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir])
    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            destination = root / "raw" / dataset / method
            destination.mkdir(parents=True)
            print(f"\n{'=' * 88}\nRUN {dataset.upper()} | {method}\n{'=' * 88}")
            run(command(args, dataset, method, destination))
            rows.append(summarize(dataset, method, destination))
    summary = pd.DataFrame(rows)
    baseline = summary[summary.method == "failfast"][["dataset", "ms_per_output_token"]]
    if not baseline.empty:
        baseline = baseline.rename(
            columns={"ms_per_output_token": "failfast_ms_per_output_token"}
        )
        summary = summary.merge(baseline, on="dataset", how="left")
        summary["speedup_vs_failfast"] = (
            summary.failfast_ms_per_output_token / summary.ms_per_output_token
        )
    summary.to_csv(root / "dataset_method_summary.csv", index=False)
    manifest = {
        "datasets": list(args.datasets),
        "methods": list(args.methods),
        "problem_ids": {dataset: selected_ids(args, dataset) for dataset in args.datasets},
        "id_offset": args.id_offset,
        "excluded_known_unrunnable_ids": {
            dataset: sorted(KNOWN_UNRUNNABLE_IDS[dataset]) for dataset in DATASETS
        },
        "tau_d": args.drafter_threshold,
        "tau_f": args.lowconf_threshold,
        "tie_ms_per_token": 1.0,
        "continue_threshold": 0.5,
        "fixed_continue_threshold": 0.5,
        "E2": "legacy clipped normalized utility plus CONTINUE class weighting",
        "U1_single": "raw abs(delta_J_ms_per_token), no class weighting, current-pair SGD",
        "U1_batch1x": "U1 objective + one uniform minibatch SGD update per resolved pair; batch=16, buffer=100",
        "replay_rng": "dedicated seed+7919 so minibatch sampling does not perturb probe RNG",
        "IPS": "logged only; not used for learner updates",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(root), "zip", root.parent, root.name)
    print("\nDATASET/METHOD SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nARCHIVE: {archive}")


if __name__ == "__main__":
    main()
