import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_quantization", default="int8")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_hindsight_delta_j_f5_math_gsm8k_test25"
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


def command(args, dataset, output_dir):
    ids = PROBLEM_IDS[dataset][: args.num_questions]
    return [
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
        "--drafter_thresholds", "0.30",
        "--sweep_lowconf_threshold", "0.50",
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--adaptive-td",
        "--adaptive-feature-schema", "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment", "hindsight_delta_j_f5",
        "--adaptive-policy-mode", "hindsight_delta_j_f5",
        "--adaptive-hindsight-delta-j-p-continue-threshold", "0.65",
        "--adaptive-hindsight-delta-j-class-balance-alpha", "5.0",
        "--adaptive-hindsight-delta-j-max-continue-weight", "3.0",
        "--adaptive-hindsight-delta-j-calibration-beta", "0.05",
        "--adaptive-hindsight-delta-j-min-pairs", "30",
        "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
        "--adaptive-hindsight-delta-j-structural-probe", "0.08",
        "--adaptive-hindsight-delta-j-floor-probe", "0.02",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
        "--seed", "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", "INFO",
    ]


def summarize(dataset, destination, root):
    benchmark = pd.read_csv(destination / "benchmark_results.csv")
    decisions = pd.read_csv(destination / "adaptive_td_decisions.csv")
    transitions_path = destination / "adaptive_full_stream_transitions.csv"
    transitions = pd.read_csv(transitions_path) if transitions_path.exists() else pd.DataFrame()
    runtime = json.loads(
        (destination / "adaptive_td_runtime_state.json").read_text(encoding="utf-8")
    )["hindsight_block_gain"]
    action_source = decisions.get("action_source", pd.Series(dtype=str)).fillna("")
    row = {
        "dataset": dataset,
        "questions": int(benchmark.problem_id.nunique()),
        "output_tokens": int(benchmark.output_tokens.sum()),
        "algorithm_time_s": float(benchmark.actual_algorithm_time.sum()),
        "ms_per_output_token": 1000.0 * benchmark.actual_algorithm_time.sum()
        / max(1, benchmark.output_tokens.sum()),
        "decisions": len(decisions),
        "learned_stop": int((action_source == "learned_stop").sum()),
        "learned_continue": int((action_source == "learned_continue").sum()),
        "cold_start_continue": int((action_source == "failfast_cold_start").sum()),
        "structural_probes": int((action_source == "structural_probe").sum()),
        "floor_probes": int((action_source == "floor_probe").sum()),
        "resolved_pairs": int(runtime["resolved_count"]),
        "censored_pairs": int(runtime["censored_count"]),
        "beneficial_continue_pairs": int(runtime["beneficial_continue_count"]),
        "stop_better_pairs": int(runtime["stop_better_count"]),
        "calibration_bias": float(runtime["calibration_bias"]),
        "normalized_training_mae": float(runtime["model"]["normalized_mae"]),
    }
    dynamics = []
    if not transitions.empty:
        transitions.to_csv(root / f"{dataset}_delta_j_pairs.csv", index=False)
        row.update({
            "mean_delta_j": float(transitions.delta_J_ms_per_token.mean()),
            "continue_rate_percent": 100.0 * float(
                (transitions.true_action_from_delta_J == "continue").mean()
            ),
            "cost_aware_accuracy_percent": 100.0 * float(
                transitions.cost_aware_correct.mean()
            ),
            "raw_prediction_mae": float(
                (transitions.normalized_delta_J - transitions.predicted_r_before_update)
                .abs().mean()
            ),
            "calibrated_prediction_mae": float(
                (
                    transitions.normalized_delta_J
                    - transitions.calibrated_predicted_r_before_update
                ).abs().mean()
            ),
        })
        transitions = transitions.reset_index(drop=True)
        transitions["quartile"] = pd.qcut(
            range(len(transitions)), q=min(4, len(transitions)),
            labels=False, duplicates="drop",
        ) + 1
        for quartile, frame in transitions.groupby("quartile"):
            dynamics.append({
                "dataset": dataset,
                "quartile": int(quartile),
                "pairs": len(frame),
                "mean_true_r": float(frame.normalized_delta_J.mean()),
                "mean_predicted_r": float(frame.predicted_r_before_update.mean()),
                "mae": float(
                    (frame.normalized_delta_J - frame.predicted_r_before_update)
                    .abs().mean()
                ),
                "continue_rate_percent": 100.0 * float(
                    (frame.true_action_from_delta_J == "continue").mean()
                ),
            })
    return row, dynamics


def main():
    args = parse_args()
    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    (root / "raw").mkdir(parents=True)
    run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir])
    summaries, dynamics = [], []
    for dataset in DATASETS:
        destination = root / "raw" / dataset / "hindsight_delta_j_f5"
        destination.mkdir(parents=True)
        print(f"\n{'=' * 88}\nHINDSIGHT DELTA-J F5 {dataset.upper()}\n{'=' * 88}")
        run(command(args, dataset, destination))
        summary, rows = summarize(dataset, destination, root)
        summaries.append(summary)
        dynamics.extend(rows)
    summary = pd.DataFrame(summaries)
    learning = pd.DataFrame(dynamics)
    summary.to_csv(root / "dataset_summary.csv", index=False)
    learning.to_csv(root / "learning_dynamics.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({
        "method": "hindsight_delta_j_f5",
        "datasets": list(DATASETS),
        "problem_ids": {
            dataset: PROBLEM_IDS[dataset][: args.num_questions]
            for dataset in DATASETS
        },
        "target": "normalized local candidate-prefix delta-J",
        "shared_T_B": "first factual verifier latency + post-verify latency",
        "tau_d": 0.30,
        "tau_f": 0.50,
        "target_quantization": args.target_quantization,
        "extra_verifier_calls": False,
    }, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(root), "zip", root.parent, root.name)
    print("\nDATASET SUMMARY")
    print(summary.to_string(index=False))
    print("\nLEARNING DYNAMICS")
    print(learning.to_string(index=False))
    print(f"\nARCHIVE: {archive}")


if __name__ == "__main__":
    main()
