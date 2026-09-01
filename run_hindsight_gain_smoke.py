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
    parser.add_argument("--num_questions", type=int, default=5)
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
            "outputs_hindsight_block_gain_smoke_math_gsm8k_test5"
        ),
    )
    parser.add_argument("--min_observations", type=int, default=8)
    parser.add_argument("--probe_initial", type=float, default=0.15)
    parser.add_argument("--probe_floor", type=float, default=0.02)
    parser.add_argument("--probe_decay_pairs", type=float, default=32.0)
    parser.add_argument("--probe_max_fraction", type=float, default=0.08)
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
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name",
        dataset,
        "--num_questions",
        str(len(ids)),
        "--problem_ids",
        *map(str, ids),
        "--warmup_questions",
        "0",
        "--benchmark_modes",
        "dllm_ar",
        "--dllm_variant",
        "failfast",
        "--decoding_strategy",
        "greedy",
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--spec_len",
        "8",
        "--block_size",
        "32",
        "--small_block_size",
        "8",
        "--target_model_name",
        "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir",
        args.dllm_dir,
        "--target_device",
        str(args.target_device),
        "--drafter_device",
        str(args.drafter_device),
        "--target_quantization",
        args.target_quantization,
        "--drafter_thresholds",
        "0.30",
        "--sweep_lowconf_threshold",
        "0.50",
        "--sweep_max_spec_len",
        "64",
        "--sweep_incr_len",
        "8",
        "--adaptive-td",
        "--adaptive-feature-schema",
        "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment",
        "hindsight_block_gain",
        "--adaptive-policy-mode",
        "hindsight_gain",
        "--adaptive-early-stop-min-observations",
        str(args.min_observations),
        "--adaptive-hindsight-probe-initial",
        str(args.probe_initial),
        "--adaptive-hindsight-probe-floor",
        str(args.probe_floor),
        "--adaptive-hindsight-probe-decay-pairs",
        str(args.probe_decay_pairs),
        "--adaptive-hindsight-probe-max-fraction",
        str(args.probe_max_fraction),
        "--adaptive-collect-raw-state",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
        "--disable_reusing_drafter_kvs",
        "--seed",
        "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir",
        str(output_dir),
        "--log_level",
        "INFO",
    ]


def summarize_dataset(dataset, output_dir):
    benchmark = pd.read_csv(output_dir / "benchmark_results.csv")
    decisions = pd.read_csv(output_dir / "adaptive_td_decisions.csv")
    transitions_path = output_dir / "adaptive_full_stream_transitions.csv"
    transitions = (
        pd.read_csv(transitions_path)
        if transitions_path.exists()
        else pd.DataFrame()
    )
    runtime = json.loads(
        (output_dir / "adaptive_td_runtime_state.json").read_text(encoding="utf-8")
    )
    model = runtime["hindsight_block_gain"]["model"]
    summary = {
        "dataset": dataset,
        "questions": int(benchmark.problem_id.nunique()),
        "output_tokens": int(benchmark.output_tokens.sum()),
        "algorithm_time_s": float(benchmark.actual_algorithm_time.sum()),
        "ms_per_output_token": (
            1000.0 * benchmark.actual_algorithm_time.sum()
            / max(1, benchmark.output_tokens.sum())
        ),
        "decisions": len(decisions),
        "stop_rate_percent": 100.0 * (decisions.action == "stop").mean(),
        "forced_continue_probes": int(
            (decisions.reason == "hindsight_uncertainty_probe").sum()
        ),
        "learned_continue_actions": int(
            (decisions.reason == "hindsight_gain_exceeds_cost").sum()
        ),
        "exploration_rate_percent": 100.0 * float(
            decisions.exploration_used.fillna(False).astype(bool).mean()
        ),
        "training_pairs": int(model["sample_count"]),
        "censored_pairs": int(runtime["hindsight_block_gain"]["censored_count"]),
        "invalid_snapshots": int(runtime["hindsight_block_gain"]["invalid_count"]),
        "normalized_training_mae": float(model["normalized_mae"]),
        "snapshot_overhead_ema_ms": runtime["hindsight_block_gain"][
            "snapshot_overhead_ema_ms"
        ],
    }
    dynamics = []
    if not transitions.empty:
        transitions = transitions.reset_index(drop=True)
        transitions["sequence"] = range(1, len(transitions) + 1)
        transitions["quartile"] = pd.qcut(
            transitions.sequence,
            q=min(4, len(transitions)),
            labels=False,
            duplicates="drop",
        ) + 1
        summary.update({
            "gain_mean": float(transitions.gain_tokens.mean()),
            "gain_positive_rate_percent": 100.0 * float(
                (transitions.gain_tokens > 0).mean()
            ),
            "prediction_mae_tokens": float(
                transitions.prediction_error_tokens.abs().mean()
            ),
            "prediction_bias_tokens": float(
                -transitions.prediction_error_tokens.mean()
            ),
            "cost_aware_accuracy_percent": 100.0 * float(
                transitions.cost_aware_correct.mean()
            ),
        })
        for quartile, frame in transitions.groupby("quartile"):
            dynamics.append({
                "dataset": dataset,
                "quartile": int(quartile),
                "pairs": len(frame),
                "actual_gain_mean": float(frame.gain_tokens.mean()),
                "predicted_gain_mean": float(
                    frame.predicted_gain_tokens_before_update.mean()
                ),
                "mae_tokens": float(frame.prediction_error_tokens.abs().mean()),
                "cost_aware_accuracy_percent": 100.0
                * float(frame.cost_aware_correct.mean()),
            })
        transitions.to_csv(
            output_dir.parent.parent / f"{dataset}_hindsight_pairs.csv", index=False
        )
    return summary, dynamics


def main():
    args = parse_args()
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    root = Path(args.output_dir)
    if root.exists():
        shutil.rmtree(root)
    raw = root / "raw"
    raw.mkdir(parents=True)
    run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir])
    summaries = []
    dynamics = []
    for dataset in DATASETS:
        destination = raw / dataset / "hindsight_block_gain"
        destination.mkdir(parents=True)
        print(f"\n{'=' * 88}\nHINDSIGHT GAIN {dataset.upper()}\n{'=' * 88}", flush=True)
        run(command(args, dataset, destination))
        summary, rows = summarize_dataset(dataset, destination)
        summaries.append(summary)
        dynamics.extend(rows)
    summary_frame = pd.DataFrame(summaries)
    dynamics_frame = pd.DataFrame(dynamics)
    summary_frame.to_csv(root / "hindsight_smoke_summary.csv", index=False)
    dynamics_frame.to_csv(root / "hindsight_learning_dynamics.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps({
            "datasets": list(DATASETS),
            "problem_ids": {
                name: PROBLEM_IDS[name][: args.num_questions] for name in DATASETS
            },
            "method": "online_bayesian_hindsight_block_gain",
            "label": "delayed committed-greedy LCP, active eight-token block",
            "outer_rewards_used": False,
            "counterfactual_model_calls": False,
            "target_quantization": args.target_quantization,
            "tau_d": 0.30,
            "tau_f": 0.50,
        }, indent=2),
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(root), "zip", root.parent, root.name)
    print("\nHINDSIGHT SMOKE SUMMARY", flush=True)
    print(summary_frame.to_string(index=False), flush=True)
    print("\nLEARNING DYNAMICS", flush=True)
    print(dynamics_frame.to_string(index=False), flush=True)
    print(f"\nARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
