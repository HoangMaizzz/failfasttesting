import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
DATASET_COUNTS = {
    "math": 50,
    "gsm8k": 50,
    "humaneval": 50,
    "aime": 25,
}
SUMMARY_FILE = "global_oracle_problem_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exact end-to-end STOP/CONTINUE oracle over the legal FailFast "
            "action graph. No learned feature, annealing policy, or rho is used."
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Problems per model-loading batch. The default favors robust resume "
            "for this long-running exact search."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--drafter_threshold", type=float, default=0.30)
    parser.add_argument("--lowconf_threshold", type=float, default=0.50)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument(
        "--target_quantization",
        choices=("none", "int8", "int4"),
        default="int8",
    )
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_perfect_stop_oracle_175"
        ),
    )
    parser.add_argument(
        "--global_oracle_max_states",
        type=int,
        default=0,
        help="Zero means exact enumeration without an approximation cutoff.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.global_oracle_max_states < 0:
        raise ValueError("--global_oracle_max_states must be non-negative")
    if args.target_device < 0 or args.drafter_device < 0:
        raise ValueError("CUDA device indices must be non-negative")
    if args.drafter_threshold != 0.30 or args.lowconf_threshold != 0.50:
        raise ValueError("the exact oracle fixes tau_D=0.30 and tau_F=0.50")
    for dataset, count in DATASET_COUNTS.items():
        if len(PROBLEM_IDS[dataset]) < count:
            raise ValueError(f"{dataset} does not provide {count} matched IDs")


def selected_problem_ids():
    return {
        dataset: list(PROBLEM_IDS[dataset][:count])
        for dataset, count in DATASET_COUNTS.items()
    }


def batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield start // batch_size, values[start:start + batch_size]


def batch_output_dir(args, dataset, batch_index):
    return Path(args.output_dir) / "raw" / dataset / f"batch_{batch_index:03d}"


def command_for(args, dataset, problem_ids, output_dir):
    return [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(value) for value in problem_ids],
        "--warmup_questions", "0",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--frontier_stop_mode", "disabled",
        "--disable_reusing_drafter_kvs",
        "--global_oracle_graph",
        "--global_oracle_max_states", str(args.global_oracle_max_states),
        "--global_oracle_log_interval", "25",
        "--global_oracle_epsilon_cost_ms", "1.0",
        "--seed", "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]


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
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def completed_batch(output_dir, expected_ids):
    summary_path = output_dir / SUMMARY_FILE
    if not summary_path.exists() or not summary_path.stat().st_size:
        return False
    summary = pd.read_csv(summary_path)
    observed = set(pd.to_numeric(summary["problem_id"]).astype(int))
    return observed == set(expected_ids)


def collect_report_frames(args):
    summaries = []
    states = []
    selected_ids = selected_problem_ids()
    for dataset, problem_ids in selected_ids.items():
        for batch_index, batch_ids in batches(problem_ids, args.batch_size):
            output_dir = batch_output_dir(args, dataset, batch_index)
            summary = pd.read_csv(output_dir / SUMMARY_FILE)
            summary.insert(0, "dataset", dataset)
            summary.insert(1, "batch_index", batch_index)
            summaries.append(summary)
            state_path = output_dir / "global_oracle_nodes.csv"
            if state_path.exists() and state_path.stat().st_size:
                state_rows = pd.read_csv(state_path)
                state_rows.insert(0, "dataset", dataset)
                state_rows.insert(1, "batch_index", batch_index)
                states.append(state_rows)
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
    )


def aggregate_datasets(problem_summary):
    rows = []
    for dataset, frame in problem_summary.groupby("dataset", sort=False):
        generated = float(frame["generated_tokens"].sum())
        baseline_ms = float(frame["baseline_total_latency_ms"].sum())
        oracle_ms = float(frame["oracle_optimal_latency_ms"].sum())
        modeled_baseline_ms = float(
            frame["modeled_baseline_total_latency_ms"].sum()
        )
        modeled_oracle_ms = float(
            frame["modeled_oracle_optimal_latency_ms"].sum()
        )
        rows.append({
            "dataset": dataset,
            "num_questions": int(frame["problem_id"].nunique()),
            "generated_tokens": int(generated),
            "failfast_replay_total_s": baseline_ms / 1000.0,
            "oracle_replay_total_s": oracle_ms / 1000.0,
            "failfast_replay_ms_per_token": baseline_ms / max(1.0, generated),
            "oracle_replay_ms_per_token": oracle_ms / max(1.0, generated),
            "oracle_replay_speedup_vs_failfast": baseline_ms / max(1e-9, oracle_ms),
            "oracle_replay_headroom_percent": 100.0 * (
                1.0 - oracle_ms / max(1e-9, baseline_ms)
            ),
            "modeled_failfast_ms_per_token": (
                modeled_baseline_ms / max(1.0, generated)
            ),
            "modeled_oracle_ms_per_token": (
                modeled_oracle_ms / max(1.0, generated)
            ),
            "modeled_oracle_speedup_vs_failfast": (
                modeled_baseline_ms / max(1e-9, modeled_oracle_ms)
            ),
            "modeled_oracle_headroom_percent": 100.0 * (
                1.0 - modeled_oracle_ms / max(1e-9, modeled_baseline_ms)
            ),
            "failfast_dllm_forwards": int(frame["baseline_dllm_forwards"].sum()),
            "oracle_dllm_forwards": int(frame["oracle_dllm_forwards"].sum()),
            "failfast_verifier_calls": int(frame["baseline_verifier_calls"].sum()),
            "oracle_verifier_calls": int(frame["oracle_verifier_calls"].sum()),
            "profiling_wall_time_hours": float(
                frame["oracle_total_profiling_wall_time_ms"].sum()
            ) / 3_600_000.0,
            "replay_oracle_not_slower_rate_percent": 100.0 * float(
                frame["replay_oracle_not_slower_than_failfast"].mean()
            ),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = selected_problem_ids()
    started = time.time()
    total_batches = sum(
        (len(values) + args.batch_size - 1) // args.batch_size
        for values in selected_ids.values()
    )
    completed = 0

    for dataset, problem_ids in selected_ids.items():
        for batch_index, batch_ids in batches(problem_ids, args.batch_size):
            completed += 1
            phase_dir = batch_output_dir(args, dataset, batch_index)
            if args.resume and completed_batch(phase_dir, batch_ids):
                print(
                    f"SKIP {completed}/{total_batches} | {dataset} | "
                    f"batch={batch_index} | complete",
                    flush=True,
                )
                continue
            if phase_dir.exists():
                shutil.rmtree(phase_dir)
            print("\n" + "=" * 100, flush=True)
            print(
                f"RUN {completed}/{total_batches} | {dataset.upper()} | "
                f"batch={batch_index} | problem_ids={batch_ids}",
                flush=True,
            )
            print("=" * 100, flush=True)
            run_streaming(command_for(args, dataset, batch_ids, phase_dir))
            if not completed_batch(phase_dir, batch_ids):
                raise RuntimeError(
                    f"batch did not produce all expected summaries: {phase_dir}"
                )

    problem_summary, state_labels = collect_report_frames(args)
    dataset_summary = aggregate_datasets(problem_summary)
    problem_summary.to_csv(
        output_dir / "perfect_stop_oracle_problem_summary.csv",
        index=False,
    )
    dataset_summary.to_csv(
        output_dir / "perfect_stop_oracle_dataset_summary.csv",
        index=False,
    )
    state_labels.to_csv(
        output_dir / "perfect_stop_oracle_state_labels.csv",
        index=False,
    )
    manifest = {
        "oracle": "exact_global_stop_continue_under_failfast",
        "objective": "minimum measured algorithm latency to terminal",
        "datasets": DATASET_COUNTS,
        "selected_problem_ids": selected_ids,
        "contains_prior_25_ids": {
            dataset: selected_ids[dataset][:25] == PROBLEM_IDS[dataset][:25]
            for dataset in DATASET_COUNTS
        },
        "arguments": vars(args),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    archive = None
    if not args.skip_archive:
        archive = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    print("\nPERFECT STOP ORACLE DATASET SUMMARY", flush=True)
    print(dataset_summary.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    if archive:
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
