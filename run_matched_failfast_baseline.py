import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS, aggregate_method


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")
METHOD = "failfast_matched"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--drafter_threshold", type=float, default=0.30)
    parser.add_argument("--lowconf_threshold", type=float, default=0.50)
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=1)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_failfast_tauD0p30_tauF0p50_test25",
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
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    available = min(len(PROBLEM_IDS[dataset]) for dataset in args.datasets)
    if args.num_questions > available:
        raise ValueError(f"--num_questions cannot exceed {available}")
    if args.warmup_questions != 1:
        raise ValueError("the matched benchmark requires one warmup question")
    if not 0.0 < args.drafter_threshold <= 1.0:
        raise ValueError("--drafter_threshold must be in (0, 1]")
    if not 0.0 <= args.lowconf_threshold <= 1.0:
        raise ValueError("--lowconf_threshold must be in [0, 1]")
    if args.target_device < 0 or args.drafter_device < 0:
        raise ValueError("CUDA device indices must be non-negative")


def command_for(args, dataset, output_dir):
    problem_ids = PROBLEM_IDS[dataset][: args.num_questions]
    command = [
        sys.executable,
        "-u",
        "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", str(len(problem_ids)),
        "--problem_ids", *[str(value) for value in problem_ids],
        "--warmup_questions", str(args.warmup_questions),
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
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.lowconf_threshold),
        "--sweep_max_spec_len", "60",
        "--sweep_incr_len", "8",
        "--seed", "42",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(output_dir),
        "--log_level", args.log_level,
    ]
    return command


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


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    selected_ids = {}
    started = time.time()

    for dataset in args.datasets:
        phase_dir = output_dir / "raw" / dataset / METHOD
        results_path = phase_dir / "benchmark_results.csv"
        selected_ids[dataset] = PROBLEM_IDS[dataset][: args.num_questions]
        if not (args.resume and results_path.exists()):
            print(f"\nRUN {dataset.upper()} | FailFast | questions={args.num_questions}", flush=True)
            run_streaming(command_for(args, dataset, phase_dir))
        results = pd.read_csv(results_path)
        summary = aggregate_method(results, METHOD)
        summary["dataset"] = dataset
        summaries.append(summary)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    manifest = {
        "method": METHOD,
        "arguments": vars(args),
        "selected_problem_ids": selected_ids,
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
    print("\nMATCHED FAILFAST SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    if archive:
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
