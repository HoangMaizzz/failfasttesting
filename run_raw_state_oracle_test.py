import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from raw_state_experiment import build_raw_oracle_dataset


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--pilot_questions", type=int, default=2)
    parser.add_argument("--boundaries_per_problem", type=int, default=5)
    parser.add_argument("--max_states_per_boundary", type=int, default=256)
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
            "outputs_raw_state_aligned_oracle_math_gsm8k_test10"
        ),
    )
    parser.add_argument("--resume", action="store_true")
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


def main():
    args = parse_args()
    root = Path(args.output_dir)
    run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir])
    command = [
        sys.executable,
        "-u",
        "run_exact_boundary_oracle.py",
        "--datasets", "math", "gsm8k",
        "--num_questions", str(args.num_questions),
        "--pilot_questions", str(args.pilot_questions),
        "--boundaries_per_problem", str(args.boundaries_per_problem),
        "--max_states_per_boundary", str(args.max_states_per_boundary),
        "--target_quantization", args.target_quantization,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--dllm_dir", args.dllm_dir,
        "--output_dir", str(root),
        "--skip_archive",
    ]
    if args.resume:
        command.append("--resume")
    run(command)
    checks = pd.read_csv(root / "final_report" / "output_replay_check.csv")
    if checks.empty or not bool(checks.output_match.astype(bool).all()):
        raise RuntimeError("exact replay output did not match behavior output")
    trees = pd.read_csv(root / "final_report" / "exact_tree_summary.csv")
    if trees.empty or not bool(trees.resolved.astype(bool).all()):
        raise RuntimeError("one or more exact boundary trees were unresolved")
    build_raw_oracle_dataset(root, root / "raw_capacity_dataset")
    archive = shutil.make_archive(str(root) + "_final", "zip", root.parent, root.name)
    print(f"\nRAW ORACLE ARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
