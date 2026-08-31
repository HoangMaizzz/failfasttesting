import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--pilot_questions", type=int, default=2)
    parser.add_argument("--boundaries_per_problem", type=int, default=5)
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
            "outputs_raw_aligned_discrimination_math_gsm8k_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command):
    print("\nRUN:", " ".join(map(str, command)), flush=True)
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
    if args.num_questions != 25:
        raise ValueError("this matched discrimination benchmark requires 25 questions")
    root = Path(args.output_dir)
    oracle = root / "oracle"
    online = root / "online"
    capacity = root / "capacity"
    checkpoint = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    common_device = [
        "--target_quantization", args.target_quantization,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--dllm_dir", args.dllm_dir,
    ]
    oracle_command = [
        sys.executable, "-u", "run_raw_state_oracle_test.py",
        "--num_questions", "25",
        "--pilot_questions", str(args.pilot_questions),
        "--boundaries_per_problem", str(args.boundaries_per_problem),
        "--output_dir", str(oracle),
        *common_device,
    ]
    online_command = [
        sys.executable, "-u", "run_raw_state_online_test.py",
        "--num_questions", "25",
        "--output_dir", str(online),
        *common_device,
    ]
    if args.resume:
        oracle_command.append("--resume")
        online_command.append("--resume")
    run(oracle_command)
    run([
        sys.executable, "-u", "run_raw_state_capacity_test.py",
        "--oracle_dir", str(oracle / "raw_capacity_dataset"),
        "--output_dir", str(capacity),
    ])
    run(online_command)
    run([
        sys.executable, "-u", "run_raw_checkpoint_oracle_test.py",
        "--online_dir", str(online),
        "--oracle_dir", str(oracle / "raw_capacity_dataset"),
        "--output_dir", str(checkpoint),
    ])

    capacity_summary = pd.read_csv(capacity / "raw_capacity_summary.csv")
    learning_curve = pd.read_csv(
        checkpoint / "checkpoint_oracle_learning_curve.csv"
    )
    final_checkpoints = learning_curve[learning_curve.final.astype(bool)].copy()
    capacity_summary.to_csv(root / "feature_capacity_summary.csv", index=False)
    final_checkpoints.to_csv(root / "online_oracle_alignment_summary.csv", index=False)
    manifest = {
        **vars(args),
        "datasets": ["math", "gsm8k"],
        "models": ["raw_linear", "raw_mlp"],
        "feature_schema": "otrc_raw_state_v1",
        "feature_version": 302,
        "metrics": [
            "advantage_auc", "sign_accuracy", "balanced_accuracy",
            "stop_recall", "continue_recall", "true_stop", "false_stop",
            "false_continue", "true_continue", "advantage_spearman",
            "mean_oracle_regret",
        ],
    }
    (root / "discrimination_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    archive = shutil.make_archive(str(root) + "_final", "zip", root.parent, root.name)
    print("\nFEATURE CAPACITY SUMMARY", flush=True)
    print(capacity_summary.to_string(index=False), flush=True)
    print("\nONLINE ORACLE ALIGNMENT SUMMARY", flush=True)
    print(final_checkpoints.to_string(index=False), flush=True)
    print(f"\nRAW DISCRIMINATION ARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
