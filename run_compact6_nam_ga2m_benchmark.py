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
    parser.add_argument("--target_quantization", choices=("int8", "none"), default="int8")
    parser.add_argument("--dllm_dir", default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--output_dir", default="/home/maihoang/failfasttesting/outputs_compact6_nam_ga2m_math_gsm8k_test25")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def command(args, model):
    destination = Path(args.output_dir) / model
    values = [
        sys.executable, "-u", "run_otrc_v2_td_benchmark.py",
        "--datasets", *DATASETS,
        "--num_questions", str(args.num_questions),
        "--feature_schema", "otrc_v2_2_compact_td",
        "--credit_assignment", "verifier_boundary_factual_no_bootstrap",
        "--value_parameterization", "shared_value_advantage",
        "--value_model", model,
        "--nonlinear_learning_rate", "0.001",
        "--nonlinear_weight_decay", "0.0",
        "--nonlinear_grad_clip", "1.0",
        "--nonlinear_device", "cpu",
        "--adaptive_rho_alpha", "0.05",
        "--rho_warmup_boundaries", "0",
        "--policy_weight_ema_beta", "0.0",
        "--adaptive_factual_ema_alpha", "0.2",
        "--adaptive_risk_beta", "1.0",
        "--adaptive_stop_probability_threshold", "0.75",
        "--adaptive_uncertainty_prior", "1.0",
        "--adaptive_epistemic_scale", "0.1",
        "--adaptive_q_margin", "0.0",
        "--adaptive_explore_epsilon", "0.10",
        "--adaptive_explore_min", "0.02",
        "--adaptive_explore_decay", "0.998",
        "--adaptive_policy_mode", "symmetric_annealed",
        "--adaptive_min_action_probability", "0.10",
        "--adaptive_max_importance_weight", "5.0",
        "--adaptive_weight_snapshot_interval", "100",
        "--warmup_questions", "1",
        "--max_new_tokens", "1024",
        "--spec_len", "8", "--incr_len", "8", "--max_spec_len", "64",
        "--block_size", "32", "--small_block_size", "8",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir", args.dllm_dir,
        "--target_device", "0", "--drafter_device", "0",
        "--target_quantization", args.target_quantization,
        "--drafter_threshold", "0.30", "--lowconf_threshold", "0.50",
        "--seed", "42", "--output_dir", str(destination), "--log_level", "INFO",
    ]
    if args.resume:
        values.append("--resume")
    return values


def run(command):
    process = subprocess.Popen(command, cwd=ROOT)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)


def collect(root, model):
    rows = []
    for dataset in DATASETS:
        matches = list((root / model / "raw" / dataset).glob("*/benchmark_results.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one {model}/{dataset} result, found {len(matches)}")
        frame = pd.read_csv(matches[0])
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "value_model", model)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()
    if args.num_questions != 25:
        raise ValueError("this matched experiment requires 25 questions")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for model in ("nam", "ga2m"):
        print(f"\nRUN COMPACT6-{model.upper()} | MATH+GSM8K | 25 each", flush=True)
        run(command(args, model))
    combined = pd.concat([collect(root, model) for model in ("nam", "ga2m")], ignore_index=True)
    combined.to_csv(root / "all_problem_results.csv", index=False)
    metric = "actual_e2e_ms_per_output_token"
    pivot = combined.pivot(index=["dataset", "problem_id"], columns="value_model", values=metric).reset_index()
    pivot["nam_speedup_vs_ga2m"] = pivot.ga2m / pivot.nam
    pivot.to_csv(root / "paired_nam_vs_ga2m.csv", index=False)
    summary = pivot.groupby("dataset").agg(
        problems=("problem_id", "size"),
        nam_ms_per_token=("nam", "mean"),
        ga2m_ms_per_token=("ga2m", "mean"),
        mean_speedup_nam_vs_ga2m=("nam_speedup_vs_ga2m", "mean"),
    ).reset_index()
    summary.to_csv(root / "comparison_summary.csv", index=False)
    (root / "matched_problem_ids.json").write_text(json.dumps({d: PROBLEM_IDS[d][:25] for d in DATASETS}, indent=2))
    archive = shutil.make_archive(str(root), "zip", root.parent, root.name)
    print(summary.to_string(index=False), flush=True)
    print(f"ARCHIVE READY: {archive}", flush=True)


if __name__ == "__main__":
    main()
