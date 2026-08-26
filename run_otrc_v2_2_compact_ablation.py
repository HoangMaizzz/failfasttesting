import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "otrc_v2_2_compact_rho_warmup_ablation_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(PROBLEM_IDS),
        default=["math", "gsm8k"],
    )
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--rho_warmup_boundaries", type=int, default=32)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_otrc_v2_2_compact_ablation_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0 or args.num_questions > 25:
        raise ValueError("--num_questions must be in [1, 25]")
    if args.rho_warmup_boundaries <= 0:
        raise ValueError("--rho_warmup_boundaries must be positive")


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


def case_command(args, output_dir, rho_warmup_boundaries):
    command = [
        sys.executable,
        "-u",
        "run_otrc_v2_td_benchmark.py",
        "--datasets",
        *args.datasets,
        "--num_questions",
        str(args.num_questions),
        "--feature_schema",
        "otrc_v2_2_compact_td",
        "--credit_assignment",
        "verifier_boundary_factual_no_bootstrap",
        "--rho_warmup_boundaries",
        str(rho_warmup_boundaries),
        "--warmup_questions",
        str(args.warmup_questions),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--spec_len",
        "8",
        "--incr_len",
        "8",
        "--max_spec_len",
        "60",
        "--block_size",
        "32",
        "--small_block_size",
        "8",
        "--target_model_name",
        args.target_model_name,
        "--dllm_dir",
        args.dllm_dir,
        "--drafter_threshold",
        "0.05",
        "--lowconf_threshold",
        "0.45",
        "--adaptive_learning_rate",
        "0.02",
        "--adaptive_mc_learning_rate",
        "0.01",
        "--adaptive_mc_mix",
        "0.5",
        "--adaptive_update_mode",
        "mixed",
        "--adaptive_rho_alpha",
        "0.05",
        "--adaptive_factual_ema_alpha",
        "0.2",
        "--adaptive_risk_beta",
        "1.0",
        "--adaptive_stop_probability_threshold",
        "0.75",
        "--adaptive_uncertainty_prior",
        "1.0",
        "--adaptive_epistemic_scale",
        "0.1",
        "--adaptive_explore_epsilon",
        "0.10",
        "--adaptive_explore_min",
        "0.01",
        "--adaptive_explore_decay",
        "0.998",
        "--adaptive_warmup_rounds",
        "20",
        "--adaptive_early_stop_min_observations",
        "32",
        "--adaptive_min_action_probability",
        "0.10",
        "--adaptive_max_importance_weight",
        "5.0",
        "--adaptive_weight_snapshot_interval",
        "100",
        "--seed",
        "42",
        "--output_dir",
        str(output_dir),
        "--log_level",
        args.log_level,
    ]
    if args.resume:
        command.append("--resume")
    return command


def read_case(case_dir, label):
    manifest = json.loads(
        (case_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    method = manifest["method"]
    summary = pd.read_csv(case_dir / "dataset_method_summary.csv")
    summary.insert(0, "case", label)
    summary.insert(1, "rho_warmup_boundaries", int(
        manifest["arguments"]["rho_warmup_boundaries"]
    ))
    factual = pd.read_csv(case_dir / "factual_target_summary.csv")
    factual.insert(0, "case", label)
    conditioning = pd.read_csv(case_dir / "feature_conditioning.csv")
    conditioning.insert(0, "case", label)
    learning = pd.read_csv(case_dir / "learning_dynamics.csv")
    learning.insert(0, "case", label)
    raw = {}
    for dataset in manifest["datasets"]:
        raw[dataset] = pd.read_csv(
            case_dir / "raw" / dataset / method / "benchmark_results.csv"
        )
    return summary, factual, conditioning, learning, raw


def paired_comparison(compact_raw, warmup_raw):
    paired_frames = []
    summaries = []
    for dataset in compact_raw:
        left = compact_raw[dataset].copy()
        right = warmup_raw[dataset].copy()
        paired = left.merge(
            right,
            on="problem_id",
            suffixes=("_compact", "_rho_warmup"),
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(left) or len(paired) != len(right):
            raise ValueError(f"paired problem coverage mismatch for {dataset}")
        paired.insert(0, "dataset", dataset)
        paired["rho_warmup_speedup_vs_compact"] = (
            paired["actual_algorithm_ms_per_output_token_compact"]
            / paired["actual_algorithm_ms_per_output_token_rho_warmup"]
        )
        paired["rho_warmup_wins"] = (
            paired["actual_algorithm_ms_per_output_token_rho_warmup"]
            < paired["actual_algorithm_ms_per_output_token_compact"]
        )
        paired["output_match"] = (
            paired["output_token_hash_compact"]
            == paired["output_token_hash_rho_warmup"]
        )
        paired_frames.append(paired)

        compact_ms = (
            1000.0 * paired["actual_algorithm_time_compact"].sum()
            / paired["output_tokens_compact"].sum()
        )
        warmup_ms = (
            1000.0 * paired["actual_algorithm_time_rho_warmup"].sum()
            / paired["output_tokens_rho_warmup"].sum()
        )
        summaries.append({
            "dataset": dataset,
            "num_questions": len(paired),
            "compact_ms_per_output_token": compact_ms,
            "rho_warmup_ms_per_output_token": warmup_ms,
            "rho_warmup_speedup_vs_compact": compact_ms / warmup_ms,
            "rho_warmup_win_rate_percent": 100.0 * paired["rho_warmup_wins"].mean(),
            "output_match_rate_percent": 100.0 * paired["output_match"].mean(),
        })
    return pd.concat(paired_frames, ignore_index=True), pd.DataFrame(summaries)


def main():
    args = parse_args()
    validate_args(args)
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("compact6", 0),
        (f"compact6_rho_warmup{args.rho_warmup_boundaries}",
         args.rho_warmup_boundaries),
    ]
    loaded = {}
    for label, warmup in cases:
        case_dir = output_dir / label
        print("\n" + "=" * 100, flush=True)
        print(
            f"RUN {label} | rho_warmup_boundaries={warmup}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(case_command(args, case_dir, warmup))
        loaded[label] = read_case(case_dir, label)

    summary = pd.concat([value[0] for value in loaded.values()], ignore_index=True)
    factual = pd.concat([value[1] for value in loaded.values()], ignore_index=True)
    conditioning = pd.concat(
        [value[2] for value in loaded.values()],
        ignore_index=True,
    )
    learning = pd.concat([value[3] for value in loaded.values()], ignore_index=True)
    compact_raw = loaded["compact6"][4]
    warmup_label = cases[1][0]
    paired, paired_summary = paired_comparison(
        compact_raw,
        loaded[warmup_label][4],
    )

    summary.to_csv(output_dir / "ablation_dataset_method_summary.csv", index=False)
    factual.to_csv(output_dir / "ablation_factual_target_summary.csv", index=False)
    conditioning.to_csv(output_dir / "ablation_feature_conditioning.csv", index=False)
    learning.to_csv(output_dir / "ablation_learning_dynamics.csv", index=False)
    paired.to_csv(output_dir / "paired_problem_comparison.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_dataset_summary.csv", index=False)
    manifest = {
        "version": VERSION,
        "cases": [label for label, _ in cases],
        "datasets": args.datasets,
        "num_questions": args.num_questions,
        "problem_ids": {
            dataset: PROBLEM_IDS[dataset][:args.num_questions]
            for dataset in args.datasets
        },
        "arguments": vars(args),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "baseline_or_oracle_executed": False,
    }
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("\nCOMPACT6 RHO-WARMUP PAIRED SUMMARY", flush=True)
    print(paired_summary.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
