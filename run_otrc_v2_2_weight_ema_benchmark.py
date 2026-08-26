import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
VERSION = "otrc_v2_2_compact_no_bootstrap_weight_ema_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["math", "gsm8k"])
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--policy_weight_ema_beta", type=float, default=0.99)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--target_model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_otrc_v2_2_weight_ema_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def validate_args(args):
    if args.datasets != ["math", "gsm8k"]:
        raise ValueError("this matched test requires --datasets math gsm8k")
    if args.num_questions <= 0 or args.num_questions > 25:
        raise ValueError("--num_questions must be in [1, 25]")
    if not math.isclose(
        args.policy_weight_ema_beta,
        0.99,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("this predeclared ablation requires beta=0.99")


def benchmark_command(args):
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
        "0",
        "--policy_weight_ema_beta",
        str(args.policy_weight_ema_beta),
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
        str(args.output_dir),
        "--log_level",
        args.log_level,
    ]
    if args.resume:
        command.append("--resume")
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


def validate_outputs(output_dir):
    feature_stats = pd.read_csv(output_dir / "feature_statistics.csv")
    ratio = feature_stats.loc[
        feature_stats.feature.eq("draft_verify_latency_ratio")
    ]
    if ratio.empty or not (ratio["variance"] > 0.0).all():
        raise RuntimeError("draft_verify_latency_ratio has no variance")
    policy = pd.read_csv(output_dir / "policy_ema_summary.csv")
    enabled = (
        policy["policy_weight_ema_enabled"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    if len(policy) != 2 or not enabled.all():
        raise RuntimeError("policy EMA diagnostics are incomplete")
    method = "otrc_v2_2_compact_factual_no_bootstrap_policy_ema0p99"
    states = {}
    for dataset in ("math", "gsm8k"):
        state_path = (
            output_dir
            / "raw"
            / dataset
            / method
            / "adaptive_td_runtime_state.json"
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        updates = state["policy_weight_ema"]
        if not all(updates[action]["update_count"] > 0 for action in updates):
            raise RuntimeError(f"policy EMA missed an action on {dataset}")
        states[dataset] = {
            action: updates[action]["update_count"]
            for action in updates
        }
    return ratio, policy, states


def main():
    args = parse_args()
    validate_args(args)
    started = time.time()
    run_streaming(benchmark_command(args))
    output_dir = Path(args.output_dir)
    ratio, policy, states = validate_outputs(output_dir)
    manifest = {
        "version": VERSION,
        "arguments": vars(args),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "baseline_or_oracle_executed": False,
        "policy_ema_update_counts": states,
    }
    (output_dir / "weight_ema_test_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("\nWEIGHT EMA FEATURE CHECK", flush=True)
    print(ratio.to_string(index=False), flush=True)
    print("\nWEIGHT EMA POLICY SUMMARY", flush=True)
    print(policy.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
