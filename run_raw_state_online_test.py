import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=25)
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
            "outputs_raw_aligned_shared_va_math_gsm8k_test25"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command):
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def command(args, model, destination):
    values = [
        sys.executable, "-u", "run_otrc_v2_td_benchmark.py",
        "--datasets", *DATASETS,
        "--num_questions", str(args.num_questions),
        "--feature_schema", "otrc_raw_state_v1",
        "--credit_assignment", "verifier_boundary_factual_no_bootstrap",
        "--value_parameterization", "shared_value_advantage",
        "--value_model", model,
        "--nonlinear_learning_rate", "0.001",
        "--nonlinear_weight_decay", "0",
        "--nonlinear_grad_clip", "1.0",
        "--nonlinear_device", "cpu",
        "--adaptive_rho_alpha", "0.05",
        "--rho_warmup_boundaries", "0",
        "--policy_weight_ema_beta", "0",
        "--adaptive_factual_ema_alpha", "0.2",
        "--adaptive_risk_beta", "1.0",
        "--adaptive_stop_probability_threshold", "0.75",
        "--adaptive_uncertainty_prior", "1.0",
        "--adaptive_epistemic_scale", "0.1",
        "--adaptive_q_margin", "0",
        "--adaptive_explore_epsilon", "0.10",
        "--adaptive_explore_min", "0.02",
        "--adaptive_explore_decay", "0.998",
        "--adaptive_warmup_rounds", "20",
        "--adaptive_early_stop_min_observations", "32",
        "--adaptive_policy_mode", "symmetric_annealed",
        "--adaptive_min_action_probability", "0.10",
        "--adaptive_max_importance_weight", "5",
        "--adaptive_weight_snapshot_interval", "100",
        "--warmup_questions", "1",
        "--max_new_tokens", "1024",
        "--spec_len", "8", "--incr_len", "8", "--max_spec_len", "64",
        "--block_size", "32", "--small_block_size", "8",
        "--target_quantization", args.target_quantization,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--dllm_dir", args.dllm_dir,
        "--drafter_threshold", "0.30",
        "--lowconf_threshold", "0.50",
        "--seed", "42",
        "--output_dir", str(destination),
    ]
    if args.resume:
        values.append("--resume")
    return values


def validate_method(directory, model):
    manifest = json.loads((directory / "benchmark_manifest.json").read_text())
    method = manifest["method"]
    summaries = pd.read_csv(directory / "dataset_method_summary.csv")
    if set(summaries.dataset) != set(DATASETS):
        raise RuntimeError(f"{model} did not complete both datasets")
    states = []
    for dataset in DATASETS:
        phase = directory / "raw" / dataset / method
        state = json.loads((phase / "adaptive_td_runtime_state.json").read_text())
        if state["value_model"] != model or state["feature_dim"] != 101:
            raise RuntimeError(f"invalid raw controller state for {dataset}/{model}")
        nonlinear = state.get("nonlinear_value") or {}
        if not nonlinear or nonlinear.get("update_count", 0) <= 0:
            raise RuntimeError(f"{dataset}/{model} did not learn")
        transitions = pd.read_csv(phase / "adaptive_full_stream_transitions.csv")
        if not transitions.td_error.map(pd.notna).all():
            raise RuntimeError(f"{dataset}/{model} produced invalid factual errors")
        decisions = pd.read_csv(phase / "adaptive_td_decisions.csv")
        matrix = np.asarray(
            [json.loads(value) for value in decisions.features],
            dtype=np.float32,
        )
        if matrix.ndim != 2 or matrix.shape[1] != 101 or not np.isfinite(matrix).all():
            raise RuntimeError(f"{dataset}/{model} raw-state matrix is invalid")
        observed_columns = [
            state_index * 48 + position * 6 + 1
            for state_index in range(2)
            for position in range(8)
        ]
        if not np.allclose(matrix[:, observed_columns], 1.0):
            raise RuntimeError(
                f"{dataset}/{model} contains incomplete raw-state decisions"
            )
        if (
            state.get("factual_verifier_latency_ema_ms") is None
            or state.get("factual_tokens_per_verifier_ema") is None
        ):
            raise RuntimeError(
                f"{dataset}/{model} did not update factual verifier features"
            )
        np.savez_compressed(
            phase / "adaptive_raw_states.npz",
            X=matrix,
            problem_id=decisions.problem_id.to_numpy(np.int64),
            round_id=decisions.round_id.to_numpy(np.int64),
            decision_id=decisions.decision_id.to_numpy(np.int64),
            action=decisions.action.to_numpy(str),
        )
        states.append({
            "dataset": dataset,
            "model": model,
            "decisions": state["decision_count"],
            "updates": nonlinear["update_count"],
            "parameter_norm": nonlinear.get("last_parameter_norm", 0.0),
            "snapshots": len(state.get("weight_snapshots") or []),
        })
    return method, summaries, pd.DataFrame(states)


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir])
    summary_frames = []
    state_frames = []
    for model in ("raw_linear", "raw_mlp"):
        destination = root / model
        run(command(args, model, destination))
        _, summary, states = validate_method(destination, model)
        summary.insert(1, "raw_model", model)
        summary_frames.append(summary)
        state_frames.append(states)
    result_frames = []
    for model in ("raw_linear", "raw_mlp"):
        method_root = root / model
        manifest = json.loads((method_root / "benchmark_manifest.json").read_text())
        method = manifest["method"]
        for dataset in DATASETS:
            result = pd.read_csv(
                method_root / "raw" / dataset / method / "benchmark_results.csv"
            )[["problem_id", "output_token_hash"]]
            result["dataset"] = dataset
            result["raw_model"] = model
            result_frames.append(result)
    hashes = pd.concat(result_frames, ignore_index=True).pivot(
        index=["dataset", "problem_id"],
        columns="raw_model",
        values="output_token_hash",
    )
    if not bool((hashes.raw_linear == hashes.raw_mlp).all()):
        raise RuntimeError("raw Linear and raw MLP changed greedy target output")
    pd.concat(summary_frames, ignore_index=True).to_csv(
        root / "raw_online_method_summary.csv", index=False
    )
    pd.concat(state_frames, ignore_index=True).to_csv(
        root / "raw_online_learning_summary.csv", index=False
    )
    archive = shutil.make_archive(str(root) + "_final", "zip", root.parent, root.name)
    print(f"\nRAW ONLINE ARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
