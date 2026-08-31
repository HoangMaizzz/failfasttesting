import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k")
METHODS = ("failfast", "always_stop", "compact6_annealed", "raw_linear", "raw_mlp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_raw_aligned_smoke_compare_math_gsm8k_test5"
        ),
    )
    parser.add_argument(
        "--controls_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_shared_va_failfast_always_stop_int8_test25"
        ),
    )
    parser.add_argument(
        "--compact6_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_compact6_annealed_from_zero_int8_test25"
        ),
    )
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
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


def only_result(path):
    matches = list(path.glob("*/benchmark_results.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result under {path}, found {matches}")
    return matches[0]


def result_paths(args, online_root, dataset):
    controls = Path(args.controls_dir)
    compact6 = Path(args.compact6_dir)
    return {
        "failfast": controls / "matched_failfast_baseline" / "raw" / dataset
        / "failfast_matched" / "benchmark_results.csv",
        "always_stop": next(
            (controls / "policy_controls" / "frozen_stop" / "raw" / dataset).glob(
                "*/benchmark_results.csv"
            )
        ),
        "compact6_annealed": next(
            (compact6 / "raw" / dataset).glob("*/benchmark_results.csv")
        ),
        "raw_linear": only_result(online_root / "raw_linear" / Path("raw") / dataset),
        "raw_mlp": only_result(online_root / "raw_mlp" / Path("raw") / dataset),
    }


def main():
    args = parse_args()
    root = Path(args.output_dir)
    online = root / "latest_raw"
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "run_raw_state_online_test.py",
        "--num_questions", "5",
        "--target_quantization", "int8",
        "--dllm_dir", args.dllm_dir,
        "--output_dir", str(online),
        "--allow_output_mismatch",
    ]
    if args.resume:
        command.append("--resume")
    run(command)

    summaries = []
    paired_parts = []
    selected_ids = {}
    for dataset in DATASETS:
        paths = result_paths(args, online, dataset)
        failfast = pd.read_csv(paths["failfast"]).sort_values("problem_id").head(5)
        ids = failfast.problem_id.astype(int).tolist()
        selected_ids[dataset] = ids
        paired = pd.DataFrame({"problem_id": ids})
        paired.insert(0, "dataset", dataset)
        for method in METHODS:
            frame = pd.read_csv(paths[method])
            frame = frame[frame.problem_id.astype(int).isin(ids)].copy()
            if set(frame.problem_id.astype(int)) != set(ids):
                raise RuntimeError(f"{dataset}/{method} does not contain matched IDs")
            frame = frame.set_index(frame.problem_id.astype(int)).loc[ids].reset_index(drop=True)
            total_ms = 1000.0 * float(frame.actual_algorithm_time.sum())
            output_tokens = int(frame.output_tokens.sum())
            summaries.append({
                "dataset": dataset,
                "method": method,
                "questions": len(frame),
                "output_tokens": output_tokens,
                "algorithm_time_s": float(frame.actual_algorithm_time.sum()),
                "ms_per_output_token": total_ms / max(output_tokens, 1),
            })
            paired[f"{method}_ms_per_output_token"] = (
                1000.0 * frame.actual_algorithm_time.to_numpy()
                / frame.output_tokens.clip(lower=1).to_numpy()
            )
            paired[f"{method}_output_hash"] = frame.output_token_hash.astype(str).to_numpy()
        paired_parts.append(paired)

    summary = pd.DataFrame(summaries)
    baseline = summary[summary.method == "failfast"][[
        "dataset", "ms_per_output_token"
    ]].rename(columns={"ms_per_output_token": "failfast_ms_per_output_token"})
    summary = summary.merge(baseline, on="dataset", validate="many_to_one")
    summary["speedup_vs_failfast"] = (
        summary.failfast_ms_per_output_token / summary.ms_per_output_token
    )
    paired = pd.concat(paired_parts, ignore_index=True)
    for method in METHODS:
        paired[f"{method}_speedup_vs_failfast"] = (
            paired.failfast_ms_per_output_token
            / paired[f"{method}_ms_per_output_token"]
        )
        paired[f"{method}_output_matches_failfast"] = (
            paired[f"{method}_output_hash"] == paired.failfast_output_hash
        )
    summary.to_csv(root / "smoke_method_summary.csv", index=False)
    paired.to_csv(root / "smoke_paired_problem_comparison.csv", index=False)
    (root / "smoke_manifest.json").write_text(json.dumps({
        "datasets": list(DATASETS),
        "problem_ids": selected_ids,
        "methods": list(METHODS),
        "new_methods_run": ["raw_linear", "raw_mlp"],
        "controls_reused": ["failfast", "always_stop", "compact6_annealed"],
        "feature_schema": "otrc_raw_state_v1",
        "feature_version": 302,
        "quantization": "int8",
        "drafter_threshold": 0.30,
        "outer_threshold": 0.50,
    }, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(root), "zip", root.parent, root.name)
    print("\nSMOKE METHOD SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nARCHIVE: {archive}", flush=True)


if __name__ == "__main__":
    main()
