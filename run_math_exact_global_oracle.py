import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "benchmark_references" / "math_failfast8_test50"
VERSION = "math_exact_global_inner_refinement_oracle_v1"
REPORT_FILES = (
    "global_oracle_problem_summary.csv",
    "global_oracle_nodes.csv",
    "global_oracle_block_curves.csv",
    "global_oracle_edges.csv",
    "global_oracle_delayed_benefit_events.csv",
    "global_oracle_patience_analysis.csv",
    "global_oracle_optimal_trace.csv",
    "global_oracle_optimal_rounds.csv",
    "global_oracle_failfast_trace.csv",
    "benchmark_results.csv",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_math_exact_global_oracle_test50",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--max_replays_per_prefix", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def run_streaming(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_reference():
    manifest = json.loads(
        (REFERENCE_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    baseline = pd.read_csv(
        REFERENCE_DIR / "raw" / "failfast" / "benchmark_results.csv"
    )
    problem_ids = [int(value) for value in manifest["problem_ids"]]
    if set(problem_ids) != set(map(int, baseline["problem_id"])):
        raise ValueError("bundled MATH reference IDs do not match its baseline CSV")
    return manifest, baseline, problem_ids


def read_available(output_dir, filename):
    frames = []
    for path in sorted((output_dir / "raw").glob(f"problem_*/{filename}")):
        if path.exists() and path.stat().st_size:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(output_dir, reference):
    frames = {name: read_available(output_dir, name) for name in REPORT_FILES}
    for name, frame in frames.items():
        if not frame.empty:
            frame.to_csv(output_dir / name, index=False)
    summaries = frames["global_oracle_problem_summary.csv"]
    if summaries.empty:
        return None
    paired = summaries.merge(
        reference,
        on="problem_id",
        how="left",
        suffixes=("_oracle_graph", "_reference"),
        validate="one_to_one",
    )
    if "output_token_hash_reference" in paired:
        paired["reference_output_match"] = (
            paired["output_token_hash_oracle_graph"].astype(str)
            == paired["output_token_hash_reference"].astype(str)
        )
    paired.to_csv(output_dir / "reference_replay_validation.csv", index=False)

    baseline_ms = float(summaries["baseline_total_latency_ms"].sum())
    oracle_ms = float(summaries["oracle_optimal_latency_ms"].sum())
    generated = int(summaries["generated_tokens"].sum())
    nodes = frames["global_oracle_nodes.csv"]
    delayed = frames["global_oracle_delayed_benefit_events.csv"]
    aggregate_row = {
        "num_questions_completed": len(summaries),
        "generated_tokens": generated,
        "baseline_replay_total_latency_ms": baseline_ms,
        "oracle_optimal_total_latency_ms": oracle_ms,
        "oracle_pooled_speedup_vs_failfast_replay": baseline_ms / oracle_ms,
        "oracle_latency_reduction_percent": 100.0 * (1.0 - oracle_ms / baseline_ms),
        "baseline_replay_ms_per_output_token": baseline_ms / generated,
        "oracle_ms_per_output_token": oracle_ms / generated,
        "baseline_dllm_forwards": int(summaries["baseline_dllm_forwards"].sum()),
        "oracle_dllm_forwards": int(summaries["oracle_dllm_forwards"].sum()),
        "baseline_verifier_calls": int(summaries["baseline_verifier_calls"].sum()),
        "oracle_verifier_calls": int(summaries["oracle_verifier_calls"].sum()),
        "oracle_search_wall_time_hours": float(
            summaries["oracle_search_wall_time_ms"].sum() / 3_600_000.0
        ),
        "unique_dp_states": int(summaries["unique_dp_states"].sum()),
        "oracle_replays": int(summaries["oracle_replays"].sum()),
        "decision_states": len(nodes),
        "global_myopic_disagreements": int(
            nodes.get("global_myopic_disagree", pd.Series(dtype=int)).sum()
        ),
        "delayed_benefit_states": len(delayed),
        "all_oracle_results_dominate_replay": bool(
            summaries["global_never_slower_validation"].all()
        ),
    }
    aggregate_frame = pd.DataFrame([aggregate_row])
    aggregate_frame.to_csv(output_dir / "global_oracle_aggregate_summary.csv", index=False)
    return aggregate_frame


def main():
    args = parse_args()
    manifest, reference, problem_ids = load_reference()
    problem_ids = problem_ids[:args.max_questions]
    source = manifest["arguments"]
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for index, problem_id in enumerate(problem_ids, start=1):
        problem_dir = raw_dir / f"problem_{problem_id:04d}"
        summary_path = problem_dir / "global_oracle_problem_summary.csv"
        if args.resume and summary_path.exists() and summary_path.stat().st_size:
            print(f"SKIP {index}/{len(problem_ids)} problem_id={problem_id}", flush=True)
            continue
        if problem_dir.exists():
            shutil.rmtree(problem_dir)
        problem_dir.mkdir(parents=True)
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", "math",
            "--num_questions", "1",
            "--problem_ids", str(problem_id),
            "--warmup_questions", "0",
            "--benchmark_modes", "dllm_ar",
            "--dllm_variant", "failfast",
            "--decoding_strategy", "greedy",
            "--max_new_tokens", str(source["max_new_tokens"]),
            "--spec_len", str(source["spec_len"]),
            "--block_size", str(source["block_size"]),
            "--small_block_size", str(source["small_block_size"]),
            "--target_model_name", source["target_model_name"],
            "--dllm_dir", args.dllm_dir,
            "--drafter_thresholds", str(source["drafter_threshold"]),
            "--sweep_lowconf_threshold", str(source["lowconf_threshold"]),
            "--sweep_max_spec_len", str(source["max_spec_len"]),
            "--sweep_incr_len", str(source["incr_len"]),
            "--seed", str(source["seed"]),
            "--collect_bucket_oracle",
            "--global_oracle_graph",
            "--global_oracle_max_states", str(args.max_replays_per_prefix),
            "--global_oracle_log_interval", str(args.log_interval),
            "--disable_reusing_drafter_kvs",
            "--quiet_generation",
            "--disable_progress",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(problem_dir),
            "--log_level", args.log_level,
        ]
        print("\n" + "=" * 100, flush=True)
        print(
            f"EXACT GLOBAL ORACLE {index}/{len(problem_ids)} | problem_id={problem_id}",
            flush=True,
        )
        print("=" * 100, flush=True)
        run_streaming(command)
        current = aggregate(output_dir, reference)
        if current is not None:
            print(current.to_string(index=False), flush=True)

    aggregate_frame = aggregate(output_dir, reference)
    report_manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "problem_ids": problem_ids,
        "source_reference": str(REFERENCE_DIR),
        "source_arguments": source,
        "oracle_definition": (
            "Exhaustive scripted replay of every legal inner STOP/CONTINUE action; "
            "the original outer FailFast EXTEND/VERIFY policy is executed inside "
            "Fast-dLLM. Every complete round proposal is greedily verified, and a "
            "DAG dynamic program minimizes measured draft+verify+post latency to EOS."
        ),
        "cache_mode": (
            "Drafter KV reuse is disabled for both the FailFast replay and oracle "
            "branches so exact memoization by target prefix is path-independent."
        ),
        "search_cost_excluded": True,
        "elapsed_runner_hours": (time.time() - started) / 3600.0,
    }
    try:
        report_manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        report_manifest["git_commit"] = None
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(report_manifest, indent=2), encoding="utf-8"
    )
    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    if aggregate_frame is not None:
        print("\nEXACT GLOBAL ORACLE SUMMARY")
        print(aggregate_frame.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive}")


if __name__ == "__main__":
    main()
