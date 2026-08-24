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
VERSION = "math_truncated_global_oracle_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_math_truncated_global_h2_test50",
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[2])
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--max_trajectories_per_prefix", type=int, default=0)
    parser.add_argument("--lcp_validation_candidates", type=int, default=16)
    parser.add_argument("--exact_results_dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if not args.horizons or any(value < 1 for value in args.horizons):
        parser.error("--horizons must contain positive integers")
    if args.max_questions < 1:
        parser.error("--max_questions must be positive")
    return args


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


def read_available(raw_dir, horizon, filename):
    frames = []
    for path in sorted((raw_dir / f"h{horizon}").glob(f"problem_*/{filename}")):
        if path.exists() and path.stat().st_size:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_horizon(output_dir, reference, horizon):
    prefix = f"truncated_global_h{horizon}"
    filenames = (
        f"{prefix}_sample_summary.csv",
        f"{prefix}_block_curves.csv",
        f"{prefix}_delayed_benefit_events.csv",
        f"{prefix}_patience_analysis.csv",
        f"{prefix}_optimal_trace.csv",
        f"{prefix}_cache_stats.csv",
        f"{prefix}_timing_profile.csv",
        "benchmark_results.csv",
    )
    frames = {
        filename: read_available(output_dir / "raw", horizon, filename)
        for filename in filenames
    }
    horizon_dir = output_dir / f"h{horizon}"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        if not frame.empty:
            frame.to_csv(horizon_dir / filename, index=False)
    samples = frames[f"{prefix}_sample_summary.csv"]
    if samples.empty:
        return None, frames
    paired = samples.merge(
        reference[["problem_id", "output_token_hash"]],
        on="problem_id",
        how="left",
        suffixes=("_oracle", "_reference"),
        validate="one_to_one",
    )
    paired["reference_output_match"] = (
        paired["output_token_hash_oracle"].astype(str)
        == paired["output_token_hash_reference"].astype(str)
    )
    paired.to_csv(horizon_dir / "reference_output_validation.csv", index=False)
    baseline_ms = float(samples["baseline_latency_ms"].sum())
    replay_ms = float(samples["real_replay_latency_ms"].sum())
    generated = int(samples["generated_tokens"].sum())
    timing = frames[f"{prefix}_timing_profile.csv"]
    timing_by_component = (
        timing.groupby("component")["time_ms"].sum().to_dict()
        if not timing.empty
        else {}
    )
    row = {
        "method": f"H{horizon}",
        "horizon": horizon,
        "num_questions_completed": len(samples),
        "generated_tokens": generated,
        "baseline_latency_ms": baseline_ms,
        "real_replay_latency_ms": replay_ms,
        "pooled_speedup_vs_baseline": baseline_ms / replay_ms,
        "latency_reduction_percent": 100.0 * (1.0 - replay_ms / baseline_ms),
        "real_replay_ms_per_output_token": replay_ms / generated,
        "dllm_forwards": int(samples["dllm_forwards"].sum()),
        "verifier_calls": int(samples["verifier_calls"].sum()),
        "search_wall_time_hours": float(
            samples["oracle_search_wall_time_ms"].sum() / 3_600_000.0
        ),
        "memo_calls": int(samples["memo_total_calls"].sum()),
        "memo_unique_states": int(samples["memo_unique_states"].sum()),
        "memo_hits": int(samples["memo_hits"].sum()),
        "baseline_tail_cache_hits": int(
            samples["baseline_tail_cache_hits"].sum()
        ),
        "baseline_tail_cache_misses": int(
            samples["baseline_tail_cache_misses"].sum()
        ),
        "mean_absolute_latency_prediction_error_percent": float(
            samples["latency_prediction_error_percent"].abs().mean()
        ),
        "lcp_semantic_validations": int(
            samples["lcp_semantic_validations"].sum()
        ),
        "reference_output_match_percent": 100.0 * paired[
            "reference_output_match"
        ].mean(),
        "delayed_benefit_events": int(
            samples["number_delayed_benefit_events"].sum()
        ),
        "counterfactual_drafter_search_hours": timing_by_component.get(
            "counterfactual_drafter", 0.0
        ) / 3_600_000.0,
        "baseline_prepass_hours": timing_by_component.get(
            "baseline_real_prepass", 0.0
        ) / 3_600_000.0,
        "selected_replay_hours": timing_by_component.get(
            "selected_path_replay", 0.0
        ) / 3_600_000.0,
    }
    return row, frames


def _first_stop_depths(frame, action_column):
    if frame.empty:
        return pd.DataFrame()
    keys = ["problem_id", "prefix_len", "block_id"]
    selected = frame[frame[action_column] == "stop"].copy()
    if selected.empty:
        return pd.DataFrame(columns=keys + ["stop_depth"])
    return (
        selected.groupby(keys, as_index=False)["refinement_step"]
        .min()
        .rename(columns={"refinement_step": "stop_depth"})
    )


def write_exact_validation(output_dir, horizons, exact_results_dir):
    exact_dir = Path(exact_results_dir)
    exact_summary_path = exact_dir / "global_oracle_problem_summary.csv"
    exact_nodes_path = exact_dir / "global_oracle_nodes.csv"
    if not exact_summary_path.exists() or not exact_nodes_path.exists():
        raise FileNotFoundError(
            "exact results must contain global_oracle_problem_summary.csv and "
            "global_oracle_nodes.csv"
        )
    exact_summary = pd.read_csv(exact_summary_path)
    exact_nodes = pd.read_csv(exact_nodes_path)
    exact_depths = _first_stop_depths(exact_nodes, "global_action")
    rows = []
    for horizon in horizons:
        prefix = f"truncated_global_h{horizon}"
        horizon_dir = output_dir / f"h{horizon}"
        sample_path = horizon_dir / f"{prefix}_sample_summary.csv"
        curve_path = horizon_dir / f"{prefix}_block_curves.csv"
        if not sample_path.exists() or not curve_path.exists():
            continue
        samples = pd.read_csv(sample_path)
        curves = pd.read_csv(curve_path)
        common_ids = sorted(
            set(map(int, samples["problem_id"])).intersection(
                map(int, exact_summary["problem_id"])
            )
        )
        if not common_ids:
            continue
        paired = samples[samples["problem_id"].isin(common_ids)].merge(
            exact_summary[exact_summary["problem_id"].isin(common_ids)],
            on="problem_id",
            validate="one_to_one",
            suffixes=("_truncated", "_exact"),
        )
        paired["latency_regret_percent"] = 100.0 * (
            paired["real_replay_latency_ms"]
            - paired["oracle_optimal_latency_ms"]
        ) / paired["oracle_optimal_latency_ms"]
        paired["verifier_call_difference"] = (
            paired["verifier_calls"] - paired["oracle_verifier_calls"]
        )
        paired["dllm_forward_difference"] = (
            paired["dllm_forwards"] - paired["oracle_dllm_forwards"]
        )
        truncated_depths = _first_stop_depths(curves, "global_action")
        depths = truncated_depths.merge(
            exact_depths,
            on=["problem_id", "prefix_len", "block_id"],
            suffixes=("_truncated", "_exact"),
        )
        action_keys = ["problem_id", "prefix_len", "block_id", "refinement_step"]
        actions = curves[action_keys + ["global_action"]].merge(
            exact_nodes[action_keys + ["global_action"]],
            on=action_keys,
            suffixes=("_truncated", "_exact"),
        )
        rows.append({
            "method": f"H{horizon}",
            "num_samples": len(paired),
            "median_latency_regret_percent": float(
                paired["latency_regret_percent"].median()
            ),
            "mean_latency_regret_percent": float(
                paired["latency_regret_percent"].mean()
            ),
            "action_agreement_percent": (
                100.0
                * (actions["global_action_truncated"] == actions["global_action_exact"]).mean()
                if not actions.empty
                else float("nan")
            ),
            "mean_absolute_stop_depth_error": (
                float(
                    (
                        depths["stop_depth_truncated"]
                        - depths["stop_depth_exact"]
                    ).abs().mean()
                )
                if not depths.empty
                else float("nan")
            ),
            "mean_verifier_call_difference": float(
                paired["verifier_call_difference"].mean()
            ),
            "mean_dllm_forward_difference": float(
                paired["dllm_forward_difference"].mean()
            ),
        })
        paired.to_csv(
            output_dir / f"{prefix}_vs_exact_per_sample.csv",
            index=False,
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "truncated_global_h2_vs_exact_validation.csv",
            index=False,
        )


def main():
    args = parse_args()
    manifest, reference, problem_ids = load_reference()
    problem_ids = problem_ids[:args.max_questions]
    source = manifest["arguments"]
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for horizon in sorted(set(args.horizons)):
        prefix = f"truncated_global_h{horizon}"
        for index, problem_id in enumerate(problem_ids, start=1):
            problem_dir = raw_dir / f"h{horizon}" / f"problem_{problem_id:04d}"
            summary_path = problem_dir / f"{prefix}_sample_summary.csv"
            if args.resume and summary_path.exists() and summary_path.stat().st_size:
                print(
                    f"SKIP H={horizon} {index}/{len(problem_ids)} "
                    f"problem_id={problem_id}",
                    flush=True,
                )
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
                "--truncated_global_horizon", str(horizon),
                "--truncated_lcp_validation_candidates",
                str(args.lcp_validation_candidates),
                "--global_oracle_max_states",
                str(args.max_trajectories_per_prefix),
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
                f"TRUNCATED GLOBAL H={horizon} {index}/{len(problem_ids)} | "
                f"problem_id={problem_id}",
                flush=True,
            )
            print("=" * 100, flush=True)
            run_streaming(command)
            row, _ = aggregate_horizon(output_dir, reference, horizon)
            if row is not None:
                print(pd.DataFrame([row]).to_string(index=False), flush=True)

    aggregate_rows = []
    for horizon in sorted(set(args.horizons)):
        row, _ = aggregate_horizon(output_dir, reference, horizon)
        if row is not None:
            aggregate_rows.append(row)
    if aggregate_rows:
        aggregate_frame = pd.DataFrame(aggregate_rows)
        aggregate_frame.to_csv(output_dir / "truncated_global_summary.csv", index=False)
    else:
        aggregate_frame = pd.DataFrame()
    if args.exact_results_dir:
        write_exact_validation(output_dir, args.horizons, args.exact_results_dir)

    report_manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "problem_ids": problem_ids,
        "horizons": sorted(set(args.horizons)),
        "source_reference": str(REFERENCE_DIR),
        "source_arguments": source,
        "oracle_definition": (
            "Exact exhaustive inner refinement and original outer FailFast behavior "
            "through H verifier boundaries, followed by a cached original-FailFast "
            "baseline tail. This is not the exact full-to-EOS global oracle."
        ),
        "deterministic_shortcut": (
            "Greedy verifier acceptance is reconstructed by exact longest-prefix "
            "matching against a target continuation obtained from a real baseline "
            "prepass. Real verifier calls validate shortcut semantics, and the final "
            "selected path is replayed with the real verifier."
        ),
        "state_key": (
            "Exact generated target-token prefix tuple plus remaining verifier horizon; "
            "drafter KV reuse is disabled so the state is path-independent."
        ),
        "search_cost_excluded_from_replay_latency": True,
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
    if not aggregate_frame.empty:
        print("\nTRUNCATED GLOBAL ORACLE SUMMARY")
        print(aggregate_frame.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive}")


if __name__ == "__main__":
    main()
