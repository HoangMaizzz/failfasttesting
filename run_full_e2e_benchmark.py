import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


DATASETS = ("math", "aime", "gsm8k", "gpqa", "humaneval")
DATASET_LIMITS = {"aime": 30}
METHOD_ORDER = (
    "ar_only",
    "ar_draft",
    "fast_dllm",
    "eagle3",
    "failfast",
    "cost_aware_no_extend",
    "cost_aware_extend",
)
LOCAL_METHODS = {
    "ar_only": {
        "mode": "verifier_ar",
        "spec_len": 1,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "ar_draft": {
        "mode": "ar_ar",
        "spec_len": 10,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "fast_dllm": {
        "mode": "dllm_ar",
        "spec_len": 10,
        "dllm_variant": "fixed",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "failfast": {
        "mode": "dllm_ar",
        "spec_len": 10,
        "dllm_variant": "failfast",
        "frontier_mode": "disabled",
        "lowconf_threshold": 0.45,
    },
    "cost_aware_no_extend": {
        "mode": "dllm_ar",
        "spec_len": 5,
        "dllm_variant": "failfast",
        "frontier_mode": "cost_aware_no_extend",
        "lowconf_threshold": 0.45,
    },
    "cost_aware_extend": {
        "mode": "dllm_ar",
        "spec_len": 5,
        "dllm_variant": "failfast",
        "frontier_mode": "cost_aware",
        "lowconf_threshold": 0.60,
    },
}
TIME_COLUMNS = (
    "actual_e2e_time",
    "actual_draft_time",
    "actual_verify_time",
    "actual_post_verify_time",
    "actual_algorithm_time",
    "actual_unattributed_core_time",
)
COUNT_COLUMNS = (
    "output_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "num_speculation_rounds",
    "total_num_forward_passes",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--drafter_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--incr_len", type=int, default=10)
    parser.add_argument("--frontier_min_steps", type=int, default=2)
    parser.add_argument("--frontier_patience", type=int, default=2)
    parser.add_argument("--frontier_cost_token_equiv", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/content/failfasttesting/outputs_full_e2e")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--eagle_base_url")
    parser.add_argument("--eagle_api_key", default="EMPTY")
    parser.add_argument("--eagle_model")
    parser.add_argument("--request_timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 0:
        raise ValueError("--warmup_questions must be non-negative")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if "eagle3" in args.methods and not args.aggregate_only and not args.eagle_base_url:
        raise ValueError("--eagle_base_url is required when --methods includes eagle3")


def run_streaming(command, cwd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
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


def local_output_dir(args, dataset, method):
    return Path(args.output_dir) / "raw" / dataset / method


def measured_questions(args, dataset):
    return min(args.num_questions, DATASET_LIMITS.get(dataset, args.num_questions))


def local_results_complete(path, expected_rows):
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    return len(rows) == expected_rows and rows["problem_id"].nunique() == expected_rows


def run_local_method(args, dataset, method):
    config = LOCAL_METHODS[method]
    output_dir = local_output_dir(args, dataset, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_results.csv"
    expected_rows = measured_questions(args, dataset)
    if args.resume and local_results_complete(benchmark_path, expected_rows):
        print(f"RESUME {dataset} | {method}", flush=True)
    else:
        if benchmark_path.exists():
            benchmark_path.unlink()
        command = [
            sys.executable,
            "-u",
            "failfast.py",
            "--dataset_name", dataset,
            "--num_questions", str(expected_rows),
            "--warmup_questions", str(args.warmup_questions),
            "--benchmark_modes", config["mode"],
            "--decoding_strategy", "greedy",
            "--max_new_tokens", str(args.max_new_tokens),
            "--spec_len", str(config["spec_len"]),
            "--block_size", str(args.block_size),
            "--small_block_size", str(args.small_block_size),
            "--target_model_name", args.target_model_name,
            "--drafter_model_name", args.drafter_model_name,
            "--dllm_dir", args.dllm_dir,
            "--dllm_variant", config["dllm_variant"],
            "--drafter_thresholds", str(args.drafter_threshold),
            "--sweep_lowconf_threshold", str(config["lowconf_threshold"]),
            "--sweep_max_spec_len", str(args.max_spec_len),
            "--sweep_incr_len", str(args.incr_len),
            "--frontier_stop_mode", config["frontier_mode"],
            "--frontier_min_steps", str(args.frontier_min_steps),
            "--frontier_patience", str(args.frontier_patience),
            "--frontier_cost_token_equiv", str(args.frontier_cost_token_equiv),
            "--seed", str(args.seed),
            "--quiet_generation",
            "--disable_progress",
            "--skip_artifacts",
            "--skip_plots",
            "--overwrite",
            "--output_dir", str(output_dir),
            "--log_level", args.log_level,
        ]
        print("\n" + "=" * 100, flush=True)
        print(f"RUN {dataset} | {method} | measured={expected_rows} | max_tokens={args.max_new_tokens}", flush=True)
        print("=" * 100, flush=True)
        run_streaming(command, Path(__file__).resolve().parent)
    rows = pd.read_csv(benchmark_path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {benchmark_path}, found {len(rows)}")
    rows["source_problem_id"] = rows["problem_id"]
    rows["dataset"] = dataset
    rows["method"] = method
    rows["backend"] = "transformers_hf"
    rows["runtime_comparable_to_ar"] = True
    rows["measurement_note"] = "synchronized generation-core E2E; greedy decoding"
    if dataset != "gsm8k":
        rows["is_correct"] = math.nan
    for key, value in config.items():
        rows[key] = value
    return rows


def api_request(url, api_key, timeout, payload=None):
    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def api_text(url, api_key, timeout):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def normalize_base_url(base_url):
    return base_url.rstrip("/")


def server_root(base_url):
    base_url = normalize_base_url(base_url)
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def fetch_spec_decode_metrics(args):
    try:
        metrics_text = api_text(
            f"{server_root(args.eagle_base_url)}/metrics",
            args.eagle_api_key,
            args.request_timeout,
        )
    except Exception:
        return None
    counters = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    found = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode_"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        metric_name = fields[0].split("{", 1)[0].removesuffix("_total")
        key = None
        if metric_name == "vllm:spec_decode_num_drafts":
            key = "drafts"
        elif metric_name == "vllm:spec_decode_num_draft_tokens":
            key = "draft_tokens"
        elif metric_name == "vllm:spec_decode_num_accepted_tokens":
            key = "accepted_tokens"
        if key is not None:
            counters[key] += float(fields[1])
            found = True
    return counters if found else None


def metric_delta(before, after, key):
    if before is None or after is None:
        return math.nan
    return max(0.0, after[key] - before[key])


def resolve_eagle_model(args):
    if args.eagle_model:
        return args.eagle_model
    response = api_request(
        f"{normalize_base_url(args.eagle_base_url)}/models",
        args.eagle_api_key,
        args.request_timeout,
    )
    return response["data"][0]["id"]


def normalize_answer(value):
    if value is None:
        return None
    return str(value).strip().replace(",", "").replace("$", "").replace(" ", "")


def extract_boxed_answer(text):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
    return normalize_answer(matches[-1]) if matches else None


def reference_answer(dataset, raw_data):
    if dataset == "gsm8k":
        return normalize_answer(raw_data.get("answer", "").rsplit("####", 1)[-1])
    if dataset == "gpqa":
        return "A"
    return None


def eagle_raw_path(args, dataset):
    return Path(args.output_dir) / "raw" / dataset / "eagle3" / "benchmark_results.csv"


def run_eagle_dataset(args, dataset):
    from utils import format_problem_and_options, get_first_user_msg, populate_dataset

    output_path = eagle_raw_path(args, dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_rows = measured_questions(args, dataset)
    if args.resume and output_path.exists() and len(pd.read_csv(output_path)) == expected_rows:
        print(f"RESUME {dataset} | eagle3", flush=True)
        return pd.read_csv(output_path)

    model = resolve_eagle_model(args)
    dataset_args = SimpleNamespace(dataset_name=dataset)
    populate_dataset(dataset_args)
    rows = []
    runs = (
        [(problem_id, True) for problem_id in range(args.warmup_questions)]
        + [(problem_id, False) for problem_id in range(expected_rows)]
    )
    for source_problem_id, is_warmup in runs:
        raw_data = format_problem_and_options(dataset_args, source_problem_id)
        prompt = get_first_user_msg(dataset_args, raw_data)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            "seed": args.seed,
        }
        metrics_before = fetch_spec_decode_metrics(args)
        start = time.perf_counter()
        response = api_request(
            f"{normalize_base_url(args.eagle_base_url)}/chat/completions",
            args.eagle_api_key,
            args.request_timeout,
            payload,
        )
        elapsed = time.perf_counter() - start
        metrics_after = fetch_spec_decode_metrics(args)
        if is_warmup:
            continue
        output_text = response["choices"][0]["message"].get("content") or ""
        output_tokens = int(response.get("usage", {}).get("completion_tokens") or 0)
        predicted = extract_boxed_answer(output_text)
        reference = reference_answer(dataset, raw_data)
        num_drafts = metric_delta(metrics_before, metrics_after, "drafts")
        drafted_tokens = metric_delta(metrics_before, metrics_after, "draft_tokens")
        accepted_tokens = metric_delta(metrics_before, metrics_after, "accepted_tokens")
        rows.append({
            "problem_id": source_problem_id,
            "source_problem_id": source_problem_id,
            "dataset": dataset,
            "method": "eagle3",
            "mode": "eagle3",
            "backend": "openai_compatible_external",
            "runtime_comparable_to_ar": False,
            "measurement_note": "request E2E; standard API exposes no draft/verify/acceptance breakdown",
            "actual_e2e_time": elapsed,
            "actual_e2e_ms_per_output_token": 1000.0 * elapsed / output_tokens if output_tokens else math.nan,
            "output_tokens_per_ms": output_tokens / (1000.0 * elapsed) if elapsed else math.nan,
            "actual_draft_time": math.nan,
            "actual_verify_time": math.nan,
            "actual_post_verify_time": math.nan,
            "actual_algorithm_time": math.nan,
            "actual_unattributed_core_time": math.nan,
            "actual_total_time": math.nan,
            "theo_total_time": math.nan,
            "theo_draft_time": math.nan,
            "theo_verify_time": math.nan,
            "acceptance_rate_percent": 100.0 * safe_ratio(accepted_tokens, drafted_tokens),
            "output_tokens": output_tokens,
            "accepted_tokens": accepted_tokens,
            "drafted_tokens": drafted_tokens,
            "num_speculation_rounds": num_drafts,
            "total_num_forward_passes": math.nan,
            "output_token_hash": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "predicted_answer": predicted,
            "reference_answer": reference,
            "is_correct": predicted == reference if predicted is not None and reference is not None else math.nan,
            "spec_len": 5,
            "dllm_variant": math.nan,
            "frontier_mode": math.nan,
            "lowconf_threshold": math.nan,
        })
        print(f"EAGLE3 {dataset} {len(rows)}/{expected_rows}: {elapsed:.3f}s, {output_tokens} tokens", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def read_existing_rows(args):
    frames = []
    for dataset in args.datasets:
        for method in args.methods:
            path = eagle_raw_path(args, dataset) if method == "eagle3" else local_output_dir(args, dataset, method) / "benchmark_results.csv"
            if not path.exists():
                continue
            rows = pd.read_csv(path)
            if method != "eagle3":
                rows["source_problem_id"] = rows["problem_id"]
                rows["dataset"] = dataset
                rows["method"] = method
                rows["backend"] = "transformers_hf"
                rows["runtime_comparable_to_ar"] = True
                rows["measurement_note"] = "synchronized generation-core E2E; greedy decoding"
                if dataset != "gsm8k":
                    rows["is_correct"] = math.nan
                for key, value in LOCAL_METHODS[method].items():
                    rows[key] = value
            frames.append(rows)
    if not frames:
        raise RuntimeError("No benchmark results were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_sum(group, column):
    if column not in group:
        return math.nan
    return pd.to_numeric(group[column], errors="coerce").sum(min_count=1)


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def add_paired_ar_metrics(rows):
    rows = rows.copy()
    rows["actual_e2e_ms_per_output_token"] = 1000.0 * pd.to_numeric(rows["actual_e2e_time"], errors="coerce") / pd.to_numeric(rows["output_tokens"], errors="coerce")
    rows["output_tokens_per_ms"] = 1.0 / rows["actual_e2e_ms_per_output_token"]
    ar = rows[rows["method"] == "ar_only"][["dataset", "problem_id", "actual_e2e_ms_per_output_token", "output_token_hash"]].rename(
        columns={"actual_e2e_ms_per_output_token": "ar_e2e_ms_per_output_token", "output_token_hash": "ar_output_token_hash"}
    )
    rows = rows.merge(ar, on=["dataset", "problem_id"], how="left")
    rows["speedup_vs_ar_e2e"] = rows["ar_e2e_ms_per_output_token"] / rows["actual_e2e_ms_per_output_token"]
    rows["output_matches_ar"] = (rows["output_token_hash"] == rows["ar_output_token_hash"]).astype("boolean")
    comparable = rows["runtime_comparable_to_ar"].astype(str).str.lower().eq("true")
    rows.loc[~comparable, "output_matches_ar"] = pd.NA
    rows.loc[rows["method"] == "ar_only", "speedup_vs_ar_e2e"] = 1.0
    rows.loc[rows["method"] == "ar_only", "output_matches_ar"] = True
    return rows


def aggregate_group(group):
    output_tokens = numeric_sum(group, "output_tokens")
    drafted_tokens = numeric_sum(group, "drafted_tokens")
    accepted_tokens = numeric_sum(group, "accepted_tokens")
    rounds = numeric_sum(group, "num_speculation_rounds")
    passes = numeric_sum(group, "total_num_forward_passes")
    e2e_time = numeric_sum(group, "actual_e2e_time")
    algorithm_time = numeric_sum(group, "actual_algorithm_time")
    row = {
        "num_samples": len(group),
        "output_tokens": output_tokens,
        "actual_e2e_time_s": e2e_time,
        "actual_e2e_time_mean_s": pd.to_numeric(group["actual_e2e_time"], errors="coerce").mean(),
        "actual_e2e_ms_per_output_token": safe_ratio(1000.0 * e2e_time, output_tokens),
        "output_tokens_per_ms": safe_ratio(output_tokens, 1000.0 * e2e_time),
        "actual_draft_time_mean_s": pd.to_numeric(group["actual_draft_time"], errors="coerce").mean(),
        "actual_verify_time_mean_s": pd.to_numeric(group["actual_verify_time"], errors="coerce").mean(),
        "actual_computation_time_mean_s": pd.to_numeric(group["actual_post_verify_time"], errors="coerce").mean(),
        "actual_algorithm_time_mean_s": pd.to_numeric(group["actual_algorithm_time"], errors="coerce").mean(),
        "actual_algorithm_ms_per_output_token": safe_ratio(1000.0 * algorithm_time, output_tokens),
        "actual_unattributed_core_time_mean_s": pd.to_numeric(group["actual_unattributed_core_time"], errors="coerce").mean(),
        "acceptance_rate_percent": safe_ratio(100.0 * accepted_tokens, drafted_tokens),
        "drafted_tokens_per_round": safe_ratio(drafted_tokens, rounds),
        "accepted_tokens_per_round": safe_ratio(accepted_tokens, rounds),
        "output_tokens_per_round": safe_ratio(output_tokens, rounds),
        "draft_forward_passes_per_100_output_tokens": safe_ratio(100.0 * passes, output_tokens),
        "verifier_rounds_per_100_output_tokens": safe_ratio(100.0 * rounds, output_tokens),
        "theoretical_ms_per_output_token": safe_ratio(numeric_sum(group, "theo_total_time"), output_tokens),
        "output_match_rate_vs_ar_percent": 100.0 * group["output_matches_ar"].mean(),
        "parsed_accuracy_percent": 100.0 * pd.to_numeric(group["is_correct"], errors="coerce").mean(),
        "runtime_comparable_to_ar": bool(group["runtime_comparable_to_ar"].all()),
        "backend": ",".join(sorted(group["backend"].dropna().unique())),
    }
    if group["method"].iloc[0] == "ar_only":
        row["acceptance_rate_percent"] = math.nan
        row["drafted_tokens_per_round"] = math.nan
        row["accepted_tokens_per_round"] = math.nan
    return row


def build_dataset_summary(rows):
    records = []
    for (dataset, method), group in rows.groupby(["dataset", "method"], sort=False):
        record = {"dataset": dataset, "method": method}
        record.update(aggregate_group(group))
        records.append(record)
    summary = pd.DataFrame(records)
    ar_tpt = summary[summary["method"] == "ar_only"].set_index("dataset")["actual_e2e_ms_per_output_token"]
    ar_theoretical = summary[summary["method"] == "ar_only"].set_index("dataset")["theoretical_ms_per_output_token"]
    summary["speedup_vs_ar_e2e"] = summary.apply(
        lambda row: safe_ratio(ar_tpt.get(row["dataset"], math.nan), row["actual_e2e_ms_per_output_token"]), axis=1
    )
    summary["theoretical_speedup_vs_ar"] = summary.apply(
        lambda row: safe_ratio(ar_theoretical.get(row["dataset"], math.nan), row["theoretical_ms_per_output_token"]), axis=1
    )
    summary.loc[summary["method"] == "ar_only", ["speedup_vs_ar_e2e", "theoretical_speedup_vs_ar"]] = 1.0
    summary["method_order"] = summary["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return summary.sort_values(["dataset", "method_order"]).drop(columns="method_order")


def build_average_summary(dataset_summary):
    numeric_columns = [
        column for column in dataset_summary.select_dtypes(include="number").columns
        if column != "num_samples"
    ]
    average = dataset_summary.groupby("method", as_index=False)[numeric_columns].mean()
    average["datasets_completed"] = dataset_summary.groupby("method")["dataset"].nunique().values
    average["method_order"] = average["method"].map({name: index for index, name in enumerate(METHOD_ORDER)})
    return average.sort_values("method_order").drop(columns="method_order")


def write_manifest(args, output_dir):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "methods": {**LOCAL_METHODS, "eagle3": {"backend": "external OpenAI-compatible EAGLE-3 server"}},
        "primary_metric": "actual_e2e_ms_per_output_token",
        "speedup_formula": "AR aggregate E2E ms/output-token divided by method aggregate E2E ms/output-token",
        "timing_scope": "generation core after tokenized model inputs are ready through final generated token; excludes model loading, dataset loading, prompt tokenization, output decoding, plots, and file writes",
        "decoding": "greedy",
        "warmup_policy": "warmup samples excluded; adaptive controller state reset immediately before the first measured sample",
        "eagle_limitation": "standard API timing includes request overhead and exposes no draft/verify/controller/acceptance breakdown; compare EAGLE speedup only with a backend-matched AR server for publication",
    }
    with (output_dir / "benchmark_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def save_reports(args, rows):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = add_paired_ar_metrics(rows)
    dataset_summary = build_dataset_summary(rows)
    average_summary = build_average_summary(dataset_summary)
    rows.to_csv(output_dir / "e2e_per_observation.csv", index=False)
    dataset_summary.to_csv(output_dir / "e2e_dataset_summary.csv", index=False)
    average_summary.to_csv(output_dir / "e2e_average_summary.csv", index=False)
    write_manifest(args, output_dir)
    display_columns = [
        "dataset",
        "method",
        "num_samples",
        "speedup_vs_ar_e2e",
        "output_tokens_per_ms",
        "actual_e2e_ms_per_output_token",
        "actual_draft_time_mean_s",
        "actual_verify_time_mean_s",
        "actual_computation_time_mean_s",
        "actual_unattributed_core_time_mean_s",
        "acceptance_rate_percent",
        "output_tokens_per_round",
        "draft_forward_passes_per_100_output_tokens",
        "verifier_rounds_per_100_output_tokens",
        "output_match_rate_vs_ar_percent",
    ]
    print("\nE2E DATASET SUMMARY", flush=True)
    print(dataset_summary[display_columns].to_string(index=False), flush=True)
    print(f"\nSaved reports: {output_dir}", flush=True)


def main():
    args = parse_args()
    validate_args(args)
    if args.aggregate_only:
        save_reports(args, read_existing_rows(args))
        return
    frames = []
    for dataset in args.datasets:
        for method in args.methods:
            if method == "eagle3":
                frames.append(run_eagle_dataset(args, dataset))
            else:
                frames.append(run_local_method(args, dataset, method))
    save_reports(args, pd.concat(frames, ignore_index=True, sort=False))


if __name__ == "__main__":
    main()
