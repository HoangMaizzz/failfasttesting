import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from run_corrected_greedy_oracle_two_datasets import run_dataset
from run_otrc_v2_td_benchmark import PROBLEM_IDS


ROOT = Path(__file__).resolve().parent
VERSION = "corrected_one_action_greedy_oracle_four_datasets_v1"
REFERENCE_VERSION = "corrected_one_action_greedy_oracle_two_datasets_v1"
REFERENCE_DATASETS = ("math", "gsm8k")
MEASURED_DATASETS = ("aime", "humaneval")
DEFAULT_REFERENCE_ZIP = (
    ROOT
    / "benchmark_references"
    / "corrected_greedy_oracle_math_gsm8k_reports.zip"
)
REFERENCE_FILES = (
    "corrected_oracle_manifest.json",
    "corrected_oracle_feature_labels.csv",
    "greedy_local_oracle_dataset_report.csv",
    "greedy_local_oracle_summary.csv",
    "verifier_profile.json",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=25)
    parser.add_argument("--paired_repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epsilon_ms", type=float, default=1.0)
    parser.add_argument(
        "--reference_oracle_zip",
        default=str(DEFAULT_REFERENCE_ZIP),
        help=(
            "Reference pack containing the corrected 50-question MATH and "
            "GSM8K oracle reports. These datasets are imported, not rerun."
        ),
    )
    parser.add_argument(
        "--dllm_dir",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/content/failfasttesting/"
            "outputs_corrected_greedy_oracle_four_datasets"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_archive", action="store_true")
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def dataset_configuration(dataset, num_questions):
    if dataset not in MEASURED_DATASETS:
        raise ValueError(f"unsupported measured dataset: {dataset}")
    available = PROBLEM_IDS[dataset]
    if num_questions <= 0 or num_questions > len(available):
        raise ValueError(
            f"{dataset} num_questions must be in [1, {len(available)}]"
        )
    source = {
        "dataset": dataset,
        "max_new_tokens": 1024,
        "block_size": 32,
        "small_block_size": 8,
        "target_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "drafter_threshold": 0.05,
        "lowconf_threshold": 0.45,
        "max_spec_len": 60,
        "seed": 42,
    }
    return source, [int(value) for value in available[:num_questions]]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_zip_entry(archive, suffix):
    normalized = suffix.replace("\\", "/")
    matches = [
        entry
        for entry in archive.infolist()
        if not entry.is_dir()
        and (
            entry.filename == normalized
            or entry.filename.endswith("/" + normalized)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"reference ZIP must contain exactly one {normalized}; "
            f"found {len(matches)}"
        )
    return matches[0]


def read_zip_json(archive, suffix):
    entry = find_zip_entry(archive, suffix)
    with archive.open(entry) as handle:
        return json.load(handle)


def read_zip_csv(archive, suffix):
    entry = find_zip_entry(archive, suffix)
    with archive.open(entry) as handle:
        return pd.read_csv(handle)


def copy_zip_entry(archive, suffix, destination):
    entry = find_zip_entry(archive, suffix)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(entry) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def import_reference_oracles(reference_zip, output_dir):
    reference_zip = Path(reference_zip)
    if not reference_zip.exists():
        raise FileNotFoundError(
            f"corrected MATH/GSM8K oracle reference not found: {reference_zip}"
        )
    imported = {}
    with zipfile.ZipFile(reference_zip) as archive:
        top_manifest = read_zip_json(archive, "benchmark_manifest.json")
        if top_manifest.get("version") != REFERENCE_VERSION:
            raise ValueError(
                "reference oracle has an incompatible version: "
                f"{top_manifest.get('version')!r}"
            )
        if set(top_manifest.get("datasets", [])) != set(REFERENCE_DATASETS):
            raise ValueError("reference oracle must contain MATH and GSM8K")

        for dataset in REFERENCE_DATASETS:
            prefix = f"{dataset}/"
            manifest = read_zip_json(
                archive,
                prefix + "corrected_oracle_manifest.json",
            )
            problem_ids = [int(value) for value in manifest["problem_ids"]]
            if len(problem_ids) != 50 or len(set(problem_ids)) != 50:
                raise ValueError(
                    f"reference {dataset} oracle must contain 50 unique IDs"
                )
            summary = read_zip_csv(
                archive,
                prefix + "greedy_local_oracle_summary.csv",
            )
            report = read_zip_csv(
                archive,
                prefix + "greedy_local_oracle_dataset_report.csv",
            )
            labels = read_zip_csv(
                archive,
                prefix + "corrected_oracle_feature_labels.csv",
            )
            if set(summary.problem_id.astype(int)) != set(problem_ids):
                raise ValueError(f"reference {dataset} summary ID mismatch")
            if len(report) != 1 or labels.empty:
                raise ValueError(f"reference {dataset} report is incomplete")

            dataset_dir = Path(output_dir) / dataset
            for filename in REFERENCE_FILES:
                copy_zip_entry(
                    archive,
                    prefix + filename,
                    dataset_dir / filename,
                )
            imported[dataset] = {
                "problem_ids": problem_ids,
                "num_questions": len(problem_ids),
                "source": "bundled_corrected_oracle_reference",
            }
    return imported, top_manifest


def load_dataset_outputs(output_dir, dataset, expected_count):
    dataset_dir = Path(output_dir) / dataset
    report = pd.read_csv(dataset_dir / "greedy_local_oracle_dataset_report.csv")
    summary = pd.read_csv(dataset_dir / "greedy_local_oracle_summary.csv")
    labels = pd.read_csv(dataset_dir / "corrected_oracle_feature_labels.csv")
    manifest = json.loads(
        (dataset_dir / "corrected_oracle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    problem_ids = [int(value) for value in manifest["problem_ids"]]
    if len(problem_ids) != expected_count or len(set(problem_ids)) != expected_count:
        raise ValueError(f"{dataset} manifest does not contain {expected_count} IDs")
    if len(report) != 1 or len(summary) != expected_count or labels.empty:
        raise ValueError(f"{dataset} oracle output is incomplete")
    if set(summary.problem_id.astype(int)) != set(problem_ids):
        raise ValueError(f"{dataset} summary ID mismatch")

    report = report.copy()
    summary = summary.copy()
    labels = labels.copy()
    if "dataset" not in report:
        report.insert(0, "dataset", dataset)
    if "dataset" not in summary:
        summary.insert(0, "dataset", dataset)
    if "dataset" not in labels:
        labels.insert(0, "dataset", dataset)
    return report, summary, labels, manifest


def aggregate_overall(dataset_reports):
    reports = pd.concat(dataset_reports, ignore_index=True)
    baseline = float(reports.baseline_real_latency_ms.sum())
    oracle = float(reports.greedy_real_latency_ms.sum())
    tokens = float(reports.total_generated_tokens.sum())
    speedups = reports.pooled_real_speedup.astype(float).clip(lower=1e-12)
    row = {
        "datasets": len(reports),
        "num_samples": int(reports.num_samples.sum()),
        "total_generated_tokens": tokens,
        "baseline_real_latency_ms": baseline,
        "oracle_real_latency_ms": oracle,
        "pooled_real_speedup": baseline / oracle,
        "latency_reduction_percent": 100.0 * (1.0 - oracle / baseline),
        "baseline_ms_per_token": baseline / tokens,
        "oracle_ms_per_token": oracle / tokens,
        "baseline_tokens_per_second": 1000.0 * tokens / baseline,
        "oracle_tokens_per_second": 1000.0 * tokens / oracle,
        "macro_speedup_arithmetic": float(speedups.mean()),
        "macro_speedup_geometric": float(
            math.exp(float(speedups.map(math.log).mean()))
        ),
        "baseline_verifier_calls": float(reports.baseline_verifier_calls.sum()),
        "oracle_verifier_calls": float(reports.greedy_verifier_calls.sum()),
        "baseline_dLLM_forwards": float(reports.baseline_dLLM_forwards.sum()),
        "oracle_dLLM_forwards": float(reports.greedy_dLLM_forwards.sum()),
        "output_hash_match_percent": (
            float(
                (
                    reports.output_hash_match_percent
                    * reports.num_samples
                ).sum()
            )
            / max(1.0, float(reports.num_samples.sum()))
        ),
    }
    return reports, pd.DataFrame([row])


def build_combined_reports(output_dir, expected_counts):
    reports = []
    summaries = []
    labels = []
    manifests = {}
    for dataset, count in expected_counts.items():
        report, summary, decision_labels, manifest = load_dataset_outputs(
            output_dir,
            dataset,
            count,
        )
        reports.append(report)
        summaries.append(summary)
        labels.append(decision_labels)
        manifests[dataset] = manifest

    dataset_report, overall = aggregate_overall(reports)
    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_labels = pd.concat(labels, ignore_index=True)
    dataset_report.to_csv(
        Path(output_dir) / "four_dataset_corrected_oracle_dataset_report.csv",
        index=False,
    )
    overall.to_csv(
        Path(output_dir) / "four_dataset_corrected_oracle_overall.csv",
        index=False,
    )
    combined_summary.to_csv(
        Path(output_dir) / "four_dataset_corrected_oracle_summary.csv",
        index=False,
    )
    combined_labels.to_csv(
        Path(output_dir) / "four_dataset_corrected_oracle_feature_labels.csv",
        index=False,
    )
    return dataset_report, overall, manifests


def main():
    args = parse_args()
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.epsilon_ms < 0.0:
        raise ValueError("--epsilon_ms must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    imported, reference_manifest = import_reference_oracles(
        args.reference_oracle_zip,
        output_dir,
    )
    measured = {}
    for dataset in MEASURED_DATASETS:
        source, problem_ids = dataset_configuration(dataset, args.num_questions)
        print("\n" + "=" * 100, flush=True)
        print(
            f"DATASET {dataset.upper()} | questions={len(problem_ids)} | "
            "oracle=corrected_one_action_baseline_rollout",
            flush=True,
        )
        print("=" * 100, flush=True)
        report, _ = run_dataset(args, dataset, source, problem_ids)
        if len(report) != 1:
            raise RuntimeError(f"{dataset} did not produce one dataset report")
        measured[dataset] = {
            "problem_ids": problem_ids,
            "num_questions": len(problem_ids),
            "source": "measured_in_this_run",
        }

    expected_counts = {
        **{dataset: value["num_questions"] for dataset, value in imported.items()},
        **{dataset: value["num_questions"] for dataset, value in measured.items()},
    }
    dataset_report, overall, manifests = build_combined_reports(
        output_dir,
        expected_counts,
    )
    benchmark_manifest = {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "oracle_definition": (
            "Corrected one-action greedy local STOP/CONTINUE oracle. The "
            "current action is oracle-controlled and subsequent actions use "
            "FailFast-8 until the next verifier boundary."
        ),
        "datasets": {**imported, **measured},
        "reference_oracle_zip": str(Path(args.reference_oracle_zip).resolve()),
        "reference_oracle_sha256": sha256_file(args.reference_oracle_zip),
        "reference_manifest": reference_manifest,
        "per_dataset_manifests": manifests,
        "paired_repetitions_for_new_datasets": args.paired_repetitions,
        "epsilon_ms": args.epsilon_ms,
        "elapsed_hours_new_work": (time.time() - started) / 3600.0,
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(benchmark_manifest, indent=2),
        encoding="utf-8",
    )

    archive_path = None
    if not args.skip_archive:
        archive_path = shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    print("\nFOUR-DATASET CORRECTED ORACLE REPORT", flush=True)
    print(dataset_report.to_string(index=False), flush=True)
    print("\nFOUR-DATASET POOLED SUMMARY", flush=True)
    print(overall.to_string(index=False), flush=True)
    print(f"\nSaved: {output_dir}", flush=True)
    if archive_path:
        print(f"Archive: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
