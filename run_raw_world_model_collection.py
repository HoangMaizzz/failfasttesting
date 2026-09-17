#!/usr/bin/env python3
"""Collect causal refinement trajectories using the existing production hooks.

This is a data-collection run, not a latency benchmark.  collect_bucket_oracle
evaluates every recorded refinement boundary with the greedy verifier while the
normal FailFast trajectory remains the source of the next refinement state.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["math", "gsm8k", "humaneval"])
    p.add_argument("--num_questions", type=int, default=25)
    p.add_argument("--target_quantization", default="int8")
    p.add_argument("--target_device", default="0")
    p.add_argument("--drafter_device", default="0")
    p.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dllm_dir", default=str(ROOT / "Fast_dLLM_v2_1.5B"))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir or (ROOT / (
        "outputs_world_model_raw_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "purpose": "causal raw trajectory collection; do not report latency",
        "controller": "unmodified FailFast",
        "dense_boundary_oracle": True,
        "oracle": "greedy verifier accepted length and first mismatch",
        "storage": "production CSV/JSONL hooks; no pickle-only dataset",
    })
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for dataset in args.datasets:
        destination = out / "raw" / dataset
        command = [sys.executable, "failfast.py",
                   "--dataset_name", dataset,
                   "--num_questions", str(args.num_questions),
                   "--benchmark_modes", "dllm_ar",
                   "--dllm_variant", "failfast",
                   "--decoding_strategy", "greedy",
                   "--max_new_tokens", "1024",
                   "--spec_len", "8", "--block_size", "32",
                   "--small_block_size", "8",
                   "--target_model_name", args.target_model_name,
                   "--dllm_dir", args.dllm_dir,
                   "--target_device", args.target_device,
                   "--drafter_device", args.drafter_device,
                   "--target_quantization", args.target_quantization,
                   "--drafter_thresholds", "0.3",
                   "--sweep_lowconf_threshold", "0.5",
                   "--sweep_max_spec_len", "64", "--sweep_incr_len", "8",
                   "--seed", "42", "--log_verifier_calls",
                   "--collect_draft_diagnostics", "--collect_bucket_oracle",
                   "--quiet_generation", "--disable_progress",
                   "--skip_artifacts", "--skip_plots", "--overwrite",
                   "--output_dir", str(destination), "--log_level", "INFO"]
        if args.resume:
            command[-2:] = ["--output_dir", str(destination)]
        run(command)

    manifest = {"config": config, "datasets": {}}
    for dataset in args.datasets:
        d = out / "raw" / dataset
        files = sorted(str(p.relative_to(out)) for p in d.rglob("*") if p.is_file())
        manifest["datasets"][dataset] = {
            "files": files,
            "oracle_csv": [x for x in files if x.endswith("bucket_oracle_snapshots.csv")],
            "trajectory_csv": [x for x in files if x.endswith("frontier_round_diagnostics.csv")],
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", out.parent, out.name)
    print(f"\nRAW WORLD-MODEL DATASET: {archive}", flush=True)


if __name__ == "__main__":
    main()
