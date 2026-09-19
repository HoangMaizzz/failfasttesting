"""Matched MATH FailFast benchmark: author N=10 versus current N=8."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_u1_sgd_ablation import selected_ids

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--num_questions", type=int, default=100)
    p.add_argument("--id_offset", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--two_gpu", action="store_true")
    p.add_argument("--no_verifier_cache", action="store_true")
    return p.parse_args()


def run(cmd, log_path=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    stream = subprocess.PIPE
    log = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log:
                log.write(line)
                log.flush()
        code = proc.wait()
    finally:
        if log:
            log.close()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def command(args, ids, n, max_spec_len, destination):
    cmd = [
        sys.executable, "-u", "failfast.py",
        "--dataset_name", "math", "--num_questions", str(len(ids)),
        "--problem_ids", *map(str, ids), "--warmup_questions", "0",
        "--benchmark_modes", "dllm_ar", "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy", "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", str(n), "--block_size", "32", "--small_block_size", "8",
        # Use the downloaded local model because benchmark execution is offline.
        "--target_model_name", str(args.assets / "target"),
        "--target_model_label", "Qwen2.5-7B-Instruct",
        "--dllm_dir", str(args.assets / "drafter"),
        "--dataset_dir", str(args.assets / "datasets" / "math"),
        "--target_device", "0", "--drafter_device", "1" if args.two_gpu else "0",
        "--target_quantization", "none", "--unquantized_dtype", "float16",
        "--drafter_thresholds", "0.5", "--sweep_lowconf_threshold", "0.7",
        "--sweep_max_spec_len", str(max_spec_len), "--sweep_incr_len", str(n),
        "--seed", "42", "--quiet_generation", "--disable_progress",
        "--skip_artifacts", "--skip_plots", "--log_verifier_calls", "--overwrite",
        "--output_dir", str(destination), "--log_level", "INFO",
    ]
    if args.two_gpu:
        cmd.append("--target_two_gpu_fp16")
    if not args.no_verifier_cache:
        cmd.append("--verifier_kv_cache")
    return cmd


def main():
    args = parse_args()
    args.assets = args.assets.resolve()
    output = (args.output or ROOT / "outputs" /
              ("math_failfast_n10_n8_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        raise FileExistsError(output)
    required = [args.assets / "target" / "config.json",
                args.assets / "drafter" / "config.json",
                args.assets / "datasets" / "math" / "state.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))
    id_args = argparse.Namespace(num_questions=args.num_questions, id_offset=args.id_offset)
    ids = selected_ids(id_args, "math")
    output.mkdir(parents=True)
    manifest = {
        "dataset": "math", "num_questions": len(ids), "id_offset": args.id_offset,
        "problem_ids": ids, "physical_block_size": 8,
        "drafter_threshold": 0.5, "lowconf_threshold": 0.7,
        "target_dtype": "float16", "target_quantization": "none",
        "verifier_cache": not args.no_verifier_cache,
        "configs": {"n10_author": {"spec_len": 10, "incr_len": 10, "max_spec_len": 60},
                    "n8_current": {"spec_len": 8, "incr_len": 8, "max_spec_len": 64}},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        run([sys.executable, "-u", "patch_fastdllm_frontier.py", str(args.assets / "drafter")],
            output / "patch.log")
        for name, n, cap in (("n10_author", 10, 60), ("n8_current", 8, 64)):
            destination = output / name / "raw" / "math" / "failfast"
            destination.mkdir(parents=True)
            cmd = command(args, ids, n, cap, destination)
            (destination / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
            run(cmd, destination / "run.log")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        archive = shutil.make_archive(str(output), "zip", output.parent, output.name)
        print("RESULTS:", output, flush=True)
        print("ARCHIVE:", archive, flush=True)


if __name__ == "__main__":
    main()
