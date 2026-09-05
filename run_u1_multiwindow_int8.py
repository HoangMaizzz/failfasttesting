"""Resumable U1 dynamic-threshold benchmark, one GPU process per problem."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_u1_sgd_ablation import command as base_command, selected_ids

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["math", "gsm8k", "humaneval"], default=["math", "gsm8k", "humaneval"])
    p.add_argument("--num_questions", type=int, default=100)
    p.add_argument("--id_offset", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--dllm_dir", default=str(ROOT / "Fast_dLLM_v2_1.5B"))
    p.add_argument("--output_dir", default=str(ROOT / "outputs_u1_multiwindow_int8_test100"))
    p.add_argument("--soft_probe", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single_process", action="store_true",
                   help="Keep models loaded across all questions in each dataset.")
    args = p.parse_args()
    args.target_quantization = "int8"
    args.target_device = args.drafter_device = 0
    args.drafter_threshold = 0.5
    args.lowconf_threshold = 0.7
    args.continue_threshold = 0.5
    args.replay_stop_to_continue_ratio = 0.0
    return args


def problem_command(args, dataset, problem_id, destination, checkpoint):
    cmd = base_command(args, dataset, "u1_batch1x", destination)
    start = cmd.index("--problem_ids") + 1
    end = cmd.index("--warmup_questions")
    cmd[start:end] = [str(problem_id)]
    cmd[cmd.index("--num_questions") + 1] = "1"
    cmd[cmd.index("--warmup_questions") + 1] = "0"
    cmd.remove("--no-adaptive-hindsight-logistic-dynamic-threshold")
    cmd.extend([
        "--unquantized_dtype", "float16",
        "--adaptive-hindsight-logistic-dynamic-threshold",
        "--adaptive-hindsight-logistic-dynamic-windows", "50,60,100",
        "--adaptive-hindsight-logistic-dynamic-min-selected", "7",
        "--adaptive-hindsight-logistic-dynamic-se-beta", "0.30",
        "--adaptive-hindsight-logistic-dynamic-use-ipw",
        "--adaptive-hindsight-logistic-dynamic-ipw-clip", "50",
        "--log_verifier_calls",
    ])
    if checkpoint is not None:
        cmd.extend(["--adaptive-state-path", str(checkpoint)])
    return cmd


def run_logged(cmd, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise


def dataset_command(args, dataset, problem_ids, destination):
    cmd = problem_command(args, dataset, problem_ids[0], destination, None)
    start = cmd.index("--problem_ids") + 1
    end = cmd.index("--warmup_questions")
    cmd[start:end] = list(map(str, problem_ids))
    cmd[cmd.index("--num_questions") + 1] = str(len(problem_ids))
    return cmd


def run_dataset(args, output, dataset, problem_ids):
    destination = output / "raw" / dataset / "continuous"
    marker = destination / "complete.json"
    if marker.exists():
        data = pd.read_csv(destination / "benchmark_results.csv")
        if data.problem_id.tolist() != problem_ids:
            raise ValueError("Completed dataset has unexpected IDs")
        return
    if destination.exists():
        destination.rename(destination.with_name(f"continuous_incomplete_{time.time_ns()}"))
        print("Restarting incomplete dataset from its first question; old files preserved.", flush=True)
    destination.mkdir(parents=True)
    cmd = dataset_command(args, dataset, problem_ids, destination)
    (destination / "command.json").write_text(json.dumps(cmd, indent=2))
    print(f"\n{dataset.upper()} | {len(problem_ids)} questions | load models ONCE", flush=True)
    run_logged(cmd, destination / "run.log")
    data = pd.read_csv(destination / "benchmark_results.csv")
    if data.problem_id.tolist() != problem_ids:
        raise ValueError("Unexpected output problem IDs")
    marker.write_text(json.dumps({"problem_ids": problem_ids}))


def summarize(output, ids):
    rows = []
    for dataset, problem_ids in ids.items():
        frames = []
        continuous = output / "raw" / dataset / "continuous" / "benchmark_results.csv"
        if continuous.exists() and continuous.stat().st_size:
            frame = pd.read_csv(continuous)
            frame["dataset"] = dataset
            frames.append(frame)
        for pid in problem_ids:
            directory = output / "raw" / dataset / f"id_{pid}"
            if (directory / "complete.json").exists():
                frame = pd.read_csv(directory / "benchmark_results.csv")
                frame["dataset"] = dataset
                frames.append(frame)
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        data.to_csv(output / f"{dataset}_benchmark_results.csv", index=False)
        seconds = float(data.actual_algorithm_time.sum())
        tokens = int(data.output_tokens.sum())
        rows.append({"dataset": dataset, "method": "u1_batch1x_multiwindow",
                     "completed": len(data), "requested": len(problem_ids),
                     "algorithm_seconds": seconds, "output_tokens": tokens,
                     "ms_per_token": 1000 * seconds / max(tokens, 1),
                     "peak_allocated_gib": float(data.gpu_peak_allocated_gib.max()),
                     "peak_reserved_gib": float(data.gpu_peak_reserved_gib.max())})
    pd.DataFrame(rows).to_csv(output / "dataset_method_summary.csv", index=False)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    ids = {d: selected_ids(args, d) for d in args.datasets}
    fingerprint = hashlib.sha256()
    for name in ("adaptive_td.py", "failfast.py", "run_u1_sgd_ablation.py", "Fast_dLLM_v2_1_5B/modeling.py", Path(__file__).name):
        fingerprint.update((ROOT / name).read_bytes())
    manifest = {"configuration": {k: v for k, v in vars(args).items() if k != "resume"},
                "problem_ids": ids, "source_sha256": fingerprint.hexdigest(),
                "windows": [50, 60, 100], "min_selected": 7, "se_beta": 0.3,
                "ipw_clip": 50, "unquantized_dtype": "float16",
                "restart_each_problem": not args.single_process, "warmup_questions": 0,
                "timing_note": "Generation timing excludes model loading; no cross-hardware speedup claim."}
    path = output / "manifest.json"
    if output.exists():
        if not args.resume or not path.exists() or json.loads(path.read_text()) != manifest:
            raise ValueError("Existing output: use --resume with identical code/config, or choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    subprocess.run([sys.executable, "patch_fastdllm_frontier.py", args.dllm_dir], cwd=ROOT, check=True)
    try:
        for dataset, problem_ids in ids.items():
            if args.single_process:
                run_dataset(args, output, dataset, problem_ids)
                summarize(output, ids)
                print("ARCHIVE:", shutil.make_archive(str(output), "zip", output.parent, output.name), flush=True)
                continue
            checkpoint = None
            for index, pid in enumerate(problem_ids, 1):
                destination = output / "raw" / dataset / f"id_{pid}"
                destination.mkdir(parents=True, exist_ok=True)
                state = destination / "adaptive_td_runtime_state.json"
                marker = destination / "complete.json"
                print(f"\n{dataset.upper()} {index}/{len(problem_ids)} | ID={pid}", flush=True)
                if not marker.exists():
                    if any(destination.iterdir()):
                        destination.rename(destination.with_name(f"id_{pid}_incomplete_{time.time_ns()}"))
                        destination.mkdir()
                    cmd = problem_command(args, dataset, pid, destination, checkpoint)
                    (destination / "command.json").write_text(json.dumps(cmd, indent=2))
                    run_logged(cmd, destination / "run.log")
                    data = pd.read_csv(destination / "benchmark_results.csv")
                    if data.problem_id.tolist() != [pid]:
                        raise ValueError("Unexpected output problem IDs")
                    saved = json.loads(state.read_text())
                    if "logistic_boundary_checkpoint" not in saved:
                        raise ValueError("Missing U1 learner checkpoint")
                    marker.write_text(json.dumps({"problem_id": pid}))
                elif not state.exists():
                    raise ValueError(f"Completed question has no checkpoint: {destination}")
                checkpoint = state
            summarize(output, ids)
            print("ARCHIVE:", shutil.make_archive(str(output), "zip", output.parent, output.name), flush=True)
    finally:
        summarize(output, ids)
        print("ARCHIVE:", shutil.make_archive(str(output), "zip", output.parent, output.name), flush=True)


if __name__ == "__main__":
    main()
