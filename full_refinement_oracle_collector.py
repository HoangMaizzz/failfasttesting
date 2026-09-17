#!/usr/bin/env python3
"""Policy-independent raw-state/oracle collection driver.

The repository's production hook already records a snapshot after each native
Fast-dLLM refinement boundary and evaluates the snapshot with the real greedy
verifier.  This driver runs that hook with a fixed 8-token block, low-confidence
threshold zero, and an 8-token hard cap.  Consequently neither FailFast's
confidence stop nor a learned controller can shorten the native trajectory.

The staging CSVs remain available for audit.  The lightweight tensors are
converted to sharded NPZ files so downstream code never has to load one giant
pickle/array.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MASK_ID = 151665


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["math", "gsm8k", "humaneval"])
    p.add_argument("--num_questions", type=int, default=25)
    p.add_argument("--problem_ids_file", type=Path)
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dllm_dir", default=str(ROOT / "Fast_dLLM_v2_1.5B"))
    p.add_argument("--target_quantization", default="int8")
    p.add_argument("--target_device", default="0")
    p.add_argument("--drafter_device", default="0")
    p.add_argument("--shard_rows", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def problem_ids(path: Path | None, dataset: str, count: int) -> list[int] | None:
    if path is None:
        return None
    result: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [x.strip() for x in raw.replace(",", " ").split()]
        if len(parts) == 1 or parts[0].lower() == dataset.lower():
            result.append(int(parts[-1]))
    return result[:count]


def run_staging(args: argparse.Namespace, dataset: str, destination: Path) -> None:
    command = [
        sys.executable, "failfast.py",
        "--dataset_name", dataset,
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--block_size", "32",
        "--small_block_size", "8",
        "--spec_len", "8",
        "--drafter_thresholds", "0.3",
        "--sweep_lowconf_threshold", "0.0",
        "--sweep_max_spec_len", "8",
        "--sweep_incr_len", "8",
        "--max_new_tokens", "1024",
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--seed", str(args.seed),
        "--collect_draft_diagnostics",
        "--collect_bucket_oracle",
        "--full_refinement_oracle",
        "--bucket_oracle_force_continue",
        "--quiet_generation", "--disable_progress",
        "--skip_artifacts", "--skip_plots", "--overwrite",
        "--output_dir", str(destination), "--log_level", "INFO",
    ]
    ids = problem_ids(args.problem_ids_file, dataset, args.num_questions)
    if ids:
        command += ["--problem_ids", *map(str, ids)]
    else:
        command += ["--num_questions", str(args.num_questions)]
    print(">>>", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_json(value: str | None):
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def state_id(dataset: str, row: dict[str, str]) -> str:
    raw = f"{dataset}|{row.get('problem_id')}|{row.get('round_id')}|{row.get('step')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def convert_dataset(dataset: str, staging: Path, output: Path, shard_rows: int) -> dict:
    source = staging / "bucket_oracle_snapshots.csv"
    if not source.exists():
        raise FileNotFoundError(f"collector produced no {source}")
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    dataset_out = output / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_out / "index.jsonl"
    token_chunks: list[list[int]] = []
    prob_chunks: list[list[float]] = []
    mask_chunks: list[list[bool]] = []
    hidden_chunks: list[list] = []
    layer_chunks: list[list[int]] = []
    top_id_chunks: list[list] = []
    top_logit_chunks: list[list] = []
    metadata_chunks: list[dict] = []
    shard_index = 0
    metadata_file = metadata_path.open("w", encoding="utf-8")

    def flush() -> None:
        nonlocal shard_index
        if not token_chunks:
            return
        name = f"shard_{shard_index:05d}.npz"
        width = max(map(len, token_chunks))
        pad = lambda values, fill: values + [fill] * (width - len(values))
        np.savez_compressed(
            dataset_out / name,
            proposal_token_ids=np.asarray([pad(x, 0) for x in token_chunks], dtype=np.int64),
            drafter_observed_prob=np.asarray([pad(x, 0.0) for x in prob_chunks], dtype=np.float16),
            proposal_mask=np.asarray([pad(x, False) for x in mask_chunks], dtype=np.bool_),
            hidden_states=np.asarray(hidden_chunks, dtype=np.float16),
            hidden_layer_indices=np.asarray(layer_chunks, dtype=np.int64),
            topk_token_ids=np.asarray(top_id_chunks, dtype=np.int64),
            topk_logits=np.asarray(top_logit_chunks, dtype=np.float16),
        )
        for offset, item in enumerate(metadata_chunks):
            item.update({"shard": name, "row": offset})
            metadata_file.write(json.dumps(item) + "\n")
        token_chunks.clear(); prob_chunks.clear(); mask_chunks.clear()
        hidden_chunks.clear(); layer_chunks.clear(); top_id_chunks.clear(); top_logit_chunks.clear()
        metadata_chunks.clear()
        shard_index += 1

    previous_by_block: dict[tuple[str, str], str] = {}
    ids = [state_id(dataset, row) for row in rows]
    next_ids = {}
    for index, row in enumerate(rows[:-1]):
        if (row.get("problem_id"), row.get("round_id")) == (
            rows[index + 1].get("problem_id"), rows[index + 1].get("round_id")
        ):
            next_ids[index] = ids[index + 1]
    for index, row in enumerate(rows):
        proposal = [int(x) for x in parse_json(row.get("draft_proposal"))]
        probabilities = [float(x) for x in parse_json(row.get("accept_probabilities"))]
        probabilities = (probabilities + [0.0] * len(proposal))[:len(proposal)]
        block = (row.get("problem_id", ""), row.get("round_id", ""))
        sid = ids[index]
        previous = previous_by_block.get(block)
        previous_by_block[block] = sid
        hidden = parse_json(row.get("hidden_states"))
        layers = [int(x) for x in parse_json(row.get("hidden_layer_indices"))]
        top_ids = parse_json(row.get("topk_token_ids"))
        top_logits = parse_json(row.get("topk_logits"))
        if not hidden or not layers or not top_ids or not top_logits:
            raise RuntimeError("full raw collection row is missing hidden/top-K tensors")
        metadata_chunks.append({
            "state_id": sid,
            "previous_state_id": previous,
            "next_state_id": next_ids.get(index),
            "dataset": dataset,
            "problem_id": int(row["problem_id"]),
            "round_id": int(row["round_id"]),
            "boundary_index": int(row.get("step") or 0),
            "context_len": int(row.get("context_len") or 0),
            "proposal_length": len(proposal),
            "accepted_len": int(row.get("accepted_len_if_stop") or 0),
            "first_mismatch": int(row.get("accepted_len_if_stop") or 0),
            "verifier_latency_ms": float(row.get("actual_verify_latency_ms") or 0),
            "source_csv": str(source),
        })
        token_chunks.append(proposal)
        prob_chunks.append(probabilities)
        mask_chunks.append([token == MASK_ID for token in proposal])
        hidden_chunks.append(hidden)
        layer_chunks.append(layers)
        top_id_chunks.append(top_ids)
        top_logit_chunks.append(top_logits)
        if len(token_chunks) >= shard_rows:
            flush()
    flush(); metadata_file.close()
    return {"states": len(rows), "shards": shard_index, "source": str(source)}


def main() -> None:
    args = args_parser()
    output = args.output_dir or ROOT / (
        "world_model_raw_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config["output_dir"] = str(output)
    config.update({
        "collector": "full_refinement_oracle",
        "policy_control": "none",
        "native_hard_cap": 8,
        "lowconf_threshold": 0.0,
        "oracle": "real greedy verifier at every recorded refinement boundary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    manifest = {"config": config, "datasets": {}}
    for dataset in args.datasets:
        staging = output / "staging" / dataset
        if not (args.resume and (staging / "bucket_oracle_snapshots.csv").exists()):
            staging.mkdir(parents=True, exist_ok=True)
            run_staging(args, dataset, staging)
        manifest["datasets"][dataset] = convert_dataset(
            dataset, staging, output / "raw", args.shard_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
