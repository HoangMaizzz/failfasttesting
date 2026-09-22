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
import shutil
import zipfile
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
    p.add_argument("--dllm_dir", default=str(ROOT / "Fast_dLLM_v2_1_5B"))
    p.add_argument("--target_quantization", default="none")
    p.add_argument("--target_device", default="0")
    p.add_argument("--drafter_device", default="1")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--spec_len", type=int, default=8)
    p.add_argument("--sweep_max_spec_len", type=int, default=8)
    p.add_argument("--sweep_incr_len", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--drafter_threshold", type=float, default=0.3)
    p.add_argument("--sweep_lowconf_threshold", type=float, default=0.0)
    p.add_argument("--log_level", default="INFO")
    p.add_argument(
        "--show_progress",
        action="store_true",
        help="keep generation text and progress bars visible for diagnostics",
    )
    p.add_argument("--shard_rows", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--drop_staging", action="store_true", help="remove large CSV staging after successful conversion")
    p.add_argument("--stream_raw", action="store_true", help="write compressed raw shards during inference")
    p.add_argument("--archive_each_dataset", action="store_true", help="archive each dataset immediately after inference")
    p.add_argument("--remove_raw_after_archive", action="store_true", help="remove unpacked raw shards after archiving")
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
        "--block_size", str(args.block_size),
        "--small_block_size", str(args.small_block_size),
        "--spec_len", str(args.spec_len),
        "--drafter_thresholds", str(args.drafter_threshold),
        "--sweep_lowconf_threshold", str(args.sweep_lowconf_threshold),
        "--sweep_max_spec_len", str(args.sweep_max_spec_len),
        "--sweep_incr_len", str(args.sweep_incr_len),
        "--max_new_tokens", str(args.max_new_tokens),
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
        "--skip_artifacts", "--skip_plots", "--overwrite",
        "--output_dir", str(destination), "--log_level", str(args.log_level),
    ]
    if not args.show_progress:
        command += ["--quiet_generation", "--disable_progress"]
    ids = problem_ids(args.problem_ids_file, dataset, args.num_questions)
    if ids:
        command += ["--num_questions", str(len(ids)), "--problem_ids", *map(str, ids)]
    else:
        command += ["--num_questions", str(args.num_questions)]
    if args.stream_raw:
        command += [
            "--raw_stream_dir", str(destination.parent.parent / "raw"),
            "--raw_stream_shard_rows", str(args.shard_rows),
        ]
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
    dataset_out = output / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_out / "index.jsonl"
    token_chunks: list[list[int]] = []
    before_token_chunks: list[list[int]] = []
    prefix_chunks: list[list[int]] = []
    prob_chunks: list[list[float]] = []
    mask_chunks: list[list[bool]] = []
    committed_mask_chunks: list[list[bool]] = []
    after_token_chunks: list[list[int]] = []
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
        prefix_flat = [token for prefix in prefix_chunks for token in prefix]
        prefix_offsets = [0]
        for prefix in prefix_chunks:
            prefix_offsets.append(prefix_offsets[-1] + len(prefix))
        np.savez_compressed(
            dataset_out / name,
            proposal_token_ids=np.asarray([pad(x, 0) for x in token_chunks], dtype=np.int64),
            proposal_token_ids_before_fill=np.asarray(
                [pad(x, MASK_ID) for x in before_token_chunks], dtype=np.int64
            ),
            drafter_observed_prob=np.asarray([pad(x, 0.0) for x in prob_chunks], dtype=np.float16),
            proposal_mask=np.asarray([pad(x, False) for x in mask_chunks], dtype=np.bool_),
            proposal_mask_before_fill=np.asarray(
                [pad(x, False) for x in mask_chunks], dtype=np.bool_
            ),
            committed_position_mask=np.asarray(
                [pad(x, False) for x in committed_mask_chunks], dtype=np.bool_
            ),
            proposal_token_ids_after_fill=np.asarray(
                [pad(x, 0) for x in after_token_chunks], dtype=np.int64
            ),
            prefix_token_ids_flat=np.asarray(prefix_flat, dtype=np.int32),
            prefix_token_ids_offsets=np.asarray(prefix_offsets, dtype=np.int64),
            hidden_states=np.asarray(hidden_chunks, dtype=np.float16),
            hidden_layer_indices=np.asarray(layer_chunks, dtype=np.int64),
            topk_token_ids=np.asarray(top_id_chunks, dtype=np.int64),
            topk_logits=np.asarray(top_logit_chunks, dtype=np.float16),
        )
        for offset, item in enumerate(metadata_chunks):
            item.update({"shard": name, "row": offset})
            metadata_file.write(json.dumps(item) + "\n")
        token_chunks.clear(); before_token_chunks.clear(); prefix_chunks.clear()
        prob_chunks.clear()
        mask_chunks.clear(); committed_mask_chunks.clear(); after_token_chunks.clear()
        hidden_chunks.clear(); layer_chunks.clear(); top_id_chunks.clear(); top_logit_chunks.clear()
        metadata_chunks.clear()
        shard_index += 1

    previous_by_block: dict[tuple[str, str], str] = {}
    state_count = 0
    with source.open(encoding="utf-8", newline="") as source_file:
        rows = csv.DictReader(source_file)
        for row in rows:
            proposal = [int(x) for x in parse_json(row.get("draft_proposal"))]
            proposal_before_fill = [
                int(x) for x in parse_json(row.get("proposal_token_ids_before_fill"))
            ] or proposal
            proposal_mask_before = [
                bool(x) for x in parse_json(row.get("proposal_mask_before_fill"))
            ]
            if not proposal_mask_before:
                proposal_mask_before = [token == MASK_ID for token in proposal_before_fill]
            proposal_mask_before = (proposal_mask_before + [False] * len(proposal_before_fill))[
                :len(proposal_before_fill)
            ]
            committed_position_mask = [not value for value in proposal_mask_before]
            proposal_after_fill = [
                int(x) for x in parse_json(row.get("proposal_token_ids_after_fill"))
            ] or proposal
            prefix_token_ids = [
                int(x) for x in parse_json(row.get("prefix_token_ids"))
            ]
            probabilities = [float(x) for x in parse_json(row.get("accept_probabilities"))]
            probabilities = (probabilities + [0.0] * len(proposal))[:len(proposal)]
            block = (row.get("problem_id", ""), row.get("round_id", ""))
            sid = state_id(dataset, row)
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
            "next_state_id": None,
            "dataset": dataset,
            "problem_id": int(row["problem_id"]),
            "round_id": int(row["round_id"]),
            "boundary_index": int(row.get("step") or 0),
            "context_len": int(row.get("context_len") or 0),
            "prefix_length": len(prefix_token_ids),
            "proposal_length": len(proposal),
            "accepted_len": int(row.get("accepted_len_if_stop") or 0),
            "emitted_len_if_stop": int(row.get("emitted_len_if_stop") or 0),
            "first_mismatch_convention": "accepted_prefix_length_for_greedy_verifier",
            "verifier_latency_ms": float(row.get("actual_verify_latency_ms") or 0),
            "accept_check_latency_ms": float(
                row.get("actual_accept_check_latency_ms") or 0
            ),
            "post_verify_latency_ms": float(
                row.get("actual_post_verify_latency_ms") or 0
            ),
            "hidden_state_stage": row.get("hidden_state_stage") or None,
            "hidden_state_source": row.get("hidden_state_source") or None,
            "hidden_state_forward_pass": int(
                row.get("hidden_state_forward_pass") or 0
            ),
            "termination_reason": row.get("termination_reason") or None,
            "draft_passes_elapsed": int(row.get("draft_passes_elapsed") or 0),
            "draft_latency_elapsed_ms": float(row.get("draft_latency_elapsed_ms") or 0),
            "masks_remaining": int(row.get("masks_remaining") or sum(proposal_mask_before)),
            "committed_tokens": int(row.get("committed_tokens") or sum(committed_position_mask)),
            "filled_tokens": int(row.get("filled_tokens") or sum(committed_position_mask)),
            "counterfactual_fill_tokens": int(
                row.get("counterfactual_fill_tokens") or sum(proposal_mask_before)
            ),
            "newly_unmasked": int(row.get("newly_unmasked") or 0),
            "newly_unmasked_positions": parse_json(row.get("newly_unmasked_positions")),
            "outer_action_if_stop": row.get("outer_action_if_stop") or None,
            "stop_total_latency_ms": float(row.get("stop_total_latency_ms") or 0),
            "stop_latency_per_output_token": float(
                row.get("stop_latency_per_output_token") or 0
            ),
            "stop_yield_tokens_per_ms": float(row.get("stop_yield_tokens_per_ms") or 0),
            "continue_available": row.get("continue_available") == "True",
            "continue_next_step": (
                int(row["continue_next_step"])
                if row.get("continue_next_step") not in (None, "")
                else None
            ),
            "continue_draft_delta_passes": (
                int(row["continue_draft_delta_passes"])
                if row.get("continue_draft_delta_passes") not in (None, "")
                else None
            ),
            "continue_draft_delta_latency_ms": (
                float(row["continue_draft_delta_latency_ms"])
                if row.get("continue_draft_delta_latency_ms") not in (None, "")
                else None
            ),
            "continue_total_latency_ms": (
                float(row["continue_total_latency_ms"])
                if row.get("continue_total_latency_ms") not in (None, "")
                else None
            ),
            "continue_latency_per_output_token": (
                float(row["continue_latency_per_output_token"])
                if row.get("continue_latency_per_output_token") not in (None, "")
                else None
            ),
            "continue_yield_tokens_per_ms": (
                float(row["continue_yield_tokens_per_ms"])
                if row.get("continue_yield_tokens_per_ms") not in (None, "")
                else None
            ),
            "future_yield_opportunity_label": int(
                row.get("future_yield_opportunity_label") or 0
            ),
            "one_step_latency_continue_label": (
                int(row["one_step_latency_continue_label"])
                if row.get("one_step_latency_continue_label") not in (None, "")
                else None
            ),
            "one_step_latency_oracle_action": row.get("one_step_latency_oracle_action") or None,
            "actual_action_taken": None,
            "trajectory_policy": "policy_independent_force_continue",
            "source_csv": str(source),
            })
            token_chunks.append(proposal)
            before_token_chunks.append(proposal_before_fill)
            prefix_chunks.append(prefix_token_ids)
            prob_chunks.append(probabilities)
            mask_chunks.append(proposal_mask_before)
            committed_mask_chunks.append(committed_position_mask)
            after_token_chunks.append(proposal_after_fill)
            hidden_chunks.append(hidden)
            layer_chunks.append(layers)
            top_id_chunks.append(top_ids)
            top_logit_chunks.append(top_logits)
            state_count += 1
            if len(token_chunks) >= shard_rows:
                flush()
    flush(); metadata_file.close()
    # Restore trajectory links without retaining the multi-GB CSV in memory.
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    for left, right in zip(metadata, metadata[1:]):
        if (left.get("problem_id"), left.get("round_id")) == (right.get("problem_id"), right.get("round_id")):
            left["next_state_id"] = right["state_id"]
    metadata_path.write_text(
        "".join(json.dumps(item) + "\n" for item in metadata), encoding="utf-8"
    )
    return {"states": state_count, "shards": shard_index, "source": str(source)}


def archive_dataset(dataset: str, output: Path, staging: Path) -> dict:
    dataset_out = output / "raw" / dataset
    archive_path = output / f"{dataset}_raw.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in dataset_out.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output))
        benchmark = staging / "benchmark_results.csv"
        if benchmark.exists():
            archive.write(benchmark, benchmark.relative_to(output))
    return {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "states": sum(1 for _ in (dataset_out / "index.jsonl").open(encoding="utf-8")),
        "shards": len(list(dataset_out.glob("shard_*.npz"))),
    }


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
        "mask_semantics": "proposal_mask_before_fill is the live pre-stop refinement mask",
        "prefix_storage": "prefix_token_ids_flat + prefix_token_ids_offsets per shard",
        "hidden_state_stage": "native_pre_counterfactual_fill",
        "verifier_target": "accepted_len; first_mismatch is the same greedy-prefix index",
        "latency_oracle": (
            "one-step stop-vs-next-boundary comparison using cumulative "
            "latency per emitted output token"
        ),
        "actual_action_taken": "not available; trajectory is forcibly continued",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    manifest = {"config": config, "datasets": {}}
    for dataset in args.datasets:
        staging = output / "staging" / dataset
        if not (args.resume and (staging / "bucket_oracle_snapshots.csv").exists()):
            staging.mkdir(parents=True, exist_ok=True)
            run_staging(args, dataset, staging)
        if args.stream_raw:
            dataset_out = output / "raw" / dataset
            index_path = dataset_out / "index.jsonl"
            manifest["datasets"][dataset] = {
                "states": sum(1 for _ in index_path.open(encoding="utf-8"))
                if index_path.exists() else 0,
                "shards": len(list(dataset_out.glob("shard_*.npz"))),
                "source": "streamed during inference",
            }
        else:
            manifest["datasets"][dataset] = convert_dataset(
                dataset, staging, output / "raw", args.shard_rows)
        if args.archive_each_dataset:
            manifest["datasets"][dataset] = archive_dataset(dataset, output, staging)
            (output / "manifest_partial.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
            if args.remove_raw_after_archive:
                shutil.rmtree(output / "raw" / dataset)
        if args.drop_staging:
            shutil.rmtree(staging)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
