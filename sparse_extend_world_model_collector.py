#!/usr/bin/env python3
"""Collect a bounded Extend/refinement graph from a raw-state ZIP backbone.

The ZIP backbone supplies native masked states and the existing raw feature
schema.  Each Extend candidate is refined by the dLLM for up to N denoising
forwards.  Submit labels are computed by comparing with one cached greedy
verifier continuation per round; verifier latency for longer proposals is an
explicit estimate scaled from the ZIP's measured 8-token verifier timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
import zipfile
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MASK_ID = 151665


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backbone_zip", type=Path, required=True,
        help="raw-state ZIP or directory containing its extracted index/shards",
    )
    p.add_argument("--dataset", choices=("math", "gsm8k"), default="gsm8k")
    p.add_argument("--num_questions", type=int, default=3)
    p.add_argument("--max_rounds_per_question", type=int, default=1,
                   help="0 means use every round for each selected question")
    p.add_argument("--root_boundaries", type=int, default=4,
                   help="number of saved 8-token backbone boundaries per round")
    p.add_argument("--extend_size", type=int, default=8)
    p.add_argument("--max_proposal_tokens", type=int, default=64)
    p.add_argument("--max_unmask_passes", type=int, default=4,
                   help="one default unmask plus up to three additional unmask forwards")
    p.add_argument("--branch_width", type=int, default=4)
    p.add_argument("--physical_block_size", type=int, default=32)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--drafter_threshold", type=float, default=0.3)
    p.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dllm_dir", required=True)
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=1)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--shard_rows", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_unpacked_index(folder: Path, dataset: str) -> Path:
    candidates = list(folder.rglob("index.jsonl"))
    raw_candidates = [
        p for p in candidates
        if p.parent.name.lower() == dataset.lower()
        and p.parent.parent.name.lower() == "raw"
    ]
    if raw_candidates:
        candidates = raw_candidates
    else:
        dataset_candidates = [p for p in candidates if p.parent.name.lower() == dataset.lower()]
        if dataset_candidates:
            candidates = dataset_candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {dataset} raw index.jsonl under {folder}; found: "
            f"{[str(p) for p in candidates]}"
        )
    return candidates[0]


def _select_backbone_rows(rows: list[dict], num_questions: int,
                          max_rounds: int, root_boundaries: int) -> list[dict]:
    by_question: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        problem_id = int(row["problem_id"])
        if problem_id < num_questions and int(row.get("proposal_length", 0)) == 8:
            by_question[problem_id].append(row)
    selected = []
    for problem_id in range(num_questions):
        rounds: dict[int, list[dict]] = defaultdict(list)
        for row in by_question.get(problem_id, []):
            rounds[int(row["round_id"])].append(row)
        round_ids = sorted(rounds)
        if max_rounds > 0:
            round_ids = round_ids[:max_rounds]
        for round_id in round_ids:
            candidates = sorted(
                rounds[round_id],
                key=lambda row: (int(row.get("boundary_index", 0)), row["state_id"]),
            )
            selected.extend(candidates[:root_boundaries])
    return selected


def _hydrate_backbone_rows(selected: list[dict], dataset: str,
                           shard_loader) -> list[dict]:
    shard_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
    result = []
    for row in selected:
        shard_key = str(row["shard"])
        if shard_key not in shard_cache:
            shard_cache[shard_key] = shard_loader(row["shard"])
            while len(shard_cache) > 3:
                shard_cache.popitem(last=False)
        else:
            shard_cache.move_to_end(shard_key)
        arrays = shard_cache[shard_key]
        i = int(row["row"])
        offsets = arrays["prefix_token_ids_offsets"]
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        record = dict(row)
        record["prefix_token_ids"] = arrays["prefix_token_ids_flat"][lo:hi].astype(np.int64).tolist()
        for name in (
            "proposal_token_ids", "proposal_token_ids_before_fill",
            "proposal_token_ids_after_fill", "proposal_mask_before_fill",
            "committed_position_mask", "drafter_observed_prob",
            "hidden_states", "hidden_layer_indices", "topk_token_ids", "topk_logits",
        ):
            if name in arrays:
                record[name] = arrays[name][i].tolist()
        width = int(record["proposal_length"])
        for name in (
            "proposal_token_ids", "proposal_token_ids_before_fill",
            "proposal_token_ids_after_fill", "proposal_mask_before_fill",
            "committed_position_mask", "drafter_observed_prob",
        ):
            if name in record:
                record[name] = record[name][:width]
        result.append(record)
    if not result:
        raise RuntimeError("No 8-token backbone states selected; check dataset/problem IDs")
    return result


def _load_backbone_rows(backbone_path: Path, dataset: str, num_questions: int,
                        max_rounds: int, root_boundaries: int) -> list[dict]:
    """Load selected states directly from either the ZIP or its extracted folder."""
    if backbone_path.is_dir():
        index_path = _find_unpacked_index(backbone_path, dataset)
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        selected = _select_backbone_rows(rows, num_questions, max_rounds, root_boundaries)

        def load_shard(shard):
            shard_path = Path(shard)
            candidates = [
                shard_path if shard_path.is_absolute() else index_path.parent / shard_path,
                backbone_path / shard_path,
                backbone_path / "raw" / dataset / shard_path.name,
            ]
            actual = next((p for p in candidates if p.is_file()), None)
            if actual is None:
                raise FileNotFoundError(f"Shard {shard!r} not found under {backbone_path}")
            with np.load(actual, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}

        return _hydrate_backbone_rows(selected, dataset, load_shard)

    if not zipfile.is_zipfile(backbone_path):
        raise ValueError(f"Backbone input is neither a ZIP nor a directory: {backbone_path}")
    with zipfile.ZipFile(backbone_path) as archive:
        index_name = f"raw/{dataset}/index.jsonl"
        if index_name not in archive.namelist():
            raise FileNotFoundError(f"{index_name} missing from {backbone_path}")
        rows = [json.loads(line) for line in archive.read(index_name).splitlines()]
        selected = _select_backbone_rows(rows, num_questions, max_rounds, root_boundaries)

        def load_shard(shard):
            member = f"raw/{dataset}/{Path(shard).name}"
            with archive.open(member) as handle:
                with np.load(handle, allow_pickle=False) as data:
                    return {name: data[name] for name in data.files}

        return _hydrate_backbone_rows(selected, dataset, load_shard)


def _load_models(args):
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_name,
        torch_dtype=torch.float16,
        device_map={"": args.target_device},
        attn_implementation="sdpa",
    )
    # Compatibility hooks used by the repository's own failfast loader.
    import transformers.modeling_rope_utils as rope_utils
    import transformers.modeling_utils as modeling_utils
    patched_rope = False
    original_tied = getattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def custom_rope_init_fn(config, device, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 1000000.0)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            return inv, 1.0
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = custom_rope_init_fn
        patched_rope = True
    if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys"):
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}
    try:
        drafter = AutoModelForCausalLM.from_pretrained(
            args.dllm_dir,
            torch_dtype=torch.float16,
            device_map={"": args.drafter_device},
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        drafter.lm_head.weight = drafter.model.embed_tokens.weight
    finally:
        if patched_rope and "default" in rope_utils.ROPE_INIT_FUNCTIONS:
            del rope_utils.ROPE_INIT_FUNCTIONS["default"]
        if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys") and original_tied is not None:
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = original_tied
    return tokenizer, target, drafter


def _greedy_reference(model, prefix: list[int], tokenizer, max_tokens: int) -> list[int]:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [int(x) for x in output[0, len(prefix):].tolist()]


def _score_greedy(proposal: list[int], reference: list[int], eos_id: int | None,
                  remaining_output_budget: int) -> tuple[int, int, list[int]]:
    accepted = 0
    while accepted < min(len(proposal), len(reference)) and proposal[accepted] == reference[accepted]:
        accepted += 1
    if accepted < len(proposal) and accepted < len(reference):
        emitted = proposal[:accepted] + [reference[accepted]]
    elif accepted == len(proposal) and accepted < len(reference):
        emitted = proposal[:accepted] + [reference[accepted]]
    else:
        emitted = proposal[:accepted]
    if eos_id is not None and eos_id in emitted:
        emitted = emitted[:emitted.index(eos_id) + 1]
    emitted = emitted[:max(0, int(remaining_output_budget))]
    return min(accepted, len(emitted)), len(emitted), emitted


def _load_verifier_profile(backbone_path: Path, dataset: str) -> list[dict]:
    """Load measured L=8 verifier timings from the ZIP index or extracted index."""
    if backbone_path.is_dir():
        index_path = _find_unpacked_index(backbone_path, dataset)
        index_lines = index_path.read_text(encoding="utf-8").splitlines()
    else:
        archive = zipfile.ZipFile(backbone_path)
        index_name = f"raw/{dataset}/index.jsonl"
        if index_name not in archive.namelist():
            archive.close()
            raise FileNotFoundError(f"{index_name} missing from {backbone_path}")
        index_lines = archive.read(index_name).decode("utf-8").splitlines()
        archive.close()
    selected = []
    for line in index_lines:
        row = json.loads(line)
        if int(row.get("proposal_length", 0)) == 8:
            selected.append(row)
    return selected


def _verifier_profile(rows: list[dict], context_len: int) -> tuple[float, float]:
    """Median measured L=8 verifier/post time near this context length."""
    candidates = rows
    if not candidates:
        raise RuntimeError("Backbone ZIP has no verifier latency samples for L=8")
    nearest = min(abs(int(r.get("context_len", 0)) - context_len) for r in candidates)
    pool = [r for r in candidates if abs(int(r.get("context_len", 0)) - context_len) <= max(128, nearest)]
    return (
        statistics.median(float(r["verifier_latency_ms"]) for r in pool),
        statistics.median(float(r.get("post_verify_latency_ms", 0.0)) for r in pool),
    )


def _state_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:20]


def _save_child(writer, child: dict) -> None:
    meta = child["metadata"]
    writer.append({
        "proposal_token_ids": child["filled_tokens"],
        "proposal_token_ids_before_fill": child["native_tokens"],
        "proposal_token_ids_after_fill": child["filled_tokens"],
        "drafter_observed_prob": child["probabilities"],
        "proposal_mask": child["mask"],
        "proposal_mask_before_fill": child["mask"],
        "committed_position_mask": [not x for x in child["mask"]],
        # Child features describe the newly extended 8-token segment. Parent
        # feature tensors remain on the parent node and are linked by ID.
        "hidden_states": child["hidden_states"],
        "hidden_layer_indices": child["hidden_layer_indices"],
        "topk_token_ids": child["topk_token_ids"],
        "topk_logits": child["topk_logits"],
        "prefix_token_ids": child["prefix_token_ids"],
        "metadata": meta,
    })


def _merge_feature_rows(parent_rows: list, update_rows: list, update_offset: int,
                        total_length: int) -> list:
    """Overlay current native-forward features on the inherited state rows."""
    merged = [list(vector) for vector in parent_rows]
    offset = int(update_offset)
    if offset < 0 or offset > len(merged):
        raise RuntimeError(
            f"feature update offset {offset} is outside parent feature length {len(merged)}"
        )
    for relative, vector in enumerate(update_rows):
        position = offset + relative
        if position > len(merged):
            raise RuntimeError("feature update left a gap in proposal positions")
        if position == len(merged):
            merged.append(list(vector))
        else:
            merged[position] = list(vector)
    if len(merged) != int(total_length):
        raise RuntimeError(
            f"composed feature length {len(merged)} != proposal length {total_length}"
        )
    return merged


def _merge_layer_features(parent_layers: list, update_layers: list,
                          update_offset: int, total_length: int) -> list:
    if len(parent_layers) != len(update_layers):
        raise RuntimeError("hidden/top-K feature layer count changed across branch")
    return [
        _merge_feature_rows(parent, update, update_offset, total_length)
        for parent, update in zip(parent_layers, update_layers)
    ]


def _extend_one(args, drafter, tokenizer, prefix: list[int], parent: dict,
                writer, reference: list[int], verifier8_ms: float,
                post_ms: float, eos_id: int | None, remaining: int,
                step_counter: list[int]) -> list[dict]:
    full_native = list(parent["native_tokens"])
    prompt_ids = prefix + full_native
    device = drafter.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    call_args = SimpleNamespace(
        target_tokenizer=tokenizer,
        full_refinement_oracle=True,
        raw_top_k=32,
        collect_bucket_oracle=True,
        frontier_stop_mode="disabled",
        adaptive_td=False,
        adaptive_freeze=False,
        global_oracle_graph=False,
        strict_greedy_local_oracle=False,
        bucket_acceptance_calibration={},
        bucket_min_observations=8,
        bucket_prior_strength=8.0,
        # The dLLM gain estimator expects a mapping even when there is no
        # precomputed calibration. Keep per-call tables local to this branch.
        bucket_gain_calibration={
            "length_score_masks": {},
            "score_masks": {},
            "length_score": {},
            "score": {},
            "step": {},
            "global": [0.0, 0],
        },
        bucket_current_context_len=len(prompt_ids),
        collector_parent_native_length=len(full_native),
        collector_parent_fill_tokens=list(parent["filled_tokens"]),
    )
    started = time.perf_counter()
    with torch.inference_mode():
        result = drafter.generate_draft_tokens_arbitrary_length(
            input_ids,
            max_new_tokens=3 * args.physical_block_size,
            small_block_size=args.small_block_size,
            block_size=args.physical_block_size,
            threshold=args.drafter_threshold,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            is_drafter=True,
            spec_len=args.extend_size,
            return_prefill_kvs=True,
            prev_prefill_output=None,
            args=call_args,
            lowconf_threshold=0.0,
            max_spec_len=args.extend_size,
            incr_len=args.extend_size,
            last_round_rejected=None,
            return_frontier_stats=True,
            max_denoising_passes=args.max_unmask_passes,
        )
    wall_ms = (time.perf_counter() - started) * 1000.0
    generated, _, _, _, latencies, stats = result
    snapshots = sorted(
        stats.get("oracle_refinement_snapshots", []),
        key=lambda s: int(s.get("unmask_forward_index", 0)),
    )
    children = []
    for snap in snapshots[:args.max_unmask_passes]:
        ext_native = [int(x) for x in snap["proposal_token_ids_before_fill"]]
        ext_filled = [int(x) for x in snap["proposal_token_ids_after_fill"]]
        if len(ext_native) != args.extend_size:
            continue
        native = [int(x) for x in (
            snap.get("collector_full_proposal_token_ids_before_fill")
            or full_native + ext_native
        )]
        filled = [int(x) for x in (
            snap.get("collector_full_proposal_token_ids_after_fill")
            or list(parent["filled_tokens"]) + ext_filled
        )]
        if len(native) != len(filled) or len(native) != len(full_native) + args.extend_size:
            raise RuntimeError("collector snapshot did not preserve the full parent+extension state")
        if len(filled) > args.max_proposal_tokens:
            continue
        mask = [int(x) == MASK_ID for x in native]
        accepted, emitted, emitted_ids = _score_greedy(
            filled, reference, eos_id, remaining
        )
        action_ms = float(snap.get("unmask_forward_latency_ms", 0.0))
        extension_elapsed_ms = float(
            snap.get("denoising_latency_elapsed_ms", action_ms)
        )
        cumulative_draft = float(parent["cumulative_draft_latency_ms"]) + extension_elapsed_ms
        proposal_verify_est = verifier8_ms * len(filled) / 8.0
        total = cumulative_draft + proposal_verify_est + post_ms
        yield_per_ms = emitted / max(total, 1e-9)
        step_counter[0] += 1
        uid = _state_id(
            parent["metadata"]["state_id"],
            len(filled),
            snap.get("unmask_forward_index"),
            step_counter[0],
        )
        feature_offset = int(snap.get("collector_full_feature_start_offset", len(full_native)))
        full_hidden = snap.get("collector_full_hidden_states")
        full_topk_ids = snap.get("collector_full_topk_token_ids")
        full_topk_logits = snap.get("collector_full_topk_logits")
        if full_hidden and full_topk_ids and full_topk_logits:
            hidden_states = _merge_layer_features(
                parent["hidden_states"], full_hidden, feature_offset, len(native)
            )
            topk_token_ids = _merge_feature_rows(
                parent["topk_token_ids"], full_topk_ids, feature_offset, len(native)
            )
            topk_logits = _merge_feature_rows(
                parent["topk_logits"], full_topk_logits, feature_offset, len(native)
            )
        else:
            hidden_states = list(parent["hidden_states"]) + list(
                snap.get("hidden_states") or []
            )
            topk_token_ids = list(parent["topk_token_ids"]) + list(
                snap.get("topk_token_ids") or []
            )
            topk_logits = list(parent["topk_logits"]) + list(
                snap.get("topk_logits") or []
            )
        child = {
            "native_tokens": native,
            "filled_tokens": filled,
            "mask": mask,
            "probabilities": list(parent["probabilities"]) + [
                float(x) for x in snap.get("accept_probabilities", [])
            ][:args.extend_size],
            "hidden_states": hidden_states,
            "hidden_layer_indices": list(parent["hidden_layer_indices"]),
            "topk_token_ids": topk_token_ids,
            "topk_logits": topk_logits,
            "prefix_token_ids": list(prefix),
            "cumulative_draft_latency_ms": cumulative_draft,
            "yield_tokens_per_ms": yield_per_ms,
            "metadata": {
                "state_id": uid,
                "parent_state_id": parent["metadata"]["state_id"],
                "previous_state_id": parent["metadata"]["state_id"],
                "child_state_ids": [],
                "dataset": args.dataset,
                "problem_id": int(parent["metadata"]["problem_id"]),
                "round_id": int(parent["metadata"]["round_id"]),
                "boundary_index": int(snap.get("step", 0)),
                "context_len": len(prefix),
                "prefix_length": len(prefix),
                "proposal_length": len(filled),
                "extension_size": args.extend_size,
                "extend_depth": len(filled) // args.extend_size - 1,
                "unmask_passes_since_extend": int(snap.get("unmask_forward_index", 0)),
                "extra_unmask_passes": max(0, int(snap.get("unmask_forward_index", 0)) - 1),
                "action": "extend_then_refine",
                "accepted_len": accepted,
                "emitted_len_if_stop": emitted,
                "emitted_token_ids_if_stop": emitted_ids,
                "verifier_latency_ms": proposal_verify_est,
                "verifier_latency_source": "backbone_l8_median_scaled_by_proposal_length",
                "verifier_latency_l8_ms": verifier8_ms,
                "post_verify_latency_ms": post_ms,
                "stop_total_latency_ms": total,
                "stop_latency_per_output_token": total / max(emitted, 1),
                "stop_yield_tokens_per_ms": yield_per_ms,
                "incoming_parent_stop_yield_tokens_per_ms": float(parent["yield_tokens_per_ms"]),
                "incoming_continue_yield_tokens_per_ms": yield_per_ms,
                "incoming_one_step_latency_continue_label": int(
                    yield_per_ms > float(parent["yield_tokens_per_ms"])
                ),
                "incoming_latency_yield_gain_tokens_per_ms": (
                    yield_per_ms - float(parent["yield_tokens_per_ms"])
                ),
                "one_unmask_latency_ms": action_ms,
                "cumulative_draft_latency_ms": cumulative_draft,
                "replayed_prefix_wall_latency_ms": wall_ms,
                "state_replay_mode": "prefix_plus_native_mask_tokens_without_saved_kv",
                "reference_policy": "cached_greedy_target_continuation",
                "reference_length": len(reference),
                "feature_scope": "full_proposal_composed_parent_plus_current_native_block_refresh",
                "feature_update_start_offset": feature_offset,
                "feature_update_token_count": len(full_hidden[0]) if full_hidden else args.extend_size,
                "hidden_state_stage": snap.get("hidden_state_stage"),
                "hidden_state_source": snap.get("hidden_state_source"),
                "hidden_state_forward_pass": snap.get("hidden_state_forward_pass"),
                "masks_remaining": sum(mask),
                "committed_tokens": len(mask) - sum(mask),
                "termination_reason": (
                    "collector_pass_limit"
                    if stats.get("collector_pass_limit_reached")
                    else stats.get("native_termination_reason")
                ),
                "selected_for_expansion": False,
                "selection_reason": None,
            },
        }
        if not child["hidden_states"] or not child["topk_token_ids"] or not child["topk_logits"]:
            raise RuntimeError("dLLM extension snapshot omitted raw hidden/top-K features")
        children.append(child)
    return children


def _select_beam(candidates: list[dict], width: int) -> list[dict]:
    """Deterministic one-per-parent pass, then fill by yield/token objective."""
    ordered = sorted(candidates, key=lambda n: (-n["yield_tokens_per_ms"], n["metadata"]["state_id"]))
    chosen, chosen_ids, parents = [], set(), set()
    for node in ordered:
        parent_id = node["metadata"]["parent_state_id"]
        if parent_id in parents:
            continue
        node["metadata"]["selected_for_expansion"] = True
        node["metadata"]["selection_reason"] = "best_yield_from_distinct_parent"
        chosen.append(node); chosen_ids.add(node["metadata"]["state_id"]); parents.add(parent_id)
        if len(chosen) >= width:
            return chosen
    for node in ordered:
        if node["metadata"]["state_id"] in chosen_ids:
            continue
        node["metadata"]["selected_for_expansion"] = True
        node["metadata"]["selection_reason"] = "next_best_yield"
        chosen.append(node); chosen_ids.add(node["metadata"]["state_id"])
        if len(chosen) >= width:
            break
    return chosen


def collect(args) -> dict:
    if args.extend_size <= 0 or args.max_proposal_tokens < 8:
        raise ValueError("extend_size must be positive and max proposal length at least 8")
    if args.max_unmask_passes != 4:
        # Kept configurable for future experiments; this smoke protocol is
        # explicitly one default pass plus three additional passes.
        if args.max_unmask_passes < 1:
            raise ValueError("max_unmask_passes must be >= 1")
    roots = _load_backbone_rows(
        args.backbone_zip, args.dataset, args.num_questions,
        args.max_rounds_per_question, args.root_boundaries,
    )
    tokenizer, target, drafter = _load_models(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_dir / "raw"
    from raw_stream_writer import RawShardWriter
    writer = RawShardWriter(raw_root, f"{args.dataset}_extend_graph", args.shard_rows)
    references: dict[tuple[int, int], list[int]] = {}
    verifier_profiles = _load_verifier_profile(args.backbone_zip, args.dataset)
    root_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for root in roots:
        key = (int(root["problem_id"]), int(root["round_id"]))
        root_groups[key].append(root)
    graph_summary = {"config": vars(args).copy(), "roots": len(roots), "levels": [], "created_utc": datetime.now(timezone.utc).isoformat()}
    step_counter = [0]
    for (problem_id, round_id), group in sorted(root_groups.items()):
        prefix = [int(x) for x in group[0]["prefix_token_ids"]]
        if any([int(x) for x in r["prefix_token_ids"]] != prefix for r in group):
            raise RuntimeError("backbone states in one round do not share the same verifier prefix")
        ref_key = (problem_id, round_id)
        if ref_key not in references:
            references[ref_key] = _greedy_reference(
                target, prefix, tokenizer, args.max_proposal_tokens + 1
            )
        reference = references[ref_key]
        verifier8_ms, post_ms = _verifier_profile(verifier_profiles, len(prefix))
        active = []
        for row in sorted(group, key=lambda r: (int(r.get("boundary_index", 0)), r["state_id"])):
            native = [int(x) for x in row["proposal_token_ids_before_fill"]]
            filled = [int(x) for x in row["proposal_token_ids_after_fill"]]
            base_accept, base_emitted, _ = _score_greedy(
                filled, reference, tokenizer.eos_token_id, args.max_proposal_tokens
            )
            if base_accept != int(row["accepted_len"]):
                raise RuntimeError(
                    f"greedy reference disagrees with ZIP state {row['state_id']}: "
                    f"reference={base_accept}, archive={row['accepted_len']}"
                )
            root_node = {
                "native_tokens": native,
                "filled_tokens": filled,
                "mask": [int(x) == MASK_ID for x in native],
                "probabilities": [float(x) for x in row.get("drafter_observed_prob", [])],
                "hidden_states": row["hidden_states"],
                "hidden_layer_indices": row["hidden_layer_indices"],
                "topk_token_ids": row["topk_token_ids"],
                "topk_logits": row["topk_logits"],
                "cumulative_draft_latency_ms": float(row["draft_latency_elapsed_ms"]),
                "yield_tokens_per_ms": float(row["stop_yield_tokens_per_ms"]),
                "metadata": {
                    **row,
                    "state_id": row["state_id"],
                },
            }
            active.append(root_node)

        max_depth = max(0, math.ceil((args.max_proposal_tokens - 8) / args.extend_size))
        for depth in range(1, max_depth + 1):
            expansion_parents = list(active)
            candidate_pool = []
            for parent in expansion_parents:
                if len(parent["filled_tokens"]) + args.extend_size > args.max_proposal_tokens:
                    continue
                candidate_pool.extend(_extend_one(
                    args, drafter, tokenizer, prefix, parent, writer, reference,
                    verifier8_ms, post_ms, tokenizer.eos_token_id,
                    args.max_proposal_tokens, step_counter,
                ))
            active = _select_beam(candidate_pool, args.branch_width)
            children_by_parent: dict[str, list[str]] = defaultdict(list)
            for candidate in candidate_pool:
                children_by_parent[candidate["metadata"]["parent_state_id"]].append(
                    candidate["metadata"]["state_id"]
                )
            parent_by_id = {
                node["metadata"]["state_id"]: node
                for node in expansion_parents
            }
            for parent_id, child_ids in children_by_parent.items():
                parent_node = parent_by_id.get(parent_id)
                if parent_node is not None:
                    parent_node["metadata"]["child_state_ids"] = child_ids
                    child_nodes = [
                        node for node in candidate_pool
                        if node["metadata"]["parent_state_id"] == parent_id
                    ]
                    best_continue = max(
                        (node["yield_tokens_per_ms"] for node in child_nodes),
                        default=float(parent_node["yield_tokens_per_ms"]),
                    )
                    parent_node["metadata"]["best_one_step_continue_yield_tokens_per_ms"] = best_continue
                    parent_node["metadata"]["one_step_latency_continue_label"] = int(
                        best_continue > float(parent_node["yield_tokens_per_ms"])
                    )
                    parent_node["metadata"]["oracle_latency_yield_action"] = (
                        "continue" if best_continue > float(parent_node["yield_tokens_per_ms"])
                        else "stop"
                    )
                    for child in child_nodes:
                        child["metadata"]["parent_best_continue_yield_tokens_per_ms"] = best_continue
                        child["metadata"]["parent_one_step_latency_continue_label"] = int(
                            best_continue > float(parent_node["yield_tokens_per_ms"])
                        )
                        child["metadata"]["parent_oracle_latency_yield_action"] = (
                            "continue" if best_continue > float(parent_node["yield_tokens_per_ms"])
                            else "stop"
                        )
            for candidate in candidate_pool:
                _save_child(writer, candidate)
            graph_summary["levels"].append({
                "problem_id": problem_id,
                "round_id": round_id,
                "extend_depth": depth,
                "candidate_states": len(candidate_pool),
                "selected_for_expansion": len(active),
                "proposal_length": 8 + depth * args.extend_size,
            })
            print(
                f"[graph] problem={problem_id} round={round_id} depth={depth} "
                f"candidates={len(candidate_pool)} selected={len(active)}",
                flush=True,
            )
            if not active:
                break
        del reference
    writer.flush()
    graph_summary["child_states"] = step_counter[0]
    if args.backbone_zip.is_dir():
        source_index = _find_unpacked_index(args.backbone_zip, args.dataset)
        graph_summary["backbone_source_index_sha256"] = sha256_file(source_index)
    else:
        graph_summary["backbone_source_sha256"] = sha256_file(args.backbone_zip)
    graph_summary["backbone_source"] = str(args.backbone_zip)
    graph_summary["submit_label_source"] = "cached_greedy_reference_lcp"
    graph_summary["verifier_latency_note"] = (
        "L=8 latency measured in backbone ZIP; longer proposals are scaled estimates, not per-node verifier measurements"
    )
    (args.output_dir / "graph_manifest.json").write_text(
        json.dumps(graph_summary, indent=2, default=str), encoding="utf-8"
    )
    archive_path = args.output_dir / f"{args.dataset}_extend_graph.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in raw_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir))
        archive.write(args.output_dir / "graph_manifest.json", "graph_manifest.json")
    print(json.dumps({
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "roots": len(roots),
        "child_states": step_counter[0],
    }, indent=2), flush=True)
    return graph_summary


def main() -> None:
    args = parse_args()
    collect(args)


if __name__ == "__main__":
    main()
