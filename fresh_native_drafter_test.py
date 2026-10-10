#!/usr/bin/env python3
"""Collect fresh native Fast-dLLM R/E trajectories and audit a predictive latent.

This is deliberately a drafter-only experiment. It runs native Fast-dLLM with
greedy sampling, threshold 0.5, 8-token logical blocks and up to three R steps.
At each block boundary it top-1 fills the previous proposal (the same zero-cost
STOP snapshot used by the collector) and asks the native dLLM for the next
8-token segment. No verifier is loaded or queried.

The audit compares a wider direct predictor, a 128D latent direct predictor,
and an action-conditioned 128D latent transition evaluated free-running through
H1-H3. All splits are question-disjoint. Candidate-content targets are learned
token embeddings from Fast-dLLM, not random token identity codes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_sufficiency_audit import (
    DirectPredictor, Edge, LatentDynamics, State, _average_metric_dict,
    _collect_direct, _collect_rollout, _fit_direct, _fit_dynamics,
    _paths_for_edges, _restrict_edges, normalize_states, question_split,
    score_records,
)
from native_elysia_graph import NativeElysiaRunner, NativeEosWithoutSnapshot


MASK_ID = 151665
SCHEMA = "fresh_native_drafter_latent_v1"


class ShardWriter:
    """Small bounded-memory writer; metadata is updated with shard/row on flush."""

    def __init__(self, root: Path, shard_rows: int = 32):
        self.root = root
        self.feature_dir = root / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = int(shard_rows)
        self.rows: list[dict] = []
        self.serial = 0
        self.index = root / "states.jsonl"
        self.index.write_text("", encoding="utf-8")

    def append(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        name = f"shard_{self.serial:05d}.npz"
        payload = {
            "x": np.stack([row["x"] for row in self.rows]).astype(np.float16),
            "mask": np.stack([row["mask"] for row in self.rows]).astype(np.bool_),
            "valid": np.stack([row["valid"] for row in self.rows]).astype(np.bool_),
            "confidence": np.stack([row["confidence"] for row in self.rows]).astype(np.float16),
            "changed": np.stack([row["changed"] for row in self.rows]).astype(np.bool_),
            "content": np.stack([row["content"] for row in self.rows]).astype(np.float16),
            "native_token_ids": np.stack([row["native_token_ids"] for row in self.rows]).astype(np.int32),
            "filled_token_ids": np.stack([row["filled"] for row in self.rows]).astype(np.int32),
            "candidate_token_ids": np.stack([row["candidate_token_ids"] for row in self.rows]).astype(np.int32),
            "topk_token_ids": np.stack([row["topk_token_ids"] for row in self.rows]).astype(np.int32),
            "topk_logits": np.stack([row["topk_logits"] for row in self.rows]).astype(np.float16),
        }
        np.savez_compressed(self.feature_dir / name, **payload)
        with self.index.open("a", encoding="utf-8") as handle:
            for offset, row in enumerate(self.rows):
                meta = dict(row["meta"])
                meta.update(shard=f"features/{name}", row=offset)
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.rows.clear()
        self.serial += 1


def _load_drafter(model_dir: Path, device: int):
    """Load the repo-instrumented Fast-dLLM without allocating the verifier."""
    import transformers.modeling_rope_utils as rope_utils
    import transformers.modeling_utils as modeling_utils

    patched_rope = False
    original_tied = getattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def custom_rope_init_fn(config, target_device, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 1_000_000.0)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                                device=target_device) / dim))
            return inv, 1.0
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = custom_rope_init_fn
        patched_rope = True
    if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys"):
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float16,
            device_map={"": int(device)}, trust_remote_code=True,
            local_files_only=True, attn_implementation="sdpa",
        )
        # Fast-dLLM's released checkpoint uses a tied input/output embedding.
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight
    finally:
        if patched_rope and "default" in rope_utils.ROPE_INIT_FUNCTIONS:
            del rope_utils.ROPE_INIT_FUNCTIONS["default"]
        if original_tied is not None:
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = original_tied
    return model.eval()


def _chat_prefix(tokenizer, question: str) -> list[int]:
    user_text = (
        "Solve the following math problem efficiently and clearly. Please reason step by step, "
        "separate logical reasoning steps with two newline characters ("
        "\n\n"
        "), and put your "
        "final answer within \\boxed{}.\nProblem: " + question
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True, add_generation_prompt=True,
    )
    return [int(token) for token in ids]


def _aligned_row(rows, offset: int, local_index: int):
    row_index = int(local_index) - int(offset)
    if row_index < 0 or row_index >= len(rows):
        return None
    return rows[row_index]


@torch.inference_mode()
def _state_row(snapshot: dict, model, context_len: int, block_index: int,
               max_blocks: int, eos_id: int, hidden_size: int) -> dict:
    native = np.asarray(snapshot["proposal_token_ids_before_fill"], dtype=np.int64)
    filled = np.asarray(snapshot["proposal_token_ids_after_fill"], dtype=np.int64)
    width = len(native)
    if width != 8 or len(filled) != width:
        raise ValueError(f"Expected 8-token native snapshot, got {width}/{len(filled)}")
    mask = np.asarray(snapshot["proposal_mask_before_fill"], dtype=np.bool_)
    if len(mask) != width or not np.array_equal(mask, native == MASK_ID):
        raise ValueError("Native mask does not match proposal token IDs")

    hidden_rows = snapshot.get("hidden_states") or []
    if not hidden_rows:
        raise ValueError("Native snapshot has no hidden states")
    hidden = np.asarray(hidden_rows[-1], dtype=np.float32)
    hidden_offset = int(snapshot.get("native_hidden_start_offset", 0))
    top_ids_rows = snapshot.get("topk_token_ids") or []
    top_logits_rows = snapshot.get("topk_logits") or []
    top_offset = int(snapshot.get("native_topk_start_offset", 0))
    if len(top_ids_rows) != len(top_logits_rows):
        raise ValueError("Native top-k token/logit row counts differ")
    topk = max((len(x) for x in top_ids_rows), default=0)
    if topk < 1:
        raise ValueError("Native snapshot has no top-k candidate")

    top_ids = np.full((width, topk), -1, dtype=np.int32)
    top_logits = np.zeros((width, topk), dtype=np.float32)
    hidden_aligned = np.zeros((width, hidden_size), dtype=np.float32)
    feature_valid = np.zeros(width, dtype=np.bool_)
    top1_ids = np.full(width, -1, dtype=np.int32)
    for pos in range(width):
        hrow = _aligned_row(hidden, hidden_offset, pos)
        irow = _aligned_row(top_ids_rows, top_offset, pos)
        lrow = _aligned_row(top_logits_rows, top_offset, pos)
        if hrow is None or irow is None or lrow is None:
            continue
        hrow = np.asarray(hrow, dtype=np.float32)
        if hrow.size != hidden_size or len(irow) != len(lrow) or not len(irow):
            continue
        count = min(topk, len(irow))
        ids = np.asarray(irow[:count], dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= model.get_input_embeddings().weight.shape[0]):
            continue
        hidden_aligned[pos] = hrow
        top_ids[pos, :count] = ids.astype(np.int32)
        top_logits[pos, :count] = np.asarray(lrow[:count], dtype=np.float32)
        top1_ids[pos] = int(ids[0])
        feature_valid[pos] = True

    confidence = np.asarray(snapshot.get("confidences", []), dtype=np.float32)
    margin = np.asarray(snapshot.get("margins", []), dtype=np.float32)
    if len(confidence) != width:
        raise ValueError("Native confidence length is not 8")
    if len(margin) != width:
        margin = np.zeros(width, dtype=np.float32)

    device = model.get_input_embeddings().weight.device
    embedding = model.get_input_embeddings()
    safe_native = torch.as_tensor(np.maximum(native, 0), device=device, dtype=torch.long)
    token_embedding = embedding(safe_native).float().cpu().numpy()
    expected_candidate = np.zeros((width, hidden_size), dtype=np.float32)
    candidate_top1_embedding = np.zeros((width, hidden_size), dtype=np.float32)
    entropy = np.zeros(width, dtype=np.float32)
    valid_positions = np.flatnonzero(feature_valid)
    if len(valid_positions):
        candidate_ids = top_ids[valid_positions].astype(np.int64)
        candidate_valid = candidate_ids >= 0
        safe_candidate_ids = torch.as_tensor(np.maximum(candidate_ids, 0), device=device)
        candidate_logits = torch.as_tensor(
            top_logits[valid_positions], device=device, dtype=torch.float32)
        candidate_valid_t = torch.as_tensor(candidate_valid, device=device)
        candidate_logits = candidate_logits.masked_fill(~candidate_valid_t, -1e4)
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        candidate_vectors = embedding(safe_candidate_ids).float()
        expected_candidate[valid_positions] = (
            candidate_probs.unsqueeze(-1) * candidate_vectors
        ).sum(1).cpu().numpy()
        candidate_top1_embedding[valid_positions] = candidate_vectors[:, 0].cpu().numpy()
        candidate_counts = candidate_valid.sum(-1)
        entropy_values = -(candidate_probs * torch.log(candidate_probs.clamp_min(1e-12))).sum(-1)
        entropy_values = entropy_values.cpu().numpy()
        multi = candidate_counts > 1
        entropy[valid_positions[multi]] = entropy_values[multi] / np.log(candidate_counts[multi])

    newly = set(int(x) for x in snapshot.get("newly_unmasked_positions", []))
    scalars = np.stack([
        mask.astype(np.float32), confidence,
        margin, entropy,
        np.arange(width, dtype=np.float32) / max(1, width - 1),
        np.full(width, block_index / max(1, max_blocks - 1), np.float32),
        np.full(width, float(snapshot.get("unmask_forward_index", 0)) / 16.0, np.float32),
        np.full(width, min(context_len, 4096) / 4096.0, np.float32),
        np.asarray([float(i in newly) for i in range(width)], dtype=np.float32),
        feature_valid.astype(np.float32),
    ], axis=-1)
    x = np.concatenate([hidden_aligned, token_embedding, expected_candidate, scalars], axis=-1)
    valid = feature_valid.copy()
    if eos_id in native:
        valid[int(np.flatnonzero(native == eos_id)[0]) + 1:] = False
    return dict(
        x=x.astype(np.float16), mask=mask, valid=valid,
        confidence=confidence, changed=np.zeros(width, dtype=np.bool_),
        content=candidate_top1_embedding.astype(np.float16),
        native_token_ids=native.astype(np.int32), candidate_token_ids=top1_ids,
        topk_token_ids=top_ids, topk_logits=top_logits.astype(np.float16),
        filled=filled.astype(np.int64),
        eos_in_native=bool(eos_id in native), eos_in_filled=bool(eos_id in filled),
        masks_remaining=int(mask.sum()), feature_valid=feature_valid,
        hidden_offset=hidden_offset, topk_offset=top_offset,
    )


def _make_edge(src: dict, dst: dict, action: str, states: dict[str, State]) -> Edge:
    src_state, dst_state = states[src["state_id"]], states[dst["state_id"]]
    valid = src_state.eligible & dst_state.eligible
    if action == "R":
        valid &= src_state.mask
        changed_valid = valid.copy()
        src_state_candidate = src["candidate_token_ids"]
        dst_state.changed = src_state_candidate != dst["candidate_token_ids"]
        # Unknown candidate positions are not valid labels.
        dst_state.changed &= (src_state_candidate >= 0) & (dst["candidate_token_ids"] >= 0)
    else:
        changed_valid = np.zeros_like(valid)
    new_suffix = np.ones_like(valid) if action == "E" else np.zeros_like(valid)
    return Edge(src_state.sid, dst_state.sid, 0 if action == "R" else 1,
                valid, changed_valid, new_suffix)


def collect(args) -> dict:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_questions > len(dataset):
        raise ValueError(f"Requested {args.num_questions}, GSM8K test has {len(dataset)}")
    selected = np.random.default_rng(args.seed).permutation(len(dataset))[:args.num_questions]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = _load_drafter(args.dllm_dir, args.drafter_device)
    embedding = model.get_input_embeddings()
    hidden_size = int(model.config.hidden_size)
    args.raw_top_k = int(args.raw_top_k)
    native_args = type("NativeArgs", (), dict(
        raw_top_k=args.raw_top_k, physical_block_size=args.physical_block_size,
        small_block_size=args.small_block_size, extend_size=args.extend_size,
        drafter_threshold=args.threshold, max_refinement_steps=args.max_refinement_steps,
    ))()
    runner = NativeElysiaRunner(model, tokenizer, native_args)
    writer = ShardWriter(out, args.shard_rows)
    questions_path = out / "questions.jsonl"
    edges: list[dict] = []
    state_objects: dict[str, State] = {}
    question_rows = []
    terminal_counts = {}
    total_start = time.time()
    with questions_path.open("w", encoding="utf-8") as question_handle:
        for local_q, source_index in enumerate(selected.tolist()):
            row = dataset[int(source_index)]
            qid = str(int(source_index))
            prefix = _chat_prefix(tokenizer, row["question"])
            completion: list[int] = []
            previous_state = None
            question_state_ids = []
            question_edge_start = len(edges)
            termination = "max_proposal_tokens"
            action_rng = np.random.default_rng(args.seed + int(source_index) * 1009)
            print(f"[capture] q={local_q + 1}/{len(selected)} source_id={qid} "
                  f"prompt_tokens={len(prefix)}", flush=True)
            for block_index in range(args.max_proposal_tokens // args.extend_size):
                prompt = prefix + completion
                # Sample the number of real R edges before the next E. The
                # same-forward top-1 fill at the selected boundary is the E prefix.
                selected_refines = int(action_rng.integers(0, args.max_refinement_steps + 1))
                native_args.max_refinement_steps = selected_refines
                try:
                    snapshots = runner.segment(prompt, max_snapshots=selected_refines + 1)
                except NativeEosWithoutSnapshot as exc:
                    termination = "eos_before_observable_snapshot"
                    terminal_counts[termination] = terminal_counts.get(termination, 0) + 1
                    break
                if not snapshots:
                    termination = "no_native_snapshot"
                    terminal_counts[termination] = terminal_counts.get(termination, 0) + 1
                    break
                snapshots = snapshots[:selected_refines + 1]
                segment_states = []
                for refine_index, snapshot in enumerate(snapshots):
                    raw = _state_row(snapshot, model, len(prompt), block_index,
                                     args.max_proposal_tokens // args.extend_size,
                                     tokenizer.eos_token_id, hidden_size)
                    sid = f"q{qid}_b{block_index:02d}_r{refine_index:02d}"
                    meta = dict(
                        state_id=sid, question_id=qid, source_question_index=int(source_index),
                        block_index=block_index, refine_index=refine_index,
                        sampled_refine_budget=selected_refines,
                        context_len=len(prompt), action_from_parent=(
                            "E" if previous_state is not None and refine_index == 0 else
                            ("R" if refine_index > 0 else None)),
                        actual_action_taken=(
                            "E" if previous_state is not None and refine_index == 0 else
                            ("R" if refine_index > 0 else None)),
                        stop_available=not raw["eos_in_native"],
                        refine_available=bool(
                            not raw["eos_in_native"] and raw["masks_remaining"] > 0
                            and refine_index < args.max_refinement_steps
                        ),
                        refinement_probe_budget_remaining=max(0, selected_refines - refine_index),
                        extend_available=bool(
                            block_index + 1 < args.max_proposal_tokens // args.extend_size
                            and not raw["eos_in_native"] and not raw["eos_in_filled"]
                        ),
                        parent_state_id=(
                            previous_state["state_id"] if refine_index == 0 and previous_state is not None
                            else (segment_states[-1]["state_id"] if refine_index > 0 else None)
                        ),
                        native_unmask_forward_index=int(snapshot["unmask_forward_index"]),
                        draft_passes_elapsed=int(snapshot["draft_passes_elapsed"]),
                        draft_latency_elapsed_ms=float(snapshot["draft_latency_elapsed_ms"]),
                        masks_remaining=raw["masks_remaining"],
                        committed_tokens=args.extend_size - raw["masks_remaining"],
                        stop_candidate_contains_eos=raw["eos_in_filled"],
                        hidden_state_stage=snapshot.get("hidden_state_stage",
                                                        "native_pre_counterfactual_fill"),
                        hidden_state_source=snapshot.get("hidden_state_source",
                                                         "native_refinement_forward_output"),
                        hidden_layer_indices=snapshot.get("hidden_layer_indices", []),
                        native_hidden_start_offset=raw["hidden_offset"],
                        native_topk_start_offset=raw["topk_offset"],
                        feature_valid_positions=np.flatnonzero(raw["feature_valid"]).tolist(),
                        newly_unmasked_positions=list(snapshot.get("newly_unmasked_positions", [])),
                        num_masked_positions=raw["masks_remaining"],
                        snapshot_forward_index=int(snapshot["unmask_forward_index"]),
                        termination_reason=("eos_committed" if raw["eos_in_native"] else None),
                    )
                    writer.append({**raw, "meta": meta})
                    state = State(
                        sid=sid, qid=qid, x=raw["x"], mask=raw["mask"],
                        eligible=raw["valid"], confidence=raw["confidence"],
                        changed=np.zeros(args.extend_size, dtype=np.bool_),
                        content=raw["content"].astype(np.float32), proposal_len=args.extend_size,
                    )
                    state_objects[sid] = state
                    entry = {**meta, **raw}
                    if refine_index == 0 and previous_state is not None:
                        edge = _make_edge(previous_state, entry, "E", state_objects)
                        edges.append(dict(src_state_id=edge.src, dst_state_id=edge.dst,
                                          action="E", valid=edge.valid.tolist(),
                                          changed_valid=edge.changed_valid.tolist(),
                                          new_suffix=edge.new_suffix.tolist()))
                    elif refine_index > 0:
                        edge = _make_edge(segment_states[-1], entry, "R", state_objects)
                        edges.append(dict(src_state_id=edge.src, dst_state_id=edge.dst,
                                          action="R", valid=edge.valid.tolist(),
                                          changed_valid=edge.changed_valid.tolist(),
                                          new_suffix=edge.new_suffix.tolist()))
                    segment_states.append(entry)
                    question_state_ids.append(sid)
                    if raw["eos_in_native"]:
                        break
                last = segment_states[-1]
                previous_state = last
                if last["eos_in_native"]:
                    termination = "eos_committed"
                    break
                # E is unavailable if native top-1 STOP fill already contains EOS.
                if last["eos_in_filled"]:
                    termination = "eos_in_stop_fill"
                    break
                filled_for_extension = [int(x) for x in last["filled"]]
                if tokenizer.eos_token_id in filled_for_extension:
                    filled_for_extension = filled_for_extension[
                        :filled_for_extension.index(tokenizer.eos_token_id) + 1]
                completion.extend(filled_for_extension)
                if len(completion) >= args.max_proposal_tokens:
                    termination = "max_proposal_tokens"
                    break
            terminal_counts[termination] = terminal_counts.get(termination, 0) + 1
            writer.flush()
            question_meta = dict(
                question_id=qid, source_question_index=int(source_index),
                question=row["question"], prefix_token_ids=prefix,
                prompt_token_count=len(prefix), completion_tokens_collected=len(completion),
                state_ids=question_state_ids,
                edge_count=len(edges) - question_edge_start,
                state_count=len(question_state_ids), termination_reason=termination,
            )
            question_rows.append(question_meta)
            question_handle.write(json.dumps(question_meta, ensure_ascii=False) + "\n")
            print(f"[capture] q={local_q + 1}/{len(selected)} states={len(question_state_ids)} "
                  f"edges={len(edges) - question_edge_start} completion={len(completion)} "
                  f"stop={termination}", flush=True)
    writer.flush()
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.get_input_embeddings().weight.device)
    del runner, model, tokenizer, embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (out / "edges.jsonl").write_text(
        "".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")
    manifest = dict(
        schema_version=SCHEMA, status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        dataset="openai/gsm8k:test", selected_question_indices=selected.tolist(),
        question_count=len(question_rows), state_count=sum(q["state_count"] for q in question_rows),
        edge_count=len(edges), action_counts={a: sum(e["action"] == a for e in edges) for a in ("R", "E")},
        terminal_counts=terminal_counts,
        config=dict(num_questions=args.num_questions, seed=args.seed,
                    model_id="Efficient-Large-Model/Fast_dLLM_v2_1.5B",
                    tokenizer_id=args.tokenizer_name,
                    threshold=args.threshold, do_sample=False, extend_size=args.extend_size,
                    max_proposal_tokens=args.max_proposal_tokens,
                    max_refinement_steps=args.max_refinement_steps,
                    refine_steps_per_segment="seeded_uniform_0_to_3",
                    physical_block_size=args.physical_block_size,
                    small_block_size=args.small_block_size, raw_top_k=args.raw_top_k),
        protocol=dict(
            observation="native Fast-dLLM forward snapshots before counterfactual fill",
            refine="consecutive snapshots from one native 8-token segment",
            extend="top-1 fill of the last native snapshot, then a fresh native 8-token segment",
            action_probe="seeded uniform choice of 0-3 R steps before each E; each state retains its same-forward STOP fill",
            kv="native generator prefix-KV path; no mutable block-cache override",
            verifier="not run; this experiment audits drafter dynamics only",
            input_feature_layout=["native_final_hidden", "current_token_embedding",
                                  "topk_expected_token_embedding", "mask", "confidence",
                                  "top1_top2_margin", "topk_entropy", "within_block_position",
                                  "proposal_block_index", "native_forward_index",
                                  "context_length", "newly_unmasked", "feature_valid"],
            target="next mask, confidence, candidate-change event and top-1 learned token embedding",
            split="question-disjoint 70/15/15, fixed seed 42",
        ),
        elapsed_capture_seconds=time.time() - total_start,
    )
    (out / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_capture(root: Path):
    states: dict[str, State] = {}
    metadata = {}
    shard_cache = {}
    with (root / "states.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        shard = row["shard"]
        if shard not in shard_cache:
            with np.load(root / shard, allow_pickle=False) as arrays:
                shard_cache[shard] = {key: arrays[key] for key in arrays.files}
        arrays = shard_cache[shard]
        index = int(row["row"])
        sid = str(row["state_id"])
        states[sid] = State(
            sid=sid, qid=str(row["question_id"]), x=arrays["x"][index].astype(np.float16),
            mask=arrays["mask"][index].astype(bool), eligible=arrays["valid"][index].astype(bool),
            confidence=arrays["confidence"][index].astype(np.float32),
            changed=arrays["changed"][index].astype(bool),
            content=arrays["content"][index].astype(np.float32), proposal_len=8,
        )
        metadata[sid] = row
    edges = []
    with (root / "edges.jsonl").open(encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            row = json.loads(row)
            edges.append(Edge(
                src=str(row["src_state_id"]), dst=str(row["dst_state_id"]),
                action=0 if row["action"] == "R" else 1,
                valid=np.asarray(row["valid"], dtype=bool),
                changed_valid=np.asarray(row["changed_valid"], dtype=bool),
                new_suffix=np.asarray(row["new_suffix"], dtype=bool),
            ))
    return states, edges, metadata


def _action_metrics(records, states, train_edges):
    result = {"all": score_records(records, states, train_edges)}
    for action, label in ((0, "R"), (1, "E")):
        selected = [(edge, pred) for edge, pred in records if edge.action == action]
        metrics = score_records(selected, states, train_edges) if selected else {
            "edge_count": 0, "position_count": 0,
        }
        if action == 1 and selected:
            # A previous proposal's mask/confidence pattern is not a valid
            # persistence baseline for a newly appended segment.
            mask_rows = [states[edge.dst].mask[edge.valid]
                         for edge, _ in selected if edge.valid.any()]
            y = np.concatenate(mask_rows) if mask_rows else np.asarray([], dtype=bool)
            metrics["extend_new_segment_all_mask_baseline_brier"] = float(
                np.mean((1.0 - y.astype(np.float32)) ** 2)) if y.size else None
            metrics["mask_persistence_brier"] = None
            metrics["current_native_rule_mask_brier"] = None
            metrics["confidence_persistence_mae"] = None
        result[label] = metrics
    return result


def audit_capture(root: Path, args) -> dict:
    states, edges, metadata = load_capture(root)
    if not states or not edges:
        raise RuntimeError("Fresh native capture produced no trainable transitions")
    capture_manifest = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
    split = question_split([state.qid for state in states.values()], seed=args.split_seed)
    question_sets = {part: {qid for qid, value in split.items() if value == part}
                     for part in ("train", "validation", "test")}
    normalize_states(states, question_sets["train"])
    train_edges = _restrict_edges(edges, question_sets["train"], states)
    val_edges = _restrict_edges(edges, question_sets["validation"], states)
    test_edges = _restrict_edges(edges, question_sets["test"], states)
    paths = _paths_for_edges(edges, states, max_horizon=3)
    paths_by_part = {
        part: [path for path in paths if path and states[path[0].src].qid in question_sets[part]]
        for part in ("train", "validation", "test")
    }
    in_dim = int(next(iter(states.values())).x.shape[-1])
    content_dim = int(next(iter(states.values())).content.shape[-1])
    report = {
        "data": {
            "question_count_with_states": len(split),
            "questions_requested": int(capture_manifest.get("question_count", len(split))),
            "questions_without_states": sorted(set(map(str, capture_manifest.get(
                "selected_question_indices", []))) - set(split)),
            "state_count": len(states), "edge_count": len(edges),
            "edge_action_counts": {a: sum(edge.action == action for edge in edges)
                                    for a, action in (("R", 0), ("E", 1))},
            "split_question_counts": {part: len(qids) for part, qids in question_sets.items()},
            "split_edge_counts": {part: len(_restrict_edges(edges, qids, states))
                                  for part, qids in question_sets.items()},
            "split_path_counts_horizon": {
                part: {f"h{h}": sum(len(path) >= h for path in paths_by_part[part])
                       for h in (1, 2, 3)} for part in paths_by_part
            },
            "feature_dim": in_dim, "candidate_embedding_dim": content_dim,
            "input_signal": "native hidden + learned token embeddings + native top-k/statistics",
        },
        "config": {"updates_per_model": args.updates, "seeds": args.seeds,
                   "latent_dim": 128, "horizons": [1, 2, 3],
                   "split_seed": args.split_seed, "question_split": "70/15/15"},
        "arms": {}, "per_seed": {}, "training_logs": {},
    }
    names = ("rich_direct_h1", "latent_direct_h1", "latent_dynamics_h1_h3")
    rows_by_arm = {name: [] for name in names}
    ablation_rows: dict[str, list[dict]] = {}
    for seed in args.seeds:
        print(f"[train] seed={seed} edges train/val/test="
              f"{len(train_edges)}/{len(val_edges)}/{len(test_edges)}", flush=True)
        models = {
            "rich_direct_h1": DirectPredictor(in_dim, 256, content_dim),
            "latent_direct_h1": DirectPredictor(in_dim, 128, content_dim),
            "latent_dynamics_h1_h3": LatentDynamics(in_dim, content_dim, latent_dim=128),
        }
        seed_result = {}
        for name in ("rich_direct_h1", "latent_direct_h1"):
            model, logs = _fit_direct(
                models[name], train_edges, val_edges, states, args.device,
                args.updates, seed, batch_size=args.batch_size,
                eval_every=max(20, args.updates // 8), label=f"{name}/seed{seed}",
            )
            records = _collect_direct(model, test_edges, states, args.device,
                                      batch_size=args.batch_size)
            seed_result[name] = {"h1": _action_metrics(records, states, train_edges)}
            report["training_logs"][f"{name}_seed{seed}"] = logs
            del model
        dynamics, logs = _fit_dynamics(
            models["latent_dynamics_h1_h3"], paths_by_part["train"],
            paths_by_part["validation"], states, args.device, args.updates, seed,
            horizon=3, batch_size=max(2, args.batch_size // 2),
            eval_every=max(20, args.updates // 8), label=f"latent_dynamics/seed{seed}",
        )
        rolled = _collect_rollout(dynamics, paths_by_part["test"], states,
                                  args.device, max_horizon=3,
                                  batch_size=max(2, args.batch_size // 2))
        seed_result["latent_dynamics_h1_h3"] = {
            f"h{h}": _action_metrics(rolled[h], states, train_edges)
            for h in (1, 2, 3)
        }
        # Test-time group ablation: zero one normalized feature group (replace
        # it with its train-set mean) and replay the learned latent dynamics.
        # This is sensitivity analysis, not a retrained causal attribution.
        groups = {
            "native_final_hidden": slice(0, content_dim),
            "current_token_embedding": slice(content_dim, 2 * content_dim),
            "topk_expected_embedding": slice(2 * content_dim, 3 * content_dim),
            "native_scalars": slice(3 * content_dim, in_dim),
        }
        seed_result["feature_ablation"] = {}
        test_state_ids = [sid for sid, state in states.items()
                          if state.qid in question_sets["test"]]
        for group_name, feature_slice in groups.items():
            original = {sid: states[sid].x[:, feature_slice].copy() for sid in test_state_ids}
            for sid in test_state_ids:
                states[sid].x[:, feature_slice] = 0.0
            ablated_rollouts = _collect_rollout(
                dynamics, paths_by_part["test"], states, args.device,
                max_horizon=3, batch_size=max(2, args.batch_size // 2),
            )
            seed_result["feature_ablation"][group_name] = {
                f"h{h}": _action_metrics(ablated_rollouts[h], states, train_edges)
                for h in (1, 3)
            }
            for sid, value in original.items():
                states[sid].x[:, feature_slice] = value
            ablation_rows.setdefault(group_name, []).append(seed_result["feature_ablation"][group_name])
        report["training_logs"][f"latent_dynamics_h1_h3_seed{seed}"] = logs
        report["per_seed"][str(seed)] = seed_result
        for name, horizons in seed_result.items():
            if name == "feature_ablation":
                continue
            for horizon, action_result in horizons.items():
                for action, metrics in action_result.items():
                    rows_by_arm[name].append({"seed": seed, "horizon": horizon,
                                              "action": action, **metrics})
        del dynamics, models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, rows in rows_by_arm.items():
        report["arms"][name] = {}
        for action in ("all", "R", "E"):
            action_rows = [row for row in rows if row["action"] == action]
            horizons = sorted({str(row["horizon"]) for row in action_rows})
            report["arms"][name][action] = {
                horizon: _average_metric_dict([
                    {key: value for key, value in row.items()
                     if key not in ("seed", "horizon", "action")}
                    for row in action_rows if str(row["horizon"]) == horizon
                ]) for horizon in horizons
            }
    report["interpretation"] = {
        "representation_gap": "rich_direct_h1 vs latent_direct_h1, same held-out questions and transitions",
        "transition_gap": "latent_direct_h1 vs latent_dynamics_h1 at H1",
        "rollout_gap": "latent_dynamics H1 vs H2 vs H3, separately for R and E destinations",
        "semantic_tokens": "candidate-content is learned Fast-dLLM input-embedding space, never random IDs",
        "not_claimed": "No exact token generation, verifier acceptance, optimal action, or latency-policy claim",
    }
    report["feature_ablation"] = {
        group: {
            horizon: {
                action: _average_metric_dict([
                    result[horizon][action] for result in group_runs
                ])
                for action in ("all", "R", "E")
            }
            for horizon in ("h1", "h3")
        }
        for group, group_runs in ablation_rows.items()
    }
    report["feature_ablation_note"] = (
        "Each feature group is replaced with its training mean only at test time; "
        "the model is not retrained, so these are sensitivity checks rather than causal importance."
    )
    report["capture"] = capture_manifest
    return report


def package(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != temporary:
                zf.write(file, file.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dllm-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--extend-size", type=int, default=8)
    p.add_argument("--max-proposal-tokens", type=int, default=64)
    p.add_argument("--max-refinement-steps", type=int, default=3)
    p.add_argument("--physical-block-size", type=int, default=32)
    p.add_argument("--small-block-size", type=int, default=8)
    p.add_argument("--raw-top-k", type=int, default=32)
    p.add_argument("--drafter-device", type=int, default=0)
    p.add_argument("--shard-rows", type=int, default=32)
    p.add_argument("--updates", type=int, default=240)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_proposal_tokens % args.extend_size:
        raise ValueError("max-proposal-tokens must be divisible by extend-size")
    if args.extend_size != 8 or args.max_refinement_steps > 3:
        raise ValueError("This experiment is pinned to native E8 and at most 3 R steps")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator")
    args.device = torch.device("cuda:0")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, flush=True)
    print("Fresh native GSM8K capture; verifier is intentionally not loaded.", flush=True)
    started = time.time()
    try:
        manifest = collect(args)
        print("Capture:", manifest["state_count"], "states;", manifest["edge_count"],
              "edges; actions:", manifest["action_counts"], flush=True)
        report = audit_capture(args.out_dir, args)
        report["elapsed_total_seconds"] = time.time() - started
        (args.out_dir / "latent_sufficiency_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (args.out_dir / "README.md").write_text(
            "# Fresh native dLLM latent dynamics test\n\n"
            "The capture is generated afresh by Fast-dLLM. It contains native R transitions "
            "and native E transitions (top-1 fill then a new 8-token segment), threshold 0.5, "
            "up to three R steps, and question-disjoint train/validation/test evaluation. "
            "The verifier is not run. See capture_manifest.json and "
            "latent_sufficiency_report.json for the exact scope and results.\n",
            encoding="utf-8")
        archive = package(args.out_dir)
        print("RESULT ZIP:", archive, f"({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        print("ELAPSED:", f"{report['elapsed_total_seconds'] / 60:.1f} min", flush=True)
    except Exception as exc:
        (args.out_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            try:
                print("PARTIAL ZIP:", package(args.out_dir), flush=True)
            except Exception as package_exc:
                print("Could not package partial output:", repr(package_exc), flush=True)
        raise


if __name__ == "__main__":
    main()
