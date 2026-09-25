"""Structured S/R/E collection: store all probes, recurse into <=2 E children.

The explicit-state backend uses block-causal, full-context replay. It never
calls generate_draft_tokens_arbitrary_length or inherits hidden-state rows.
See STRUCTURED_SPARSE_PROTOCOL.md for action and timing conventions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import zipfile

import numpy as np
import torch

from sparse_extend_world_model_collector import (
    MASK_ID, _find_unpacked_index, _hydrate_backbone_rows, _load_models,
    _greedy_reference, _score_greedy,
)
from raw_stream_writer import RawShardWriter


def identity(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def regime(accepted, length):
    if accepted >= length:
        return "full"
    if accepted >= max(1, length - 2):
        return "near_full"
    if accepted <= 1:
        return "early_mismatch"
    return "mid"


def annotate_verifier_acceptance(row, recomputed_accepted):
    """Keep the archive label, but make the active label match this verifier run."""
    checked = dict(row)
    recorded = int(row["accepted_len"])
    recomputed = int(recomputed_accepted)
    checked["backbone_recorded_accepted_len"] = recorded
    checked["verifier_recomputed_accepted_len"] = recomputed
    checked["verifier_acceptance_matches_backbone"] = recorded == recomputed
    # All newly collected labels must use the verifier/reference generated in
    # this run. Preserve the source value explicitly for auditing.
    checked["accepted_len"] = recomputed
    return checked


def hash_observation(prefix, native, observation):
    """Stable content hash for exact raw states; excludes timing noise."""
    digest = hashlib.sha256()
    identity_part = json.dumps(
        {"prefix_token_ids": list(prefix),
         "proposal_token_ids_before_fill": list(native),
         "hidden_layer_indices": observation["hidden_layer_indices"]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest.update(identity_part)
    arrays = (
        ("predictions", observation["predictions"], "<i8"),
        ("probabilities", observation["probabilities"], "<f4"),
        ("filled", observation["filled"], "<i8"),
        ("filled_probabilities", observation["filled_probabilities"], "<f4"),
        ("hidden_states", observation["hidden_states"], "<f4"),
        ("topk_token_ids", observation["topk_token_ids"], "<i8"),
        ("topk_logits", observation["topk_logits"], "<f4"),
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def select_anchors(rows, num_questions, anchors_per_question, max_rounds=0):
    """Deterministic round-robin across acceptance regimes, not first rounds."""
    questions = sorted({int(r["problem_id"]) for r in rows})[:num_questions]
    selected = []
    for q in questions:
        candidates = [r for r in rows if int(r["problem_id"]) == q
                      and int(r.get("proposal_length", 0)) == 8]
        buckets = {k: [] for k in ("full", "near_full", "mid", "early_mismatch")}
        for row in sorted(candidates, key=lambda r: (int(r["round_id"]),
                          int(r.get("boundary_index", 0)), str(r["state_id"]))):
            buckets[regime(int(row["accepted_len"]), 8)].append(row)
        chosen_rounds = set()
        count = 0
        while any(buckets.values()) and count < anchors_per_question:
            for bucket in buckets.values():
                while bucket:
                    row = bucket.pop(0)
                    rid = int(row["round_id"])
                    if max_rounds and rid not in chosen_rounds and len(chosen_rounds) >= max_rounds:
                        continue
                    selected.append(row)
                    chosen_rounds.add(rid)
                    count += 1
                    break
                if count >= anchors_per_question:
                    break
    if not selected:
        raise ValueError("No L=8 anchors found")
    return selected


def load_anchors(path, args):
    required = ("prefix_token_ids_offsets", "prefix_token_ids_flat",
                "proposal_token_ids_before_fill", "proposal_token_ids_after_fill")
    if path.is_dir():
        index = _find_unpacked_index(path, args.dataset)
        rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines()]
        selected = select_anchors(rows, args.num_questions, args.anchors_per_question,
                                  args.max_rounds_per_question)
        def load(shard):
            candidates = [index.parent / shard, index.parent / Path(shard).name]
            actual = next(p for p in candidates if p.is_file())
            with np.load(actual, allow_pickle=False) as arrays:
                return {key: arrays[key] for key in required}
        result = _hydrate_backbone_rows(selected, args.dataset, load)
    else:
        with zipfile.ZipFile(path) as archive:
            index = f"raw/{args.dataset}/index.jsonl"
            rows = [json.loads(x) for x in archive.read(index).decode().splitlines()]
            selected = select_anchors(rows, args.num_questions, args.anchors_per_question,
                                      args.max_rounds_per_question)
            def load(shard):
                name = str(Path(index).parent / Path(shard).name).replace("\\", "/")
                with archive.open(name) as handle, np.load(handle, allow_pickle=False) as arrays:
                    return {key: arrays[key] for key in required}
            result = _hydrate_backbone_rows(selected, args.dataset, load)
    return result


def commit_one(native, predictions, probabilities, small_block, threshold, start=0):
    """Commit in the first unresolved proposal-relative logical frame only."""
    masked = [i for i in range(start, len(native)) if native[i] == MASK_ID]
    if not masked:
        return list(native), []
    frame = (masked[0] // small_block) * small_block
    eligible = [i for i in masked if i < frame + small_block]
    chosen = [i for i in eligible if probabilities[i] > threshold]
    if not chosen:
        chosen = [max(eligible, key=lambda i: (probabilities[i], -i))]
    result = list(native)
    for i in chosen:
        if int(predictions[i]) == MASK_ID:
            raise ValueError("Drafter predicted MASK as a committed token")
        result[i] = int(predictions[i])
    return result, chosen


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class ExplicitDrafter:
    def __init__(self, model, args):
        self.model, self.args = model.eval(), args
        self.device = model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def observe(self, prefix, native):
        if not prefix:
            raise ValueError("A nonempty verifier prefix is required for next-token alignment")
        tokens = list(prefix) + list(native)
        pad = (-len(tokens)) % self.args.physical_block_size
        ids = torch.tensor([tokens + [MASK_ID] * pad], device=self.device)
        begin, end = len(prefix), len(prefix) + len(native)
        logit_positions = torch.arange(begin - 1, end - 1, device=self.device)
        sync(self.device)
        began = time.perf_counter()
        output = self.model(input_ids=ids, use_cache=False, update_past_key_values=False,
                            block_size=self.args.physical_block_size,
                            output_hidden_states=True, logits_to_keep=logit_positions)
        sync(self.device)
        forward_ms = (time.perf_counter() - began) * 1000
        # A prediction for absolute token p comes from logits[p-1]. Hidden
        # states, by contrast, are representations at token position p.
        logits = output.logits[0].float()
        probs = logits.softmax(-1)
        predicted_prob, predicted = probs.max(-1)
        k = min(self.args.raw_top_k, logits.shape[-1])
        top_values, top_ids = logits.topk(k, dim=-1)
        hidden = output.hidden_states
        if hidden is None:
            raise RuntimeError("Model omitted hidden states; refusing a partial observation")
        last = len(hidden) - 1
        layers = sorted({0, last // 4, last // 2, 3 * last // 4, last})
        result = {
            "hidden_states": [hidden[i][0, begin:end].half().cpu().tolist() for i in layers],
            "hidden_layer_indices": layers,
            "topk_token_ids": top_ids.cpu().tolist(),
            "topk_logits": top_values.half().cpu().tolist(),
            "predictions": predicted.cpu().tolist(),
            "probabilities": predicted_prob.cpu().tolist(),
            "forward_ms": forward_ms,
            "input_state_hash": identity(prefix, native),
        }
        filled = [result["predictions"][i] if token == MASK_ID else token
                  for i, token in enumerate(native)]
        if MASK_ID in filled:
            raise RuntimeError("Counterfactual fill retained MASK; cannot label Submit")
        result["filled"] = filled
        token_ids = torch.tensor(filled, device=self.device).unsqueeze(-1)
        result["filled_probabilities"] = probs.gather(-1, token_ids).squeeze(-1).cpu().tolist()
        if any(len(layer) != len(native) for layer in result["hidden_states"]):
            raise RuntimeError("Hidden-state absolute position alignment failed")
        return result


class KVExplicitDrafter(ExplicitDrafter):
    """Reuse only complete, immutable 32-token blocks before the first mask.

    A cache is keyed by the exact token prefix, never by branch ID or length.
    The active/masked block is always recomputed. This is block-prefix KV reuse,
    not reuse of a mutable denoising-block cache.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self._prefix_cache = OrderedDict()
        self._max_cached_prefixes = 2

    @torch.inference_mode()
    def observe(self, prefix, native):
        if not prefix:
            raise ValueError("A nonempty verifier prefix is required")
        tokens = list(prefix) + list(native)
        block_size = self.args.physical_block_size
        first_mask = next((i for i, value in enumerate(native) if value == MASK_ID), len(native))
        cached_count = ((len(prefix) + first_mask) // block_size) * block_size
        if cached_count == 0:
            result = super().observe(prefix, native)
            result.update(kv_cache_hit=False, kv_cached_prefix_len=0)
            return result
        pad = (-len(tokens)) % block_size
        all_ids = torch.tensor([tokens + [MASK_ID] * pad], device=self.device)
        key = identity(tokens[:cached_count])
        sync(self.device)
        began = time.perf_counter()
        cached = self._prefix_cache.get(key)
        hit = cached is not None
        if hit:
            self._prefix_cache.move_to_end(key)
        else:
            if len(prefix) - 1 < cached_count:
                prefill_indices = torch.arange(max(0, len(prefix) - 1), cached_count,
                                               device=self.device)
            else:
                prefill_indices = torch.tensor([cached_count - 1], device=self.device)
            prefill = self.model(input_ids=all_ids[:, :cached_count], use_cache=True,
                                 update_past_key_values=True, block_size=block_size,
                                 output_hidden_states=True, logits_to_keep=prefill_indices)
            cached = (prefill, int(prefill_indices[0].item()))
            self._prefix_cache[key] = cached
            if len(self._prefix_cache) > self._max_cached_prefixes:
                self._prefix_cache.popitem(last=False)
        prefill, first_logit = cached
        suffix = None
        if cached_count < all_ids.shape[1]:
            suffix = self.model(input_ids=all_ids[:, cached_count:],
                past_key_values=prefill.past_key_values, use_cache=True,
                update_past_key_values=False, block_size=block_size,
                output_hidden_states=True)
        sync(self.device)
        forward_ms = (time.perf_counter() - began) * 1000
        begin, end = len(prefix), len(tokens)
        selected_logits = []
        for token_position in range(begin, end):
            # Production keeps the first logit of each physical block and
            # shifts the other block logits by one position.
            logit_position = (token_position if token_position % block_size == 0
                              else token_position - 1)
            selected_logits.append((prefill.logits[0, logit_position - first_logit]
                if logit_position < cached_count else
                suffix.logits[0, logit_position - cached_count]))
        logits = torch.stack(selected_logits).float()
        probs = logits.softmax(-1)
        predicted_prob, predicted = probs.max(-1)
        k = min(self.args.raw_top_k, logits.shape[-1])
        top_values, top_ids = logits.topk(k, dim=-1)
        layers_count = len(prefill.hidden_states)
        layers = sorted({0, (layers_count - 1) // 4, (layers_count - 1) // 2,
                         3 * (layers_count - 1) // 4, layers_count - 1})
        hidden_rows = []
        for layer in layers:
            hidden_rows.append(torch.stack([
                (prefill.hidden_states[layer][0, absolute]
                 if absolute < cached_count else
                 suffix.hidden_states[layer][0, absolute - cached_count])
                for absolute in range(begin, end)
            ]).half().cpu().tolist())
        filled = [int(predicted[i]) if value == MASK_ID else int(value)
                  for i, value in enumerate(native)]
        filled_ids = torch.tensor(filled, device=self.device).unsqueeze(-1)
        filled_probabilities = probs.gather(-1, filled_ids).squeeze(-1)
        return dict(hidden_states=hidden_rows, hidden_layer_indices=layers,
            topk_token_ids=top_ids.cpu().tolist(),
            topk_logits=top_values.half().cpu().tolist(),
            predictions=predicted.cpu().tolist(),
            probabilities=predicted_prob.cpu().tolist(),
            forward_ms=forward_ms, input_state_hash=identity(prefix, native),
            filled=filled, filled_probabilities=filled_probabilities.cpu().tolist(),
            kv_cache_hit=hit, kv_cached_prefix_len=cached_count)


class CachedVerifier:
    """Greedy labels plus measured prefix-KV verification calibration per context."""
    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model.eval(), tokenizer, args
        self.device = model.get_input_embeddings().weight.device
        self.records = []

    @torch.inference_mode()
    def prepare(self, prefix):
        args = self.args
        model_key = {
            "name": args.target_model_name,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
            "dtype": str(self.model.dtype), "greedy": True,
            "max_tokens": args.max_proposal_tokens + 1,
            # Greedy token IDs can differ at near-ties under different sharding.
            "device_map": sorted((str(k), str(v)) for k, v in
                                  getattr(self.model, "hf_device_map", {}).items()),
        }
        key = identity(prefix, model_key)
        cache_dir = getattr(args, "reference_cache_dir", None) or args.output_dir / "reference_cache"
        cache_file = cache_dir / f"{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if (cached["prefix_token_ids"] != prefix or cached["model_key"] != model_key
                    or cached.get("context_hash", identity(prefix)) != identity(prefix)):
                raise RuntimeError("Reference cache identity mismatch")
            if cached.get("verifier_calibration_key", key) != key:
                raise RuntimeError("Reference cache calibration-key mismatch")
            reference = cached["token_ids"]
        else:
            reference = _greedy_reference(self.model, prefix, self.tokenizer,
                                          args.max_proposal_tokens + 1)
            cache_file.write_text(json.dumps(dict(prefix_token_ids=prefix,
                context_hash=identity(prefix), verifier_calibration_key=key,
                model_key=model_key, token_ids=reference)), encoding="utf-8")
        exported = args.output_dir / "reference_cache" / cache_file.name
        exported.parent.mkdir(parents=True, exist_ok=True)
        if exported != cache_file:
            exported.write_bytes(cache_file.read_bytes())
        if len(reference) < args.max_proposal_tokens + 1 and (
                not reference or reference[-1] != self.tokenizer.eos_token_id):
            raise RuntimeError("Reference ended before required length without EOS")
        # Cache prefix except its last token. Query L+1 then produces logits
        # for all L proposed tokens and the correction/bonus token.
        past = None
        if len(prefix) > 1:
            ids = torch.tensor([prefix[:-1]], device=self.device)
            past = self.model(input_ids=ids, use_cache=True).past_key_values
        timings = {}
        lengths = list(range(8, args.max_proposal_tokens + 1, args.extend_size))
        for length in lengths:
            suffix = (reference + [self.tokenizer.eos_token_id] * length)[:length]
            ids = torch.tensor([[prefix[-1]] + suffix], device=self.device)
            samples = []
            for repeat in range(args.calibration_repeats + 1):
                # Cache copies are collection overhead; each timed probe gets
                # independent prefix KV even if transformers mutates its input.
                cache = copy.deepcopy(past)
                sync(self.device)
                began = time.perf_counter()
                result = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
                result.logits.argmax(-1)
                sync(self.device)
                elapsed = (time.perf_counter() - began) * 1000
                del result, cache
                if repeat:
                    samples.append(elapsed)
            timings[length] = statistics.median(samples)
            self.records.append(dict(context_hash=identity(prefix),
                verifier_calibration_key=key, reference_key=key,
                context_length=len(prefix),
                proposal_length=length, samples_ms=samples, median_ms=timings[length],
                source="measured_prefix_kv_calibration", device=str(self.device),
                model_key=model_key, query_tokens=length + 1))
        del past
        return reference, timings, key


class FullContextVerifier:
    """Measure and score each submitted proposal with the production no-KV path."""

    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model.eval(), tokenizer, args
        self.device = model.get_input_embeddings().weight.device
        self.records = []

    def prepare(self, prefix):
        model_key = dict(name=self.args.target_model_name,
                         revision=getattr(self.model.config, "_commit_hash", None),
                         dtype=str(self.model.dtype),
                         device_map=sorted((str(k), str(v)) for k, v in
                                           getattr(self.model, "hf_device_map", {}).items()),
                         mode="full_context_no_kv_greedy")
        return None, None, identity(prefix, model_key)

    @torch.inference_mode()
    def score(self, prefix, proposal, remaining_output_budget):
        if not prefix or not proposal:
            raise ValueError("A nonempty prefix and proposal are required")
        ids = torch.tensor([list(prefix) + list(proposal)], dtype=torch.long,
                           device=self.device)
        attention_mask = torch.ones_like(ids)
        sync(self.device)
        began = time.perf_counter()
        outputs = self.model(input_ids=ids, attention_mask=attention_mask,
                             use_cache=False, logits_to_keep=len(proposal) + 1)
        sync(self.device)
        elapsed_ms = (time.perf_counter() - began) * 1000
        predictions = outputs.logits[0].argmax(dim=-1)
        if predictions.numel() != len(proposal) + 1:
            raise RuntimeError("Verifier did not return proposal logits plus bonus logit")
        predicted = predictions.tolist()
        accepted = 0
        while accepted < len(proposal) and int(proposal[accepted]) == int(predicted[accepted]):
            accepted += 1
        emitted = [int(x) for x in proposal[:accepted]] + [int(predicted[accepted])]
        if self.tokenizer.eos_token_id in emitted:
            emitted = emitted[:emitted.index(self.tokenizer.eos_token_id) + 1]
        emitted = emitted[:int(remaining_output_budget)]
        # Production reports accepted_len before EOS/budget truncation.
        self.records.append(dict(measurement_id=len(self.records),
            context_hash=identity(prefix),
            proposal_hash=identity(prefix, proposal),
            proposal_length=len(proposal), latency_ms=elapsed_ms,
            source="measured_full_context_no_kv_per_proposal"))
        return accepted, len(emitted), emitted, elapsed_ms


def choose_children(candidates, width=2, min_ratio=0.5):
    eligible = [n for n in candidates if n["meta"]["current_submit_acceptance_ratio"] >= min_ratio
                and not n["meta"]["state_eos_committed"]]
    ordered = sorted(eligible, key=lambda n: (-n["meta"]["current_submit_acceptance_ratio"],
                                               -n["meta"]["submit_accepted_len"],
                                               n["meta"]["state_id"]))
    if not ordered:
        return []
    best = ordered[0]
    selected = [best]
    best["meta"]["selection_reason"] = "best_accepted_prefix"
    if width > 1 and len(ordered) > 1:
        def diversity(node):
            a, b = node["meta"], best["meta"]
            mask_distance = sum((x == MASK_ID) != (y == MASK_ID)
                                for x, y in zip(node["native"], best["native"]))
            return (a["source_refine_steps"] != b["source_refine_steps"],
                    mask_distance,
                    abs(a["current_submit_acceptance_ratio"] -
                        b["current_submit_acceptance_ratio"]),
                    a["current_submit_acceptance_ratio"])
        diverse = max(ordered[1:], key=diversity)
        diverse["meta"]["selection_reason"] = "diverse_refinement_mask_outcome"
        selected.append(diverse)
    for node in selected:
        node["meta"]["selected_for_expansion"] = True
    return selected


class GraphCollector:
    def __init__(self, args, engine, eos_id, verifier=None):
        self.args, self.engine, self.eos_id = args, engine, eos_id
        self.verifier = verifier
        self.nodes, self.edges, self.serial = [], [], 0
        self.writer = RawShardWriter(args.output_dir / "raw", args.dataset + "_structured", args.shard_rows)

    def node(self, prefix, native, observation, refine_steps, draft_ms, source, parent=None,
             action=None, action_ms=0.0, committed=None):
        self.serial += 1
        remaining = getattr(self.args, "remaining_output_budget", None) or self.args.max_proposal_tokens + 1
        if self.verifier is None:
            accepted, emitted, emitted_ids = _score_greedy(observation["filled"], self.reference,
                self.eos_id, remaining)
            verifier_ms = self.timings[len(native)]
            latency_source = "measured_prefix_kv_calibration_estimate_for_node"
            label_source = "cached_greedy_reference_lcp"
        else:
            accepted, emitted, emitted_ids, verifier_ms = self.verifier.score(
                prefix, observation["filled"], remaining)
            latency_source = "measured_full_context_no_kv_per_proposal"
            label_source = "direct_full_context_no_kv_greedy"
        length = len(native)
        masks = sum(t == MASK_ID for t in native)
        eos_committed = (self.eos_id in native and
                         MASK_ID not in native[:native.index(self.eos_id)])
        meta = dict(
            state_id=identity(self.anchor_id, self.serial, native)[:24],
            anchor_state_id=self.anchor_id, dataset=self.args.dataset,
            problem_id=source["problem_id"], round_id=source["round_id"],
            proposal_length=length, context_len=len(prefix),
            extend_depth=(length - 8) // self.args.extend_size,
            refine_steps_since_extend=refine_steps,
            source_refine_steps=parent["meta"]["refine_steps_since_extend"] if parent else 0,
            masks_remaining=masks, committed_tokens=length - masks,
            state_masks_resolved=masks == 0, state_eos_committed=eos_committed,
            submit_available=True,
            refine_available=bool(masks and not eos_committed and refine_steps < self.args.max_refinement_steps),
            extend_available=bool(length + self.args.extend_size <= self.args.max_proposal_tokens and not eos_committed),
            submit_accepted_len=accepted, submit_emitted_len=emitted,
            current_submit_regime=regime(accepted, length),
            current_submit_acceptance_ratio=accepted / max(length, 1),
            submit_emitted_token_ids=emitted_ids,
            submit_verifier_latency_ms=verifier_ms,
            submit_verifier_measurement_key=(identity(prefix, observation["filled"])
                if self.verifier is not None else None),
            submit_verifier_measurement_id=(self.verifier.records[-1]["measurement_id"]
                if self.verifier is not None else None),
            submit_latency_source=latency_source,
            submit_latency_is_node_measurement=self.verifier is not None,
            submit_label_source=label_source,
            reference_key=self.reference_key, verifier_calibration_key=self.reference_key,
            context_hash=identity(prefix),
            reference_length=len(self.reference) if self.reference is not None else None,
            draft_latency_from_anchor_ms=draft_ms,
            draft_passes_from_anchor=(parent["meta"]["draft_passes_from_anchor"] + 1) if parent else 0,
            backbone_draft_latency_elapsed_ms=source.get("draft_latency_elapsed_ms"),
            stop_latency_per_output_token_from_anchor=(draft_ms + verifier_ms) / max(emitted, 1),
            hidden_state_stage="exact_native_state_pre_counterfactual_fill",
            hidden_state_source=("exact_state_block_causal_kv_prefill_plus_suffix"
                if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix"
                else "full_context_block_causal_reencode"),
            hidden_state_input_hash=observation["input_state_hash"],
            observation_hash=hash_observation(prefix, native, observation),
            hidden_token_positions="prefix_length + proposal_index",
            logits_token_positions=("physical_block_first_self_else_previous"
                if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix"
                else "prefix_length + proposal_index - 1"),
            feature_scope=("entire_proposal_cached_prefill_plus_suffix"
                if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix"
                else "entire_proposal_single_forward"),
            feature_merge_mode=("immutable_prefix_kv_reuse"
                if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix"
                else "none"),
            observation_forward_ms=observation["forward_ms"],
            drafter_kv_mode=getattr(self.args, "drafter_kv_mode", "none"),
            drafter_kv_cache_hit=observation.get("kv_cache_hit", False),
            drafter_kv_cached_prefix_len=observation.get("kv_cached_prefix_len", 0),
            selected_for_expansion=False, selection_reason=None,
            collection_disposition="stored", parent_state_id=parent["meta"]["state_id"] if parent else None,
            action_from_parent=action, newly_unmasked_positions=committed or [],
        )
        if meta["hidden_state_input_hash"] != identity(prefix, native):
            raise RuntimeError("Observation does not describe this native state")
        node = dict(native=list(native), obs=observation, meta=meta)
        self.nodes.append(meta)
        self.writer.append(dict(
            proposal_token_ids=observation["filled"], proposal_token_ids_before_fill=list(native),
            proposal_token_ids_after_fill=observation["filled"],
            proposal_mask=[t == MASK_ID for t in native],
            proposal_mask_before_fill=[t == MASK_ID for t in native],
            committed_position_mask=[t != MASK_ID for t in native],
            drafter_observed_prob=observation["filled_probabilities"],
            hidden_states=observation["hidden_states"],
            hidden_layer_indices=observation["hidden_layer_indices"],
            topk_token_ids=observation["topk_token_ids"], topk_logits=observation["topk_logits"],
            prefix_token_ids=list(prefix), metadata=meta))
        if parent:
            self.edges.append(dict(src_state_id=parent["meta"]["state_id"], dst_state_id=meta["state_id"],
                action=action, action_cost_ms=action_ms,
                action_cost_source=("measured_post_decision_kv_drafter_forwards"
                    if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix"
                    else "measured_full_context_drafter_forward"),
                action_cost_includes_kv_reuse=(getattr(self.args, "drafter_kv_mode", "none")
                                               == "stable_block_prefix"),
                extend_delta=length - len(parent["native"]),
                native_unmask_forwards=1, terminal=False,
                action_forward_count=(2 if action == "E" and
                    getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix" else 1),
                destination_observation_forward_ms=observation["forward_ms"]))
        return node

    def transition(self, parent, action):
        native = list(parent["native"])
        if action == "E":
            old_length = len(native)
            native += [MASK_ID] * self.args.extend_size
            observation = self.engine.observe(self.prefix, native)
            start, refine_steps = old_length, 0
        else:
            observation = parent["obs"]
            start, refine_steps = 0, parent["meta"]["refine_steps_since_extend"] + 1
        updated, committed = commit_one(native, observation["predictions"], observation["probabilities"],
                                       self.args.small_block_size, self.args.drafter_threshold, start)
        if not committed:
            raise RuntimeError("An R/E action must include one actual unmask forward")
        obs = self.engine.observe(self.prefix, updated)
        if getattr(self.args, "drafter_kv_mode", "none") == "stable_block_prefix":
            # The parent observation exists before this decision. The child
            # observation is the next forward needed after choosing R/E.
            action_ms = obs["forward_ms"] + (observation["forward_ms"] if action == "E" else 0.0)
            forward_count = 2 if action == "E" else 1
        else:
            action_ms = observation["forward_ms"]
            forward_count = 1
        child = self.node(self.prefix, updated, obs, refine_steps,
                          parent["meta"]["draft_latency_from_anchor_ms"] + action_ms,
                          self.source, parent, action, action_ms, committed)
        if forward_count != 1:
            child["meta"]["draft_passes_from_anchor"] = parent["meta"]["draft_passes_from_anchor"] + forward_count
        return child

    def backbone(self, node, steps):
        result = [node]
        for _ in range(steps):
            if not result[-1]["meta"]["refine_available"]:
                break
            result.append(self.transition(result[-1], "R"))
        return result

    def run_anchor(self, row, reference, timings, key):
        self.reference, self.timings, self.reference_key = reference, timings, key
        self.prefix, self.source, self.anchor_id = row["prefix_token_ids"], row, row["state_id"]
        native = row["proposal_token_ids_before_fill"][:8]
        root = self.node(self.prefix, native, self.engine.observe(self.prefix, native), 0, 0.0, row)
        recorded_acceptance = row.get("backbone_recorded_accepted_len", row["accepted_len"])
        root["meta"]["backbone_accepted_len"] = recorded_acceptance
        root["meta"]["backbone_recomputed_accepted_len"] = row.get(
            "verifier_recomputed_accepted_len", row["accepted_len"])
        root["meta"]["backbone_verifier_mismatch"] = not row.get(
            "verifier_acceptance_matches_backbone", True)
        root["meta"]["backbone_acceptance_regime"] = regime(recorded_acceptance, 8)
        root["meta"]["backbone_boundary_index"] = row.get("boundary_index")
        root["meta"]["refinement_budget_origin"] = "reset_at_sampled_anchor"
        active = [root]
        while active:
            candidates = []
            for parent in active:
                for boundary in self.backbone(parent, self.args.max_refinement_steps):
                    if boundary["meta"]["extend_available"]:
                        candidates.append(self.transition(boundary, "E"))
            selected = choose_children(candidates, self.args.branch_width, self.args.min_expand_acceptance_ratio)
            selected_ids = {n["meta"]["state_id"] for n in selected}
            bad = [n for n in candidates if n["meta"]["current_submit_acceptance_ratio"] < self.args.min_expand_acceptance_ratio
                   and n["meta"]["refine_available"]]
            # One deterministic representative bad rollout per level, never E.
            for node in sorted(bad, key=lambda n: n["meta"]["state_id"])[:self.args.bad_probe_branches]:
                for probe in self.backbone(node, self.args.bad_refinement_steps):
                    probe["meta"]["collection_disposition"] = "negative_refinement_probe"
            for node in candidates:
                if node["meta"]["state_id"] not in selected_ids and node["meta"]["collection_disposition"] == "stored":
                    node["meta"]["collection_disposition"] = "pruned_from_deep_expansion"
            print(f"[structured] problem={row['problem_id']} round={row['round_id']} "
                  f"L={len(active[0]['native'])} E_children={len(candidates)} selected={len(selected)}", flush=True)
            active = selected
        self.writer.flush()

    def finish(self):
        self.writer.flush()
        duplicate_groups = defaultdict(list)
        for meta in self.nodes:
            duplicate_groups[meta["observation_hash"]].append(meta["state_id"])
        for meta in self.nodes:
            meta["observation_duplicate_count"] = len(duplicate_groups[meta["observation_hash"]])
            meta["observation_is_duplicate"] = meta["observation_duplicate_count"] > 1
        # Metadata may be updated after a shard flush; nodes.jsonl is canonical.
        with self.writer.metadata_path.open(encoding="utf-8") as handle:
            storage = {r["state_id"]: r for r in map(json.loads, handle)}
        for meta in self.nodes:
            meta.update(shard=storage[meta["state_id"]]["shard"], row=storage[meta["state_id"]]["row"])
        content = "".join(json.dumps(r) + "\n" for r in self.nodes)
        (self.args.output_dir / "nodes.jsonl").write_text(content, encoding="utf-8")
        self.writer.metadata_path.write_text(content, encoding="utf-8")
        (self.args.output_dir / "edges.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.edges), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone_zip", required=True, type=Path)
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--num_questions", type=int, default=3)
    p.add_argument("--anchors_per_question", "--root_boundaries", type=int, default=4)
    p.add_argument("--max_rounds_per_question", type=int, default=0)
    p.add_argument("--extend_size", type=int, default=8)
    p.add_argument("--max_proposal_tokens", type=int, default=64)
    p.add_argument("--remaining_output_budget", type=int, default=None,
                   help="Optional emitted-token cap; default allows Lmax plus one bonus token")
    p.add_argument("--max_refinement_steps", type=int, default=3)
    p.add_argument("--max_unmask_passes", type=int, default=None,
                   help="Compatibility option: total passes = 1 initial + refinement steps")
    p.add_argument("--branch_width", type=int, choices=[1, 2], default=2)
    p.add_argument("--min_expand_acceptance_ratio", type=float, default=0.5)
    p.add_argument("--bad_probe_branches", type=int, default=1)
    p.add_argument("--bad_refinement_steps", type=int, default=2)
    p.add_argument("--physical_block_size", type=int, default=32)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--drafter_threshold", type=float, default=0.3)
    p.add_argument("--drafter_kv_mode", choices=["none", "stable_block_prefix"],
                   default="none")
    p.add_argument("--raw_top_k", type=int, default=32)
    p.add_argument("--calibration_repeats", type=int, default=3)
    p.add_argument("--verifier_mode", choices=["prefix_kv_calibration", "full_context_no_kv"],
                   default="prefix_kv_calibration")
    p.add_argument("--reference_cache_dir", type=Path, default=None)
    p.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dllm_dir", required=True)
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=1)
    p.add_argument("--target_gpu_memory_gib", type=int, default=9)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--shard_rows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect(args):
    if args.max_unmask_passes is not None:
        args.max_refinement_steps = args.max_unmask_passes - 1
    if args.max_proposal_tokens < 8 or args.extend_size < 1 or (args.max_proposal_tokens - 8) % args.extend_size:
        raise ValueError("Maximum length must be 8 + n * extend_size")
    if min(args.small_block_size, args.physical_block_size, args.calibration_repeats, args.shard_rows) < 1:
        raise ValueError("Block sizes, calibration repeats and shard rows must be positive")
    if not 0 <= args.min_expand_acceptance_ratio <= 1 or min(args.max_refinement_steps,
            args.bad_refinement_steps, args.bad_probe_branches) < 0:
        raise ValueError("Invalid acceptance threshold or refinement budget")
    if args.remaining_output_budget is not None and args.remaining_output_budget < 1:
        raise ValueError("Remaining output budget must be positive")
    if getattr(args, "target_gpu_memory_gib", 9) < 1:
        raise ValueError("Target GPU memory budget must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory; existing data will not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rows = load_anchors(args.backbone_zip, args)
    tokenizer, target, drafter = _load_models(args)
    kv_drafter = getattr(args, "drafter_kv_mode", "none") == "stable_block_prefix"
    engine = (KVExplicitDrafter if kv_drafter else ExplicitDrafter)(drafter, args)
    full_verifier = getattr(args, "verifier_mode", "prefix_kv_calibration") == "full_context_no_kv"
    oracle = (FullContextVerifier if full_verifier else CachedVerifier)(target, tokenizer, args)
    graph = GraphCollector(args, engine, tokenizer.eos_token_id,
                           verifier=oracle if full_verifier else None)
    status, failure = "complete", None
    verifier_mismatches = []
    try:
        # Reuse references/calibration across anchors with the same exact prefix.
        prepared = {}
        collection_began = time.perf_counter()
        for anchor_number, row in enumerate(rows, 1):
            prefix = row["prefix_token_ids"]
            key = identity(prefix)
            if key not in prepared:
                print(f"[prepare] problem={row['problem_id']} round={row['round_id']} calibration", flush=True)
                prepared[key] = oracle.prepare(prefix)
            reference, timings, ref_key = prepared[key]
            if full_verifier:
                accepted, _, _, _ = oracle.score(prefix, row["proposal_token_ids_after_fill"][:8],
                                                  args.max_proposal_tokens)
            else:
                accepted, _, _ = _score_greedy(row["proposal_token_ids_after_fill"][:8], reference,
                                               tokenizer.eos_token_id, args.max_proposal_tokens)
            checked_row = annotate_verifier_acceptance(row, accepted)
            if not checked_row["verifier_acceptance_matches_backbone"]:
                mismatch = dict(state_id=row["state_id"],
                    problem_id=int(row["problem_id"]), round_id=int(row["round_id"]),
                    backbone_recorded_accepted_len=int(row["accepted_len"]),
                    verifier_recomputed_accepted_len=accepted)
                verifier_mismatches.append(mismatch)
                print("[anchor] WARNING verifier acceptance differs from backbone; "
                      f"using recomputed value for labels: {mismatch}", flush=True)
            graph.run_anchor(checked_row, reference, timings, ref_key)
            elapsed = time.perf_counter() - collection_began
            eta = elapsed * (len(rows) - anchor_number) / anchor_number
            print(f"[progress] dataset={args.dataset} anchors={anchor_number}/{len(rows)} "
                  f"nodes={len(graph.nodes)} elapsed_min={elapsed/60:.1f} "
                  f"eta_min={eta/60:.1f}", flush=True)
    except BaseException as exc:
        status, failure = "partial", repr(exc)
        raise
    finally:
        graph.finish()
        (args.output_dir / "verifier_calibration.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in oracle.records), encoding="utf-8")
        roots = [n for n in graph.nodes if n["parent_state_id"] is None]
        duplicate_groups = defaultdict(list)
        for node in graph.nodes:
            duplicate_groups[node["observation_hash"]].append(node["state_id"])
        duplicate_groups = [dict(observation_hash=key, state_ids=ids, count=len(ids))
                            for key, ids in sorted(duplicate_groups.items()) if len(ids) > 1]
        all_regimes = ["full", "near_full", "mid", "early_mismatch"]
        manifest = dict(schema_version=("structured_sparse_sre_v4" if full_verifier or kv_drafter
                                        else "structured_sparse_sre_v3"),
            status=status, failure=failure,
            collector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config=vars(args), nodes=len(graph.nodes), edges=len(graph.edges),
            node_acceptance_histogram=dict(Counter(n["submit_accepted_len"] for n in graph.nodes)),
            current_submit_regime_histogram=dict(Counter(
                n["current_submit_regime"] for n in graph.nodes)),
            backbone_anchor_regime_counts=dict(Counter(
                regime(int(r.get("backbone_recorded_accepted_len", r["accepted_len"])), 8)
                for r in rows)),
            current_anchor_submit_regime_counts=dict(Counter(
                n["current_submit_regime"] for n in roots)),
            anchor_sampling_regime_source="backbone_recorded_accepted_len",
            missing_current_anchor_regimes=sorted(set(all_regimes) -
                {n["current_submit_regime"] for n in roots}),
            observation_duplicate_group_count=len(duplicate_groups),
            observation_duplicate_state_count=sum(g["count"] for g in duplicate_groups),
            observation_duplicate_groups=duplicate_groups,
            backbone_verifier_mismatch_count=len(verifier_mismatches),
            backbone_verifier_mismatches=verifier_mismatches,
            anchors=len(rows), action_semantics={"E": "append masks + one unmask forward in new segment",
                "R": "one unmask forward in earliest unresolved logical frame", "S": "counterfactual fill + cached greedy LCP"},
            transition_backend="full_context_block_causal_replay_without_kv",
            verifier_mode=getattr(args, "verifier_mode", "prefix_kv_calibration"),
            drafter_kv_mode=getattr(args, "drafter_kv_mode", "none"),
            latency_caveat=("Drafter replay costs measured; verifier measured per proposal with full context and no KV. Raw observation/export overhead excluded from action costs."
                if full_verifier else
                "Drafter replay costs measured; verifier cost calibrated with prefix KV, not measured per node. Raw observation/export overhead excluded from action costs."),
            backbone_missing_anchor_regimes=sorted(set(all_regimes) -
                {regime(int(r.get("backbone_recorded_accepted_len", r["accepted_len"])), 8)
                 for r in rows}))
        (args.output_dir / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        archive = args.output_dir / f"{args.dataset}_structured_graph.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for path in args.output_dir.rglob("*"):
                if path.is_file() and path != archive:
                    z.write(path, path.relative_to(args.output_dir),
                            compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED)
        print(f"[archive] status={status} path={archive}", flush=True)


if __name__ == "__main__":
    collect(parse_args())
