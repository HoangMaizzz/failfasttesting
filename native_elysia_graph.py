"""Build sparse S/R/E graphs from Elysia's native denoising snapshots.

The old raw archive supplies only verifier prefixes and anchor IDs. No archived
draft token or confidence is used to create a new node: every eight-token
segment is generated afresh by generate_draft_tokens_arbitrary_length.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from sparse_extend_world_model_collector import MASK_ID
from structured_sparse_collector import GraphCollector, choose_children


class NativeEosWithoutSnapshot(RuntimeError):
    """The native drafter emitted EOS before exposing a feature snapshot."""

    def __init__(self, candidate_token_ids, stats):
        self.candidate_token_ids = list(candidate_token_ids)
        self.stats = {key: stats.get(key) for key in (
            "stop_reason", "native_termination_reason", "actual_spec_len",
            "oracle_snapshot_attempts", "oracle_snapshot_skipped_missing_fill",
            "forward_pass_breakdown")}
        super().__init__(
            "Native Elysia terminated on EOS before exposing a raw snapshot; "
            f"candidate_len={len(self.candidate_token_ids)}, stats={self.stats}"
        )


class NativeElysiaRunner:
    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model.eval(), tokenizer, args
        self.device = model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def segment(self, prompt, max_snapshots=None):
        """Return every native boundary of the next eight-token segment."""
        args = self.args
        snapshot_limit = (args.max_refinement_steps + 1
            if max_snapshots is None else int(max_snapshots))
        base_pass_limit = args.extend_size + args.max_refinement_steps
        # A segment beginning near a physical-block boundary can need several
        # bridge forwards before all positions in its first logical proposal
        # have same-forward predictions. Reserve one complete physical-block
        # worth of extra forwards for a retry if the first attempt observes no
        # usable boundary.
        retry_pass_limit = base_pass_limit + max(
            1, (args.physical_block_size + args.small_block_size - 1)
            // args.small_block_size)

        def run_generator(pass_limit):
            # Optional V2 native-tail instrumentation. Reset on BOTH attempts so
            # forward IDs stay aligned with hidden_state_forward_pass on retry.
            reset = getattr(self.model, '_wm_v2_reset_capture', None)
            if reset is not None:
                reset()
            call_args = SimpleNamespace(
                target_tokenizer=self.tokenizer,
                full_refinement_oracle=True,
                raw_top_k=args.raw_top_k,
                collect_bucket_oracle=True,
                bucket_oracle_force_continue=True,
                frontier_stop_mode="disabled",
                adaptive_td=False,
                adaptive_freeze=False,
                global_oracle_graph=False,
                strict_greedy_local_oracle=False,
                bucket_acceptance_calibration={},
                bucket_min_observations=8,
                bucket_prior_strength=8.0,
                bucket_gain_calibration={
                    "length_score_masks": {}, "score_masks": {}, "length_score": {},
                    "score": {}, "step": {}, "global": [0.0, 0],
                },
                bucket_current_context_len=len(prompt),
                collector_parent_native_length=0,
                collector_parent_fill_tokens=[],
                collector_max_oracle_snapshots=snapshot_limit,
                # V2 learns native latent-feature dynamics only.  It does not
                # consume token IDs, token probabilities, or verifier labels.
                wm_v2_capture_native_latent=bool(
                    getattr(args, "wm_v2_capture_native_latent", False)
                ),
                wm_v2_cut_layer=int(getattr(args, "wm_v2_cut_layer", 24)),
                wm_v2_suffix_layers=int(getattr(args, "wm_v2_suffix_layers", 4)),
                wm_v2_memory_window=int(getattr(args, "wm_v2_memory_window", 128)),
            )
            return self.model.generate_draft_tokens_arbitrary_length(
                inputs,
                max_new_tokens=3 * args.physical_block_size,
                small_block_size=args.small_block_size,
                block_size=args.physical_block_size,
                threshold=args.drafter_threshold,
                do_sample=False, temperature=0.0, top_p=1.0, top_k=0.0,
                # Match the failfast.py production call: prefix KV, but no mutable
                # block cache (use_block_cache is not passed there).
                is_drafter=True, spec_len=args.extend_size,
                return_prefill_kvs=True, prev_prefill_output=None,
                args=call_args, lowconf_threshold=0.0,
                max_spec_len=args.extend_size, incr_len=args.extend_size,
                last_round_rejected=None, return_frontier_stats=True,
                max_denoising_passes=pass_limit,
            )

        inputs = torch.tensor([prompt], dtype=torch.long, device=self.device)
        result = run_generator(base_pass_limit)
        stats = result[-1]
        snapshots = sorted(stats.get("oracle_refinement_snapshots", []),
                           key=lambda x: int(x["unmask_forward_index"]))
        if not snapshots:
            first_attempt = stats
            if first_attempt.get("native_termination_reason") == "eos":
                generated = result[0][0, len(prompt):].detach().cpu().tolist()
                eos_id = self.tokenizer.eos_token_id
                if eos_id in generated:
                    eos_pos = generated.index(eos_id)
                    if MASK_ID not in generated[:eos_pos]:
                        raise NativeEosWithoutSnapshot(generated[:eos_pos + 1], first_attempt)
            result = run_generator(retry_pass_limit)
            stats = result[-1]
            snapshots = sorted(stats.get("oracle_refinement_snapshots", []),
                               key=lambda x: int(x["unmask_forward_index"]))
            if not snapshots:
                def brief(value):
                    return {key: value.get(key) for key in (
                        "stop_reason", "native_termination_reason",
                        "actual_spec_len", "oracle_snapshot_attempts",
                        "oracle_snapshot_skipped_missing_fill",
                        "collector_pass_limit_reached", "collector_pass_limit",
                        "forward_pass_breakdown")}
                if stats.get("native_termination_reason") == "eos":
                    generated = result[0][0, len(prompt):].detach().cpu().tolist()
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id in generated:
                        eos_pos = generated.index(eos_id)
                        if MASK_ID not in generated[:eos_pos]:
                            raise NativeEosWithoutSnapshot(generated[:eos_pos + 1], brief(stats))
                raise RuntimeError(
                    "Native Elysia generator returned no oracle refinement snapshots "
                    f"after retry (prefix_len={len(prompt)}, base_pass_limit={base_pass_limit}, "
                    f"retry_pass_limit={retry_pass_limit}, first_stats={brief(first_attempt)}, "
                    f"retry_stats={brief(stats)})"
                )
        previous = None
        previous_forward = None
        previous_v2 = None
        for snap in snapshots:
            native = [int(x) for x in snap["proposal_token_ids_before_fill"]]
            filled = [int(x) for x in snap["proposal_token_ids_after_fill"]]
            forward_index = int(snap["unmask_forward_index"])
            if len(native) != args.extend_size or len(filled) != args.extend_size:
                raise RuntimeError("Native snapshot does not cover the requested segment")
            if MASK_ID in filled or forward_index < 1:
                raise RuntimeError("Native snapshot has unresolved STOP or invalid forward index")
            if previous_forward is not None and forward_index != previous_forward + 1:
                raise RuntimeError("Nonconsecutive native snapshots cannot form a one-step R edge")
            v2_capture = bool(getattr(args, "wm_v2_capture_native_latent", False))
            if not v2_capture and (
                not snap.get("hidden_states") or not snap.get("topk_token_ids") or not snap.get("topk_logits")
            ):
                raise RuntimeError("Native snapshot lacks raw hidden/top-k features")
            if v2_capture:
                _validate_v2_native_snapshot(snap, previous_v2, len(prompt), args)
                previous_v2 = snap
            if any(a != MASK_ID and a != b for a, b in zip(native, filled)):
                raise RuntimeError("STOP fill modified a committed token")
            if previous is not None:
                if any(a != MASK_ID and a != b for a, b in zip(previous, native)):
                    raise RuntimeError("Native unmask sequence changed a committed token")
                # A native forward can update a different physical sub-block.
                # Its target proposal may therefore remain unchanged while the
                # observable STOP fill or elapsed cost changes. Keep that R
                # edge; only remasking a committed token is impossible.
                if sum(x == MASK_ID for x in native) > sum(x == MASK_ID for x in previous):
                    raise RuntimeError("Native forward increased proposal mask count")
            previous = native
            previous_forward = forward_index
        return snapshots


def _validate_v2_native_snapshot(snapshot, previous, proposal_start, args):
    """Audit actual native forwards rather than infer one-step R from mask counts."""
    required = (
        "wm_v2_native_latent_block", "wm_v2_physical_mask", "wm_v2_physical_event",
        "wm_v2_mask_before_forward", "wm_v2_proposal_span", "wm_v2_native_active_region",
        "wm_v2_forward_region", "wm_v2_proposal_start", "wm_v2_proposal_length",
        "wm_v2_physical_block_start", "wm_v2_prefix_kv_length", "wm_v2_prefix_kv_summary",
        "wm_v2_memory_layer_indices", "wm_v2_valid_positions",
        "wm_v2_prefix_kv_valid", "wm_v2_native_refine_available", "wm_v2_native_eos_ready",
        "hidden_state_forward_pass", "draft_passes_elapsed",
    )
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise RuntimeError(f"V2 native capture missing corrected physical fields: {missing}")
    if int(snapshot["wm_v2_proposal_start"]) != int(proposal_start):
        raise RuntimeError("V2 proposal absolute start differs from real native prefix")
    if int(snapshot["wm_v2_proposal_length"]) != int(args.extend_size):
        raise RuntimeError("V2 proposal geometry changed requested native segment length")
    # A usable same-forward STOP snapshot crossing a physical boundary can only
    # observe the final physical block; the earlier proposal part is committed.
    expected_frame = (proposal_start + int(args.extend_size) - 1) // int(args.physical_block_size)
    if int(snapshot["wm_v2_physical_block_start"]) != expected_frame * int(args.physical_block_size):
        raise RuntimeError("V2 observable native frame differs from final proposal block")
    if int(snapshot["hidden_state_forward_pass"]) != int(snapshot["draft_passes_elapsed"]):
        raise RuntimeError("V2 residual and postcommit structure do not share a native forward")
    if snapshot["wm_v2_memory_layer_indices"] != [0, 7, 15, 23]:
        raise RuntimeError("V2 prefix memory must summarize upstream layers [0,7,15,23]")
    before = torch.as_tensor(snapshot["wm_v2_mask_before_forward"], dtype=torch.bool)
    after = torch.as_tensor(snapshot["wm_v2_physical_mask"], dtype=torch.bool)
    event = torch.as_tensor(snapshot["wm_v2_physical_event"], dtype=torch.bool)
    region = torch.as_tensor(snapshot["wm_v2_forward_region"], dtype=torch.bool)
    if any(vector.shape != (int(args.physical_block_size),) for vector in (before, after, event, region)):
        raise RuntimeError("V2 native physical structural arrays have wrong shape")
    if (after & ~before).any() or not torch.equal(event, before & ~after):
        raise RuntimeError("V2 native commit event is not the full physical mask delta")
    if not event.any() or (event & ~region).any():
        raise RuntimeError("V2 native commit escaped its actual scheduler sub-block")
    if previous is None:
        return
    if not bool(previous["wm_v2_native_refine_available"]):
        raise RuntimeError("V2 R edge follows a native terminal proposal")
    if int(snapshot["draft_passes_elapsed"]) != int(previous["draft_passes_elapsed"]) + 1:
        raise RuntimeError("V2 one-step R must correspond to exactly one actual native forward")
    if int(snapshot["wm_v2_physical_block_start"]) != int(previous["wm_v2_physical_block_start"]):
        raise RuntimeError("V2 R changed physical frame")
    if not torch.equal(before, torch.as_tensor(previous["wm_v2_physical_mask"], dtype=torch.bool)):
        raise RuntimeError("V2 next native forward input mask differs from prior postcommit mask")
    if not torch.equal(region, torch.as_tensor(previous["wm_v2_native_active_region"], dtype=torch.bool)):
        raise RuntimeError("V2 next native forward did not execute the recorded NEXT R region")
    if (
        int(snapshot["wm_v2_prefix_kv_length"]) != int(previous["wm_v2_prefix_kv_length"])
        or bool(snapshot["wm_v2_prefix_kv_valid"]) != bool(previous["wm_v2_prefix_kv_valid"])
        or not torch.equal(torch.as_tensor(snapshot["wm_v2_prefix_kv_summary"]),
                           torch.as_tensor(previous["wm_v2_prefix_kv_summary"]))
    ):
        raise RuntimeError("V2 native prefix KV changed during one-step R")


class NativeElysiaGraphCollector(GraphCollector):
    """R = successive snapshots; E = top-1-filled parent + new native segment."""

    def __init__(self, args, feature_engine, native_runner, eos_id, verifier):
        super().__init__(args, feature_engine, eos_id, verifier=verifier)
        self.native_runner = native_runner
    def _node_from_snapshot(self, source, parent, action, snapshot,
                            remaining_snapshots, prior):
        native = prior + [int(x) for x in snapshot["proposal_token_ids_before_fill"]]
        filled = prior + [int(x) for x in snapshot["proposal_token_ids_after_fill"]]
        if MASK_ID in prior or len(native) != len(filled):
            raise RuntimeError("Extend must start from a fully top-1-committed parent")
        observation = self.engine.observe(self.prefix, native)
        # Feature re-encoding is diagnostic. Submission comes only from the
        # native forward snapshot; never substitute the re-encode's top-1.
        observation = dict(observation)
        observation["filled"] = filled
        observation["native_active_hidden_states"] = snapshot.get("hidden_states", [])
        observation["native_active_hidden_layer_indices"] = snapshot.get("hidden_layer_indices", [])
        observation["native_active_topk_token_ids"] = snapshot.get("topk_token_ids", [])
        observation["native_active_topk_logits"] = snapshot.get("topk_logits", [])
        confidences = [float(x) for x in snapshot["confidences"]]
        if len(confidences) != self.args.extend_size:
            raise RuntimeError("Native snapshot confidence length does not match segment")
        observation["filled_probabilities"] = (
            ([] if parent is None or action is None else
             list(parent["obs"]["filled_probabilities"][:len(prior)]))
            + confidences)
        predictions = list(observation["predictions"])
        for i, token in enumerate(native):
            if token == MASK_ID:
                predictions[i] = filled[i]
        observation["predictions"] = predictions
        elapsed = float(snapshot["draft_latency_elapsed_ms"])
        if parent is None:
            action_ms, cumulative_ms, refine_steps = 0.0, elapsed, 0
            prefill = []
        elif action == "E":
            action_ms = elapsed
            cumulative_ms = parent["meta"]["draft_latency_from_anchor_ms"] + action_ms
            refine_steps = 0
            prefill = [i for i, token in enumerate(parent["native"]) if token == MASK_ID]
        else:
            previous_elapsed = float(parent["meta"]["native_segment_elapsed_ms"])
            action_ms = elapsed - previous_elapsed
            if action_ms < 0:
                raise RuntimeError("Native forward latency went backwards")
            cumulative_ms = parent["meta"]["draft_latency_from_anchor_ms"] + action_ms
            refine_steps = parent["meta"]["refine_steps_since_extend"] + 1
            prefill = []
        native_passes = int(snapshot["draft_passes_elapsed"])
        if parent is None:
            total_passes = native_passes
        elif action == "E":
            total_passes = parent["meta"]["draft_passes_from_anchor"] + native_passes
        else:
            total_passes = (parent["meta"]["draft_passes_from_anchor"]
                + native_passes - parent["meta"]["native_draft_passes_elapsed"])
        extra_meta = dict(
            submit_candidate_source="native_elysia_same_forward_top1",
            feature_source="full_state_reencode_diagnostic_not_action_forward",
            native_unmask_forward_index=int(snapshot["unmask_forward_index"]),
            native_segment_elapsed_ms=elapsed,
            native_draft_passes_elapsed=native_passes,
            native_snapshot_masks_remaining=int(snapshot["masks_remaining"]),
            native_snapshot_newly_unmasked_positions=list(
                snapshot.get("newly_unmasked_positions", [])),
            native_snapshot_stop_token_ids=list(snapshot["proposal_token_ids_after_fill"]),
            native_snapshot_confidences=confidences,
            native_snapshot_hidden_state_source=snapshot.get("hidden_state_source"),
            native_hidden_start_offset=snapshot.get("native_hidden_start_offset"),
            native_topk_start_offset=snapshot.get("native_topk_start_offset"),
            anchor_source_proposal_used_as_state=False,
            refine_available=bool(remaining_snapshots),
            draft_passes_from_anchor=total_passes,
        )
        newly_committed = [len(prior) + int(x) for x in
                           snapshot.get("newly_unmasked_positions", [])]
        node = self.node(self.prefix, native, observation, refine_steps, cumulative_ms,
                         source, parent, action, action_ms,
                         newly_committed, prefill, [prior[i] for i in prefill], extra_meta)
        meta = node["meta"]
        if parent is not None:
            edge = self.edges[-1]
            previous_passes = (parent["meta"]["native_draft_passes_elapsed"]
                               if action == "R" else 0)
            previous_unmask = (parent["meta"]["native_unmask_forward_index"]
                               if action == "R" else 0)
            edge.update(action_cost_ms=action_ms,
                action_cost_source="native_elysia_generator_reported_forwards",
                action_forward_count=native_passes - previous_passes,
                native_unmask_forwards=int(snapshot["unmask_forward_index"]) - previous_unmask,
                action_cost_includes_kv_reuse=True,
                pre_extension_fill_source=("native_parent_same_forward_top1" if action == "E" else None))
        return node

    def _rollout(self, source, parent=None):
        prior = [] if parent is None else list(parent["obs"]["filled"])
        snapshots = self.native_runner.segment(self.prefix + prior)
        chain = []
        for index, snapshot in enumerate(snapshots):
            previous = parent if index == 0 else chain[-1]
            action = None if previous is None else ("E" if index == 0 else "R")
            node = self._node_from_snapshot(source, previous, action, snapshot,
                                            snapshots[index + 1:], prior)
            chain.append(node)
        return chain

    def run_anchor(self, row, reference, timings, key):
        self.reference, self.timings, self.reference_key = reference, timings, key
        self.prefix, self.source, self.anchor_id = row["prefix_token_ids"], row, row["state_id"]
        chain = self._rollout(row)
        root = chain[0]
        root["meta"]["backbone_accepted_len"] = row.get("backbone_recorded_accepted_len", row["accepted_len"])
        root["meta"]["backbone_boundary_index"] = row.get("boundary_index")
        active = [chain]
        while active:
            candidate_chains = []
            for parent_chain in active:
                for boundary in parent_chain:
                    if boundary["meta"]["extend_available"]:
                        candidate_chains.append(self._rollout(row, boundary))
            heads = [c[0] for c in candidate_chains]
            # The confidence threshold governs native token commitment, not
            # whether a counterfactual extension is allowed. Keep expanding
            # the best/diverse heads even when the verifier accepts nothing.
            # EOS and max_proposal_tokens are the only extension stops.
            selected = choose_children(heads, self.args.branch_width, 0.0)
            selected_ids = {n["meta"]["state_id"] for n in selected}
            for child_chain in candidate_chains:
                if child_chain[0]["meta"]["state_id"] not in selected_ids:
                    for child in child_chain:
                        child["meta"]["collection_disposition"] = "pruned_from_deep_expansion"
            print(f"[native] problem={row['problem_id']} round={row['round_id']} "
                  f"L={len(active[0][0]['native'])} E_children={len(heads)} "
                  f"selected={len(selected)}", flush=True)
            active = [c for c in candidate_chains if c[0]["meta"]["state_id"] in selected_ids]
        self.writer.flush()
