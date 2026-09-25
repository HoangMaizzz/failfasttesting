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


class NativeElysiaRunner:
    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model.eval(), tokenizer, args
        self.device = model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def segment(self, prompt):
        """Return every native boundary of the next eight-token segment."""
        args = self.args
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
            collector_max_oracle_snapshots=args.max_refinement_steps + 1,
        )
        inputs = torch.tensor([prompt], dtype=torch.long, device=self.device)
        result = self.model.generate_draft_tokens_arbitrary_length(
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
            # A draft crossing a physical block may need bridge forwards before
            # its first complete same-forward STOP candidate is observable.
            max_denoising_passes=args.extend_size + args.max_refinement_steps,
        )
        stats = result[-1]
        snapshots = sorted(stats.get("oracle_refinement_snapshots", []),
                           key=lambda x: int(x["unmask_forward_index"]))
        if not snapshots:
            raise RuntimeError("Native Elysia generator returned no oracle refinement snapshots")
        previous = None
        previous_forward = None
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
            if not snap.get("hidden_states") or not snap.get("topk_token_ids") or not snap.get("topk_logits"):
                raise RuntimeError("Native snapshot lacks raw hidden/top-k features")
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
            selected = choose_children(heads, self.args.branch_width,
                                       self.args.min_expand_acceptance_ratio)
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
