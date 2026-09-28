"""Real native R/E experiences. Bounded deterministic replay, not full-tree collection.

The current native generator cannot suspend/resume Python execution. R therefore
replays the frozen segment to the next snapshot and checks its predecessor. This
preserves audited native transitions, but replay wall time is NOT action latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F

from world_model_core import Observation

MASK_ID = 151665


def native_observation(uid, question, round_id, prefix, prior, snapshot, parent,
                       hidden_dim, top_k, args, refinement):
    before = list(prior) + list(snapshot["proposal_token_ids_before_fill"])
    filled = list(prior) + list(snapshot["proposal_token_ids_after_fill"])
    length, start = len(before), len(prior)
    if len(before) != len(filled) or MASK_ID in filled:
        raise ValueError("Invalid native STOP snapshot")
    if any(a != MASK_ID and a != b for a, b in zip(before, filled)):
        raise ValueError("STOP changed committed tokens")
    ids = torch.tensor(list(zip(before, filled)), dtype=torch.long)
    selected_layers = getattr(args, "hidden_layers", [14, 28])
    hidden = torch.zeros(length, len(selected_layers), hidden_dim, dtype=torch.float16)
    gaps = torch.zeros(length, top_k, dtype=torch.float16)
    topk_ids = torch.zeros(length, top_k, dtype=torch.long)
    scalars = torch.zeros(length, 16)
    if parent is not None:
        count = min(length, parent.length)
        hidden[:count] = parent.hidden[:count]
        gaps[:count] = parent.gaps[:count]
        scalars[:count] = parent.scalars[:count]
        if parent.topk_ids is not None:
            topk_ids[:count] = parent.topk_ids[:count]
        # Ages count decision transitions, not elapsed milliseconds or forwards.
        scalars[:count, 5:7] += 1 / 8
    layer_ids = list(snapshot["hidden_layer_indices"])
    if not selected_layers or not all(layer in layer_ids for layer in selected_layers):
        raise ValueError(f"Need native layers {selected_layers}, received {layer_ids}")
    raw_hidden = snapshot["hidden_states"]
    h_offset = int(snapshot["native_hidden_start_offset"])
    for slot, layer in enumerate(selected_layers):
        values = torch.as_tensor(raw_hidden[layer_ids.index(layer)], dtype=torch.float16)
        if values.ndim != 2 or values.shape[-1] != hidden_dim:
            raise ValueError("Native hidden shape mismatch")
        for row in range(len(values)):
            position = start + h_offset + row
            if start <= position < length:
                hidden[position, slot] = values[row]
                scalars[position, 3] = 1
                scalars[position, 5] = 0
    raw_logits = torch.as_tensor(snapshot["topk_logits"], dtype=torch.float32)
    raw_ids = torch.as_tensor(snapshot["topk_token_ids"], dtype=torch.long)
    if raw_logits.ndim != 2 or raw_logits.shape[-1] < top_k:
        raise ValueError("Native top-k shape mismatch")
    logit_offset = int(snapshot["native_topk_start_offset"])
    for row in range(len(raw_logits)):
        position = start + logit_offset + row
        if start <= position < length:
            gaps[position] = (raw_logits[row, :top_k] - raw_logits[row, 0]).half()
            topk_ids[position] = raw_ids[row, :top_k]
            scalars[position, 4] = 1
            scalars[position, 6] = 0
    confidences = torch.tensor(snapshot["confidences"], dtype=torch.float32)
    if len(confidences) != length-start or not bool(torch.isfinite(confidences).all()):
        raise ValueError("Native confidence shape/value mismatch")
    scalars[start:, 7] = confidences
    scalars[start:, 8] = 1
    scalars[:, 0] = ids[:, 0].eq(MASK_ID)
    scalars[:, 1] = ~ids[:, 0].eq(MASK_ID)
    scalars[:, 2] = 0
    for position in snapshot.get("newly_unmasked_positions", []):
        if 0 <= position < length-start:
            scalars[start+position, 2] = 1
    relative = torch.arange(length).float()
    absolute = relative + len(prefix)
    scalars[:, 9] = relative / max(1, length)
    scalars[:, 10] = absolute.remainder(args.physical_block_size) / args.physical_block_size
    scalars[:, 11] = absolute.remainder(args.small_block_size) / args.small_block_size
    scalars[:, 12] = relative >= start
    scalars[:, 13] = 1  # captured by native forward, before counterfactual fill
    scalars[:, 14] = ids[:, 0].ne(ids[:, 1])
    scalars[:, 15] = args.drafter_threshold
    context = torch.tensor([len(prefix)/1024, length/64, start/64, refinement/3,
        args.drafter_threshold, args.physical_block_size/64,
        args.small_block_size/64, args.max_proposal_tokens/64], dtype=torch.float32)
    if not torch.isfinite(hidden).all() or not torch.isfinite(gaps).all():
        raise ValueError("Nonfinite native features")
    if not scalars[start:, 3].any() or not scalars[start:, 4].any():
        raise ValueError("No native coverage of current segment")
    history = torch.zeros(length, 4)
    if parent is not None:
        count = min(parent.length, length)
        valid = (parent.scalars[:count, 3] > 0) & (scalars[:count, 3] > 0)
        history[:count, 0] = scalars[:count, 7] - parent.scalars[:count, 7]
        history[:count, 1] = ids[:count, 1].ne(parent.ids[:count, 1]).float()
        history[:count, 2] = (1-F.cosine_similarity(hidden[:count].float().flatten(1),
                               parent.hidden[:count].float().flatten(1), dim=-1))*valid
        history[:count, 3] = valid.float()
    return Observation(uid, question, round_id, ids, hidden, gaps, scalars, context,
                       prefix_ids=torch.tensor(prefix, dtype=torch.long), topk_ids=topk_ids, history=history)


@dataclass
class NativeState:
    observation: Observation
    prefix: list
    prior: list
    snapshot: dict
    snapshot_index: int
    emitted: list
    terminal_reason: str | None
    refine_exhausted: bool = False
    extend_exhausted: bool = False
    submitted: bool = False


class NativeTrainingEnvironment:
    def __init__(self, runner, verifier, eos_id, hidden_dim, args, on_state):
        self.runner, self.verifier, self.eos_id = runner, verifier, eos_id
        self.hidden_dim, self.args, self.on_state = hidden_dim, args, on_state
        self.counter = 0
        self.stats = dict(native_calls=0, verifier_calls=0, unavailable_R=0,
                          executed_S=0, executed_R=0, executed_E=0)

    def segment(self, prompt, limit):
        self.stats["native_calls"] += 1
        print(f"[native-step] call={self.stats['native_calls']} prefix_tokens={len(prompt)} "
              f"snapshot_limit={limit}", flush=True)
        return self.runner.segment(prompt, max_snapshots=limit)

    def make_state(self, question, round_id, prefix, prior, snapshot, index, parent,
                   remaining, action, wall_seconds):
        self.counter += 1
        observation = native_observation(f"state_{self.counter:08d}", question, round_id,
            prefix, prior, snapshot, None if parent is None else parent.observation,
            self.hidden_dim, self.args.raw_top_k, self.args, index)
        proposal = observation.ids[:, 1].tolist()
        terminal = "candidate_eos" if self.eos_id in proposal else None
        state = NativeState(observation, list(prefix), list(prior), snapshot, index,
                            [], terminal)
        if parent is not None:
            previous = parent.observation
            if action == "E":
                if proposal[:previous.length] != previous.ids[:, 1].tolist():
                    raise RuntimeError("E failed to commit the parent's STOP candidate")
            elif any(a != MASK_ID and a != b for a, b in
                     zip(previous.ids[:, 0].tolist(), observation.ids[:, 0].tolist())):
                raise RuntimeError("R changed a committed token")
        if action in ("R", "E"):
            self.stats[f"executed_{action}"] += 1
        self.on_state(state, parent, action, dict(native_replay_wall_seconds=wall_seconds,
            verifier_ms=None, timing_is_training_overhead_not_action_cost=True))
        return state

    def start(self, question, round_id, prefix, remaining):
        began = time.perf_counter()
        snapshots = self.segment(prefix, 1)
        return self.make_state(question, round_id, prefix, [], snapshots[0], 0, None,
                               remaining, None, time.perf_counter()-began)

    def actions(self, state):
        if state.submitted:
            return []
        if state.terminal_reason:
            return ["S"]
        actions = ["S"]
        if (not state.refine_exhausted and state.snapshot_index < self.args.max_refinement_steps
                and bool(state.observation.ids[:, 0].eq(MASK_ID).any())):
            actions.append("R")
        if (not state.extend_exhausted and
                state.observation.length + self.args.extend_size <= self.args.max_proposal_tokens):
            actions.append("E")
        return actions

    def submit(self, state, remaining):
        """Only an actual STOP invokes the verifier. No re-encoding of the candidate."""
        if "S" not in self.actions(state):
            raise ValueError("State already submitted")
        accepted, _, emitted, elapsed = self.verifier.score(
            state.prefix, state.observation.ids[:, 1].tolist(), remaining)
        if not 0 <= accepted <= state.observation.length or not emitted:
            raise RuntimeError("Invalid verifier result")
        self.stats["verifier_calls"] += 1
        self.stats["executed_S"] += 1
        state.observation.accepted = int(accepted)
        teacher = getattr(self.verifier, "last_teacher", None)
        if teacher is not None:
            margins = torch.tensor(teacher["margin"], dtype=torch.float32)
            if len(margins) != state.observation.length or not torch.isfinite(margins).all():
                raise RuntimeError("Invalid verifier teacher target")
            state.observation.teacher_margin = margins
        state.emitted = list(emitted)
        state.submitted = True
        return elapsed

    def step(self, state, action, remaining):
        if action not in ("R", "E") or action not in self.actions(state):
            raise ValueError(f"Action {action} is not observable/legal at this state")
        began = time.perf_counter()
        o = state.observation
        if action == "E":
            prior = o.ids[:, 1].tolist()
            try:
                snapshots = self.segment(state.prefix + prior, 1)
            except RuntimeError as error:
                # Disable only E for this state if native generation exposes no
                # complete next snapshot; STOP and any legal R remain usable.
                if str(error) != "Native Elysia generator returned no oracle refinement snapshots":
                    raise
                state.extend_exhausted = True
                self.stats["unavailable_E"] = self.stats.get("unavailable_E", 0) + 1
                return None
            if not snapshots:
                state.extend_exhausted = True
                self.stats["unavailable_E"] = self.stats.get("unavailable_E", 0) + 1
                return None
            snapshot, index = snapshots[0], 0
        else:
            prior = state.prior
            snapshots = self.segment(state.prefix + prior, state.snapshot_index + 2)
            if len(snapshots) <= state.snapshot_index:
                raise RuntimeError("Native replay lost the source snapshot")
            replayed = snapshots[state.snapshot_index]
            for key in ("proposal_token_ids_before_fill", "proposal_token_ids_after_fill",
                        "unmask_forward_index"):
                if replayed[key] != state.snapshot[key]:
                    raise RuntimeError(f"Native predecessor replay drift in {key}; refusing false R edge")
            if len(snapshots) <= state.snapshot_index + 1:
                self.stats["unavailable_R"] += 1
                state.refine_exhausted = True
                return None  # exhaustion is not a zero-yield training label
            index = state.snapshot_index + 1
            snapshot = snapshots[index]
        return self.make_state(o.question, o.round_id, state.prefix, prior, snapshot,
            index, state, remaining, action, time.perf_counter()-began)
