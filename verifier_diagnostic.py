"""Diagnostic-only tracing and strict replay of recorded controller actions."""
import json
from pathlib import Path


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def install_replay(controller, replay_path):
    from adaptive_td import AdaptiveDecision, ActionEstimate
    replay = json.loads(Path(replay_path).read_text())
    offsets = {}

    def choose(features, **kwargs):
        pid = str(controller.hindsight_problem_id)
        rows = replay["problems"][pid]["decisions"]
        offset = offsets.get(pid, 0)
        if offset >= len(rows):
            raise RuntimeError(f"Diagnostic replay exhausted at problem={pid}, decision={offset}")
        row = rows[offset]
        for key in ("context_len", "refinement_step", "active_block_start", "active_block_end"):
            if int(kwargs[key]) != int(row[key]):
                raise RuntimeError(f"Diagnostic replay diverged: problem={pid}, decision={offset}, field={key}, actual={kwargs[key]}, expected={row[key]}")
        if list(kwargs["draft_proposal"]) != row["draft_proposal"]:
            raise RuntimeError(f"Diagnostic replay proposal diverged at problem={pid}, decision={offset}")
        action = row["action"]
        if action == "stop" and not kwargs["allow_stop"]:
            raise RuntimeError("Recorded STOP is unavailable.")
        offsets[pid] = offset + 1
        controller.hindsight_current_snapshot = None
        controller.decision_count += 1
        zero = ActionEstimate(0., 0., 0., 0.)
        return AdaptiveDecision(
            action=action, reason="diagnostic_action_replay", stop=zero, continue_=zero,
            rho_tokens_per_ms=0., exploration_used=False, latency_ms=0.,
            early_stop_observations=0, calibration_active=False, advantage_mean=0.,
            advantage_risk=0., stop_probability=float(action == "stop"),
            behavior_stop_probability=float(action == "stop"), selected_action_probability=1.,
            importance_weight=1., diagnostics={"action_source": "diagnostic_action_replay"},
        )
    controller.choose = choose
    return replay, offsets


def trace_verifier(model, full_ids, mask, cached_logits, proposal, path, problem_id, round_id):
    import torch
    with torch.inference_mode():
        reference = model(input_ids=full_ids, attention_mask=mask, use_cache=False,
                          logits_to_keep=len(proposal) + 1).logits
        cval, cidx = cached_logits[0].float().topk(2, dim=-1)
        rval, ridx = reference[0].float().topk(2, dim=-1)
        # topk tie ordering is not the production argmax tie-break.
        cached_predictions = cached_logits[0].argmax(dim=-1).tolist()
        reference_predictions = reference[0].argmax(dim=-1).tolist()
        row = {
            "prediction_rule": "argmax", "diagnostic_schema_version": 2,
            "problem_id": int(problem_id), "round_id": int(round_id),
            "context_length": full_ids.shape[1] - len(proposal),
            "full_input_ids": full_ids[0].tolist(), "proposal": list(proposal),
            "cached_predictions": cached_predictions, "full_predictions": reference_predictions,
            "cached_top2_ids": cidx.tolist(), "cached_top2_logits": cval.tolist(),
            "full_top2_ids": ridx.tolist(), "full_top2_logits": rval.tolist(),
            "max_abs_logit_difference": float((cached_logits.float() - reference.float()).abs().max()),
            "different_prediction_positions": [i for i, (a, b) in enumerate(zip(cached_predictions, reference_predictions)) if a != b],
        }
    append_record(path, row)
