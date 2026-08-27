import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from adaptive_td import CONTINUE, STOP


def build_oracle_state_key(
    problem_id,
    context_len,
    proposal_length,
    refinement_step,
    draft_proposal,
):
    payload = json.dumps(
        {
            "problem_id": int(problem_id),
            "context_len": int(context_len),
            "proposal_length": int(proposal_length),
            "refinement_step": int(refinement_step),
            "draft_proposal": [
                None if token_id is None else int(token_id)
                for token_id in (draft_proposal or [])
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GreedyBranch:
    local_cost_ms: float
    emitted_tokens: int


@dataclass(frozen=True)
class GreedyDecision:
    action: str
    immediate_action: str
    stop_score_ms: float
    continue_score_ms: float
    stop_penalty_ms: float
    continue_penalty_ms: float
    stop_extra_calls: float
    continue_extra_calls: float
    delta_j_ms: float
    tie_fallback_used: bool


def load_verifier_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("mean_verify_latency_ms", "mean_tokens_per_verify"):
        value = float(profile[field])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid verifier profile: {field}={value}")
    return profile


def predict_verifier_latency_ms(profile, context_len, proposal_len):
    """Predict from a frozen binned profile, falling back to its global mean."""
    bins = profile.get("latency_bins") or []
    if not bins:
        return float(profile["mean_verify_latency_ms"])
    context_bucket = max(0, int(context_len) // int(profile["context_bucket_size"]))
    proposal_bucket = max(
        1,
        math.ceil(int(proposal_len) / int(profile["proposal_bucket_size"])),
    )
    best = min(
        bins,
        key=lambda row: (
            4 * abs(int(row["context_bucket"]) - context_bucket)
            + abs(int(row["proposal_bucket"]) - proposal_bucket),
            -int(row["observations"]),
        ),
    )
    return float(best["mean_verify_latency_ms"])


def format_outer_path(extension_count):
    extension_count = int(extension_count)
    if extension_count < 0:
        raise ValueError("extension_count must be non-negative")
    return " -> ".join(["EXTEND"] * extension_count + ["VERIFY"])


def one_action_rollout_scripts(action_script):
    """Control only the current action; later decisions use baseline CONTINUE."""
    prefix = tuple(action_script)
    return prefix + (STOP,), prefix + (CONTINUE,)


def choose_strict_greedy_action(
    stop,
    continue_,
    *,
    mean_verify_latency_ms,
    mean_tokens_per_verify,
    epsilon_ms=1.0,
    baseline_action=CONTINUE,
):
    mean_verify_latency_ms = float(mean_verify_latency_ms)
    mean_tokens_per_verify = float(mean_tokens_per_verify)
    epsilon_ms = float(epsilon_ms)
    if mean_verify_latency_ms <= 0.0 or mean_tokens_per_verify <= 0.0:
        raise ValueError("verifier means must be positive")
    if epsilon_ms < 0.0:
        raise ValueError("epsilon_ms must be non-negative")
    if baseline_action not in (STOP, CONTINUE):
        raise ValueError(f"invalid baseline action: {baseline_action}")

    stop_extra_calls = max(
        0.0,
        float(continue_.emitted_tokens - stop.emitted_tokens),
    ) / mean_tokens_per_verify
    continue_extra_calls = max(
        0.0,
        float(stop.emitted_tokens - continue_.emitted_tokens),
    ) / mean_tokens_per_verify
    stop_penalty_ms = stop_extra_calls * mean_verify_latency_ms
    continue_penalty_ms = continue_extra_calls * mean_verify_latency_ms
    stop_score_ms = float(stop.local_cost_ms) + stop_penalty_ms
    continue_score_ms = float(continue_.local_cost_ms) + continue_penalty_ms
    delta_j_ms = continue_score_ms - stop_score_ms
    immediate_delta = float(continue_.local_cost_ms) - float(stop.local_cost_ms)
    immediate_action = STOP if immediate_delta > 0.0 else CONTINUE
    tie = abs(delta_j_ms) <= epsilon_ms
    if tie:
        action = baseline_action
    else:
        action = STOP if delta_j_ms > 0.0 else CONTINUE
    return GreedyDecision(
        action=action,
        immediate_action=immediate_action,
        stop_score_ms=stop_score_ms,
        continue_score_ms=continue_score_ms,
        stop_penalty_ms=stop_penalty_ms,
        continue_penalty_ms=continue_penalty_ms,
        stop_extra_calls=stop_extra_calls,
        continue_extra_calls=continue_extra_calls,
        delta_j_ms=delta_j_ms,
        tie_fallback_used=tie,
    )
