from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RenewalDecision:
    stop_ms_per_output: float
    continue_ms_per_output: float | None
    predicted_gain: float | None
    should_continue: bool


def position_bucket(position: int) -> str:
    position = max(0, int(position))
    if position < 2:
        return "0-1"
    if position < 4:
        return "2-3"
    if position < 8:
        return "4-7"
    if position < 16:
        return "8-15"
    if position < 24:
        return "16-23"
    if position < 32:
        return "24-31"
    return "32+"


def expected_accepted_prefix(probabilities: Sequence[float]) -> float:
    expected_prefix = 0.0
    survival = 1.0
    for probability in probabilities:
        survival *= min(0.98, max(0.02, float(probability)))
        expected_prefix += survival
    return expected_prefix


def calibrated_acceptance_probability(
    raw_probability: float,
    bucket_counts: Sequence[Sequence[float] | None],
    *,
    min_observations: int = 8,
    prior_strength: float = 8.0,
) -> float:
    raw_probability = min(0.98, max(0.02, float(raw_probability)))
    fallback = None
    for stats in bucket_counts:
        if not stats:
            continue
        accepted, total = stats
        total = float(total)
        probability = (
            float(accepted) + float(prior_strength) * raw_probability
        ) / (total + float(prior_strength))
        probability = min(0.98, max(0.02, probability))
        if total >= int(min_observations):
            return probability
        if fallback is None or total > fallback[0]:
            fallback = (total, probability)
    return raw_probability if fallback is None else fallback[1]


def predict_next_gain(
    expected_prefix_history: Sequence[float],
    bucket_estimate: float | None = None,
    bucket_weight: float = 0.0,
) -> float | None:
    if len(expected_prefix_history) < 2:
        return (
            None
            if bucket_estimate is None
            else max(0.0, float(bucket_estimate))
        )
    current_gain = max(
        0.0,
        float(expected_prefix_history[-1]) - float(expected_prefix_history[-2]),
    )
    trajectory_estimate = current_gain
    if len(expected_prefix_history) >= 3:
        previous_gain = max(
            0.0,
            float(expected_prefix_history[-2]) - float(expected_prefix_history[-3]),
        )
        if previous_gain <= 0.0:
            trajectory_estimate = 0.0
        else:
            decay = min(1.0, current_gain / previous_gain)
            trajectory_estimate = current_gain * decay
    if bucket_estimate is None:
        return trajectory_estimate
    weight = min(1.0, max(0.0, float(bucket_weight)))
    return (
        weight * max(0.0, float(bucket_estimate))
        + (1.0 - weight) * trajectory_estimate
    )


def compare_renewal_costs(
    *,
    elapsed_draft_ms: float,
    next_draft_ms: float,
    verify_round_ms: float,
    controller_ms: float,
    expected_prefix: float,
    predicted_gain: float | None,
    hysteresis: float = 0.0,
) -> RenewalDecision:
    expected_output = 1.0 + max(0.0, float(expected_prefix))
    common_ms = (
        max(0.0, float(elapsed_draft_ms))
        + max(0.0, float(verify_round_ms))
        + max(0.0, float(controller_ms))
    )
    stop_cost = common_ms / expected_output
    if predicted_gain is None:
        return RenewalDecision(stop_cost, None, None, True)

    gain = max(0.0, float(predicted_gain))
    continue_cost = (
        common_ms + max(0.0, float(next_draft_ms))
    ) / (expected_output + gain)
    should_continue = continue_cost < stop_cost * (1.0 - float(hysteresis))
    return RenewalDecision(stop_cost, continue_cost, gain, should_continue)
