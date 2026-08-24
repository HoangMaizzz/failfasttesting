import json
import math
from pathlib import Path


PROFILE_FIELDS = (
    "tokens_per_round",
    "draft_ms_per_round",
    "verify_ms_per_round",
    "post_verify_ms_per_round",
)


def load_future_cost_profile(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "global" not in payload or "per_problem" not in payload:
        raise ValueError("future-cost profile must contain global and per_problem")
    _validate_stats(payload["global"])
    for stats in payload["per_problem"].values():
        _validate_stats(stats)
    return payload


def _validate_stats(stats):
    for field in PROFILE_FIELDS:
        value = float(stats[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid future-cost profile value: {field}={value}")
    if float(stats["tokens_per_round"]) <= 0.0:
        raise ValueError("tokens_per_round must be positive")


def stats_for_problem(profile, problem_id):
    return profile["per_problem"].get(str(int(problem_id)), profile["global"])


def adjusted_candidate_cost(row, reference_emitted_len, stats):
    emitted_len = max(0.0, float(row["emitted_len_if_stop"]))
    deficit = max(0.0, float(reference_emitted_len) - emitted_len)
    extra_rounds = deficit / float(stats["tokens_per_round"])
    draft_penalty = extra_rounds * float(stats["draft_ms_per_round"])
    verify_penalty = extra_rounds * float(stats["verify_ms_per_round"])
    post_penalty = extra_rounds * float(stats["post_verify_ms_per_round"])
    future_penalty = draft_penalty + verify_penalty + post_penalty
    return {
        "future_reference_emitted_len": float(reference_emitted_len),
        "future_token_deficit": deficit,
        "expected_extra_verifier_rounds": extra_rounds,
        "future_draft_penalty_ms": draft_penalty,
        "future_verify_penalty_ms": verify_penalty,
        "future_post_verify_penalty_ms": post_penalty,
        "future_round_penalty_ms": future_penalty,
        "adjusted_counterfactual_total_latency_ms": (
            float(row["counterfactual_total_latency_ms"]) + future_penalty
        ),
    }


def select_greedy_future_adjusted_candidate(rows, stats):
    if not rows:
        raise ValueError("at least one oracle candidate is required")
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item["draft_passes_elapsed"]),
            int(item["target_len"]),
            int(item["candidate_index"]),
        ),
    )
    global_reference = max(float(row["emitted_len_if_stop"]) for row in ordered)
    for row in ordered:
        row.update(adjusted_candidate_cost(row, global_reference, stats))
    trace = []
    selected = ordered[-1]
    for current, following in zip(ordered, ordered[1:]):
        pair_reference = max(
            float(current["emitted_len_if_stop"]),
            float(following["emitted_len_if_stop"]),
        )
        stop_cost = adjusted_candidate_cost(current, pair_reference, stats)
        continue_cost = adjusted_candidate_cost(following, pair_reference, stats)
        action = (
            "stop"
            if stop_cost["adjusted_counterfactual_total_latency_ms"]
            <= continue_cost["adjusted_counterfactual_total_latency_ms"]
            else "continue"
        )
        trace.append({
            "stop_candidate_index": int(current["candidate_index"]),
            "continue_candidate_index": int(following["candidate_index"]),
            "stop_step": int(current["step"]),
            "continue_step": int(following["step"]),
            "reference_emitted_len": pair_reference,
            "stop_emitted_len": int(current["emitted_len_if_stop"]),
            "continue_emitted_len": int(following["emitted_len_if_stop"]),
            "stop_expected_extra_verifier_rounds": stop_cost[
                "expected_extra_verifier_rounds"
            ],
            "continue_expected_extra_verifier_rounds": continue_cost[
                "expected_extra_verifier_rounds"
            ],
            "stop_future_verify_penalty_ms": stop_cost[
                "future_verify_penalty_ms"
            ],
            "continue_future_verify_penalty_ms": continue_cost[
                "future_verify_penalty_ms"
            ],
            "stop_future_round_penalty_ms": stop_cost[
                "future_round_penalty_ms"
            ],
            "continue_future_round_penalty_ms": continue_cost[
                "future_round_penalty_ms"
            ],
            "stop_adjusted_total_latency_ms": stop_cost[
                "adjusted_counterfactual_total_latency_ms"
            ],
            "continue_adjusted_total_latency_ms": continue_cost[
                "adjusted_counterfactual_total_latency_ms"
            ],
            "action": action,
        })
        if action == "stop":
            selected = current
            break
    selected_index = ordered.index(selected)
    for index, row in enumerate(ordered):
        if index < selected_index:
            row["oracle_action"] = "continue"
        elif index == selected_index:
            row["oracle_action"] = "stop"
        else:
            row["oracle_action"] = "not_reached"
        row["selected"] = index == selected_index
    return selected, trace
