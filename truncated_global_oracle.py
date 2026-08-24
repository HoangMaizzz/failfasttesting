import math
import sys
from statistics import median


def greedy_lcp_verification(
    target_tokens,
    prefix_len,
    draft_proposal,
    *,
    max_append_tokens,
    eos_token_id,
):
    prefix_len = int(prefix_len)
    remaining = list(target_tokens[prefix_len:])
    accepted_len = 0
    for draft_token, target_token in zip(draft_proposal, remaining):
        if int(draft_token) != int(target_token):
            break
        accepted_len += 1
    if accepted_len >= len(remaining):
        tokens_to_append = remaining[:max_append_tokens]
        final_token = tokens_to_append[-1] if tokens_to_append else eos_token_id
    else:
        final_token = int(remaining[accepted_len])
        tokens_to_append = [
            int(token_id) for token_id in draft_proposal[:accepted_len]
        ] + [final_token]
        tokens_to_append = tokens_to_append[:max_append_tokens]
    if eos_token_id in tokens_to_append:
        tokens_to_append = tokens_to_append[
            :tokens_to_append.index(eos_token_id) + 1
        ]
    return {
        "accepted_len": min(accepted_len, len(tokens_to_append)),
        "emitted_len": len(tokens_to_append),
        "tokens_to_append": tokens_to_append,
        "final_token": final_token,
    }


class VerifierLatencyProfile:
    def __init__(self, context_bucket_size=256, proposal_bucket_size=8):
        self.context_bucket_size = int(context_bucket_size)
        self.proposal_bucket_size = int(proposal_bucket_size)
        self.observations = []

    def add(self, context_len, proposal_len, accepted_len, latency_ms):
        self.observations.append({
            "context_bucket": int(context_len) // self.context_bucket_size,
            "proposal_bucket": max(
                1,
                math.ceil(int(proposal_len) / self.proposal_bucket_size),
            ),
            "accepted_bucket": max(
                0,
                math.ceil(int(accepted_len) / self.proposal_bucket_size),
            ),
            "latency_ms": max(0.0, float(latency_ms)),
        })

    def estimate(self, context_len, proposal_len, accepted_len):
        if not self.observations:
            raise RuntimeError("verifier latency profile has no observations")
        target = (
            int(context_len) // self.context_bucket_size,
            max(1, math.ceil(int(proposal_len) / self.proposal_bucket_size)),
            max(0, math.ceil(int(accepted_len) / self.proposal_bucket_size)),
        )
        ranked = []
        for row in self.observations:
            distance = (
                4 * abs(row["context_bucket"] - target[0])
                + 2 * abs(row["proposal_bucket"] - target[1])
                + abs(row["accepted_bucket"] - target[2])
            )
            ranked.append((distance, row["latency_ms"]))
        minimum = min(item[0] for item in ranked)
        nearest = [latency for distance, latency in ranked if distance == minimum]
        return float(median(nearest))

    def summary(self):
        latencies = [row["latency_ms"] for row in self.observations]
        return {
            "observations": len(latencies),
            "median_latency_ms": float(median(latencies)) if latencies else math.nan,
            "min_latency_ms": min(latencies) if latencies else math.nan,
            "max_latency_ms": max(latencies) if latencies else math.nan,
        }


def estimate_cache_bytes(*objects):
    seen = set()

    def size(value):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(size(key) + size(item) for key, item in value.items())
        elif isinstance(value, (list, tuple, set, frozenset)):
            total += sum(size(item) for item in value)
        return total

    return sum(size(value) for value in objects)


def solve_truncated_horizon(initial_state, horizon, expand, baseline_value):
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    memo = {}
    policy = {}
    stats = {"calls": 0, "hits": 0}

    def value(state, remaining_horizon):
        stats["calls"] += 1
        key = (state, int(remaining_horizon))
        if key in memo:
            stats["hits"] += 1
            return memo[key]
        candidates = expand(state)
        if not candidates:
            memo[key] = 0.0
            return 0.0
        scored = []
        for edge in candidates:
            if edge.get("terminal"):
                suffix = 0.0
            elif remaining_horizon == 1:
                suffix = float(baseline_value(edge["child_state"]))
            else:
                suffix = value(edge["child_state"], remaining_horizon - 1)
            scored.append((float(edge["edge_latency_ms"]) + suffix, edge))
        cost, selected = min(
            scored,
            key=lambda item: (
                item[0],
                int(item[1].get("draft_passes", 0)),
                int(item[1].get("proposal_len", 0)),
            ),
        )
        memo[key] = cost
        policy[key] = selected
        return cost

    total = value(initial_state, horizon)
    path = []
    state = initial_state
    remaining_horizon = horizon
    while (state, remaining_horizon) in policy:
        edge = policy[(state, remaining_horizon)]
        path.append(edge)
        if edge.get("terminal"):
            break
        state = edge["child_state"]
        remaining_horizon -= 1
        if remaining_horizon == 0:
            break
    return {
        "cost_ms": total,
        "path": path,
        "policy": policy,
        "memo": memo,
        "memo_calls": stats["calls"],
        "memo_hits": stats["hits"],
        "memo_hit_rate": stats["hits"] / max(1, stats["calls"]),
    }
