from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


STOP = "stop"
CONTINUE = "continue"


@dataclass(frozen=True)
class BoundaryLeaf:
    script: tuple[str, ...]
    emitted_tokens: int
    draft_passes: int
    predicted_verify_ms: float
    measured_verify_ms: float
    proposal_length: int
    proposal_hash: str


def leaf_profile_cost_ms(
    leaf: BoundaryLeaf,
    *,
    mean_draft_forward_ms: float,
    mean_post_verify_ms: float,
) -> float:
    return (
        max(0, int(leaf.draft_passes)) * float(mean_draft_forward_ms)
        + float(leaf.predicted_verify_ms)
        + float(mean_post_verify_ms)
    )


def leaf_profile_utility(
    leaf: BoundaryLeaf,
    *,
    rho_tokens_per_ms: float,
    mean_draft_forward_ms: float,
    mean_post_verify_ms: float,
) -> float:
    return float(leaf.emitted_tokens) - float(rho_tokens_per_ms) * (
        leaf_profile_cost_ms(
            leaf,
            mean_draft_forward_ms=mean_draft_forward_ms,
            mean_post_verify_ms=mean_post_verify_ms,
        )
    )


def _best_descendant(
    leaves: Sequence[BoundaryLeaf],
    prefix: tuple[str, ...],
    utilities: Mapping[tuple[str, ...], float],
) -> BoundaryLeaf | None:
    candidates = [
        leaf for leaf in leaves if leaf.script[: len(prefix)] == prefix
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda leaf: (
            float(utilities[leaf.script]),
            -len(leaf.script),
            tuple(leaf.script),
        ),
    )


def solve_exact_boundary_tree(
    node_prefixes: Iterable[tuple[str, ...]],
    leaves: Sequence[BoundaryLeaf],
    *,
    rho_tokens_per_ms: float,
    mean_draft_forward_ms: float,
    mean_post_verify_ms: float,
    epsilon_ms: float,
) -> list[dict]:
    """Solve every binary decision in one verifier-boundary tree.

    Leaf utility is measured from the round root.  The cost before a descendant
    node is shared by both actions at that node, so it cancels in the advantage.
    This lets the solver use stable profile cost without reconstructing a noisy
    per-edge wall-clock ledger.
    """
    rho = max(0.0, float(rho_tokens_per_ms))
    epsilon_tokens = rho * max(0.0, float(epsilon_ms))
    leaves = list(leaves)
    utilities = {
        leaf.script: leaf_profile_utility(
            leaf,
            rho_tokens_per_ms=rho,
            mean_draft_forward_ms=mean_draft_forward_ms,
            mean_post_verify_ms=mean_post_verify_ms,
        )
        for leaf in leaves
    }
    rows = []
    for prefix in sorted(set(node_prefixes), key=lambda value: (len(value), value)):
        stop_leaf = _best_descendant(leaves, prefix + (STOP,), utilities)
        continue_leaf = _best_descendant(
            leaves, prefix + (CONTINUE,), utilities
        )
        if stop_leaf is None or continue_leaf is None:
            raise RuntimeError(
                f"incomplete exact boundary tree at action prefix {prefix}"
            )
        q_stop = float(utilities[stop_leaf.script])
        q_continue = float(utilities[continue_leaf.script])
        advantage = q_stop - q_continue
        tie = abs(advantage) <= epsilon_tokens + 1e-12
        action = CONTINUE if tie else (STOP if advantage > 0.0 else CONTINUE)
        rows.append({
            "action_prefix": list(prefix),
            "decision_depth": len(prefix),
            "q_stop_profile_tokens": q_stop,
            "q_continue_profile_tokens": q_continue,
            "oracle_advantage_tokens": advantage,
            "oracle_action": action,
            "oracle_label": "tie" if tie else action,
            "tie": int(tie),
            "epsilon_tokens": epsilon_tokens,
            "best_stop_leaf_script": list(stop_leaf.script),
            "best_continue_leaf_script": list(continue_leaf.script),
            "best_stop_emitted_tokens": int(stop_leaf.emitted_tokens),
            "best_continue_emitted_tokens": int(continue_leaf.emitted_tokens),
            "best_stop_profile_cost_ms": leaf_profile_cost_ms(
                stop_leaf,
                mean_draft_forward_ms=mean_draft_forward_ms,
                mean_post_verify_ms=mean_post_verify_ms,
            ),
            "best_continue_profile_cost_ms": leaf_profile_cost_ms(
                continue_leaf,
                mean_draft_forward_ms=mean_draft_forward_ms,
                mean_post_verify_ms=mean_post_verify_ms,
            ),
        })
    return rows
