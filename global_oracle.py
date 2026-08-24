import math

from adaptive_td import (
    CONTINUE,
    STOP,
    ActionEstimate,
    AdaptiveDecision,
    AdaptiveTDConfig,
    OnlineTDRefinementController,
)


class OracleBranchRequired(RuntimeError):
    def __init__(self, decision_index, state):
        super().__init__(f"oracle action required at decision {decision_index}")
        self.decision_index = int(decision_index)
        self.state = dict(state)


class ScriptedOracleRefinementController(OnlineTDRefinementController):
    controller_name = "exact_global_oracle"

    def __init__(self, max_refinement_steps=64):
        super().__init__(
            AdaptiveTDConfig(
                max_refinement_steps=int(max_refinement_steps),
                warmup_rounds=0,
                early_stop_min_observations=0,
            )
        )
        self.action_script = ()
        self.script_position = 0

    def set_script(self, actions):
        unknown = set(actions).difference({STOP, CONTINUE})
        if unknown:
            raise ValueError(f"unknown oracle actions: {sorted(unknown)}")
        self.action_script = tuple(actions)
        self.script_position = 0

    def choose(
        self,
        features,
        *,
        allow_stop,
        refinement_step,
        **state,
    ):
        if not allow_stop:
            action = CONTINUE
            reason = "oracle_mandatory_continue"
        elif self.script_position >= len(self.action_script):
            raise OracleBranchRequired(
                self.script_position,
                {
                    "refinement_step": int(refinement_step),
                    "features": list(features),
                    **state,
                },
            )
        else:
            action = self.action_script[self.script_position]
            self.script_position += 1
            reason = "oracle_scripted_action"
        zero = ActionEstimate(0.0, 0.0, 0.0, 0.0)
        return AdaptiveDecision(
            action=action,
            reason=reason,
            stop=zero,
            continue_=zero,
            rho_tokens_per_ms=0.0,
            exploration_used=False,
            latency_ms=0.0,
            early_stop_observations=0,
            calibration_active=False,
            advantage_mean=0.0,
            advantage_risk=0.0,
            stop_probability=0.5,
            behavior_stop_probability=1.0 if action == STOP else 0.0,
            selected_action_probability=1.0,
            importance_weight=1.0,
            diagnostics={
                "controller_name": self.controller_name,
                "oracle_script_index": self.script_position - 1,
                "oracle_elapsed_draft_ms": float(
                    state.get("elapsed_draft_ms", 0.0)
                ),
            },
        )

    def resolve_pending_stop(self, *args, **kwargs):
        return None

    def observe_transition(self, *args, **kwargs):
        return None

    def observe_continue_transition(self, *args, **kwargs):
        return None

    def complete_trajectory(self, *args, **kwargs):
        return None

    def observe_round(self, *args, **kwargs):
        return None


def _edge_order(edge):
    return (
        int(edge["draft_passes"]),
        int(edge["step"]),
        int(edge["candidate_index"]),
    )


def _local_edge_order(edge):
    return (
        float(edge["edge_latency_ms"]) / max(1, int(edge["emitted_len"])),
        *_edge_order(edge),
    )


def solve_canonical_oracle_graph(edges_by_state, terminal_state):
    terminal_state = int(terminal_state)
    if terminal_state < 0:
        raise ValueError("terminal_state must be non-negative")
    reachable = {terminal_state}
    for state in sorted(edges_by_state, reverse=True):
        if state >= terminal_state:
            continue
        edges = edges_by_state[state]
        if not edges:
            raise ValueError(f"state {state} has no canonical edges")
        for edge in edges:
            child = int(edge["child_state"])
            if child <= state or child > terminal_state:
                raise ValueError(f"invalid edge {state}->{child}")
        reachable.add(int(state))
    if 0 not in reachable:
        raise ValueError("graph must contain root state 0")

    policies = {
        "global": {},
        "local": {},
        "failfast": {},
    }
    values = {
        name: {terminal_state: 0.0}
        for name in policies
    }
    for state in sorted((key for key in edges_by_state if key < terminal_state), reverse=True):
        edges = edges_by_state[state]
        missing = [
            int(edge["child_state"])
            for edge in edges
            if int(edge["child_state"]) not in values["global"]
        ]
        if missing:
            raise ValueError(
                f"state {state} references unsolved children: {sorted(set(missing))}"
            )
        global_edge = min(
            edges,
            key=lambda edge: (
                float(edge["edge_latency_ms"])
                + values["global"][int(edge["child_state"])],
                *_edge_order(edge),
            ),
        )
        local_edge = min(edges, key=_local_edge_order)
        failfast_candidates = [edge for edge in edges if edge.get("is_failfast")]
        if len(failfast_candidates) > 1:
            raise ValueError(
                f"state {state} has multiple FailFast replay edges"
            )
        failfast_edge = (
            failfast_candidates[0]
            if failfast_candidates
            else max(edges, key=_edge_order)
        )
        selected = {
            "global": global_edge,
            "local": local_edge,
            "failfast": failfast_edge,
        }
        for name, edge in selected.items():
            child = int(edge["child_state"])
            policies[name][state] = edge
            values[name][state] = (
                float(edge["edge_latency_ms"])
                + values[name][child]
            )

    paths = {
        name: reconstruct_policy_path(policy, terminal_state)
        for name, policy in policies.items()
    }
    node_rows = []
    for state in sorted(edges_by_state):
        global_edge = policies["global"][state]
        local_edge = policies["local"][state]
        failfast_edge = policies["failfast"][state]
        local_global_q = (
            float(local_edge["edge_latency_ms"])
            + values["global"][int(local_edge["child_state"])]
        )
        global_q = values["global"][state]
        delayed_benefit = (
            _edge_order(global_edge) > _edge_order(local_edge)
            and _local_edge_order(global_edge)[0] >= _local_edge_order(local_edge)[0]
            and global_q < local_global_q
        )
        node_rows.append({
            "state": int(state),
            "num_candidates": len(edges_by_state[state]),
            "global_step": int(global_edge["step"]),
            "local_step": int(local_edge["step"]),
            "failfast_step": int(failfast_edge["step"]),
            "global_child_state": int(global_edge["child_state"]),
            "local_child_state": int(local_edge["child_state"]),
            "failfast_child_state": int(failfast_edge["child_state"]),
            "global_immediate_ms_per_token": _local_edge_order(global_edge)[0],
            "local_immediate_ms_per_token": _local_edge_order(local_edge)[0],
            "global_total_to_terminal_ms": global_q,
            "local_choice_global_total_ms": local_global_q,
            "failfast_total_to_terminal_ms": values["failfast"][state],
            "global_continues_beyond_local": int(
                _edge_order(global_edge) > _edge_order(local_edge)
            ),
            "delayed_benefit_reversal": int(delayed_benefit),
            "global_savings_vs_local_choice_ms": max(
                0.0,
                local_global_q - global_q,
            ),
        })
    if values["global"][0] > values["failfast"][0] + 1e-6:
        raise RuntimeError("global oracle is slower than its FailFast fallback")
    return {
        "values": values,
        "policies": policies,
        "paths": paths,
        "node_rows": node_rows,
    }


def reconstruct_policy_path(policy, terminal_state):
    state = 0
    path = []
    visited = set()
    while state != terminal_state:
        if state in visited:
            raise RuntimeError("policy contains a cycle")
        visited.add(state)
        if state not in policy:
            raise ValueError(f"policy has no action for state {state}")
        edge = policy[state]
        path.append(edge)
        state = int(edge["child_state"])
    return path


def summarize_policy_path(path):
    total_latency_ms = sum(float(edge["edge_latency_ms"]) for edge in path)
    return {
        "rounds": len(path),
        "draft_passes": sum(int(edge["draft_passes"]) for edge in path),
        "draft_latency_ms": sum(float(edge["draft_latency_ms"]) for edge in path),
        "verify_latency_ms": sum(float(edge["verify_latency_ms"]) for edge in path),
        "post_verify_latency_ms": sum(
            float(edge["post_verify_latency_ms"])
            for edge in path
        ),
        "total_latency_ms": total_latency_ms,
        "accepted_tokens": sum(int(edge["accepted_len"]) for edge in path),
        "drafted_tokens": sum(int(edge["proposal_len"]) for edge in path),
        "mean_step": (
            sum(int(edge["step"]) for edge in path) / len(path)
            if path
            else math.nan
        ),
    }


def analyze_stop_depth_curves(rows, epsilon_cost_ms=1.0):
    epsilon = max(0.0, float(epsilon_cost_ms))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["prefix_len"], row["block_key"]), []).append(row)
    annotated = []
    events = []
    patience_records = []
    event_id = 0
    for (prefix_len, block_key), group in grouped.items():
        ordered = sorted(group, key=lambda row: row["refinement_step"])
        costs = [float(row["stop_global_cost_ms"]) for row in ordered]
        best_index = min(range(len(ordered)), key=lambda index: costs[index])
        patience_failures = {}
        patience_stops = {}
        for patience in (1, 2, 3):
            running_best = 0
            non_improving = 0
            stopped = None
            for index in range(1, len(ordered)):
                if costs[index] < costs[running_best] - epsilon:
                    running_best = index
                    non_improving = 0
                else:
                    non_improving += 1
                if non_improving >= patience:
                    stopped = running_best
                    break
            if stopped is None:
                stopped = best_index
            patience_stops[patience] = stopped
            patience_failures[patience] = stopped != best_index
            patience_records.append({
                "prefix_len": int(prefix_len),
                "block_key": block_key,
                "patience": patience,
                "patience_stop_step": int(ordered[stopped]["refinement_step"]),
                "global_best_step": int(ordered[best_index]["refinement_step"]),
                "would_fail": int(stopped != best_index),
                "extra_latency_ms": max(0.0, costs[stopped] - costs[best_index]),
                "missed_depth": int(
                    ordered[best_index]["refinement_step"]
                    - ordered[stopped]["refinement_step"]
                ),
            })

        running_best = 0
        non_improving = 0
        delayed_event_by_later_index = {}
        for index in range(1, len(ordered)):
            if costs[index] < costs[running_best] - epsilon:
                if non_improving:
                    event_id += 1
                    early = ordered[running_best]
                    late = ordered[index]
                    event = {
                        "delayed_benefit_event_id": event_id,
                        "prefix_len": int(prefix_len),
                        "block_key": block_key,
                        "early_best_step": int(early["refinement_step"]),
                        "later_better_step": int(late["refinement_step"]),
                        "gap_steps": int(non_improving),
                        "early_global_cost_ms": costs[running_best],
                        "later_global_cost_ms": costs[index],
                        "improvement_ms": costs[running_best] - costs[index],
                        "improvement_percent": (
                            100.0 * (costs[running_best] - costs[index])
                            / max(costs[running_best], 1e-9)
                        ),
                        "additional_dllm_forwards": max(
                            0,
                            int(late.get("stop_draft_passes", 0))
                            - int(early.get("stop_draft_passes", 0)),
                        ),
                        "verifier_calls_early": int(
                            early.get("stop_future_verifier_calls", 0)
                        ),
                        "verifier_calls_late": int(
                            late.get("stop_future_verifier_calls", 0)
                        ),
                        "verifier_calls_saved": int(
                            early.get("stop_future_verifier_calls", 0)
                            - late.get("stop_future_verifier_calls", 0)
                        ),
                        "accepted_early": int(early.get("stop_accepted_tokens", 0)),
                        "accepted_late": int(late.get("stop_accepted_tokens", 0)),
                        "emitted_early": int(early.get("stop_emitted_tokens", 0)),
                        "emitted_late": int(late.get("stop_emitted_tokens", 0)),
                        "early_outer_action": early.get("outer_action_if_stop"),
                        "late_outer_action": late.get("outer_action_if_stop"),
                        "patience1_would_fail": int(patience_failures[1]),
                        "patience2_would_fail": int(patience_failures[2]),
                        "patience3_would_fail": int(patience_failures[3]),
                    }
                    if (
                        event["early_outer_action"] == "verify"
                        and event["late_outer_action"] == "extend"
                    ):
                        event["delayed_benefit_type"] = "outer_verify_to_extend"
                    elif event["verifier_calls_saved"] > 0:
                        event["delayed_benefit_type"] = "future_verifier_reduction"
                    elif event["accepted_late"] > event["accepted_early"]:
                        event["delayed_benefit_type"] = "acceptance_improvement"
                    else:
                        event["delayed_benefit_type"] = "combined_or_other"
                    events.append(event)
                    delayed_event_by_later_index[index] = event_id
                running_best = index
                non_improving = 0
            else:
                non_improving += 1

        for index, row in enumerate(ordered):
            local_minimum = (
                0 < index < len(ordered) - 1
                and costs[index] < costs[index - 1]
                and costs[index] < costs[index + 1]
            )
            annotated.append({
                **row,
                "is_local_minimum": int(local_minimum),
                "is_global_best_for_block": int(index == best_index),
                "delayed_benefit_event_id": delayed_event_by_later_index.get(index),
                "patience1_failure": int(patience_failures[1]),
                "patience2_failure": int(patience_failures[2]),
                "patience3_failure": int(patience_failures[3]),
            })
    return annotated, events, patience_records
