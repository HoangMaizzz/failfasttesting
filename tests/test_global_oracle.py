import unittest

from global_oracle import (
    CONTINUE,
    STOP,
    OracleBranchRequired,
    ScriptedOracleRefinementController,
    analyze_stop_depth_curves,
    solve_canonical_oracle_graph,
    summarize_policy_path,
)


def edge(index, state, child, step, latency, emitted, passes=None):
    return {
        "candidate_index": index,
        "state": state,
        "child_state": child,
        "step": step,
        "draft_passes": step if passes is None else passes,
        "edge_latency_ms": latency,
        "draft_latency_ms": latency * 0.4,
        "verify_latency_ms": latency * 0.5,
        "post_verify_latency_ms": latency * 0.1,
        "emitted_len": emitted,
        "accepted_len": max(0, emitted - 1),
        "proposal_len": 8,
    }


class GlobalOracleTests(unittest.TestCase):
    def test_scripted_controller_requests_and_replays_both_actions(self):
        controller = ScriptedOracleRefinementController()
        features = controller.build_features(
            proposal_length=8,
            remaining_masks=4,
            newly_unmasked=2,
            recoverable_confidences=[0.6, 0.7],
            recoverable_margins=[0.1, 0.2],
            first_remaining_position=4,
            frontier_length=4,
            proposal_change_ratio=0.0,
            recoverable_change_ratio=0.0,
            refinement_step=1,
        )
        controller.set_script(())
        with self.assertRaises(OracleBranchRequired):
            controller.choose(features, allow_stop=True, refinement_step=1)
        for action in (STOP, CONTINUE):
            controller.set_script((action,))
            decision = controller.choose(
                features, allow_stop=True, refinement_step=1
            )
            self.assertEqual(decision.action, action)

    def test_global_solver_finds_delayed_benefit_reversal(self):
        graph = {
            0: [
                edge(0, 0, 1, 1, 10.0, 1),
                edge(1, 0, 2, 2, 25.0, 2),
            ],
            1: [edge(0, 1, 3, 1, 100.0, 2)],
            2: [edge(0, 2, 3, 1, 10.0, 1)],
        }
        solved = solve_canonical_oracle_graph(graph, terminal_state=3)
        self.assertEqual(solved["policies"]["local"][0]["step"], 1)
        self.assertEqual(solved["policies"]["global"][0]["step"], 2)
        self.assertEqual(solved["values"]["global"][0], 35.0)
        self.assertEqual(solved["node_rows"][0]["delayed_benefit_reversal"], 1)

    def test_global_oracle_never_loses_to_failfast_fallback(self):
        graph = {
            0: [
                edge(0, 0, 1, 1, 12.0, 1),
                edge(1, 0, 2, 2, 30.0, 2),
            ],
            1: [edge(0, 1, 2, 1, 8.0, 1)],
        }
        solved = solve_canonical_oracle_graph(graph, terminal_state=2)
        self.assertLessEqual(
            solved["values"]["global"][0],
            solved["values"]["failfast"][0],
        )

    def test_policy_summary_aggregates_measured_components(self):
        path = [
            edge(0, 0, 1, 1, 10.0, 1),
            edge(0, 1, 3, 1, 20.0, 2),
        ]
        summary = summarize_policy_path(path)
        self.assertEqual(summary["rounds"], 2)
        self.assertEqual(summary["draft_passes"], 2)
        self.assertEqual(summary["total_latency_ms"], 30.0)
        self.assertEqual(summary["accepted_tokens"], 1)

    def test_graph_rejects_non_progressing_edge(self):
        graph = {0: [edge(0, 0, 0, 1, 10.0, 1)]}
        with self.assertRaisesRegex(ValueError, "invalid edge"):
            solve_canonical_oracle_graph(graph, terminal_state=1)

    def test_dp_matches_exhaustive_policy_enumeration(self):
        graph = {
            0: [
                edge(0, 0, 1, 1, 9.0, 1),
                edge(1, 0, 2, 2, 14.0, 2),
            ],
            1: [
                edge(0, 1, 3, 1, 30.0, 2),
                edge(1, 1, 2, 2, 8.0, 1),
            ],
            2: [edge(0, 2, 3, 1, 6.0, 1)],
        }
        solved = solve_canonical_oracle_graph(graph, terminal_state=3)

        def enumerate_cost(state):
            if state == 3:
                return [0.0]
            return [
                candidate["edge_latency_ms"] + suffix
                for candidate in graph[state]
                for suffix in enumerate_cost(candidate["child_state"])
            ]

        self.assertEqual(solved["values"]["global"][0], min(enumerate_cost(0)))

    def test_outer_extend_and_future_verifier_saving_can_win(self):
        shallow_verify = edge(0, 0, 1, 1, 5.0, 1)
        shallow_verify.update({"verifier_calls": 1, "outer_action": "verify"})
        deep_extend = edge(1, 0, 2, 3, 12.0, 2)
        deep_extend.update({"verifier_calls": 0, "outer_action": "extend"})
        graph = {
            0: [shallow_verify, deep_extend],
            1: [edge(0, 1, 3, 1, 80.0, 2)],
            2: [edge(0, 2, 3, 1, 8.0, 1)],
        }
        solved = solve_canonical_oracle_graph(graph, terminal_state=3)
        self.assertEqual(solved["policies"]["global"][0]["outer_action"], "extend")
        self.assertEqual(solved["values"]["global"][0], 20.0)

    def test_delayed_benefit_curve_escapes_non_monotonic_steps(self):
        rows = []
        for step, cost in enumerate((100.0, 105.0, 108.0, 80.0), start=1):
            rows.append({
                "prefix_len": 0,
                "block_key": "",
                "refinement_step": step,
                "stop_global_cost_ms": cost,
                "stop_draft_passes": step,
                "stop_future_verifier_calls": 2,
                "stop_accepted_tokens": step,
                "stop_emitted_tokens": step + 1,
                "outer_action_if_stop": "verify",
            })
        annotated, events, patience = analyze_stop_depth_curves(rows, 1.0)
        self.assertEqual(
            next(row for row in annotated if row["is_global_best_for_block"])[
                "refinement_step"
            ],
            4,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gap_steps"], 2)
        failures = {row["patience"]: row["would_fail"] for row in patience}
        self.assertEqual(failures, {1: 1, 2: 1, 3: 0})


if __name__ == "__main__":
    unittest.main()
