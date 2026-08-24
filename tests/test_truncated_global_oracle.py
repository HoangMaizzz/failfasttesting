import unittest

from global_oracle import analyze_stop_depth_curves
from truncated_global_oracle import (
    VerifierLatencyProfile,
    greedy_lcp_verification,
    solve_truncated_horizon,
)


def edge(state, child, latency, *, terminal=False, label="", blocks=1):
    return {
        "state": state,
        "child_state": child,
        "edge_latency_ms": float(latency),
        "terminal": bool(terminal),
        "draft_passes": blocks,
        "proposal_len": 8 * blocks,
        "label": label,
        "blocks": blocks,
    }


class TruncatedGlobalOracleTests(unittest.TestCase):
    def test_lcp_rejection_emits_target_correction(self):
        result = greedy_lcp_verification(
            [10, 11, 12, 13],
            0,
            [10, 11, 99],
            max_append_tokens=4,
            eos_token_id=2,
        )
        self.assertEqual(result["accepted_len"], 2)
        self.assertEqual(result["tokens_to_append"], [10, 11, 12])
        self.assertEqual(result["final_token"], 12)

    def test_lcp_full_accept_emits_bonus_token(self):
        result = greedy_lcp_verification(
            [10, 11, 12, 2],
            0,
            [10, 11],
            max_append_tokens=3,
            eos_token_id=2,
        )
        self.assertEqual(result["accepted_len"], 2)
        self.assertEqual(result["tokens_to_append"], [10, 11, 12])

    def test_lcp_honors_eos_and_output_limit(self):
        result = greedy_lcp_verification(
            [10, 2, 12],
            0,
            [10, 2, 12],
            max_append_tokens=3,
            eos_token_id=2,
        )
        self.assertEqual(result["tokens_to_append"], [10, 2])
        self.assertEqual(result["accepted_len"], 2)

    def test_latency_profile_uses_nearest_bucket_median(self):
        profile = VerifierLatencyProfile()
        profile.add(100, 8, 4, 10.0)
        profile.add(110, 8, 5, 14.0)
        profile.add(700, 32, 16, 40.0)
        self.assertEqual(profile.estimate(120, 8, 6), 12.0)

    def test_horizon_decrements_once_per_verifier_edge(self):
        graph = {
            "root": [edge("root", "p1", 4.0, label="multi_block", blocks=3)],
            "p1": [edge("p1", "p2", 5.0, label="second_verify")],
        }
        baseline_calls = []

        def baseline(state):
            baseline_calls.append(state)
            return 7.0

        solved = solve_truncated_horizon(
            "root",
            2,
            lambda state: graph[state],
            baseline,
        )
        self.assertEqual(solved["cost_ms"], 16.0)
        self.assertEqual([item["label"] for item in solved["path"]], [
            "multi_block",
            "second_verify",
        ])
        self.assertEqual(baseline_calls, ["p2"])

    def test_memoization_reuses_converged_verifier_state(self):
        graph = {
            "root": [
                edge("root", "shared", 3.0, label="a"),
                edge("root", "shared", 4.0, label="b"),
            ],
            "shared": [edge("shared", "end", 2.0, terminal=True)],
        }
        solved = solve_truncated_horizon(
            "root",
            2,
            lambda state: graph[state],
            lambda state: 100.0,
        )
        self.assertEqual(solved["cost_ms"], 5.0)
        self.assertGreaterEqual(solved["memo_hits"], 1)
        self.assertEqual(len(solved["memo"]), 2)

    def test_horizon_one_attaches_baseline_after_first_verify(self):
        expanded = []

        def expand(state):
            expanded.append(state)
            return [edge(state, "tail", 3.0)]

        solved = solve_truncated_horizon(
            "root",
            1,
            expand,
            lambda state: 9.0,
        )
        self.assertEqual(solved["cost_ms"], 12.0)
        self.assertEqual(expanded, ["root"])

    def test_delayed_benefit_and_patience_two_are_preserved(self):
        rows = []
        for step, cost in enumerate((100.0, 105.0, 110.0, 80.0), start=1):
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
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["later_better_step"], 4)
        patience_two = next(row for row in patience if row["patience"] == 2)
        self.assertEqual(patience_two["would_fail"], 1)
        self.assertEqual(
            next(row for row in annotated if row["is_global_best_for_block"])[
                "refinement_step"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
