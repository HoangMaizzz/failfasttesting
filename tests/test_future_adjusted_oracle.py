import tempfile
import unittest
from pathlib import Path

import pandas as pd

from future_oracle import (
    adjusted_candidate_cost,
    load_future_cost_profile,
    select_greedy_future_adjusted_candidate,
)
from run_math_future_adjusted_oracle import (
    build_future_cost_profile,
    posthoc_policy_upper_bound,
)


def candidate(index, step, cost_ms, emitted):
    return {
        "candidate_index": index,
        "step": step,
        "target_len": 8,
        "draft_passes_elapsed": step,
        "counterfactual_total_latency_ms": cost_ms,
        "emitted_len_if_stop": emitted,
        "selected": False,
    }


def stats():
    return {
        "tokens_per_round": 5.0,
        "draft_ms_per_round": 40.0,
        "verify_ms_per_round": 75.0,
        "post_verify_ms_per_round": 5.0,
    }


class FutureAdjustedOracleTests(unittest.TestCase):
    def test_future_round_penalty_can_reverse_local_stop(self):
        rows = [
            candidate(0, 1, 100.0, 4),
            candidate(1, 2, 145.0, 7),
        ]
        selected, trace = select_greedy_future_adjusted_candidate(rows, stats())
        self.assertEqual(selected["candidate_index"], 1)
        self.assertEqual(trace[0]["action"], "continue")
        self.assertAlmostEqual(
            trace[0]["stop_expected_extra_verifier_rounds"],
            0.6,
        )
        self.assertAlmostEqual(trace[0]["stop_future_verify_penalty_ms"], 45.0)
        self.assertAlmostEqual(trace[0]["stop_adjusted_total_latency_ms"], 172.0)

    def test_greedy_selector_stops_when_extra_pass_is_not_worthwhile(self):
        rows = [
            candidate(0, 1, 100.0, 6),
            candidate(1, 2, 150.0, 7),
            candidate(2, 3, 170.0, 8),
        ]
        selected, trace = select_greedy_future_adjusted_candidate(rows, stats())
        self.assertEqual(selected["candidate_index"], 0)
        self.assertEqual(trace[0]["action"], "stop")
        self.assertEqual(rows[1]["oracle_action"], "not_reached")

    def test_profile_uses_measured_failfast_round_costs(self):
        baseline = pd.DataFrame({
            "problem_id": [1, 2],
            "output_tokens": [20, 30],
            "num_speculation_rounds": [4, 5],
            "actual_draft_time": [0.4, 0.5],
            "actual_verify_time": [0.8, 1.0],
            "actual_post_verify_time": [0.04, 0.05],
        })
        profile = build_future_cost_profile(baseline)
        self.assertAlmostEqual(profile["per_problem"]["1"]["tokens_per_round"], 5.0)
        self.assertAlmostEqual(profile["per_problem"]["1"]["verify_ms_per_round"], 200.0)
        self.assertAlmostEqual(profile["global"]["tokens_per_round"], 50.0 / 9.0)

    def test_profile_loader_validates_serialized_profile(self):
        baseline = pd.DataFrame({
            "problem_id": [1],
            "output_tokens": [20],
            "num_speculation_rounds": [4],
            "actual_draft_time": [0.4],
            "actual_verify_time": [0.8],
            "actual_post_verify_time": [0.04],
        })
        profile = build_future_cost_profile(baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            import json

            path.write_text(json.dumps(profile), encoding="utf-8")
            loaded = load_future_cost_profile(path)
        cost = adjusted_candidate_cost(candidate(0, 1, 100.0, 4), 7, loaded["global"])
        self.assertGreater(cost["future_verify_penalty_ms"], 0.0)

    def test_posthoc_upper_bound_selects_the_faster_executed_policy(self):
        baseline = pd.DataFrame({
            "problem_id": [1, 2],
            "actual_algorithm_time": [1.0, 2.0],
            "output_tokens": [10, 20],
        })
        oracle = pd.DataFrame({
            "problem_id": [1, 2],
            "actual_algorithm_time": [0.8, 2.4],
            "output_tokens": [10, 20],
        })
        summary, paired = posthoc_policy_upper_bound(baseline, oracle)
        self.assertEqual(summary.iloc[0].oracle_selected_questions, 1)
        self.assertEqual(list(paired.selected_policy), [
            "future_adjusted_oracle",
            "failfast",
        ])
        self.assertGreater(summary.iloc[0].posthoc_best_speedup_vs_failfast, 1.0)


if __name__ == "__main__":
    unittest.main()
