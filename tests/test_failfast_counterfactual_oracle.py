import unittest

import pandas as pd

from run_failfast_counterfactual_oracle import (
    add_latency_estimates,
    build_transitions,
    policy_summary,
    select_round_candidates,
)


class FailFastCounterfactualOracleTests(unittest.TestCase):
    def snapshots(self):
        return pd.DataFrame({
            "problem_id": [1, 1, 1],
            "round_id": [0, 0, 0],
            "context_len": [300, 300, 300],
            "target_len": [8, 8, 8],
            "step": [1, 2, 3],
            "draft_passes_elapsed": [1, 2, 3],
            "draft_latency_elapsed_ms": [10.0, 18.0, 26.0],
            "actual_verify_latency_ms": [20.0, 20.0, 20.0],
            "actual_post_verify_latency_ms": [1.0, 1.0, 1.0],
            "emitted_len_if_stop": [3, 6, 6],
            "adaptive_policy_action": ["continue", "stop", "stop"],
            "adaptive_policy_reason": ["continue", "stop", "stop"],
            "adaptive_stop_probability": [0.2, 0.8, 0.9],
            "adaptive_q_stop_mean": [1.0, 3.0, 3.0],
            "adaptive_q_continue_mean": [2.0, 2.0, 2.0],
            "adaptive_rho_tokens_per_ms": [0.1, 0.1, 0.1],
            "adaptive_stop_available": [True, True, True],
        })

    def test_transition_uses_exact_next_pass_and_frozen_policy_action(self):
        transitions = build_transitions(
            add_latency_estimates(self.snapshots()),
            fallback_rho=0.1,
        )
        first = transitions.iloc[0]
        second = transitions.iloc[1]

        self.assertEqual(first["actual_next_gain_tokens"], 3.0)
        self.assertEqual(first["next_draft_latency_ms"], 8.0)
        self.assertEqual(first["oracle_action"], "continue")
        self.assertEqual(first["decision_correct"], 1)
        self.assertEqual(second["oracle_action"], "stop")
        self.assertEqual(second["decision_correct"], 1)

    def test_policy_summary_reports_false_stops(self):
        transitions = pd.DataFrame({
            "predicted_action": ["stop", "stop", "continue"],
            "oracle_action": ["stop", "continue", "stop"],
            "decision_correct": [1, 0, 0],
            "regret_tokens": [0.0, 1.0, 2.0],
            "regret_ms_per_output_token": [0.0, 1.0, 2.0],
            "regret_ms_equivalent": [0.0, 10.0, 20.0],
        })
        summary = policy_summary(transitions).iloc[0]
        self.assertEqual(summary["predicted_stop_count"], 2)
        self.assertEqual(summary["false_stop_rate_percent"], 50.0)
        self.assertAlmostEqual(summary["decision_accuracy_percent"], 100.0 / 3.0)

    def test_round_oracle_selects_lowest_latency_per_emitted_token(self):
        rounds = select_round_candidates(add_latency_estimates(self.snapshots()))
        row = rounds.iloc[0]
        self.assertEqual(row["factual_step"], 3)
        self.assertEqual(row["policy_step"], 2)
        self.assertEqual(row["oracle_step"], 2)
        self.assertEqual(row["policy_matches_oracle_step"], 1)


if __name__ == "__main__":
    unittest.main()
