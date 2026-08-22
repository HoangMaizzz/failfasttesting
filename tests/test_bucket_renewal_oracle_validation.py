import unittest

import pandas as pd

from run_bucket_renewal_oracle_validation import (
    build_oracle_transitions,
    summarize_transitions,
)


class BucketRenewalOracleValidationTests(unittest.TestCase):
    def test_transition_uses_measured_stop_and_continue_outcomes(self):
        snapshots = pd.DataFrame({
            "dataset": ["gsm8k", "gsm8k"],
            "problem_id": [1, 1],
            "round_id": [0, 0],
            "target_len": [8, 8],
            "step": [1, 2],
            "draft_passes_elapsed": [1, 2],
            "draft_latency_elapsed_ms": [10.0, 18.0],
            "actual_verify_latency_ms": [20.0, 22.0],
            "actual_post_verify_latency_ms": [1.0, 1.0],
            "emitted_len_if_stop": [3, 6],
            "predicted_expected_output": [4.0, 6.0],
            "predicted_next_gain": [2.0, None],
            "predicted_stop_ms_per_output": [8.0, 7.0],
            "predicted_continue_ms_per_output": [7.0, None],
            "predicted_should_continue": [True, None],
            "predicted_gain_source": ["step", "step"],
            "gain_bucket_count": [8, 8],
            "gain_bucket_weight": [0.5, 0.5],
            "calibration_tokens": [100, 100],
        })

        transition = build_oracle_transitions(snapshots).iloc[0]

        self.assertEqual(transition["actual_next_gain"], 3.0)
        self.assertAlmostEqual(transition["actual_stop_ms_per_output"], 31.0 / 3.0)
        self.assertAlmostEqual(transition["actual_continue_ms_per_output"], 41.0 / 6.0)
        self.assertEqual(transition["oracle_action"], "continue")
        self.assertEqual(transition["decision_correct"], 1)

    def test_transition_rejects_non_adjacent_forward_passes(self):
        snapshots = pd.DataFrame({
            "dataset": ["math", "math"],
            "problem_id": [1, 1],
            "round_id": [0, 0],
            "target_len": [8, 8],
            "step": [1, 2],
            "draft_passes_elapsed": [1, 3],
        })
        self.assertTrue(build_oracle_transitions(snapshots).empty)

    def test_summary_reports_gain_cost_and_decision_errors(self):
        transitions = pd.DataFrame({
            "dataset": ["aime", "aime"],
            "gain_error": [1.0, -1.0],
            "predicted_next_gain": [2.0, 0.0],
            "actual_next_gain": [1.0, 1.0],
            "predicted_current_output": [3.0, 3.0],
            "actual_current_output": [3.0, 3.0],
            "predicted_next_output": [5.0, 3.0],
            "actual_next_output": [4.0, 4.0],
            "stop_cost_error": [1.0, -1.0],
            "continue_cost_error": [2.0, -2.0],
            "predicted_action": ["continue", "stop"],
            "oracle_action": ["continue", "continue"],
            "decision_correct": [1, 0],
            "decision_regret_ms_per_output": [0.0, 2.0],
        })

        summary = summarize_transitions(transitions, ["dataset"]).iloc[0]

        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(summary["gain_mae"], 1.0)
        self.assertEqual(summary["decision_accuracy_percent"], 50.0)
        self.assertEqual(summary["continue_false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
