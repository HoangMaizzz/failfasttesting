import unittest

import pandas as pd

from run_cost_aware_v2_benchmark import (
    aggregate_method,
    build_threshold_comparison,
)


class CostAwareV2BenchmarkTests(unittest.TestCase):
    def test_aggregate_method_reports_measured_components(self):
        frame = pd.DataFrame({
            "output_tokens": [10],
            "actual_draft_time": [0.2],
            "actual_verify_time": [0.3],
            "actual_post_verify_time": [0.01],
            "theo_total_time": [0.4],
            "drafted_tokens": [8],
            "accepted_tokens": [4],
            "num_speculation_rounds": [2],
            "total_num_forward_passes": [3],
            "frontier_v2_extend_actions": [1],
            "frontier_v2_verify_actions": [1],
            "frontier_v2_refinement_stop_actions": [1],
            "frontier_v2_extension_stop_actions": [0],
            "frontier_v2_fallback_steps": [2],
            "frontier_v2_hazard_ready_steps": [3],
            "frontier_v2_extension_history_ready_steps": [1],
            "frontier_v2_predicted_extension_gain_mean": [2.5],
            "frontier_fill_forward_passes": [1],
            "frontier_denoising_forward_passes": [2],
            "frontier_expected_output_mean": [5.0],
            "is_correct": [1],
        })

        result = aggregate_method(frame)

        self.assertAlmostEqual(result["actual_measured_time_s"], 0.51)
        self.assertAlmostEqual(result["actual_measured_ms_per_output_token"], 51.0)
        self.assertAlmostEqual(result["output_tokens_per_second"], 10.0 / 0.51)
        self.assertAlmostEqual(result["frontier_v2_refinement_stops_per_100_rounds"], 50.0)

    def test_threshold_comparison_pairs_identical_problem_ids(self):
        common = {
            "dataset": "gsm8k",
            "problem_id": 0,
            "actual_draft_time": 0.2,
            "actual_verify_time": 0.3,
            "actual_post_verify_time": 0.01,
            "output_tokens": 10,
            "num_speculation_rounds": 2,
            "total_num_forward_passes": 3,
            "accepted_tokens": 4,
            "drafted_tokens": 8,
            "output_token_hash": "same",
        }
        rows = pd.DataFrame([
            {
                **common,
                "method": "cost_aware_v2_lowconf_0p45",
                "actual_measured_time": 0.51,
                "actual_measured_ms_per_output_token": 51.0,
            },
            {
                **common,
                "method": "cost_aware_v2_lowconf_0p60",
                "actual_measured_time": 0.45,
                "actual_measured_ms_per_output_token": 45.0,
            },
        ])

        paired, summary = build_threshold_comparison(rows)

        self.assertEqual(len(paired), 1)
        self.assertTrue(bool(paired.iloc[0]["lowconf_0p60_faster"]))
        self.assertAlmostEqual(summary.iloc[0]["pooled_speedup_0p60_vs_0p45"], 51.0 / 45.0)
        self.assertEqual(summary.iloc[0]["output_match_rate_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
