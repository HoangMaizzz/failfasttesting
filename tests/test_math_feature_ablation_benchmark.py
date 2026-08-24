import unittest

import pandas as pd

from adaptive_td import FEATURE_NAMES
from run_math_feature_ablation_benchmark import (
    FEATURE_GROUPS,
    aggregate_method,
    paired_ablation_summary,
)


def result_frame(scale=1.0):
    return pd.DataFrame({
        "problem_id": [1, 2],
        "output_tokens": [10, 20],
        "drafted_tokens": [20, 40],
        "accepted_tokens": [8, 16],
        "actual_algorithm_time": [1.0 * scale, 2.0 * scale],
        "actual_draft_time": [0.4 * scale, 0.8 * scale],
        "actual_verify_time": [0.5 * scale, 1.0 * scale],
        "actual_post_verify_time": [0.1 * scale, 0.2 * scale],
        "total_num_forward_passes": [4, 8],
        "num_speculation_rounds": [2, 4],
        "adaptive_decisions": [3, 6],
        "adaptive_stop_actions": [2, 4],
        "is_correct": [True, False],
        "output_token_hash": ["a", "b"],
    })


class MathFeatureAblationTests(unittest.TestCase):
    def test_feature_groups_partition_non_bias_features(self):
        grouped = [
            feature
            for name, features in FEATURE_GROUPS.items()
            if name != "full"
            for feature in features
        ]
        self.assertEqual(set(grouped), set(FEATURE_NAMES[1:]))
        self.assertEqual(len(grouped), len(set(grouped)))

    def test_aggregate_method_uses_algorithm_time(self):
        summary = aggregate_method(result_frame(), "avg_td_full")
        self.assertAlmostEqual(summary["ms_per_output_token"], 100.0)
        self.assertAlmostEqual(summary["adaptive_stop_rate_percent"], 200.0 / 3.0)
        self.assertAlmostEqual(summary["acceptance_rate_percent"], 40.0)

    def test_paired_ablation_reports_full_controller_speedup(self):
        frames = {
            "avg_td_full": result_frame(),
            "avg_td_drop_frontier": result_frame(scale=1.25),
        }
        summary = paired_ablation_summary(frames).iloc[0]
        self.assertAlmostEqual(summary["full_speedup_vs_ablated"], 1.25)
        self.assertEqual(summary["full_win_rate_percent"], 100.0)
        self.assertEqual(summary["output_hash_match_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
