import unittest
from types import SimpleNamespace

import pandas as pd

from run_v2_extension_margin_sweep import (
    build_reference_comparison,
    margin_label,
    validate_args,
)


class V2ExtensionMarginSweepTests(unittest.TestCase):
    def test_margin_labels_distinguish_gain_and_loss(self):
        self.assertEqual(margin_label(-0.05), "require_gain_5pct")
        self.assertEqual(margin_label(0.0), "break_even")
        self.assertEqual(margin_label(0.10), "allow_loss_10pct")

    def test_reference_margin_uses_tolerant_float_matching(self):
        args = SimpleNamespace(
            num_questions=20,
            warmup_questions=1,
            max_new_tokens=1024,
            extension_cost_margins=[-0.05, -0.0300000000001, 0.0],
            reference_margin=-0.03,
        )

        validate_args(args)

        self.assertEqual(args.reference_margin, -0.0300000000001)

    def test_reference_comparison_is_paired_by_problem(self):
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
            "frontier_v2_extend_actions": 1,
            "frontier_v2_extension_stop_actions": 1,
            "output_token_hash": "same",
        }
        rows = pd.DataFrame([
            {
                **common,
                "extension_cost_margin": -0.03,
                "actual_measured_time": 0.51,
                "actual_measured_ms_per_output_token": 51.0,
            },
            {
                **common,
                "extension_cost_margin": 0.05,
                "actual_measured_time": 0.45,
                "actual_measured_ms_per_output_token": 45.0,
                "frontier_v2_extend_actions": 2,
            },
        ])

        paired, summary = build_reference_comparison(rows, -0.03)

        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired.iloc[0]["candidate_speedup_vs_reference"], 51.0 / 45.0)
        self.assertEqual(summary.iloc[0]["paired_win_rate_percent"], 100.0)
        self.assertEqual(summary.iloc[0]["extend_action_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
