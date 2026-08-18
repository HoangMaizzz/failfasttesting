import math
import unittest

import pandas as pd

from run_cost_aware_threshold_validation import (
    build_calibration_table,
    prediction_metrics,
    summarize_bucket_gain_ablation,
    summarize_gain_gap_decomposition,
    summarize_next_step_gain,
    summarize_rounds,
)


class CostAwareThresholdValidationTests(unittest.TestCase):
    def test_prediction_metrics_report_bias_and_error(self):
        frame = pd.DataFrame({"predicted": [1.0, 3.0], "actual": [2.0, 2.0]})

        metrics = prediction_metrics(frame, "predicted", "actual")

        self.assertEqual(metrics["num_observations"], 2)
        self.assertAlmostEqual(metrics["bias_predicted_minus_actual"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["predicted_to_actual_ratio"], 1.0)

    def test_round_summary_separates_full_accept_with_capacity(self):
        frame = pd.DataFrame({
            "dataset": ["gsm8k"] * 4,
            "method": ["cost_aware_lowconf_0p45"] * 4,
            "full_accept": [1, 1, 0, 1],
            "full_accept_with_extension_capacity": [1, 0, 0, 1],
            "extension_count": [0, 1, 0, 2],
            "cost_stop_requested": [1, 0, 1, 0],
            "stop_reason": ["failfast_low_confidence", "failfast_max_spec_len", "cost_aware_low_expected_gain", "failfast_low_confidence"],
            "draft_len": [8, 60, 8, 16],
            "accepted_len": [8, 60, 2, 16],
        })

        summary = summarize_rounds(frame).iloc[0]

        self.assertEqual(summary["num_rounds"], 4)
        self.assertEqual(summary["full_accept_rounds"], 3)
        self.assertEqual(summary["full_accept_with_extension_capacity_rounds"], 2)
        self.assertEqual(summary["full_accept_with_capacity_zero_extension_rounds"], 1)
        self.assertAlmostEqual(
            summary["full_accept_with_capacity_rate_given_full_accept_percent"],
            200.0 / 3.0,
        )

    def test_empty_calibration_input_is_supported(self):
        result = build_calibration_table(
            pd.DataFrame(),
            "predicted",
            "actual",
            [-math.inf, 1.0, math.inf],
        )

        self.assertTrue(result.empty)

    def test_next_step_gain_supports_only_extension_transitions(self):
        frame = pd.DataFrame({
            "dataset": ["gsm8k", "gsm8k"],
            "method": ["cost_aware_lowconf_0p45"] * 2,
            "same_target_len": [0, 0],
            "predicted_next_gain": [0.2, 0.4],
            "actual_next_gain": [0.1, 0.5],
        })

        summary = summarize_next_step_gain(frame).iloc[0]

        self.assertEqual(summary["num_observations"], 2)
        self.assertEqual(summary["same_target_len_rate_percent"], 0.0)

    def test_bucket_ablation_separates_prefix_and_global_prior_error(self):
        frame = pd.DataFrame({
            "dataset": ["gsm8k", "gsm8k"],
            "method": ["cost_aware_lowconf_0p45"] * 2,
            "trigger": ["high_confidence_extend"] * 2,
            "extension_size": [2, 2],
            "extension_prior_probability": [0.5, 0.5],
            "original_prefix_fully_accepted": [1, 0],
            "actual_extension_accepted_tokens": [2, 0],
            "prefix_survival_probability": [0.25, 0.25],
            "prefix_survival_raw_confidence": [0.5, 0.5],
            "prefix_survival_without_confidence_bucket": [0.3, 0.3],
            "prefix_survival_without_margin_bucket": [0.4, 0.4],
            "prefix_survival_without_forced_bucket": [0.2, 0.2],
            "predicted_extension_gain": [0.1875, 0.1875],
            "predicted_extension_gain_raw_confidence": [0.375, 0.375],
            "predicted_extension_gain_without_confidence_bucket": [0.225, 0.225],
            "predicted_extension_gain_without_margin_bucket": [0.3, 0.3],
            "predicted_extension_gain_without_forced_bucket": [0.15, 0.15],
        })

        ablation = summarize_bucket_gain_ablation(frame)
        full = ablation[ablation["variant"] == "full_calibration"].iloc[0]
        no_margin = ablation[ablation["variant"] == "without_margin_bucket"].iloc[0]
        oracle = ablation[ablation["variant"] == "oracle_prefix_global_q"].iloc[0]

        self.assertAlmostEqual(full["predicted_mean"], 0.1875)
        self.assertAlmostEqual(no_margin["predicted_gain_change_vs_full"], 0.1125)
        self.assertAlmostEqual(oracle["predicted_mean"], 0.375)

        decomposition = summarize_gain_gap_decomposition(ablation).iloc[0]
        self.assertAlmostEqual(decomposition["actual_extension_gain"], 1.0)
        self.assertAlmostEqual(decomposition["prefix_product_gap"], 0.1875)
        self.assertAlmostEqual(decomposition["global_q_gap"], 0.625)


if __name__ == "__main__":
    unittest.main()
