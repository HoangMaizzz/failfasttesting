import unittest

import pandas as pd

from run_v2_gain_calibration_validation import (
    add_analysis_columns,
    summarize_events,
)


class V2GainCalibrationValidationTests(unittest.TestCase):
    def test_calibrated_prediction_improves_error_metrics(self):
        events = pd.DataFrame({
            "problem_id": [0, 0, 1],
            "raw_predicted_extension_gain": [2.0, 2.0, 2.0],
            "predicted_extension_gain": [3.8, 4.0, 4.2],
            "actual_extension_accepted_tokens": [4.0, 4.0, 4.0],
            "extension_gain_correction": [1.9, 2.0, 2.0],
            "extension_gain_uncertainty": [0.5, 0.4, 0.3],
            "extension_gain_ucb_bonus": [0.5, 0.4, 0.3],
            "original_prefix_fully_accepted": [1, 1, 1],
        })

        summary = summarize_events(events)

        self.assertAlmostEqual(summary["raw_bias_pred_minus_actual"], -2.0)
        self.assertAlmostEqual(summary["calibrated_bias_pred_minus_actual"], 0.0)
        self.assertGreater(summary["mae_reduction_percent"], 90.0)

    def test_analysis_columns_preserve_signed_errors_and_stage(self):
        events = pd.DataFrame({
            "raw_predicted_extension_gain": [2.0],
            "predicted_extension_gain": [3.0],
            "actual_extension_accepted_tokens": [4.0],
            "from_len": [16],
            "gain_calibration_count": [12],
            "gain_calibration_source": ["block"],
        })

        result = add_analysis_columns(events).iloc[0]

        self.assertEqual(result["raw_error"], -2.0)
        self.assertEqual(result["calibrated_error"], -1.0)
        self.assertEqual(result["proposal_length_bucket"], "9-16")
        self.assertEqual(result["calibration_stage"], "block:1-31")


if __name__ == "__main__":
    unittest.main()
