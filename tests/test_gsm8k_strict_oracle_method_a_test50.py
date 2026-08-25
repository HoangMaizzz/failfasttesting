import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adaptive_td import FEATURE_NAMES
from run_gsm8k_strict_oracle_method_a_test50 import (
    confusion_summary,
    feature_alignment,
    match_decisions,
    method_a_phase_complete,
)


def feature_row(first_value):
    values = [0.0] * len(FEATURE_NAMES)
    values[0] = 1.0
    values[1] = float(first_value)
    return json.dumps(values)


class Gsm8kStrictOracleMethodATests(unittest.TestCase):
    def test_method_a_resume_requires_all_expected_problem_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame({"problem_id": [1, 2]}).to_csv(
                directory / "benchmark_results.csv",
                index=False,
            )
            pd.DataFrame({"action": ["stop"]}).to_csv(
                directory / "adaptive_td_decisions.csv",
                index=False,
            )
            (directory / "adaptive_td_runtime_state.json").write_text(
                json.dumps({"controller_name": "avg_td"}),
                encoding="utf-8",
            )
            self.assertTrue(method_a_phase_complete(directory, [1, 2]))
            self.assertFalse(method_a_phase_complete(directory, [1, 2, 3]))

    def test_exact_state_matching_rejects_different_proposal(self):
        method = pd.DataFrame({
            "problem_id": [7, 7],
            "context_len": [100, 100],
            "target_len": [8, 8],
            "step": [1, 1],
            "draft_proposal": ["[1, 2]", "[1, 9]"],
            "features": [feature_row(0.2), feature_row(0.8)],
            "action": ["stop", "continue"],
            "stop_probability": [0.8, 0.2],
            "advantage_mean": [1.0, -1.0],
        })
        oracle = pd.DataFrame({
            "sample_id": [7],
            "context_len": [100],
            "accumulated_proposal_length": [8],
            "refinement_step": [1],
            "draft_proposal": ["[1, 2]"],
            "chosen_action": ["stop"],
        })
        _, _, matched = match_decisions(method, oracle)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["method_a_action"], "stop")
        self.assertEqual(matched.iloc[0]["confusion_class"], "true_stop")

    def test_confusion_summary_reports_stop_and_continue_errors(self):
        matched = pd.DataFrame({
            "confusion_class": [
                "true_stop",
                "false_stop",
                "false_continue",
                "true_continue",
            ],
            "oracle_stop": [1, 0, 1, 0],
            "method_a_stop": [1, 1, 0, 0],
            "stop_probability": [0.9, 0.8, 0.2, 0.1],
            "advantage_mean": [1.0, 0.5, -0.5, -1.0],
        })
        row = confusion_summary(matched).iloc[0]
        self.assertEqual(row["true_stop"], 1)
        self.assertEqual(row["false_continue"], 1)
        self.assertAlmostEqual(row["accuracy_percent"], 50.0)
        self.assertAlmostEqual(row["stop_recall_percent"], 50.0)
        self.assertAlmostEqual(row["continue_recall_percent"], 50.0)

    def test_feature_alignment_detects_opposite_learned_direction(self):
        matched = pd.DataFrame({"oracle_stop": [1, 1, 0, 0]})
        for name in FEATURE_NAMES:
            matched[f"feature_{name}"] = [0.0, 0.1, 0.8, 0.9]
        state = {
            "actions": {
                "stop": {"theta": [1.0] * len(FEATURE_NAMES)},
                "continue": {"theta": [0.0] * len(FEATURE_NAMES)},
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            report = feature_alignment(matched, path)
        remaining = report.loc[
            report.feature.eq("remaining_mask_ratio")
        ].iloc[0]
        self.assertEqual(remaining["oracle_direction"], "lower_means_stop")
        self.assertEqual(remaining["learned_direction"], "higher_means_stop")
        self.assertEqual(remaining["alignment"], "opposite")


if __name__ == "__main__":
    unittest.main()
