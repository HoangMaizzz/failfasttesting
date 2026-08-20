import json
import unittest
from types import SimpleNamespace

import pandas as pd

from run_online_unmask_calibration_validation import (
    evaluate_dataset,
    summarize_predictions,
)


def snapshot(problem_id, round_id, step, confidences, recoverable, emitted, draft_ms):
    target_len = len(confidences)
    accepted = emitted - 1
    return {
        "dataset": "gsm8k",
        "problem_id": problem_id,
        "round_id": round_id,
        "step": step,
        "target_len": target_len,
        "draft_latency_elapsed_ms": draft_ms,
        "masks_remaining": sum(recoverable),
        "committed_tokens": target_len - sum(recoverable),
        "filled_tokens": sum(recoverable),
        "frontier_k": target_len - sum(recoverable),
        "frontier_score": float(accepted),
        "unmasked_this_step": 2,
        "token_confidences": json.dumps(confidences),
        "token_margins": json.dumps([0.4] * target_len),
        "token_forced": json.dumps([False] * target_len),
        "token_recoverable": json.dumps(recoverable),
        "context_len": 128,
        "actual_verify_latency_ms": 100.0,
        "accepted_len_if_stop": accepted,
        "emitted_len_if_stop": emitted,
    }


class OnlineUnmaskCalibrationValidationTests(unittest.TestCase):
    def test_evaluation_uses_warmup_but_reports_only_measured_problem(self):
        rows = [
            snapshot(0, 0, 1, [0.8, 0.7, 0.4, 0.3], [0, 0, 1, 1], 3, 40.0),
            snapshot(0, 0, 2, [0.8, 0.7, 0.7, 0.6], [0, 0, 0, 0], 4, 80.0),
            snapshot(1, 0, 1, [0.9, 0.8, 0.5, 0.4], [0, 0, 1, 1], 3, 42.0),
            snapshot(1, 0, 2, [0.9, 0.8, 0.8, 0.7], [0, 0, 0, 0], 5, 82.0),
        ]
        args = SimpleNamespace(
            warmup_questions=1,
            initial_draft_latency_ms=40.0,
            initial_verify_latency_ms=100.0,
            ema_alpha=0.2,
        )

        result = evaluate_dataset(pd.DataFrame(rows), args, "gsm8k")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["problem_id"], 1)
        self.assertGreater(result.iloc[0]["hazard_observations_before"], 0)
        self.assertGreater(result.iloc[0]["transition_observations_before"], 0)
        self.assertEqual(result.iloc[0]["actual_gain"], 2)

    def test_summary_reports_gain_and_decision_metrics(self):
        predictions = pd.DataFrame({
            "dataset": ["math", "math"],
            "problem_id": [1, 2],
            "predicted_gain": [1.0, 0.0],
            "actual_gain": [2.0, 0.0],
            "gain_error": [-1.0, 0.0],
            "predicted_current_y": [3.0, 2.0],
            "actual_current_y": [3.0, 2.0],
            "predicted_next_y": [4.0, 2.0],
            "actual_next_y": [5.0, 2.0],
            "predicted_next_draft_ms": [40.0, 40.0],
            "actual_next_draft_ms": [42.0, 38.0],
            "predicted_verify_ms": [100.0, 100.0],
            "actual_verify_ms": [105.0, 95.0],
            "predicted_action": ["continue", "stop"],
            "oracle_action": ["continue", "stop"],
            "decision_correct": [1, 1],
            "decision_regret_ms_per_token": [0.0, 0.0],
        })

        summary = summarize_predictions(predictions, ["dataset"]).iloc[0]

        self.assertEqual(summary["transitions"], 2)
        self.assertAlmostEqual(summary["gain_mae"], 0.5)
        self.assertAlmostEqual(summary["decision_accuracy_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
