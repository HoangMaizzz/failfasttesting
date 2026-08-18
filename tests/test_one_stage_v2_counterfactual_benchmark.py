import unittest
from types import SimpleNamespace

import pandas as pd

from run_one_stage_v2_counterfactual_benchmark import (
    counterfactual_summary,
    sampled_problem_ids,
)


class OneStageV2CounterfactualBenchmarkTests(unittest.TestCase):
    def test_problem_sampling_is_reproducible_and_excludes_warmup(self):
        args = SimpleNamespace(
            datasets=["aime", "gsm8k"],
            num_questions=5,
            warmup_questions=1,
            sample_seed=2026,
        )

        first = sampled_problem_ids(args)
        second = sampled_problem_ids(args)

        self.assertEqual(first, second)
        self.assertEqual(len(first["aime"]), 5)
        self.assertNotIn(0, first["aime"])

    def test_counterfactual_summary_uses_realized_gain_for_cost(self):
        events = pd.DataFrame({
            "dataset": ["gsm8k", "gsm8k"],
            "problem_id": [1, 2],
            "trigger": [
                "cost_aware_v2_counterfactual_extend",
                "cost_aware_v2_counterfactual_extend",
            ],
            "actual_extension_accepted_tokens": [8, 0],
            "predicted_extension_gain": [2.0, 2.0],
            "extension_size": [8, 8],
            "decision_expected_output": [5.0, 5.0],
            "stop_ms_per_output": [10.0, 10.0],
            "predicted_extend_ms_per_output": [12.0, 12.0],
            "estimated_extension_total_ms": [100.0, 100.0],
        })

        analyzed, summary = counterfactual_summary(events, 0.05)

        self.assertTrue(bool(analyzed.iloc[0]["strictly_beneficial"]))
        self.assertFalse(bool(analyzed.iloc[1]["strictly_beneficial"]))
        self.assertEqual(summary.iloc[0]["positive_gain_rate_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
