import unittest

import pandas as pd

from run_oracle_refinement_profile import (
    summarize_datasets,
    summarize_oracle_rounds,
    summarize_oracle_steps,
)


class OracleRefinementProfileTests(unittest.TestCase):
    def test_round_summary_reports_early_saturation(self):
        oracle = pd.DataFrame({
            "dataset": ["gsm8k", "gsm8k", "gsm8k"],
            "problem_id": [1, 1, 1],
            "round_id": [0, 0, 0],
            "step": [1, 2, 3],
            "accepted_len_if_stop": [2, 5, 5],
            "emitted_len_if_stop": [3, 6, 6],
            "draft_latency_elapsed_ms": [10.0, 20.0, 30.0],
            "masks_remaining": [5, 2, 0],
        })

        summary = summarize_oracle_rounds(oracle).iloc[0]

        self.assertEqual(summary["final_step"], 3)
        self.assertEqual(summary["best_accept_step"], 2)
        self.assertEqual(summary["accept_saturates_before_final"], 1)
        self.assertEqual(summary["wasted_steps_by_accept"], 1)

    def test_step_and_dataset_summaries_average_expected_columns(self):
        oracle = pd.DataFrame({
            "dataset": ["math", "math"],
            "problem_id": [1, 1],
            "round_id": [0, 0],
            "step": [1, 2],
            "masks_remaining": [4, 0],
            "committed_tokens": [4, 8],
            "filled_tokens": [4, 0],
            "accepted_len_if_stop": [3, 6],
            "emitted_len_if_stop": [4, 7],
            "delta_accepted_len": [3, 3],
            "delta_emitted_len": [4, 3],
            "draft_latency_elapsed_ms": [12.0, 24.0],
        })

        step_summary = summarize_oracle_steps(oracle)
        round_summary = summarize_oracle_rounds(oracle)
        dataset_summary = summarize_datasets(round_summary).iloc[0]

        self.assertEqual(len(step_summary), 2)
        self.assertEqual(dataset_summary["rounds"], 1)
        self.assertEqual(dataset_summary["mean_final_step"], 2)
        self.assertEqual(dataset_summary["mean_best_accepted_len"], 6)


if __name__ == "__main__":
    unittest.main()
