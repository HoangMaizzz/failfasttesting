import argparse
import unittest

import pandas as pd

from run_adaptive_td_benchmark import aggregate, paired_comparison, sampled_problem_ids


class AdaptiveTDBenchmarkTests(unittest.TestCase):
    def test_sampling_is_deterministic(self):
        args = argparse.Namespace(
            datasets=["math", "aime", "gsm8k", "humaneval"],
            warmup_questions=1,
            num_questions=5,
            sample_seed=2026,
        )
        self.assertEqual(sampled_problem_ids(args), sampled_problem_ids(args))

    def test_paired_comparison_requires_identical_problem_keys(self):
        rows = pd.DataFrame({
            "dataset": ["math", "math", "math", "math"],
            "problem_id": [1, 2, 1, 2],
            "method": ["failfast", "failfast", "adaptive_td", "adaptive_td"],
            "measured_ms_per_output_token": [10.0, 5.0, 5.0, 10.0],
            "e2e_ms_per_output_token": [11.0, 6.0, 6.0, 12.0],
            "output_token_hash": ["a", "b", "a", "c"],
            "num_speculation_rounds": [2, 1, 1, 2],
            "total_num_forward_passes": [4, 2, 2, 4],
            "output_tokens": [10, 10, 10, 10],
        })
        paired = paired_comparison(rows)
        self.assertEqual(paired["measured_speedup_vs_failfast"].tolist(), [2.0, 0.5])
        self.assertEqual(paired["output_match"].tolist(), [1, 0])

    def test_aggregate_reports_stop_feasibility_separately(self):
        rows = pd.DataFrame({
            "dataset": ["gsm8k"],
            "method": ["adaptive_td"],
            "output_tokens": [8],
            "drafted_tokens": [8],
            "accepted_tokens": [4],
            "measured_time_s": [1.0],
            "actual_e2e_time": [1.1],
            "actual_draft_time": [0.4],
            "actual_verify_time": [0.5],
            "actual_post_verify_time": [0.1],
            "num_speculation_rounds": [2],
            "total_num_forward_passes": [3],
            "adaptive_decisions": [4],
            "adaptive_stop_actions": [1],
            "adaptive_exploration_actions": [1],
            "adaptive_stop_available_decisions": [3],
            "adaptive_candidate_coverage_decisions": [4],
            "adaptive_outer_verify_eligible_decisions": [3],
            "adaptive_mean_refinement_step": [1.5],
            "adaptive_controller_ms": [0.2],
        })
        summary = aggregate(rows).iloc[0]
        self.assertEqual(summary["adaptive_stop_available_rate_percent"], 75.0)
        self.assertEqual(summary["adaptive_candidate_coverage_rate_percent"], 100.0)
        self.assertEqual(summary["adaptive_outer_verify_eligible_rate_percent"], 75.0)


if __name__ == "__main__":
    unittest.main()
