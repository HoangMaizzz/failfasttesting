import argparse
import unittest

import pandas as pd

from run_adaptive_td_benchmark import paired_comparison, sampled_problem_ids


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


if __name__ == "__main__":
    unittest.main()
