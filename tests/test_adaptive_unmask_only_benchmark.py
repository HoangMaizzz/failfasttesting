import argparse
import unittest

import pandas as pd

from run_adaptive_unmask_only_benchmark import (
    build_extension_policy_summary,
    sampled_problem_ids,
)


class AdaptiveUnmaskOnlyBenchmarkTests(unittest.TestCase):
    def test_sampling_matches_reference_seed_policy(self):
        args = argparse.Namespace(
            datasets=["math", "aime", "gsm8k", "humaneval"],
            warmup_questions=1,
            num_questions=15,
            sample_seed=2026,
        )

        first = sampled_problem_ids(args)
        second = sampled_problem_ids(args)

        self.assertEqual(first, second)
        self.assertTrue(all(len(ids) == 15 for ids in first.values()))
        self.assertTrue(all(0 not in ids for ids in first.values()))

    def test_extension_summary_confirms_naive_policy(self):
        diagnostics = pd.DataFrame({
            "dataset": ["math", "math"],
            "trigger": ["high_confidence_extend", "high_confidence_extend"],
        })

        summary = build_extension_policy_summary(diagnostics).iloc[0]

        self.assertEqual(summary["total_extensions"], 2)
        self.assertEqual(summary["high_confidence_extensions"], 2)
        self.assertEqual(summary["cost_aware_extensions"], 0)


if __name__ == "__main__":
    unittest.main()
