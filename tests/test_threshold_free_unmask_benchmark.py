import argparse
import unittest

import run_threshold_free_unmask_benchmark as threshold_free


class ThresholdFreeUnmaskBenchmarkTests(unittest.TestCase):
    def test_method_uses_threshold_free_refinement_mode(self):
        self.assertEqual(threshold_free.benchmark.METHOD["name"], "threshold_free_unmask_only")
        self.assertEqual(
            threshold_free.benchmark.METHOD["frontier_mode"],
            "cost_aware_v2_refinement_no_threshold",
        )
        self.assertEqual(threshold_free.benchmark.METHOD["spec_len"], 8)
        self.assertEqual(threshold_free.benchmark.METHOD["incr_len"], 8)

    def test_sampling_uses_same_problem_policy(self):
        args = argparse.Namespace(
            datasets=["math", "aime", "gsm8k", "humaneval"],
            warmup_questions=1,
            num_questions=10,
            sample_seed=2026,
        )

        problem_ids = threshold_free.benchmark.sampled_problem_ids(args)

        self.assertTrue(all(len(ids) == 10 for ids in problem_ids.values()))
        self.assertTrue(all(0 not in ids for ids in problem_ids.values()))


if __name__ == "__main__":
    unittest.main()
