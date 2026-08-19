import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_failfast_spec8_matched_benchmark import (
    METHOD,
    load_matched_reference,
    sampled_problem_ids,
)


class FailFastSpec8MatchedBenchmarkTests(unittest.TestCase):
    def test_method_is_matched_to_adaptive_only(self):
        self.assertEqual(METHOD["spec_len"], 8)
        self.assertEqual(METHOD["incr_len"], 8)
        self.assertEqual(METHOD["lowconf_threshold"], 0.45)
        self.assertEqual(METHOD["frontier_mode"], "disabled")

    def test_sampling_matches_adaptive_only(self):
        args = argparse.Namespace(
            datasets=["math", "aime", "gsm8k", "humaneval"],
            warmup_questions=1,
            num_questions=15,
            sample_seed=2026,
        )
        problem_ids = sampled_problem_ids(args)
        self.assertEqual(problem_ids["math"][0], 53)
        self.assertEqual(problem_ids["aime"][0], 1)
        self.assertEqual(problem_ids["gsm8k"][0], 318)
        self.assertEqual(problem_ids["humaneval"][0], 6)

    def test_reference_requires_exact_problem_keys(self):
        expected = {"gsm8k": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({
                "dataset": ["gsm8k", "gsm8k"],
                "problem_id": [1, 3],
                "method": ["adaptive_unmask_only", "adaptive_unmask_only"],
            }).to_csv(path / "per_observation.csv", index=False)
            with self.assertRaises(ValueError):
                load_matched_reference(path, expected)


if __name__ == "__main__":
    unittest.main()
