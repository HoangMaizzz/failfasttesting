import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_bucket_renewal_benchmark import (
    METHODS,
    load_reference,
    paired_comparison,
    sampled_problem_ids,
)


class BucketRenewalBenchmarkTests(unittest.TestCase):
    def test_bucket_renewal_changes_only_unmask_policy(self):
        candidate = METHODS["bucket_renewal_spec8"]
        self.assertEqual(set(METHODS), {"bucket_renewal_spec8"})
        self.assertEqual(candidate["spec_len"], 8)
        self.assertEqual(candidate["incr_len"], 8)
        self.assertEqual(candidate["frontier_mode"], "bucket_renewal")

    def test_problem_sampling_is_deterministic_and_shared(self):
        args = argparse.Namespace(
            datasets=["math", "aime", "gsm8k", "humaneval"],
            warmup_questions=1,
            num_questions=10,
            sample_seed=2026,
        )
        first = sampled_problem_ids(args)
        second = sampled_problem_ids(args)
        self.assertEqual(first, second)
        self.assertTrue(all(len(values) == 10 for values in first.values()))

    def test_paired_comparison_uses_identical_problem_ids(self):
        rows = pd.DataFrame({
            "dataset": ["math", "math", "math", "math"],
            "problem_id": [1, 2, 1, 2],
            "method": ["candidate", "candidate", "baseline", "baseline"],
            "measured_ms_per_output_token": [5.0, 10.0, 10.0, 5.0],
            "output_token_hash": ["a", "b", "a", "c"],
        })
        paired = paired_comparison(rows, "candidate", "baseline")
        self.assertEqual(paired["speedup"].tolist(), [2.0, 0.5])
        self.assertEqual(paired["output_match"].tolist(), [1, 0])

    def test_reference_loader_reuses_saved_failfast_rows(self):
        frame = pd.DataFrame({
            "dataset": ["math"],
            "problem_id": [53],
            "output_tokens": [10],
            "accepted_tokens": [4],
            "drafted_tokens": [8],
            "num_speculation_rounds": [2],
            "total_num_forward_passes": [3],
            "output_token_hash": ["hash"],
            "actual_measured_time": [0.5],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            frame.to_csv(path, index=False)
            args = argparse.Namespace(reference_csv=str(path), datasets=["math"])
            loaded = load_reference(args, {"math": [53]})
        self.assertEqual(loaded.loc[0, "method"], "failfast_spec8_reference")
        self.assertEqual(loaded.loc[0, "measured_ms_per_output_token"], 50.0)


if __name__ == "__main__":
    unittest.main()
