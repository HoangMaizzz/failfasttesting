import unittest

import pandas as pd

from run_greedy_losslessness_audit import compare_token_traces, summarize_audit


class GreedyLosslessnessAuditTests(unittest.TestCase):
    def test_compare_token_traces_finds_first_difference(self):
        failfast = pd.DataFrame({
            "output_position": [0, 1, 2],
            "token_id": [10, 11, 12],
        })
        method_a = pd.DataFrame({
            "output_position": [0, 1, 2, 3],
            "token_id": [10, 99, 12, 13],
        })
        result = compare_token_traces(failfast, method_a).iloc[0]
        self.assertEqual(result["first_different_position"], 1)
        self.assertEqual(result["failfast_output_length"], 3)
        self.assertEqual(result["method_a_output_length"], 4)

    def test_summary_separates_internal_and_causal_mismatches(self):
        rows = pd.DataFrame({
            "emitted_matches_batched": [1, 0, 1],
            "batched_matches_prefix": [1, 1, 0],
            "absolute_output_position": [4, 5, 6],
            "batched_margin": [2.0, 1.0, 0.1],
            "prefix_margin": [1.5, 0.9, 0.2],
        })
        result = summarize_audit("method_a", rows)
        self.assertEqual(result["internal_token_mismatches"], 1)
        self.assertEqual(result["batched_prefix_argmax_mismatches"], 1)
        self.assertEqual(result["first_internal_mismatch_position"], 5)
        self.assertEqual(result["first_batched_prefix_mismatch_position"], 6)


if __name__ == "__main__":
    unittest.main()
