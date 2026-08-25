import unittest

from run_corrected_greedy_oracle_two_datasets import (
    gsm8k_configuration,
    math_configuration,
)


class CorrectedGreedyOracleTwoDatasetTests(unittest.TestCase):
    def test_math_uses_bundled_matched_problem_ids(self):
        source, problem_ids = math_configuration(3)
        self.assertEqual(source["dataset"], "math")
        self.assertEqual(problem_ids, [2, 6, 42])

    def test_gsm8k_sampling_is_deterministic_and_matches_previous_sample(self):
        first_source, first_ids = gsm8k_configuration(50, 2026)
        second_source, second_ids = gsm8k_configuration(50, 2026)
        self.assertEqual(first_source, second_source)
        self.assertEqual(first_ids, second_ids)
        self.assertIn(1202, first_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))


if __name__ == "__main__":
    unittest.main()
