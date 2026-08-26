import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_corrected_greedy_oracle_four_datasets import (
    DEFAULT_REFERENCE_ZIP,
    aggregate_overall,
    dataset_configuration,
    import_reference_oracles,
)
from run_otrc_v2_td_benchmark import PROBLEM_IDS


class CorrectedGreedyOracleFourDatasetTests(unittest.TestCase):
    def test_aime_and_humaneval_use_fixed_twenty_five_problem_ids(self):
        for dataset in ("aime", "humaneval"):
            source, problem_ids = dataset_configuration(dataset, 25)
            self.assertEqual(source["dataset"], dataset)
            self.assertEqual(problem_ids, PROBLEM_IDS[dataset][:25])
            self.assertEqual(len(problem_ids), len(set(problem_ids)))
            self.assertNotIn(0, problem_ids)
            self.assertEqual(source["max_new_tokens"], 1024)
            self.assertEqual(source["max_spec_len"], 60)

    def test_bundled_reference_contains_corrected_math_and_gsm8k_fifty(self):
        self.assertTrue(DEFAULT_REFERENCE_ZIP.exists())
        with tempfile.TemporaryDirectory() as directory:
            imported, manifest = import_reference_oracles(
                DEFAULT_REFERENCE_ZIP,
                Path(directory),
            )
            self.assertEqual(set(imported), {"math", "gsm8k"})
            self.assertEqual(imported["math"]["num_questions"], 50)
            self.assertEqual(imported["gsm8k"]["num_questions"], 50)
            self.assertEqual(
                manifest["version"],
                "corrected_one_action_greedy_oracle_two_datasets_v1",
            )
            for dataset in imported:
                dataset_dir = Path(directory) / dataset
                self.assertTrue(
                    (dataset_dir / "greedy_local_oracle_summary.csv").exists()
                )
                self.assertTrue(
                    (dataset_dir / "corrected_oracle_feature_labels.csv").exists()
                )

    def test_overall_aggregation_uses_pooled_real_latency(self):
        first = pd.DataFrame([{
            "dataset": "math",
            "num_samples": 2,
            "total_generated_tokens": 100.0,
            "baseline_real_latency_ms": 1000.0,
            "greedy_real_latency_ms": 800.0,
            "pooled_real_speedup": 1.25,
            "baseline_verifier_calls": 20.0,
            "greedy_verifier_calls": 18.0,
            "baseline_dLLM_forwards": 40.0,
            "greedy_dLLM_forwards": 30.0,
            "output_hash_match_percent": 100.0,
        }])
        second = pd.DataFrame([{
            "dataset": "aime",
            "num_samples": 1,
            "total_generated_tokens": 50.0,
            "baseline_real_latency_ms": 500.0,
            "greedy_real_latency_ms": 500.0,
            "pooled_real_speedup": 1.0,
            "baseline_verifier_calls": 10.0,
            "greedy_verifier_calls": 10.0,
            "baseline_dLLM_forwards": 20.0,
            "greedy_dLLM_forwards": 20.0,
            "output_hash_match_percent": 100.0,
        }])

        datasets, overall = aggregate_overall([first, second])
        row = overall.iloc[0]
        self.assertEqual(len(datasets), 2)
        self.assertEqual(row.num_samples, 3)
        self.assertAlmostEqual(row.pooled_real_speedup, 1500.0 / 1300.0)
        self.assertAlmostEqual(row.baseline_ms_per_token, 10.0)
        self.assertAlmostEqual(row.oracle_ms_per_token, 1300.0 / 150.0)
        self.assertEqual(row.output_hash_match_percent, 100.0)


if __name__ == "__main__":
    unittest.main()
