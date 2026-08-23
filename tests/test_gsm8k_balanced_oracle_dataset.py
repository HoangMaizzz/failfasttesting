import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from run_gsm8k_balanced_oracle_dataset import (
    balanced_quotas,
    causal_oracle_comparison,
    completed_failfast_batch,
    pass_class,
    partition_problem_ids,
    problem_profiles,
    select_balanced_problems,
)


class Gsm8kBalancedOracleDatasetTests(unittest.TestCase):
    def test_problem_ids_are_partitioned_without_loss(self):
        batches = partition_problem_ids(list(range(7)), 3)
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5], [6]])

    def test_resume_requires_every_expected_problem(self):
        with TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            pd.DataFrame({"problem_id": [1, 2]}).to_csv(
                batch_dir / "benchmark_results.csv", index=False
            )
            self.assertTrue(completed_failfast_batch(batch_dir, [1, 2]))
            self.assertFalse(completed_failfast_batch(batch_dir, [1, 2, 3]))

    def test_pass_classes_use_total_draft_passes(self):
        self.assertEqual(pass_class(1), "step1")
        self.assertEqual(pass_class(1.5), "step2")
        self.assertEqual(pass_class(2), "step2")
        self.assertEqual(pass_class(3), "step3plus")

    def test_problem_class_uses_median_oracle_passes(self):
        rounds = pd.DataFrame({
            "problem_id": [1, 1, 1, 2, 2, 2],
            "oracle_draft_passes": [1, 2, 2, 2, 3, 4],
            "factual_draft_passes": [3, 3, 3, 4, 4, 4],
        })
        profiles = problem_profiles(rounds).set_index("problem_id")
        self.assertEqual(profiles.loc[1, "problem_oracle_class"], "step2")
        self.assertEqual(profiles.loc[2, "problem_oracle_class"], "step3plus")
        self.assertEqual(profiles.loc[1, "oracle_step2_rounds"], 2)

    def test_balanced_selection_is_deterministic_and_exact(self):
        profiles = pd.DataFrame({
            "problem_id": range(12),
            "problem_oracle_class": ["step1"] * 4 + ["step2"] * 4 + ["step3plus"] * 4,
            "class_purity_percent": [75.0] * 12,
            "oracle_step1_rounds": [1] * 12,
            "oracle_step2_rounds": [1] * 12,
            "oracle_step3plus_rounds": [1] * 12,
            "oracle_step1_round_percent": [34.0] * 12,
            "oracle_step2_round_percent": [33.0] * 12,
            "oracle_step3plus_round_percent": [33.0] * 12,
        })
        rounds = pd.DataFrame({
            "problem_id": [problem_id for problem_id in range(12) for _ in range(3)],
            "round_id": [0, 1, 2] * 12,
            "oracle_pass_class": ["step1", "step2", "step3plus"] * 12,
        })
        first, anchors, quotas = select_balanced_problems(
            profiles, rounds, 9, 17
        )
        second, second_anchors, _ = select_balanced_problems(
            profiles, rounds, 9, 17
        )
        self.assertEqual(quotas, {"step1": 3, "step2": 3, "step3plus": 3})
        self.assertEqual(
            first["selection_stratum"].value_counts().to_dict(),
            quotas,
        )
        self.assertEqual(first["problem_id"].tolist(), second["problem_id"].tolist())
        self.assertEqual(anchors["problem_id"].tolist(), second_anchors["problem_id"].tolist())
        self.assertEqual(anchors["problem_id"].nunique(), 9)

    def test_selection_rejects_fake_balance_when_a_class_is_short(self):
        profiles = pd.DataFrame({
            "problem_id": range(5),
            "problem_oracle_class": ["step1", "step1", "step2", "step2", "step3plus"],
            "class_purity_percent": [100.0] * 5,
            "oracle_step1_rounds": [1, 1, 0, 0, 0],
            "oracle_step2_rounds": [0, 0, 1, 1, 0],
            "oracle_step3plus_rounds": [0, 0, 0, 0, 1],
            "oracle_step1_round_percent": [100.0, 100.0, 0.0, 0.0, 0.0],
            "oracle_step2_round_percent": [0.0, 0.0, 100.0, 100.0, 0.0],
            "oracle_step3plus_round_percent": [0.0, 0.0, 0.0, 0.0, 100.0],
        })
        rounds = pd.DataFrame({
            "problem_id": range(5),
            "round_id": [0] * 5,
            "oracle_pass_class": ["step1", "step1", "step2", "step2", "step3plus"],
        })
        with self.assertRaisesRegex(ValueError, "cannot provide"):
            select_balanced_problems(profiles, rounds, 6, 1)

    def test_causal_comparison_uses_executed_algorithm_times(self):
        failfast = pd.DataFrame({
            "problem_id": [1],
            "actual_algorithm_time": [0.06],
            "actual_draft_time": [0.04],
            "actual_verify_time": [0.02],
            "actual_post_verify_time": [0.0],
            "output_tokens": [3],
            "theo_total_time": [30.0],
            "total_num_forward_passes": [3],
            "num_speculation_rounds": [1],
            "accepted_tokens": [2],
            "drafted_tokens": [3],
            "output_token_hash": ["same"],
            "is_correct": [True],
        })
        causal = failfast.copy()
        causal["actual_algorithm_time"] = 0.03
        causal["actual_draft_time"] = 0.01
        causal["theo_total_time"] = 15.0
        causal["total_num_forward_passes"] = 1
        decisions = pd.DataFrame({
            "counterfactual_probe_wall_time_ms": [10.0],
            "excluded_extra_draft_latency_ms": [5.0],
            "counterfactual_matches_execution": [1],
        })
        summary = causal_oracle_comparison(failfast, causal, decisions).iloc[0]
        self.assertEqual(summary["causal_oracle_speedup_vs_failfast"], 2.0)
        self.assertEqual(summary["modeled_causal_oracle_speedup_vs_failfast"], 2.0)
        self.assertEqual(summary["excluded_oracle_diagnostic_time_s"], 0.015)
        self.assertEqual(balanced_quotas(50), {"step1": 17, "step2": 17, "step3plus": 16})


if __name__ == "__main__":
    unittest.main()
