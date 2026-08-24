import unittest
from types import SimpleNamespace

import pandas as pd

from run_gsm8k_round_balanced_question_benchmark import (
    round_balance_metrics,
    round_balance_score,
    nested_pool_problem_ids,
    select_round_balanced_questions,
)


class Gsm8kRoundBalancedQuestionBenchmarkTests(unittest.TestCase):
    def test_larger_pool_preserves_previous_candidate_prefix(self):
        small = nested_pool_problem_ids(SimpleNamespace(
            warmup_questions=1,
            sample_seed=2026,
            pool_size=20,
        ))
        large = nested_pool_problem_ids(SimpleNamespace(
            warmup_questions=1,
            sample_seed=2026,
            pool_size=40,
        ))
        self.assertEqual(small, large[:20])

    def test_balance_score_is_zero_for_equal_round_counts(self):
        self.assertEqual(round_balance_score([10, 10, 10]), 0.0)
        self.assertGreater(round_balance_score([20, 9, 1]), 0.0)

    def test_selector_returns_unique_questions_and_all_round_counts(self):
        profiles = pd.DataFrame({
            "problem_id": list(range(9)),
            "oracle_step1_rounds": [8, 7, 6, 1, 1, 2, 1, 2, 1],
            "oracle_step2_rounds": [1, 2, 1, 8, 7, 6, 1, 1, 2],
            "oracle_step3plus_rounds": [1, 1, 2, 1, 2, 1, 8, 7, 6],
        })

        selected, diagnostics = select_round_balanced_questions(
            profiles,
            selected_size=6,
            selection_seed=17,
            restarts=8,
            max_iterations=50,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected["problem_id"].nunique(), 6)
        self.assertLessEqual(
            diagnostics["optimized"]["selection_objective"],
            diagnostics["random_reference"]["selection_objective"],
        )

    def test_selector_is_deterministic(self):
        profiles = pd.DataFrame({
            "problem_id": list(range(12)),
            "oracle_step1_rounds": [index % 5 + 1 for index in range(12)],
            "oracle_step2_rounds": [(index * 2) % 5 + 1 for index in range(12)],
            "oracle_step3plus_rounds": [
                (index * 3) % 5 + 1 for index in range(12)
            ],
        })
        first, _ = select_round_balanced_questions(
            profiles, 6, 23, restarts=4, max_iterations=20
        )
        second, _ = select_round_balanced_questions(
            profiles, 6, 23, restarts=4, max_iterations=20
        )
        self.assertEqual(
            first["problem_id"].tolist(), second["problem_id"].tolist()
        )

    def test_round_balance_metrics_report_all_classes(self):
        metrics = round_balance_metrics([30, 30, 20])
        self.assertEqual(metrics["total_rounds"], 80)
        self.assertEqual(metrics["step3plus_rounds"], 20)
        self.assertAlmostEqual(metrics["step1_percent"], 37.5)


if __name__ == "__main__":
    unittest.main()
