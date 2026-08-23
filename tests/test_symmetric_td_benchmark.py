import unittest
from types import SimpleNamespace

from run_symmetric_td_benchmark import sampled_problem_ids, validate_args


class SymmetricTDBenchmarkTests(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "datasets": ["gsm8k"],
            "train_questions": 50,
            "num_questions": 50,
            "spec_len": 8,
            "incr_len": 8,
            "adaptive_min_action_probability": 0.1,
            "adaptive_max_importance_weight": 5.0,
            "sample_seed": 2026,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_training_and_evaluation_ids_are_disjoint_and_deterministic(self):
        args = self.args()
        first = sampled_problem_ids(args)
        second = sampled_problem_ids(args)
        self.assertEqual(first, second)
        train = set(first["gsm8k"]["train"])
        evaluation = set(first["gsm8k"]["evaluation"])
        self.assertEqual(len(train), 50)
        self.assertEqual(len(evaluation), 50)
        self.assertFalse(train.intersection(evaluation))

    def test_validation_rejects_overlapping_capacity(self):
        args = self.args(
            datasets=["aime"],
            train_questions=20,
            num_questions=20,
        )
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
