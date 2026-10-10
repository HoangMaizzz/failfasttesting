import unittest

import numpy as np
import torch

from latent_sufficiency_audit import (
    Edge, Encoder, State, _auc, _topk_distribution, question_split, score_records,
)


class LatentSufficiencyAuditTests(unittest.TestCase):
    def test_question_split_is_deterministic_and_disjoint(self):
        ids = [str(i) for i in range(100)]
        first = question_split(ids)
        second = question_split(list(reversed(ids)))
        self.assertEqual(first, second)
        self.assertEqual({key: list(first.values()).count(key)
                          for key in ("train", "validation", "test")},
                         {"train": 70, "validation": 15, "test": 15})

    def test_topk_distribution_is_finite_and_sums_to_one(self):
        values = _topk_distribution(np.asarray([[1000.0, 999.0], [-1000.0, -1001.0]]))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values.sum(-1), np.ones(2), rtol=1e-6)

    def test_encoder_handles_physical_canvas(self):
        encoder = Encoder(input_dim=7, width=128)
        result = encoder(torch.zeros(2, 32, 7))
        self.assertEqual(tuple(result.shape), (2, 32, 128))

    def test_auc_and_edge_metrics_have_expected_values(self):
        self.assertEqual(_auc(np.asarray([0, 1]), np.asarray([.1, .9])), 1.0)
        source = State("a", "q1", np.zeros((4, 3), np.float16),
                       np.asarray([True, True, False, False]), np.ones(4, bool),
                       np.asarray([.2, .3, .4, .5], np.float32), np.zeros(4, bool),
                       np.ones((4, 2), np.float32), 4)
        target = State("b", "q1", np.zeros((4, 3), np.float16),
                       np.asarray([False, True, False, False]), np.ones(4, bool),
                       np.asarray([.8, .3, .4, .5], np.float32),
                       np.asarray([True, False, False, False]),
                       np.ones((4, 2), np.float32), 4)
        edge = Edge("a", "b", 0, np.asarray([True, True, False, False]),
                    np.asarray([True, True, False, False]), np.zeros(4, bool))
        prediction = {
            "mask": np.zeros(4, np.float32), "confidence": np.asarray([.8, .3, .4, .5]),
            "changed": np.asarray([2., -2., 0., 0.]), "content": np.ones((4, 2), np.float32),
        }
        result = score_records([(edge, prediction)], {"a": source, "b": target}, [])
        self.assertEqual(result["position_count"], 2)
        self.assertAlmostEqual(result["confidence_mae"], 0.0, places=6)
        self.assertEqual(result["candidate_change_auc"], 1.0)
        self.assertIn("current_native_rule_mask_brier", result)

    def test_mask_logit_is_scored_as_probability_of_remaining_masked(self):
        source = State("a", "q", np.zeros((2, 1), np.float16), np.ones(2, bool),
                       np.ones(2, bool), np.zeros(2, np.float32), np.zeros(2, bool),
                       np.ones((2, 1), np.float32), 2)
        target = State("b", "q", np.zeros((2, 1), np.float16),
                       np.asarray([False, True]), np.ones(2, bool),
                       np.zeros(2, np.float32), np.zeros(2, bool),
                       np.ones((2, 1), np.float32), 2)
        edge = Edge("a", "b", 0, np.ones(2, bool), np.zeros(2, bool), np.zeros(2, bool))
        pred = {"mask": np.asarray([-8.0, 8.0]), "confidence": np.zeros(2),
                "changed": np.zeros(2), "content": np.ones((2, 1))}
        result = score_records([(edge, pred)], {"a": source, "b": target}, [])
        self.assertEqual(result["mask_auc"], 1.0)
        self.assertLess(result["mask_brier"], 1e-5)


if __name__ == "__main__":
    unittest.main()
