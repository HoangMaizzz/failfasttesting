"""Tests for the trainable single-hidden-layer ablation."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model_core import Observation
from world_model_retrained_feature_compare import feature_observations, summarize_seed


class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        self.obs = Observation(
            uid='s0', question='q0', round_id=0,
            ids=torch.zeros(5, 2, dtype=torch.long),
            hidden=torch.arange(5 * 3 * 4, dtype=torch.float16).reshape(5, 3, 4),
            gaps=torch.zeros(5, 32, dtype=torch.float16),
            scalars=torch.zeros(5, 16), context=torch.zeros(8), accepted=3,
            prefix_ids=torch.tensor([1, 2]), topk_ids=torch.ones(5, 32, dtype=torch.long),
            history=torch.zeros(5, 3))

    def test_hidden28_keeps_one_layer_and_all_other_features(self):
        selected = feature_observations({'s0': self.obs}, selected_index=2)['s0']
        self.assertEqual(tuple(selected.hidden.shape), (5, 1, 4))
        self.assertTrue(torch.equal(selected.hidden[:, 0], self.obs.hidden[:, 2]))
        self.assertIs(selected.gaps, self.obs.gaps)
        self.assertIs(selected.topk_ids, self.obs.topk_ids)
        self.assertIs(selected.prefix_ids, self.obs.prefix_ids)
        self.assertEqual(selected.accepted, self.obs.accepted)

    def test_out_of_range_layer_fails_loudly(self):
        with self.assertRaises(ValueError):
            feature_observations({'s0': self.obs}, selected_index=3)

    def test_paired_retraining_metrics_keep_full_as_reference(self):
        rows = []
        for question in ('q0', 'q1'):
            for variant, prediction in (('full', 2.0), ('hidden28', 3.0)):
                rows.append(dict(split='validation', variant=variant, question=question,
                    source=question, state_id=question, depth=0, actions='', group='current',
                    length=8, K=4, expected_K=prediction, nll=1.0))
        result = summarize_seed(rows, seed=42, update=512, train_questions=80,
                                model_parameters={'full': 100, 'hidden28': 80})
        current = result['metrics']['current']
        self.assertEqual(current['full_question_macro_mae'], 2.0)
        self.assertEqual(current['hidden28_question_macro_mae'], 1.0)
        self.assertEqual(current['hidden28_minus_full_question_macro_mae'], -1.0)


if __name__ == '__main__':
    unittest.main()
