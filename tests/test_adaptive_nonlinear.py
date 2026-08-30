import unittest

import torch

from adaptive_nonlinear import CompactNonlinearVA, OnlineNonlinearVA


FEATURES = [1.0, 0.2, -0.3, 0.5, 0.7, 0.1]


class AdaptiveNonlinearTests(unittest.TestCase):
    def test_neutral_initialization_and_sign_convention(self):
        learner = OnlineNonlinearVA(
            "nam", learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, huber_delta=32.0, seed=42,
        )
        value, advantage, q_stop, q_continue = learner.predict(FEATURES)
        self.assertEqual((value, advantage, q_stop, q_continue), (0.0,) * 4)
        with torch.no_grad():
            learner.model.bias[1] = 2.0
        _, advantage, q_stop, q_continue = learner.predict(FEATURES)
        self.assertGreater(advantage, 0.0)
        self.assertGreater(q_stop, q_continue)

    def test_nam_and_ga2m_composition(self):
        x = torch.tensor([[0.2, -0.3, 0.5, 0.7, 0.1]])
        for model_type in ("nam", "ga2m"):
            model = CompactNonlinearVA(model_type, seed=7)
            with torch.no_grad():
                model.bias[:] = torch.tensor([1.0, -1.0])
                for module in list(model.main_effects) + list(model.interactions):
                    module.network[-1].bias[:] = torch.tensor([0.2, 0.1])
            total, main, interactions = model.components(x)
            expected = model.bias + main.sum(1) + interactions.sum(1)
            self.assertTrue(torch.allclose(total, expected))
            self.assertEqual(interactions.shape[1], 0 if model_type == "nam" else 10)

    def test_partial_feedback_and_checkpoint_round_trip(self):
        learner = OnlineNonlinearVA(
            "ga2m", learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, huber_delta=32.0, seed=42,
        )
        before = learner.predict(FEATURES)
        update = learner.update("stop", FEATURES, target=4.0, weight=2.0)
        self.assertAlmostEqual(update["residual"], 4.0)
        self.assertEqual(learner.update_count, 1)
        snapshot = learner.snapshot()
        restored = OnlineNonlinearVA(
            "ga2m", learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, huber_delta=32.0, seed=99,
        )
        restored.load_snapshot(snapshot)
        self.assertEqual(learner.predict(FEATURES), restored.predict(FEATURES))
        self.assertNotEqual(before, learner.predict(FEATURES))

    def test_parameter_counts(self):
        nam = CompactNonlinearVA("nam")
        ga2m = CompactNonlinearVA("ga2m")
        self.assertEqual(sum(p.numel() for p in nam.parameters()), 312)
        self.assertEqual(sum(p.numel() for p in ga2m.parameters()), 1012)


if __name__ == "__main__":
    unittest.main()
