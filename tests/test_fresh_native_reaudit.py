import unittest

import numpy as np
import torch

from fresh_native_reaudit import ActionDirectPredictor
from latent_sufficiency_audit import Edge, State, edge_loss


class ReauditTests(unittest.TestCase):
    def test_direct_baseline_receives_action(self):
        model = ActionDirectPredictor(7, 128, 4).eval()
        x = torch.zeros(2, 8, 7)
        with torch.no_grad():
            r = model(x, torch.zeros(2, dtype=torch.long))
            e = model(x, torch.ones(2, dtype=torch.long))
        self.assertFalse(torch.allclose(r["mask"], e["mask"]))

    def test_content_direction_loss_is_invariant_to_target_scale(self):
        state = State("a", "q", np.zeros((2, 1)), np.zeros(2, bool), np.ones(2, bool),
                      np.zeros(2), np.zeros(2, bool), np.ones((2, 3), np.float32), 2)
        edge = Edge("a", "a", 0, np.ones(2, bool), np.zeros(2, bool), np.zeros(2, bool))
        prediction = {"mask": torch.zeros(1, 2), "confidence": torch.zeros(1, 2),
                      "changed": torch.zeros(1, 2), "content": -torch.ones(1, 2, 3)}
        first = edge_loss(prediction, [edge], {"a": state}, torch.device("cpu"), "cosine")
        state.content *= .001
        second = edge_loss(prediction, [edge], {"a": state}, torch.device("cpu"), "cosine")
        self.assertAlmostEqual(float(first), float(second), places=6)


if __name__ == "__main__":
    unittest.main()
