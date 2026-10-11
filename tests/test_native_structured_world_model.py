"""CPU shape/gradient checks for the native-structured latent feasibility model."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_structured_world_model import NativeStructuredWorldModel
from world_model_core import (ExperienceReplay, Observation, WorldModelLearner,
                              acceptance_nll, prefix_log_distribution)


def batch(width=16, prefix_width=24):
    torch.manual_seed(19)
    b, layers, hidden, token_dim, top_k = 2, 3, 16, 24, 4
    scalars = torch.zeros(b, width, 16)
    scalars[:, :, 3] = 1
    scalars[:, :, 4] = 1
    scalars[:, :4, 0] = 1
    scalars[:, :4, 1] = 0
    scalars[:, 4:, 1] = 1
    scalars[:, :, 7] = .7
    scalars[:, :, 15] = .5
    lengths = torch.tensor([8, 12])
    return dict(
        hidden=torch.randn(b, width, layers, hidden),
        token_vectors=torch.randn(b, width, 2, token_dim),
        topk_vectors=torch.randn(b, width, token_dim),
        gaps=torch.randn(b, width, top_k).clamp(-5, 0),
        scalars=scalars,
        history=torch.randn(b, width, 4),
        context=torch.tensor([[.2, .125, 0, 0, .5, .5, .125, 1.],
                              [.2, .1875, 0, .25, .5, .5, .125, 1.]]),
        lengths=lengths,
        prefix_vectors=torch.randn(b, prefix_width, token_dim),
        prefix_lengths=torch.tensor([prefix_width, prefix_width-5]),
        labels=torch.tensor([3, 4]),
    )


def observation(uid, length, accepted, masks=2, newly=1):
    ids = torch.randint(1, 24, (length, 2))
    scalars = torch.zeros(length, 16)
    scalars[:masks, 0] = 1
    scalars[masks:, 1] = 1
    scalars[:, 3] = 1
    scalars[:, 4] = 1
    scalars[:newly, 2] = 1
    scalars[:, 7] = .65
    scalars[:, 15] = .5
    return Observation(uid, "q", 0, ids, torch.randn(length, 3, 16),
        torch.randn(length, 4).clamp(-5, 0), scalars,
        torch.tensor([.1, length/64, 0., 0., .5, .5, .125, 1.]),
        accepted=accepted, prefix_ids=torch.tensor([1, 2, 3, 4]),
        topk_ids=torch.randint(1, 24, (length, 4)), history=torch.randn(length, 4))


class NativeStructuredModelTests(unittest.TestCase):
    def test_current_acceptance_distribution_and_loss(self):
        model = NativeStructuredWorldModel(16, 24, top_k=4, dim=32,
                                           num_hidden_layers=3, dropout=0).eval()
        values = batch()
        latent = model.encoder(values)
        logits = model.acceptance(latent)
        logp = prefix_log_distribution(logits, latent.lengths)
        self.assertEqual(tuple(latent.tokens.shape), (2, 16, 32))
        self.assertTrue(torch.allclose(logp.exp().sum(-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isneginf(logp[0, 9:]).all())
        loss = acceptance_nll(logits, latent.lengths, values["labels"])
        self.assertTrue(torch.isfinite(loss))

    def test_action_conditioned_rollouts_preserve_native_legality(self):
        model = NativeStructuredWorldModel(16, 24, top_k=4, dim=32,
                                           num_hidden_layers=3, dropout=0)
        values = batch()
        state = model.encoder(values)
        actions = torch.tensor([0, 1])  # R, E
        child = model.transition(state, actions, 8)
        self.assertEqual(child.lengths.tolist(), [8, 20])
        self.assertEqual(tuple(child.tokens.shape), (2, 20, 32))
        self.assertTrue(torch.all(child.mask_probs[0, :8] <= state.mask_probs[0, :8] + 1e-7))
        self.assertTrue(torch.equal(child.mask_probs[1, :12], torch.zeros(12)))
        self.assertTrue(torch.equal(child.event_probs[0, 4:8], torch.zeros(4)))
        self.assertTrue(torch.equal(child.event_probs[1, :12], torch.zeros(12)))
        self.assertTrue(torch.isfinite(model.acceptance(child)).all())
        second = model.transition(child, torch.tensor([0, 0]), 8)
        third = model.transition(second, torch.tensor([1, 0]), 8)
        self.assertEqual(third.lengths.tolist(), [16, 20])

    def test_structural_loss_backpropagates_and_feature_lesions_run(self):
        model = NativeStructuredWorldModel(16, 24, top_k=4, dim=32,
                                           num_hidden_layers=3, dropout=0)
        values = batch()
        state = model.encoder(values)
        child = model.transition(state, torch.tensor([0, 1]), 8)
        future = batch(width=20)
        future["lengths"] = torch.tensor([8, 20])
        future["scalars"][:, :, 0] = 0
        future["scalars"][1, 12:20, 0] = 1
        loss = model.structural_loss(child, future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        for variant in model.encoder.VARIANTS:
            model.encoder.variant = variant
            with torch.no_grad():
                result = model.encoder(values)
            self.assertTrue(torch.isfinite(result.tokens).all(), variant)
        model.encoder.variant = "full"

    def test_world_model_learner_trains_on_native_R_E_edges(self):
        torch.manual_seed(23)
        model = NativeStructuredWorldModel(16, 24, top_k=4, dim=32,
                                           num_hidden_layers=3, dropout=0)
        root = observation("root", 8, 4, masks=4, newly=2)
        refined = observation("refined", 8, 5, masks=2, newly=2)
        extended = observation("extended", 16, 9, masks=4, newly=4)
        replay = ExperienceReplay(16, seed=7)
        replay.add(root, refined, "R")
        replay.add(refined, extended, "E")
        learner = WorldModelLearner(model, torch.randn(64, 24), "cpu", 8,
            lr=1e-3, warmup_updates=0, horizon_warmup=0)
        result = learner.update(replay, batch_size=2, max_horizon=2)
        self.assertIsNotNone(result)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertGreater(result["structural_loss"], 0)
        self.assertEqual(result["horizon"], 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
