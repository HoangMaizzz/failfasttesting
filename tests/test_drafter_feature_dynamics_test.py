import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import drafter_feature_dynamics_test as dynamics


class FeatureDynamicsTests(unittest.TestCase):
    def _observation(self, sid, qid, seed):
        rng = np.random.default_rng(seed)
        state = dynamics.Observation(
            sid=sid, qid=qid, hidden=rng.normal(size=(8, 4)).astype(np.float32),
            scalars=rng.random((8, 10), dtype=np.float32),
            valid=np.ones(8, dtype=bool), history=np.zeros((8, 5), dtype=np.float32),
        )
        state._target_hidden = rng.normal(size=(8, 3)).astype(np.float32)
        state._target_scalars = rng.normal(size=(8, 3)).astype(np.float32)
        state._target_mask = (rng.random(8) > 0.5).astype(np.float32)
        state._target_event = (rng.random(8) > 0.5).astype(np.float32)
        return state

    def test_question_split_is_exhaustive_and_disjoint(self):
        qids = [f"q{i}" for i in range(40)]
        split = dynamics.question_split(qids, seed=17)
        self.assertEqual(set(split), set(qids))
        self.assertEqual(sum(value == "train" for value in split.values()), 24)
        self.assertEqual(sum(value == "validation" for value in split.values()), 8)
        self.assertEqual(sum(value == "test" for value in split.values()), 8)

    def test_capture_trim_discards_token_content_and_question_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "features").mkdir()
            hidden_dim = 4
            raw = np.arange(2 * 8 * (3 * hidden_dim + 10), dtype=np.float32)
            raw = raw.reshape(2, 8, 3 * hidden_dim + 10).astype(np.float16)
            shard = root / "features" / "shard_00000.npz"
            np.savez_compressed(
                shard, x=raw, valid=np.ones((2, 8), dtype=bool),
                native_token_ids=np.ones((2, 8), dtype=np.int32),
                candidate_token_ids=np.full((2, 8), 7, dtype=np.int32),
                content=np.ones((2, 8, hidden_dim), dtype=np.float16),
            )
            (root / "states.jsonl").write_text("".join(
                json.dumps({"state_id": f"s{i}", "question_id": "q0",
                            "shard": "features/shard_00000.npz", "row": i}) + "\n"
                for i in range(2)), encoding="utf-8")
            (root / "questions.jsonl").write_text(json.dumps({
                "question_id": "q0", "question": "private prompt",
                "prefix_token_ids": [1, 2, 3], "state_ids": ["s0", "s1"],
            }) + "\n", encoding="utf-8")
            (root / "capture_manifest.json").write_text(json.dumps({"schema_version": "test"}),
                                                          encoding="utf-8")

            self.assertEqual(dynamics.trim_capture_to_features(root), hidden_dim)
            with np.load(shard, allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"hidden", "scalars", "valid"})
                np.testing.assert_array_equal(arrays["hidden"], raw[..., :hidden_dim])
                np.testing.assert_array_equal(
                    arrays["scalars"], raw[..., 3 * hidden_dim:3 * hidden_dim + 10])
            question = json.loads((root / "questions.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("question", question)
            self.assertNotIn("prefix_token_ids", question)
            manifest = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("token embeddings", manifest["feature_dynamics_audit"][
                "excluded_before_training"])

    def test_action_conditioned_latent_rollout_has_feature_only_outputs(self):
        torch.manual_seed(4)
        states = [self._observation(f"s{i}", f"q{i}", i) for i in range(2)]
        for variant in dynamics.FEATURE_VARIANTS:
            input_dim = len(dynamics.state_input(states[0], variant)[0])
            model = dynamics.FeatureDynamics(input_dim, latent_dim=64, continuous_dim=6)
            x = torch.as_tensor(np.stack([
                dynamics.state_input(state, variant) for state in states]), dtype=torch.float32)
            valid = torch.ones((2, 8), dtype=torch.bool)
            latent = model.encoder(x, valid)
            latent = model.transition(latent, torch.tensor([0, 1]), valid)
            cont, mask, event = model.decoder(latent)
            self.assertEqual(tuple(cont.shape), (2, 8, 6))
            self.assertEqual(tuple(mask.shape), (2, 8))
            self.assertEqual(tuple(event.shape), (2, 8))
            loss = dynamics._feature_loss(cont, mask, event, states, valid)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.encoder.input[0].weight.grad)
            self.assertFalse(any("token" in name or "vocab" in name
                                 for name, _ in model.named_parameters()))

    def test_training_validity_mask_uses_destination_observation(self):
        source = self._observation("src", "q0", 1)
        child = self._observation("dst", "q0", 2)
        child.valid[4:] = False
        cont = torch.zeros((1, 8, 6), requires_grad=True)
        mask = torch.zeros((1, 8), requires_grad=True)
        event = torch.zeros((1, 8), requires_grad=True)
        valid = torch.as_tensor(child.valid[None])
        loss = dynamics._feature_loss(
            cont, mask, event, [child], valid)
        self.assertTrue(torch.isfinite(loss))
        baseline = float(loss)
        child._target_hidden[4:] = 1e6
        child._target_scalars[4:] = -1e6
        child._target_mask[4:] = 1.0 - child._target_mask[4:]
        child._target_event[4:] = 1.0 - child._target_event[4:]
        changed_invalid_targets = dynamics._feature_loss(
            cont, mask, event, [child], valid)
        self.assertAlmostEqual(float(changed_invalid_targets), baseline, places=6)

    def test_three_step_open_loop_training_and_validation(self):
        states = {f"s{i}": self._observation(f"s{i}", f"q{i // 4}", i)
                  for i in range(8)}
        train_path = ("s0", [
            dynamics.Edge("s0", "s1", 0),
            dynamics.Edge("s1", "s2", 1),
            dynamics.Edge("s2", "s3", 0),
        ])
        validation_path = ("s4", [
            dynamics.Edge("s4", "s5", 1),
            dynamics.Edge("s5", "s6", 0),
            dynamics.Edge("s6", "s7", 1),
        ])
        model, logs, best = dynamics.fit_model(
            states, [train_path], [validation_path], "scalars", latent_dim=64,
            device=torch.device("cpu"), updates=1, batch_size=1, seed=9,
            label="unit_h3")
        self.assertTrue(np.isfinite(best))
        self.assertEqual(len(logs), 1)
        rows = dynamics.evaluate_paths(model, [validation_path], states, "scalars",
                                       torch.device("cpu"))
        self.assertEqual([row["horizon"] for row in rows], [1, 2, 3])
        self.assertTrue(all(np.isfinite(row["pred_cont"]).all() for row in rows))


if __name__ == "__main__":
    unittest.main()
