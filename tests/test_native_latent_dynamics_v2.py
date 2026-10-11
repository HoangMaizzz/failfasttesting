import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from native_latent_dynamics_v2 import (
    ACTION_TO_ID, Edge, NativeLatentDynamics, State, build_paths,
    split_questions, supervised_loss, train_and_audit,
)


class NativeLatentDynamicsV2Tests(unittest.TestCase):
    def test_question_split_is_deterministic_and_disjoint(self):
        qids = [f"q{i}" for i in range(25)]
        a = split_questions(qids, seed=11)
        b = split_questions(qids, seed=11)
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {"train", "validation", "test"})
        self.assertEqual(len(a), len(qids))

    def test_paths_respect_action_horizon_and_question_split(self):
        def state(sid, qid):
            return State(sid, qid, np.zeros((32, 16), np.float32),
                         np.zeros(4096, np.float32), np.zeros(32, np.float32),
                         np.zeros(32, np.float32), np.zeros(32, np.float32),
                         np.zeros(7, np.float32), True, {})
        states = {f"q{i}_s{j}": state(f"q{i}_s{j}", f"q{i}")
                  for i in range(5) for j in range(4)}
        split = {f"q{i}": "test" if i == 0 else "train" for i in range(5)}
        edges = [Edge(f"q0_s{i}", f"q0_s{i+1}", ACTION_TO_ID["R" if i % 2 == 0 else "E"])
                 for i in range(3)]
        paths = build_paths(states, edges, split, "test", max_horizon=3)
        root = next(chain for sid, chain in paths if sid == "q0_s0")
        self.assertEqual(len(root), 3)
        self.assertEqual([edge.action for edge in root], [0, 1, 0])
        self.assertTrue(all(states[e.dst].qid == "q0" for e in root))

    def test_model_outputs_only_latent_and_structural_features(self):
        for feature_set in ("native_only", "native_structure", "full"):
            model = NativeLatentDynamics(hidden_size=16, memory_size=4096,
                                         dim=32, layers=1, feature_set=feature_set)
            batch = {
                "u": torch.randn(2, 32, 16),
                "memory": torch.randn(2, 4096),
                "mask": torch.randint(0, 2, (2, 32)).float(),
                "event": torch.randint(0, 2, (2, 32)).float(),
                "active": torch.randint(0, 2, (2, 32)).float(),
                "context": torch.randn(2, 7),
                "memory_valid": torch.ones(2),
            }
            z = model.encode(batch)
            next_z, output = model.step(z, torch.tensor([0, 1]))
            self.assertEqual(next_z.shape, (2, 32, 32))
            self.assertEqual(set(output), {"u", "memory", "mask", "event", "active", "context"})
            self.assertEqual(output["u"].shape, (2, 32, 16))
            self.assertEqual(output["memory"].shape, (2, 4096))
            self.assertEqual(output["context"].shape, (2, 7))
            self.assertFalse(any("token" in name or "logit" in name for name in output))

    def test_loss_has_finite_gradients(self):
        b, n, d, m = 3, 32, 12, 4096
        pred = {
            "u": torch.randn(b, n, d, requires_grad=True),
            "memory": torch.randn(b, m, requires_grad=True),
            "mask": torch.randn(b, n, requires_grad=True),
            "event": torch.randn(b, n, requires_grad=True),
            "active": torch.randn(b, n, requires_grad=True),
            "context": torch.randn(b, 7, requires_grad=True),
        }
        target = {
            "u": torch.randn(b, n, d), "memory": torch.randn(b, m),
            "mask": torch.randint(0, 2, (b, n)).float(),
            "event": torch.randint(0, 2, (b, n)).float(),
            "active": torch.randint(0, 2, (b, n)).float(),
            "context": torch.randn(b, 7),
        }
        loss, _ = supervised_loss(pred, target, torch.tensor([True, False, True]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(pred["u"].grad)
        self.assertTrue(torch.isfinite(pred["u"].grad).all())

    def test_tiny_train_validate_test_run(self):
        rng = np.random.default_rng(7)
        states = {}
        edges = []
        for q in range(20):
            qid = f"q{q:02d}"
            previous = None
            for step in range(3):
                sid = f"{qid}_s{step}"
                states[sid] = State(
                    sid=sid, qid=qid,
                    u=rng.normal(size=(32, 8)).astype(np.float32),
                    memory=rng.normal(size=4096).astype(np.float32),
                    mask=rng.integers(0, 2, size=32).astype(np.float32),
                    event=rng.integers(0, 2, size=32).astype(np.float32),
                    active=rng.integers(0, 2, size=32).astype(np.float32),
                    context=rng.normal(size=7).astype(np.float32),
                    memory_valid=True, meta={},
                )
                if previous is not None:
                    edges.append(Edge(previous, sid, (q + step) % 2))
                previous = sid
        split = split_questions({state.qid for state in states.values()}, seed=9)
        args = SimpleNamespace(
            train_device="cpu", horizon=3, latent_dim=16, model_layers=1,
            updates=2, batch_size=4, seed=9, cut_layer=24,
        )
        with TemporaryDirectory() as tmp:
            report, scalers = train_and_audit(args, states, edges, split, Path(tmp))
            self.assertEqual(set(report["variants"]), {"native_only", "native_structure", "full"})
            self.assertIn("action_R_h1", report["variants"]["full"]["test"])
            self.assertTrue((Path(tmp) / "model_full.pt").is_file())
            self.assertEqual(scalers["u_mean"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
