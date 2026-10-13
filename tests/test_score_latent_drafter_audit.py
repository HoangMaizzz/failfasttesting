import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from native_latent_dynamics_v2 import Edge, fit_scalers
from native_latent_rules_v2 import next_geometry
from score_latent_drafter import ScoreEncoder, ScoreTransition, _state_from_batch, pack_states
from score_latent_drafter_audit import _auc, _binary_metrics, exact_paths, locate_run_root
from test_score_latent_drafter import make_state


class ScoreLatentDrafterAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.geometry = np.array([96, 96, 96, 8, 0], dtype=np.int64)

    def test_exact_horizon_coverage_and_question_split_isolation(self):
        qid = "heldout"
        states = {}
        chain = []
        geometry = self.geometry.copy()
        for index in range(5):
            if index == 2:
                geometry = next_geometry(torch.as_tensor(geometry[None]), torch.tensor([1]))[0].numpy()
            state = make_state(f"s{index}", qid, geometry, u_offset=index * .1)
            states[state.sid] = state
            chain.append(state.sid)
        train_root = make_state("train0", "train_q", self.geometry)
        train_next = make_state("train1", "train_q", self.geometry, u_offset=.2)
        states[train_root.sid] = train_root
        states[train_next.sid] = train_next
        edges = [Edge(chain[i], chain[i + 1], (0, 0, 1, 0)[i]) for i in range(4)]
        edges.append(Edge("train0", "train1", 0))
        split = {qid: "test", "train_q": "train"}

        paths1 = exact_paths(states, edges, split, 1)
        paths2 = exact_paths(states, edges, split, 2)
        paths3 = exact_paths(states, edges, split, 3)
        self.assertEqual(len(paths1), 4)
        self.assertEqual(len(paths2), 3)
        self.assertEqual(len(paths3), 2)
        self.assertTrue(all(len(path) == horizon
                            for horizon, paths in ((1, paths1), (2, paths2), (3, paths3))
                            for _, path in paths))
        self.assertTrue(all(states[root].qid == "heldout"
                            for paths in (paths1, paths2, paths3) for root, _ in paths))
        # Every observed edge is scored exactly once at H1, without slicing a path pool.
        self.assertEqual({(path[-1].src, path[-1].dst) for _, path in paths1},
                         {(edge.src, edge.dst) for edge in edges if edge.src != "train0"})

    def test_auc_and_probability_metrics_handle_ties_and_single_class(self):
        self.assertAlmostEqual(_auc([0, 1], [.1, .9]), 1.0)
        self.assertAlmostEqual(_auc([0, 1], [.5, .5]), .5)
        metrics = _binary_metrics([0, 1], [.5, .5])
        self.assertAlmostEqual(metrics["brier"], .25)
        self.assertIsNone(_auc([1, 1], [.2, .8]))

    def test_input_folder_can_contain_a_nested_extracted_run(self):
        required = (
            "run_status.json", "config.json", "split.json",
            "latent_score_drafter_report.json", "teacher_features/states.jsonl",
            "teacher_features/edges.jsonl", "models/full_context.pt",
        )
        with TemporaryDirectory() as temp:
            run_root = Path(temp) / "dataset" / "exported_run"
            for name in required:
                path = run_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(locate_run_root(Path(temp)), run_root)

    def test_all_feature_ablations_keep_hard_native_constraints(self):
        root = make_state("root", "q", self.geometry)
        child = make_state("child", "q", self.geometry, before=root.mask, u_offset=.1)
        states = {root.sid: root, child.sid: child}
        scalers = fit_scalers(states, {"q": "train"})
        batch = pack_states([root.sid], states, scalers, "cpu")
        ablations = (
            "native_residual", "position_embedding", "scheduler_structure",
            "scheduler_context", "geometry_embedding", "prefix_kv",
            "action_conditioning",
        )
        encoder = ScoreEncoder(12, 16, layers=1, feature_set="full_context")
        transition = ScoreTransition(16, layers=1, feature_set="full_context")
        for feature in ablations:
            z, memory_z = encoder(batch, ablate_features=(feature,))
            self.assertEqual(tuple(z.shape), (1, 32, 16))
            self.assertTrue(torch.isfinite(z).all())
            state = _state_from_batch(batch, (z, memory_z))
            predicted, aux = transition(state, torch.tensor([0]), ablate_features=(feature,))
            self.assertTrue(torch.equal(predicted["geometry"], batch["geometry"]))
            self.assertTrue(torch.all(predicted["mask"] <= batch["mask"] + 1e-6))
            self.assertTrue(torch.all(predicted["event"] <= aux["eligible"] + 1e-6))
            self.assertGreaterEqual(float(predicted["event"].sum()), 1.0 - 1e-6)
            if feature == "prefix_kv":
                self.assertEqual(float(memory_z.abs().sum()), 0.0)
                self.assertEqual(float(predicted["memory_z"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
