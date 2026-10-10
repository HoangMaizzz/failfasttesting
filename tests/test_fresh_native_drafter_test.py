import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh_native_drafter_test import (
    MASK_ID, ShardWriter, _make_edge, _state_row, audit_capture, load_capture,
)
from latent_sufficiency_audit import State


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(MASK_ID + 64, 4)
        with torch.no_grad():
            values = torch.arange(self.embedding.weight.numel()).reshape_as(self.embedding.weight)
            self.embedding.weight.copy_(values / 100.0)
        self.config = type("Config", (), {"hidden_size": 4})()

    def get_input_embeddings(self):
        return self.embedding


def fake_snapshot():
    return {
        "proposal_token_ids_before_fill": [MASK_ID, MASK_ID, 12, 13, 14, 15, 16, 17],
        "proposal_token_ids_after_fill": [20, 21, 12, 13, 14, 15, 16, 17],
        "proposal_mask_before_fill": [True, True, False, False, False, False, False, False],
        "hidden_states": [
            [[float(layer), float(pos), 1.0, -1.0] for pos in range(8)]
            for layer in range(5)
        ],
        "hidden_layer_indices": [0, 1, 2, 3, 4],
        "native_hidden_start_offset": 1,
        "topk_token_ids": [[30, 31] for _ in range(8)],
        "topk_logits": [[2.0, 1.0] for _ in range(8)],
        "native_topk_start_offset": 0,
        "confidences": [0.7] * 8,
        "margins": [0.2] * 8,
        "newly_unmasked_positions": [0],
        "unmask_forward_index": 1,
    }


class FreshNativeDrafterTests(unittest.TestCase):
    def test_native_feature_offsets_are_respected(self):
        row = _state_row(fake_snapshot(), FakeModel(), 50, 0, 8, 151645, 4)
        self.assertFalse(row["feature_valid"][0])
        self.assertTrue(row["feature_valid"][1:].all())
        # The final hidden layer's first available row aligns to proposal position 1.
        self.assertEqual(float(row["x"][1, 0]), 4.0)
        self.assertEqual(int(row["candidate_token_ids"][1]), 30)
        self.assertEqual(row["content"].shape, (8, 4))
        self.assertEqual(row["x"].shape, (8, 22))  # 3*hidden + ten scalar features

    def test_r_edge_scores_only_source_masks_and_marks_candidate_change(self):
        a = {"state_id": "a", "candidate_token_ids": np.asarray([10, 11, 12, 13]),
             "valid": np.ones(4, dtype=bool)}
        b = {"state_id": "b", "candidate_token_ids": np.asarray([10, 20, 12, 30]),
             "valid": np.ones(4, dtype=bool)}
        states = {
            "a": State("a", "q", np.zeros((4, 3)),
                       np.asarray([True, True, False, False]), np.ones(4, bool),
                       np.zeros(4), np.zeros(4, bool), np.ones((4, 2)), 4),
            "b": State("b", "q", np.zeros((4, 3)),
                       np.asarray([False, False, False, False]), np.ones(4, bool),
                       np.zeros(4), np.zeros(4, bool), np.ones((4, 2)), 4),
        }
        edge = _make_edge(a, b, "R", states)
        self.assertEqual(edge.valid.tolist(), [True, True, False, False])
        self.assertEqual(edge.changed_valid.tolist(), [True, True, False, False])
        self.assertEqual(states["b"].changed.tolist(), [False, True, False, True])

    def test_shards_round_trip_capture_state_and_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = ShardWriter(root, shard_rows=2)
            base = _state_row(fake_snapshot(), FakeModel(), 50, 0, 8, 151645, 4)
            for sid in ("a", "b"):
                meta = {"state_id": sid, "question_id": "q0"}
                writer.append({**base, "meta": meta})
            writer.flush()
            (root / "edges.jsonl").write_text(
                json.dumps({"src_state_id": "a", "dst_state_id": "b", "action": "R",
                            "valid": [True] * 8, "changed_valid": [True] * 8,
                            "new_suffix": [False] * 8}) + "\n", encoding="utf-8")
            states, edges, _ = load_capture(root)
            self.assertEqual(set(states), {"a", "b"})
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0].action, 0)
            self.assertEqual(states["a"].x.shape, (8, 22))
            with np.load(root / "features" / "shard_00000.npz") as arrays:
                self.assertIn("filled_token_ids", arrays.files)

    def test_audit_trains_and_reports_separate_r_e_horizons(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = ShardWriter(root, shard_rows=16)
            edge_rows = []
            rng = np.random.default_rng(7)
            for question in range(10):
                entries = []
                for step in range(5):
                    sid = f"q{question}_s{step}"
                    mask = np.asarray([(pos + step) % 3 != 0 for pos in range(8)])
                    valid = np.ones(8, dtype=bool)
                    row = dict(
                        x=rng.normal(size=(8, 22)).astype(np.float16),
                        mask=mask, valid=valid,
                        confidence=rng.uniform(.2, .9, size=8).astype(np.float32),
                        changed=np.zeros(8, dtype=bool),
                        content=rng.normal(size=(8, 4)).astype(np.float16),
                        native_token_ids=np.arange(8, dtype=np.int32),
                        filled=np.arange(8, dtype=np.int32) + 100,
                        candidate_token_ids=np.arange(8, dtype=np.int32) + step,
                        topk_token_ids=np.ones((8, 2), dtype=np.int32),
                        topk_logits=np.ones((8, 2), dtype=np.float16),
                        meta={"state_id": sid, "question_id": str(question)},
                    )
                    writer.append(row)
                    entries.append({"state_id": sid, "mask": mask,
                                    "valid": valid,
                                    "candidate_token_ids": row["candidate_token_ids"]})
                for step in range(4):
                    action = "E" if step == 2 else "R"
                    src, dst = entries[step], entries[step + 1]
                    valid = src["valid"] & dst["valid"]
                    if action == "R":
                        valid &= src["mask"]
                    edge_rows.append({
                        "src_state_id": src["state_id"], "dst_state_id": dst["state_id"],
                        "action": action, "valid": valid.tolist(),
                        "changed_valid": (valid if action == "R" else np.zeros(8, bool)).tolist(),
                        "new_suffix": (np.ones(8, bool) if action == "E" else np.zeros(8, bool)).tolist(),
                    })
            writer.flush()
            (root / "edges.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in edge_rows), encoding="utf-8")
            (root / "capture_manifest.json").write_text(json.dumps({"schema_version": "test"}),
                                                          encoding="utf-8")
            args = SimpleNamespace(device=torch.device("cpu"), updates=2, seeds=[42],
                                   split_seed=42, batch_size=2)
            report = audit_capture(root, args)
            self.assertEqual(report["data"]["edge_action_counts"], {"R": 30, "E": 10})
            self.assertIn("R", report["arms"]["latent_dynamics_h1_h3"])
            self.assertIn("E", report["arms"]["latent_dynamics_h1_h3"])
            self.assertIn("native_final_hidden", report["feature_ablation"])


if __name__ == "__main__":
    unittest.main()
