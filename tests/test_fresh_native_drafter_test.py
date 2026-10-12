import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh_native_drafter_test import (
    MASK_ID, ShardWriter, _make_edge, _state_row, audit_capture, load_capture, collect,
)
from native_elysia_graph import NativeEosWithoutSnapshot
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


def native_structure(**overrides):
    # Exercise the exact instrumentation helper without importing a 1.5B
    # remote-code model and all its optional Transformer integrations.
    source = Path(__file__).resolve().parents[1] / "Fast_dLLM_v2_1_5B" / "modeling.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                      and node.name == "native_v2_physical_structure")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    values = dict(tokens=torch.tensor([10] * 56 + [MASK_ID] * 8),
                  mask_before_forward=torch.tensor([False] * 24 + [True] * 8),
                  physical_block_start=32, proposal_start=50, proposal_length=8,
                  block_size=32, small_block_size=8, scheduler_start=24,
                  generated_start=50, mask_id=MASK_ID, stop_token=151645)
    values.update(overrides)
    return namespace["native_v2_physical_structure"](**values)


def fake_v2_snapshot():
    snapshot = fake_snapshot()
    tokens = torch.tensor([10] * 56 + [MASK_ID] * 8)
    tokens[58] = 20  # physical position 26 is outside logical proposal 18..25.
    structure = native_structure(tokens=tokens)
    snapshot.update(
        proposal_token_ids_before_fill=[10] * 6 + [MASK_ID] * 2,
        proposal_token_ids_after_fill=[10] * 6 + [21, 22],
        proposal_mask_before_fill=[False] * 6 + [True] * 2,
        wm_v2_native_latent_block=np.ones((32, 4)).tolist(),
        wm_v2_proposal_start=50, wm_v2_proposal_length=8,
        wm_v2_physical_block_start=32, wm_v2_prefix_kv_length=32,
        wm_v2_prefix_kv_summary=np.zeros((4, 2, 2, 2, 128)).tolist(),
        wm_v2_prefix_kv_valid=True, wm_v2_context=[.1, 0, .1, .1, 24 / 31],
        wm_v2_memory_layer_indices=[0, 7, 15, 23],
        wm_v2_native_refine_available=structure["refine_available"],
        wm_v2_native_eos_ready=structure["eos_ready"],
        wm_v2_valid_positions=structure["valid_positions"].tolist(),
        draft_passes_elapsed=2, draft_latency_elapsed_ms=1., hidden_state_forward_pass=2,
        wm_v2_forward_pass_breakdown={"prefill": 1, "cache_update": 0, "denoising": 1, "fill": 0, "total": 2},
    )
    for name, key in (("physical_mask", "mask"), ("physical_event", "event"),
                      ("mask_before_forward", "mask_before"), ("proposal_span", "proposal_span"),
                      ("native_active_region", "active"), ("forward_region", "forward_region")):
        snapshot["wm_v2_" + name] = structure[key].tolist()
    return snapshot


class FreshNativeDrafterTests(unittest.TestCase):
    def test_v2_capture_preserves_terminal_e_without_fabricating_hidden(self):
        class Tokenizer:
            eos_token_id = 151645
            def apply_chat_template(self, *args, **kwargs):
                return list(range(50))
        class Runner:
            def __init__(self, *args):
                self.calls = 0
            def segment(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [fake_v2_snapshot()]
                raise NativeEosWithoutSnapshot([151645], {
                    "native_termination_reason": "eos",
                    "forward_pass_breakdown": {"prefill": 1, "denoising": 1, "total": 2},
                })
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(out_dir=Path(folder) / "out", seed=42, num_questions=1,
                tokenizer_name="fake", dllm_dir=Path(folder), drafter_device=0, raw_top_k=1,
                physical_block_size=32, small_block_size=8, extend_size=8, threshold=.5,
                max_refinement_steps=3, max_proposal_tokens=64, shard_rows=2,
                wm_v2_capture_native_latent=True)
            with patch("fresh_native_drafter_test.load_dataset", return_value=[{"question": "q"}]), \
                 patch("fresh_native_drafter_test.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                 patch("fresh_native_drafter_test._load_drafter", return_value=FakeModel()), \
                 patch("fresh_native_drafter_test.NativeElysiaRunner", Runner), \
                 patch("fresh_native_drafter_test.torch.cuda.is_available", return_value=False):
                manifest = collect(args)
            terminal = json.loads((args.out_dir / "terminal_transitions.jsonl").read_text())
            question = json.loads((args.out_dir / "questions.jsonl").read_text())
            self.assertEqual(manifest["terminal_counts"], {"eos_before_observable_snapshot": 1})
            self.assertEqual(manifest["state_count"], 1)
            self.assertEqual(manifest["terminal_transition_count"], 1)
            self.assertEqual(question["completion_tokens_collected"], 9)
            self.assertEqual(terminal["src_state_id"], "q0_b00_r00")
            self.assertEqual(terminal["action"], "E")
            self.assertFalse(terminal["hidden_available"])
            self.assertNotIn("candidate_token_ids", terminal)

    def test_full_physical_event_and_next_region_survive_logical_crop(self):
        snapshot = fake_v2_snapshot()
        class NoEmbeddingModel:
            def get_input_embeddings(self):
                raise AssertionError("V2 must not compute token embeddings")
        row = _state_row(snapshot, NoEmbeddingModel(), 50, 3, 8, 151645, 4)
        self.assertEqual(np.flatnonzero(row["physical_event"]).tolist(), [26])
        self.assertEqual(np.flatnonzero(row["native_active_region"]).tolist(), list(range(24, 32)))
        self.assertEqual(row["geometry"].tolist(), [50, 32, 32, 8, 3])
        self.assertNotIn("x", row)
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder), shard_rows=1)
            writer.append({**row, "meta": {"state_id": "s", "question_id": "q"}})
            with np.load(Path(folder) / "features/shard_00000.npz") as data:
                self.assertIn("mask_before_forward", data.files)
                self.assertIn("forward_active_region", data.files)
                self.assertEqual(data["geometry"].dtype, np.dtype("int64"))
                self.assertNotIn("native_token_ids", data.files)
                self.assertNotIn("topk_logits", data.files)
                self.assertNotIn("content", data.files)

    def test_v2_omitted_event_outside_proposal_is_rejected(self):
        snapshot = fake_v2_snapshot()
        snapshot["wm_v2_physical_event"][26] = False
        with self.assertRaisesRegex(ValueError, "same-forward mask resolution"):
            _state_row(snapshot, object(), 50, 0, 8, 151645, 4)

    def test_native_region_empty_when_proposal_done_despite_physical_tail_masks(self):
        tokens = torch.tensor([10] * 58 + [MASK_ID] * 6)
        structure = native_structure(tokens=tokens)
        self.assertFalse(structure["refine_available"])
        self.assertFalse(structure["active"].any())
        self.assertTrue(structure["mask"].any())
        self.assertEqual(torch.nonzero(structure["event"]).flatten().tolist(), [24, 25])

    def test_native_eos_waits_for_preceding_masks(self):
        tokens = torch.tensor([10] * 56 + [MASK_ID] * 8)
        tokens[57] = 151645
        structure = native_structure(tokens=tokens)
        self.assertFalse(structure["eos_ready"])
        self.assertTrue(structure["refine_available"])
        tokens[56] = 20
        structure = native_structure(tokens=tokens)
        self.assertTrue(structure["eos_ready"])
        self.assertFalse(structure["active"].any())

    def test_native_partial_proposal_requires_older_positions_committed(self):
        tokens = torch.tensor([10] * 32 + [MASK_ID] * 32)
        tokens[33] = 20
        structure = native_structure(tokens=tokens, mask_before_forward=torch.ones(32, dtype=torch.bool),
                                     proposal_start=29, scheduler_start=0, generated_start=29)
        self.assertEqual(int(structure["proposal_span"].sum()), 5)
        tokens[30] = MASK_ID
        with self.assertRaisesRegex(RuntimeError, "missing proposal positions"):
            native_structure(tokens=tokens, mask_before_forward=torch.ones(32, dtype=torch.bool),
                             proposal_start=29, scheduler_start=0, generated_start=29)

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

    def test_legacy_change_labels_are_rebuilt_from_candidate_ids(self):
        # Force immediate flush: later in-memory mutations cannot repair disk.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = ShardWriter(root, shard_rows=1)
            base = _state_row(fake_snapshot(), FakeModel(), 50, 0, 8, 151645, 4)
            for sid, token in (("a", 30), ("b", 31)):
                candidate = base["candidate_token_ids"].copy()
                candidate[1] = token
                writer.append({**base, "candidate_token_ids": candidate,
                               "meta": {"state_id": sid, "question_id": "q0"}})
            (root / "edges.jsonl").write_text(json.dumps({
                "src_state_id": "a", "dst_state_id": "b", "action": "R",
                "valid": [True] * 8, "changed_valid": [True] * 8,
                "new_suffix": [False] * 8}) + "\n")
            states, _, _ = load_capture(root)
            self.assertTrue(states["b"].changed[1])
            self.assertEqual(int(states["b"].changed.sum()), 1)

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
