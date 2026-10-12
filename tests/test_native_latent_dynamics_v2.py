import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from native_latent_dynamics_v2 import (
    Edge, NativeLatentDynamics, State, audit_native_contract, build_paths,
    fit_scalers, load_teacher, pack_state, sanitize_capture, split_questions, supervised_loss, train_and_audit,
    main,
)
from native_latent_rules_v2 import (
    align_frame, constrained_commit, initial_segment_mask, next_geometry, proposal_span, scheduler_region,
)


def make_state(sid, qid, start=96, segment=0, before=None, memory=None, forward=1):
    frame = (start + 7) // 32 * 32
    geometry = np.array([start, frame, frame, 8, segment], np.int64)
    before = initial_segment_mask(torch.from_numpy(geometry[None]))[0].numpy().astype(np.float32) if before is None else before.copy()
    region = scheduler_region(torch.from_numpy(before[None]), torch.from_numpy(geometry[None]))[0].numpy()
    event = np.zeros(32, np.float32)
    event[np.flatnonzero(before.astype(bool) & region)[0]] = 1
    mask = before * (1 - event)
    native_active = scheduler_region(torch.from_numpy(mask[None]), torch.from_numpy(geometry[None]), True)[0].numpy()
    span = proposal_span(torch.from_numpy(geometry[None]))[0].numpy()
    active = native_active & span
    context = np.array([start/4096, (frame//32-start//32)/2, forward/16, forward/16,
                        np.flatnonzero(active)[0]/31 if active.any() else -1, 0, 0], np.float32)
    return State(sid, qid, np.full((32, 8), forward*.01, np.float32),
                 np.zeros(4096, np.float32) if memory is None else memory.copy(),
                 mask, event, active.astype(np.float32), context, True,
                 {"native_unmask_forward_index": forward, "refine_index": forward-1}, geometry, before,
                 native_active.astype(np.float32), region.astype(np.float32), np.ones(32, np.float32))


def fixture():
    states, edges = {}, []
    for q in range(20):
        qid = f"q{q:02}"
        a = make_state(qid+"a", qid)
        b = make_state(qid+"b", qid, before=a.mask, memory=a.memory, forward=2)
        c = make_state(qid+"c", qid, start=104, segment=1, memory=np.full(4096, .1, np.float32))
        d = make_state(qid+"d", qid, start=104, segment=1, before=c.mask, memory=c.memory, forward=2)
        for s in (a, b, c, d):
            s.u += q * .01
            states[s.sid] = s
        edges.extend((Edge(a.sid, b.sid, 0), Edge(b.sid, c.sid, 1), Edge(c.sid, d.sid, 0)))
    return states, edges


class NativeLatentDynamicsV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_split_and_horizon(self):
        states, edges = fixture()
        split = split_questions([s.qid for s in states.values()], 9)
        self.assertEqual(split, split_questions(split, 9))
        self.assertEqual(set(split.values()), {"train", "validation", "test"})
        paths = build_paths(states, edges, split, "test", 3)
        self.assertTrue(any([e.action for e in chain] == [0, 1, 0] for _, chain in paths))

    def test_physical_progress_can_be_outside_proposal(self):
        geom = torch.tensor([[1, 0, 0, 8, 0]])
        before = torch.zeros(1, 32)
        before[:, 8:] = 1
        region = scheduler_region(before, geom)
        score = torch.full((1, 32), -9.)
        score[:, 9] = -1.
        event, eligible = constrained_commit(score, before, region)
        self.assertEqual(event.sum().item(), 1)
        self.assertEqual(event[0, 9].item(), 1)
        self.assertEqual(event[0, 8].item(), 0)
        self.assertEqual(eligible.sum().item(), 8)
        self.assertFalse(bool(proposal_span(geom)[0, 9]))

    def test_absolute_frame_alignment(self):
        geom = torch.tensor([[120, 96, 96, 8, 3]])
        child = next_geometry(geom, torch.tensor([1]))
        self.assertEqual(child.tolist(), [[128, 128, 128, 8, 4]])
        aligned, overlap = align_frame(torch.ones(1, 32, 8), geom, child)
        self.assertEqual(aligned.sum().item(), 0)
        self.assertFalse(overlap.any())
        aligned, overlap = align_frame(torch.ones(1, 32, 8), geom, geom)
        self.assertTrue(overlap.all())
        self.assertEqual(aligned.sum().item(), 256)

    def test_cross_frame_transition_preserves_source_conditioning_and_positions(self):
        s = make_state("a", "q", start=120, segment=3)
        scales = {"u_mean": np.zeros(8), "u_std": np.ones(8),
                  "memory_mean": np.zeros(4096), "memory_std": np.ones(4096)}
        packed = pack_state(s, scales, "cpu")
        model = NativeLatentDynamics(8, 4096, 16, 1, "native_only")
        a, b = torch.randn(1, 32, 16), torch.randn(1, 32, 16)
        za = model.transition(a, torch.tensor([1]), packed)
        zb = model.transition(b, torch.tensor([1]), packed)
        self.assertGreater((za-zb).abs().max().item(), 1e-3)
        self.assertGreater(za.std(dim=1).mean().item(), 1e-4)

    def test_R_hard_rules_and_identity_readout(self):
        s = make_state("a", "q")
        scalers = {"u_mean": np.zeros(8), "u_std": np.ones(8),
                   "memory_mean": np.zeros(4096), "memory_std": np.ones(4096)}
        packed = pack_state(s, scalers, "cpu")
        for feature_set in ("native_only", "native_structure", "full"):
            model = NativeLatentDynamics(8, 4096, 16, 1, feature_set)
            torch.nn.init.zeros_(model.event_out.weight)
            torch.nn.init.constant_(model.event_out.bias, -8.)
            _, pred = model.step(model.encode(packed), torch.tensor([0]), packed)
            self.assertTrue(torch.equal(pred["memory"], packed["memory"]))
            self.assertTrue(torch.equal(pred["u"], packed["u"]))
            self.assertTrue(torch.equal(pred["geometry"], packed["geometry"]))
            self.assertTrue(torch.all(pred["mask"] <= packed["mask"] + 1e-6))
            self.assertGreaterEqual(pred["event"].sum().item(), 1)
            self.assertEqual((pred["event"] * (1-pred["eligible"])).sum().item(), 0)
            self.assertAlmostEqual(pred["context"][0, 2].item(), 2/16)
            self.assertFalse(any("token" in key or "vocab" in key for key in pred))

    def test_loss_and_predicted_state_feedback(self):
        states, edges = fixture()
        split = split_questions([s.qid for s in states.values()])
        scalers = fit_scalers(states, split)
        model = NativeLatentDynamics(8, 4096, 16, 1)
        edge = edges[0]
        source, target = (pack_state(states[sid], scalers, "cpu") for sid in (edge.src, edge.dst))
        z, pred = model.step(model.encode(source), torch.tensor([0]), source)
        loss, _ = supervised_loss(pred, target, torch.tensor([True]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.u_out[-1].weight.grad).all())
        current = {key: pred[key] for key in source}
        _, next_pred = model.step(z, torch.tensor([1]), current)
        self.assertEqual(next_pred["geometry"][0, 0].item(), source["geometry"][0, 0].item()+8)

    def test_native_capture_audit_rejects_corruption(self):
        states, edges = fixture()
        self.assertEqual(audit_native_contract(states, edges)["status"], "passed")
        states[edges[0].dst].event[:] = 0
        with self.assertRaisesRegex(ValueError, "physical event"):
            audit_native_contract(states, edges)

    def test_legacy_capture_rejected_not_silently_upgraded(self):
        s = make_state("a", "q")
        s.geometry = None
        with self.assertRaisesRegex(ValueError, "Fresh native-rules"):
            pack_state(s, {}, "cpu")

    def test_tiny_train_package_all_variants(self):
        states, edges = fixture()
        split = split_questions([s.qid for s in states.values()], 9)
        args = SimpleNamespace(train_device="cpu", horizon=3, latent_dim=16,
                               model_layers=1, updates=3, batch_size=4, seed=9, cut_layer=24)
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            report, scalers = train_and_audit(args, states, edges, split, out)
            sanitize_capture(out, out, states, edges, split, scalers)
            reloaded, reloaded_edges = load_teacher(out)
            self.assertEqual(audit_native_contract(reloaded, reloaded_edges)["status"], "passed")
            self.assertEqual(len(reloaded), len(states))
            self.assertEqual(set(report["variants"]), {"native_only", "native_structure", "full"})
            self.assertIn("sequences", report["variants"]["full"]["test"])
            ckpt = torch.load(out / "model_full.pt", weights_only=True)
            self.assertIn("normalization", ckpt)
            self.assertIn("model_config", ckpt)
            with np.load(out / "teacher_features/latent_0000.npz") as arr:
                self.assertTrue({"geometry", "mask_before_forward", "proposal_span", "valid_positions"}.issubset(arr.files))

    def test_failed_replay_preserves_features_and_source(self):
        states, edges = fixture()
        split = split_questions([s.qid for s in states.values()])
        with TemporaryDirectory() as tmp:
            source, out = Path(tmp)/"source", Path(tmp)/"output"
            source.mkdir()
            sanitize_capture(source, source, states, edges, split, fit_scalers(states, split))
            argv = ["test", "--capture-input", str(source), "--out-dir", str(out),
                    "--num-questions", "20", "--train-device", "cpu"]
            with patch("sys.argv", argv), patch("native_latent_dynamics_v2.train_and_audit", side_effect=RuntimeError("synthetic training failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic training"):
                    main()
            self.assertTrue((source/"teacher_features/states.jsonl").is_file())
            self.assertTrue((out/"teacher_features/states.jsonl").is_file())
            self.assertTrue(out.with_suffix(".zip").is_file())
            status = json.loads((out/"run_status.json").read_text())
            self.assertEqual(status["status"], "partial")
            self.assertTrue(status["sanitized_features_preserved"])

    def test_terminal_edge_has_no_fabricated_hidden_or_token_payload(self):
        states, edges = fixture()
        split = split_questions([s.qid for s in states.values()])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"src_state_id": "q00d", "question_id": "q00", "action": "E",
                   "candidate_contains_eos": True, "terminal_reason": "eos_before_observable_snapshot",
                   "hidden_available": False, "candidate_token_ids": [151645], "native_stats": {"secret": 1}}
            (root/"terminal_transitions.jsonl").write_text(json.dumps(row)+"\n")
            sanitize_capture(root, root, states, edges, split, fit_scalers(states, split))
            clean = json.loads((root/"teacher_features/terminal_transitions.jsonl").read_text())
            self.assertFalse(clean["hidden_available"])
            self.assertTrue(clean["native_terminal"])
            self.assertNotIn("candidate_token_ids", clean)
            self.assertNotIn("native_stats", clean)


if __name__ == "__main__":
    unittest.main()
