import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from native_latent_dynamics_v2 import Edge, State, fit_scalers
from native_latent_rules_v2 import (
    initial_segment_mask, next_geometry, proposal_span, scheduler_region,
)
from score_latent_drafter import (
    ScoreEncoder, ScoreTransition, _state_from_batch, loss_for_step,
    pack_states, target_change_labels, train_and_evaluate,
)


def make_state(sid, qid, geometry, before=None, u_offset=0.0, event_index=None,
               memory=None, refine_index=0, forward_index=1):
    geometry_t = torch.as_tensor(np.asarray(geometry, dtype=np.int64)[None])
    span = proposal_span(geometry_t)[0].numpy()
    if before is None:
        before = initial_segment_mask(geometry_t)[0].numpy().astype(np.float32)
    region = scheduler_region(torch.as_tensor(before[None]), geometry_t)
    eligible = np.flatnonzero((before >= .5) & region[0].numpy())
    event = np.zeros(32, dtype=np.float32)
    if len(eligible):
        idx = int(eligible[0] if event_index is None else event_index)
        event[idx] = 1.0
    mask = before * (1.0 - event)
    active = scheduler_region(torch.as_tensor(mask[None]), geometry_t, logical_only=True)[0].numpy()
    valid = np.ones(32, dtype=np.float32)
    valid[~span] = 1.0  # The physical frame contains valid native residual rows.
    rng = np.random.default_rng(abs(hash(sid)) % (2**32))
    u = rng.normal(size=(32, 12)).astype(np.float32) + u_offset
    if memory is None:
        memory = rng.normal(size=(4, 2, 2, 2, 128)).astype(np.float32).reshape(-1) * .01
    frame = int(geometry[1]); start = int(geometry[0])
    context = np.array([start / 4096, float(geometry[4]) / 7,
                        2 / 16, 1 / 16,
                        (np.flatnonzero(active)[0] / 31 if active.any() else -1),
                        0, 0], dtype=np.float32)
    return State(
        sid=sid, qid=qid, u=u, memory=memory, mask=mask.astype(np.float32),
        event=event, active=active.astype(np.float32), context=context,
        memory_valid=True,
        meta={"native_terminal": False, "native_unmask_forward_index": forward_index,
              "refine_index": refine_index},
        geometry=np.asarray(geometry, dtype=np.int64),
        mask_before=before.astype(np.float32), native_active=active.astype(np.float32),
        forward_active=region[0].numpy().astype(np.float32), valid=valid,
    )


class ScoreLatentDrafterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.geometry = np.array([96, 96, 96, 8, 0], dtype=np.int64)
        self.root = make_state("root", "q0", self.geometry)
        self.child_r = make_state("child_r", "q0", self.geometry,
                                  before=self.root.mask, u_offset=.1)
        e_geometry = next_geometry(torch.as_tensor(self.geometry[None]), torch.tensor([1]))[0].numpy()
        self.child_e = make_state("child_e", "q0", e_geometry, u_offset=.2)
        self.states = {s.sid: s for s in (self.root, self.child_r, self.child_e)}
        self.split = {"q0": "train"}
        self.scalers = fit_scalers(self.states, self.split)

    def _state_batch(self, sid):
        return pack_states([sid], self.states, self.scalers, "cpu")

    def test_encoder_and_transition_shapes_and_hard_native_rules(self):
        for feature_set in ("native_only", "native_structure", "full_context"):
            encoder = ScoreEncoder(12, 16, layers=1, feature_set=feature_set)
            transition = ScoreTransition(16, layers=1, feature_set=feature_set)
            batch = self._state_batch("root")
            z, memory_z = encoder(batch)
            self.assertEqual(tuple(z.shape), (1, 32, 16))
            self.assertEqual(tuple(memory_z.shape), (1, 16))
            state = _state_from_batch(batch, (z, memory_z))

            refined, refine_aux = transition(state, torch.tensor([0]))
            self.assertTrue(torch.equal(refined["geometry"], batch["geometry"]))
            self.assertTrue(torch.all(refined["mask"] <= batch["mask"] + 1e-6))
            # The hard rule forces one commit, but summing float32 straight-through
            # values can round a few ulps below exactly 1.0.
            self.assertGreaterEqual(float(refined["event"].sum()), 1.0 - 1e-6)
            self.assertEqual(float((refined["event"] * (1 - refine_aux["eligible"])).sum()), 0.0)

            extended, extend_aux = transition(state, torch.tensor([1]))
            self.assertEqual(int(extended["geometry"][0, 0]), int(self.geometry[0] + 8))
            expected_before = initial_segment_mask(extended["geometry"]).float()
            self.assertTrue(torch.equal(extended["mask_before"], expected_before))
            self.assertTrue(torch.all(extended["event"] <= extend_aux["eligible"] + 1e-6))

    def test_change_labels_align_by_absolute_position_on_extend(self):
        labels, valid = target_change_labels(
            self.root, self.child_e, self.scalers, [0.1, 0.2], "cpu")
        self.assertEqual(tuple(labels.shape), (1, 32))
        self.assertEqual(tuple(valid.shape), (1, 32))
        # E's physical frame is still 96 here, so all physical rows overlap.
        self.assertEqual(int(valid.sum()), 32)

    def test_transition_loss_has_gradient_and_uses_predicted_source(self):
        encoder = ScoreEncoder(12, 16, layers=1, feature_set="full_context")
        transition = ScoreTransition(16, layers=1, feature_set="full_context")
        source_batch = self._state_batch("root")
        target_batch = self._state_batch("child_r")
        source_z, source_mem = encoder(source_batch)
        state = _state_from_batch(source_batch, (source_z, source_mem))
        predicted, aux = transition(state, torch.tensor([0]))
        with torch.no_grad():
            target_z, target_mem = encoder(target_batch)
        target_batch["_memory_z"] = target_mem
        target_batch["_state_obj"] = self.child_r
        loss, pieces = loss_for_step(
            predicted, aux, target_batch, target_z, self.root,
            source_z + .05, source_z, self.scalers, [0.1, 0.2], 1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in pieces.values()))
        loss.backward()
        self.assertIsNotNone(transition.delta[-1].weight.grad)
        self.assertTrue(torch.isfinite(transition.delta[-1].weight.grad).all())

    def test_tiny_train_eval_runs_all_ablation_and_h3_rollout(self):
        states, edges = {}, []
        for q in range(20):
            qid = f"train_q{q:02d}"
            root = make_state(f"{qid}_root", qid, self.geometry, u_offset=q * .01)
            r1 = make_state(f"{qid}_r1", qid, self.geometry, before=root.mask,
                            u_offset=q * .01 + .1, memory=root.memory,
                            refine_index=1, forward_index=2)
            e_geometry = next_geometry(torch.as_tensor(r1.geometry[None]), torch.tensor([1]))[0].numpy()
            extended = make_state(f"{qid}_e", qid, e_geometry,
                                  u_offset=q * .01 + .2, refine_index=0, forward_index=1)
            r2 = make_state(f"{qid}_r2", qid, e_geometry, before=extended.mask,
                            u_offset=q * .01 + .3, memory=extended.memory,
                            refine_index=1, forward_index=2)
            for state in (root, r1, extended, r2):
                states[state.sid] = state
            edges.extend((Edge(root.sid, r1.sid, 0), Edge(r1.sid, extended.sid, 1),
                          Edge(extended.sid, r2.sid, 0)))
        args = SimpleNamespace(
            train_device="cpu", horizon=3, paths_per_root=3, eval_paths=16,
            test_paths=24, latent_dim=16, encoder_layers=1, transition_layers=1,
            updates=3, batch_size=2, seed=17, learning_rate=2e-4,
            ema_momentum=.9, log_every=99,
        )
        split = {qid: ("train" if int(qid[-2:]) < 12 else
                       "validation" if int(qid[-2:]) < 16 else "test")
                 for qid in {state.qid for state in states.values()}}
        with TemporaryDirectory() as tmp:
            report = train_and_evaluate(args, states, edges, split, self.scalers,
                                        Path(tmp))
            self.assertEqual(set(report["variants"]),
                             {"native_only", "native_structure", "full_context"})
            self.assertIn("H3_R", report["variants"]["full_context"]["test"]["by_horizon_action"])
            self.assertIn("feature_impact_vs_native_only", report)


if __name__ == "__main__":
    unittest.main()
