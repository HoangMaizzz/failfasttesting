"""CPU synthetic contracts for frozen D coordinates and independent D/V models.

Run directly or with ``python -B -m unittest discover -s tests
-p test_factorized_wm_models.py -v``. No dataset, checkpoint, GPU, or LLM is used.
Contract failures are intentional evidence of model defects, not expected failures.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from factorized_wm_models import (
    DirectOutcome,
    HiddenCompressor,
    Layout,
    LinearDynamics,
    Preprocessor,
    State,
    StructuredDynamics,
    TabularVerifier,
    TokenDecoder,
    VerifierTransformer,
    drafter_loss,
    feature_view,
    materialize,
    pack,
    tabular_features,
    verifier_loss,
)


class DOnlyRow(dict):
    """Fail on label access even when the accessed value is subsequently ignored."""

    forbidden = {"accepted", "teacher", "margin", "current_verifier",
                 "teacher_features", "teacher_margin", "labels"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"D reconstruction read verifier target {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"D reconstruction read verifier target {key!r}")
        return super().get(key, default)

    def __contains__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"D reconstruction inspected verifier target {key!r}")
        return super().__contains__(key)


class FactorizedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1701)
        self.layout = Layout(hidden_dim=4, layers=3)
        # Orthogonal identity codes make decoder and OOV assertions unambiguous.
        codes = torch.eye(self.layout.code_dim)[:5].clone()
        self.prep = Preprocessor(
            self.layout, [-1, 1, 2, 3, 151665], codes,
            HiddenCompressor(raw_dim=16, layers=3, dim=4),
        )
        self.decoder = TokenDecoder(self.prep)
        self.prior = torch.randn(8, self.layout.size) * .05
        s = self.prior[:, self.layout.structure]
        s.zero_()
        s[:, 0] = 1
        s[:, 3:5] = 1
        s[:, 12] = 1
        s[:, 14] = 1
        self.prior[:, self.layout.native] = self.prep.token_codes(torch.tensor([151665]))
        self.prior[:, self.layout.stop] = self.prep.token_codes(torch.tensor([2]))

    def raw(self, uid="source", length=8, frontier=3, question="train", seed=31):
        rng = torch.Generator().manual_seed(seed)
        pos = torch.arange(length)
        stop = pos.remainder(3) + 1
        masked = (pos >= frontier) & (pos.remainder(2) == 1)
        native = torch.where(masked, 151665, stop)
        scalars = torch.zeros(length, 16)
        scalars[:, 0] = masked.float()
        scalars[:, 1] = (~masked).float()
        scalars[:, 3:5] = 1
        scalars[:frontier, 5] = .25
        scalars[:frontier, 6] = .375
        scalars[:, 7] = .2
        scalars[:, 8] = .3
        scalars[:, 9] = pos / length
        scalars[:, 10] = (pos + 2).remainder(16) / 16
        scalars[:, 11] = (pos + 2).remainder(8) / 8
        scalars[:, 12] = (pos >= frontier).float()
        scalars[:, 13] = .4
        scalars[:, 14] = masked.float()
        scalars[:, 15] = .5
        gaps = -torch.rand(length, 32, generator=rng) * 20
        surface = torch.cat([gaps, torch.randn(length, 4, generator=rng)], -1)
        teacher = torch.zeros(length, 35)
        teacher[:, 0] = .2
        teacher[:, 1] = .6
        teacher[:, 2] = pos.remainder(2).float()
        return SimpleNamespace(
            uid=uid, question=question, round_id=0, length=length,
            ids=torch.stack([native, stop], -1),
            hidden=torch.randn(length, 3, 16, generator=rng),
            surface=surface, scalars=scalars,
            context=torch.tensor([2 / 1024, length / 64, frontier / 64,
                                  1 / 3, .5, 16 / 64, 8 / 64, .7]),
            prefix_ids=torch.tensor([1, 2, 3, 2, 1]),
            topk_ids=(torch.arange(32).remainder(3) + 1).expand(length, -1).clone(),
            gaps=gaps, accepted=min(4, length), teacher_features=teacher,
            teacher_margin=teacher[:, 0].clone(),
        )

    def rows(self, *raw):
        prepared = self.prep.prepare({r.uid: r for r in raw}, "cpu")
        return [prepared[r.uid] for r in raw]

    def source(self, lengths=(8, 16), frontiers=None):
        if frontiers is None:
            frontiers = [3] * len(lengths)
        rows = self.rows(*(self.raw(f"source-{i}", n, f, seed=31 + i)
                           for i, (n, f) in enumerate(zip(lengths, frontiers))))
        return pack(rows, "cpu")

    def dynamics(self, kind="transformer", nonzero=True, representation="SHT"):
        if kind == "linear":
            model = LinearDynamics(self.prep, self.prior.clone())
            local, pooled = model.design(self.source((8,)))
            model.r_weight = torch.randn(local.shape[-1], self.layout.size) * .005
            model.e_weight = torch.randn(pooled.shape[-1], 8 * self.layout.size) * .005
        else:
            model = StructuredDynamics(
                self.prep, self.prior, kind=kind, representation=representation,
                width=16, layers=1, dropout=0,
            )
            if nonzero:
                # Default zero output weights would conceal rollout dependencies.
                with torch.no_grad():
                    model.output.weight.normal_(std=.04)
                    model.output.bias.normal_(std=.02)
        return model.eval()

    def identity_variants(self, state):
        """Change each identity channel independently, keeping D core/metadata fixed."""
        native, stop = state.x.clone(), state.x.clone()
        native[..., self.layout.native] = native[..., self.layout.native].roll(7, dims=-1)
        stop[..., self.layout.stop] = stop[..., self.layout.stop].roll(11, dims=-1)
        return {
            "prefix": replace(state, prefix=state.prefix + torch.linspace(-2, 3, 64)),
            "native_codes": replace(state, x=native),
            "stop_codes": replace(state, x=stop),
        }

    def configure_history_scaler(self):
        # Nonidentity scaling catches implementations that write raw history into x.
        self.prep.mean[-4:] = torch.tensor([.2, -.1, .4, .75])
        self.prep.std[-4:] = torch.tensor([.5, .25, .2, .5])

    def raw_history(self, state, model):
        history = state.x[..., self.layout.h + 32:self.layout.h + 36]
        return history * model.history_std + model.history_mean

    def snapshot(self, state):
        return State(*(value.detach().clone() for value in
                       (state.x, state.lengths, state.context, state.prefix)))

    def assert_state_equal(self, actual, expected):
        for name in ("x", "lengths", "context", "prefix"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                       rtol=0, atol=0, msg=name)

    def assert_finite_gradients(self, model, required=()):
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(grads, "backpropagation produced no parameter gradients")
        for name, grad in grads.items():
            self.assertTrue(bool(torch.isfinite(grad).all()), f"nonfinite gradient: {name}")
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads.values()))
        for name in required:
            self.assertIn(name, grads)
            self.assertGreater(float(grads[name].abs().sum()), 0, name)

    def assert_heads(self, heads, batch):
        self.assertEqual(set(heads), {"hazard", "tf", "probability", "margin"})
        for name, value in heads.items():
            self.assertEqual(tuple(value.shape), (batch, 64), name)
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(bool(torch.isfinite(value).all()), name)
        self.assertTrue(bool(((heads["probability"] >= 0) &
                              (heads["probability"] <= 1)).all()))
        self.assertTrue(bool((heads["margin"].abs() <= 1).all()))

    def test_layout_and_pack_cpu_shapes_copy_and_padding(self):
        l = self.layout
        self.assertEqual((l.h, l.size), (12, 128))
        self.assertEqual((l.surface.start, l.surface.stop), (12, 48))
        self.assertEqual((l.structure.start, l.structure.stop), (48, 64))
        self.assertEqual((l.native.start, l.native.stop), (64, 96))
        self.assertEqual((l.stop.start, l.stop.stop), (96, 128))
        rows = self.rows(self.raw("a", 8), self.raw("b", 64))
        saved = rows[0]["x"].clone()
        state = pack(rows, "cpu")
        self.assertEqual(tuple(state.x.shape), (2, 64, l.size))
        self.assertEqual(tuple(state.context.shape), (2, 8))
        self.assertEqual(tuple(state.prefix.shape), (2, 64))
        self.assertEqual(state.lengths.tolist(), [8, 64])
        self.assertEqual(state.x.device.type, "cpu")
        self.assertEqual(state.lengths.dtype, torch.long)
        self.assertEqual(float(state.x[0, 8:].abs().sum()), 0)
        self.assert_state_equal(state.to("cpu"), state)
        state.x[0, 0, 0] += 10
        torch.testing.assert_close(rows[0]["x"], saved, rtol=0, atol=0)

    def test_feature_views_keep_structure_and_do_not_mutate_source(self):
        state = self.source()
        before = self.snapshot(state)
        for rep in ("S", "H", "SH", "SHT"):
            with self.subTest(representation=rep):
                expected = state.x.clone()
                if rep == "S":
                    expected[..., :self.layout.h] = 0
                if rep == "H":
                    expected[..., self.layout.surface] = 0
                if rep != "SHT":
                    expected[..., self.layout.native] = 0
                    expected[..., self.layout.stop] = 0
                torch.testing.assert_close(feature_view(state, self.layout, rep), expected,
                                           rtol=0, atol=0)
        with self.assertRaises(ValueError):
            feature_view(state, self.layout, "invalid")
        self.assert_state_equal(state, before)

    def test_structured_identity_ablation_ignores_prefix_and_token_codes(self):
        source = self.source()
        before = self.snapshot(source)
        variants = self.identity_variants(source)
        actions = torch.tensor([0, 1])
        for kind in ("transformer", "mlp"):
            for rep in ("S", "H", "SH", "SHT"):
                model = self.dynamics(kind, representation=rep)
                with torch.no_grad():
                    # Also make the mask head sensitive to its learned inputs.
                    model.mask_head.weight.normal_(std=.04)
                    reference = model(source, actions)
                    gates = model.last_gates.clone()
                    for channel, changed in variants.items():
                        with self.subTest(kind=kind, representation=rep, channel=channel):
                            result = model(changed, actions)
                            # History's decoded STOP-change bit is known bookkeeping.
                            # Compare learned hidden/gaps/structure, excluding history4.
                            core = torch.cat([result.x[..., :self.layout.h + 32],
                                              result.x[..., self.layout.structure]], -1)
                            expected = torch.cat([reference.x[..., :self.layout.h + 32],
                                                  reference.x[..., self.layout.structure]], -1)
                            if rep == "SHT":
                                self.assertGreater(float((core - expected).abs().max()), 1e-6,
                                                   f"SHT must respond to {channel}")
                            else:
                                # Copied STOP/native blocks may change; learned D outputs may not.
                                torch.testing.assert_close(core, expected, rtol=0, atol=0)
                                torch.testing.assert_close(model.last_gates, gates, rtol=0, atol=0)
                            torch.testing.assert_close(result.prefix, changed.prefix,
                                                       rtol=0, atol=0)
        self.assert_state_equal(source, before)

    def test_verifier_identity_ablation_ignores_prefix_and_token_codes(self):
        source = self.source()
        variants = self.identity_variants(source)
        for rep in ("S", "H", "SH", "SHT"):
            model = VerifierTransformer(self.layout, representation=rep,
                                        width=16, layers=1, dropout=0).eval()
            with torch.no_grad():
                reference = model(source)
                gates = model.last_gates.clone()
                for channel, changed in variants.items():
                    with self.subTest(representation=rep, channel=channel):
                        result = model(changed)
                        self.assert_heads(result, 2)
                        if rep == "SHT":
                            difference = max(float((result[name] - reference[name]).abs().max())
                                             for name in result)
                            self.assertGreater(difference, 1e-6, f"SHT must respond to {channel}")
                        else:
                            for name in result:
                                torch.testing.assert_close(result[name], reference[name],
                                                           rtol=0, atol=0, msg=name)
                            torch.testing.assert_close(model.last_gates, gates, rtol=0, atol=0)

    def test_tabular_identity_ablation_excludes_prefix_and_token_codes(self):
        source = self.source()
        variants = self.identity_variants(source)
        for rep in ("S", "H", "SH", "SHT"):
            reference = torch.as_tensor(tabular_features(source, self.layout, rep))
            expected_prefix = (source.prefix if rep == "SHT" else
                               torch.zeros_like(source.prefix))
            torch.testing.assert_close(reference[..., -64:],
                                       expected_prefix[:, None].expand(-1, 64, -1),
                                       rtol=0, atol=0)
            for channel, changed in variants.items():
                with self.subTest(representation=rep, channel=channel):
                    result = torch.as_tensor(tabular_features(changed, self.layout, rep))
                    if rep == "SHT":
                        self.assertGreater(float((result - reference).abs().max()), 1e-6)
                    else:
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_identity_ablation_has_no_gradient_path_from_learned_outputs(self):
        for kind in ("transformer", "mlp", "verifier"):
            for rep in ("S", "H", "SH", "SHT"):
                with self.subTest(kind=kind, representation=rep):
                    source = self.source()
                    source = replace(source, x=source.x.clone().requires_grad_(),
                                     prefix=source.prefix.clone().requires_grad_())
                    if kind == "verifier":
                        model = VerifierTransformer(self.layout, representation=rep,
                                                    width=16, layers=1, dropout=0).eval()
                        loss = sum(value.square().mean() for value in model(source).values())
                    else:
                        model = self.dynamics(kind, representation=rep)
                        result = model(source, torch.tensor([0, 1]))
                        # Exclude deterministic token/history bookkeeping from this contract.
                        loss = result.x[..., :self.layout.h + 32].square().mean()
                    loss.backward()
                    self.assertTrue(bool(torch.isfinite(source.x.grad).all()))
                    if rep == "SHT":
                        self.assertIsNotNone(source.prefix.grad)
                        self.assertTrue(bool(torch.isfinite(source.prefix.grad).all()))
                        self.assertGreater(float(source.prefix.grad.abs().sum()), 0)
                        for block in (self.layout.native, self.layout.stop):
                            self.assertGreater(float(source.x.grad[..., block].abs().sum()), 0)
                    else:
                        if source.prefix.grad is not None:
                            self.assertEqual(float(source.prefix.grad.abs().sum()), 0)
                        for block in (self.layout.native, self.layout.stop):
                            self.assertEqual(float(source.x.grad[..., block].abs().sum()), 0)

    def test_raw_compressor_shape_reconstruction_and_backward(self):
        compressor = HiddenCompressor(raw_dim=16, layers=3, dim=4)
        raw = torch.randn(7, 3, 16, requires_grad=True)
        projected = compressor(raw)
        self.assertEqual(tuple(projected.shape), (7, 3, 4))
        self.assertTrue(projected.requires_grad)
        loss = compressor.reconstruction_loss(raw)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assert_finite_gradients(compressor, ("encoders.0.weight", "decoders.0.weight"))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_preparation_projects_with_frozen_compressor_and_masks_invalid_rows(self):
        raw = self.raw()
        raw.hidden.requires_grad_()
        raw.scalars[0, 3] = 0
        raw.scalars[1, 4] = 0
        before = raw.surface.clone()
        row = self.rows(raw)[0]
        expected = self.prep.compressor(raw.hidden).flatten(1).detach()
        expected[0] = 0
        torch.testing.assert_close(row["x"][:, :self.layout.h], expected)
        self.assertEqual(float(row["x"][1, self.layout.h:self.layout.h + 32].abs().sum()), 0)
        self.assertFalse(row["x"].requires_grad)
        self.assertIsNone(raw.hidden.grad)
        self.assertFalse(self.prep.compressor.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in self.prep.compressor.parameters()))
        torch.testing.assert_close(raw.surface, before, rtol=0, atol=0)

    def test_projection_targets_and_codes_stay_frozen_after_dynamics_update(self):
        raw = self.raw("target", 16, frontier=8)
        row = self.rows(raw)[0]
        saved_x = row["x"].clone()
        saved_codes = self.prep.codes.clone()
        saved_weights = {k: v.clone() for k, v in self.prep.compressor.state_dict().items()}
        source = self.source((8,))
        model = self.dynamics("mlp")
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        prediction = model(source, torch.tensor([1]))
        loss, _ = drafter_loss(prediction, [row], source, torch.tensor([1]),
                               model.decoder, self.layout)
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(self.rows(raw)[0]["x"], saved_x, rtol=0, atol=0)
        torch.testing.assert_close(self.prep.codes, saved_codes, rtol=0, atol=0)
        self.assertFalse(self.prep.codes.requires_grad)
        self.assertFalse(model.decoder.codes.requires_grad)
        for name, value in self.prep.compressor.state_dict().items():
            torch.testing.assert_close(value, saved_weights[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.prep.compressor.parameters()))

    def test_preprocessor_scaler_and_vocabulary_use_only_training_questions(self):
        train = [self.raw("train-a", 8), self.raw("train-b", 12, seed=32)]
        held = self.raw("held", 10, question="validation", seed=99)
        held.ids.fill_(999999)
        held.topk_ids.fill_(999999)
        held.hidden.mul_(1000)
        held.surface.fill_(10000)
        held.scalars[:, 3:5] = 0  # Held-out rows are never eligible for training.
        options = dict(question_ids=["train"], dim=4, updates=2, device="cpu", seed=19)
        only_train = Preprocessor.fit({r.uid: r for r in train}, **options)
        with_held = Preprocessor.fit({r.uid: r for r in [*train, held]}, **options)
        self.assertEqual(with_held.vocabulary.tolist(), [-1, 1, 2, 3, 151665])
        for name in ("mean", "std", "codes", "vocabulary"):
            torch.testing.assert_close(getattr(with_held, name), getattr(only_train, name),
                                       rtol=0, atol=0, msg=name)
        for name, value in with_held.compressor.state_dict().items():
            torch.testing.assert_close(value, only_train.compressor.state_dict()[name],
                                       rtol=0, atol=0)
        unscaled = Preprocessor(with_held.layout, with_held.vocabulary, with_held.codes,
                                with_held.compressor)
        prepared = unscaled.prepare({r.uid: r for r in train})
        values = torch.cat([prepared[r.uid]["x"][r.scalars[:, 3] > 0,
                           :with_held.layout.h + 36] for r in train])
        torch.testing.assert_close(with_held.mean, values.mean(0))
        torch.testing.assert_close(with_held.std, values.std(0, unbiased=False).clamp_min(.05))
        saved_mean, saved_std = with_held.mean.clone(), with_held.std.clone()
        with_held.prepare({held.uid: held})
        torch.testing.assert_close(with_held.mean, saved_mean, rtol=0, atol=0)
        torch.testing.assert_close(with_held.std, saved_std, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "No training questions"):
            Preprocessor.fit({held.uid: held}, **options)

    def test_preprocessor_restore_preserves_frozen_coordinates(self):
        raw = self.raw()
        restored = Preprocessor.restore(self.prep.save_payload())
        a, b = self.rows(raw)[0], restored.prepare({raw.uid: raw})[raw.uid]
        for name in ("x", "context", "prefix", "token_targets", "topk_classes"):
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in restored.compressor.parameters()))

    def test_oov_codes_decode_to_minus_one_and_count_as_wrong(self):
        raw = self.raw()
        raw.ids[:, 1] = torch.tensor([777, 888, 999, 777, 888, 999, 777, 888])
        raw.topk_ids.fill_(777)
        row = self.rows(raw)[0]
        prediction = self.decoder.ids(row["x"]).squeeze(-1)
        self.assertTrue(bool((row["token_targets"] == 0).all()))
        self.assertTrue(bool((row["topk_classes"] == 0).all()))
        self.assertEqual(prediction.tolist(), [-1] * raw.length)
        self.assertEqual(int((prediction == row["raw_stop_ids"]).sum()), 0,
                         "OOV accuracy must compare original IDs, not shared UNK classes")
        coverage = torch.isin(row["raw_stop_ids"], self.prep.vocabulary)
        self.assertEqual(int(coverage.sum()), 0)
        torch.testing.assert_close(row["raw_stop_ids"], raw.ids[:, 1], rtol=0, atol=0)

    def test_decoder_known_tokens_topk_shapes_and_differentiability(self):
        ids = torch.tensor([-1, 1, 2, 3, 151665])
        x = torch.zeros(2, 5, self.layout.size)
        x[..., self.layout.stop] = self.prep.token_codes(ids)
        x.requires_grad_()
        self.assertEqual(tuple(self.decoder.logits(x).shape), (2, 5, 5))
        self.assertEqual(tuple(self.decoder.ids(x, k=99).shape), (2, 5, 5))
        torch.testing.assert_close(self.decoder.ids(x).squeeze(-1), ids.expand(2, -1))
        self.decoder.logits(x).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertEqual(list(self.decoder.parameters()), [])

    def test_materialize_mixed_lengths_commit_stop_before_append_without_mutation(self):
        state = self.source((8, 56))
        saved = self.snapshot(state)
        prior = self.prior.clone()
        result, new = materialize(state, torch.tensor([0, 1]), self.layout, self.prior)
        self.assertEqual(result.lengths.tolist(), [8, 64])
        self.assertEqual(new.sum(-1).tolist(), [0, 8])
        torch.testing.assert_close(result.x[1, :56, self.layout.native],
                                   state.x[1, :56, self.layout.stop], rtol=0, atol=0)
        torch.testing.assert_close(result.x[1, :56, self.layout.stop],
                                   state.x[1, :56, self.layout.stop], rtol=0, atol=0)
        torch.testing.assert_close(result.x[1, 56:, :self.layout.h + 32],
                                   prior[:, :self.layout.h + 32], rtol=0, atol=0)
        torch.testing.assert_close(result.x[1, 56:, self.layout.native],
                                   prior[:, self.layout.native], rtol=0, atol=0)
        torch.testing.assert_close(result.x[1, 56:, self.layout.stop],
                                   prior[:, self.layout.stop], rtol=0, atol=0)
        s = result.x[1, :56, self.layout.structure]
        self.assertEqual(float(s[:, [0, 2, 12, 14]].abs().sum()), 0)
        self.assertTrue(bool((s[:, 1] == 1).all()))
        self.assert_state_equal(state, saved)
        torch.testing.assert_close(self.prior, prior, rtol=0, atol=0)
        torch.testing.assert_close(result.prefix, state.prefix, rtol=0, atol=0)
        self.assertNotEqual(result.prefix.data_ptr(), state.prefix.data_ptr())

    def test_materialize_refine_copies_content_with_known_metadata_and_age_changes(self):
        source = self.source((8,))
        result, new = materialize(source, torch.tensor([0]), self.layout, self.prior)
        self.assertEqual(int(new.sum()), 0)
        expected = source.x.clone()
        s = expected[..., self.layout.structure]
        s[:, :8, 5:7] += 1 / 8
        pos = torch.arange(64).float()
        s[0, :, 9] = pos / 8
        s[0, :, 10] = (pos + 2).remainder(16) / 16
        s[0, :, 11] = (pos + 2).remainder(8) / 8
        s[0, :, 12] = (pos >= 3).float()
        s[0, :, 15] = source.context[0, 4]
        expected[:, 8:] = 0
        torch.testing.assert_close(result.x, expected, rtol=0, atol=0)
        expected_context = source.context.clone()
        expected_context[:, 3] += 1 / 3
        torch.testing.assert_close(result.context, expected_context, rtol=0, atol=0)
        torch.testing.assert_close(result.prefix, source.prefix, rtol=0, atol=0)

    def test_materialize_commit_is_differentiable_in_actual_stop_vectors(self):
        source = self.source((8,))
        source = replace(source, x=source.x.detach().clone().requires_grad_())
        result, _ = materialize(source, torch.tensor([1]), self.layout, self.prior)
        result.x[0, :8, self.layout.native].sum().backward()
        torch.testing.assert_close(source.x.grad[0, :8, self.layout.stop],
                                   torch.ones(8, self.layout.code_dim), rtol=0, atol=0)
        self.assertEqual(float(source.x.grad[0, :8, self.layout.native].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(source.x.grad).all()))

    def test_length_cap_and_invalid_action_rejected(self):
        for kind in ("materialize", "transformer", "mlp", "linear"):
            with self.subTest(kind=kind):
                model = None if kind == "materialize" else self.dynamics(kind)

                def advance(state, action):
                    if model is None:
                        return materialize(state, action, self.layout, self.prior)[0]
                    return model(state, action)

                self.assertEqual(advance(self.source((64,)), torch.tensor([0])).lengths.tolist(), [64])
                self.assertEqual(advance(self.source((56,)), torch.tensor([1])).lengths.tolist(), [64])
                for n in (57, 64):
                    with self.assertRaisesRegex(ValueError, "64"):
                        advance(self.source((n,)), torch.tensor([1]))
                for invalid in (-1, 2):
                    with self.assertRaisesRegex(ValueError, "R/E"):
                        advance(self.source((8,)), torch.tensor([invalid]))
        with self.assertRaisesRegex(ValueError, "64"):
            self.rows(self.raw(length=65))

    def test_zero_initialized_structured_outputs_use_extension_prior(self):
        source = self.source((8,))
        for kind in ("transformer", "mlp"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind, nonzero=False)
                result = model(source, torch.tensor([1]))
                torch.testing.assert_close(result.x[0, 8:16, :self.layout.h + 32],
                                           self.prior[:, :self.layout.h + 32], rtol=0, atol=0)
                expected_history = (-model.history_mean / model.history_std).expand(8, -1)
                torch.testing.assert_close(result.x[0, 8:16, self.layout.h + 32:self.layout.h + 36],
                                           expected_history, rtol=0, atol=0)
                torch.testing.assert_close(result.x[0, 8:16, self.layout.stop],
                                           self.prior[:, self.layout.stop], rtol=0, atol=0)
                self.assertEqual(float(result.x[0, 16:].abs().sum()), 0)

    def test_structured_extend_copies_all_old_hidden_gap_rows_in_mixed_batches(self):
        source = self.source((8, 16), (3, 4))
        # Invalid/cache flags and nonzero ages must survive E for the former frontier too.
        s = source.x[..., self.layout.structure]
        s[0, 3:8, 3:5] = 0
        s[1, 4:16, 3:5] = 0
        s[0, 3:8, 5:7] = .25
        s[1, 4:16, 5:7] = .375
        before = self.snapshot(source)
        for kind in ("transformer", "mlp"):
            for rep in ("S", "H", "SH", "SHT"):
                model = self.dynamics(kind, representation=rep)
                for sequence in ([1, 0], [0, 1]):
                    with self.subTest(kind=kind, representation=rep, actions=sequence):
                        actions = torch.tensor(sequence)
                        with torch.no_grad():
                            result = model(source, actions)
                        self.assertEqual(result.lengths.tolist(),
                                         (source.lengths + actions * 8).tolist())
                        for i, action in enumerate(sequence):
                            n = int(source.lengths[i])
                            if action:
                                # Include every old row, not just context[2]'s committed prefix.
                                for block in (slice(0, self.layout.h),
                                              slice(self.layout.h, self.layout.h + 32)):
                                    torch.testing.assert_close(result.x[i, :n, block],
                                                               source.x[i, :n, block],
                                                               rtol=0, atol=0)
                                old_s = source.x[i, :n, self.layout.structure]
                                new_s = result.x[i, :n, self.layout.structure]
                                torch.testing.assert_close(new_s[:, 3:5], old_s[:, 3:5],
                                                           rtol=0, atol=0)
                                torch.testing.assert_close(new_s[:, 5:7], old_s[:, 5:7] + 1 / 8,
                                                           rtol=0, atol=0)
                                tail = result.x[i, n:n + 8, :self.layout.h + 32]
                                self.assertGreater(float((tail - self.prior[:, :self.layout.h + 32])
                                                         .abs().max()), 1e-6,
                                                   "E must still learn the new tail")
                            else:
                                start = round(float(source.context[i, 2]) * 64)
                                delta = (result.x[i, start:n, :self.layout.h + 32] -
                                         source.x[i, start:n, :self.layout.h + 32])
                                self.assertGreater(float(delta.abs().max()), 1e-6,
                                                   "R must still update its active rows")
                        self.assert_state_equal(source, before)

    def test_structured_repeated_extend_preserves_the_predicted_first_tail(self):
        for kind in ("transformer", "mlp"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind)
                with torch.no_grad():
                    state = model(self.source(), torch.tensor([0, 0]))
                    for _ in range(2):
                        before = self.snapshot(state)
                        state = model(state, torch.tensor([1, 1]))
                        for i, length in enumerate(before.lengths.tolist()):
                            torch.testing.assert_close(state.x[i, :length, :self.layout.h + 32],
                                                       before.x[i, :length, :self.layout.h + 32],
                                                       rtol=0, atol=0)
                            old_s = before.x[i, :length, self.layout.structure]
                            new_s = state.x[i, :length, self.layout.structure]
                            torch.testing.assert_close(new_s[:, 5:7], old_s[:, 5:7] + 1 / 8,
                                                       rtol=0, atol=0)
                        torch.testing.assert_close(state.prefix, before.prefix, rtol=0, atol=0)

    def test_dynamics_immutable_prefix_content_codes_and_metadata_age_exceptions(self):
        source = self.source((8, 16))
        saved = self.snapshot(source)
        for kind in ("transformer", "mlp", "linear"):
            with self.subTest(kind=kind):
                result = self.dynamics(kind)(source, torch.tensor([0, 1]))
                self.assertEqual(result.lengths.tolist(), [8, 24])
                torch.testing.assert_close(result.prefix, source.prefix, rtol=0, atol=0)
                torch.testing.assert_close(result.x[:, :3, :self.layout.h + 32],
                                           source.x[:, :3, :self.layout.h + 32], rtol=0, atol=0)
                for block in (self.layout.native, self.layout.stop):
                    torch.testing.assert_close(result.x[:, :3, block],
                                               source.x[:, :3, block], rtol=0, atol=0)
                old = source.x[:, :3, self.layout.structure]
                new = result.x[:, :3, self.layout.structure]
                stable = [0, 1, 2, 3, 4, 7, 8, 13, 14, 15]
                torch.testing.assert_close(new[..., stable], old[..., stable], rtol=0, atol=0)
                torch.testing.assert_close(new[..., 5:7], old[..., 5:7] + 1 / 8,
                                           rtol=0, atol=0)
                self.assertTrue(bool(torch.isfinite(result.x).all()))
                self.assert_state_equal(source, saved)

    def test_extend_history_recomputes_prefix_and_clears_old_new_unmask_flags(self):
        self.configure_history_scaler()
        source = self.source()
        hist = slice(self.layout.h + 32, self.layout.h + 36)
        valid = torch.arange(64)[None] < source.lengths[:, None]
        stale = torch.tensor([-.3, 1., .65, 0.])
        source.x[..., hist][valid] = (stale - self.prep.mean[-4:]) / self.prep.std[-4:]
        source.x[..., self.layout.structure][..., 2][valid] = 1
        saved = self.snapshot(source)
        for kind in ("transformer", "mlp", "linear"):
            model = self.dynamics(kind)
            for sequence in ([1, 0], [0, 1]):
                with self.subTest(kind=kind, actions=sequence):
                    with torch.no_grad():
                        result = model(source, torch.tensor(sequence))
                    for i, action in enumerate(sequence):
                        if not action:
                            continue
                        n = int(source.lengths[i])
                        # E copies hidden/gaps but recomputes confidence/ID/cos/valid history.
                        expected = torch.tensor([0., 0., 0., 1.]).expand(n, -1)
                        normalized = (expected - model.history_mean) / model.history_std
                        torch.testing.assert_close(result.x[i, :n, hist], normalized,
                                                   rtol=0, atol=0)
                        old_s = source.x[i, :n, self.layout.structure]
                        new_s = result.x[i, :n, self.layout.structure]
                        torch.testing.assert_close(new_s[:, 7], old_s[:, 7], rtol=0, atol=0)
                        self.assertEqual(float(new_s[:, 2].abs().sum()), 0,
                                         "E prefix must clear stale newly-unmasked flags")
                        normalized_zero = (-model.history_mean / model.history_std).expand(8, -1)
                        torch.testing.assert_close(result.x[i, n:n + 8, hist], normalized_zero,
                                                   rtol=0, atol=0)
                        self.assert_state_equal(source, saved)

    def test_refine_history_tracks_confidence_decoded_id_and_active_valid_cosine(self):
        self.configure_history_scaler()
        a, b = self.raw("history-r"), self.raw("history-e", 16, seed=44)
        a.scalars[3:, 0] = 1
        a.scalars[3:, 1] = 0
        a.ids[3:, 0] = 151665
        a.scalars[4, 3] = 0  # Fresh child hidden does not make parent hidden valid.
        a.surface[:, 32:] = torch.tensor([-.1, 1., .4, 0.])
        b.surface[:, 32:] = torch.tensor([-.1, 1., .4, 0.])
        source = pack(self.rows(a, b), "cpu")
        hist = slice(self.layout.h + 32, self.layout.h + 36)
        for kind in ("transformer", "mlp"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind, nonzero=False)
                model.frontier_gate = False
                with torch.no_grad():
                    model.output.bias[self.layout.structure.start + 7] = .1
                    model.output.bias[self.layout.h + 34] = .5 / model.history_std[2]
                    model.output.bias[self.layout.stop] = 3 * self.prep.token_codes(torch.tensor([3]))[0]
                result = model(source, torch.tensor([0, 1]))
                raw_hist = self.raw_history(result, model)
                s = result.x[..., self.layout.structure]
                parent = source.x[..., self.layout.structure]
                torch.testing.assert_close(raw_hist[0, :8, 0], s[0, :8, 7] - parent[0, :8, 7])
                expected_changed = (model.decoder.ids(result.x[0, 3:8]).squeeze(-1) !=
                                    model.decoder.ids(source.x[0, 3:8]).squeeze(-1)).float()
                self.assertGreater(float(expected_changed.sum()), 0)
                self.assertTrue(bool((expected_changed == 0).any()),
                                "a changed vector with the same decoded ID must count as unchanged")
                torch.testing.assert_close(raw_hist[0, 3:8, 1], expected_changed)
                expected_cos = torch.zeros(64)
                expected_cos[3:8] = .9
                expected_cos[4] = 0
                torch.testing.assert_close(raw_hist[0, :8, 2], expected_cos[:8])
                torch.testing.assert_close(raw_hist[1, :24, 2], torch.zeros(24))
                expected_valid = torch.ones(8)
                expected_valid[4] = 0
                torch.testing.assert_close(raw_hist[0, :8, 3], expected_valid)
                torch.testing.assert_close(raw_hist[1, :16, 3], torch.ones(16))
                torch.testing.assert_close(raw_hist[1, 16:24, 3], torch.zeros(8))
                # Copied-row cosine history cannot receive a learned residual.
                copied = result.x[0, :3, hist][:, 2].sum() + result.x[1, :24, hist][:, 2].sum()
                copied_grad, = torch.autograd.grad(copied, model.output.bias, retain_graph=True)
                self.assertEqual(float(copied_grad.abs().sum()), 0)
                raw_hist[0, 3:8, 2].sum().backward()
                self.assertTrue(bool(torch.isfinite(model.output.bias.grad).all()))
                self.assertGreater(float(model.output.bias.grad[self.layout.h + 34]), 0)

    def test_committed_stop_vectors_never_change_across_mixed_rollout(self):
        for kind in ("transformer", "mlp", "linear"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind)
                state = self.source((8, 16))
                for actions in ([0, 1], [1, 0], [0, 1], [1, 0]):
                    a = torch.tensor(actions)
                    before = self.snapshot(state)
                    pos = torch.arange(64)[None]
                    valid = pos < state.lengths[:, None]
                    committed = valid & ((state.x[..., self.layout.structure][..., 0] <= 0)
                                         | a[:, None].bool())
                    state = model(state, a)
                    torch.testing.assert_close(state.x[..., self.layout.stop][committed],
                                               before.x[..., self.layout.stop][committed],
                                               rtol=0, atol=0,
                                               msg=f"{kind} changed committed STOP content")
                    for i, action in enumerate(actions):
                        if action:
                            n = int(before.lengths[i])
                            torch.testing.assert_close(state.x[i, :n, self.layout.native],
                                                       before.x[i, :n, self.layout.stop],
                                                       rtol=0, atol=0)

    def test_refinement_masks_are_bounded_monotone_over_repeated_steps(self):
        for kind in ("transformer", "mlp", "linear"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind)
                state = self.source()
                for _ in range(3):
                    previous = state.x[..., self.layout.structure][..., 0].clone()
                    state = model(state, torch.zeros(2, dtype=torch.long))
                    mask = state.x[..., self.layout.structure][..., 0]
                    self.assertTrue(bool(((mask >= 0) & (mask <= 1)).all()))
                    self.assertTrue(bool((mask <= previous).all()))
                    valid = torch.arange(64)[None] < state.lengths[:, None]
                    torch.testing.assert_close(state.x[..., self.layout.structure][..., 1][valid],
                                               (1 - mask)[valid], rtol=0, atol=0)

    def test_free_extend_extend_uses_predicted_first_tail_and_responds_to_perturbation(self):
        for kind in ("transformer", "mlp", "linear"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind)
                source = self.source((8,))
                action = torch.tensor([1])
                first = model(source, action)
                first_saved = self.snapshot(first)
                seen = []
                hook = None
                if kind != "linear":
                    hook = model.input.register_forward_pre_hook(
                        lambda _module, args: seen.append(args[0].detach().clone()))
                try:
                    second = model(first, action)
                    perturbed = first.x.clone()
                    perturbed[:, 8:16, :self.layout.h] += .7
                    perturbed[:, 8:16, self.layout.stop] += torch.linspace(
                        -.6, .8, self.layout.code_dim)
                    changed = model(replace(first, x=perturbed), action)
                finally:
                    if hook is not None:
                        hook.remove()
                self.assertEqual(second.lengths.tolist(), [24])
                torch.testing.assert_close(second.x[0, 8:16, self.layout.native],
                                           first.x[0, 8:16, self.layout.stop], rtol=0, atol=0)
                difference = (second.x[0, 16:24, :self.layout.h + 32] -
                              changed.x[0, 16:24, :self.layout.h + 32]).abs().max()
                self.assertGreater(float(difference.detach()), 1e-6,
                                   f"{kind}: second E did not depend on the predicted first tail")
                if seen:
                    torch.testing.assert_close(seen[0][:, 8:16, self.layout.stop],
                                               first.x[:, 8:16, self.layout.stop], rtol=0, atol=0)
                    torch.testing.assert_close(seen[1][:, 8:16, self.layout.stop],
                                               perturbed[:, 8:16, self.layout.stop], rtol=0, atol=0)
                self.assert_state_equal(first, first_saved)

    def test_native_masked_codes_have_correct_shape_mixture_and_gradients(self):
        for kind in ("transformer", "mlp"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind)
                source = self.source()
                source = replace(source, x=source.x.detach().clone().requires_grad_())
                result = model(source, torch.tensor([0, 1]))
                native, stop = result.x[..., self.layout.native], result.x[..., self.layout.stop]
                self.assertEqual(tuple(native.shape), (2, 64, 32))
                self.assertTrue(native.requires_grad)
                mask = result.x[..., self.layout.structure][..., 0]
                valid = torch.arange(64)[None] < result.lengths[:, None]
                old_valid = torch.arange(64)[None] < source.lengths[:, None]
                immutable = old_valid & ((source.x[..., self.layout.structure][..., 0] <= 0)
                                         | torch.tensor([0, 1])[:, None].bool())
                editable = valid & ~immutable
                expected = mask[..., None] * model.mask_code + (1 - mask[..., None]) * stop
                torch.testing.assert_close(native[editable], expected[editable])
                native[editable].square().mean().backward()
                self.assert_finite_gradients(model, ("mask_head.bias", "output.weight"))
                self.assertIsNotNone(source.x.grad)
                self.assertTrue(bool(torch.isfinite(source.x.grad).all()))

    def test_linear_native_masked_codes_follow_mask_stop_mixture(self):
        model = self.dynamics("linear")
        source = self.source()
        result = model(source, torch.tensor([0, 1]))
        mask = result.x[..., self.layout.structure][..., 0]
        valid = torch.arange(64)[None] < result.lengths[:, None]
        expected = mask[..., None] * self.prep.token_codes(torch.tensor([151665]))[0]
        expected = expected + (1 - mask[..., None]) * result.x[..., self.layout.stop]
        torch.testing.assert_close(result.x[..., self.layout.native][valid], expected[valid],
                                   msg="LinearDynamics native codes must match mask/STOP content")

    def test_mixed_refine_extend_drafter_loss_backpropagates_through_both_steps(self):
        for kind in ("transformer", "mlp"):
            with self.subTest(kind=kind):
                model = self.dynamics(kind).train()
                source = self.source()
                first = model(source, torch.tensor([0, 1]))
                first.x.retain_grad()
                action = torch.tensor([1, 0])
                prediction = model(first, action)
                target = self.rows(self.raw("t0", 16, 8), self.raw("t1", 24, 16, seed=44))
                loss, metrics = drafter_loss(prediction, target, first, action,
                                             model.decoder, self.layout)
                self.assertEqual(prediction.lengths.tolist(), [16, 24])
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertEqual(set(metrics), {"hidden", "surface", "token_ce", "mask", "confidence"})
                loss.backward()
                self.assert_finite_gradients(model, ("input.weight", "global_input.weight",
                                                     "output.weight", "mask_head.weight"))
                self.assertTrue(bool(torch.isfinite(first.x.grad).all()))
                self.assertGreater(float(first.x.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in self.prep.compressor.parameters()))

    def test_linear_mixed_actions_are_differentiable_in_source(self):
        model = self.dynamics("linear")
        source = self.source()
        source = replace(source, x=source.x.detach().clone().requires_grad_())
        prediction = model(source, torch.tensor([0, 1]))
        prediction.x.square().mean().backward()
        self.assertIsNotNone(source.x.grad)
        self.assertTrue(bool(torch.isfinite(source.x.grad).all()))
        for i in range(2):
            self.assertGreater(float(source.x.grad[i].abs().sum()), 0)

    def test_linear_ridge_fit_on_synthetic_refine_and_extend_edges(self):
        rows = self.rows(self.raw("parent"), self.raw("refined", seed=32),
                         self.raw("extended", 16, 8, seed=33))
        model = LinearDynamics(self.prep, self.prior.clone())
        model.fit(dict(zip(("parent", "refined", "extended"), rows)),
                  [(("parent", "refined"), (0,)), (("parent", "extended"), (1,))],
                  "cpu", seed=7, alpha=1)
        self.assertTrue(bool(torch.isfinite(model.r_weight).all()))
        self.assertTrue(bool(torch.isfinite(model.e_weight).all()))
        source = pack([rows[0], rows[0]], "cpu")
        result = model(source, torch.tensor([0, 1]))
        self.assertEqual(result.lengths.tolist(), [8, 16])
        self.assertTrue(bool(torch.isfinite(result.x).all()))

    def test_drafter_loss_never_reads_verifier_fields(self):
        source = self.source((40, 40))
        action = torch.tensor([0, 1])
        model = self.dynamics("mlp")
        prediction = model(source, action)
        targets = self.rows(self.raw("t0", 40), self.raw("t1", 48, 40, seed=40))
        torch.manual_seed(701)
        normal, metrics = drafter_loss(prediction, targets, source, action,
                                       model.decoder, self.layout)
        torch.manual_seed(701)
        guarded, guarded_metrics = drafter_loss(prediction, [DOnlyRow(r) for r in targets],
                                                source, action, model.decoder, self.layout)
        torch.testing.assert_close(normal, guarded, rtol=0, atol=0)
        self.assertEqual(metrics, guarded_metrics)

    def test_drafter_loss_is_identical_after_label_teacher_mutation_with_reset_seed(self):
        raw = [self.raw("t0", 40), self.raw("t1", 48, 40, seed=40)]
        original = self.rows(*raw)
        for r in raw:
            r.accepted = r.length
            r.teacher_features = torch.full_like(r.teacher_features, float("nan"))
            r.teacher_margin = torch.full_like(r.teacher_margin, float("nan"))
        mutated = self.rows(*raw)
        for before, after in zip(original, mutated):
            for name in ("x", "context", "prefix", "token_targets"):
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        source = self.source((40, 40))
        action = torch.tensor([0, 1])
        model = self.dynamics("transformer")
        torch.manual_seed(702)
        a, ma = drafter_loss(model(source, action), original, source, action,
                             model.decoder, self.layout)
        torch.manual_seed(702)
        b, mb = drafter_loss(model(source, action), mutated, source, action,
                             model.decoder, self.layout)
        self.assertTrue(bool(torch.isfinite(a)))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(ma, mb)

    def test_drafter_hidden_and_surface_losses_ignore_invalid_or_stale_targets(self):
        raw = self.raw("target")
        raw.scalars[0, 3] = 0
        raw.scalars[1, 5] = 1 / 8
        raw.scalars[2, 4] = 0
        raw.scalars[3, 6] = 1 / 8
        row = self.rows(raw)[0]
        changed = dict(row, x=row["x"].clone())
        changed["x"][[0, 1], :self.layout.h] += 100
        changed["x"][[2, 3], self.layout.surface] += 100
        source = self.source((8,), (0,))
        prediction = self.dynamics("mlp")(source, torch.tensor([0]))
        a, ma = drafter_loss(prediction, [row], source, torch.tensor([0]),
                             self.decoder, self.layout)
        b, mb = drafter_loss(prediction, [changed], source, torch.tensor([0]),
                             self.decoder, self.layout)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(ma, mb)

    def test_drafter_confidence_loss_uses_only_fresh_valid_confidence_targets(self):
        raw = self.raw("confidence", frontier=0)
        raw.scalars[:, 8] = 1
        raw.scalars[2, 8] = 0
        raw.scalars[4, 4] = 0
        raw.scalars[5, 6] = 1 / 8
        rows = self.rows(raw)
        source = pack(rows, "cpu")
        baseline, base_metrics = drafter_loss(source, rows, source, torch.tensor([0]),
                                              self.decoder, self.layout)
        confidence_col = self.layout.structure.start + 7
        x = source.x.clone()
        x[0, 0, confidence_col] += .1
        x[0, 1, confidence_col] += .2
        prediction = replace(source, x=x.requires_grad_())
        loss, metrics = drafter_loss(prediction, rows, source, torch.tensor([0]),
                                     self.decoder, self.layout)
        self.assertAlmostEqual(base_metrics["confidence"], 0)
        self.assertAlmostEqual(metrics["confidence"], (.1 ** 2 + .2 ** 2) / 5, places=6)
        self.assertAlmostEqual(float((loss - baseline).detach()), .3 * metrics["confidence"], places=6)
        ignored = x.detach().clone()
        ignored[0, [2, 4, 5], confidence_col] += 10
        ignored_loss, ignored_metrics = drafter_loss(replace(source, x=ignored), rows, source,
                                                     torch.tensor([0]), self.decoder, self.layout)
        torch.testing.assert_close(loss, ignored_loss, rtol=0, atol=0)
        self.assertEqual(metrics, ignored_metrics)
        loss.backward()
        confidence_grad = prediction.x.grad[0, :, confidence_col]
        self.assertTrue(bool(torch.isfinite(confidence_grad).all()))
        self.assertGreater(float(confidence_grad[:2].abs().sum()), 0)
        self.assertEqual(float(confidence_grad[[2, 4, 5]].abs().sum()), 0)

    def test_structured_confidence_predictions_are_bounded_on_updated_rows(self):
        source = self.source()
        for kind in ("transformer", "mlp"):
            for bias, expected in ((-5., 0.), (5., 1.)):
                with self.subTest(kind=kind, bias=bias):
                    model = self.dynamics(kind, nonzero=False)
                    model.frontier_gate = False
                    with torch.no_grad():
                        model.output.bias[self.layout.structure.start + 7] = bias
                        result = model(source, torch.tensor([0, 1]))
                    s = result.x[..., self.layout.structure]
                    torch.testing.assert_close(s[0, 3:8, 7], torch.full((5,), expected))
                    torch.testing.assert_close(s[1, 16:24, 7], torch.full((8,), expected))
                    torch.testing.assert_close(s[1, :16, 7],
                                               source.x[1, :16, self.layout.structure][:, 7],
                                               rtol=0, atol=0)

    def test_tabular_verifier_rejects_missing_hazards_but_allows_missing_teachers(self):
        teacher_only = self.raw("teacher-only")
        teacher_only.accepted = None
        missing = self.raw("missing")
        missing.accepted = missing.teacher_features = missing.teacher_margin = None
        teacher_row, missing_row = self.rows(teacher_only, missing)
        for rows in ([], [teacher_row], [missing_row], [teacher_row, missing_row]):
            with self.subTest(rows=len(rows)):
                with self.assertRaisesRegex(ValueError, "(?i)accepted.*hazard|hazard.*supervision"):
                    TabularVerifier(self.layout).fit(rows)
        # Accepted length zero is a real hazard label, not missing supervision.
        labeled = self.raw("hazard-only")
        labeled.accepted = 0
        labeled.teacher_features = labeled.teacher_margin = None
        rows = self.rows(labeled)
        model = TabularVerifier(self.layout).fit(rows)
        heads = model(pack(rows, "cpu"))
        self.assertEqual(tuple(heads["hazard"].shape), (1, 64))
        self.assertTrue(bool(torch.isfinite(heads["hazard"]).all()))
        for name in ("tf", "probability", "margin"):
            self.assertIsNone(heads[name])

    def test_verifier_shapes_finite_separate_loss_and_backward(self):
        raw = [self.raw("v0"), self.raw("v1", 16, seed=40)]
        raw[1].accepted = None
        raw[1].teacher_features = None
        raw[1].teacher_margin = None
        rows = self.rows(*raw)
        state = pack(rows, "cpu")
        state = replace(state, x=state.x.detach().clone().requires_grad_())
        verifier = VerifierTransformer(self.layout, width=16, layers=1, dropout=0).train()
        heads = verifier(state)
        self.assert_heads(heads, 2)
        loss = verifier_loss(heads, rows, "cpu")
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assert_finite_gradients(verifier, ("readout.weight", "input.weight",
                                               "global_input.weight", "gate.weight"))
        self.assertTrue(bool(torch.isfinite(state.x.grad).all()))
        self.assertTrue(all(p.grad is None for p in self.prep.compressor.parameters()))

    def test_verifier_predictions_do_not_depend_on_current_labels_or_teachers(self):
        raw = [self.raw("v0"), self.raw("v1", 16, seed=40)]
        original = self.rows(*raw)
        verifier = VerifierTransformer(self.layout, width=16, layers=1, dropout=0).eval()
        a = verifier(pack(original, "cpu"))
        for r in raw:
            r.accepted = 0
            r.teacher_features = r.teacher_features.clone()
            r.teacher_features[:, 0] = -.8
            r.teacher_features[:, 1] = .1
            r.teacher_features[:, 2] = 1 - r.teacher_features[:, 2]
            r.teacher_margin = torch.full_like(r.teacher_margin, -.8)
        mutated = self.rows(*raw)
        b = verifier(pack(mutated, "cpu"))
        self.assert_heads(b, 2)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
        self.assertGreater(float((verifier_loss(a, original, "cpu") -
                                  verifier_loss(b, mutated, "cpu")).detach().abs()), 1e-6,
                           "label mutation must affect supervision, while predictions stay fixed")

    def test_verifier_missing_supervision_is_zero_differentiable_loss(self):
        raw = self.raw()
        raw.accepted = raw.teacher_features = raw.teacher_margin = None
        rows = self.rows(raw)
        verifier = VerifierTransformer(self.layout, width=16, layers=1, dropout=0)
        loss = verifier_loss(verifier(pack(rows, "cpu")), rows, "cpu")
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        for parameter in verifier.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertEqual(float(parameter.grad.abs().sum()), 0)

    def test_direct_outcome_action_sequences_shapes_and_independent_backward(self):
        rows = self.rows(self.raw("o0"), self.raw("o1", 16, seed=40))
        state = pack(rows, "cpu")
        model = DirectOutcome(self.layout, width=16)
        for horizon in (1, 2, 3):
            with self.subTest(horizon=horizon):
                actions = torch.tensor([[0, 1, 0], [1, 0, 1]])[:, :horizon]
                self.assert_heads(model(state, actions), 2)
        actions = torch.tensor([[0, 1, 0], [1, 0, 1]])
        heads = model(state, actions)
        flipped = model(state, 1 - actions)
        self.assertGreater(float((heads["hazard"] - flipped["hazard"]).detach().abs().max()),
                           1e-6)
        loss = verifier_loss(heads, rows, "cpu")
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assert_finite_gradients(model, ("input.weight", "summary.weight"))


if __name__ == "__main__":
    unittest.main()
