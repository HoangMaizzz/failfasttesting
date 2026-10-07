"""Offline architecture, leakage, objectives, and committed-resume checks."""
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from native_qwen_probe import (QuestionActionSampler, call_model, make_model,
                               parameter_counts, probe_loss, survival_truth, train_model)
from paired_latent_models import PairedStudent, acceptance_loss, outputs


def spec(kind='raw', variant='HC', depth=7, dim=128, **kwargs):
    return dict(kind=kind, variant=variant, depth=depth, dim=dim, **kwargs)


def batch_fixture(width=64, hidden_dim=12, count=2):
    rng = torch.Generator().manual_seed(29)
    return dict(hidden=torch.randn(count, width, hidden_dim, generator=rng),
                candidate_embedding=torch.randn(count, width, hidden_dim, generator=rng),
                structural=torch.randn(count, width, 4, generator=rng),
                z_D=torch.randn(count, width, 128, generator=rng),
                c=torch.randn(count, width, 20, generator=rng),
                context=torch.randn(count, 8, generator=rng),
                lengths=torch.tensor([width - (i % width) for i in range(count)]),
                accepted=torch.tensor([min(2, width - (i % width)) for i in range(count)]))


class ProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_same_raw_architecture_capacity_and_initialization_across_depths(self):
        for variant, width in [('H', 12), ('HC', 24)]:
            models = []
            for depth in (7, 14, 21, 28):
                torch.manual_seed(42)
                model = make_model(spec(variant=variant, depth=depth), {}, 12)
                models.append(model)
                layers = list(model.encoder)
                self.assertEqual([type(x) for x in layers],
                                 [nn.LayerNorm, nn.Linear, nn.GELU, nn.Dropout, nn.Linear])
                self.assertEqual(layers[0].normalized_shape, (width,))
                self.assertEqual((layers[1].in_features, layers[1].out_features), (width, 256))
                self.assertEqual(layers[3].p, .05)
                self.assertEqual((layers[4].in_features, layers[4].out_features), (256, 1))
                self.assertIsNone(model.head)
            self.assertEqual(len({parameter_counts(m)['trainable'] for m in models}), 1)
            for model in models[1:]:
                for key, value in models[0].state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_survival_truth_zero_based_boundary_and_full_prefix(self):
        truth = survival_truth(torch.tensor([0, 1, 3, 4]), 4)
        self.assertEqual(truth.tolist(), [[0, 0, 0, 0], [1, 0, 0, 0],
                                         [1, 1, 1, 0], [1, 1, 1, 1]])

    def test_raw_structural_default_false_preserves_sweep(self):
        for variant in ('H', 'HC'):
            torch.manual_seed(42)
            default = make_model(spec(variant=variant), {}, 12).eval()
            torch.manual_seed(42)
            explicit = make_model(spec(variant=variant, structural_input=False), {}, 12).eval()
            self.assertFalse(default.uses_structural)
            for key, value in default.state_dict().items():
                self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
            batch = batch_fixture(width=4)
            before = call_model(default, spec(variant=variant), batch)
            batch['structural'].fill_(float('nan'))
            self.assertTrue(torch.equal(before['q'], call_model(default, spec(variant=variant), batch)['q']))

    def test_raw_structural_reference_same_inputs_as_latent_and_detached(self):
        models = []
        for depth in (7, 14, 21, 28):
            s = spec(depth=depth, structural_input=True)
            torch.manual_seed(42)
            model = make_model(s, {}, 12).eval()
            models.append(model)
            self.assertEqual(model.encoder[0].normalized_shape, (28,))
            self.assertEqual(model.encoder[1].in_features, 28)
            self.assertEqual(model.encoder[-1].out_features, 1)
            self.assertIsNone(model.head)
            self.assertIsNone(model.reconstruction)
            batch = batch_fixture(width=4)
            for name in ('hidden', 'candidate_embedding', 'structural'):
                batch[name].requires_grad_(True)
            result = call_model(model, s, batch)
            changed = dict(batch, structural=torch.zeros_like(batch['structural']))
            self.assertFalse(torch.equal(result['q'], call_model(model, s, changed)['q']))
            loss, parts = probe_loss(s, result, batch)
            self.assertTrue(torch.equal(loss, parts['bce'] + .1 * parts['K_huber']))
            loss.backward()
            for name in ('hidden', 'candidate_embedding', 'structural'):
                self.assertIsNone(batch[name].grad)
            with self.assertRaisesRegex(ValueError, 'four positional scalars'):
                call_model(model, s, {k: v for k, v in batch.items() if k != 'structural'})
        self.assertEqual(len({parameter_counts(m)['trainable'] for m in models}), 1)
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_native_q_is_direct_sigmoid_not_hazard_product(self):
        for kind in ('raw', 'latent'):
            s = spec(kind)
            model = make_model(s, {}, 12).eval()
            for p in model.parameters():
                nn.init.zeros_(p)
            batch = batch_fixture()
            result = call_model(model, s, batch)
            self.assertTrue(torch.equal(result['q'][0], torch.full((64,), .5)))
            self.assertEqual(result['K'].tolist(), [32., 31.5])
            self.assertEqual(float(result['q'][1, 63]), 0.)

    def test_labels_parent_k_and_final_logits_never_enter_any_forward(self):
        batch = batch_fixture(width=5)
        for kind in ('raw', 'latent', 'direct'):
            s = spec(kind)
            model = make_model(s, {}, 12).eval()
            before = call_model(model, s, batch)
            changed = dict(batch, accepted=torch.tensor([0, 0]), parent_K=torch.tensor([999, 999]),
                           parent_accepted=torch.tensor([888, 888]),
                           final_logits=torch.randn(2, 5, 100), survival_true=torch.zeros(2, 5))
            after = call_model(model, s, changed)
            self.assertTrue(torch.equal(before['q'], after['q']))
            self.assertTrue(torch.equal(before['K'], after['K']))
            unlabelled = {k: v for k, v in changed.items() if k not in
                          ('accepted', 'parent_K', 'parent_accepted', 'final_logits', 'survival_true')}
            self.assertTrue(torch.equal(before['q'], call_model(model, s, unlabelled)['q']))

    def test_direct_is_fresh_original_architecture_and_current_objective(self):
        torch.manual_seed(42)
        model = make_model(spec('direct'), {'dropout': .05}, 12).eval()
        torch.manual_seed(42)
        reference = PairedStudent('Direct', 128, layers=2, dropout=.05).eval()
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        batch = batch_fixture(width=4)
        # A Direct-only pack has no verifier hidden, candidate embedding or structure.
        direct_batch = {k: v for k, v in batch.items() if k not in
                        ('hidden', 'candidate_embedding', 'structural')}
        observed = call_model(model, spec('direct'), direct_batch)
        expected = reference(batch['z_D'], batch['c'], batch['context'], batch['lengths'])
        self.assertTrue(torch.equal(observed['q'], outputs(expected['hazard'], batch['lengths'])['q']))
        loss, parts = probe_loss(spec('direct'), observed, batch)
        legacy, legacy_parts = acceptance_loss(expected['hazard'], batch['lengths'], batch['accepted'])
        self.assertTrue(torch.equal(loss, legacy))
        self.assertEqual(parts.keys(), legacy_parts.keys())
        contaminated = dict(batch, hidden=torch.full_like(batch['hidden'], float('nan')),
                            candidate_embedding=torch.full_like(batch['candidate_embedding'], float('nan')),
                            structural=torch.full_like(batch['structural'], float('nan')))
        self.assertTrue(torch.equal(observed['q'], call_model(model, spec('direct'), contaminated)['q']))

    def test_native_and_direct_inputs_are_detached(self):
        for kind in ('raw', 'latent', 'direct'):
            s = spec(kind, lambda_rec=.01 if kind == 'latent' else 0.)
            batch = batch_fixture(width=4)
            for name in ('hidden', 'candidate_embedding', 'structural', 'z_D', 'c', 'context'):
                batch[name].requires_grad_(True)
            model = make_model(s, {}, 12)
            loss, _ = probe_loss(s, call_model(model, s, batch), batch)
            loss.backward()
            for name in ('hidden', 'candidate_embedding', 'structural', 'z_D', 'c', 'context'):
                self.assertIsNone(batch[name].grad, (kind, name))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_hidden_only_does_not_read_embedding_or_structural(self):
        s = spec(variant='H')
        model = make_model(s, {}, 12).eval()
        batch = batch_fixture()
        before = call_model(model, s, batch)
        del batch['candidate_embedding'], batch['structural']
        self.assertTrue(torch.equal(before['q'], call_model(model, s, batch)['q']))

    def test_latent_architecture_and_primary_reconstruction_zero(self):
        for dim in (64, 128, 256):
            s = spec('latent', dim=dim)
            model = make_model(s, {}, 12)
            self.assertEqual(model.encoder[0].normalized_shape, (28,))
            self.assertEqual(model.encoder[-1].out_features, dim)
            self.assertEqual([type(m) for m in model.head], [nn.LayerNorm, nn.Linear])
            self.assertIsNone(model.reconstruction)
            result = call_model(model, s, batch_fixture())
            self.assertEqual(result['z'].shape, (2, 64, dim))
            self.assertNotIn('reconstructed_hidden', result)
            loss, parts = probe_loss(s, result, batch_fixture())
            self.assertEqual(float(parts['reconstruction']), 0.)
            self.assertTrue(torch.equal(loss, parts['bce'] + .1 * parts['K_huber']))
        s = spec('latent', lambda_rec=.1)
        model = make_model(s, {}, 12)
        batch = batch_fixture()
        result = call_model(model, s, batch)
        self.assertEqual(result['reconstructed_hidden'].shape, batch['hidden'].shape)
        loss, parts = probe_loss(s, result, batch)
        self.assertGreater(float(parts['reconstruction'].detach()), 0.)
        self.assertTrue(torch.equal(loss, parts['bce'] + .1 * parts['K_huber'] + .1 * parts['reconstruction']))

    def test_bce_only_covers_all_valid_positions_after_rejection(self):
        s = spec(lambda_K=0.)
        logits = torch.tensor([[0., 1., 2., 3.], [4., 5., 99., 99.]], requires_grad=True)
        valid = torch.tensor([[True] * 4, [True, True, False, False]])
        q = logits.sigmoid().masked_fill(~valid, 0.)
        batch = dict(lengths=torch.tensor([4, 2]), accepted=torch.tensor([1, 0]))
        loss, parts = probe_loss(s, dict(logits=logits, q=q, K=q.sum(-1)), batch)
        expected = F.binary_cross_entropy_with_logits(logits[valid], torch.tensor([1., 0., 0., 0., 0., 0.]))
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertGreater(float(logits.grad[0, 3]), 0.)
        self.assertTrue(torch.equal(logits.grad[1, 2:], torch.zeros(2)))

    def test_padding_is_ignored_even_with_nan_observations(self):
        s = spec('latent', lambda_rec=.01)
        model = make_model(s, {}, 12).eval()
        batch = batch_fixture(width=4)
        before = call_model(model, s, batch)
        for name in ('hidden', 'candidate_embedding', 'structural'):
            batch[name][1, 3] = float('nan')
        after = call_model(model, s, batch)
        self.assertTrue(torch.equal(before['q'], after['q']))
        loss, _ = probe_loss(s, after, batch)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_invalid_spec_and_missing_labels_fail_fast(self):
        for s in (spec('unknown'), spec(variant='bad'), spec('latent', dim=32),
                  spec(lambda_rec=.1), spec(lambda_K=-.1), spec(structural_input='yes')):
            with self.assertRaises(ValueError):
                make_model(s, {}, 12)
        batch = batch_fixture(width=4)
        s = spec()
        result = call_model(make_model(s, {}, 12), s, batch)
        for labels in (torch.tensor([-1, 0]), torch.tensor([5, 0]), torch.tensor([.5, 0])):
            with self.assertRaises(ValueError):
                probe_loss(s, result, dict(batch, accepted=labels))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.rows = {str(i): dict(question_id=f'q{i // 2}', action='R' if i % 2 else 'E', split='train')
                     for i in range(6)}
        self.uids = list(self.rows)
        self.cfg = dict(updates=6, eval_every=2, batch_size=3, lr=.0003,
                        dropout=.05, study_signature='frozen-source-and-question-split')
        self.batch = batch_fixture(width=4, count=6)

    def pack(self, uids, s, device):
        indices = [int(uid) for uid in uids]
        # Consume each RNG so the resumed checkpoint must restore all of them.
        random.random()
        np.random.random()
        torch.rand(1)
        return {k: v[indices].to(device) for k, v in self.batch.items()}

    def validate(self, model, s):
        self.assertFalse(model.training)
        self.assertFalse(torch.is_grad_enabled())
        batch = {k: v[:2] for k, v in self.batch.items()}
        result = call_model(model, s, batch)
        score = float((result['K'] - batch['accepted']).abs().mean())
        return score, {'ALL': {'K_question_macro_MAE': score, 'num_states': 2,
                                'num_questions': 1, 'num_positions': 7}}

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_exact_optimizer_rng_and_model_resume_on_cpu(self):
        s = spec('latent', dim=64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, full_record = train_model(s, 42, self.uids, self.rows, self.pack,
                                           self.validate, self.cfg, root / 'full', 'cpu')
            full_rng = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
            pack_calls = 0

            def interrupted_pack(uids, s, device):
                nonlocal pack_calls
                pack_calls += 1
                if pack_calls == 4:  # initial shape inspection, 2 updates, then failure
                    raise RuntimeError('simulated interruption')
                return self.pack(uids, s, device)

            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                train_model(s, 42, self.uids, self.rows, interrupted_pack,
                            self.validate, self.cfg, root / 'resume', 'cpu')
            checkpoint = torch.load(root / 'resume/last.pt', weights_only=True)
            self.assertEqual(checkpoint['step'], 2)
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            resumed, resumed_record = train_model(s, 42, self.uids, self.rows, self.pack,
                                                 self.validate, self.cfg, root / 'resume', 'cpu')
            self.assert_nested_equal(full.state_dict(), resumed.state_dict())
            self.assert_nested_equal(full_record, resumed_record)
            self.assertTrue(torch.equal(full_rng[0], torch.get_rng_state()))
            self.assertEqual(full_rng[1], random.getstate())
            np.testing.assert_array_equal(full_rng[2][1], np.random.get_state()[1])
            self.assertEqual(full_rng[2][2:], np.random.get_state()[2:])
            for name in ('last.pt', 'best.pt'):
                a = torch.load(root / 'full' / name, weights_only=True)
                b = torch.load(root / 'resume' / name, weights_only=True)
                self.assert_nested_equal(a, b)
                self.assertIn('optimizer', b)
                self.assertIn('rng', b)
                self.assertFalse(b['test_access'])
            # Finished-job reuse restores the committed RNG, with no training.
            again, record = train_model(s, 42, self.uids, self.rows, self.pack,
                                        self.validate, self.cfg, root / 'resume', 'cpu')
            self.assert_nested_equal(record, resumed_record)
            self.assert_nested_equal(again.state_dict(), resumed.state_dict())

    def test_signature_rejects_config_seed_spec_and_training_subset_changes(self):
        s = spec()
        with tempfile.TemporaryDirectory() as temporary:
            train_model(s, 42, self.uids, self.rows, self.pack,
                        self.validate, self.cfg, temporary, 'cpu')
            changes = [(s, 43, self.uids, self.cfg),
                       (spec(depth=14), 42, self.uids, self.cfg),
                       (spec(structural_input=True), 42, self.uids, self.cfg),
                       (s, 42, self.uids[:-1], self.cfg),
                       (s, 42, self.uids, dict(self.cfg, lr=.001)),
                       (s, 42, self.uids, dict(self.cfg, study_signature='different-capture'))]
            for changed_spec, seed, uids, cfg in changes:
                with self.assertRaisesRegex(ValueError, 'Resume signature differs'):
                    train_model(changed_spec, seed, uids, self.rows, self.pack,
                                self.validate, cfg, temporary, 'cpu')

    def test_batch_stream_matches_across_depth_variant_and_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = []
            for index, s in enumerate([spec(depth=7), spec(variant='H', depth=14),
                                       spec('latent', depth=21, dim=64), spec('direct'),
                                       spec(depth=7, structural_input=True)]):
                _, record = train_model(s, 42, self.uids, self.rows, self.pack,
                                        self.validate, self.cfg, Path(temporary) / str(index), 'cpu')
                records.append(record)
                self.assertFalse(record['test_access'])
                self.assertEqual(record['selection_split'], 'validation')
            self.assertEqual(len({r['batch_digest'] for r in records}), 1)

    def test_validation_selects_earlier_step_and_recovers_stale_best_file(self):
        s = spec()
        scores = iter([3., 1., 2.])

        def validation(model, s):
            return next(scores), {'ALL': {'num_questions': 2}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, record = train_model(s, 42, self.uids, self.rows, self.pack,
                                        validation, self.cfg, root, 'cpu')
            self.assertEqual(record['selected_step'], 4)
            last = torch.load(root / 'last.pt', weights_only=True)
            best = torch.load(root / 'best.pt', weights_only=True)
            self.assertEqual(last['step'], 6)
            self.assertEqual(best['step'], 4)
            self.assert_nested_equal(model.state_dict(), best['model'])
            # Simulate a stale/torn externally materialized best checkpoint.
            torch.save({'step': 999}, root / 'best.pt')
            recovered, recovered_record = train_model(s, 42, self.uids, self.rows, self.pack,
                                                       validation, self.cfg, root, 'cpu')
            self.assert_nested_equal(record, recovered_record)
            self.assert_nested_equal(model.state_dict(), recovered.state_dict())
            self.assert_nested_equal(best, torch.load(root / 'best.pt', weights_only=True))

    def test_sampler_balances_action_and_question_not_state_count(self):
        rows = {'a': dict(question='q1', action='R'),
                'b': dict(question='q2', action='R'),
                **{str(i): dict(question='q3', action='E') for i in range(100)}}
        sampler = QuestionActionSampler(list(rows), rows, 42)
        samples = sampler.sample(6000)
        r_fraction = sum(rows[u]['action'] == 'R' for u in samples) / len(samples)
        q1_fraction = samples.count('a') / len(samples)
        self.assertAlmostEqual(r_fraction, .5, delta=.03)
        self.assertAlmostEqual(q1_fraction, .25, delta=.03)
        repeated = QuestionActionSampler(list(reversed(rows)), rows, 42).sample(6000)
        self.assertEqual(samples, repeated)

    def test_sampler_rejects_test_data_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            QuestionActionSampler(['x'], {'x': dict(question='q-test', split='test')}, 42)
        with self.assertRaisesRegex(ValueError, 'unique'):
            QuestionActionSampler(['0', '0'], self.rows, 42)


if __name__ == '__main__':
    unittest.main()
