"""Offline fixtures exercise all stages, atomic artifacts and strict resume."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_factorized_wm_data import make_run, read_jsonl, write_jsonl
from phase0_wm_data import load_dataset, prepare_embedding, NativeTargets, prepare_rows, pack_observations
from run_latent_wm_phase0 import run, package, heads_to_rows


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trace = make_run(self.root / 'trace', questions=10)
        # The old adapter fixture intentionally uses invalid teacher values;
        # replace its final JSONL records with meaningful BCE/normalized targets.
        teachers = read_jsonl(self.trace, 'teacher_targets.jsonl')
        for row in teachers:
            row['features'] = [[.2, .8, 1.] + [0.] * 32 for _ in row['features']]
        write_jsonl(self.trace, 'teacher_targets.jsonl', teachers)
        self.cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/latent_wm_phase0.json').read_text())
        self.cfg.update(num_questions=10, latent_dims=[16], embedding_dim=8,
            token_embedding='learned', device_encoder='cpu', device_dynamics='cpu',
            stage_a_updates=2, raw_updates=2, direct_updates=2, dynamics_updates_per_horizon=2,
            eval_every=2, minimum_updates=1, batch_size=2, reconstruction_fit_rows=16,
            reconstruction_hidden_dim=2, bootstrap_samples=4, continue_diagnostics=True)
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(self.cfg))
        self.output = self.root / 'results'
        self.args = argparse.Namespace(input=str(self.trace), output=str(self.output),
                                       config=str(self.config), resume=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_free_rollout_and_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(self.args)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['LLM_forwards'], 0)
        width = summary['widths']['16']
        self.assertEqual(width['completed_horizons'], [1, 2, 3])
        self.assertTrue(width['continued_after_failed_gate'])
        rows = read_jsonl(self.output / 'latent_16/test', 'composition_predictions.jsonl')
        self.assertEqual({r['horizon'] for r in rows}, {1, 2, 3})
        self.assertTrue(all('direct_verifier' in r and 'oracle_verifier' in r for r in rows))
        self.assertTrue(all(len(r['actions']) == r['horizon'] for r in rows))
        self.assertEqual({r['question'] for r in rows},
                         set(json.loads((self.output / 'split_manifest.json').read_text())['test']))
        with zipfile.ZipFile(self.output.with_suffix('.zip')) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any(name.endswith('dynamics/H3/best.pt') for name in archive.namelist()))
            self.assertFalse(any('/experience/' in name for name in archive.namelist()))
        self.args.resume = True
        with contextlib.redirect_stdout(io.StringIO()), patch('run_latent_wm_phase0.fit_observation', wraps=__import__('run_latent_wm_phase0').fit_observation) as fit:
            resumed = run(self.args)
        # Completed observation checkpoint reused; dynamics/width training skipped.
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(resumed['widths'], summary['widths'])

    def test_failed_gate_stops_without_claiming_h3(self):
        self.cfg['continue_diagnostics'] = False
        self.config.write_text(json.dumps(self.cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(self.args)
        width = summary['widths']['16']
        self.assertEqual(width['status'], 'stage_A_gate_failed')
        self.assertFalse((self.output / 'latent_16/dynamics/H3/best.pt').exists())
        self.assertTrue(self.output.with_suffix('.zip').is_file())

    def test_resume_rejects_changed_config_and_packages_partial(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.args)
        self.cfg['seed'] += 1
        self.config.write_text(json.dumps(self.cfg)); self.args.resume = True
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Resume source/config/data differs'):
            run(self.args)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['status'], 'partial')
        with zipfile.ZipFile(self.output.with_suffix('.zip')) as archive:
            self.assertTrue(any(name.endswith('error.txt') for name in archive.namelist()))
            self.assertTrue(any(name.endswith('best.pt') for name in archive.namelist()))

    def test_pretrained_local_embedding_only_train_projection(self):
        from safetensors.torch import save_file
        torch.manual_seed(10)
        weights = torch.randn(151936, 12)
        checkpoint = self.root / 'model.safetensors'
        save_file({'model.embed_tokens.weight': weights}, checkpoint)
        dataset = load_dataset(self.trace)
        cfg = dict(self.cfg, token_embedding='pretrained', embedding_path=str(checkpoint),
                   embedding_fit_tokens=64)
        a, report = prepare_embedding(dataset, ['gsm8k:0'], cfg, self.root, 'cpu')
        self.assertEqual(a['weights'].shape, (151936, 8))
        self.assertFalse(a['learned'])
        self.assertEqual(report['LLM_forwards'], 0)
        self.assertEqual(report['projection_fit'], 'train token IDs only')
        # Held-out observation edits cannot alter the fitted input projection.
        dataset.states['q9s0'].ids += 5000
        b, _ = prepare_embedding(dataset, ['gsm8k:0'], cfg, self.root, 'cpu')
        torch.testing.assert_close(a['weights'], b['weights'])

    def test_fixed_reconstruction_serialization_and_observation_padding(self):
        dataset = load_dataset(self.trace)
        targets = NativeTargets.fit(dataset, ['gsm8k:0'], 2, 16)
        path = self.root / 'target.pt'; torch.save(vars(targets), path)
        restored = NativeTargets(**torch.load(path, weights_only=True))
        torch.testing.assert_close(targets.basis, restored.basis)
        rows = prepare_rows(dataset, targets)
        batch = pack_observations([rows['q0s1'], rows['q0s2']], 'cpu')
        self.assertEqual(batch.hidden.shape, (2, 64, 3, 1536))
        self.assertEqual(batch.lengths.tolist(), [8, 16])
        self.assertEqual(batch.prefix_lengths.tolist(), [2, 2])
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            pack_observations([rows['q0s0']], 'cpu', prefix_max_tokens=1)

    def test_expected_K_excludes_bonus_and_missing_teacher_is_missing(self):
        logits = torch.full((1, 64), torch.logit(torch.tensor(.5)))
        heads = dict(hazard=logits, tf=logits, probability=logits.sigmoid(), margin=logits.tanh())
        row = dict(uid='x', question='q', length=8, accepted=1, teacher=None)
        prediction = heads_to_rows(heads, [row])[0]
        self.assertAlmostEqual(prediction['expected_yield'], sum(.5**i for i in range(1, 9)))
        self.assertIsNone(prediction['tf_truth'])
        self.assertEqual(prediction['mode'], 0)

    def test_atomic_packaging_omits_large_raw_cache(self):
        self.output.mkdir()
        (self.output / 'experience').mkdir()
        (self.output / 'experience/shard.npz').write_bytes(b'raw')
        (self.output / 'best.pt').write_bytes(b'compact')
        (self.output / 'unfinished.tmp').write_bytes(b'partial')
        with contextlib.redirect_stdout(io.StringIO()):
            archive_path = package(self.output)
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(archive.namelist(), ['results/best.pt'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
