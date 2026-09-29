"""No model downloads/GPU/verifier required. Exercises actual offline evaluator."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_feature_audit import Archive, audit, encode, make_report, perturb, load_observations, add_diagnostics
from world_model_core import Observation, pack_observations, expected_acceptance
from world_model_probe import ProbeWorldModel

torch.set_num_threads(1)


def observation(uid, question, length, segment=0):
    scalars = torch.zeros(length, 16)
    scalars[:, 0] = 1; scalars[:, 3:5] = 1; scalars[:, 7] = .7
    context = torch.tensor([.1, length/64, segment/64, 0, .5, .5, .125, 1.])
    return Observation(uid, question, 0, torch.ones(length, 2, dtype=torch.long),
        torch.randn(length, 3, 4).half(), torch.zeros(length, 2).half(), scalars, context,
        min(length, 1), torch.tensor([1, 2, 3]), torch.ones(length, 2, dtype=torch.long),
        torch.randn(length, 4))


class AuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ProbeWorldModel(4, 4, 2, 16, 3, 0).eval()
        self.table = torch.randn(8, 4).half()

    def test_perturbation_does_not_mutate_input_or_rules(self):
        b = pack_observations([observation('a', 'q', 2)], self.table, 'cpu')
        old = {k: v.clone() for k, v in b.items()}
        z = encode(self.model, b, 'mask_encoder')
        self.assertTrue(torch.equal(z.mask_probs, b['scalars'][:, :, 0]))
        self.assertEqual(int(z.frontier[0]), 0)
        for variant in ('hidden_all', 'hidden_layer_1', 'topk', 'token_stop', 'history',
                        'encoder_context', 'confidence', 'position_age', 'prefix_content'):
            encode(self.model, b, variant)
        for k in old:
            self.assertTrue(torch.equal(old[k], b[k]), k)
        self.assertFalse(self.model.encoder.context._forward_hooks)

    def test_teacher_and_labels_are_not_inputs(self):
        b = pack_observations([observation('a', 'q', 2)], self.table, 'cpu')
        before = encode(self.model, b, 'full')
        b['labels'].fill_(99); b['teacher_margin'].fill_(99); b['teacher_valid'].fill_(True)
        after = encode(self.model, b, 'full')
        self.assertTrue(torch.equal(before.tokens, after.tokens))

    def test_followup_channels_and_last_layer(self):
        b = pack_observations([observation('a', 'q', 2)], self.table, 'cpu')
        b['gaps'].fill_(-2)
        b['topk_uniform_vectors'] = b['topk_vectors'] + 1
        gaps = perturb(b, 'topk_gaps_channel')
        self.assertEqual(float(gaps['gaps'].abs().sum()), 0)
        self.assertTrue(torch.equal(gaps['topk_vectors'], b['topk_vectors']))
        candidates = perturb(b, 'topk_candidate_channel')
        self.assertTrue(torch.equal(candidates['gaps'], b['gaps']))
        self.assertEqual(float(candidates['topk_vectors'].abs().sum()), 0)
        no_probabilities = perturb(b, 'topk_no_probabilities')
        self.assertTrue(torch.equal(no_probabilities['topk_vectors'], b['topk_uniform_vectors']))
        compact = perturb(b, 'compact_last')
        self.assertTrue(torch.equal(compact['hidden'][:, :, -1], b['hidden'][:, :, -1]))
        self.assertEqual(float(compact['hidden'][:, :, :-1].abs().sum()), 0)
        self.assertEqual(float(compact['history'].abs().sum()), 0)
        self.assertTrue(torch.equal(compact['scalars'], b['scalars']))

    def test_change_groups_and_matched_horizon(self):
        source = observation('a', 'q', 8); source.accepted = 2
        rows = []
        for depth, target in [(1, 4), (2, 5), (3, 5)]:
            row = dict(split='validation', variant='full', group=f'h{depth}_R',
                source='a', state_id=str(depth), depth=depth, actions='R'*depth,
                question='q', expected_K=3., K=target, nll=1., persistence_error=abs(2.-target))
            add_diagnostics(row, source, 2.)
            rows.append(row)
        report = make_report(rows, 10)
        self.assertEqual(report['validation/full/h1_R_gain']['n'], 1)
        self.assertEqual(report['validation/full/h1_R_changed']['delta_K_mae'], 1.)
        self.assertEqual(report['validation/full/h1_R_source8']['n'], 1)
        self.assertEqual(report['validation/full/h1_common3']['n'], 1)
        self.assertEqual(report['validation/full/h1_R']['dynamics_minus_persistence_macro'], -1.)

    def test_paired_metrics_cluster_bootstrap(self):
        rows = []
        for i in range(3):
            for variant, pred in [('full', 2.), ('history', 3.)]:
                rows.append(dict(split='validation', variant=variant, group='current',
                    source=str(i), state_id=str(i), depth=0, actions='', question=str(i),
                    expected_K=pred, K=2, nll=1.))
        report = make_report(rows, 100)
        changed = report['validation/history/current']
        self.assertEqual(changed['delta_mae'], 1.)
        self.assertEqual(changed['delta_question_macro_mae_ci95'], [1., 1.])

    def test_complete_offline_zip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root/'run'; run.mkdir(); (run/'experience').mkdir()
            metadata = []; labels = []; edges = []; saved = []
            for i, split in enumerate(('train', 'validation')):
                q = f'q{i}'
                nodes = [observation(q+'a', q, 2), observation(q+'b', q, 2), observation(q+'c', q, 4, 2)]
                lengths = np.asarray([o.length for o in nodes]); offsets = np.r_[0, lengths.cumsum()]
                arrays = {k: torch.cat([getattr(o, k) for o in nodes]).numpy()
                          for k in ('ids', 'hidden', 'gaps', 'scalars', 'history')}
                arrays.update(offsets=offsets, context=torch.stack([o.context for o in nodes]).numpy(),
                    aligned_topk_token_ids=torch.cat([o.topk_ids for o in nodes]).numpy())
                shard = f'experience/shard_{i:04d}.npz'; np.savez_compressed(run/shard, **arrays)
                for j, o in enumerate(nodes):
                    metadata.append(dict(state_id=o.uid, question=q, round_id=0, split=split,
                        shard=shard, row=j, prefix_token_ids=o.prefix_ids.tolist()))
                    labels.append(dict(state_id=o.uid, accepted_len=o.accepted, label_valid=True))
                    if split == 'validation':
                        z = self.model.encoder(pack_observations([o], self.table, 'cpu'))
                        saved.append(dict(kind='current', state_id=o.uid,
                            expected_K=float(expected_acceptance(self.model.acceptance(z), z.lengths)[0])))
                edges += [dict(parent=nodes[0].uid, child=nodes[1].uid, action='R'),
                          dict(parent=nodes[1].uid, child=nodes[2].uid, action='E')]
            for name, rows in [('states', metadata), ('labels', labels), ('edges', edges), ('validation_predictions', saved)]:
                (run/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            torch.save(dict(model_config=self.model.config, model=self.model.state_dict(),
                            args={'extend_size': 2}), run/'checkpoint.pt')
            embeddings = root/'embeddings.safetensors'
            save_file({'model.embed_tokens.weight': self.table}, str(embeddings))
            archive = root/'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for path in run.rglob('*'):
                    if path.is_file(): z.write(path, 'nested/'+path.relative_to(run).as_posix())
            # Both Kaggle extracted folders and original ZIPs must work.
            self.assertEqual(len(load_observations(Archive(run), metadata[:3], {r['state_id']:r for r in labels})), 3)
            out = root/'result'
            audit(SimpleNamespace(input=archive, output=out, trust_checkpoint=True, device='cpu',
                embeddings=str(embeddings), embedding_repo='unused', embedding_revision='unused',
                cache=str(root), batch_size=2, horizon=3, bootstrap=30, ablate_train=False))
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['parity_with_saved_predictions']['passed'])
            self.assertEqual(summary['llm_forward_calls'], 0)
            self.assertEqual(summary['metrics']['validation/full/h1_R']['n'], 1)
            self.assertEqual(summary['metrics']['validation/full/h2_E']['n'], 1)
            self.assertNotIn('train/history/current', summary['metrics'])
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertIn('report.md', z.namelist())
                self.assertIsNone(z.testzip())
            followup = root/'followup'
            audit(SimpleNamespace(input=archive, output=followup, trust_checkpoint=True, device='cpu',
                embeddings=str(embeddings), embedding_repo='unused', embedding_revision='unused',
                cache=str(root), batch_size=2, horizon=3, bootstrap=30, ablate_train=False,
                suite='followup', validation_only=True))
            result = json.loads((followup/'summary.json').read_text())
            self.assertTrue(result['parity_with_saved_predictions']['passed'])
            self.assertNotIn('train/full/current', result['metrics'])
            self.assertEqual(result['metrics']['validation/compact_last/current']['n'], 3)
            self.assertEqual(result['metrics']['validation/full/h1_R_same']['n'], 1)


if __name__ == '__main__':
    unittest.main()
