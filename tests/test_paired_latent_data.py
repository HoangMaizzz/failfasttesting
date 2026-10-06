"""Offline paired-loader checks, including an actual frozen Phase0 encoder."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import behavior_aware_source as source
import factorized_wm_data as raw_data
import paired_latent_data as paired
from phase0_wm_data import NativeTargets, dataset_digest, load_dataset, pack_observations, prepare_rows
from phase0_wm_models import DynamicsPair, LatentEncoder, TokenEmbedding, VerifierReadout
from test_behavior_aware_source import source_archive
from test_factorized_wm_data import archive_run, make_run, read_jsonl, write_jsonl


def small_row(uid='train', question='train', length=2, teacher=True):
    return dict(uid=uid, question=question, length=length, accepted=1,
                ids=torch.tensor([[1, 3]] * length), c=torch.zeros(length, 20),
                context=torch.zeros(8), teacher_hidden=torch.arange(length * 32).reshape(length, 32).float()
                if teacher else None,
                candidate_embedding=torch.arange(length * 64).reshape(length, 64).float(),
                action='R', parent_uid='evaluation-only', parent_length=64,
                parent_accepted=63, parent_token_changed=True)


class PairedHelpersTests(unittest.TestCase):
    def test_normalization_fits_train_only_and_preserves_raw(self):
        train = small_row()
        missing = small_row('missing', teacher=False)
        val = small_row('val', 'val')
        test = small_row('test', 'test')
        rows = {r['uid']: r for r in (train, missing, val, test)}
        for field, width in (('teacher_hidden', 32), ('candidate_embedding', 64)):
            with self.subTest(field=field):
                original = train[field].clone()
                expected = torch.cat([r[field] for r in (train, missing) if r[field] is not None])
                report = paired.fit_normalization(rows, ['train'], field, width)
                torch.testing.assert_close(torch.tensor(report['mean']), expected.mean(0))
                torch.testing.assert_close(torch.tensor(report['std']), expected.std(0, unbiased=False))
                val[field].fill_(1e6)
                test[field].fill_(-1e9)
                self.assertEqual(report, paired.fit_normalization(rows, ['train'], field, width))
                torch.testing.assert_close(train[field], original)
                transformed = paired.apply_normalization(train[field], report)
                torch.testing.assert_close(transformed.mean(0), torch.zeros(width))
                self.assertEqual(report['fit_split'], 'train')
                self.assertEqual(report['questions'], ['train'])
        self.assertEqual(paired.fit_normalization(rows, ['train'], 'teacher_hidden', 32)['tokens'], 2)
        self.assertEqual(paired.fit_normalization(rows, ['train'], 'candidate_embedding', 64)['tokens'], 4)

    def test_absent_train_teachers_do_not_fit_held_out_teachers(self):
        rows = {'train': small_row(teacher=False), 'val': small_row('val', 'val')}
        report = paired.fit_normalization(rows, ['train'], 'teacher_hidden', 32)
        self.assertFalse(report['fitted'])
        self.assertEqual(report['tokens'], 0)
        self.assertEqual(report['mean'], [0.] * 32)
        self.assertEqual(report['std'], [1.] * 32)
        self.assertIsNone(paired.apply_normalization(None, report))

    def test_first_three_teacher_columns_never_enter_hidden_input(self):
        features = torch.arange(70).reshape(2, 35).float()
        result = paired.project_teacher_hidden(features, 2)
        torch.testing.assert_close(result, features[:, 3:35])
        features[:, :3] = 1e30
        torch.testing.assert_close(result, paired.project_teacher_hidden(features, 2))
        self.assertEqual(tuple(result.shape), (2, 32))
        self.assertIsNone(paired.project_teacher_hidden(None, 2))
        with self.assertRaisesRegex(ValueError, 'shape'):
            paired.project_teacher_hidden(torch.zeros(2, 34), 2)
        features[0, 3] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            paired.project_teacher_hidden(features, 2)

    def test_candidate_embeddings_use_stop_column_and_frozen_weights(self):
        table = torch.arange(6 * 64).reshape(6, 64).half()
        original = table.clone()
        embedding = dict(weights=table, vocabulary=None, learned=False)
        ids = torch.tensor([[1, 3], [2, 4]])
        result = paired.lookup_candidate_embedding(embedding, ids)
        torch.testing.assert_close(result, table[[3, 4]].float())
        ids[:, 0] = 5
        torch.testing.assert_close(result, paired.lookup_candidate_embedding(embedding, ids))
        result.zero_()
        torch.testing.assert_close(table, original)
        with self.assertRaisesRegex(ValueError, 'pretrained'):
            paired.lookup_candidate_embedding(dict(embedding, learned=True), ids)
        with self.assertRaisesRegex(ValueError, 'pretrained'):
            paired.lookup_candidate_embedding(dict(embedding, vocabulary=[1, 2]), ids)
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            paired.lookup_candidate_embedding(embedding, torch.tensor([[0, 6]]))

    def test_pack_shapes_masks_and_unchanged_frozen_latents(self):
        teacher = small_row()
        missing = small_row('missing', teacher=False, length=3)
        missing['accepted'] = None
        cache = {'train': torch.randn(64, 128).half(), 'missing': torch.randn(64, 128).half()}
        before = {k: v.clone() for k, v in cache.items()}
        packed = paired.pack_paired_rows([teacher, missing], cache, 'cpu')
        for field, shape in dict(z_D=(2, 64, 128), c=(2, 64, 20), context=(2, 8),
                                 lengths=(2,), teacher_hidden=(2, 64, 32),
                                 candidate_embedding=(2, 64, 64), accepted=(2,),
                                 teacher_valid=(2, 64)).items():
            self.assertEqual(tuple(packed[field].shape), shape)
            self.assertEqual(packed[field].dtype, torch.bool if field == 'teacher_valid' else
                             torch.long if field in ('accepted', 'lengths') else torch.float32)
        self.assertEqual(packed['lengths'].tolist(), [2, 3])
        self.assertEqual(packed['accepted'].tolist(), [1, -1])
        self.assertEqual(packed['teacher_valid'].sum(1).tolist(), [2, 0])
        self.assertFalse(packed['teacher_hidden'][1].any())
        self.assertTrue(packed['candidate_embedding'][1, :3].any())
        self.assertFalse(packed['candidate_embedding'][1, 3:].any())
        for i, uid in enumerate(('train', 'missing')):
            torch.testing.assert_close(packed['z_D'][i], before[uid].float(), rtol=0, atol=0)
            torch.testing.assert_close(cache[uid], before[uid], rtol=0, atol=0)
        packed['z_D'].zero_()
        torch.testing.assert_close(cache['train'], before['train'], rtol=0, atol=0)

    def test_D_inputs_ignore_labels_teacher_and_all_parent_truth(self):
        class NoEvaluationReads(dict):
            def __getitem__(self, key):
                if key.startswith('parent_') or key == 'action':
                    raise AssertionError(f'evaluation metadata read: {key}')
                return super().__getitem__(key)

        class SourceCache(dict):
            def __getitem__(self, uid):
                if uid != 'train':
                    raise AssertionError('Future child latent read')
                return super().__getitem__(uid)

        row = NoEvaluationReads(small_row())
        cache = SourceCache(train=torch.randn(64, 128))
        before = paired.pack_paired_rows([row], cache, 'cpu')
        row['teacher_hidden'] = torch.full((2, 32), 1e7)
        row['candidate_embedding'] = torch.full((2, 64), -1e6)
        row['accepted'] = 0
        row['parent_accepted'] = object()
        after = paired.pack_paired_rows([row], cache, 'cpu')
        self.assertEqual(paired.D_INPUT_FIELDS, ('z_D', 'c', 'context', 'lengths'))
        self.assertTrue(set(paired.D_INPUT_FIELDS).isdisjoint(
            {'accepted', 'teacher_hidden', 'candidate_embedding', 'teacher_valid', 'parent_accepted', 'action'}))
        for key in paired.D_INPUT_FIELDS:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before['teacher_hidden'], after['teacher_hidden']))
        self.assertFalse(torch.equal(before['candidate_embedding'], after['candidate_embedding']))

    def test_pack_rejects_silent_truncation_and_malformed_teachers(self):
        row = small_row()
        cache = {'train': torch.zeros(64, 128)}
        for field, value in (('length', 65), ('teacher_hidden', torch.zeros(2, 35)),
                             ('candidate_embedding', torch.zeros(2, 63)), ('accepted', -1),
                             ('teacher_hidden', torch.full((2, 32), float('inf')))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired.pack_paired_rows([dict(row, **{field: value})], cache, 'cpu')
        with self.assertRaisesRegex(ValueError, 'empty'):
            paired.pack_paired_rows([], cache, 'cpu')


class DynamicsArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'upload.dat'
        source_archive(self.archive)
        self.weights = {'R.weight': torch.arange(12).reshape(3, 4).float()}
        buffer = io.BytesIO()
        torch.save(self.weights, buffer)
        self.bytes = buffer.getvalue()

    def tearDown(self):
        self.tmp.cleanup()

    def add_weights(self):
        with zipfile.ZipFile(self.archive, 'a') as archive:
            archive.writestr('arbitrary/' + paired.DYNAMICS_MEMBER, self.bytes)
            archive.writestr('arbitrary/huge-unrelated.bin', b'not extracted')

    def test_zip_and_folder_materialize_only_optional_weights_safely(self):
        self.add_weights()
        original_load = torch.load
        with patch.object(torch, 'load', wraps=original_load) as load:
            weights, report = paired.materialize_dynamics(self.archive, self.root / 'cache')
        load.assert_called_once()
        self.assertTrue(load.call_args.kwargs['weights_only'])
        self.assertEqual(load.call_args.kwargs['map_location'], 'cpu')
        torch.testing.assert_close(weights['R.weight'], self.weights['R.weight'])
        self.assertEqual(report['sha256'], hashlib.sha256(self.bytes).hexdigest())
        cached = list((self.root / 'cache').rglob('*'))
        self.assertEqual([p.name for p in cached if p.is_file()], ['best.pt'])
        self.assertEqual(Path(report['cache_path']).read_bytes(), self.bytes)
        folder = source.materialize_phase0(self.archive, self.root / 'folder')
        member = folder / paired.DYNAMICS_MEMBER
        member.parent.mkdir(parents=True)
        member.write_bytes(self.bytes)
        weights2, report2 = paired.materialize_dynamics(folder, self.root / 'cache')
        torch.testing.assert_close(weights2['R.weight'], weights['R.weight'])
        self.assertEqual(report2['sha256'], report['sha256'])

    def test_absent_weights_do_not_reuse_stale_cached_weights(self):
        cache = self.root / 'cache'
        weights, report = paired.materialize_dynamics(self.archive, cache)
        self.assertIsNone(weights)
        self.assertFalse(report['available'])
        self.add_weights()
        self.assertIsNotNone(paired.materialize_dynamics(self.archive, cache)[0])
        absent = self.root / 'absent.zip'
        source_archive(absent)
        self.assertIsNone(paired.materialize_dynamics(absent, cache)[0])

    def test_unsafe_archives_rejected_before_any_cache_write(self):
        for index, name in enumerate(('../escape', 'C:/outside', '/outside',
                                      'arbitrary/../../escape', 'arbitrary\\escape',
                                      'arbitrary/file:stream', 'arbitrary/./alias')):
            with self.subTest(name=name):
                archive_path = self.root / f'unsafe{index}.zip'
                source_archive(archive_path)
                # Windows ZipInfo constructors normalize backslashes. Preserve
                # the actual unsafe stored spelling for this adversarial ZIP.
                info = zipfile.ZipInfo('placeholder')
                info.filename = info.orig_filename = name
                with zipfile.ZipFile(archive_path, 'a') as archive:
                    archive.writestr(info, b'bad')
                cache = self.root / f'cache{index}'
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    paired.materialize_dynamics(archive_path, cache)
                self.assertFalse(cache.exists())

    def test_duplicate_and_symlink_archive_members_rejected(self):
        self.add_weights()
        with zipfile.ZipFile(self.archive, 'a') as archive, warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('arbitrary/' + paired.DYNAMICS_MEMBER, self.bytes)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired.materialize_dynamics(self.archive, self.root / 'cache')
        link_zip = self.root / 'link.zip'
        source_archive(link_zip)
        link = zipfile.ZipInfo('arbitrary/' + paired.DYNAMICS_MEMBER)
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(link_zip, 'a') as archive:
            archive.writestr(link, 'outside.pt')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            paired.materialize_dynamics(link_zip, self.root / 'cache')

    def test_present_invalid_checkpoint_is_not_treated_as_absent(self):
        buffer = io.BytesIO()
        torch.save({'bad': torch.tensor(float('nan'))}, buffer)
        with zipfile.ZipFile(self.archive, 'a') as archive:
            archive.writestr('arbitrary/' + paired.DYNAMICS_MEMBER, buffer.getvalue())
        with self.assertRaisesRegex(ValueError, 'finite tensor state dict'):
            paired.materialize_dynamics(self.archive, self.root / 'cache')
        self.assertFalse((self.root / 'cache').exists())


class PairedModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_oracle_student_forwards_and_objectives_accept_packed_fields(self):
        from paired_latent_models import ORACLE_METHODS, STUDENT_METHODS, objective
        from run_paired_native_latent import call_model, make_model

        cfg = dict(latent_dim=128, dropout=0., student_layers=1,
                   bridge_state_weight=1., bridge_behavior_weight=1., distill_weight=.1)
        rows = [small_row(), small_row('second', length=3)]
        batch = paired.pack_paired_rows(rows, {r['uid']: torch.randn(64, 128) for r in rows}, 'cpu')
        native = make_model('V_joint', cfg).eval().requires_grad_(False)
        target_z = native.encode(batch['teacher_hidden'], batch['candidate_embedding'], batch['lengths'])
        for method in ORACLE_METHODS + STUDENT_METHODS:
            with self.subTest(method=method):
                model = make_model(method, cfg, native).train()
                prediction = call_model(model, method, batch)
                self.assertEqual(tuple(prediction['hazard'].shape), (2, 64))
                self.assertEqual(tuple(prediction['z_V'].shape), (2, 64, 128))
                loss, parts = objective(method, prediction, batch, target_z, cfg)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if method.startswith('Bridge'):
                    self.assertTrue(all(p.grad is None for p in model.readout.parameters()))
        self.assertTrue(all(p.grad is None for p in native.parameters()))

    def test_student_dispatch_cannot_access_candidate_teacher_or_acceptance(self):
        from paired_latent_models import STUDENT_METHODS
        from run_paired_native_latent import call_model, make_model

        class DOnly(dict):
            def __getitem__(self, key):
                if key not in paired.D_INPUT_FIELDS:
                    raise AssertionError(f'Student read privileged field {key}')
                return super().__getitem__(key)

        row = small_row()
        batch = DOnly(paired.pack_paired_rows([row], {'train': torch.randn(64, 128)}, 'cpu'))
        cfg = dict(latent_dim=128, dropout=0., student_layers=1)
        with torch.no_grad():
            for method in STUDENT_METHODS:
                with self.subTest(method=method):
                    model = make_model(method, cfg).eval()
                    prediction = call_model(model, method, batch)
                    selected = model(**{k: batch[k] for k in paired.D_INPUT_FIELDS})
                    torch.testing.assert_close(prediction['hazard'], selected['hazard'], rtol=0, atol=0)

    def test_objective_masks_missing_teacher_and_unresolved_acceptance_independently(self):
        from paired_latent_models import objective

        rows = [small_row(), small_row('missing', teacher=False, length=3)]
        rows[1]['accepted'] = None
        batch = paired.pack_paired_rows(rows, {r['uid']: torch.zeros(64, 128) for r in rows}, 'cpu')
        prediction = dict(hazard=torch.zeros(2, 64, requires_grad=True),
                          z_V=torch.zeros(2, 64, 128, requires_grad=True))
        target = torch.ones(2, 64, 128, requires_grad=True)
        cfg = dict(bridge_state_weight=1., bridge_behavior_weight=1., distill_weight=1.)
        loss, parts = objective('Bridge_behavior', prediction, batch, target, cfg)
        changed_target = target.detach().clone()
        changed_target[1] = 1e6
        changed_target[0, rows[0]['length']:] = -1e6
        changed_prediction = dict(prediction, hazard=prediction['hazard'].detach().clone())
        changed_prediction['hazard'][1] = 100
        changed_loss, changed_parts = objective('Bridge_behavior', changed_prediction, batch, changed_target, cfg)
        torch.testing.assert_close(loss, changed_loss, rtol=0, atol=0)
        torch.testing.assert_close(parts['latent'], torch.tensor(.5), rtol=0, atol=0)
        torch.testing.assert_close(parts['behavioral'], changed_parts['behavioral'], rtol=0, atol=0)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertFalse(prediction['hazard'].grad[1].any())
        self.assertFalse(prediction['z_V'].grad[1].any())

    def test_string_root_R_E_metadata_is_sampler_and_metrics_compatible(self):
        from paired_latent_metrics import build_report
        from paired_latent_models import outputs
        from run_paired_native_latent import QuestionSampler, prediction_record

        rows = {name: dict(small_row(name), action=name) for name in ('root', 'R', 'E')}
        rows['root'].update(parent_uid=None, parent_length=None, parent_accepted=None,
                            parent_token_changed=None)
        sampler = QuestionSampler(list(rows), rows, seed=17)
        self.assertEqual(sampler.actions, ['E', 'R', 'root'])
        self.assertEqual(set(sampler.sample(100)), set(rows))
        result = outputs(torch.zeros(3, 64), torch.tensor([2, 2, 2]))
        records = [prediction_record(row, 17, 'Direct', result, i)
                   for i, row in enumerate(rows.values())]
        self.assertEqual({r['action'] for r in records}, {'root', 'R', 'E'})
        self.assertIsInstance(build_report(records, bootstrap_samples=2), dict)


class PairedSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.trace = make_run(cls.root / 'original', questions=100)
        teachers = read_jsonl(cls.trace, 'teacher_targets.jsonl')
        for row in teachers:
            question = int(row['state_id'].split('s')[0][1:])
            row['features'] = [[1000., .8, 1.] + [question + i / 8 + j / 32 for j in range(32)]
                               for i in range(len(row['features']))]
        write_jsonl(cls.trace, 'teacher_targets.jsonl', teachers)
        dataset = load_dataset(cls.trace)
        cls.split = raw_data.split_questions(dataset.question_ids)
        # Teachers without acceptance still fit train normalization, but cannot
        # enter paired verifier observation IDs on either train or validation.
        cls.unlabeled_teacher_questions = {cls.split[name][0] for name in ('train', 'val')}
        labels = read_jsonl(cls.trace, 'labels.jsonl')
        for label in labels:
            question = label['state_id'].split('s')[0].replace('q', 'gsm8k:')
            if question in cls.unlabeled_teacher_questions and label['state_id'].endswith('s1'):
                label['label_valid'] = False
        write_jsonl(cls.trace, 'labels.jsonl', labels)
        dataset = load_dataset(cls.trace)
        cls.cfg = dict(num_questions=100, latent_dims=[128], token_embedding='pretrained',
                       encoder_layers=1, dynamics_layers=1, dropout=0., prefix_max_tokens=64)
        # This deterministic synthetic table stands in for a saved projected
        # pretrained artifact; the loader must use its values without resampling.
        table = (torch.arange(3400).float()[:, None] / 100 + torch.arange(64)[None] / 64).half()
        embedding = dict(weights=table, vocabulary=None, learned=False)
        targets = NativeTargets(torch.zeros(3, 1536), torch.zeros(3, 1536, 1), torch.ones(3, 1))
        preprocessing = dict(embedding=embedding, native_targets=dict(
            mean=targets.mean, basis=targets.basis, std=targets.std))
        torch.manual_seed(17)
        encoder = LatentEncoder(TokenEmbedding(**embedding), 128, 1, 0.).eval().requires_grad_(False)
        rows = prepare_rows(dataset, targets)
        cls.cachez = {}
        with torch.no_grad():
            values = list(rows.values())
            for start in range(0, len(values), 16):
                group = values[start:start + 16]
                latent = encoder(pack_observations(group, 'cpu', 64)).z
                cls.cachez.update({r['uid']: latent[i].half().clone() for i, r in enumerate(group)})
        cls.phase0 = cls.root / 'phase0'
        (cls.phase0 / 'latent_128/stage_A').mkdir(parents=True)
        hashes = {name: hashlib.sha256((Path(source.__file__).parent / name).read_bytes()
                                      .replace(b'\r\n', b'\n')).hexdigest()
                  for name in ('phase0_wm_models.py', 'phase0_wm_data.py', 'factorized_wm_data.py')}
        fingerprint = hashlib.sha256(json.dumps(dict(config=cls.cfg, source_hashes=hashes,
                                                     data=dataset_digest(dataset)), sort_keys=True).encode()).hexdigest()
        metadata = {'config.json': cls.cfg, 'summary.json': dict(schema='latent_world_model_phase0_v1'),
                    'split_manifest.json': cls.split, 'source_hashes.json': hashes,
                    'study_manifest.json': dict(fingerprint=fingerprint),
                    'latent_128/stage_A/complete.json': dict(status='complete')}
        for name, value in metadata.items():
            (cls.phase0 / name).write_text(json.dumps(value), encoding='utf-8')
        torch.save(preprocessing, cls.phase0 / 'preprocessing.pt')
        torch.save(dict(encoder=encoder.state_dict(), readout=VerifierReadout(128, 1, 0.).state_dict(),
                        reconstruction={'test': torch.zeros(1)}), cls.phase0 / 'latent_128/stage_A/best.pt')
        torch.save(cls.cachez, cls.phase0 / 'latent_128/frozen_latents.pt')
        cls.original_zip = archive_run(cls.trace, cls.root / 'original.dat', 'arbitrary/raw/')
        cls.phase0_zip = archive_run(cls.phase0, cls.root / 'phase0.dat', 'arbitrary/phase0/')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_real_folder_source_exact_split_frozen_verification_and_cached_reload(self):
        cache = self.root / 'folder-cache'
        original_source = source.prepare_source
        original_shard = raw_data._shard_arrays
        with patch.object(source, 'prepare_source', wraps=original_source) as prepare, \
                patch.object(raw_data, '_shard_arrays', wraps=original_shard) as shard:
            result = paired.prepare_paired_source(self.trace, self.phase0, cache)
        prepare.assert_called_once_with(self.trace, self.phase0, cache,
                                        verification_samples=32, device='cpu')
        self.assertEqual(shard.call_count, 1)  # sidecar reload hits the trusted raw cache
        self.assertEqual(result['split'], self.split)
        self.assertEqual(result['provenance']['checked_encoder_train_samples'], 32)
        self.assertEqual(result['provenance']['test_encoder_verifications'], 0)
        self.assertEqual(result['provenance']['LLM_forwards'], 0)
        self.assertEqual(result['summary_marker'], 'projected32_teacher_pilot')
        self.assertFalse(result['provenance']['full_qwen_compression_claimed'])
        self.assertEqual((result['teacher_input_dim'], result['embedding_dim'], result['latent_dim']), (32, 64, 128))
        self.assertEqual(result['teacher_states_per_split'], dict(train=70, val=15, test=15))
        self.assertIsNone(result['dynamics_weights'])
        self.assertFalse(result['dynamics_provenance']['available'])
        self.assertEqual(result['data_audit']['full_teacher_states'], 100)
        for name, questions in result['split'].items():
            self.assertEqual(len(result['observation_ids'][name]), len(questions) - int(name != 'test'))
            self.assertTrue(all(result['rows'][uid]['accepted'] is not None and
                                result['rows'][uid]['teacher_hidden'] is not None and
                                result['rows'][uid]['question'] in questions
                                for uid in result['observation_ids'][name]))
        for uid, row in result['rows'].items():
            self.assertTrue({'teacher', 'native_target', 'hidden', 'teacher_features', 'teacher_margin'}
                            .isdisjoint(row))
            torch.testing.assert_close(result['cachez'][uid], self.cachez[uid], rtol=0, atol=0)
            if uid.endswith('s0'):
                self.assertEqual(row['action'], 'root')
                self.assertIsNone(row['teacher_hidden'])
                self.assertIsNotNone(row['candidate_embedding'])
            else:
                self.assertEqual(row['action'], 'E' if uid.endswith('s2') else 'R')
                self.assertFalse(row['parent_token_changed'])
        train_uid = next(uid for uid, row in result['rows'].items() if row['question'] in self.split['train'])
        packed = paired.pack_paired_rows([result['rows'][train_uid]], result['cachez'], 'cpu')
        torch.testing.assert_close(packed['z_D'][0], self.cachez[train_uid].float(), rtol=0, atol=0)
        train_teachers = torch.cat([r['teacher_hidden'] for r in result['rows'].values()
                                   if r['question'] in self.split['train'] and r['teacher_hidden'] is not None])
        torch.testing.assert_close(train_teachers.mean(0), torch.zeros(32), atol=2e-6, rtol=0)
        torch.testing.assert_close(train_teachers.std(0, unbiased=False), torch.ones(32), atol=2e-6, rtol=0)
        self.assertEqual(result['normalization']['teacher_hidden']['tokens'], 70 * 8)
        self.assertEqual(len(result['provenance']['teacher_checksum']), 64)
        self.assertEqual(json.loads(json.dumps(result['normalization'], allow_nan=False)), result['normalization'])
        # Worker payloads must round-trip through restricted torch loading while
        # preserving normalized teacher fields, native c and frozen cache values.
        buffer = io.BytesIO()
        torch.save(result, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu', weights_only=True)
        self.assertEqual(restored['normalization'], result['normalization'])
        restored_batch = paired.pack_paired_rows([restored['rows'][train_uid]], restored['cachez'], 'cpu')
        for field in packed:
            torch.testing.assert_close(restored_batch[field], packed[field], rtol=0, atol=0)
        with patch.object(raw_data, '_shard_arrays', side_effect=AssertionError('Raw cache not reused')):
            cached = load_dataset(self.trace, cache_dir=cache / 'raw')
        self.assertTrue(all(isinstance(action, str) and action in ('R', 'E') for _, _, action in cached.edges))
        for uid, state in cached.states.items():
            torch.testing.assert_close(result['rows'][uid]['c'], torch.cat([state.scalars, state.surface[:, 32:]], -1),
                                       rtol=0, atol=0)
            self.assertEqual(result['rows'][uid]['teacher_hidden'] is None, state.teacher_features is None)

    def test_real_zip_source_optional_H1_and_no_whole_archive_extraction(self):
        weights = DynamicsPair(128, self.cfg['dynamics_layers'], self.cfg['dropout']).state_dict()
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        archive = self.root / 'with-dynamics.zip'
        shutil.copyfile(self.phase0_zip, archive)
        with zipfile.ZipFile(archive, 'a') as z:
            z.writestr('arbitrary/phase0/' + paired.DYNAMICS_MEMBER, buffer.getvalue())
            z.writestr('arbitrary/phase0/unrelated.txt', 'never materialized')
        cache = self.root / 'zip-cache'
        result = paired.prepare_paired_source(self.original_zip, archive, cache, verification_samples=3)
        self.assertEqual(result['split'], self.split)
        self.assertEqual(result['provenance']['checked_encoder_train_samples'], 3)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in result['dynamics_weights'].values()))
        self.assertEqual(set(result['dynamics_weights']), set(weights))
        dynamics = DynamicsPair(128, result['source_config']['dynamics_layers'], result['source_config']['dropout'])
        dynamics.load_state_dict(result['dynamics_weights'], strict=True)
        self.assertEqual(source.weight_hash(dynamics.state_dict()), source.weight_hash(weights))
        self.assertEqual(result['dynamics_provenance']['horizon'], 1)
        files = {p.relative_to(cache / 'phase0').as_posix() for p in (cache / 'phase0').rglob('*') if p.is_file()}
        self.assertEqual(files, set(source.REQUIRED))
        self.assertFalse(list(cache.rglob('*.npz')))
        self.assertFalse(list(cache.rglob('unrelated.txt')))
        from run_paired_native_latent import evaluate_rollouts, make_model
        cfg = dict(latent_dim=128, dropout=0., student_layers=1, batch_size=8)
        models = {method: make_model(method, cfg).eval().requires_grad_(False)
                  for method in ('V_joint', 'Direct')}
        before, status = evaluate_rollouts(models, result, cfg, 'cpu', seed=17)
        self.assertEqual(status['paths_by_horizon'], {'1': 15, '2': 0, '3': 0})
        self.assertFalse(status['dynamics_retrained'])
        self.assertEqual(status['real_intermediate_state_injections'], 0)
        # Missing future teachers prevent H2/H3 oracle comparisons. For observed
        # endpoints, changing real child teachers/STOP embeddings affects only
        # privileged scoring; the free D rollout predictions stay identical.
        for uid in result['observation_ids']['test']:
            result['rows'][uid]['teacher_hidden'] *= -3
            result['rows'][uid]['candidate_embedding'] += 100
        after, _ = evaluate_rollouts(models, result, cfg, 'cpu', seed=17)
        student_before = [r for r in before if r['method'] in ('Direct', 'Frozen_D_readout')]
        student_after = [r for r in after if r['method'] in ('Direct', 'Frozen_D_readout')]
        self.assertEqual(len(student_before), 30)
        for a, b in zip(student_before, student_after):
            for field in ('uid', 'method', 'K_pred', 'q_pred', 'hazard_pred'):
                self.assertEqual(a[field], b[field])

    def test_changed_latent_cache_rejected_by_original_train_verification(self):
        folder = self.root / 'tampered-latents'
        shutil.copytree(self.phase0, folder)
        cachez = {uid: z.clone() for uid, z in self.cachez.items()}
        train_uid = sorted(uid for uid in cachez if uid.split('s')[0].replace('q', 'gsm8k:') in self.split['train'])[0]
        cachez[train_uid][0, 0] += 100
        torch.save(cachez, folder / 'latent_128/frozen_latents.pt')
        with self.assertRaisesRegex(ValueError, 'Frozen latent cache differs'):
            paired.prepare_paired_source(self.trace, folder, self.root / 'tamper-cache', verification_samples=1)

    def test_original_fingerprint_and_source_hash_checks_cannot_be_bypassed(self):
        folder = self.root / 'tampered-hashes'
        shutil.copytree(self.phase0, folder)
        hashes = json.loads((folder / 'source_hashes.json').read_text())
        hashes['phase0_wm_models.py'] = 'not the frozen implementation'
        (folder / 'source_hashes.json').write_text(json.dumps(hashes))
        with self.assertRaisesRegex(ValueError, 'source implementation hash mismatch'):
            paired.prepare_paired_source(self.trace, folder, self.root / 'hash-cache', verification_samples=0)
        fingerprint_folder = self.root / 'tampered-fingerprint'
        shutil.copytree(self.phase0, fingerprint_folder)
        (fingerprint_folder / 'study_manifest.json').write_text(json.dumps(dict(fingerprint='wrong')))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            paired.prepare_paired_source(self.trace, fingerprint_folder, self.root / 'fingerprint-cache', verification_samples=0)


if __name__ == '__main__':
    unittest.main()
