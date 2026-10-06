"""Offline sidecar checks: no source downloads, GPU queries or training."""
import builtins
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_paired_native_latent as launcher


def config_fixture():
    # Independent of the config owned by the main study, including while it is
    # being edited. This tests the sidecar, not the runner's training defaults.
    config = copy.deepcopy(launcher.DEFAULT_CONFIG)
    config['protocol'] = {'questions': 100, 'split': [70, 15, 15]}
    config['projected_dim'] = 32
    return config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def zip_tree(source, archive):
    with zipfile.ZipFile(archive, 'w') as zipped:
        for path in source.rglob('*'):
            if path.is_file():
                zipped.write(path, 'wrapper/result/' + path.relative_to(source).as_posix())
    return archive


def original_result(root):
    root.mkdir(parents=True)
    write_json(root / 'summary.json', {
        'schema': 'interactive_acceptance_two_source_v1',
        'status': 'complete', 'questions_completed': 100,
    })
    for name in ('states.jsonl', 'edges.jsonl', 'labels.jsonl', 'teacher_targets.jsonl'):
        (root / name).write_text('', encoding='utf-8')
    (root / 'experience').mkdir()
    (root / 'experience/trace.npz').write_bytes(b'metadata fixture, never loaded')
    return root


def phase0_result(root, pretrained=True):
    root.mkdir(parents=True)
    documents = {
        'config.json': {'num_questions': 100, 'latent_dims': [64, 128],
                        'token_embedding': 'pretrained' if pretrained else 'learned'},
        'summary.json': {'schema': 'latent_world_model_phase0_v1', 'status': 'complete'},
        'split_manifest.json': {'train': list(range(70)), 'val': list(range(70, 85)),
                                'test': list(range(85, 100))},
        'study_manifest.json': {'fingerprint': 'a' * 64},
        'source_hashes.json': {'phase0_wm_models.py': 'b' * 64},
        'latent_128/stage_A/complete.json': {'status': 'complete'},
    }
    for name, value in documents.items():
        write_json(root / name, value)
    for name in ('preprocessing.pt', 'latent_128/stage_A/best.pt', 'latent_128/frozen_latents.pt'):
        (root / name).write_bytes(b'fixture checkpoint: do not load')
    return root


def prior_result(root, config=None):
    config = config if config is not None else config_fixture()
    write_json(root / 'config.json', config)
    write_json(root / 'study_manifest.json', {
        'schema': 'paired_native_latent_v1', 'signature': 'c' * 64,
        'config': config, 'status': 'running',
    })
    for name in ('best.pt', 'last.pt'):
        checkpoint = root / 'jobs/bridge_seed42' / name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'model/optimizer/RNG fixture: runner validates payload')
    return root


def resolver_dependencies():
    """The real metadata reader needs no tensor code; stub only its imports."""
    modules = {}
    for name, exports in {
        'torch': (),
        'phase0_wm_data': ('load_dataset', 'dataset_digest', 'NativeTargets',
                           'prepare_rows', 'pack_observations'),
        'phase0_wm_models': ('TokenEmbedding', 'LatentEncoder'),
        'run_latent_wm_phase0': ('select_paths', 'cpu_weights'),
    }.items():
        module = ModuleType(name)
        for export in exports:
            setattr(module, export, lambda *a, **k: (_ for _ in ()).throw(AssertionError('tensor work')))
        modules[name] = module
    return modules


class ConfigImportAndCellTests(unittest.TestCase):
    def test_full_preserves_main_adjusted_budgets_and_nested_config(self):
        base = config_fixture()
        base.update(oracle_updates=777, max_updates=999, eval_every=33, workers=1,
                    seeds=[44], batch_size=7, bridge_state_weight=2.5)
        before = copy.deepcopy(base)
        result = launcher.build_config(base, {})
        self.assertEqual(result, base)
        result['protocol']['split'].append(999)
        self.assertEqual(base, before)

    def test_missing_standard_fields_use_defaults_without_editing_source(self):
        self.assertEqual(launcher.build_config({}, {}), launcher.DEFAULT_CONFIG)

    def test_smoke_changes_only_budgets_and_keeps_full_split(self):
        base = config_fixture()
        result = launcher.build_config(base, {'MODE': 'smoke'})
        for key, value in launcher.SMOKE_OVERRIDES.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['num_questions'], 100)
        self.assertEqual(result['protocol'], base['protocol'])
        self.assertEqual(result['workers'], 2)
        self.assertEqual(base['oracle_updates'], 1000)
        result = launcher.build_config(base, {'MODE': 'smoke', 'SMOKE_CONFIG': {
            'oracle_updates': 4, 'max_updates': 6, 'seeds': [44], 'pipeline_check_only': False,
        }})
        self.assertEqual((result['oracle_updates'], result['max_updates'], result['seeds']), (4, 6, [44]))
        self.assertTrue(result['pipeline_check_only'])

    def test_invalid_mode_split_schema_and_question_overrides(self):
        for scope in ({'MODE': 'typo'}, {'NUM_QUESTIONS': 8}, {'NUM_QUESTIONS': 100.0},
                      {'MODE': 'smoke', 'SMOKE_CONFIG': {'num_questions': 8}},
                      {'MODE': 'smoke', 'SMOKE_CONFIG': []}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                launcher.build_config(config_fixture(), scope)
        for key, value in (('num_questions', 8), ('schema', 'behavior_aware_h1_v1'), ('latent_dim', 32),
                           ('protocol', {'split': [80, 10, 10]})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.build_config(dict(config_fixture(), **{key: value}), {})

    def test_import_cannot_change_cwd_start_process_download_or_import_torch(self):
        spec = importlib.util.spec_from_file_location('_paired_launcher_import', ROOT / 'kaggle_paired_native_latent.py')
        module = importlib.util.module_from_spec(spec)
        real_import = builtins.__import__
        def import_guard(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise AssertionError('torch/GPU import')
            return real_import(name, *args, **kwargs)
        with patch('os.chdir', side_effect=AssertionError('chdir')), \
                patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir')), \
                patch('subprocess.run', side_effect=AssertionError('process')), \
                patch('subprocess.check_output', side_effect=AssertionError('download')), \
                patch('builtins.__import__', side_effect=import_guard), \
                patch.object(launcher.behavior, 'launch', side_effect=AssertionError('old study')), \
                patch.object(launcher.phase0, 'launch', side_effect=AssertionError('old study')):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.launch))
        self.assertIs(module.resolve_phase0_input, launcher.behavior.resolve_phase0_input)
        self.assertIs(module.download_source, launcher.phase0.download_source)

    def test_device_placement_does_not_query_gpus(self):
        self.assertEqual(launcher.select_devices({}, config_fixture()), ('cuda:0', 'cuda:1'))
        self.assertEqual(launcher.select_devices({'DEVICES': 'cpu'}, config_fixture()), ('cpu', 'cpu'))
        self.assertEqual(launcher.select_devices({'DEVICES': ['cpu']}, config_fixture()), ('cpu', 'cpu'))
        self.assertEqual(launcher.select_devices({'DEVICES': 'cuda:0,cuda:1'}, config_fixture()), ('cuda:0', 'cuda:1'))
        self.assertEqual(launcher.select_devices({'DEVICES': ['cuda:1']}, {'workers': 1}), ('cuda:1',))
        for devices in (['cuda:2', 'cpu'], ['cuda:0'], [], None):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                launcher.select_devices({'DEVICES': devices}, config_fixture())

    def test_source_fetch_is_detached_for_branch_and_pinned_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, ref in enumerate((launcher.DEFAULT_SOURCE_REF, 'a' * 40)):
                repo = Path(directory) / str(index)
                with patch.object(launcher.phase0, 'run_command') as run, \
                        patch('subprocess.check_output', return_value='b' * 40 + '\n'):
                    self.assertEqual(launcher.download_source(repo, ref, {}), 'b' * 40)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[2], ['git', '-C', repo, 'fetch', '--depth', '1', 'origin', ref])
                self.assertEqual(commands[3][-2:], ['--detach', 'FETCH_HEAD'])

    def test_copy_cell_reads_same_ref_and_executes_main_once_with_parameters(self):
        cell = (ROOT / 'PAIRED_NATIVE_LATENT.md').read_text(encoding='utf-8').split('```python\n', 1)[1].split('```', 1)[0]
        payloads = {name: (ROOT / name).read_text(encoding='utf-8') for name in
                    ('kaggle_latent_wm_phase0.py', 'kaggle_behavior_aware_h1.py', 'kaggle_paired_native_latent.py')}
        payloads['kaggle_paired_native_latent.py'] = payloads['kaggle_paired_native_latent.py'].replace(
            'if __name__ == "__main__":',
            'from unittest.mock import Mock\nlaunch = Mock(return_value="mock-output")\nif __name__ == "__main__":')
        scope = {'SOURCE_REF': 'a' * 40, 'RUN_DIR': '/original.payload', 'PHASE0_INPUT': '/phase0-folder',
                 'MODE': 'smoke', 'DEVICES': 'cpu', 'RESUME_INPUT': '/prior.zip',
                 'SMOKE_CONFIG': {'oracle_updates': 4}}
        def read(url, **kwargs):
            return io.BytesIO(payloads[url.rsplit('/', 1)[-1]].encode())
        with patch('urllib.request.urlopen', side_effect=read) as download, \
                patch('os.chdir') as change, patch.dict(sys.modules), \
                patch('pathlib.Path.mkdir', side_effect=AssertionError('implicit activation')):
            exec(compile(cell, 'PAIRED_NATIVE_LATENT.md:cell', 'exec'), scope)
        self.assertEqual(download.call_count, 1)
        self.assertTrue(all('/' + 'a' * 40 + '/' in call.args[0] for call in download.call_args_list))
        change.assert_not_called()  # Activation is mocked; the launcher owns cwd.
        activate = scope['launcher']['launch']
        activate.assert_called_once_with(scope['launcher'])
        self.assertEqual(scope['OUTPUT'], 'mock-output')
        self.assertEqual(scope['launcher']['__name__'], '__main__')
        for name in ('RUN_DIR', 'PHASE0_INPUT', 'MODE', 'DEVICES', 'RESUME_INPUT', 'SMOKE_CONFIG'):
            self.assertEqual(scope['launcher'][name], scope[name])

    def test_missing_helpers_are_deferred_on_ordinary_import_without_network(self):
        source = (ROOT / 'kaggle_paired_native_latent.py').read_text(encoding='utf-8')
        real_import = builtins.__import__
        def import_without_siblings(name, *args, **kwargs):
            if name == 'kaggle_behavior_aware_h1':
                raise ModuleNotFoundError('fresh notebook has no sibling module', name=name)
            return real_import(name, *args, **kwargs)
        scope = {'__name__': '_fresh_import', 'NUM_QUESTIONS': 8}
        with patch('builtins.__import__', side_effect=import_without_siblings), \
                patch('subprocess.run', side_effect=AssertionError('network')), \
                patch('subprocess.check_output', side_effect=AssertionError('git query')), \
                patch('os.chdir', side_effect=AssertionError('chdir')):
            exec(compile(source, 'raw-import', 'exec'), scope)
        self.assertIsNone(scope['behavior'])
        self.assertIsNone(scope['_BOOTSTRAP_CHECKOUT'])
        self.assertEqual(scope['NUM_QUESTIONS'], 8)
        with self.assertRaisesRegex(ValueError, 'fixed at 100'):
            scope['build_config'](config_fixture(), scope)


class ContentResumeAndArchiveTests(unittest.TestCase):
    def test_original_trace_folder_zip_and_extensionless_mount_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = original_result(root / 'folder/arbitrary-root')
            self.assertEqual(launcher.resolve_original_input(ROOT, source.parent), source.resolve())
            mount = root / 'mount'
            mount.mkdir()
            archive = zip_tree(source, mount / 'unexpected-filename.payload')
            self.assertEqual(launcher.resolve_original_input(ROOT, archive), archive.resolve())
            self.assertEqual(launcher.resolve_original_input(ROOT, mount), archive.resolve())
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                launcher.resolve_original_input(ROOT, root)

    def test_real_phase0_reader_uses_metadata_for_arbitrary_zip_and_folder(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, resolver_dependencies()):
            root = Path(directory)
            source = phase0_result(root / 'folder/unexpected-root')
            before = list(sys.path)
            self.assertEqual(launcher.resolve_phase0_input(ROOT, source.parent), source.resolve())
            mount = root / 'mount'
            mount.mkdir()
            archive = zip_tree(source, mount / 'unexpected-phase0.payload')
            self.assertEqual(launcher.resolve_phase0_input(ROOT, archive), archive.resolve())
            self.assertEqual(launcher.resolve_phase0_input(ROOT, mount), archive.resolve())
            self.assertEqual(sys.path, before)
            self.assertNotIn('_behavior_aware_launcher_source', sys.modules)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                launcher.resolve_phase0_input(ROOT, root)

    def test_phase0_resolver_loads_new_downloaded_code_not_old_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'unused.json', {})
            (root / 'behavior_aware_source.py').write_text(
                'from pathlib import Path\ndef resolve_phase0_input(path):\n    return Path(path) / "new_reader"\n',
                encoding='utf-8')
            old = ModuleType('behavior_aware_source')
            old.resolve_phase0_input = lambda path: (_ for _ in ()).throw(AssertionError('stale reader'))
            with patch.dict(sys.modules, {'behavior_aware_source': old}):
                self.assertEqual(launcher.resolve_phase0_input(root, root), root / 'new_reader')

    def test_phase0_rejects_behavior_zip_learned_missing_cache_and_wrong_split(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, resolver_dependencies()):
            root = Path(directory)
            wrong = root / 'latest-behavior.zip'
            with zipfile.ZipFile(wrong, 'w') as zipped:
                zipped.writestr('summary.json', '{"schema":"behavior_aware_h1_v1"}')
            for source in (wrong, phase0_result(root / 'learned', False)):
                with self.assertRaises(ValueError):
                    launcher.resolve_phase0_input(ROOT, source)
            missing = phase0_result(root / 'missing')
            (missing / 'latent_128/frozen_latents.pt').unlink()
            with self.assertRaises(ValueError):
                launcher.resolve_phase0_input(ROOT, missing)
            split = phase0_result(root / 'split')
            write_json(split / 'split_manifest.json', {'train': list(range(100)), 'val': [], 'test': []})
            with self.assertRaises(ValueError):
                launcher.resolve_phase0_input(ROOT, split)

    def test_resume_folder_and_wrapped_zip_keep_jobs_filter_assets_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = prior_result(root / 'saved')
            (saved / 'preprocessing.pt').write_bytes(b'input asset')
            (saved / 'inputsource').mkdir()
            (saved / 'inputsource/original.payload').write_bytes(b'raw source')
            archive = zip_tree(saved, root / 'prior.payload')
            for index, path in enumerate((saved, archive)):
                output = root / f'working/new{index}'
                self.assertEqual(launcher.restore_resume_input(path, output, root / f'temp{index}'), output.resolve())
                self.assertTrue((output / 'jobs/bridge_seed42/best.pt').is_file())
                self.assertTrue((output / 'jobs/bridge_seed42/last.pt').is_file())
                self.assertFalse((output / 'inputsource').exists())
                self.assertFalse((output / 'preprocessing.pt').exists())
            self.assertTrue((saved / 'preprocessing.pt').is_file())
            self.assertTrue(archive.is_file())

    def test_resume_rejects_missing_signature_jobs_and_overlapping_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = prior_result(root / 'saved')
            write_json(saved / 'study_manifest.json', {'signature': ''})
            with self.assertRaisesRegex(ValueError, 'signature'):
                launcher.validate_resume_root(saved)
            write_json(saved / 'study_manifest.json', {'signature': 'a' * 64})
            (saved / 'jobs/bridge_seed42/best.pt').unlink()
            (saved / 'jobs/bridge_seed42/last.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'jobs checkpoints'):
                launcher.validate_resume_root(saved)
            prior_result(saved)
            with self.assertRaisesRegex(ValueError, 'separate folders'):
                launcher.restore_resume_input(saved, saved / 'new', root / 'temp')

    def test_safe_resume_zip_rejects_traversal_and_symlinks_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(('../escape', '/absolute', 'C:/escape', 'wrapper/../../escape')):
                archive = root / f'bad{index}.zip'
                with zipfile.ZipFile(archive, 'w') as zipped:
                    zipped.writestr(name, 'bad')
                target = root / f'output{index}'
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    launcher.safe_extract(archive, target)
                self.assertFalse(target.exists())
            archive = root / 'symlink.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                info = zipfile.ZipInfo('link')
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zipped.writestr(info, 'outside')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                launcher.safe_extract(archive, root / 'symlink-output')

    def test_archive_keeps_optimizer_checkpoints_predictions_reports_logs_not_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prior_result(Path(directory) / 'working/result')
            keep = ('predictions.jsonl', 'reports/probe.json', 'logs/train.log')
            excluded = ('_cache/blob.pt', 'inputsource/arbitrary.payload', 'raw/original.data',
                        'preprocessing.pt', 'frozen_latents.pt', 'latent_128/frozen_latents.pt',
                        'states.jsonl', 'edges.jsonl', 'labels.jsonl', 'teacher_targets.jsonl',
                        'original_trace.npz', 'phase0.zip')
            for name in (*keep, *excluded):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'test')
            with zipfile.ZipFile(output / 'arbitrary-original.payload', 'w') as zipped:
                zipped.writestr('experience/trace.npz', b'raw')
            archive = launcher.archive_partial(output)
            self.assertEqual(archive.parent, output.parent)
            self.assertFalse(archive.with_suffix('.zip.tmp').exists())
            with zipfile.ZipFile(archive) as zipped:
                for name in (*keep, 'jobs/bridge_seed42/best.pt', 'jobs/bridge_seed42/last.pt'):
                    self.assertIn(name, zipped.namelist())
                self.assertTrue(set(excluded).isdisjoint(zipped.namelist()))
                self.assertNotIn('arbitrary-original.payload', zipped.namelist())
            self.assertTrue(launcher._compact_member('jobs/bridge_seed42/last.pt', 129 * 1024 ** 2))
            self.assertFalse(launcher._compact_member('arbitrary-original.bin', int(1.4 * 1024 ** 3)))

    def test_existing_stage_zip_is_merged_filtered_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prior_result(Path(directory) / 'result')
            archive = output.with_suffix('.zip')
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('stage_report.json', '{"stage":"probes"}')
                zipped.writestr('preprocessing.pt', b'input')
                zipped.writestr('inputsource/original.bin', b'input')
                nested = io.BytesIO()
                with zipfile.ZipFile(nested, 'w') as raw:
                    raw.writestr('experience/trace.npz', b'raw')
                zipped.writestr('arbitrary-original.payload', nested.getvalue())
            (output / 'launcher_error.txt').write_text('interrupted', encoding='utf-8')
            before = archive.read_bytes()
            with patch('pathlib.Path.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    launcher.archive_partial(output)
            self.assertEqual(archive.read_bytes(), before)
            launcher.archive_partial(output)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn('stage_report.json', zipped.namelist())
                self.assertIn('launcher_error.txt', zipped.namelist())
                self.assertIn('jobs/bridge_seed42/last.pt', zipped.namelist())
                self.assertNotIn('preprocessing.pt', zipped.namelist())
                self.assertNotIn('inputsource/original.bin', zipped.namelist())
                self.assertNotIn('arbitrary-original.payload', zipped.namelist())

    def test_runner_prefixed_zip_merges_to_single_resume_root_with_current_replacements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = prior_result(root / 'working/paired_native_latent_full_fixture')
            archive = output.with_suffix('.zip')
            prefix = output.name + '/'
            # Match run_latent_wm_phase0.package: output.name / relative.
            with zipfile.ZipFile(archive, 'w') as zipped:
                for path in output.rglob('*'):
                    if path.is_file():
                        zipped.write(path, prefix + path.relative_to(output).as_posix())
                zipped.writestr(prefix + 'reports/completed_probe.json', '{"stage":"probes"}')
                zipped.writestr('external_stage/report.json', '{"preserved":true}')
                zipped.writestr(prefix + '../unsafe_before.json', '{}')
                zipped.writestr(prefix + '/unsafe_after.json', '{}')
            config = config_fixture()
            config['oracle_updates'] = 999
            prior_result(output, config)
            checkpoint = 'jobs/bridge_seed42/last.pt'
            (output / checkpoint).write_bytes(b'current optimizer/RNG replacement')

            launcher.archive_partial(output)
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertEqual(len(names), len(set(names)))
                self.assertFalse(any(name.startswith(prefix) for name in names))
                self.assertEqual(names.count('config.json'), 1)
                self.assertEqual(names.count(checkpoint), 1)
                self.assertEqual(names.count('jobs/bridge_seed42/best.pt'), 1)
                self.assertEqual(json.loads(zipped.read('config.json')), config)
                self.assertEqual(zipped.read(checkpoint), b'current optimizer/RNG replacement')
                self.assertIn('reports/completed_probe.json', names)
                self.assertIn('external_stage/report.json', names)
                self.assertFalse(any('unsafe_' in name for name in names))
            extracted = launcher.safe_extract(archive, root / 'extracted')
            self.assertEqual(list(extracted.rglob('config.json')), [extracted / 'config.json'])
            self.assertEqual(launcher.locate_resume_root(extracted), extracted.resolve())
            restored = launcher.restore_resume_input(archive, root / 'resumed', root / 'temp')
            self.assertEqual((restored / checkpoint).read_bytes(), b'current optimizer/RNG replacement')


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = zip_tree(original_result(self.root / 'trace'), self.root / 'original.payload')
        self.phase0 = zip_tree(phase0_result(self.root / 'frozen'), self.root / 'phase0.payload')
        self.working = self.root / 'working'
        self.working.mkdir()
        (self.working / 'sentinel.txt').write_text('existing output', encoding='utf-8')
        self.scope = {'RUN_DIR': str(self.original), 'PHASE0_INPUT': str(self.phase0),
                      'WORKING_DIR': str(self.working), 'TEMP_DIR': str(self.root / 'temp'),
                      'MODE': 'smoke', 'DEVICES': 'cpu'}
        self.commands, self.environments, self.checkouts = [], [], []

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    def fake_download(self, repo, ref, env):
        self.assertEqual(Path.cwd(), self.working)
        self.checkouts.append(repo)
        self.environments.append(env)
        write_json(repo / launcher.CONFIG_RELATIVE, config_fixture())
        for name in (*launcher.TEST_FILES, str(launcher.RUNNER_RELATIVE)):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture; never execute training\n', encoding='utf-8')
        for name in ('kaggle_factorized_wm_feasibility.py', 'factorized_wm_data.py', 'behavior_aware_source.py',
                     'kaggle_latent_wm_phase0.py', 'kaggle_behavior_aware_h1.py'):
            shutil.copyfile(ROOT / name, repo / name)
        return 'd' * 40

    def fake_run(self, command, *, cwd, env):
        command = [str(part) for part in command]
        self.commands.append(command)
        if any(Path(part).name == str(launcher.RUNNER_RELATIVE) for part in command):
            output = Path(command[command.index('--output') + 1])
            config = json.loads(Path(command[command.index('--config') + 1]).read_text(encoding='utf-8'))
            prior_result(output, config)
            (output / 'predictions.jsonl').write_text('{}\n', encoding='utf-8')

    def launch_mocked(self, scope=None, run=None, download=None, show=None):
        with patch.object(launcher, 'on_kaggle', return_value=False), \
                patch.object(launcher, 'download_source', side_effect=download or self.fake_download), \
                patch.object(launcher, 'run_command', side_effect=run or self.fake_run), \
                patch.object(launcher, 'show_archive', side_effect=show) as displayed, \
                patch.dict(sys.modules, resolver_dependencies()):
            output = launcher.launch(scope or self.scope)
            return output, displayed.call_args.args[0]

    def test_command_resolved_inputs_config_revision_and_no_gpu_query(self):
        output, archive = self.launch_mocked()
        config = json.loads((output / 'launcher_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['num_questions'], 100)
        self.assertEqual(config['protocol']['split'], [70, 15, 15])
        self.assertEqual(config['devices'], ['cpu', 'cpu'])
        metadata = json.loads((output / 'launcher_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_revision'], 'd' * 40)
        self.assertEqual(metadata['source_ref'], launcher.DEFAULT_SOURCE_REF)
        self.assertEqual(metadata['LLM_forwards'], 0)
        self.assertEqual(metadata['LLM_downloads'], 0)
        self.assertEqual(self.commands[0][-1:], ['numpy'])
        self.assertNotIn('transformers', ' '.join(self.commands[0]).lower())
        self.assertEqual(Path(self.commands[1][1]).name, Path(launcher.TEST_FILES[0]).name)
        command = self.commands[-1]
        self.assertEqual(Path(command[2]).name, 'run_paired_native_latent.py')
        for flag, path in (('--input', self.original), ('--phase0_input', self.phase0),
                           ('--output', output), ('--config', output / 'launcher_config.json')):
            self.assertEqual(command[command.index(flag) + 1], str(path))
        self.assertNotIn('--resume', command)
        self.assertEqual(archive.parent, self.working)
        self.assertTrue(archive.is_file())
        self.assertEqual((self.working / 'sentinel.txt').read_text(encoding='utf-8'), 'existing output')
        for key in ('HF_HOME', 'TMPDIR', 'TORCH_HOME'):
            self.assertTrue(Path(self.environments[0][key]).is_relative_to(self.root / 'temp'))
        self.assertEqual(self.environments[0]['HF_HUB_OFFLINE'], '1')
        second, _ = self.launch_mocked()
        self.assertNotEqual(output, second)
        self.assertNotEqual(self.checkouts[0], self.checkouts[1])

    def test_one_url_fresh_cell_bootstraps_one_checkout_and_reuses_verified_head(self):
        cell = (ROOT / 'PAIRED_NATIVE_LATENT.md').read_text(encoding='utf-8').split('```python\n', 1)[1].split('```', 1)[0]
        source = (ROOT / 'kaggle_paired_native_latent.py').read_bytes()
        scope = dict(self.scope, SOURCE_REF='f' * 40, DEVICES=['cpu'])
        git_commands = []
        def fake_process(command, *, cwd, env, check):
            command = [str(part) for part in command]
            self.assertTrue(check)
            if command[0] == 'git':
                self.assertEqual(Path.cwd(), self.working)
                self.assertEqual(Path(cwd), self.working)
                self.assertEqual(env['GIT_LFS_SKIP_SMUDGE'], '1')
                git_commands.append(command)
                if command[-2:] == ['--detach', 'FETCH_HEAD']:
                    self.fake_download(Path(command[2]), scope['SOURCE_REF'], env)
            else:
                self.fake_run(command, cwd=cwd, env=env)
        clean_path = [path for path in sys.path if path and Path(path).resolve() != ROOT]
        with patch.dict(sys.modules, resolver_dependencies()), patch.object(sys, 'path', clean_path), \
                patch('urllib.request.urlopen', return_value=io.BytesIO(source)) as read, \
                patch('subprocess.run', side_effect=fake_process), \
                patch('subprocess.check_output', return_value='f' * 40 + '\n') as head:
            sys.modules.pop('kaggle_behavior_aware_h1', None)
            sys.modules.pop('kaggle_latent_wm_phase0', None)
            os.chdir(self.working)  # No repository path or helpers in the fresh runtime.
            exec(compile(cell, 'PAIRED_NATIVE_LATENT.md:fresh-cell', 'exec'), scope)
        read.assert_called_once()
        self.assertEqual(len(git_commands), 4)
        self.assertEqual(git_commands[2][-5:], ['fetch', '--depth', '1', 'origin', 'f' * 40])
        self.assertEqual(git_commands[3][-2:], ['--detach', 'FETCH_HEAD'])
        head.assert_called_once()
        self.assertEqual(len(self.checkouts), 1)
        output = scope['OUTPUT']
        metadata = json.loads((output / 'launcher_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['source_revision'], 'f' * 40)
        self.assertEqual(metadata['source_ref'], 'f' * 40)
        self.assertEqual(scope['launcher']['_BOOTSTRAP_CHECKOUT'], self.checkouts[0])
        command = self.commands[-1]
        self.assertEqual(Path(command[2]).parent, self.checkouts[0])
        config = json.loads((output / 'launcher_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['devices'], ['cpu', 'cpu'])
        self.assertTrue(output.with_suffix('.zip').is_file())

    def test_resume_adds_flag_preserves_raw_input_and_exact_config(self):
        config = launcher.build_config(config_fixture(), self.scope)
        config['devices'] = ['cpu', 'cpu']
        saved = prior_result(self.root / 'saved', config)
        archive = zip_tree(saved, self.root / 'resume.payload')
        output, result_zip = self.launch_mocked(dict(self.scope, RESUME_INPUT=str(archive)))
        self.assertIn('--resume', self.commands[-1])
        self.assertEqual(self.commands[-1][self.commands[-1].index('--input') + 1], str(self.original))
        self.assertTrue((output / 'jobs/bridge_seed42/last.pt').is_file())
        self.assertTrue(result_zip.is_file())

    def test_new_paired_latent_tests_are_discovered_in_downloaded_checkout(self):
        def with_new_tests(repo, ref, env):
            revision = self.fake_download(repo, ref, env)
            for name in ('test_paired_latent_models.py', 'test_paired_latent_runner.py'):
                (repo / 'tests' / name).write_text('# future main-owned test fixture\n', encoding='utf-8')
            return revision
        output, _ = self.launch_mocked(download=with_new_tests)
        expected = ['test_paired_latent_models.py', 'test_paired_latent_runner.py',
                    'test_paired_native_launcher.py']
        self.assertEqual([Path(c[1]).name for c in self.commands[1:-1]], expected)
        metadata = json.loads((output / 'launcher_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual([Path(path).name for path in metadata['test_files']], expected)

    def test_resume_config_mismatch_prevents_runner_and_packages_error(self):
        saved = prior_result(self.root / 'saved')
        with self.assertRaisesRegex(ValueError, 'effective config differs'):
            self.launch_mocked(dict(self.scope, RESUME_INPUT=str(saved)))
        self.assertFalse(any('run_paired_native_latent.py' in ' '.join(command) for command in self.commands))
        with zipfile.ZipFile(next(self.working.glob('*.zip'))) as zipped:
            self.assertIn('launcher_error.txt', zipped.namelist())
            self.assertIn('jobs/bridge_seed42/last.pt', zipped.namelist())

    def test_failed_runner_preserves_partial_checkpoint_and_earlier_stage_report(self):
        shown = []
        def fail(command, *, cwd, env):
            self.fake_run(command, cwd=cwd, env=env)
            if any(Path(str(part)).name == str(launcher.RUNNER_RELATIVE) for part in command):
                output = Path(command[command.index('--output') + 1])
                with zipfile.ZipFile(output.with_suffix('.zip'), 'w') as zipped:
                    zipped.writestr('completed_probe.json', '{"status":"complete"}')
                raise subprocess.CalledProcessError(7, command)
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch_mocked(run=fail, show=shown.append)
        self.assertEqual(len(shown), 1)
        with zipfile.ZipFile(shown[0]) as zipped:
            self.assertIn('completed_probe.json', zipped.namelist())
            self.assertIn('jobs/bridge_seed42/best.pt', zipped.namelist())
            self.assertIn('jobs/bridge_seed42/last.pt', zipped.namelist())
            self.assertIn('CalledProcessError', zipped.read('launcher_error.txt').decode())

    def test_missing_inputs_fail_before_download_and_still_make_root_zip(self):
        for name in ('RUN_DIR', 'PHASE0_INPUT'):
            with self.subTest(name=name), \
                    patch.object(launcher, 'on_kaggle', return_value=False), \
                    patch.object(launcher, 'download_source') as download, \
                    patch.object(launcher, 'show_archive') as displayed:
                with self.assertRaises(FileNotFoundError):
                    launcher.launch(dict(self.scope, **{name: str(self.root / 'missing')}))
                download.assert_not_called()
                self.assertEqual(displayed.call_args.args[0].parent, self.working)
                self.assertTrue(displayed.call_args.args[0].is_file())


if __name__ == '__main__':
    unittest.main()
