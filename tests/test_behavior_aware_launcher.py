"""Offline launch/selection/resume checks; no model training or downloads."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_behavior_aware_h1 as launcher


def read_config():
    return json.loads((ROOT / launcher.CONFIG_RELATIVE).read_text(encoding="utf-8"))


def phase0_result(root, pretrained=True):
    root.mkdir(parents=True)
    documents = {
        "config.json": {"num_questions": 100, "latent_dims": [64, 128],
                        "token_embedding": "pretrained" if pretrained else "learned"},
        "summary.json": {"schema": "latent_world_model_phase0_v1", "status": "complete"},
        "split_manifest.json": {"train": [f"q{i}" for i in range(70)],
                                "val": [f"q{i}" for i in range(70, 85)],
                                "test": [f"q{i}" for i in range(85, 100)]},
        "study_manifest.json": {"fingerprint": "a" * 64},
        "source_hashes.json": {"phase0_wm_models.py": "a" * 64},
        "latent_128/stage_A/complete.json": {"status": "complete"},
    }
    for name, value in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    for name in ("preprocessing.pt", "latent_128/stage_A/best.pt", "latent_128/frozen_latents.pt"):
        (root / name).write_bytes(b"resolver metadata fixture; do not load")
    return root


def make_zip(source, path):
    with zipfile.ZipFile(path, "w") as zipped:
        for item in source.rglob("*"):
            if item.is_file():
                zipped.write(item, "wrapper/result/" + item.relative_to(source).as_posix())
    return path


def prior_result(root, config):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "study_manifest.json").write_text(json.dumps({
        "schema": "behavior_aware_h1_v1", "fingerprint": "b" * 64, "status": "running",
    }), encoding="utf-8")
    checkpoint = root / "A_pure/seed42/resume.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"mock optimizer/RNG checkpoint; payload validated by runner")
    return root


class ConfigAndImportTests(unittest.TestCase):
    def test_full_config_unchanged_and_no_device_or_source_edits(self):
        base = read_config()
        self.assertEqual(launcher.build_config(base, {}), base)
        self.assertEqual(base["seeds"], [42, 43, 44])
        self.assertNotIn("devices", launcher.build_config(base, {}))

    def test_smoke_changes_budgets_only_preserving_full128_split(self):
        base = read_config()
        before = copy.deepcopy(base)
        effective = launcher.build_config(base, {"MODE": "smoke"})
        for key, value in launcher.SMOKE_OVERRIDES.items():
            self.assertEqual(effective[key], value)
        self.assertEqual(effective["latent_dim"], 128)
        self.assertEqual(effective["protocol"]["questions"], 100)
        self.assertEqual(effective["protocol"]["split"], [70, 15, 15])
        self.assertEqual(effective["workers"], 2)
        self.assertEqual(effective["bootstrap_samples"], 2000)
        self.assertEqual(base, before)

    def test_invalid_mode_or_full_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MODE"):
            launcher.build_config(read_config(), {"MODE": "typo"})
        for key, value in (("latent_dim", 64), ("seeds", [42]), ("workers", 1)):
            base = read_config()
            base[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract"):
                launcher.build_config(base, {})

    def test_import_does_not_start_either_study(self):
        spec = importlib.util.spec_from_file_location("_behavior_import_check", ROOT / "kaggle_behavior_aware_h1.py")
        module = importlib.util.module_from_spec(spec)
        with patch("os.chdir", side_effect=AssertionError("changed cwd")), \
                patch("subprocess.run", side_effect=AssertionError("ran command")), \
                patch.object(launcher.phase0, "launch", side_effect=AssertionError("old study")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.launch))
        self.assertIs(module.download_source, launcher.phase0.download_source)
        self.assertIs(module.select_devices, launcher.phase0.select_devices)

    def test_two_gpu_requirement_and_explicit_local_cpu(self):
        from types import SimpleNamespace
        def fake_torch(count):
            return SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count,
                is_available=lambda: count > 0, get_device_name=lambda index: f"GPU{index}"))
        self.assertEqual(launcher.select_devices({}, True, fake_torch(2)), ("cuda:0", "cuda:1"))
        with self.assertRaisesRegex(RuntimeError, "two visible GPUs"):
            launcher.select_devices({}, True, fake_torch(1))
        with self.assertRaisesRegex(ValueError, "non-Kaggle"):
            launcher.select_devices({"ALLOW_CPU": True}, True, fake_torch(2))
        self.assertEqual(launcher.select_devices({"ALLOW_CPU": True}, False, fake_torch(0)), ("cpu", "cpu"))

    def test_pinned_sha_fetch_and_detached_checkout_use_shared_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            ref = "a" * 40
            with patch.object(launcher.phase0, "run_command") as run, \
                    patch("subprocess.check_output", return_value=ref + "\n"):
                self.assertEqual(launcher.download_source(repo, ref, {}), ref)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[2], ["git", "-C", repo, "fetch", "--depth", "1", "origin", ref])
            self.assertEqual(commands[3][-2:], ["--detach", "FETCH_HEAD"])

    def test_fresh_documented_cell_downloads_both_helpers_and_activates_once(self):
        cell = (ROOT / "BEHAVIOR_AWARE_H1.md").read_text(encoding="utf-8").split("```python\n", 1)[1].split("```", 1)[0]
        helpers = (ROOT / "kaggle_latent_wm_phase0.py").read_text(encoding="utf-8")
        behavior = (ROOT / "kaggle_behavior_aware_h1.py").read_text(encoding="utf-8")
        behavior += "\nfrom unittest.mock import Mock\nlaunch = Mock(return_value='mock-output')\n"
        scope = {"__name__": "__main__", "SOURCE_REF": "a" * 40,
                 "MODE": "smoke", "RUN_DIR": "/kaggle/input/original.data",
                 "PHASE0_INPUT": "/kaggle/input/extracted/result", "RESUME_INPUT": "/kaggle/input/resume.zip"}
        def read(url, **kwargs):
            payload = helpers if url.endswith("kaggle_latent_wm_phase0.py") else behavior
            return io.BytesIO(payload.encode())
        with patch("urllib.request.urlopen", side_effect=read) as download, \
                patch("os.chdir") as change, patch.dict(sys.modules), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("implicit activation")):
            exec(compile(cell, "BEHAVIOR_AWARE_H1.md:cell", "exec"), scope)
        self.assertEqual(download.call_count, 2)
        self.assertTrue(all("/" + "a" * 40 + "/" in call.args[0] for call in download.call_args_list))
        change.assert_called_once_with("/kaggle/working")
        activate = scope["launcher"]["launch"]
        self.assertEqual(activate.call_count, 1)
        self.assertIs(activate.call_args.args[0], scope)
        self.assertEqual(scope["OUTPUT"], "mock-output")
        self.assertEqual(scope["PHASE0_INPUT"], "/kaggle/input/extracted/result")


class SourceAndArchiveTests(unittest.TestCase):
    def test_actual_reader_selects_extracted_folder_and_wrapped_arbitrary_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = phase0_result(root / "extracted/result")
            before = list(sys.path)
            self.assertEqual(launcher.resolve_phase0_input(ROOT, source.parent), source.resolve())
            archive = make_zip(source, root / "not_an_exact_name.payload")
            self.assertEqual(launcher.resolve_phase0_input(ROOT, archive), archive.resolve())
            self.assertEqual(sys.path, before)
            self.assertNotIn("_behavior_aware_launcher_source", sys.modules)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                launcher.resolve_phase0_input(ROOT, root)

    def test_reader_rejects_learned_or_original_only_input_and_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = phase0_result(root / "learned", pretrained=False)
            with self.assertRaises(ValueError):
                launcher.resolve_phase0_input(ROOT, source)
            original = root / "only2source.zip"
            with zipfile.ZipFile(original, "w") as zipped:
                zipped.writestr("summary.json", '{"schema":"interactive_acceptance_two_source_v1"}')
            with self.assertRaises(ValueError):
                launcher.resolve_phase0_input(ROOT, original)
            source = phase0_result(root / "missingcache")
            (source / "latent_128/frozen_latents.pt").unlink()
            with self.assertRaises(ValueError):
                launcher.resolve_phase0_input(ROOT, source)

    def test_restore_interrupted_job_and_exclude_native_source_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = prior_result(root / "prior", launcher.build_config(read_config(), {"MODE": "smoke"}))
            (prior / "preprocessing.pt").write_bytes(b"source embedding must not be restored into working")
            archive = make_zip(prior, root / "resume.payload")
            temp = root / "temp"
            temp.mkdir()
            output = root / "working/new"
            self.assertEqual(launcher.restore_resume_input(archive, output, temp), output.resolve())
            self.assertTrue((output / "A_pure/seed42/resume.pt").is_file())
            self.assertFalse((output / "preprocessing.pt").exists())
            self.assertTrue(archive.is_file())

    def test_resume_rejects_missing_signature_or_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = prior_result(Path(directory) / "result", read_config())
            manifest = root / "study_manifest.json"
            manifest.write_text('{"schema":"behavior_aware_h1_v1"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature"):
                launcher.validate_resume_root(root)
            manifest.write_text(json.dumps({"fingerprint": "b" * 64}), encoding="utf-8")
            (root / "A_pure/seed42/resume.pt").unlink()
            with self.assertRaisesRegex(ValueError, "checkpoints"):
                launcher.validate_resume_root(root)

    def test_resume_extraction_rejects_unsafe_members_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, member in enumerate(("../escape", "C:\\escape", "/absolute", "..\\escape")):
                archive = root / f"bad{i}.zip"
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("safe.txt", "not written")
                    zipped.writestr(member, "bad")
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    launcher.safe_extract(archive, root / f"extract{i}")
                self.assertFalse((root / f"extract{i}").exists())

    def test_fallback_includes_larger_optimizer_checkpoint_without_mutating_phase0_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = prior_result(Path(directory) / "result", read_config())
            checkpoint = root / "A_pure/seed42/resume.pt"
            with checkpoint.open("wb") as stream:
                stream.truncate(launcher.phase0.MAX_ARCHIVE_FILE_BYTES + 1)
            for name in ("preprocessing.pt", "latent_128/frozen_latents.pt", "phase0_input/embedding.pt", "experience/raw.npz"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"source/cache excluded")
            with zipfile.ZipFile(launcher.archive_partial(root)) as zipped:
                self.assertIn("A_pure/seed42/resume.pt", zipped.namelist())
                self.assertNotIn("preprocessing.pt", zipped.namelist())
                self.assertNotIn("latent_128/frozen_latents.pt", zipped.namelist())
                self.assertEqual(json.loads(zipped.read("launcher_packaging.json"))["max_file_bytes"], 2 * 1024**3)
            self.assertEqual(launcher.phase0.MAX_ARCHIVE_FILE_BYTES, 128 * 1024**2)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.working = self.root / "working"
        self.working.mkdir()
        self.original = self.root / "original.payload"
        with zipfile.ZipFile(self.original, "w") as zipped:
            zipped.writestr("summary.json", "{}")
        self.source = phase0_result(self.root / "phase0/result")
        self.sentinel = self.working / "keep.txt"
        self.sentinel.write_text("user output", encoding="utf-8")
        self.scope = {"MODE": "smoke", "WORKING_DIR": str(self.working),
                      "TEMP_DIR": str(self.root / "temp"), "RUN_DIR": str(self.original),
                      "PHASE0_INPUT": str(self.source.parent), "ALLOW_CPU": True}
        self.commands, self.envs = [], []

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def fake_download(self, repo, ref, env):
        self.assertEqual(Path.cwd(), self.working)
        self.assertTrue(repo.is_relative_to(self.root / "temp"))
        self.assertEqual(ref, self.scope.get("SOURCE_REF", launcher.DEFAULT_SOURCE_REF))
        (repo / "configs").mkdir(parents=True)
        (repo / launcher.CONFIG_RELATIVE).write_text(json.dumps(read_config()), encoding="utf-8")
        for name in (*launcher.TEST_FILES, "tests/test_behavior_aware_metrics.py", str(launcher.RUNNER_RELATIVE)):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# mocked only\n", encoding="utf-8")
        return "c" * 40

    def fake_run(self, command, *, cwd, env):
        parts = [str(part) for part in command]
        self.commands.append(parts)
        self.envs.append(env)
        if any(part.endswith(str(launcher.RUNNER_RELATIVE)) for part in parts):
            output = Path(parts[parts.index("--output") + 1])
            config = json.loads(Path(parts[parts.index("--config") + 1]).read_text(encoding="utf-8"))
            prior_result(output, config)

    def mocked_launch(self, scope=None, run=None, reader=None):
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "select_devices", return_value=("cpu", "cpu")), \
                patch.object(launcher, "download_source", side_effect=self.fake_download), \
                patch.object(launcher, "resolve_original_input", return_value=self.original), \
                patch.object(launcher, "resolve_phase0_input", side_effect=reader, return_value=self.source) as resolve, \
                patch.object(launcher, "run_command", side_effect=run or self.fake_run), \
                patch.object(launcher, "show_archive") as show, \
                patch("shutil.rmtree", side_effect=AssertionError("deleted existing directory")):
            output = launcher.launch(scope or self.scope)
            return output, show.call_args.args[0], resolve.call_args

    def test_cli_resolves_folder_source_keeps_full128_and_runs_study_tests_as_scripts(self):
        output, archive, resolved = self.mocked_launch()
        config = json.loads((output / "launcher_config.json").read_text())
        self.assertEqual(config["latent_dim"], 128)
        self.assertEqual(config["protocol"]["questions"], 100)
        self.assertEqual(config["max_updates"], 8)
        self.assertNotIn("devices", config)
        self.assertEqual(self.commands[0][-1:], ["numpy"])
        self.assertFalse(any(word in self.commands[0] for word in ("torch", "transformers", "huggingface_hub")))
        self.assertEqual([Path(command[1]).name for command in self.commands[1:-1]],
                         ["test_behavior_aware_launcher.py", "test_behavior_aware_metrics.py"])
        runner = self.commands[-1]
        self.assertEqual(runner[runner.index("--input") + 1], str(self.original))
        self.assertEqual(runner[runner.index("--phase0_input") + 1], str(self.source.resolve()))
        self.assertEqual(Path(resolved.args[1]), self.source.parent.resolve())
        self.assertNotIn("--resume", runner)
        self.assertEqual(archive.parent, self.working)
        self.assertTrue(archive.is_file())
        self.assertTrue(Path(self.envs[-1]["TMPDIR"]).is_relative_to(self.root / "temp"))
        self.assertEqual(self.sentinel.read_text(), "user output")

    def test_resume_restores_optimizer_file_and_passes_resume_without_replacing_inputs(self):
        prior = prior_result(self.root / "prior", launcher.build_config(read_config(), self.scope))
        output, archive, _ = self.mocked_launch(dict(self.scope, RESUME_INPUT=str(prior)))
        self.assertTrue((output / "A_pure/seed42/resume.pt").is_file())
        self.assertIn("--resume", self.commands[-1])
        self.assertTrue(prior.is_dir())
        self.assertTrue(archive.is_file())

    def test_error_zip_at_working_root_and_original_outputs_preserved(self):
        def fail(command, *, cwd, env):
            self.fake_run(command, cwd=cwd, env=env)
            if any(str(part).endswith(str(launcher.RUNNER_RELATIVE)) for part in command):
                raise subprocess.CalledProcessError(9, command)
        with self.assertRaises(subprocess.CalledProcessError):
            self.mocked_launch(run=fail)
        archive = next(self.working.glob("*.zip"))
        with zipfile.ZipFile(archive) as zipped:
            self.assertIn("launcher_error.txt", zipped.namelist())
            self.assertIn("A_pure/seed42/resume.pt", zipped.namelist())
        self.assertTrue(self.sentinel.is_file())

    def test_missing_new_phase0_zip_has_upload_guidance_and_no_training(self):
        with self.assertRaisesRegex(ValueError, "176 MB"):
            self.mocked_launch(reader=ValueError("only original 2source2 is mounted"))
        self.assertEqual(len(self.commands), 1)  # dependency installation only
        self.assertTrue(next(self.working.glob("*.zip")).is_file())

    def test_resume_config_mismatch_does_not_train(self):
        prior = prior_result(self.root / "prior", read_config())
        with self.assertRaisesRegex(ValueError, "effective config differs"):
            self.mocked_launch(dict(self.scope, RESUME_INPUT=str(prior)))
        self.assertEqual(len(self.commands), 1)

    def test_missing_phase0_mount_fails_before_source_fetch(self):
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "download_source") as download, patch.object(launcher, "show_archive") as show:
            with self.assertRaisesRegex(FileNotFoundError, "Add Notebook Output"):
                launcher.launch(dict(self.scope, PHASE0_INPUT=str(self.root / "missing")))
        download.assert_not_called()
        self.assertTrue(show.call_args.args[0].is_file())

    def test_default_phase0_scan_root_and_basename_download_link(self):
        scope = dict(self.scope)
        scope.pop("PHASE0_INPUT")
        with patch.object(launcher, "phase0_preflight", return_value=self.source.parent) as preflight:
            _, archive, _ = self.mocked_launch(scope)
        preflight.assert_called_once_with("/kaggle/input")
        links = []
        from types import SimpleNamespace
        with patch.dict(sys.modules, {"IPython.display": SimpleNamespace(
                FileLink=lambda name: links.append(name) or name, display=lambda value: None)}):
            launcher.show_archive(archive)
        self.assertEqual(links, [archive.name])


if __name__ == "__main__":
    unittest.main()
