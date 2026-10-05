"""Launcher checks use only local fixtures and mocks; never download source."""
import contextlib
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
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_factorized_wm_feasibility as launcher


def read_config():
    return json.loads((ROOT / launcher.CONFIG_RELATIVE).read_text(encoding="utf-8"))


def previous_result(root, config=None):
    root.mkdir(parents=True, exist_ok=True)
    config = config or read_config()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "study_manifest.json").write_text(json.dumps({
        "fingerprint": "a" * 64, "config": config,
        "source": "fixture/original.zip", "status": "running",
    }), encoding="utf-8")
    job = root / "jobs" / "saved_job"
    job.mkdir(parents=True)
    (job / "checkpoint.pt").write_bytes(b"small checkpoint fixture")
    (job / "result.json").write_text('{"job": {"name": "saved_job"}, "reports": {}}', encoding="utf-8")
    return root


class ConfigTests(unittest.TestCase):
    def test_full_defaults_retain_entire_requested_grid(self):
        base = read_config()
        saved = copy.deepcopy(base)
        effective = launcher.build_config(base, {})
        for key, value in launcher.FULL_CONTRACT.items():
            self.assertEqual(effective[key], value, key)
        self.assertEqual(base, saved)
        self.assertEqual(effective["mode"], "full")
        self.assertEqual(effective["protocol"]["split_counts"],
                         {"train": 70, "validation": 15, "test": 15})
        self.assertEqual(effective["protocol"]["curriculum"], ["H1", "H2", "H3"])
        self.assertTrue(effective["protocol"]["hidden_projector"]["freeze_before_dynamics_targets"])
        self.assertEqual(effective["protocol"]["teacher_support"], "native_top32_conditional_only")
        self.assertFalse(effective["protocol"]["full_vocabulary_kl_available"])

    def test_smoke_is_explicit_and_has_enough_validation_questions(self):
        base = read_config()
        effective = launcher.build_config(base, {"MODE": "smoke"})
        for key, value in launcher.SMOKE_OVERRIDES.items():
            self.assertEqual(effective[key], value, key)
        self.assertEqual(effective["protocol"]["split_counts"],
                         {"train": 14, "validation": 3, "test": 3})
        self.assertEqual(effective["horizon"], 3)
        self.assertEqual(effective["protocol"]["teacher_support"], "native_top32_conditional_only")
        self.assertEqual(base["num_questions"], 100)
        self.assertEqual(effective["device_drafter"], "cuda:1")
        self.assertEqual(effective["device_verifier"], "cuda:0")

    def test_full_rejects_a_shrunken_config_and_invalid_mode(self):
        base = read_config()
        base["num_questions"] = 20
        with self.assertRaisesRegex(ValueError, "Full config"):
            launcher.build_config(base, {"MODE": "full"})
        with self.assertRaisesRegex(ValueError, "MODE"):
            launcher.build_config(read_config(), {"MODE": "typo"})

    def test_effective_config_does_not_share_nested_objects(self):
        base = read_config()
        config = launcher.build_config(base, {})
        config["capacities"]["small"]["width"] = 999
        config["protocol"]["curriculum"].append("H4")
        self.assertEqual(base["capacities"]["small"]["width"], 64)
        self.assertEqual(base["protocol"]["curriculum"], ["H1", "H2", "H3"])


class ImportAndDeviceTests(unittest.TestCase):
    def test_import_has_no_launch_chdir_or_download_side_effect(self):
        spec = importlib.util.spec_from_file_location("_launcher_import_check", ROOT / launcher.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("os.chdir", side_effect=AssertionError("import changed cwd")), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("import created folder")), \
                patch("subprocess.run", side_effect=AssertionError("import ran command")), \
                patch("subprocess.check_output", side_effect=AssertionError("import downloaded source")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.launch))

    def test_two_gpu_assignment_and_explicit_local_cpu(self):
        def torch_fixture(count):
            return SimpleNamespace(cuda=SimpleNamespace(
                device_count=lambda: count, is_available=lambda: count > 0,
                get_device_name=lambda i: f"GPU{i}"))
        self.assertEqual(launcher.select_devices({}, True, torch_fixture(2)), ("cuda:1", "cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "two visible GPUs"):
            launcher.select_devices({}, False, torch_fixture(0))
        self.assertEqual(launcher.select_devices({"ALLOW_CPU": True}, False, torch_fixture(0)),
                         ("cpu", "cpu"))
        with self.assertRaisesRegex(ValueError, "non-Kaggle"):
            launcher.select_devices({"ALLOW_CPU": True}, True, torch_fixture(2))

    def test_download_link_is_a_basename(self):
        links, displayed = [], []
        module = SimpleNamespace(FileLink=lambda path: links.append(path) or path,
                                 display=displayed.append)
        with patch.dict(sys.modules, {"IPython.display": module}):
            launcher.show_archive(Path("/kaggle/working/result.zip"))
        self.assertEqual(links, ["result.zip"])
        self.assertEqual(displayed, ["result.zip"])


class InputAndResumeTests(unittest.TestCase):
    def test_preflight_accepts_arbitrary_zip_name_and_folder_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            (original / "states.jsonl").write_text("{}\n", encoding="utf-8")
            archive = root / "any_name.data"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("nested/summary.json", "{}")
            self.assertEqual(launcher.preflight_input(original, root), original.resolve())
            self.assertEqual(launcher.preflight_input(archive, root), archive.resolve())

    def test_missing_mount_prints_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "mount"
            (mount / "datasets" / "owner" / "actual_folder").mkdir(parents=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(FileNotFoundError):
                launcher.preflight_input(mount / "absent", mount)
            self.assertIn("actual_folder", output.getvalue())
            self.assertIn("Requested RUN_DIR", output.getvalue())

    def test_resolver_bridge_uses_actual_data_module_and_accepts_nested_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "anything.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                prefix = "wrapper/original/"
                zipped.writestr(prefix + "summary.json", json.dumps({
                    "schema": "interactive_acceptance_two_source_v1", "status": "complete",
                    "questions_completed": 100}))
                for name in ("states.jsonl", "edges.jsonl", "labels.jsonl", "teacher_targets.jsonl"):
                    zipped.writestr(prefix + name, "")
                zipped.writestr(prefix + "experience/arbitrary.npz", b"not loaded by resolver")
            paths = list(sys.path)
            self.assertEqual(launcher.resolve_original_input(ROOT, root), archive.resolve())
            self.assertEqual(sys.path, paths)
            self.assertNotIn("_factorized_launcher_data", sys.modules)

    def test_safe_resume_zip_preserves_config_manifest_and_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = previous_result(root / "old")
            archive = root / "uploaded.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for path in source.rglob("*"):
                    if path.is_file():
                        zipped.write(path, "wrapper/old/" + path.relative_to(source).as_posix())
            temp = root / "temp"
            temp.mkdir()
            output = root / "working" / "fresh"
            restored = launcher.restore_resume_input(archive, output, temp)
            self.assertEqual(restored, output.resolve())
            self.assertTrue((restored / "jobs/saved_job/checkpoint.pt").is_file())
            self.assertTrue(archive.is_file())

    def test_resume_rejects_traversal_windows_paths_and_symlinks_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("../escape", "/absolute", "C:\\escape", "..\\escape")):
                archive = root / f"unsafe_{index}.zip"
                destination = root / f"extract_{index}"
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("safe.txt", "must not be extracted")
                    zipped.writestr(name, "bad")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                    launcher.safe_extract(archive, destination)
                self.assertFalse(destination.exists())
            symlink = zipfile.ZipInfo("link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive = root / "symlink.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(symlink, "../../outside")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                launcher.safe_extract(archive, root / "links")

    def test_resume_requires_unique_root_and_real_saved_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = previous_result(root / "first")
            self.assertEqual(launcher.locate_resume_root(root), first.resolve())
            previous_result(root / "second")
            with self.assertRaisesRegex(ValueError, "one previous result root"):
                launcher.locate_resume_root(root)
            (first / "study_manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                launcher.validate_resume_root(first)

    def test_resume_validates_runner_manifest_fields_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = previous_result(Path(directory) / "result")
            path = root / "study_manifest.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            for key, value, message in (
                ("fingerprint", "not-a-digest", "SHA256 fingerprint"),
                ("config", {"num_questions": 1}, "config differs"),
                ("source", "", "original source"),
                ("status", "unknown", "status must"),
            ):
                invalid = dict(original, **{key: value})
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    launcher.validate_resume_root(root)
            complete = dict(original, status="complete")
            path.write_text(json.dumps(complete), encoding="utf-8")
            self.assertEqual(launcher.validate_resume_root(root), root.resolve())

    def test_legacy_manifest_name_is_not_a_runner_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = previous_result(Path(directory) / "result")
            (root / "study_manifest.json").rename(root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "study_manifest.json"):
                launcher.validate_resume_root(root)

    def test_partial_archive_keeps_small_checkpoints_and_excludes_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = previous_result(Path(directory) / "result")
            for name in ("input_cache/large.npz", "experience/shard.npz", "raw_experiences/raw.npz"):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"raw fixture")
            with (output / "huge_weights.bin").open("wb") as stream:
                stream.truncate(launcher.MAX_ARCHIVE_FILE_BYTES + 1)
            archive = launcher.archive_partial(output)
            self.assertEqual(archive, output.with_suffix(".zip"))
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn("jobs/saved_job/checkpoint.pt", names)
                self.assertIn("config.json", names)
                self.assertFalse(any(".npz" in name for name in names))
                self.assertNotIn("huge_weights.bin", names)
                self.assertEqual(len(json.loads(zipped.read("launcher_packaging.json"))["excluded"]), 4)
                self.assertIsNone(zipped.testzip())


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.working = self.root / "working"
        self.working.mkdir()
        self.input = self.root / "input"
        self.input.mkdir()
        (self.input / "summary.json").write_text("{}", encoding="utf-8")
        self.sentinel = self.working / "keep.txt"
        self.sentinel.write_text("user output", encoding="utf-8")
        self.scope = {"WORKING_DIR": str(self.working), "TEMP_DIR": str(self.root / "temp"),
                      "RUN_DIR": str(self.input), "MODE": "smoke", "ALLOW_CPU": True}
        self.commands = []

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary.cleanup()

    def fake_download(self, repo, ref, env):
        self.assertEqual(Path.cwd(), self.working.resolve())
        self.assertEqual(ref, launcher.DEFAULT_SOURCE_REF)
        self.assertTrue(repo.is_relative_to(self.root / "temp"))
        (repo / "configs").mkdir(parents=True)
        (repo / launcher.CONFIG_RELATIVE).write_text(json.dumps(read_config()), encoding="utf-8")
        return "a" * 40

    def fake_run(self, command, *, cwd, env):
        command = [str(part) for part in command]
        self.commands.append(command)
        if any(part.endswith("run_factorized_wm_feasibility.py") for part in command):
            output = Path(command[command.index("--output") + 1])
            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not (output / "config.json").exists():
                previous_result(output, config)

    def launch_mocked(self, scope=None, command_runner=None, show_callback=None):
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "select_devices", return_value=("cpu", "cpu")), \
                patch.object(launcher, "download_source", side_effect=self.fake_download), \
                patch.object(launcher, "resolve_original_input", return_value=self.input), \
                patch.object(launcher, "run_command", side_effect=command_runner or self.fake_run), \
                patch.object(launcher, "show_archive", side_effect=show_callback) as show, \
                patch("shutil.rmtree", side_effect=AssertionError("launcher removed a folder")):
            result = launcher.launch(scope or self.scope)
            return result, show.call_args.args[0]

    def test_launch_contract_test_order_devices_and_unique_temp(self):
        output, archive = self.launch_mocked()
        self.assertTrue(output.is_relative_to(self.working.resolve()))
        self.assertTrue(archive.is_file())
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "user output")
        expected_tests = (
            "tests/test_factorized_wm_data.py",
            "tests/test_factorized_wm_metrics.py",
            "tests/test_factorized_wm_models.py",
            "tests/test_factorized_kaggle_launcher.py",
            "tests/test_factorized_wm_runner.py",
        )
        self.assertEqual(launcher.TEST_FILES, expected_tests)
        self.assertEqual([Path(command[1]).name for command in self.commands[1:1+len(expected_tests)]],
                         [Path(path).name for path in expected_tests])
        runner = self.commands[-1]
        self.assertEqual(runner[runner.index("--input") + 1], str(self.input))
        self.assertNotIn("--resume", runner)
        config = json.loads((output / "launcher_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["device_drafter"], config["device_verifier"]), ("cpu", "cpu"))
        self.assertEqual(config["num_questions"], 20)
        self.assertEqual(self.commands[0][-3:], ["numpy", "scikit-learn", "matplotlib"])
        first_temp = Path(self.commands[-1][2]).parent.parent
        self.commands.clear()
        self.launch_mocked()
        self.assertNotEqual(first_temp, Path(self.commands[-1][2]).parent.parent)

    def test_failure_creates_zip_and_reports_it_before_raising(self):
        def fail_runner(command, *, cwd, env):
            self.fake_run(command, cwd=cwd, env=env)
            if any(str(part).endswith("run_factorized_wm_feasibility.py") for part in command):
                raise subprocess.CalledProcessError(7, command)
        shown = []
        def show(archive):
            self.assertTrue(archive.is_file())
            shown.append(archive)
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch_mocked(command_runner=fail_runner, show_callback=show)
        archives = list(self.working.glob("*.zip"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(shown, archives)
        with zipfile.ZipFile(archives[0]) as zipped:
            self.assertIn("launcher_error.txt", zipped.namelist())
            self.assertIn("CalledProcessError", zipped.read("launcher_error.txt").decode())
        self.assertTrue(self.sentinel.is_file())

    def test_preflight_failure_happens_before_any_download(self):
        scope = dict(self.scope, RUN_DIR=str(self.root / "missing"))
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "download_source") as download, \
                patch.object(launcher, "run_command") as run, \
                patch.object(launcher, "show_archive") as show:
            with self.assertRaises(FileNotFoundError):
                launcher.launch(scope)
        download.assert_not_called()
        run.assert_not_called()
        self.assertTrue(show.call_args.args[0].is_file())

    def test_resume_output_uses_exact_existing_path(self):
        previous = previous_result(self.working / "previous", launcher.build_config(read_config(), self.scope))
        output, archive = self.launch_mocked(dict(self.scope, RESUME_OUTPUT=str(previous)))
        self.assertEqual(output, previous.resolve())
        self.assertEqual(archive, previous.with_suffix(".zip"))
        self.assertIn("--resume", self.commands[-1])
        self.assertTrue((previous / "jobs/saved_job/checkpoint.pt").is_file())

    def test_resume_output_outside_working_is_rejected_without_download(self):
        previous = previous_result(self.root / "outside")
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "download_source") as download, \
                patch.object(launcher, "show_archive"):
            with self.assertRaisesRegex(ValueError, "inside the working"):
                launcher.launch(dict(self.scope, RESUME_OUTPUT=str(previous)))
        download.assert_not_called()
        self.assertFalse((previous / "launcher_error.txt").exists())


if __name__ == "__main__":
    unittest.main()
