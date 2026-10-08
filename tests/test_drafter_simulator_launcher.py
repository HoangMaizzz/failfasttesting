"""Launcher contract checks using stdlib mocks: no network, GPU or training."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_drafter_simulator as launcher


def versions():
    return {"python": "3.11.0", "packages": {
        "torch": "2.6.0+cu124", "transformers": "4.53.1", "huggingface_hub": "0.36.0",
    }}


class ConfigTests(unittest.TestCase):
    def test_full_defaults_and_independent_configs(self):
        original = copy.deepcopy(launcher.DEFAULT_CONFIG)
        config = launcher.build_config({})
        self.assertEqual(config, original)
        self.assertEqual(config["num_questions"], 100)
        self.assertEqual(config["max_new_tokens"], 256)
        self.assertEqual(config["max_context_tokens"], 4096)
        self.assertEqual((config["physical_block_size"], config["small_block_size"]), (32, 8))
        self.assertEqual(config["threshold"], .5)
        self.assertEqual(config["learning_milestones"], [20, 40, 70])
        self.assertEqual(config["seeds"], [42, 43, 44])
        self.assertEqual((config["encoder_updates"], config["updates"], config["eval_every"]), (400, 600, 100))
        self.assertEqual((config["benchmark_repetitions"], config["benchmark_warmup"]), (100, 20))
        self.assertEqual((config["collect_device"], config["train_device"]), ("cuda:0", "cuda:1"))
        config["seeds"].append(99)
        self.assertEqual(launcher.DEFAULT_CONFIG, original)

    def test_smoke_and_requested_scope_overrides(self):
        config = launcher.build_config({"MODE": "smoke"})
        self.assertEqual(config["num_questions"], 3)
        self.assertEqual(config["learning_milestones"], [1])
        self.assertEqual(config["seeds"], [42])
        config = launcher.build_config({
            "MODE": "smoke", "NUM_QUESTIONS": 100, "MAX_NEW_TOKENS": 64,
            "ENCODER_UPDATES": 10, "UPDATES": 20, "SEEDS": (43, 44),
        })
        self.assertEqual([config[k] for k in ("num_questions", "max_new_tokens", "encoder_updates", "updates")],
                         [100, 64, 10, 20])
        self.assertEqual(config["seeds"], [43, 44])

    def test_data_parameters_and_explicit_override_precedence(self):
        config = launcher.build_config({"DATAPARAM": {"num_questions": 3, "max_context_tokens": 2048},
                                        "NUM_QUESTIONS": 20})
        self.assertEqual(config["num_questions"], 20)
        self.assertEqual(config["max_context_tokens"], 2048)
        self.assertEqual(config["learning_milestones"], [14])
        for value in (None, [], {"target_device": "cuda:0"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                launcher.build_config({"DATAPARAM": value})

    def test_invalid_configs_fail_early(self):
        for scope in ({"MODE": "unknown"}, {"NUM_QUESTIONS": 2}, {"NUM_QUESTIONS": True},
                      {"MAX_NEW_TOKENS": 0}, {"MAX_NEW_TOKENS": 8}, {"UPDATES": "600"}, {"ENCODER_UPDATES": -1},
                      {"SEEDS": []}, {"SEEDS": [42, 42]}, {"SEEDS": [True]},
                      {"SEEDS": [-1]}, {"SEEDS": [2**32]}, {"SEEDS": "42"},
                      {"DATAPARAM": {"physical_block_size": 8}},
                      {"DATAPARAM": {"threshold": .6}},
                      {"DATAPARAM": {"threshold": float("nan")}}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                launcher.build_config(scope)

    def test_import_is_inert_even_with_notebook_overrides(self):
        spec = importlib.util.spec_from_file_location("_launcher_import_test", ROOT / "kaggle_drafter_simulator.py")
        module = importlib.util.module_from_spec(spec)
        with patch("subprocess.run", side_effect=AssertionError("import launched work")), \
                patch("subprocess.check_output", side_effect=AssertionError("import queried runtime")), \
                patch("tempfile.mkdtemp", side_effect=AssertionError("import created folder")), \
                patch("os.chdir", side_effect=AssertionError("import changed cwd")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.launch))

    def test_main_guard_calls_launch_with_exec_scope(self):
        tree = __import__("ast").parse((ROOT / "kaggle_drafter_simulator.py").read_text(encoding="utf-8"))
        guard = tree.body[-1]
        code = compile(__import__("ast").Module(body=[guard], type_ignores=[]), "guard", "exec")
        launch = Mock()
        scope = {"__name__": "__main__", "launch": launch, "NUM_QUESTIONS": 3}
        exec(code, scope)
        launch.assert_called_once_with(scope)
        launch.reset_mock()
        exec(code, dict(scope, __name__="imported"))
        launch.assert_not_called()


class HelpersTests(unittest.TestCase):
    def test_source_fetch_branch_or_sha_is_detached_and_preserves_parent(self):
        for ref in (launcher.DEFAULT_SOURCE_REF, "b" * 40):
            with self.subTest(ref=ref), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory) / "repo"
                with patch.object(launcher, "run_command") as run, \
                        patch("subprocess.check_output", return_value="a" * 40 + "\n"):
                    self.assertEqual(launcher.download_source(repo, ref, {}), "a" * 40)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[2][-2:], ["origin", ref])
                self.assertEqual(commands[3][-3:], ["checkout", "--detach", "FETCH_HEAD"])
                self.assertTrue(all(call.kwargs["cwd"] == repo.parent for call in run.call_args_list))
        with patch.object(launcher, "run_command") as run, self.assertRaises(ValueError):
            launcher.download_source(Path("unused"), "--upload-pack=bad", {})
        run.assert_not_called()

    def test_dependencies_pin_transformers_hub_and_constrain_existing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launcher.metadata, "version", return_value="2.6.0+cu124"), \
                    patch.object(launcher, "installed_versions", return_value=versions()), \
                    patch.object(launcher, "run_command") as run:
                launcher.install_dependencies(root, root, {})
            command = [str(part) for part in run.call_args.args[0]]
            self.assertIn("transformers==4.53.1", command)
            self.assertIn("huggingface_hub<1", command)
            for dependency in ("datasets", "scipy", "numpy", "matplotlib", "accelerate", "einops"):
                self.assertIn(dependency, command)
            self.assertNotIn("torch", command)
            constraint = Path(command[command.index("--constraint") + 1])
            self.assertEqual(constraint.read_text(encoding="utf-8"), "torch==2.6.0+cu124\n")
            self.assertEqual(json.loads((root / "versions.json").read_text(encoding="utf-8")), versions())

    def test_torch_replacement_is_a_failure(self):
        changed = versions()
        changed["packages"]["torch"] = "2.7.0"
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(launcher.metadata, "version", return_value="2.6.0+cu124"), \
                patch.object(launcher, "installed_versions", return_value=changed), \
                patch.object(launcher, "run_command"), self.assertRaisesRegex(RuntimeError, "changed PyTorch"):
            launcher.install_dependencies(Path(directory), Path(directory), {})

    def test_gpu_selection_two_one_and_no_gpu(self):
        for count, expected in ((2, ("cuda:0", "cuda:1")), (1, ("cuda:0", "cuda:0"))):
            with self.subTest(count=count), patch("subprocess.check_output", return_value=json.dumps({
                "available": True, "count": count, "names": ["T4"] * count,
            })):
                self.assertEqual(launcher.select_devices({}, cwd=ROOT, env={}), expected)
        with patch("subprocess.check_output", return_value=json.dumps({"available": False, "count": 0})), \
                self.assertRaisesRegex(RuntimeError, "T4 x2"):
            launcher.select_devices({}, cwd=ROOT, env={})

    def test_weight_metadata_resolves_sha_before_snapshot_and_overlays_modeling(self):
        self.run_weight_fixture()

    def test_weight_download_failure_retains_revision_without_fake_completion(self):
        self.run_weight_fixture(fail=True)

    def run_weight_fixture(self, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "hf" / "weights"
            overlay = root / "modeling.py"
            overlay.write_text("# repository observer overlay\n", encoding="utf-8")
            record_path = root / "model_metadata.json"
            api = Mock()
            api.model_info.return_value = SimpleNamespace(sha="c" * 40)
            hub = ModuleType("huggingface_hub")
            hub.HfApi = Mock(return_value=api)
            def snapshot(**kwargs):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                self.assertEqual(record["resolved_revision"], "c" * 40)
                self.assertFalse(record["download_complete"])
                self.assertEqual(kwargs["revision"], "c" * 40)
                self.assertEqual(kwargs["repo_id"], launcher.MODEL_ID)
                self.assertIn("*.py", kwargs["allow_patterns"])
                self.assertIn("*.safetensors", kwargs["allow_patterns"])
                if fail:
                    raise OSError("model download interrupted")
                destination.mkdir(parents=True)
                (destination / "config.json").write_text("{}", encoding="utf-8")
                (destination / "model.safetensors").write_bytes(b"fixture")
                (destination / "modeling.py").write_text("# upstream\n", encoding="utf-8")
            hub.snapshot_download = Mock(side_effect=snapshot)
            argv = ["-c", launcher.MODEL_ID, "main", str(destination), str(overlay),
                    str(record_path), json.dumps(launcher.WEIGHT_PATTERNS)]
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(sys, "argv", argv):
                if fail:
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        exec(launcher.WEIGHT_DOWNLOAD_CODE, {})
                else:
                    exec(launcher.WEIGHT_DOWNLOAD_CODE, {})
                    self.assertEqual((destination / "modeling.py").read_bytes(), overlay.read_bytes())
            api.model_info.assert_called_once_with(launcher.MODEL_ID, revision="main")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["download_complete"], not fail)
            self.assertEqual(len(record["overlay_sha256"]), 64)

    def test_filelink_uses_filename(self):
        display_module = ModuleType("IPython.display")
        display_module.FileLink = Mock(return_value="link")
        display_module.display = Mock()
        printed = io.StringIO()
        with patch.dict(sys.modules, {"IPython.display": display_module}), contextlib.redirect_stdout(printed):
            launcher.show_archive(Path("/kaggle/working/result.zip"))
        display_module.FileLink.assert_called_once_with("result.zip")
        self.assertIn("Output ZIP ready:", printed.getvalue())
        self.assertIn("result.zip", printed.getvalue())

    def test_archive_refresh_excludes_weights_and_keeps_small_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            for name in ("config.json", "models/encoder.pt", "metrics.json", "hf/model.safetensors",
                         "weights/pytorch_model.bin", "model-00001-of-00002.safetensors"):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            archive = launcher.archive_partial(output)
            first_bytes = archive.read_bytes()
            launcher.archive_partial(output)
            self.assertEqual(archive.read_bytes(), first_bytes)
            (output / "launcher_error.txt").write_text("late error", encoding="utf-8")
            self.assertEqual(launcher.archive_partial(output), archive)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("launcher_error.txt", zipped.namelist())
                self.assertIn("models/encoder.pt", zipped.namelist())
                self.assertFalse(any(name.endswith((".safetensors", ".bin")) for name in zipped.namelist()))
                self.assertIsNone(zipped.testzip())
            self.assertIsNone(launcher.archive_partial(Path(directory) / "absent"))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.working = self.root / "working"
        self.working.mkdir()
        self.scope = {"WORKING_DIR": str(self.working), "TEMP_DIR": str(self.root / "temp")}
        self.commands = []
        self.sentinel = self.working / "keep.txt"
        self.sentinel.write_text("existing user output", encoding="utf-8")
        self.shown = []

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary.cleanup()

    def fake_source(self, repo, ref, env):
        self.assertEqual(Path.cwd(), self.working)
        repo.mkdir()
        (repo / launcher.RUNNER_RELATIVE).write_text("# runner fixture", encoding="utf-8")
        return "a" * 40

    def fake_install(self, temp, output, env):
        launcher.write_json(output / "versions.json", versions())

    def fake_weights(self, repo, temp, output, scope, env):
        launcher.write_json(output / "model_metadata.json", {
            "model_id": launcher.MODEL_ID, "resolved_revision": "b" * 40, "download_complete": True,
        })
        destination = temp / "hf" / "Fast_dLLM_v2_1_5B"
        destination.mkdir(parents=True)
        return destination

    def fake_run(self, command, *, cwd, env, check=True):
        command = [str(part) for part in command]
        self.commands.append((command, Path(cwd), env.copy(), check))
        if str(launcher.RUNNER_RELATIVE) in [Path(part).name for part in command]:
            output = Path(command[command.index("--output_dir") + 1])
            self.assertFalse(list(output.iterdir()), "runner must receive an empty fresh output folder")
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            launcher.write_json(output / "summary.json", {"status": "complete"})
            # Mimic the runner's own finally packaging before launcher metadata.
            launcher.archive_partial(output)
        return SimpleNamespace(returncode=0)

    def launch_mocked(self, scope=None, fail_stage=None):
        def maybe_fail(name, callback):
            if name == fail_stage:
                return Mock(side_effect=RuntimeError(f"{name} failed"))
            return callback
        with patch.object(launcher, "download_source", side_effect=maybe_fail("source", self.fake_source)), \
                patch.object(launcher, "install_dependencies", side_effect=maybe_fail("dependencies", self.fake_install)), \
                patch.object(launcher, "select_devices", return_value=("cuda:0", "cuda:1")), \
                patch.object(launcher, "download_weights", side_effect=maybe_fail("weights", self.fake_weights)), \
                patch.object(launcher, "run_command", side_effect=self.fake_run), \
                patch.object(launcher, "show_archive", side_effect=self.shown.append), \
                contextlib.redirect_stdout(io.StringIO()):
            return launcher.launch(scope or self.scope)

    def test_full_launch_exact_cli_metadata_tests_and_unique_paths(self):
        output = self.launch_mocked()
        self.assertEqual(output.parent, self.working)
        self.assertTrue(self.shown[0].is_file())
        metadata = json.loads((output / "launcher_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_revision"], "a" * 40)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["runner_summary_status"], "complete")
        self.assertEqual(metadata["runner_returncode"], 0)
        test, runner = self.commands
        self.assertEqual(test[0][1:], ["-m", "unittest", "discover", "-s", "tests", "-p", "test_drafter_simulator_*.py"])
        command, cwd, env, check = runner
        self.assertFalse(check)
        self.assertEqual(command[3::2], ["--config_json", "--output_dir", "--dllm_dir"])
        self.assertEqual(Path(command[2]).name, "run_drafter_simulator.py")
        config_path, output_path, weights_path = map(Path, command[4::2])
        self.assertEqual(output_path, output)
        self.assertTrue(config_path.is_relative_to(self.root / "temp"))
        self.assertTrue(weights_path.is_relative_to(self.root / "temp"))
        self.assertFalse(weights_path.is_relative_to(self.working))
        self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")),
                         dict(launcher.DEFAULT_CONFIG, source_revision="a" * 40))
        self.assertTrue(Path(env["HF_HOME"]).is_relative_to(self.root / "temp"))
        self.assertEqual(env["GIT_LFS_SKIP_SMUDGE"], "1")
        self.assertEqual(env["USE_TF"], "0")
        self.assertEqual(env["USE_FLAX"], "0")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertTrue(cwd.is_relative_to(self.root / "temp"))
        self.assertEqual(Path.cwd(), self.working)
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "existing user output")
        with zipfile.ZipFile(self.shown[0]) as zipped:
            self.assertEqual(json.loads(zipped.read("launcher_metadata.json"))["status"], "complete")
            self.assertIn("launcher_config.json", zipped.namelist())
            self.assertIn("versions.json", zipped.namelist())
            self.assertIn("model_metadata.json", zipped.namelist())
        first_temp = metadata["temp_dir"]
        output2 = self.launch_mocked()
        self.assertNotEqual(output, output2)
        self.assertNotEqual(first_temp, json.loads((output2 / "launcher_metadata.json").read_text())["temp_dir"])

    def test_setup_failures_package_diagnostics_and_never_run_model(self):
        for stage in ("source", "dependencies", "weights"):
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                self.launch_mocked(fail_stage=stage)
            archive = self.shown[-1]
            self.assertEqual(archive.parent, self.working)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("launcher_error.txt", zipped.namelist())
                self.assertEqual(json.loads(zipped.read("launcher_metadata.json"))["status"], "failed")
                if stage in ("dependencies", "weights"):
                    config = json.loads(zipped.read("launcher_config.json"))
                    self.assertEqual(config["source_revision"], "a" * 40)
        self.assertFalse(any(Path(cmd[0][2]).name == str(launcher.RUNNER_RELATIVE) for cmd in self.commands))

    def test_nonzero_runner_exit_refreshes_runner_zip_and_raises(self):
        original_run = self.fake_run
        def failed_run(command, **kwargs):
            result = original_run(command, **kwargs)
            if any(str(part).endswith(str(launcher.RUNNER_RELATIVE)) for part in command):
                result.returncode = 7
            return result
        with patch.object(self, "fake_run", side_effect=failed_run), self.assertRaises(subprocess.CalledProcessError) as caught:
            self.launch_mocked()
        self.assertEqual(caught.exception.returncode, 7)
        with zipfile.ZipFile(self.shown[-1]) as zipped:
            self.assertIn("metrics.json", zipped.namelist())
            self.assertIn("launcher_error.txt", zipped.namelist())
            record = json.loads(zipped.read("launcher_metadata.json"))
            self.assertEqual(record["runner_returncode"], 7)
            self.assertEqual(record["status"], "failed")

    def test_bad_scope_and_missing_runner_fail_before_install(self):
        with patch.object(launcher, "download_source") as source, \
                patch.object(launcher, "install_dependencies") as install, \
                patch.object(launcher, "show_archive", side_effect=self.shown.append):
            with self.assertRaises(ValueError):
                launcher.launch(dict(self.scope, NUM_QUESTIONS=2))
            source.assert_not_called()
            def missing(repo, ref, env):
                repo.mkdir()
                return "d" * 40
            source.side_effect = missing
            with self.assertRaises(FileNotFoundError):
                launcher.launch(self.scope)
            install.assert_not_called()
        self.assertEqual(len(self.shown), 2)

    def test_inherited_offline_flags_are_cleared_for_children_only(self):
        flags = {key: "1" for key in launcher.OFFLINE_FLAGS}
        with patch.dict(os.environ, flags):
            self.launch_mocked()
            self.assertTrue(all(os.environ[key] == "1" for key in flags))
        for command, cwd, env, check in self.commands:
            self.assertFalse(any(key in env for key in flags))

    def test_zero_exit_without_complete_manifest_is_failure(self):
        original_run = self.fake_run
        for summary in (None, {"status": "partial"}, {"status": "running"}, [], "invalid JSON"):
            def incomplete_run(command, **kwargs):
                result = original_run(command, **kwargs)
                if any(str(part).endswith(str(launcher.RUNNER_RELATIVE)) for part in command):
                    output = Path(command[command.index("--output_dir") + 1])
                    path = output / "summary.json"
                    if summary is None:
                        path.unlink()
                    elif summary == "invalid JSON":
                        path.write_text(summary, encoding="utf-8")
                    else:
                        launcher.write_json(path, summary)
                return result
            with self.subTest(summary=summary), patch.object(self, "fake_run", side_effect=incomplete_run), \
                    self.assertRaisesRegex(RuntimeError, "summary.json"):
                self.launch_mocked()
            with zipfile.ZipFile(self.shown[-1]) as zipped:
                record = json.loads(zipped.read("launcher_metadata.json"))
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["runner_returncode"], 0)
                self.assertIn("launcher_error.txt", zipped.namelist())

    def test_packaging_failure_does_not_hide_original_setup_exception(self):
        with patch.object(launcher, "archive_partial", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "dependencies failed"):
            self.launch_mocked(fail_stage="dependencies")


if __name__ == "__main__":
    unittest.main()
