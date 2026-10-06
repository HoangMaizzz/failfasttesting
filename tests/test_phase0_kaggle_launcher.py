"""Offline launcher contract checks: no network, embeddings or model training."""
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
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_latent_wm_phase0 as launcher


def read_config():
    return json.loads((ROOT / launcher.CONFIG_RELATIVE).read_text(encoding="utf-8"))


def original_members():
    members = {"summary.json": json.dumps({
        "schema": "interactive_acceptance_two_source_v1", "status": "complete",
        "questions_completed": 100,
    })}
    for name in ("states.jsonl", "edges.jsonl", "labels.jsonl", "teacher_targets.jsonl"):
        members[name] = ""
    members["experience/not_a_checkpoint.npz"] = b"resolver never loads these fixture arrays"
    return members


def trace_zip(path, prefix="wrapper/original/"):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, content in original_members().items():
            zipped.writestr(prefix + name, content)
    return path


def previous_result(root, config=None):
    root.mkdir(parents=True, exist_ok=True)
    config = config if config is not None else launcher.build_config(read_config(), {})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    declared = config.get("protocol", {}).get("resume", {}).get("runner_manifest", "study_manifest.json")
    (root / declared).write_text(json.dumps({
        "schema": "latent_wm_phase0_v1", "fingerprint": "b" * 64,
        "config": config, "status": "running", "source": "fixture/original_trace.data",
    }), encoding="utf-8")
    checkpoint = root / "models" / "latent64" / "stage_a.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"small model checkpoint fixture")
    return root


class ConfigTests(unittest.TestCase):
    def test_full_preserves_json_budgets_and_pinned_pretrained_embedding(self):
        base = read_config()
        before = copy.deepcopy(base)
        config = launcher.build_config(base, {})
        for key in ("stage_a_updates", "raw_updates", "dynamics_updates_per_horizon",
                    "direct_updates", "batch_size", "eval_every", "minimum_updates"):
            self.assertEqual(config[key], base[key], key)
        self.assertEqual(config["num_questions"], 100)
        self.assertEqual(config["latent_dims"], [64, 128])
        self.assertEqual(config["token_embedding"], "pretrained")
        self.assertEqual(config["embedding_repo_id"], "Efficient-Large-Model/Fast_dLLM_v2_1.5B")
        self.assertEqual(config["embedding_revision"], "cd3af22d326325d015267a7845b5cd5e91a28fa7")
        self.assertEqual(config["embedding_dim"], 64)
        self.assertEqual((config["device_encoder"], config["device_dynamics"]), ("cuda:0", "cuda:1"))
        self.assertEqual(config["protocol"]["split_counts"], {"train": 70, "validation": 15, "test": 15})
        self.assertFalse(config["continue_diagnostics"])
        self.assertEqual(base, before)

    def test_smoke_effective_config_shrinks_all_actual_training_budgets(self):
        base = read_config()
        config = launcher.build_config(base, {"MODE": "smoke"})
        self.assertEqual(config["num_questions"], 20)
        self.assertEqual(config["latent_dims"], [64])
        self.assertEqual(config["stage_a_updates"], 10)
        for key in ("raw_updates", "dynamics_updates_per_horizon", "direct_updates"):
            self.assertEqual(config[key], 20, key)
        self.assertEqual(config["minimum_updates"], 5)
        self.assertEqual(config["eval_every"], 5)
        self.assertEqual(config["patience"], 2)
        self.assertEqual(config["protocol"]["split_counts"], {"train": 14, "validation": 3, "test": 3})
        self.assertEqual(config["embedding_revision"], base["embedding_revision"])
        self.assertEqual(config["token_embedding"], "pretrained")
        self.assertFalse(config["continue_diagnostics"])
        self.assertEqual(base["stage_a_updates"], 1600)
        self.assertEqual(base["latent_dims"], [64, 128])

    def test_embedding_and_diagnostics_overrides_are_effective_and_explicit(self):
        config = launcher.build_config(read_config(), {
            "TOKEN_EMBEDDING": "learned", "EMBEDDING_PATH": Path("/kaggle/input/embedding/table.safetensors"),
            "CONTINUE_DIAGNOSTICS": True,
        })
        self.assertEqual(config["token_embedding"], "learned")
        self.assertEqual(config["embedding_path"], str(Path("/kaggle/input/embedding/table.safetensors")))
        self.assertTrue(config["continue_diagnostics"])
        self.assertFalse(read_config()["continue_diagnostics"])
        self.assertIsNone(launcher.build_config(read_config(), {"EMBEDDING_PATH": None})["embedding_path"])

    def test_invalid_contract_or_override_fails_instead_of_falling_back(self):
        for scope in ({"MODE": "typo"}, {"TOKEN_EMBEDDING": "auto"},
                      {"EMBEDDING_PATH": ""}, {"CONTINUE_DIAGNOSTICS": "yes"}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                launcher.build_config(read_config(), scope)
        for key, value in (("num_questions", 20), ("latent_dims", [64]), ("protocol", [])):
            base = read_config()
            base[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.build_config(base, {})
        base = read_config()
        base["protocol"]["split_fractions"] = [.8, .1, .1]
        with self.assertRaisesRegex(ValueError, "split fractions"):
            launcher.build_config(base, {})


class ImportDeviceSourceTests(unittest.TestCase):
    def test_import_is_standalone_and_has_no_launch_side_effects(self):
        spec = importlib.util.spec_from_file_location("_phase0_import_check", ROOT / "kaggle_latent_wm_phase0.py")
        module = importlib.util.module_from_spec(spec)
        with patch("os.chdir", side_effect=AssertionError("changed cwd")), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("created folder")), \
                patch("subprocess.run", side_effect=AssertionError("ran command")), \
                patch("subprocess.check_output", side_effect=AssertionError("downloaded source")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.launch))

    def test_documented_notebook_cell_activates_once_with_existing_scope(self):
        document = (ROOT / "PHASE0_LATENT_WM.md").read_text(encoding="utf-8")
        cell = document.split("```python\n", 1)[1].split("```", 1)[0]
        source = (ROOT / "kaggle_latent_wm_phase0.py").read_text(encoding="utf-8")
        # Load the real launcher definitions with its normal activation guard,
        # then mock only the explicit activation to avoid any experiment work.
        payload = source + "\nfrom unittest.mock import Mock\nlaunch = Mock(return_value='mock-output')\n"
        scope = {"__name__": "__main__", "SOURCE_REF": "a" * 40,
                 "MODE": "smoke", "RUN_DIR": "/kaggle/input/original.data",
                 "TOKEN_EMBEDDING": "learned", "CONTINUE_DIAGNOSTICS": True}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload.encode())) as download, \
                patch("os.chdir") as change_directory, \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("implicit activation")), \
                patch("subprocess.run", side_effect=AssertionError("unmocked command")):
            exec(compile(cell, "PHASE0_LATENT_WM.md:one-cell", "exec"), scope)
        change_directory.assert_called_once_with("/kaggle/working")
        self.assertIn("/" + "a" * 40 + "/kaggle_latent_wm_phase0.py", download.call_args.args[0])
        activate = scope["launcher"]["launch"]
        self.assertEqual(activate.call_count, 1)
        self.assertIs(activate.call_args.args[0], scope)
        self.assertEqual(scope["OUTPUT"], "mock-output")
        self.assertEqual(scope["MODE"], "smoke")
        self.assertEqual(scope["RUN_DIR"], "/kaggle/input/original.data")
        self.assertEqual(scope["TOKEN_EMBEDDING"], "learned")
        self.assertTrue(scope["CONTINUE_DIAGNOSTICS"])

    def test_gpu_assignment_and_no_implicit_cpu_fallback(self):
        def fake_torch(count):
            return SimpleNamespace(cuda=SimpleNamespace(
                device_count=lambda: count, is_available=lambda: count > 0,
                get_device_name=lambda i: f"GPU{i}"))
        self.assertEqual(launcher.select_devices({}, True, fake_torch(2)), ("cuda:0", "cuda:1"))
        for count in (0, 1):
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, "two visible GPUs"):
                launcher.select_devices({}, True, fake_torch(count))
        self.assertEqual(launcher.select_devices({"ALLOW_CPU": True}, False, fake_torch(0)), ("cpu", "cpu"))
        with self.assertRaisesRegex(ValueError, "non-Kaggle"):
            launcher.select_devices({"ALLOW_CPU": True}, True, fake_torch(2))

    def test_source_ref_supports_branch_and_sha_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, ref in enumerate((launcher.DEFAULT_SOURCE_REF, "a" * 40)):
                repo = Path(directory) / str(index)
                with patch.object(launcher, "run_command") as run, \
                        patch("subprocess.check_output", return_value="b" * 40 + "\n"):
                    self.assertEqual(launcher.download_source(repo, ref, {}), "b" * 40)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[2], ["git", "-C", repo, "fetch", "--depth", "1", "origin", ref])
                self.assertEqual(commands[3][-2:], ["--detach", "FETCH_HEAD"])
                self.assertIn(launcher.SOURCE_REPO, commands[1])
            for ref in ("--upload-pack=bad", "main;bad", "$(bad)", "", None):
                with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "SOURCE_REF"):
                    launcher.download_source(Path(directory) / "invalid", ref, {})
            self.assertFalse((Path(directory) / "invalid").exists())

    def test_filelink_uses_only_working_root_zip_basename(self):
        links, displayed = [], []
        module = SimpleNamespace(FileLink=lambda path: links.append(path) or path, display=displayed.append)
        with patch.dict(sys.modules, {"IPython.display": module}):
            launcher.show_archive(Path("/kaggle/working/latent_phase0.zip"))
        self.assertEqual(links, ["latent_phase0.zip"])
        self.assertEqual(displayed, links)


class ContentAndResumeTests(unittest.TestCase):
    def test_content_discovery_arbitrary_zip_suffix_folder_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = trace_zip(root / "arbitrary.payload")
            self.assertEqual(launcher.preflight_input(archive, root), archive.resolve())
            before = list(sys.path)
            self.assertEqual(launcher.resolve_original_input(ROOT, archive), archive.resolve())
            self.assertEqual(launcher.resolve_original_input(ROOT, root), archive.resolve())
            self.assertEqual(sys.path, before)
            self.assertNotIn("_factorized_launcher_data", sys.modules)
            original = root / "extracted" / "original"
            original.mkdir(parents=True)
            for name, content in original_members().items():
                path = original / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode() if isinstance(content, str) else content)
            self.assertEqual(launcher.resolve_original_input(ROOT, original.parent), original.resolve())
            self.assertFalse(any(original.rglob("*.pt")))
            with self.assertRaisesRegex(ValueError, "Ambiguous input"):
                launcher.resolve_original_input(ROOT, root)

    def test_report_zip_is_not_original_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "result.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("summary.json", '{"status":"complete","LLM_forwards":0}')
                zipped.writestr("report.json", "{}")
            with self.assertRaisesRegex(ValueError, "No ORIGINAL complete"):
                launcher.resolve_original_input(ROOT, archive)

    def test_preflight_missing_mount_lists_actual_dataset_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets/owner/actual_mount").mkdir(parents=True)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), self.assertRaises(FileNotFoundError):
                launcher.preflight_input(root / "missing", root)
            self.assertIn("actual_mount", captured.getvalue())
            self.assertIn("Requested RUN_DIR", captured.getvalue())

    def test_restore_wrapped_zip_and_folder_never_change_input_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = previous_result(root / "uploaded_result")
            archive = root / "anything.data"
            with zipfile.ZipFile(archive, "w") as zipped:
                for path in source.rglob("*"):
                    if path.is_file():
                        zipped.write(path, "wrapper/result/" + path.relative_to(source).as_posix())
            temp = root / "temp"
            temp.mkdir()
            output = root / "working/result"
            self.assertEqual(launcher.restore_resume_input(archive, output, temp), output.resolve())
            self.assertTrue((output / "models/latent64/stage_a.pt").is_file())
            self.assertTrue(archive.is_file())
            second = root / "working/second"
            (source / "embedding.pt").write_bytes(b"never copy embeddings into working output")
            self.assertEqual(launcher.restore_resume_input(source, second, temp), second.resolve())
            self.assertTrue((source / "models/latent64/stage_a.pt").is_file())
            self.assertFalse((second / "embedding.pt").exists())

    def test_resume_requires_unique_phase0_manifest_and_checkpoint_not_old_jobs_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = previous_result(root / "saved")
            self.assertEqual(launcher.locate_resume_root(root), saved.resolve())
            previous_result(root / "another")
            with self.assertRaisesRegex(ValueError, "one previous Phase 0"):
                launcher.locate_resume_root(root)
            (saved / "models/latent64/stage_a.pt").unlink()
            with self.assertRaisesRegex(ValueError, "checkpoints"):
                launcher.validate_resume_root(saved)
            old = root / "old"
            old.mkdir()
            (old / "config.json").write_text('{"num_questions":100,"hidden_dims":[8]}', encoding="utf-8")
            (old / "study_manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a latent WM Phase 0"):
                launcher.validate_resume_root(old)

    def test_resume_manifest_config_mismatch_and_empty_manifest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = previous_result(Path(directory) / "result")
            name = read_config().get("protocol", {}).get("resume", {}).get("runner_manifest", "study_manifest.json")
            manifest = root / name
            manifest.write_text('{"config":{"latent_dims":[1]}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config differs"):
                launcher.validate_resume_root(root)
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                launcher.validate_resume_root(root)

    def test_resume_checks_runner_fingerprint_source_status_and_real_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = previous_result(Path(directory) / "result")
            manifest = root / "study_manifest.json"
            valid = json.loads(manifest.read_text(encoding="utf-8"))
            for key, value, message in (("fingerprint", "bad", "SHA256"),
                                        ("source", "", "original trace source"),
                                        ("status", "unknown", "status must")):
                manifest.write_text(json.dumps(dict(valid, **{key: value})), encoding="utf-8")
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    launcher.validate_resume_root(root)
            manifest.write_text(json.dumps(valid), encoding="utf-8")
            (root / "models/latent64/stage_a.pt").unlink()
            (root / "preprocessing.pt").write_bytes(b"only preprocessing; no trained model")
            (root / "frozen_latents.pt").write_bytes(b"only cached latents; no trained model")
            with self.assertRaisesRegex(ValueError, "model checkpoints"):
                launcher.validate_resume_root(root)

    def test_resume_zip_validates_all_entries_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("../outside", "/absolute", "C:\\escape", "..\\escape", "x/../../bad", "a:stream")):
                archive = root / f"unsafe{index}.zip"
                destination = root / f"extract{index}"
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("safe.txt", "must not write")
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
                launcher.safe_extract(archive, root / "symlinks")
            duplicate = root / "duplicate.zip"
            with warnings.catch_warnings(), zipfile.ZipFile(duplicate, "w") as zipped:
                warnings.simplefilter("ignore", UserWarning)
                zipped.writestr("folder/file.txt", "first")
                zipped.writestr("folder\\file.txt", "second")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                launcher.safe_extract(duplicate, root / "duplicate")

    def test_fallback_archive_keeps_compact_checkpoints_and_excludes_raw_or_embedding_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = previous_result(root / "result")
            for name in ("input_cache/raw.pt", "experience/shard.npz", "raw_experiences/data.pt",
                         "hf_cache/model.safetensors", "embedding.pt", "other/raw.npz"):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"exclude")
            with (output / "oversized.bin").open("wb") as stream:
                stream.truncate(launcher.MAX_ARCHIVE_FILE_BYTES + 1)
            archive = launcher.archive_partial(output)
            self.assertEqual(archive.parent, root)
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn("models/latent64/stage_a.pt", names)
                self.assertIn("config.json", names)
                self.assertEqual(len(json.loads(zipped.read("launcher_packaging.json"))["excluded"]), 7)
                self.assertIsNone(zipped.testzip())
            before = archive.read_bytes()
            self.assertEqual(launcher.archive_partial(output), archive)
            self.assertEqual(archive.read_bytes(), before)

    def test_restored_fallback_metadata_has_no_duplicate_zip_members(self):
        with tempfile.TemporaryDirectory() as directory:
            output = previous_result(Path(directory) / "result")
            (output / "launcher_packaging.json").write_text('{"fallback":true,"excluded":[]}', encoding="utf-8")
            with zipfile.ZipFile(launcher.archive_partial(output)) as zipped:
                self.assertEqual(zipped.namelist().count("launcher_packaging.json"), 1)

    def test_corrupt_runner_zip_is_rebuilt_from_available_compact_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = previous_result(Path(directory) / "result")
            output.with_suffix(".zip").write_bytes(b"interrupted ZIP write")
            with zipfile.ZipFile(launcher.archive_partial(output)) as zipped:
                self.assertIn("models/latent64/stage_a.pt", zipped.namelist())
                self.assertIsNone(zipped.testzip())


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.working = self.root / "working"
        self.working.mkdir()
        self.input = trace_zip(self.root / "original_trace.data")
        self.sentinel = self.working / "keep.txt"
        self.sentinel.write_text("existing notebook output", encoding="utf-8")
        self.scope = {"RUN_DIR": str(self.input), "WORKING_DIR": str(self.working),
                      "TEMP_DIR": str(self.root / "temp"), "MODE": "smoke", "ALLOW_CPU": True}
        self.commands, self.checkouts, self.environments = [], [], []

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary.cleanup()

    def fake_download(self, repo, ref, env):
        self.assertEqual(Path.cwd(), self.working.resolve())
        self.assertTrue(repo.is_relative_to(self.root / "temp"))
        self.assertEqual(ref, self.scope.get("SOURCE_REF", launcher.DEFAULT_SOURCE_REF))
        self.checkouts.append(repo)
        self.environments.append(env)
        (repo / "configs").mkdir(parents=True)
        (repo / launcher.CONFIG_RELATIVE).write_text(json.dumps(read_config()), encoding="utf-8")
        for name in (*launcher.TEST_FILES, str(launcher.RUNNER_RELATIVE)):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# mock only; never execute training\n", encoding="utf-8")
        return "a" * 40

    def fake_run(self, command, *, cwd, env):
        command = [str(part) for part in command]
        self.commands.append(command)
        if any(part.endswith(str(launcher.RUNNER_RELATIVE)) for part in command):
            output = Path(command[command.index("--output") + 1])
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            previous_result(output, config)
            (output / "summary.json").write_text('{"status":"complete","LLM_forwards":0}', encoding="utf-8")

    def launch_mocked(self, scope=None, run=None, download=None, show=None):
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "select_devices", return_value=("cpu", "cpu")), \
                patch.object(launcher, "download_source", side_effect=download or self.fake_download), \
                patch.object(launcher, "resolve_original_input", return_value=self.input), \
                patch.object(launcher, "run_command", side_effect=run or self.fake_run), \
                patch.object(launcher, "show_archive", side_effect=show) as displayed, \
                patch("shutil.rmtree", side_effect=AssertionError("launcher deleted a directory")):
            output = launcher.launch(scope or self.scope)
            return output, displayed.call_args.args[0]

    def test_smoke_invokes_new_cli_checks_tests_and_uses_unique_cache_checkouts(self):
        output, archive = self.launch_mocked()
        config = json.loads((output / "launcher_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["num_questions"], 20)
        self.assertEqual(config["latent_dims"], [64])
        self.assertEqual(config["dynamics_updates_per_horizon"], 20)
        self.assertEqual((config["device_encoder"], config["device_dynamics"]), ("cpu", "cpu"))
        self.assertEqual(self.commands[0][-4:], list(launcher.DEPENDENCIES))
        self.assertNotIn("transformers", " ".join(self.commands[0]).lower())
        self.assertNotIn("torch", self.commands[0])
        self.assertEqual([Path(c[1]).name for c in self.commands[1:-1]],
                         [Path(name).name for name in launcher.TEST_FILES])
        runner = self.commands[-1]
        self.assertEqual(Path(runner[2]).name, "run_latent_wm_phase0.py")
        self.assertEqual(runner[runner.index("--input") + 1], str(self.input))
        self.assertEqual(runner[runner.index("--output") + 1], str(output))
        self.assertNotIn("--resume", runner)
        self.assertTrue(archive.is_file())
        self.assertEqual(archive.parent, self.working)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TMPDIR", "TORCH_HOME"):
            self.assertTrue(Path(self.environments[0][key]).is_relative_to(self.root / "temp"), key)
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "existing notebook output")
        second, _ = self.launch_mocked()
        self.assertNotEqual(output, second)
        self.assertNotEqual(self.checkouts[0], self.checkouts[1])

    def test_error_packages_partial_zip_before_reraising(self):
        shown = []
        def fail(command, *, cwd, env):
            self.fake_run(command, cwd=cwd, env=env)
            if any(str(part).endswith(str(launcher.RUNNER_RELATIVE)) for part in command):
                raise subprocess.CalledProcessError(7, command)
        def show(archive):
            self.assertTrue(archive.is_file())
            shown.append(archive)
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch_mocked(run=fail, show=show)
        self.assertEqual(shown, list(self.working.glob("*.zip")))
        with zipfile.ZipFile(shown[0]) as zipped:
            self.assertIn("CalledProcessError", zipped.read("launcher_error.txt").decode())
            self.assertIn("models/latent64/stage_a.pt", zipped.namelist())
        self.assertTrue(self.sentinel.is_file())

    def test_runner_stage_zip_is_primary_and_launcher_error_is_added(self):
        def fail_after_archive(command, *, cwd, env):
            self.fake_run(command, cwd=cwd, env=env)
            parts = [str(part) for part in command]
            if str(launcher.RUNNER_RELATIVE) not in [Path(part).name for part in parts]:
                return
            output = Path(parts[parts.index("--output") + 1])
            with zipfile.ZipFile(output.with_suffix(".zip"), "w") as zipped:
                zipped.writestr("runner_stage_report.json", '{"stage":"A"}')
            raise RuntimeError("failed after completed stage")
        with self.assertRaisesRegex(RuntimeError, "completed stage"):
            self.launch_mocked(run=fail_after_archive)
        with zipfile.ZipFile(next(self.working.glob("*.zip"))) as zipped:
            self.assertIn("runner_stage_report.json", zipped.namelist())
            self.assertIn("launcher_error.txt", zipped.namelist())
            self.assertNotIn("launcher_packaging.json", zipped.namelist())

    def test_missing_input_fails_before_download_and_still_has_zip(self):
        with patch.object(launcher, "on_kaggle", return_value=False), \
                patch.object(launcher, "download_source") as download, \
                patch.object(launcher, "run_command") as run, \
                patch.object(launcher, "show_archive") as displayed:
            with self.assertRaises(FileNotFoundError):
                launcher.launch(dict(self.scope, RUN_DIR=str(self.root / "missing")))
        download.assert_not_called()
        run.assert_not_called()
        self.assertEqual(displayed.call_args.args[0].parent, self.working)
        self.assertTrue(displayed.call_args.args[0].is_file())

    def test_missing_runner_test_is_reported_before_training(self):
        def missing_test(repo, ref, env):
            revision = self.fake_download(repo, ref, env)
            (repo / launcher.TEST_FILES[-1]).unlink()
            return revision
        with self.assertRaisesRegex(FileNotFoundError, "required Phase 0 test"):
            self.launch_mocked(download=missing_test)
        self.assertFalse(any(str(launcher.RUNNER_RELATIVE) in " ".join(command) for command in self.commands))
        self.assertTrue(next(self.working.glob("*.zip")).is_file())

    def test_resume_restores_checkpoint_passes_flag_and_keeps_original_trace_input(self):
        saved_config = launcher.build_config(read_config(), self.scope)
        saved_config["device_encoder"] = saved_config["device_dynamics"] = "cpu"
        saved = previous_result(self.root / "saved", saved_config)
        output, archive = self.launch_mocked(dict(self.scope, RESUME_INPUT=str(saved)))
        self.assertNotEqual(output, saved)
        self.assertTrue((output / "models/latent64/stage_a.pt").is_file())
        self.assertIn("--resume", self.commands[-1])
        self.assertEqual(self.commands[-1][self.commands[-1].index("--input") + 1], str(self.input))
        self.assertTrue(archive.is_file())

    def test_resume_config_mismatch_does_not_run_training(self):
        saved = previous_result(self.root / "saved", launcher.build_config(read_config(), {}))
        with self.assertRaisesRegex(ValueError, "effective config differs"):
            self.launch_mocked(dict(self.scope, RESUME_INPUT=str(saved)))
        self.assertFalse(any(str(launcher.RUNNER_RELATIVE) in " ".join(command) for command in self.commands))
        self.assertTrue(next(self.working.glob("*.zip")).is_file())


if __name__ == "__main__":
    unittest.main()
