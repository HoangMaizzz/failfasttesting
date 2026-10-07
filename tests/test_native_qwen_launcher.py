"""Launcher contracts without downloads, model loading, GPU queries or training."""
import builtins
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_native_qwen_verifier as launcher


def write(path, data=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def document(path, value):
    return write(path, json.dumps(value).encode())


def config_fixture():
    return dict(copy.deepcopy(launcher.FULL_CONTRACT), schema="native_qwen_verifier_latent_v1",
                protocol={"questions": 100, "split": [70, 15, 15]},
                capture_states_per_question=None, eval_every=100)


def prior_result(root, config=None, revision="a" * 40):
    document(root / "config.json", config or config_fixture())
    document(root / "study_manifest.json", {"signature": "c" * 64})
    document(root / "launcher_metadata.json", {"source_revision": revision})
    write(root / "native_hidden/layer25/hidden.npy")
    write(root / "native_hidden/candidate_embeddings.npy")
    write(root / "jobs/raw_seed42/best.pt")
    write(root / "jobs/raw_seed42/last.pt")
    write(root / "hf_cache/model.safetensors")
    return root


def zip_tree(root, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in root.rglob("*"):
            if path.is_file():
                zipped.write(path, "arbitrary/wrapper/" + path.relative_to(root).as_posix())
    return archive


def phase0_fixture(root):
    document(root / "config.json", {"num_questions": 100, "latent_dims": [64, 128],
                                     "token_embedding": "pretrained"})
    document(root / "summary.json", {"schema": "latent_world_model_phase0_v1", "status": "complete"})
    document(root / "split_manifest.json", {"train": list(range(70)), "val": list(range(70, 85)),
                                             "test": list(range(85, 100))})
    document(root / "study_manifest.json", {"fingerprint": "a" * 64})
    document(root / "source_hashes.json", {"source": "b" * 64})
    document(root / "latent_128/stage_A/complete.json", {"status": "complete"})
    for name in ("preprocessing.pt", "latent_128/stage_A/best.pt", "latent_128/frozen_latents.pt"):
        write(root / name)
    return root


def phase0_reader_stubs():
    modules = {}
    for name, exports in {
        "torch": (),
        "phase0_wm_data": ("load_dataset", "dataset_digest", "NativeTargets", "prepare_rows", "pack_observations"),
        "phase0_wm_models": ("TokenEmbedding", "LatentEncoder"),
        "run_latent_wm_phase0": ("select_paths", "cpu_weights"),
    }.items():
        module = ModuleType(name)
        for export in exports:
            setattr(module, export, lambda *a, **k: (_ for _ in ()).throw(AssertionError("tensor work")))
        modules[name] = module
    return modules


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_cwd = Path.cwd()
        self.bootstrap = patch.multiple(launcher, _BOOTSTRAP_CHECKOUT=None,
                                        _BOOTSTRAP_REVISION=None, _BOOTSTRAP_SOURCE_REF=None)
        self.bootstrap.start()
        # Simulated failure cases intentionally print ERROR diagnostics; keep
        # them out of the real Kaggle launch log so they cannot look like a
        # live fetch/model failure. unittest failures still use stderr.
        self.captured_stdout = io.StringIO()
        self.stdout_capture = redirect_stdout(self.captured_stdout)
        self.stdout_capture.__enter__()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.bootstrap.stop()
        self.stdout_capture.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_explicit_directories_override_real_kaggle_mounts(self):
        scope = dict(WORKING_DIR=self.root / 'isolated_working', TEMP_DIR=self.root / 'isolated_temp')
        with patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': 'Batch'}), \
                patch.object(Path, 'exists', return_value=True):
            working, temporary = launcher.launch_directories(scope)
        self.assertEqual(working, scope['WORKING_DIR'].resolve())
        self.assertEqual(temporary, scope['TEMP_DIR'].resolve())
        self.assertFalse(working.exists())
        self.assertFalse(temporary.exists())

    def test_normal_kaggle_launch_keeps_production_defaults(self):
        for detected_by_mount in (False, True):
            with self.subTest(mount=detected_by_mount), \
                    patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': '' if detected_by_mount else 'Batch'}), \
                    patch.object(Path, 'exists', return_value=detected_by_mount):
                working, temporary = launcher.launch_directories({})
            self.assertEqual(working, Path('/kaggle/working').resolve())
            self.assertEqual(temporary, Path('/kaggle/temp').resolve())

    def test_ordinary_import_is_standalone_and_has_no_side_effects(self):
        spec = importlib.util.spec_from_file_location("_native_launcher_import", ROOT / "kaggle_native_qwen_verifier.py")
        module = importlib.util.module_from_spec(spec)
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "transformers", "kaggle_behavior_aware_h1",
                                       "kaggle_latent_wm_phase0", "run_native_qwen_verifier"):
                raise AssertionError("heavy/sibling import: " + name)
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded), \
                patch("os.chdir", side_effect=AssertionError("cwd mutation")), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("directory mutation")), \
                patch("subprocess.run", side_effect=AssertionError("process")), \
                patch("subprocess.check_output", side_effect=AssertionError("fetch")):
            spec.loader.exec_module(module)
        self.assertIsNone(module._BOOTSTRAP_CHECKOUT)
        self.assertTrue(callable(module.launch))

    def test_full_and_smoke_keep_population_split_and_source_config(self):
        base = config_fixture()
        original = copy.deepcopy(base)
        full = launcher.build_config(base, {})
        self.assertEqual(full, base)
        smoke = launcher.build_config(base, {"MODE": "smoke"})
        for name, value in launcher.SMOKE_OVERRIDES.items():
            self.assertEqual(smoke[name], value)
        for name in ("num_questions", "selected_layers", "latent_dims", "bootstrap_samples",
                     "max_reproduction_mismatch_rate", "package_raw_hidden", "protocol"):
            self.assertEqual(smoke[name], base[name])
        smoke["protocol"]["split"].append(999)
        self.assertEqual(base, original)
        self.assertNotIn("model_id", full)
        self.assertNotIn("revision", full)

    def test_actual_main_config_matches_launcher_contract(self):
        # The launcher remains testable while parallel work has not written it.
        path = ROOT / launcher.CONFIG_RELATIVE
        if not path.exists():
            self.skipTest("main-owned config has not arrived")
        base = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(launcher.build_config(base, {}), base)
        self.assertTrue(launcher.build_config(base, {"MODE": "smoke"})["pipeline_check_only"])

    def test_invalid_mode_budget_split_or_reproduction_override_fails(self):
        for scope in ({"MODE": "typo"}, {"NUM_QUESTIONS": 8}, {"NUM_QUESTIONS": 100.0}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                launcher.build_config(config_fixture(), scope)
        for key, value in (("num_questions", 99), ("probe_updates", 8),
                           ("schema", "paired_native_latent_v1"),
                           ("latent_dims", [128]), ("seeds", [42]),
                           ("max_reproduction_mismatch_rate", .01),
                           ("package_raw_hidden", False), ("capture_states_per_question", 2),
                           ("protocol", {"split": [80, 10, 10]})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.build_config(dict(config_fixture(), **{key: value}), {})

    def test_dependency_contract_does_not_request_torch(self):
        self.assertIn("transformers==4.53.1", launcher.DEPENDENCIES)
        self.assertIn("accelerate>=1,<2", launcher.DEPENDENCIES)
        self.assertIn("huggingface_hub<1", launcher.DEPENDENCIES)
        self.assertFalse(any("torch" in dependency.lower() for dependency in launcher.DEPENDENCIES))

    def test_readme_cell_uses_one_url_and_passes_explicit_replay_globals(self):
        readme = (ROOT / "NATIVE_QWEN_VERIFIER.md").read_text(encoding="utf-8")
        cell = readme.split("```python\n", 1)[1].split("```", 1)[0]
        scope = dict(SOURCE_REF="a" * 40, MODE="smoke", CAPTURE_INPUT="/mounted/native-capture",
                     RESUME_INPUT="/mounted/prior-results")
        payload = b"OUTPUT = 'fake-output'\n"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as download:
            exec(compile(cell, "README copy cell", "exec"), scope)
        download.assert_called_once_with(
            "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/" + "a" * 40
            + "/kaggle_native_qwen_verifier.py", timeout=120)
        self.assertEqual(scope["OUTPUT"], "fake-output")
        self.assertEqual(scope["launcher"]["__name__"], "__main__")
        self.assertEqual(scope["launcher"]["CAPTURE_INPUT"], scope["CAPTURE_INPUT"])
        self.assertEqual(scope["launcher"]["RESUME_INPUT"], scope["RESUME_INPUT"])
        self.assertEqual(scope["launcher"]["RUN_DIR"], launcher.DEFAULT_RUN_DIR)
        self.assertEqual(scope["launcher"]["PHASE0_INPUT"], launcher.DEFAULT_PHASE0_INPUT)

    def test_online_flags_and_all_cache_paths_are_temporary(self):
        with patch.dict(os.environ, {name: "1" for name in launcher.OFFLINE_FLAGS}):
            env = launcher.build_environment(self.root)
            self.assertFalse(set(launcher.OFFLINE_FLAGS) & set(env))
            for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
                         "HF_DATASETS_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE", "XDG_CACHE_HOME",
                         "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                         "MPLCONFIGDIR", "TMP", "TEMP", "TMPDIR"):
                self.assertTrue(Path(env[name]).is_relative_to(self.root))
            previous_temp = tempfile.tempdir
            with launcher.helper_environment(env):
                self.assertFalse(set(launcher.OFFLINE_FLAGS) & set(os.environ))
                self.assertEqual(tempfile.gettempdir(), str(self.root))
            self.assertEqual(tempfile.tempdir, previous_temp)
            self.assertTrue(all(os.environ[name] == "1" for name in launcher.OFFLINE_FLAGS))

    def test_one_checkout_supports_branch_sha_and_reuses_clone(self):
        for ref in (launcher.DEFAULT_SOURCE_REF, "a" * 40):
            with self.subTest(ref=ref), patch.multiple(launcher, _BOOTSTRAP_CHECKOUT=None,
                    _BOOTSTRAP_REVISION=None, _BOOTSTRAP_SOURCE_REF=None), \
                    patch.object(launcher, "run_command") as commands, \
                    patch("subprocess.check_output", return_value="a" * 40 + "\n"):
                scope = {"SOURCE_REF": ref}
                result = launcher._bootstrap_checkout(scope, self.root, self.root, {})
                second = launcher._bootstrap_checkout(scope, self.root, self.root, {})
                self.assertEqual(result, second)
                self.assertEqual(commands.call_count, 4)
                self.assertEqual(result[0], self.root / "repo")
                self.assertEqual(commands.call_args_list[2].args[0][-1], ref)
                for call in commands.call_args_list:
                    self.assertEqual(call.kwargs["cwd"], self.root)
                with self.assertRaises(ValueError):
                    launcher._bootstrap_checkout({"SOURCE_REF": "other"}, self.root, self.root, {})

    def test_checkout_rejects_bad_ref_or_changed_head(self):
        with self.assertRaises(ValueError):
            launcher._bootstrap_checkout({"SOURCE_REF": "--help"}, self.root, self.root, {})
        with patch.multiple(launcher, _BOOTSTRAP_CHECKOUT=self.root / "repo",
                            _BOOTSTRAP_REVISION="a" * 40, _BOOTSTRAP_SOURCE_REF="branch"), \
                patch("subprocess.check_output", return_value="b" * 40):
            with self.assertRaises(ValueError):
                launcher._bootstrap_checkout({"SOURCE_REF": "branch"}, self.root, self.root, {})

    def test_discovers_all_native_tests(self):
        for name in ("launcher", "capture", "runner", "metrics", "data"):
            write(self.root / f"tests/test_native_qwen_{name}.py")
        write(self.root / "tests/test_paired_native_launcher.py")
        tests = launcher.discover_test_files(self.root)
        self.assertEqual(len(tests), 5)
        self.assertTrue(all("test_native_qwen_" in name for name in tests))

    def test_full_gpu_check_only_queries_count(self):
        with patch.object(launcher, "run_command") as command:
            launcher.assert_two_gpus({}, self.root)
        script = command.call_args.args[0][-1]
        self.assertIn("device_count", script)
        self.assertIn("n >= 2", script)
        self.assertNotIn("from_pretrained", script)

    def test_fallback_keeps_hidden_embedding_and_checkpoints_filters_inputs(self):
        output = self.root / "result"
        retained = ("config.json", "native_hidden/layer25/h.npy",
                    "native_hidden/candidate_embeddings.npy", "frozen_candidate_embeddings.pt",
                    "jobs/seed42/best.pt", "jobs/seed42/last.pt", "plots/example.png")
        excluded = ("raw/states.jsonl", "input/arbitrary.dat", "hf_cache/model.safetensors",
                    "model_weights/weight.bin", "model-00001-of-00004.safetensors",
                    "pytorch_model.bin", "input_cache/experience.npz", "nested.zip")
        for name in retained + excluded:
            write(output / name)
        archive = launcher.archive_partial(output)
        self.assertEqual(archive.parent, output.parent)
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            self.assertTrue(set(retained) <= names)
            self.assertFalse(set(excluded) & names)
            self.assertIsNone(json.loads(zipped.read("launcher_packaging.json"))["file_size_cutoff"])

    def test_root_runner_zip_is_primary(self):
        output = self.root / "result"
        write(output / "config.json", b"current")
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("config.json", b"current")
            zipped.writestr("runner_report.json", b"saved")
        before = archive.read_bytes()
        self.assertEqual(launcher.archive_partial(output), archive)
        self.assertEqual(archive.read_bytes(), before)

    def test_prefixed_runner_merge_normalizes_only_exact_output_name(self):
        output = self.root / "result"
        write(output / "config.json", b"new")
        write(output / "jobs/seed42/last.pt", b"new checkpoint")
        with zipfile.ZipFile(output.with_suffix(".zip"), "w") as zipped:
            zipped.writestr("result/config.json", b"old")
            zipped.writestr("result/jobs/seed42/last.pt", b"old checkpoint")
            zipped.writestr("result/native_hidden/layer25/h.npy", b"hidden")
            zipped.writestr("unrelated/result/early_report.json", b"report")
            zipped.writestr("result/hf_cache/model.safetensors", b"weights")
            zipped.writestr("result//escape.json", b"bad")
            zipped.writestr("../unsafe.json", b"bad")
        with zipfile.ZipFile(launcher.archive_partial(output)) as zipped:
            names = zipped.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(zipped.read("config.json"), b"new")
            self.assertEqual(zipped.read("jobs/seed42/last.pt"), b"new checkpoint")
            self.assertEqual(zipped.read("native_hidden/layer25/h.npy"), b"hidden")
            self.assertIn("unrelated/result/early_report.json", names)
            self.assertFalse(any(name.startswith("result/") for name in names))
            self.assertNotIn("escape.json", names)
            self.assertFalse(any("model.safetensors" in name for name in names))

    def test_nested_input_zip_and_archived_symlinks_are_filtered(self):
        output = self.root / "result"
        output.mkdir()
        nested = self.root / "nested.zip"
        with zipfile.ZipFile(nested, "w") as zipped:
            zipped.writestr("input.txt", b"input")
        write(output / "trace.dat", nested.read_bytes())
        write(output / "jobs/last.pt", nested.read_bytes())  # Torch uses ZIP serialization.
        with zipfile.ZipFile(output.with_suffix(".zip"), "w") as zipped:
            symlink = zipfile.ZipInfo("link.json")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(symlink, "outside")
            zipped.writestr("arbitrary.dat", nested.read_bytes())
        with zipfile.ZipFile(launcher.archive_partial(output)) as zipped:
            self.assertIn("jobs/last.pt", zipped.namelist())
            self.assertNotIn("trace.dat", zipped.namelist())
            self.assertNotIn("arbitrary.dat", zipped.namelist())
            self.assertNotIn("link.json", zipped.namelist())

    def test_failed_atomic_replacement_preserves_previous_zip(self):
        output = self.root / "result"
        write(output / "launcher_error.txt", b"failure")
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("earlier_report.json", b"report")
        before = archive.read_bytes()
        with patch("pathlib.Path.replace", side_effect=OSError("full disk")):
            with self.assertRaises(OSError):
                launcher.archive_partial(output)
        self.assertEqual(archive.read_bytes(), before)

    def test_resume_uses_new_output_leaves_old_mount_untouched(self):
        source = prior_result(self.root / "mounted/arbitrary/wrapper")
        before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        output = self.root / "new_output"
        selected = launcher.restore_resume_input(source.parent.parent, output, self.root,
                    SimpleNamespace(), config_fixture(), "a" * 40)
        self.assertEqual(selected, source)
        self.assertEqual((output / "jobs/raw_seed42/last.pt").read_bytes(), b"fixture")
        self.assertTrue((output / "native_hidden/layer25/hidden.npy").is_file())
        self.assertFalse((output / "hf_cache").exists())
        self.assertEqual(before, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})

    def test_arbitrarily_named_wrapped_resume_zip_is_restored(self):
        source = prior_result(self.root / "source")
        archive = zip_tree(source, self.root / "mount/random-upload")
        original_archive = archive.read_bytes()
        phase0, _ = launcher._load_helpers(ROOT)
        output = self.root / "output"
        launcher.restore_resume_input(archive.parent, output, self.root, phase0, config_fixture(), "a" * 40)
        self.assertEqual(archive.read_bytes(), original_archive)
        self.assertTrue((output / "native_hidden/candidate_embeddings.npy").is_file())
        self.assertTrue((output / "jobs/raw_seed42/last.pt").is_file())
        self.assertFalse((output / "hf_cache/model.safetensors").exists())

    def test_unsafe_resume_zip_is_rejected_before_extraction(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../outside.txt", b"unsafe")
        phase0, _ = launcher._load_helpers(ROOT)
        with self.assertRaises(ValueError):
            launcher.restore_resume_input(archive, self.root / "output", self.root,
                                          phase0, config_fixture(), "a" * 40)
        self.assertFalse((self.root / "outside.txt").exists())

    def test_existing_content_resolvers_support_zip_and_extracted_inputs(self):
        original = self.root / "raw_source"
        document(original / "summary.json", {"schema": "interactive_acceptance_two_source_v1",
                                             "status": "complete", "questions_completed": 100})
        for name in ("states.jsonl", "edges.jsonl", "labels.jsonl", "teacher_targets.jsonl"):
            write(original / name, b"")
        write(original / "experience/something.npz")
        original_zip = zip_tree(original, self.root / "trace_mount/unpredictable-upload")
        phase0_source = phase0_fixture(self.root / "phase0_source")
        phase0_zip = zip_tree(phase0_source, self.root / "phase0_mount/some-other-upload")
        phase0, behavior = launcher._load_helpers(ROOT)
        self.assertEqual(phase0.resolve_original_input(ROOT, original), original)
        self.assertEqual(phase0.resolve_original_input(ROOT, original_zip.parent), original_zip)
        with patch.dict(sys.modules, phase0_reader_stubs()):
            self.assertEqual(behavior.resolve_phase0_input(ROOT, phase0_source), phase0_source)
            self.assertEqual(behavior.resolve_phase0_input(ROOT, phase0_zip.parent), phase0_zip)

    def test_resume_checks_config_and_git_revision_before_copy(self):
        source = prior_result(self.root / "mounted")
        output = self.root / "new_output"
        for config, revision in ((launcher.build_config(config_fixture(), {"MODE": "smoke"}), "a" * 40),
                                 (config_fixture(), "b" * 40)):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                launcher.restore_resume_input(source, output, self.root, SimpleNamespace(), config, revision)
            self.assertFalse(output.exists())

    def test_replay_is_explicit_pass_through_without_offline_inference(self):
        folder = self.root / "capture"
        folder.mkdir()
        archive = self.root / "arbitrary-upload"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("capture_audit.json", "{}")
        self.assertEqual(launcher.replay_input(folder), folder)
        self.assertEqual(launcher.replay_input(archive), archive)
        with self.assertRaises(FileNotFoundError):
            launcher.replay_input(self.root / "missing")
        with self.assertRaises(ValueError):
            launcher.replay_input(write(self.root / "not_zip"))

    def test_launch_cli_cwd_online_env_and_tests_before_model_work(self):
        repo = self.root / "repo"
        document(repo / launcher.CONFIG_RELATIVE, config_fixture())
        write(repo / launcher.RUNNER_RELATIVE)
        write(repo / "tests/test_native_qwen_launcher.py")
        write(repo / "tests/test_native_qwen_capture.py")
        original, phase0, capture = (self.root / name for name in ("original", "phase0", "capture"))
        for path in (original, phase0, capture):
            path.mkdir()
        events = []
        def command(argv, *, cwd, env):
            self.assertEqual(Path(cwd), self.root / "working")
            self.assertFalse(set(launcher.OFFLINE_FLAGS) & set(env))
            events.append([str(part) for part in argv])
        def helpers(_):
            self.assertTrue(any("unittest" in event for event in events))
            events.append(["helpers"])
            return (SimpleNamespace(preflight_input=lambda p: p, resolve_original_input=lambda r, p: p),
                    SimpleNamespace(phase0_preflight=lambda p: p, resolve_phase0_input=lambda r, p: p))
        with patch.object(launcher, "_bootstrap_checkout", return_value=(repo, "a" * 40)), \
                patch.object(launcher, "_load_helpers", side_effect=helpers), \
                patch.object(launcher, "run_command", side_effect=command), \
                patch.object(launcher, "show_archive"), \
                patch.object(launcher, "assert_two_gpus") as gpu, \
                patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch", "HF_HUB_OFFLINE": "1"}):
            output = launcher.launch(dict(MODE="smoke", WORKING_DIR=self.root / "working",
                TEMP_DIR=self.root / "temp", RUN_DIR=original, PHASE0_INPUT=phase0, CAPTURE_INPUT=capture))
        gpu.assert_not_called()
        runner = events[-1]
        self.assertEqual(runner[2], str(repo / launcher.RUNNER_RELATIVE))
        for flag, value in (("--input", original), ("--phase0_input", phase0), ("--output", output),
                            ("--config", output / "launcher_config.json"), ("--capture_input", capture)):
            self.assertEqual(runner[runner.index(flag) + 1], str(value))
        self.assertNotIn("--resume", runner)
        self.assertTrue(json.loads((output / "launcher_config.json").read_text())["pipeline_check_only"])
        self.assertEqual(Path.cwd(), self.root / "working")

    def test_failure_packages_diagnostic_and_reraises(self):
        with patch.object(launcher, "_bootstrap_checkout", side_effect=RuntimeError("fetch failed")), \
                patch.object(launcher, "show_archive"), \
                patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            with self.assertRaisesRegex(RuntimeError, "fetch failed"):
                launcher.launch(dict(WORKING_DIR=self.root / "working", TEMP_DIR=self.root / "temp"))
        archives = list((self.root / "working").glob("*.zip"))
        self.assertEqual(len(archives), 1)
        with zipfile.ZipFile(archives[0]) as zipped:
            self.assertIn(b"fetch failed", zipped.read("launcher_error.txt"))

    def test_full_launch_resume_runs_gpu_check_and_preserves_runner_failure_artifacts(self):
        repo = self.root / "repo"
        document(repo / launcher.CONFIG_RELATIVE, config_fixture())
        write(repo / launcher.RUNNER_RELATIVE)
        write(repo / "tests/test_native_qwen_launcher.py")
        source = prior_result(self.root / "mounted")
        inputs = self.root / "input"
        inputs.mkdir()
        events = []
        def command(argv, *, cwd, env):
            argv = [str(part) for part in argv]
            self.assertEqual(Path(cwd), self.root / "working")
            events.append(argv)
            if str(repo / launcher.RUNNER_RELATIVE) in argv:
                self.assertIn("--resume", argv)
                output = Path(argv[argv.index("--output") + 1])
                self.assertTrue((output / "jobs/raw_seed42/last.pt").exists())
                with zipfile.ZipFile(output.with_suffix(".zip"), "w") as zipped:
                    zipped.writestr(output.name + "/early_runner_report.json", "saved")
                    zipped.writestr(output.name + "/native_hidden/layer100/hidden.npy", "hidden")
                raise RuntimeError("runner failed")
        helpers = (SimpleNamespace(preflight_input=lambda p: p, resolve_original_input=lambda r, p: p),
                   SimpleNamespace(phase0_preflight=lambda p: p, resolve_phase0_input=lambda r, p: p))
        with patch.object(launcher, "_bootstrap_checkout", return_value=(repo, "a" * 40)), \
                patch.object(launcher, "_load_helpers", return_value=helpers), \
                patch.object(launcher, "run_command", side_effect=command), \
                patch.object(launcher, "assert_two_gpus", side_effect=lambda *a: events.append(["gpu"])) as gpu, \
                patch.object(launcher, "show_archive"), \
                patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                launcher.launch(dict(WORKING_DIR=self.root / "working", TEMP_DIR=self.root / "temp",
                                     RUN_DIR=inputs, PHASE0_INPUT=inputs, RESUME_INPUT=source))
        gpu.assert_called_once()
        self.assertLess(next(i for i, event in enumerate(events) if "unittest" in event), events.index(["gpu"]))
        archive, = (self.root / "working").glob("*.zip")
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(zipped.read("early_runner_report.json"), b"saved")
            self.assertEqual(zipped.read("native_hidden/layer100/hidden.npy"), b"hidden")
            self.assertIn(b"runner failed", zipped.read("launcher_error.txt"))
            self.assertEqual(len(zipped.namelist()), len(set(zipped.namelist())))

    def test_test_failure_stops_before_gpu_checks_or_input_readers(self):
        repo = self.root / "repo"
        document(repo / launcher.CONFIG_RELATIVE, config_fixture())
        write(repo / launcher.RUNNER_RELATIVE)
        write(repo / "tests/test_native_qwen_launcher.py")
        def command(argv, **kwargs):
            if "unittest" in argv:
                raise RuntimeError("tests failed")
        with patch.object(launcher, "_bootstrap_checkout", return_value=(repo, "a" * 40)), \
                patch.object(launcher, "run_command", side_effect=command), \
                patch.object(launcher, "_load_helpers") as helpers, \
                patch.object(launcher, "assert_two_gpus") as gpu, \
                patch.object(launcher, "show_archive"), \
                patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            with self.assertRaisesRegex(RuntimeError, "tests failed"):
                launcher.launch(dict(WORKING_DIR=self.root / "working", TEMP_DIR=self.root / "temp"))
        helpers.assert_not_called()
        gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
