"""Standalone, import-safe Kaggle launcher for the native Qwen verifier study.

Only launch() fetches source, installs packages, queries GPUs or starts work.
The runner owns model identification, capture/signature validation and results.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import zipfile

SOURCE_REPO = "https://github.com/HoangMaizzz/failfasttesting.git"
DEFAULT_SOURCE_REF = "codex/native-qwen-verifier-latent"
DEFAULT_RUN_DIR = "/kaggle/input/datasets/ainzkhail/2source2"
DEFAULT_PHASE0_INPUT = "/kaggle/input/datasets/ainzkhail/phase0"
CONFIG_RELATIVE = Path("configs/native_qwen_verifier.json")
RUNNER_RELATIVE = Path("run_native_qwen_verifier.py")
SCHEMA = "native_qwen_verifier_latent_v1"
DEPENDENCIES = ("transformers==4.53.1", "accelerate>=1,<2", "huggingface_hub<1",
                "safetensors", "numpy", "matplotlib")
FULL_CONTRACT = {
    "num_questions": 100, "probe_updates": 1000, "latent_updates": 1000,
    "selected_layers": 2, "latent_dims": [64, 128, 256], "seeds": [42, 43, 44],
    "benchmark_repetitions": 100, "benchmark_warmup": 10,
    "package_raw_hidden": True, "max_reproduction_mismatch_rate": 0.0,
    "bootstrap_samples": 2000, "pipeline_check_only": False,
}
SMOKE_OVERRIDES = {
    "capture_states_per_question": 2, "probe_updates": 8, "latent_updates": 8,
    "eval_every": 4, "seeds": [42], "benchmark_repetitions": 3,
    "benchmark_warmup": 1, "pipeline_check_only": True,
}
OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
EXCLUDED_PARTS = frozenset({
    ".git", "__pycache__", "repo", "source", "input", "inputsource", "input_source",
    "input_assets", "source_assets", "original", "original_raw", "original_trace",
    "experience", "experiences", "raw", "raw_experiences", "traces", "cache",
    "_cache", "input_cache", "raw_cache", "hf", "hf_cache", "hf_home", "hub",
    "model_cache", "model_weights", "models", "torch_cache", ".cache", "hf_assets",
    "datasets_cache", "phase0_input",
    "phase0_source", "phase0_assets", "frozen_cache", "latent_cache", "tmp", "temp",
})
EXCLUDED_FILES = frozenset({
    "states.jsonl", "edges.jsonl", "labels.jsonl", "teacher_targets.jsonl",
    "preprocessing.pt", "frozen_latents.pt", "pytorch_model.bin", "model.safetensors",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
})
_BOOTSTRAP_CHECKOUT = None
_BOOTSTRAP_REVISION = None
_BOOTSTRAP_SOURCE_REF = None


def run_command(command, *, cwd, env):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def build_environment(temp):
    """Keep downloads and all default temporary/cache storage out of Output."""
    temp = Path(temp).resolve()
    env = os.environ.copy()
    for key in OFFLINE_FLAGS:
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", GIT_LFS_SKIP_SMUDGE="1",
               USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_TELEMETRY="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               WANDB_MODE="disabled", HF_HOME=str(temp / "hf_cache"),
               HF_HUB_CACHE=str(temp / "hf_cache/hub"),
               HUGGINGFACE_HUB_CACHE=str(temp / "hf_cache/hub"),
               HF_ASSETS_CACHE=str(temp / "hf_cache/assets"),
               HF_XET_CACHE=str(temp / "hf_cache/xet"),
               TRANSFORMERS_CACHE=str(temp / "hf_cache/hub"),
               HF_DATASETS_CACHE=str(temp / "datasets_cache"),
               XDG_CACHE_HOME=str(temp / "cache"), TORCH_HOME=str(temp / "torch_cache"),
               TORCH_EXTENSIONS_DIR=str(temp / "torch_extensions"),
               TRITON_CACHE_DIR=str(temp / "triton_cache"), CUDA_CACHE_PATH=str(temp / "cuda_cache"),
               MPLCONFIGDIR=str(temp / "matplotlib"), TMPDIR=str(temp),
               TMP=str(temp), TEMP=str(temp))
    return env


@contextmanager
def helper_environment(env):
    # In-process content resolvers also use tempfile and may import torch.
    keys = set(OFFLINE_FLAGS) | {key for key in env if key in (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE", "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME", "MPLCONFIGDIR",
        "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "TMPDIR", "TEMP", "TMP", "PYTHONDONTWRITEBYTECODE")}
    previous = {key: os.environ.get(key) for key in keys}
    previous_temp = tempfile.tempdir
    try:
        for key in keys:
            if key in env:
                os.environ[key] = env[key]
            else:
                os.environ.pop(key, None)
        tempfile.tempdir = env["TMPDIR"]
        yield
    finally:
        tempfile.tempdir = previous_temp
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _bootstrap_checkout(scope, temp, working, env):
    """Fetch one branch/SHA checkout; reuse it without deleting any repository."""
    global _BOOTSTRAP_CHECKOUT, _BOOTSTRAP_REVISION, _BOOTSTRAP_SOURCE_REF
    ref = scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("SOURCE_REF must be a branch name or commit SHA")
    if _BOOTSTRAP_CHECKOUT is not None:
        if ref != _BOOTSTRAP_SOURCE_REF:
            raise ValueError("SOURCE_REF differs from this launcher's existing checkout; reload the launcher")
        repo = _BOOTSTRAP_CHECKOUT
    else:
        repo = Path(temp) / "repo"
        for command in (
            ["git", "init", repo],
            ["git", "-C", repo, "remote", "add", "origin", SOURCE_REPO],
            ["git", "-C", repo, "fetch", "--depth", "1", "origin", ref],
            ["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"],
        ):
            run_command(command, cwd=working, env=env)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       cwd=str(working), env=env, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Source checkout did not return a valid Git HEAD")
    if _BOOTSTRAP_REVISION is not None and revision != _BOOTSTRAP_REVISION:
        raise ValueError("Existing source checkout HEAD changed")
    _BOOTSTRAP_CHECKOUT, _BOOTSTRAP_REVISION, _BOOTSTRAP_SOURCE_REF = repo, revision, ref
    return repo, revision


def _load_helpers(repo):
    """Load existing content resolvers from the selected checkout only."""
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, Path(repo) / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    phase0 = load("_native_qwen_phase0_helpers", "kaggle_latent_wm_phase0.py")
    previous = sys.modules.get("kaggle_latent_wm_phase0")
    sys.modules["kaggle_latent_wm_phase0"] = phase0
    try:
        behavior = load("_native_qwen_behavior_helpers", "kaggle_behavior_aware_h1.py")
    finally:
        if previous is None:
            sys.modules.pop("kaggle_latent_wm_phase0", None)
        else:
            sys.modules["kaggle_latent_wm_phase0"] = previous
    return phase0, behavior


def build_config(base, scope):
    if not isinstance(base, dict):
        raise ValueError("Native Qwen config must be a JSON object")
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    if type(scope.get("NUM_QUESTIONS", 100)) is not int or scope.get("NUM_QUESTIONS", 100) != 100:
        raise ValueError("Both modes require all 100 questions")
    config = copy.deepcopy(base)
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Requires {SCHEMA} source configuration")
    for key, value in FULL_CONTRACT.items():
        if key in config and config[key] != value:
            raise ValueError(f"Source full config differs from required {key}={value!r}")
        config[key] = copy.deepcopy(value)
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    for key, expected in (("questions", 100), ("split", [70, 15, 15]),
                          ("split_counts", [70, 15, 15]), ("split_fractions", [.7, .15, .15])):
        if key in protocol and protocol[key] != expected:
            raise ValueError("Requires the existing exact 100-question 70/15/15 split")
    if config.get("capture_states_per_question") not in (None, 0):
        raise ValueError("Full source config must capture all eligible states")
    if mode == "smoke":
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
    return config


def assert_two_gpus(env, working):
    """Run only during full launch; inspect GPU count without loading an LLM."""
    code = ("import torch; n=torch.cuda.device_count(); "
            "assert n >= 2, f'Full native Qwen study requires 2 GPUs; found {n}'; "
            "print('Visible GPUs:', n)")
    run_command([sys.executable, "-c", code], cwd=working, env=env)


def discover_test_files(repo):
    files = sorted((Path(repo) / "tests").glob("test_native_qwen_*.py"))
    if Path(repo) / "tests/test_native_qwen_launcher.py" not in files:
        raise FileNotFoundError("Selected ref is missing tests/test_native_qwen_launcher.py")
    return tuple(path.relative_to(repo).as_posix() for path in files)


def _safe_member(name):
    portable = PurePosixPath(name.replace("\\", "/"))
    return (bool(name) and not portable.is_absolute() and not PureWindowsPath(name).drive
            and ".." not in portable.parts and ":" not in name and "\x00" not in name)


def _result_member(name):
    portable = PurePosixPath(name.replace("\\", "/"))
    return (_safe_member(name)
            and not any(part.lower() in EXCLUDED_PARTS for part in portable.parts)
            and portable.name.lower() not in EXCLUDED_FILES
            and not re.fullmatch(r"(?:model|pytorch_model)-\d+-of-\d+\.(?:bin|safetensors)",
                                 portable.name.lower())
            and portable.suffix.lower() not in (".zip", ".npz", ".tmp", ".pyc"))


def is_result_file(path, root):
    return (not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
            and _result_member(path.relative_to(root).as_posix())
            and (path.suffix.lower() in (".pt", ".pth") or not zipfile.is_zipfile(path)))


def archive_partial(output, force=False):
    """Prefer the runner ZIP; atomically repair/merge fallback results if needed.

    No size cutoff: raw native_hidden/*.npy, frozen candidate embeddings and
    experiment checkpoints are intentional study outputs.
    """
    output = Path(output)
    archive = output.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    files = {path.relative_to(output).as_posix(): path for path in sorted(output.rglob("*"))
             if path.is_file() and is_result_file(path, output)}
    files.pop("launcher_packaging.json", None)
    excluded = [path.relative_to(output).as_posix() for path in sorted(output.rglob("*"))
                if path.is_file() and not is_result_file(path, output)]
    previous = None
    members = []
    needs_rebuild = True
    if archive.is_file():
        try:
            previous = zipfile.ZipFile(archive)
            if previous.testzip():
                raise zipfile.BadZipFile("Runner ZIP failed CRC check")
            seen = set()
            needs_rebuild = False
            prefix = output.name + "/"
            for member in previous.infolist():
                if member.is_dir():
                    if not _safe_member(member.orig_filename):
                        excluded.append(member.orig_filename)
                        needs_rebuild = True
                    continue
                original = member.orig_filename.replace("\\", "/")
                name = original[len(prefix):] if original.startswith(prefix) else original
                valid = (_safe_member(original) and _result_member(name)
                         and not stat.S_ISLNK(member.external_attr >> 16)
                         and not member.flag_bits & 1)
                if valid and PurePosixPath(name).suffix.lower() not in (".pt", ".pth"):
                    with previous.open(member) as source:
                        valid = source.read(4) not in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
                if not valid:
                    excluded.append(original)
                    needs_rebuild = True
                    continue
                if original != name or name in seen:
                    needs_rebuild = True
                seen.add(name)
                members.append((name, member))
            if set(files) - seen:
                needs_rebuild = True
            # A failure diagnostic must replace a restored older diagnostic.
            if "launcher_error.txt" in files:
                needs_rebuild = True
            if not needs_rebuild:
                if not force:
                    previous.close()
                    return archive
        except (zipfile.BadZipFile, OSError, RuntimeError) as error:
            if previous is not None:
                previous.close()
            previous, members = None, []
            print("Rebuilding unreadable runner ZIP:", error, flush=True)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
            seen = set()
            if previous is not None:
                for name, member in members:
                    if name in files or name in seen or name == "launcher_packaging.json":
                        continue
                    with previous.open(member) as source, zipped.open(name, "w", force_zip64=True) as target:
                        shutil.copyfileobj(source, target)
                    seen.add(name)
            for name, path in files.items():
                zipped.write(path, name)
            zipped.writestr("launcher_packaging.json", json.dumps({
                "fallback": True, "excluded": sorted(set(excluded)),
                "raw_native_hidden_retained": True, "file_size_cutoff": None,
            }, indent=2))
    finally:
        if previous is not None:
            previous.close()
    temporary.replace(archive)
    return archive


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Requires a nonempty JSON object: {path}")
    return value


def locate_resume_root(container):
    container = Path(container).resolve()
    roots = set()
    for candidate in [container] + [path.parent for path in container.rglob("config.json")]:
        if (candidate / "config.json").is_file() and (candidate / "study_manifest.json").is_file():
            if _read_object(candidate / "config.json").get("schema") != SCHEMA:
                continue
            manifest = _read_object(candidate / "study_manifest.json")
            signature = manifest.get("signature", manifest.get("fingerprint"))
            if ((isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature))
                    or (isinstance(signature, dict) and signature)):
                roots.add(candidate)
    if len(roots) != 1:
        raise ValueError(f"Requires one native result root with exact study signature; found {len(roots)}")
    root = roots.pop()
    if not (root / "native_hidden").is_dir():
        raise ValueError("Resume requires the native_hidden capture; use CAPTURE_INPUT for a separate capture")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Resume input may not contain symlinks")
    return root


def restore_resume_input(path, output, temp, helpers, config, revision):
    """Copy into new Output; runner verifies model+source+config before reuse."""
    path, output = Path(path).resolve(), Path(output).resolve()
    if not path.exists():
        raise FileNotFoundError(f"RESUME_INPUT is not mounted: {path}")
    if path.is_file():
        source = locate_resume_root(helpers.safe_extract(path, Path(temp) / "resume_input"))
    else:
        try:
            source = locate_resume_root(path)
        except ValueError:
            archives = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
                        and zipfile.is_zipfile(p) and p.suffix.lower() not in (".pt", ".pth")]
            if len(archives) != 1:
                raise ValueError("Set RESUME_INPUT to one native result ZIP or unpacked result")
            source = locate_resume_root(helpers.safe_extract(archives[0], Path(temp) / "resume_input"))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("New Output and mounted resume input must be separate folders")
    if _read_object(source / "config.json") != config:
        raise ValueError("Resume effective config differs; retain the original mode and source configuration")
    prior_metadata = source / "launcher_metadata.json"
    if prior_metadata.is_file() and _read_object(prior_metadata).get("source_revision") != revision:
        raise ValueError("Resume source Git revision differs; pin SOURCE_REF to the original SHA")
    for item in source.rglob("*"):
        if item.is_file() and is_result_file(item, source):
            target = output / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    return source


def replay_input(path):
    """Pass an explicit capture mount unchanged; the runner validates contents."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CAPTURE_INPUT is not mounted: {path}")
    if path.is_symlink() or (path.is_file() and not zipfile.is_zipfile(path)):
        raise ValueError("CAPTURE_INPUT must be a native capture/result ZIP or unpacked folder")
    return path


def show_archive(archive):
    print("DOWNLOAD ZIP (working root):", Path(archive).name, flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(Path(archive).name))
    except ImportError:
        pass


def launch_directories(scope):
    """Explicit test/embedding paths override environment-derived defaults.

    On Kaggle, normal callers supply neither override and retain the working
    root and temporary cache. Unit tests must remain isolated even when the
    real Kaggle mounts exist or KAGGLE_KERNEL_RUN_TYPE is inherited.
    """
    kaggle = Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    working = Path(scope.get("WORKING_DIR", "/kaggle/working" if kaggle else Path.cwd())).resolve()
    temporary = Path(scope.get("TEMP_DIR", "/kaggle/temp" if kaggle else tempfile.gettempdir())).resolve()
    return working, temporary


def launch(scope):
    working, temp_root = launch_directories(scope)
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    output = Path(tempfile.mkdtemp(prefix=f"native_qwen_verifier_{stamp}_", dir=working))
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = (_BOOTSTRAP_CHECKOUT.parent if _BOOTSTRAP_CHECKOUT is not None
                else Path(tempfile.mkdtemp(prefix="native_qwen_verifier_", dir=temp_root)))
        env = build_environment(temp)
        repo, revision = _bootstrap_checkout(scope, temp, working, env)
        metadata = dict(source_ref=scope.get("SOURCE_REF", DEFAULT_SOURCE_REF),
                        source_revision=revision, temporary_root=str(temp),
                        model_identification="runner reads the actual original input summary",
                        mode=scope.get("MODE", "full"))
        metadata_path = output / "launcher_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for relative in (CONFIG_RELATIVE, RUNNER_RELATIVE):
            if not (repo / relative).is_file():
                raise FileNotFoundError(f"Selected source ref is missing {relative}")
        tests = discover_test_files(repo)
        config = build_config(_read_object(repo / CONFIG_RELATIVE), scope)
        run_command([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *DEPENDENCIES],
                    cwd=working, env=env)
        # All focused tests run before input readers, GPU checks and the heavy runner.
        run_command([sys.executable, "-m", "unittest", "discover", "-s", str(repo / "tests"),
                     "-p", "test_native_qwen_*.py"], cwd=working, env=env)
        if scope.get("MODE", "full") == "full":
            assert_two_gpus(env, working)
        with helper_environment(env):
            phase0, behavior = _load_helpers(repo)
            original = Path(phase0.resolve_original_input(repo, phase0.preflight_input(
                scope.get("RUN_DIR", DEFAULT_RUN_DIR)))).resolve()
            selected = Path(behavior.resolve_phase0_input(repo, behavior.phase0_preflight(
                scope.get("PHASE0_INPUT", DEFAULT_PHASE0_INPUT)))).resolve()
            resume = bool(scope.get("RESUME_INPUT"))
            if resume:
                restore_resume_input(scope["RESUME_INPUT"], output, temp, phase0, config, revision)
        capture = replay_input(scope["CAPTURE_INPUT"]) if scope.get("CAPTURE_INPUT") else None
        external_config = output / "launcher_config.json"
        external_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        metadata.update(input=str(original), phase0_input=str(selected), resume=resume,
                        capture_input=str(capture) if capture else None, test_files=list(tests),
                        pipeline_check_only=config["pipeline_check_only"])
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Native Qwen verifier:", metadata["mode"], "100 questions / 70/15/15; source", revision,
              "pipeline only:", config["pipeline_check_only"], flush=True)
        command = [sys.executable, "-u", repo / RUNNER_RELATIVE, "--input", original,
                   "--phase0_input", selected, "--output", output, "--config", external_config]
        if resume:
            command.append("--resume")
        if capture is not None:
            command.extend(["--capture_input", capture])
        run_command(command, cwd=working, env=env)
        show_archive(archive_partial(output))
        return output
    except BaseException as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        (output / "launcher_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            show_archive(archive_partial(output))
        except Exception as packaging_error:
            print("Could not package partial results; existing ZIP is preserved:", packaging_error, flush=True)
        raise


if __name__ == "__main__":
    OUTPUT = launch(globals())
