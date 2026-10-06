"""Import-safe Kaggle sidecar for the projected32 paired native latent pilot.

Notebook parameters: RUN_DIR, PHASE0_INPUT, SOURCE_REF, MODE, NUM_QUESTIONS,
SMOKE_CONFIG, DEVICES, RESUME_INPUT. No study, GPU query or download at import.
"""
from __future__ import annotations

import copy
import importlib
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

try:
    import kaggle_behavior_aware_h1 as behavior
except ModuleNotFoundError as error:
    if error.name not in ("kaggle_behavior_aware_h1", "kaggle_latent_wm_phase0"):
        raise
    # Fresh raw-file imports remain usable for configuration inspection. Only
    # launch/main activation fetches a checkout and loads the sibling helpers.
    behavior = None

phase0 = behavior.phase0 if behavior is not None else None
_BOOTSTRAP_CHECKOUT = None
_BOOTSTRAP_REVISION = None
_BOOTSTRAP_SOURCE_REF = None
SOURCE_REPO = "https://github.com/HoangMaizzz/failfasttesting.git"
DEFAULT_RUN_DIR = "/kaggle/input/datasets/ainzkhail/2source2"
DEFAULT_PHASE0_INPUT = "/kaggle/input/datasets/ainzkhail/phase0"
DEFAULT_SOURCE_REF = "codex/paired-native-latent-feasibility"
DEFAULT_NUM_QUESTIONS = 100
NUM_QUESTIONS = globals().get("NUM_QUESTIONS", DEFAULT_NUM_QUESTIONS)
CONFIG_RELATIVE = Path("configs/paired_native_latent.json")
RUNNER_RELATIVE = Path("run_paired_native_latent.py")
TEST_FILES = ("tests/test_paired_native_launcher.py",)
DEPENDENCIES = ("numpy",)
DEFAULT_CONFIG = {
    "schema": "paired_native_latent_v1", "num_questions": DEFAULT_NUM_QUESTIONS,
    "latent_dim": 128, "seeds": [42, 43, 44], "workers": 2,
    "oracle_updates": 1000, "max_updates": 1500, "eval_every": 100,
    "batch_size": 32, "learning_rate": .0003, "bootstrap_samples": 2000,
    "encoder_verification_samples": 32, "rollout_existing_dynamics": True,
    "bridge_state_weight": 1., "bridge_behavior_weight": .5, "distill_weight": .3,
    "pipeline_check_only": False,
}
SMOKE_OVERRIDES = {
    "oracle_updates": 8, "max_updates": 12, "eval_every": 4,
    "seeds": [42], "bootstrap_samples": 100, "pipeline_check_only": True,
}
EXCLUDED_PARTS = (behavior.EXCLUDED_PARTS if behavior is not None else frozenset()) | frozenset({
    "inputsource", "input_source", "original", "originalraw", "original_raw",
    "original_trace", "traces", "source", "repo",
})
EXCLUDED_FILES = (behavior.EXCLUDED_FILES if behavior is not None else frozenset()) | frozenset({
    "states.jsonl", "edges.jsonl", "labels.jsonl", "teacher_targets.jsonl",
})
# Keep optimizer/RNG resume payloads even above Phase 0's 128 MiB limit.
MAX_ARCHIVE_FILE_BYTES = 2 * 1024 ** 3
MAX_NON_CHECKPOINT_FILE_BYTES = 128 * 1024 ** 2

# Only stable import-safe utilities are reused. Phase 0 selection loads
# behavior_aware_source.py from the newly downloaded checkout on every call.
def _missing_helper(*args, **kwargs):
    raise RuntimeError("Source helpers are deferred until launch() fetches the selected checkout")


def _bind_helpers(module):
    global behavior, phase0, EXCLUDED_PARTS, EXCLUDED_FILES
    behavior, phase0 = module, module.phase0
    EXCLUDED_PARTS |= module.EXCLUDED_PARTS
    EXCLUDED_FILES |= module.EXCLUDED_FILES
    for name in ("on_kaggle", "timestamp", "preflight_input", "download_source",
                 "resolve_original_input", "safe_extract", "run_command", "show_archive"):
        globals()[name] = getattr(phase0, name)
    for name in ("phase0_preflight", "resolve_phase0_input", "_read_object"):
        globals()[name] = getattr(behavior, name)


for _helper_name in ("on_kaggle", "timestamp", "preflight_input", "phase0_preflight",
                     "download_source", "resolve_original_input", "resolve_phase0_input",
                     "safe_extract", "run_command", "show_archive", "_read_object"):
    globals()[_helper_name] = _missing_helper
if behavior is not None:
    _bind_helpers(behavior)


def _bootstrap_checkout(scope):
    """Stdlib-only one-checkout bootstrap for a fresh raw-file notebook exec."""
    global _BOOTSTRAP_CHECKOUT, _BOOTSTRAP_REVISION, _BOOTSTRAP_SOURCE_REF
    ref = scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("SOURCE_REF must be a branch name or commit SHA")
    working = Path(scope.get("WORKING_DIR", "/kaggle/working")).resolve()
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    temp_root = Path(scope.get("TEMP_DIR", "/kaggle/temp")).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    repo = Path(tempfile.mkdtemp(prefix="paired_native_latent_", dir=temp_root)) / "repo"
    env = os.environ.copy()
    env.update(GIT_LFS_SKIP_SMUDGE="1", PYTHONDONTWRITEBYTECODE="1")
    for command in (["git", "init", str(repo)],
                    ["git", "-C", str(repo), "remote", "add", "origin", SOURCE_REPO],
                    ["git", "-C", str(repo), "fetch", "--depth", "1", "origin", ref],
                    ["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"]):
        subprocess.run(command, cwd=str(working), env=env, check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       cwd=str(working), env=env, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Source checkout did not return a valid git HEAD")
    _BOOTSTRAP_CHECKOUT, _BOOTSTRAP_REVISION, _BOOTSTRAP_SOURCE_REF = repo, revision, ref
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    return repo


def _ensure_helpers(scope):
    if behavior is None:
        _bootstrap_checkout(scope)
        _bind_helpers(importlib.import_module("kaggle_behavior_aware_h1"))


def build_config(base, scope):
    """Preserve source budgets; smoke changes budgets, never the question split."""
    if not isinstance(base, dict):
        raise ValueError("Paired native config must be a JSON object")
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    if (type(scope.get("NUM_QUESTIONS", DEFAULT_NUM_QUESTIONS)) is not int
            or scope.get("NUM_QUESTIONS", DEFAULT_NUM_QUESTIONS) != 100):
        raise ValueError("NUM_QUESTIONS is fixed at 100 in full and smoke modes")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(copy.deepcopy(base))
    if config["schema"] != "paired_native_latent_v1" or config["latent_dim"] != 128:
        raise ValueError("Requires paired_native_latent_v1 with latent_dim 128")
    if config["num_questions"] != 100:
        raise ValueError("Source config must retain num_questions=100")
    protocol = config.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    if (protocol.get("questions", 100) != 100
            or protocol.get("split", [70, 15, 15]) != [70, 15, 15]
            or protocol.get("split_fractions", [.7, .15, .15]) != [.7, .15, .15]):
        raise ValueError("Requires exact 100-question 70/15/15 split")
    if mode == "smoke":
        overrides = scope.get("SMOKE_CONFIG", {})
        if not isinstance(overrides, dict) or set(overrides) - set(SMOKE_OVERRIDES):
            raise ValueError("SMOKE_CONFIG may only change smoke budgets, seeds and pipeline_check_only")
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
        config.update(copy.deepcopy(overrides))
        # Every smoke result must retain its pipeline-only interpretation.
        config["pipeline_check_only"] = True
    # The runner owns budget/range validation and the actual frozen-source split.
    return config


def select_devices(scope, config):
    """Choose worker placement without importing torch or querying GPUs."""
    workers = config["workers"]
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    devices = scope.get("DEVICES", config.get("devices", ["cuda:0", "cuda:1"]))
    if isinstance(devices, str):
        devices = ["cpu"] * workers if devices == "cpu" else devices.split(",")
    if not isinstance(devices, (list, tuple)) or not devices:
        raise ValueError("DEVICES must be 'cpu' or a list/comma-separated worker devices")
    if any(device not in ("cpu", "cuda:0", "cuda:1") for device in devices):
        raise ValueError("DEVICES supports cpu, cuda:0 and cuda:1 only")
    if list(devices) == ["cpu"]:
        devices = ["cpu"] * workers
    if len(devices) < workers:
        raise ValueError("DEVICES needs one device per configured worker")
    return tuple(devices[:workers])


def _compact_member(name, size):
    portable = PurePosixPath(name.replace("\\", "/"))
    checkpoint = portable.name in ("best.pt", "last.pt") and "jobs" in portable.parts
    limit = MAX_ARCHIVE_FILE_BYTES if checkpoint else MAX_NON_CHECKPOINT_FILE_BYTES
    return (not portable.is_absolute() and not PureWindowsPath(name).drive
            and ".." not in portable.parts and ":" not in name and "\x00" not in name
            and not any(part.lower() in EXCLUDED_PARTS for part in portable.parts)
            and portable.name.lower() not in EXCLUDED_FILES
            and portable.suffix.lower() not in (".npz", ".zip", ".tmp")
            and size <= limit)


def is_compact_file(path, root):
    return (not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
            and _compact_member(path.relative_to(root).as_posix(), path.stat().st_size)
            # PyTorch checkpoints are ZIP containers too. Other nested ZIPs,
            # including original traces with arbitrary extensions, are inputs.
            and (path.suffix.lower() in (".pt", ".pth") or not zipfile.is_zipfile(path)))


def archive_partial(output):
    """Atomically merge current compact results with any earlier runner ZIP.

    Strip the runner's exact output-folder prefix before merging root-relative
    results. Existing stage reports survive failure. Input assets are filtered
    from both sources, and the previous ZIP survives a packaging failure.
    """
    output = Path(output)
    archive = output.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    files = {path.relative_to(output).as_posix(): path for path in sorted(output.rglob("*"))
             if path.is_file() and is_compact_file(path, output)}
    files.pop("launcher_packaging.json", None)
    excluded = [path.relative_to(output).as_posix() for path in sorted(output.rglob("*"))
                if path.is_file() and not is_compact_file(path, output)]
    previous = None
    if archive.is_file():
        try:
            previous = zipfile.ZipFile(archive)
            if previous.testzip():
                raise zipfile.BadZipFile("Runner ZIP failed CRC check")
        except (zipfile.BadZipFile, OSError, RuntimeError) as error:
            if previous is not None:
                previous.close()
            previous = None
            print("Rebuilding unreadable runner ZIP:", error, flush=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            seen = set()
            if previous is not None:
                runner_prefix = output.name + "/"
                for member in previous.infolist():
                    if member.is_dir():
                        continue
                    original_name = member.orig_filename.replace("\\", "/")
                    if (stat.S_ISLNK(member.external_attr >> 16)
                            or not _compact_member(original_name, member.file_size)):
                        excluded.append(original_name)
                        continue
                    name = (original_name[len(runner_prefix):]
                            if original_name.startswith(runner_prefix) else original_name)
                    # Validate again: stripping can expose an absolute path,
                    # for example output.name + '//escape.json'.
                    if not _compact_member(name, member.file_size):
                        excluded.append(original_name)
                        continue
                    if name == "launcher_packaging.json" or name in files or name in seen:
                        continue
                    if PurePosixPath(name).suffix.lower() not in (".pt", ".pth"):
                        with previous.open(member) as source:
                            if source.read(4) in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
                                excluded.append(name)
                                continue
                    with previous.open(member) as source, zipped.open(name, "w", force_zip64=True) as target:
                        shutil.copyfileobj(source, target)
                    seen.add(name)
            for name, path in files.items():
                zipped.write(path, name)
            zipped.writestr("launcher_packaging.json", json.dumps({
                "excluded": sorted(set(excluded)), "max_file_bytes": MAX_ARCHIVE_FILE_BYTES,
                "max_non_checkpoint_file_bytes": MAX_NON_CHECKPOINT_FILE_BYTES,
            }, indent=2))
    finally:
        if previous is not None:
            previous.close()
    temporary.replace(archive)
    return archive


def validate_resume_root(root):
    root = Path(root).resolve()
    config = _read_object(root / "config.json")
    if config.get("schema") != "paired_native_latent_v1" or config.get("latent_dim") != 128:
        raise ValueError("Resume config is not a paired_native_latent_v1 study")
    manifest = _read_object(root / "study_manifest.json")
    if "config" in manifest and manifest["config"] != config:
        raise ValueError("Resume manifest config differs from config.json")
    signature = manifest.get("signature", manifest.get("fingerprint"))
    if not ((isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature))
            or (isinstance(signature, dict) and signature)):
        raise ValueError("Resume study_manifest.json requires its exact source/config/input signature")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Resume results may not contain symlinks")
    jobs = root / "jobs"
    checkpoints = [path for path in jobs.rglob("*") if path.is_file()
                   and path.name in ("best.pt", "last.pt") and path.stat().st_size > 0
                   and is_compact_file(path, root)]
    if not checkpoints:
        raise ValueError("Resume root requires jobs checkpoints (best.pt or last.pt)")
    # The runner checks signature equality and optimizer/RNG checkpoint contents.
    return root


def locate_resume_root(container):
    container = Path(container).resolve()
    roots, problems = [], []
    candidates = dict.fromkeys([container] + [p.parent for p in container.rglob("config.json")])
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        try:
            roots.append(validate_resume_root(candidate))
        except ValueError as error:
            problems.append(str(error))
    if len(roots) != 1:
        raise ValueError(f"Need one previous paired native result root; found {len(roots)}. {'; '.join(problems[:3])}")
    return roots[0]


def restore_resume_input(path, output, temp):
    path, output, temp = Path(path).resolve(), Path(output).resolve(), Path(temp)
    if not path.exists():
        raise FileNotFoundError(f"RESUME_INPUT is not mounted: {path}")
    if path.is_file():
        source = locate_resume_root(safe_extract(path, temp / "resume_input"))
    else:
        try:
            source = locate_resume_root(path)
        except ValueError:
            archives = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink() and zipfile.is_zipfile(p)]
            if len(archives) != 1:
                raise ValueError("Set RESUME_INPUT to one previous result ZIP or extracted folder")
            source = locate_resume_root(safe_extract(archives[0], temp / "resume_input"))
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("New output and RESUME_INPUT must be separate folders")
    for item in source.rglob("*"):
        if item.is_file() and is_compact_file(item, source):
            target = output / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    return validate_resume_root(output)


def discover_test_files(repo):
    repo = Path(repo)
    if any(not (repo / name).is_file() for name in TEST_FILES):
        raise FileNotFoundError("Selected ref is missing tests/test_paired_native_launcher.py")
    tests = set(repo / name for name in TEST_FILES)
    tests.update((repo / "tests").glob("test_paired_latent*.py"))
    return tuple(path.relative_to(repo).as_posix() for path in sorted(tests))


def launch(scope):
    _ensure_helpers(scope)
    kaggle = on_kaggle()
    working = (Path("/kaggle/working") if kaggle else Path(scope.get("WORKING_DIR", Path.cwd()))).resolve()
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)  # A notebook may start inside a read-only input mount.
    mode = scope.get("MODE", "full")
    safe_mode = mode if mode in ("full", "smoke") else "invalid"
    output = Path(tempfile.mkdtemp(prefix=f"paired_native_latent_{safe_mode}_{timestamp()}_", dir=working))
    try:
        if mode not in ("full", "smoke"):
            raise ValueError("MODE must be 'full' or 'smoke'")
        original_mount = preflight_input(scope.get("RUN_DIR", DEFAULT_RUN_DIR))
        phase0_mount = phase0_preflight(scope.get("PHASE0_INPUT", DEFAULT_PHASE0_INPUT))
        temp_root = Path("/kaggle/temp") if kaggle else Path(scope.get("TEMP_DIR", tempfile.gettempdir()))
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = (_BOOTSTRAP_CHECKOUT.parent if _BOOTSTRAP_CHECKOUT is not None
                else Path(tempfile.mkdtemp(prefix="paired_native_latent_", dir=temp_root)))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", GIT_LFS_SKIP_SMUDGE="1",
                   WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(temp / "hf_cache"), HF_HUB_CACHE=str(temp / "hf_cache/hub"),
                   HUGGINGFACE_HUB_CACHE=str(temp / "hf_cache/hub"),
                   XDG_CACHE_HOME=str(temp / "cache"), TORCH_HOME=str(temp / "torch_cache"),
                   TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp))
        ref = scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)
        if _BOOTSTRAP_CHECKOUT is not None:
            if ref != _BOOTSTRAP_SOURCE_REF:
                raise ValueError("SOURCE_REF differs from the bootstrapped checkout")
            repo, revision = _BOOTSTRAP_CHECKOUT, _BOOTSTRAP_REVISION
        else:
            repo = temp / "repo"
            revision = download_source(repo, ref, env)
        # Record provenance immediately, including failures in input/config checks.
        metadata = {"source_ref": ref, "source_revision": revision, "mode": mode,
                    "LLM_downloads": 0, "LLM_forwards": 0, "projected_dim": 32,
                    "temporary_root": str(temp)}
        metadata_path = output / "launcher_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for relative in (*TEST_FILES, str(CONFIG_RELATIVE), str(RUNNER_RELATIVE)):
            if not (repo / relative).is_file():
                raise FileNotFoundError(f"Selected source ref is missing {relative}")
        test_files = discover_test_files(repo)
        run_command([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *DEPENDENCIES], cwd=repo, env=env)
        original = Path(resolve_original_input(repo, original_mount)).resolve()
        try:
            selected = Path(resolve_phase0_input(repo, phase0_mount)).resolve()
        except ValueError as error:
            raise ValueError("PHASE0_INPUT needs the original full100 pretrained latent128 Phase 0 "
                             "result (approximately 176 MB), not the latest behavior-aware ZIP. "
                             f"Reader detail: {error}") from error
        base = json.loads((repo / CONFIG_RELATIVE).read_text(encoding="utf-8"))
        config = build_config(base, scope)
        devices = select_devices(scope, config)
        config["devices"] = list(devices)
        resume = bool(scope.get("RESUME_INPUT"))
        if resume:
            restore_resume_input(scope["RESUME_INPUT"], output, temp)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if _read_object(output / "config.json") != config:
                raise ValueError("Resume effective config differs; use the original MODE, devices and source ref")
        external_config = output / "launcher_config.json"
        external_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        metadata.update(input=str(original), phase0_input=str(phase0_mount), phase0_selection=str(selected),
                        resume=resume, worker_devices=list(devices), test_files=list(test_files))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Source commit:", revision, flush=True)
        print("Paired native projected32 pilot:", mode, "100q / exact 70/15/15 / latent128;",
              "workers=", config["workers"], "devices=", devices, "seeds=", config["seeds"], flush=True)
        print("Original trace:", original, "Phase 0:", selected, flush=True)
        print("Saved projected Qwen teacher + STOP embedding; LLM downloads/forwards: 0.", flush=True)
        for relative in test_files:
            run_command([sys.executable, repo / relative], cwd=repo, env=env)
        command = [sys.executable, "-u", repo / RUNNER_RELATIVE,
                   "--input", original, "--phase0_input", selected,
                   "--output", output, "--config", external_config]
        if resume:
            command.append("--resume")
        run_command(command, cwd=repo, env=env)
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
