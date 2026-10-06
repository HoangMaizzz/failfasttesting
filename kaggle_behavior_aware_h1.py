"""Import-safe launcher for paired behavior-aware H1 latent dynamics.

Notebook inputs: RUN_DIR, PHASE0_INPUT, SOURCE_REF, MODE, RESUME_INPUT.
PyTorch is preinstalled; this launcher never downloads a language model.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback
from types import FunctionType
import zipfile

import kaggle_latent_wm_phase0 as phase0


DEFAULT_RUN_DIR = "/kaggle/input/datasets/ainzkhail/2source2"
DEFAULT_PHASE0_INPUT = "/kaggle/input"
DEFAULT_SOURCE_REF = "codex/behavior-aware-latent-dynamics"
CONFIG_RELATIVE = Path("configs/behavior_aware_h1.json")
RUNNER_RELATIVE = Path("run_behavior_aware_h1.py")
DEPENDENCIES = ("numpy",)
TEST_FILES = ("tests/test_behavior_aware_launcher.py",)
LAMBDA_GRID = [[.1, .1], [.3, .1], [.1, .3], [.3, .3], [1., .3], [.3, 1.]]
FULL_CONTRACT = {
    "latent_dim": 128, "workers": 2, "seeds": [42, 43, 44],
    "max_updates": 2400, "pilot_updates": 1200, "eval_every": 100,
    "batch_size": 16, "lambda_grid": LAMBDA_GRID, "bootstrap_samples": 2000,
}
SMOKE_OVERRIDES = {
    "max_updates": 8, "pilot_updates": 4, "eval_every": 4,
    "seeds": [42], "lambda_grid": LAMBDA_GRID[:2],
}
EXCLUDED_PARTS = phase0.EXCLUDED_PARTS | frozenset({
    "_cache", "phase0_input", "phase0_source", "phase0_assets", "native_assets",
    "frozen_cache", "latent_cache", "input_assets", "source_assets",
})
EXCLUDED_FILES = phase0.EXCLUDED_FILES | frozenset({
    "preprocessing.pt", "frozen_latents.pt", "native_targets.pt",
    "projected_embeddings.pt", "phase0.zip",
})
# Worker optimizer/RNG checkpoints must survive fallback packaging. Unlike
# Phase 0's 128 MiB cutoff, this permits a single result file up to 2 GiB.
MAX_ARCHIVE_FILE_BYTES = 2 * 1024 ** 3

# Pure/import-safe helpers only; never call the Phase 0 experiment launcher.
on_kaggle = phase0.on_kaggle
timestamp = phase0.timestamp
preflight_input = phase0.preflight_input
select_devices = phase0.select_devices
download_source = phase0.download_source
resolve_original_input = phase0.resolve_original_input
run_command = phase0.run_command
safe_extract = phase0.safe_extract
show_archive = phase0.show_archive


def build_config(base, scope):
    if not isinstance(base, dict):
        raise ValueError("Behavior-aware config must be a JSON object")
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    mismatches = [key for key, value in FULL_CONTRACT.items() if base.get(key) != value]
    if mismatches:
        raise ValueError(f"Source JSON differs from the full behavior-aware contract: {mismatches}")
    if "num_questions" in base and base["num_questions"] != 100:
        raise ValueError("Behavior-aware inputs must retain the full 100-question split")
    config = copy.deepcopy(base)
    if mode == "smoke":
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
        config['pipeline_check_only'] = True
    # GPU worker placement is selected by the runner, not injected into the
    # study config or the frozen Phase 0 source signature.
    return config


def resolve_phase0_input(repo, path):
    """Delegate content/schema selection to the NEW downloaded source reader."""
    module_path = Path(repo) / "behavior_aware_source.py"
    if not module_path.is_file():
        raise FileNotFoundError("Selected source ref is missing behavior_aware_source.py")
    name = "_behavior_aware_launcher_source"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    old_module = sys.modules.get(name)
    old_path = list(sys.path)
    sys.modules[name] = module
    sys.path.insert(0, str(repo))
    try:
        spec.loader.exec_module(module)
        result = module.resolve_phase0_input(path)
        if result is None:
            raise ValueError("Phase 0 source reader returned no matching result")
        return result
    finally:
        sys.path[:] = old_path
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module


def phase0_preflight(path):
    path = Path(path)
    print("Requested PHASE0_INPUT:", path, flush=True)
    if not path.exists():
        raise FileNotFoundError(
            f"PHASE0_INPUT is not mounted: {path}. Upload the new 176 MB full100 "
            "pretrained Phase 0 result ZIP or use Add Notebook Output.")
    if path.is_file() and not zipfile.is_zipfile(path):
        raise ValueError("PHASE0_INPUT must be a Phase 0 result ZIP or extracted folder")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"Unsupported PHASE0_INPUT: {path}")
    return path.resolve()


def _read_object(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Resume root requires {path.name}: {path.parent}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Resume {path.name} must contain a nonempty JSON object")
    return value


def is_compact_file(path, root):
    relative = path.relative_to(root)
    return (not path.is_symlink()
            and not any(part.lower() in EXCLUDED_PARTS for part in relative.parts)
            and path.name.lower() not in EXCLUDED_FILES
            and path.suffix.lower() not in (".npz", ".zip", ".tmp")
            and path.stat().st_size <= MAX_ARCHIVE_FILE_BYTES)


def archive_partial(output):
    """Reuse Phase 0's atomic fallback with behavior-study exclusions.

    Give the function private globals so this study does not change the shared
    Phase 0 helper's packaging rules or truncate larger optimizer checkpoints.
    """
    namespace = dict(phase0.archive_partial.__globals__)
    namespace.update(is_compact_file=is_compact_file, MAX_ARCHIVE_FILE_BYTES=MAX_ARCHIVE_FILE_BYTES)
    package = FunctionType(phase0.archive_partial.__code__, namespace,
                           name="behavior_aware_archive_partial")
    return package(Path(output))


def validate_resume_root(root):
    root = Path(root).resolve()
    config = _read_object(root / "config.json")
    if (config.get("latent_dim") != 128 or "max_updates" not in config
            or "pilot_updates" not in config or not isinstance(config.get("lambda_grid"), list)):
        raise ValueError("Resume config is not a behavior-aware H1 study")
    names = ("study_manifest.json", "behavior_aware_manifest.json", "manifest.json")
    manifests = [root / name for name in names if (root / name).is_file()]
    if len(manifests) != 1:
        raise ValueError("Resume root requires one behavior-aware runner manifest")
    manifest = _read_object(manifests[0])
    if "config" in manifest and manifest["config"] != config:
        raise ValueError("Resume runner manifest config differs from config.json")
    signature = manifest.get("fingerprint", manifest.get("signature"))
    if not ((isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature))
            or (isinstance(signature, dict) and signature)):
        raise ValueError("Resume runner manifest requires its source/config/input signature")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Resume results may not contain symlinks")
    checkpoints = [path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in (".pt", ".pth")
                   and path.stat().st_size > 0 and is_compact_file(path, root)]
    if not checkpoints:
        raise ValueError("Resume root requires saved worker model/optimizer checkpoints")
    # The runner validates signature contents and optimizer/RNG payloads.
    return root


def locate_resume_root(container):
    container = Path(container).resolve()
    candidates = list(dict.fromkeys([container] + [path.parent for path in container.rglob("config.json")]))
    roots, problems = [], []
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        try:
            roots.append(validate_resume_root(candidate))
        except ValueError as error:
            problems.append(str(error))
    if len(roots) != 1:
        raise ValueError(f"Need one previous behavior-aware result root; found {len(roots)}. {'; '.join(problems[:3])}")
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
            archives = [item for item in path.rglob("*") if item.is_file()
                        and not item.is_symlink() and zipfile.is_zipfile(item)]
            if len(archives) != 1:
                raise ValueError("Set RESUME_INPUT to one prior behavior-aware ZIP or result folder")
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
    if any(not (Path(repo) / name).is_file() for name in TEST_FILES):
        raise FileNotFoundError("Selected ref is missing tests/test_behavior_aware_launcher.py")
    return tuple(path.relative_to(repo).as_posix()
                 for path in sorted((Path(repo) / "tests").glob("test_behavior_aware_*.py")))


def launch(scope):
    kaggle = on_kaggle()
    working = (Path("/kaggle/working") if kaggle
               else Path(scope.get("WORKING_DIR", Path.cwd()))).resolve()
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    mode = scope.get("MODE", "full")
    safe_mode = mode if mode in ("full", "smoke") else "invalid"
    output = Path(tempfile.mkdtemp(prefix=f"behavior_aware_h1_{safe_mode}_{timestamp()}_", dir=working))
    try:
        if mode not in ("full", "smoke"):
            raise ValueError("MODE must be 'full' or 'smoke'")
        original_mount = preflight_input(scope.get("RUN_DIR", DEFAULT_RUN_DIR))
        phase0_mount = phase0_preflight(scope.get("PHASE0_INPUT", DEFAULT_PHASE0_INPUT))
        devices = select_devices(scope, kaggle)
        temp_root = Path("/kaggle/temp") if kaggle else Path(scope.get("TEMP_DIR", tempfile.gettempdir()))
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="behavior_aware_h1_", dir=temp_root))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", GIT_LFS_SKIP_SMUDGE="1",
                   WANDB_MODE="disabled", TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp),
                   XDG_CACHE_HOME=str(temp / "cache"), TORCH_HOME=str(temp / "torch_cache"))
        repo = temp / "repo"
        ref = scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)
        revision = download_source(repo, ref, env)
        run_command([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *DEPENDENCIES],
                    cwd=repo, env=env)
        original = resolve_original_input(repo, original_mount)
        try:
            selected = resolve_phase0_input(repo, phase0_mount)
        except ValueError as error:
            raise ValueError(
                "Need one matching full100 pretrained latent128 Phase 0 result. Upload the "
                "new 176 MB ZIP or use Add Notebook Output and set PHASE0_INPUT if ambiguous. "
                f"Reader detail: {error}") from error
        selected = Path(selected).resolve()
        print("Resolved Phase 0 source:", selected, flush=True)
        base = json.loads((repo / CONFIG_RELATIVE).read_text(encoding="utf-8"))
        config = build_config(base, scope)
        resume = bool(scope.get("RESUME_INPUT"))
        if resume:
            restore_resume_input(scope["RESUME_INPUT"], output, temp)
            if _read_object(output / "config.json") != config:
                raise ValueError("Resume effective config differs; use the original MODE and source ref")
        external_config = output / "launcher_config.json"
        external_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        test_files = discover_test_files(repo)
        (output / "launcher_metadata.json").write_text(json.dumps({
            "source_ref": ref, "source_revision": revision, "input": str(original),
            "phase0_input": str(phase0_mount), "phase0_selection": str(selected),
            "mode": mode, "resume": resume, "worker_devices": list(devices),
            "temporary_root": str(temp), "test_files": list(test_files),
            "LLM_forwards": 0, "LLM_downloads": 0, "encoder_retraining": False,
            "frozen_latent_dim": 128,
        }, indent=2), encoding="utf-8")
        print("Study:", mode, "fixed full100 / latent128; workers=", config["workers"],
              "devices=", devices, "seeds=", config["seeds"], flush=True)
        print("Saved traces and frozen Phase 0 only; no LLM forwards/downloads or encoder retraining.", flush=True)
        for relative in test_files:
            run_command([sys.executable, repo / relative], cwd=repo, env=env)
        if not (repo / RUNNER_RELATIVE).is_file():
            raise FileNotFoundError(f"Selected ref is missing {RUNNER_RELATIVE}")
        command = [sys.executable, "-u", repo / RUNNER_RELATIVE,
                   "--input", original, "--phase0_input", selected,
                   "--output", output, "--config", external_config]
        if resume:
            command.append("--resume")
        run_command(command, cwd=repo, env=env)
        show_archive(archive_partial(output))
        return output
    except BaseException as error:
        diagnostic = traceback.format_exc()
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        (output / "launcher_error.txt").write_text(diagnostic, encoding="utf-8")
        try:
            archive = archive_partial(output)
            with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as zipped:
                if "launcher_error.txt" not in zipped.namelist():
                    zipped.writestr("launcher_error.txt", diagnostic)
                elif zipped.read("launcher_error.txt").decode("utf-8") != diagnostic:
                    zipped.writestr(f"launcher_failures/{timestamp()}.txt", diagnostic)
            show_archive(archive)
        except Exception as packaging_error:
            print(f"Could not package partial results: {packaging_error}", flush=True)
        raise


if __name__ == "__main__":
    launch(globals())
