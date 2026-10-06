"""Import-safe, one-cell Kaggle launcher for offline latent WM Phase 0.

Use launch(globals()). Notebook overrides: RUN_DIR, SOURCE_REF, MODE,
TOKEN_EMBEDDING, EMBEDDING_PATH, RESUME_INPUT, CONTINUE_DIAGNOSTICS. Local checks can additionally
use WORKING_DIR, TEMP_DIR and explicit ALLOW_CPU=True.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.util
import itertools
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

DEFAULT_RUN_DIR = "/kaggle/input/datasets/ainzkhail/2source2"
DEFAULT_SOURCE_REF = "codex/latent-wm-phase0"
SOURCE_REPO = "https://github.com/HoangMaizzz/failfasttesting.git"
DEFAULT_EMBEDDING_MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
CONFIG_RELATIVE = Path("configs/latent_wm_phase0.json")
RUNNER_RELATIVE = Path("run_latent_wm_phase0.py")
DEPENDENCIES = ("numpy", "scikit-learn", "huggingface_hub", "safetensors")
TEST_FILES = (
    "tests/test_phase0_wm_models.py",
    "tests/test_phase0_wm_metrics.py",
    "tests/test_phase0_kaggle_launcher.py",
    "tests/test_phase0_wm_runner.py",
)
SMOKE_OVERRIDES = {"num_questions": 20, "latent_dims": [64]}
EXCLUDED_PARTS = frozenset({
    "input", "input_cache", "raw", "raw_experiences", "experience",
    "experiences", "cache", "hf_cache", "model_cache", "embeddings",
    "embedding_cache", "__pycache__", ".git", "fast_dllm_v2_1_5b",
})
EXCLUDED_FILES = frozenset({
    "embedding.pt", "embeddings.pt", "token_embedding.pt", "token_embeddings.pt",
    "embedding.safetensors", "embeddings.safetensors", "model.safetensors",
    "pytorch_model.bin",
})
MAX_ARCHIVE_FILE_BYTES = 128 * 1024 * 1024


def timestamp():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S_%f_ICT")
    except (ImportError, KeyError):
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")


def on_kaggle():
    return Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def preflight_input(path, mount_root=Path("/kaggle/input")):
    """Show mount diagnostics before any source/package download."""
    path, mount_root = Path(path), Path(mount_root)
    print("Input mount roots:", flush=True)
    if mount_root.is_dir():
        for item in itertools.islice(mount_root.iterdir(), 30):
            print(" ", item, flush=True)
            if item.is_dir() and item.name == "datasets":
                for owner in itertools.islice(item.iterdir(), 20):
                    print("   ", owner, flush=True)
                    if owner.is_dir():
                        for dataset in itertools.islice(owner.iterdir(), 20):
                            print("     ", dataset, flush=True)
    else:
        print(" ", mount_root, "(not present; local run)", flush=True)
    print("Requested RUN_DIR:", path, flush=True)
    if not path.exists():
        raise FileNotFoundError(f"Input is not mounted: {path}; set RUN_DIR to a printed mount")
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ValueError("RUN_DIR must be an original trace ZIP or extracted folder")
    elif path.is_dir():
        samples = list(itertools.islice((p for p in path.rglob("*") if p.is_file()), 25))
        print("Input files:", [str(p.relative_to(path)) for p in samples], flush=True)
        if not samples:
            raise FileNotFoundError(f"Input folder is empty: {path}")
    else:
        raise ValueError(f"Unsupported RUN_DIR: {path}")
    return path.resolve()


def _resolve_original_input(repo, path):
    # Load only the existing launcher's import-safe helpers, from the selected
    # checkout. A notebook can bootstrap this file alone from a raw source URL.
    spec = importlib.util.spec_from_file_location(
        "_phase0_feasibility_helpers", Path(repo) / "kaggle_factorized_wm_feasibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_original_input(repo, path)


def build_config(base, scope):
    """Preserve JSON budgets in full mode; explicitly shrink smoke budgets."""
    if not isinstance(base, dict):
        raise ValueError("Phase 0 config must be a JSON object")
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    config = copy.deepcopy(base)
    if base.get("num_questions") != 100 or base.get("latent_dims") != [64, 128]:
        raise ValueError("Full source JSON must specify 100 questions and latent_dims [64, 128]")
    protocol = config.setdefault("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    fractions = protocol.get("split_fractions", [0.7, 0.15, 0.15])
    if fractions != [0.7, 0.15, 0.15]:
        raise ValueError("Phase 0 requires question split fractions 70/15/15")
    if mode == "smoke":
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
        if "seeds" in config:
            config["seeds"] = [config.get("seed", 42)]
        # The runner owns the names and full budgets; cap every update budget
        # it publishes, including later additions to the Phase 0 JSON.
        for key, value in list(config.items()):
            if (key.endswith("_updates") or key == "dynamics_updates_per_horizon") and isinstance(value, int) and not isinstance(value, bool):
                config[key] = min(value, 5 if key.endswith("min_updates") or key == "minimum_updates" else
                                  10 if key in ("encoder_updates", "stage_a_updates") else 20)
        for key, cap in (("eval_every", 5), ("early_patience", 2), ("patience", 2), ("bootstrap_samples", 50)):
            if key in config:
                config[key] = min(config[key], cap)
    config["mode"] = mode
    protocol.update(split_unit="question", split_fractions=[0.7, 0.15, 0.15],
                    fixed_split_across_jobs=True,
                    split_counts=(dict(train=14, validation=3, test=3) if mode == "smoke"
                                  else dict(train=70, validation=15, test=15)))
    config["device_encoder"], config["device_dynamics"] = "cuda:0", "cuda:1"
    embedding = scope.get("TOKEN_EMBEDDING", config.get("token_embedding", "pretrained"))
    if embedding not in ("pretrained", "learned"):
        raise ValueError("TOKEN_EMBEDDING must be 'pretrained' or 'learned'")
    config["token_embedding"] = embedding
    config.setdefault("embedding_repo_id", DEFAULT_EMBEDDING_MODEL_ID)
    if "CONTINUE_DIAGNOSTICS" in scope:
        if not isinstance(scope["CONTINUE_DIAGNOSTICS"], bool):
            raise ValueError("CONTINUE_DIAGNOSTICS must be True or False")
        config["continue_diagnostics"] = scope["CONTINUE_DIAGNOSTICS"]
    if "EMBEDDING_PATH" in scope:
        path = scope["EMBEDDING_PATH"]
        if path is not None and (not isinstance(path, (str, os.PathLike)) or not str(path).strip()):
            raise ValueError("EMBEDDING_PATH must be a nonempty path or None")
        config["embedding_path"] = str(path) if path is not None else None
    return config


def select_devices(scope, kaggle, torch_module=None):
    """Return encoder/verifier then R/E dynamics devices, in that order."""
    if kaggle and scope.get("ALLOW_CPU") is True:
        raise ValueError("ALLOW_CPU is only supported for an explicit local, non-Kaggle run")
    if torch_module is None:
        import torch as torch_module
    count = torch_module.cuda.device_count()
    if torch_module.cuda.is_available() and count >= 2:
        print("GPUs:", [torch_module.cuda.get_device_name(i) for i in (0, 1)], flush=True)
        return "cuda:0", "cuda:1"
    if not kaggle and scope.get("ALLOW_CPU") is True:
        print("Explicit local CPU check.", flush=True)
        return "cpu", "cpu"
    raise RuntimeError(f"Need two visible GPUs (found {count}); select Kaggle GPU T4 x2")


def run_command(command, *, cwd, env):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def download_source(repo, ref, env):
    """Fetch the requested branch or SHA into a new, detached checkout."""
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("SOURCE_REF must be a branch name or commit SHA")
    repo = Path(repo)
    repo.mkdir(exist_ok=False)
    for command in (
        ["git", "init", repo],
        ["git", "-C", repo, "remote", "add", "origin", SOURCE_REPO],
        ["git", "-C", repo, "fetch", "--depth", "1", "origin", ref],
        ["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"],
    ):
        run_command(command, cwd=repo.parent, env=env)
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], env=env, text=True).strip()


def resolve_original_input(repo, path):
    """Use factorized_wm_data.resolve_input; also discover extensionless ZIPs."""
    path = Path(path)
    if not path.is_dir():
        return _resolve_original_input(repo, path)
    candidates = []
    original_error = None
    try:
        candidates.append(_resolve_original_input(repo, path))
    except ValueError as error:
        if "Ambiguous input" in str(error):
            raise
        original_error = error
    for item in path.rglob("*"):
        if (item.is_file() and not item.is_symlink() and item.suffix.lower() != ".zip"
                and zipfile.is_zipfile(item)):
            try:
                candidates.append(_resolve_original_input(repo, item))
            except ValueError as error:
                if "Ambiguous input" in str(error):
                    raise
    candidates = sorted({Path(item).resolve() for item in candidates})
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous input: multiple ORIGINAL complete runs: {candidates}")
    if not candidates:
        raise original_error or ValueError(f"No ORIGINAL complete trace in {path}")
    return candidates[0]


def safe_extract(archive, destination):
    """Validate all members before writing; never traverse or follow links."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for member in zipped.infolist():
            spelling = member.orig_filename.replace("\\", "/")
            portable = PurePosixPath(spelling)
            windows = PureWindowsPath(spelling)
            mode = member.external_attr >> 16
            target = (destination / portable).resolve()
            normalized = portable.as_posix().rstrip("/")
            if (not normalized or portable.is_absolute() or ".." in portable.parts
                    or windows.drive or ":" in spelling or "\x00" in member.orig_filename
                    or stat.S_ISLNK(mode) or not target.is_relative_to(destination)
                    or normalized in seen or member.flag_bits & 1):
                raise ValueError(f"Unsafe path/symlink/duplicate in resume ZIP: {member.filename}")
            seen.add(normalized)
        destination.mkdir(parents=True, exist_ok=False)
        # Normalize Windows separators identically during validation and writing.
        for member in zipped.infolist():
            target = destination / PurePosixPath(member.filename.replace("\\", "/"))
            if member.filename.replace("\\", "/").endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
    return destination


def _read_object(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Resume root requires {path.name}: {path.parent}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Resume {path.name} must contain a nonempty JSON object")
    return value


def validate_resume_root(root):
    """Check Phase 0 config, its runner manifest, and actual saved checkpoints.

    The runner remains responsible for content/config/code fingerprints and
    checkpoint payload validation. This is deliberately not the old jobs schema.
    """
    root = Path(root).resolve()
    config = _read_object(root / "config.json")
    if (not isinstance(config.get("latent_dims"), list) or not config["latent_dims"]
            or "device_encoder" not in config or "device_dynamics" not in config):
        raise ValueError("Resume config is not a latent WM Phase 0 runner config")
    protocol = config.get("protocol", {})
    resume = protocol.get("resume", {}) if isinstance(protocol, dict) else {}
    declared = resume.get("runner_manifest") if isinstance(resume, dict) else None
    names = [declared or "study_manifest.json"]
    if any(not isinstance(name, str) or Path(name).name != name or ":" in name or "\\" in name
           for name in names):
        raise ValueError("Resume runner_manifest must be a basename")
    manifests = [root / name for name in names if (root / name).is_file()]
    if len(manifests) != 1:
        raise ValueError("Resume root requires one Phase 0 runner manifest")
    manifest = _read_object(manifests[0])
    if manifest.get("config") != config:
        raise ValueError("Resume runner manifest config differs from config.json")
    if not isinstance(manifest.get("fingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["fingerprint"]):
        raise ValueError("Resume runner manifest requires a SHA256 fingerprint")
    if not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError("Resume runner manifest requires the original trace source")
    if manifest.get("status") not in ("running", "complete"):
        raise ValueError("Resume runner manifest status must be 'running' or 'complete'")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Resume results may not contain symlinks")
    checkpoints = [path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in (".pt", ".pth", ".safetensors")
                   and path.stat().st_size > 0
                   and not any(part.lower() in EXCLUDED_PARTS for part in path.relative_to(root).parts)
                   and path.name.lower() not in EXCLUDED_FILES | {"preprocessing.pt", "frozen_latents.pt"}]
    if not checkpoints:
        raise ValueError("Resume root requires saved latent model checkpoints")
    return root


def locate_resume_root(container):
    container = Path(container).resolve()
    candidates = list(dict.fromkeys([container] + [p.parent for p in container.rglob("config.json")]))
    roots, problems = [], []
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        try:
            roots.append(validate_resume_root(candidate))
        except (ValueError, json.JSONDecodeError) as error:
            problems.append(str(error))
    if len(roots) != 1:
        raise ValueError(f"Need one previous Phase 0 result root; found {len(roots)}. {'; '.join(problems[:3])}")
    return roots[0]


def restore_resume_input(path, output, temp):
    """Restore to the fresh output; original RUN_DIR is never replaced."""
    path, output, temp = Path(path).resolve(), Path(output).resolve(), Path(temp)
    if not path.exists():
        raise FileNotFoundError(f"RESUME_INPUT is not mounted: {path}")
    if path.is_file():
        source = locate_resume_root(safe_extract(path, temp / "resume_input"))
    else:
        try:
            source = locate_resume_root(path)
        except ValueError:
            archives = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
                        and zipfile.is_zipfile(p)]
            if len(archives) != 1:
                raise ValueError("Set RESUME_INPUT to one previous Phase 0 ZIP or extracted result root")
            source = locate_resume_root(safe_extract(archives[0], temp / "resume_input"))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("New output and RESUME_INPUT must be separate folders")
    # Uploaded results should already be compact. Also keep unexpected raw
    # assets/caches in temporary storage rather than copying them into Output.
    for item in source.rglob("*"):
        if item.is_file() and is_compact_file(item, source):
            target = output / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    return validate_resume_root(output)


def is_compact_file(path, root):
    relative = path.relative_to(root)
    return (not path.is_symlink()
            and not any(part.lower() in EXCLUDED_PARTS for part in relative.parts)
            and path.name.lower() not in EXCLUDED_FILES
            and path.suffix.lower() not in (".npz", ".zip")
            and path.stat().st_size <= MAX_ARCHIVE_FILE_BYTES)


def archive_partial(output):
    """Prefer the runner's root ZIP, otherwise atomically package compact files."""
    output = Path(output)
    archive = output.with_suffix(".zip")
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zipped:
                damaged = zipped.testzip()
            if damaged:
                raise zipfile.BadZipFile(f"CRC check failed: {damaged}")
            return archive
        except (zipfile.BadZipFile, OSError, RuntimeError) as error:
            print(f"Runner ZIP is unreadable; packaging available outputs: {error}", flush=True)
    temporary = archive.with_suffix(".zip.tmp")
    excluded = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(output)
            if relative.as_posix() == "launcher_packaging.json":
                continue  # Refresh restored fallback metadata without duplicate ZIP members.
            if not is_compact_file(path, output):
                excluded.append(relative.as_posix())
                continue
            zipped.write(path, relative.as_posix())
        zipped.writestr("launcher_packaging.json", json.dumps({
            "fallback": True, "excluded": excluded, "max_file_bytes": MAX_ARCHIVE_FILE_BYTES,
        }, indent=2))
    temporary.replace(archive)
    return archive


def show_archive(archive):
    print("DOWNLOAD ZIP (working root):", Path(archive).name, flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(Path(archive).name))
    except ImportError:
        pass


def launch(scope):
    """Fetch, validate and invoke only run_latent_wm_phase0.py."""
    kaggle = on_kaggle()
    working = (Path("/kaggle/working") if kaggle
               else Path(scope.get("WORKING_DIR", Path.cwd()))).resolve()
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    mode = scope.get("MODE", "full")
    safe_mode = mode if mode in ("full", "smoke") else "invalid"
    output = Path(tempfile.mkdtemp(prefix=f"latent_wm_phase0_{safe_mode}_{timestamp()}_", dir=working))
    try:
        if mode not in ("full", "smoke"):
            raise ValueError("MODE must be 'full' or 'smoke'")
        original_mount = preflight_input(scope.get("RUN_DIR", DEFAULT_RUN_DIR))
        devices = select_devices(scope, kaggle)
        temp_root = Path("/kaggle/temp") if kaggle else Path(scope.get("TEMP_DIR", tempfile.gettempdir()))
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="latent_wm_phase0_", dir=temp_root))
        cache = temp / "hf_cache"
        cache.mkdir()
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", GIT_LFS_SKIP_SMUDGE="1", WANDB_MODE="disabled",
                   HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
                   HUGGINGFACE_HUB_CACHE=str(cache / "hub"), XDG_CACHE_HOME=str(temp / "cache"),
                   TORCH_HOME=str(temp / "torch_cache"), TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp))
        repo = temp / "repo"
        ref = scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)
        revision = download_source(repo, ref, env)
        print("Source commit:", revision, flush=True)
        run_command([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *DEPENDENCIES],
                    cwd=repo, env=env)
        original = resolve_original_input(repo, original_mount)
        base = json.loads((repo / CONFIG_RELATIVE).read_text(encoding="utf-8"))
        config = build_config(base, scope)
        config["device_encoder"], config["device_dynamics"] = devices
        resume = bool(scope.get("RESUME_INPUT"))
        if resume:
            restore_resume_input(scope["RESUME_INPUT"], output, temp)
            saved = _read_object(output / "config.json")
            if saved != config:
                raise ValueError("Resume effective config differs from saved config; use the original MODE and embedding overrides")
        external_config = output / "launcher_config.json"
        external_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "launcher_metadata.json").write_text(json.dumps({
            "source_ref": ref, "source_revision": revision, "input": str(original),
            "mode": mode, "resume": resume, "LLM_forwards": 0,
            "embedding_request": {"mode": config["token_embedding"],
                                  "model_id": config.get("embedding_repo_id"),
                                  "revision": config.get("embedding_revision"),
                                  "path": config.get("embedding_path"),
                                  "actual_provenance_reported_by": "runner"},
            "cache_root": str(temp), "timestamp_timezone": "Asia/Bangkok (ICT); UTC fallback",
        }, indent=2), encoding="utf-8")
        print("Experiment:", mode, "questions=", config["num_questions"],
              "latent_dims=", config["latent_dims"], "split=", config["protocol"]["split_counts"],
              "encoder/verifier=", devices[0], "R/E=", devices[1], flush=True)
        print("Offline saved traces; LLM forwards: 0; token embedding:", config["token_embedding"], flush=True)
        for relative in TEST_FILES:
            if not (repo / relative).is_file():
                raise FileNotFoundError(f"Source ref is missing required Phase 0 test: {relative}")
            run_command([sys.executable, repo / relative], cwd=repo, env=env)
        if not (repo / RUNNER_RELATIVE).is_file():
            raise FileNotFoundError(f"Source ref is missing {RUNNER_RELATIVE}")
        command = [sys.executable, "-u", repo / RUNNER_RELATIVE,
                   "--input", original, "--output", output, "--config", external_config]
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
            # Preserve a runner ZIP produced at an earlier stage, and include
            # this launcher failure without replacing its contents.
            with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as zipped:
                if "launcher_error.txt" not in zipped.namelist():
                    zipped.writestr("launcher_error.txt", diagnostic)
            show_archive(archive)
        except Exception as packaging_error:
            print(f"Could not package partial results: {packaging_error}", flush=True)
        raise


if __name__ == "__main__":
    launch(globals())
