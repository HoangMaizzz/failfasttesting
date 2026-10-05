"""Import-safe Kaggle launcher for the offline factorized WM experiment.

Notebook globals: RUN_DIR, SOURCE_REF, MODE, RESUME_INPUT, RESUME_OUTPUT.
Local checks may additionally set WORKING_DIR, TEMP_DIR and ALLOW_CPU=True.
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
DEFAULT_SOURCE_REF = "codex/factorized-wm-feasibility"
SOURCE_REPO = "https://github.com/HoangMaizzz/failfasttesting.git"
CONFIG_RELATIVE = Path("configs/factorized_wm_feasibility.json")
TEST_FILES = (
    "tests/test_factorized_wm_data.py",
    "tests/test_factorized_wm_metrics.py",
    "tests/test_factorized_wm_models.py",
    "tests/test_factorized_kaggle_launcher.py",
    "tests/test_factorized_wm_runner.py",
)
FULL_CONTRACT = {
    "num_questions": 100, "seed": 42, "seeds": [42, 43, 44],
    "hidden_dims": [8, 16, 32], "representations": ["S", "H", "SH", "SHT"],
    "scaling_questions": [20, 40, 60, 70],
    "capacities": {
        "small": {"width": 64, "layers": 1},
        "medium": {"width": 128, "layers": 2},
        "large": {"width": 256, "layers": 3},
    },
    "projection_updates": 300, "drafter_updates": 1200,
    "verifier_updates": 1200, "direct_updates": 800, "batch_size": 16,
    "eval_every": 100, "early_patience": 4, "h1_min_updates": 400,
    "horizon": 3, "device_drafter": "cuda:1", "device_verifier": "cuda:0",
    "max_proposal_tokens": 64, "extend_size": 8,
    "bootstrap_samples": 1000, "save_every_stage": True,
}
SMOKE_OVERRIDES = {
    "num_questions": 20, "seeds": [42], "hidden_dims": [8],
    "representations": ["SHT"], "scaling_questions": [14],
    "capacities": {"medium": {"width": 32, "layers": 1}},
    "projection_updates": 10, "drafter_updates": 20,
    "verifier_updates": 20, "direct_updates": 20,
    "eval_every": 10, "h1_min_updates": 5,
}
EXCLUDED_PARTS = frozenset({
    "input", "input_cache", "raw", "raw_experiences", "experience",
    "experiences", "cache", "hf_cache", "model_cache", "__pycache__",
    ".git", "fast_dllm_v2_1_5b",
})
MAX_ARCHIVE_FILE_BYTES = 128 * 1024 * 1024


def build_config(base, scope):
    """Make an independent effective config; full mode never reduces the grid."""
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    if not isinstance(base, dict):
        raise ValueError("Experiment config must be a JSON object")
    config = copy.deepcopy(base)
    if mode == "full":
        mismatches = [key for key, expected in FULL_CONTRACT.items()
                      if config.get(key) != expected]
        # Explicit local CPU operation is the only full-contract device exception.
        if scope.get("ALLOW_CPU") is True:
            mismatches = [key for key in mismatches
                          if key not in ("device_drafter", "device_verifier")]
        if mismatches:
            raise ValueError(f"Full config differs from the requested experiment: {mismatches}")
    else:
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
    config["mode"] = mode
    protocol = config.setdefault("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    protocol.update(split_unit="question", split_seed=42,
                    split_fractions=[0.7, 0.15, 0.15], fixed_split_across_jobs=True)
    protocol["split_counts"] = (dict(train=14, validation=3, test=3) if mode == "smoke"
                                else dict(train=70, validation=15, test=15))
    return config


def timestamp():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S_%f_ICT")
    except (ImportError, KeyError):
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")


def on_kaggle():
    return Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def preflight_input(path, mount_root=Path("/kaggle/input")):
    """Filesystem diagnostics happen before any source or package download."""
    path, mount_root = Path(path), Path(mount_root)
    print("Input mount roots:", flush=True)
    if mount_root.is_dir():
        roots = sorted(mount_root.iterdir())
        for root in roots[:30]:
            print(" ", root, flush=True)
            if root.is_dir() and root.name == "datasets":
                for owner in sorted(root.iterdir())[:20]:
                    print("   ", owner, flush=True)
                    if owner.is_dir():
                        for dataset in sorted(owner.iterdir())[:20]:
                            print("     ", dataset, flush=True)
    else:
        print(" ", mount_root, "(not present; local run)", flush=True)
    print("Requested RUN_DIR:", path, flush=True)
    if not path.exists():
        raise FileNotFoundError(f"Input is not mounted: {path}; set RUN_DIR to a printed mount")
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ValueError("RUN_DIR must be an original experience ZIP or extracted folder")
    elif path.is_dir():
        samples = list(itertools.islice((p for p in path.rglob("*") if p.is_file()), 25))
        print("Input files:", [str(p.relative_to(path)) for p in samples], flush=True)
        if not samples:
            raise FileNotFoundError(f"Input folder is empty: {path}")
    else:
        raise ValueError(f"Unsupported RUN_DIR: {path}")
    return path.resolve()


def select_devices(scope, kaggle, torch_module=None):
    if kaggle and scope.get("ALLOW_CPU") is True:
        raise ValueError("ALLOW_CPU is only supported for an explicit local, non-Kaggle run")
    if torch_module is None:
        import torch as torch_module
    count = torch_module.cuda.device_count()
    if torch_module.cuda.is_available() and count >= 2:
        print("GPUs:", [torch_module.cuda.get_device_name(i) for i in (0, 1)], flush=True)
        return "cuda:1", "cuda:0"
    if not kaggle and scope.get("ALLOW_CPU") is True:
        print("Explicit local CPU run; both small models train on CPU.", flush=True)
        return "cpu", "cpu"
    raise RuntimeError(f"Need two visible GPUs (found {count}); select Kaggle GPU T4 x2")


def run_command(command, *, cwd, env):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def download_source(repo, ref, env):
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("SOURCE_REF must be a branch name or commit SHA")
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
    """Delegate content discovery to the downloaded data module."""
    module_path = repo / "factorized_wm_data.py"
    spec = importlib.util.spec_from_file_location("_factorized_launcher_data", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(repo))
    try:
        spec.loader.exec_module(module)
        # The data module reads ZIP members in place and keeps nested prefixes internal.
        return Path(module.resolve_input(path)).resolve()
    finally:
        sys.path.pop(0)
        sys.modules.pop(spec.name, None)


def safe_extract(archive, destination):
    """Validate every ZIP entry before extracting any entry."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            portable = PurePosixPath(member.filename.replace("\\", "/"))
            windows = PureWindowsPath(member.filename)
            mode = member.external_attr >> 16
            target = (destination / member.filename).resolve()
            if (portable.is_absolute() or ".." in portable.parts or windows.drive
                    or stat.S_ISLNK(mode) or not target.is_relative_to(destination)):
                raise ValueError(f"Unsafe path/symlink in resume ZIP: {member.filename}")
        destination.mkdir(parents=True, exist_ok=False)
        zipped.extractall(destination)
    return destination


def validate_resume_root(root):
    root = Path(root).resolve()
    documents = {}
    for name in ("config.json", "study_manifest.json"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Resume root requires {name}: {root}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Resume {name} must contain a nonempty JSON object")
        documents[name] = value
    manifest = documents["study_manifest.json"]
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Resume study_manifest.json requires a SHA256 fingerprint")
    if manifest.get("config") != documents["config.json"]:
        raise ValueError("Resume study_manifest.json config differs from root config.json")
    if not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError("Resume study_manifest.json requires the original source")
    if manifest.get("status") not in ("running", "complete"):
        raise ValueError("Resume study_manifest.json status must be 'running' or 'complete'")
    if not (root / "jobs").is_dir() or (root / "jobs").is_symlink():
        raise ValueError("Resume root must contain a jobs directory with saved job artifacts")
    if not any(p.is_dir() for p in (root / "jobs").iterdir()):
        raise ValueError("Resume root has no saved job directories")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Resume results may not contain symlinks")
    return root


def locate_resume_root(container):
    container = Path(container)
    candidates = [container] + [p.parent for p in container.rglob("config.json")]
    roots = list(dict.fromkeys(p.resolve() for p in candidates
                              if (p / "config.json").is_file()
                              and (p / "study_manifest.json").is_file()
                              and (p / "jobs").is_dir()))
    if len(roots) != 1:
        raise ValueError(f"Need one previous result root with config/study_manifest/jobs; found {roots}")
    return validate_resume_root(roots[0])


def restore_resume_input(path, output, temp):
    """Copy an uploaded result to a new working root; input mounts stay read-only."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RESUME_INPUT is not mounted: {path}")
    if path.is_dir():
        try:
            source = locate_resume_root(path)
        except ValueError:
            archives = [p for p in path.rglob("*") if p.is_file() and zipfile.is_zipfile(p)]
            if len(archives) != 1:
                raise ValueError("Set RESUME_INPUT to one previous result ZIP or extracted result root")
            source = locate_resume_root(safe_extract(archives[0], temp / "resume_input"))
    else:
        source = locate_resume_root(safe_extract(path, temp / "resume_input"))
    if Path(output).resolve().is_relative_to(source):
        raise ValueError("The new output folder must not be inside RESUME_INPUT")
    shutil.copytree(source, output, dirs_exist_ok=True)
    return validate_resume_root(output)


def archive_partial(output):
    """Use the runner archive, or package available outputs if it is missing."""
    output = Path(output)
    archive = output.with_suffix(".zip")
    if archive.is_file():
        return archive
    temporary = archive.with_suffix(".zip.tmp")
    excluded = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(output)
            if (any(part.lower() in EXCLUDED_PARTS for part in relative.parts)
                    or path.stat().st_size > MAX_ARCHIVE_FILE_BYTES):
                excluded.append(str(relative))
                continue
            zipped.write(path, relative.as_posix())
        zipped.writestr("launcher_packaging.json", json.dumps({
            "fallback": True, "excluded": excluded,
            "max_file_bytes": MAX_ARCHIVE_FILE_BYTES,
        }, indent=2))
    temporary.replace(archive)
    return archive


def show_archive(archive):
    print("DOWNLOAD ZIP (working root):", archive.name, flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(archive.name))
    except ImportError:
        pass


def launch(scope):
    kaggle = on_kaggle()
    working = (Path("/kaggle/working") if kaggle
               else Path(scope.get("WORKING_DIR", Path.cwd()))).resolve()
    working.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    mode = scope.get("MODE", "full")
    safe_mode = mode if mode in ("full", "smoke") else "invalid"
    output = working / f"factorized_wm_feasibility_{safe_mode}_{timestamp()}"
    resume = False
    try:
        if mode not in ("full", "smoke"):
            raise ValueError("MODE must be 'full' or 'smoke'")
        if scope.get("RESUME_INPUT") and scope.get("RESUME_OUTPUT"):
            raise ValueError("Use only one of RESUME_INPUT or RESUME_OUTPUT")
        if scope.get("RESUME_OUTPUT"):
            requested = Path(scope["RESUME_OUTPUT"])
            if not requested.is_absolute():
                raise ValueError("RESUME_OUTPUT must be an exact absolute path in the working folder")
            requested = requested.resolve()
            if requested == working or not requested.is_relative_to(working):
                raise ValueError("RESUME_OUTPUT must be an existing result folder inside the working folder")
            output = validate_resume_root(requested)
            resume = True
        output.mkdir(parents=True, exist_ok=True)
        source_input = preflight_input(scope.get("RUN_DIR", DEFAULT_RUN_DIR))
        devices = select_devices(scope, kaggle)
        temp_root = Path("/kaggle/temp") if kaggle else Path(scope.get("TEMP_DIR", tempfile.gettempdir()))
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="factorized_wm_", dir=temp_root))
        if scope.get("RESUME_INPUT"):
            restore_resume_input(scope["RESUME_INPUT"], output, temp)
            resume = True
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", GIT_LFS_SKIP_SMUDGE="1", WANDB_MODE="disabled")
        repo = temp / "repo"
        revision = download_source(repo, scope.get("SOURCE_REF", DEFAULT_SOURCE_REF), env)
        print("Source commit:", revision, flush=True)
        run_command([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                     "numpy", "scikit-learn", "matplotlib"], cwd=repo, env=env)
        original = resolve_original_input(repo, source_input)
        print("Resolved original input:", original, flush=True)
        config_path = output / "config.json" if resume else repo / CONFIG_RELATIVE
        config = build_config(json.loads(config_path.read_text(encoding="utf-8")), scope)
        config["device_drafter"], config["device_verifier"] = devices
        external_config = output / "launcher_config.json"
        external_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "launcher_metadata.json").write_text(json.dumps({
            "source_ref": scope.get("SOURCE_REF", DEFAULT_SOURCE_REF), "source_revision": revision,
            "input": str(original), "mode": mode, "resume": resume,
            "timestamp_timezone": "Asia/Bangkok (ICT); UTC suffix if zoneinfo unavailable",
        }, indent=2), encoding="utf-8")
        print("Experiment:", mode, "questions=", config["num_questions"],
              "split=", config["protocol"]["split_counts"], "devices=", devices, flush=True)
        print("Offline saved traces only; LLM forwards: 0.", flush=True)
        for relative in TEST_FILES:
            run_command([sys.executable, repo / relative], cwd=repo, env=env)
        command = [sys.executable, "-u", repo / "run_factorized_wm_feasibility.py",
                   "--input", original, "--output", output, "--config", external_config]
        if resume:
            command.append("--resume")
        run_command(command, cwd=repo, env=env)
        archive = archive_partial(output)
        with zipfile.ZipFile(archive) as zipped:
            damaged = zipped.testzip()
            if damaged:
                raise RuntimeError(f"Result ZIP failed CRC check: {damaged}")
        show_archive(archive)
        return output
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        (output / "launcher_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            show_archive(archive_partial(output))
        except Exception as packaging_error:
            print(f"Could not package partial results: {packaging_error}", flush=True)
        raise


if __name__ == "__main__":
    launch(globals())
