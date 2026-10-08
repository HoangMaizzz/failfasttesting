"""Import-safe, fresh-run Kaggle launcher for the drafter-only simulator.

Only launch(scope) installs dependencies, fetches source/weights or starts work.
Notebook execution uses the __main__ guard; importing this module is inert.
"""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile


SOURCE_REPO = "https://github.com/HoangMaizzz/failfasttesting.git"
DEFAULT_SOURCE_REF = "codex/drafter-latent-simulator"
MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
RUNNER_RELATIVE = Path("run_drafter_simulator.py")
OVERLAY_RELATIVE = Path("Fast_dLLM_v2_1_5B/modeling.py")
DEPENDENCIES = (
    "transformers==4.53.1", "datasets", "scipy", "numpy", "matplotlib",
    "accelerate", "einops", "huggingface_hub<1", "safetensors", "sentencepiece",
)
WEIGHT_PATTERNS = ("*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.jinja")
DEFAULT_CONFIG = {
    "num_questions": 100, "max_new_tokens": 256, "max_context_tokens": 4096,
    "physical_block_size": 32, "small_block_size": 8, "threshold": 0.5,
    "seeds": [42, 43, 44], "latent_dim": 128, "horizon": 3,
    "encoder_updates": 400, "updates": 600, "eval_every": 100,
    "learning_milestones": [20, 40, 70], "batch_size": 32,
    "train_device": "cuda:1", "collect_device": "cuda:0",
    "benchmark_repetitions": 100, "benchmark_warmup": 20,
}
SMOKE_OVERRIDES = {
    "num_questions": 3, "max_new_tokens": 32, "encoder_updates": 2,
    "updates": 2, "eval_every": 1, "seeds": [42],
    "learning_milestones": [1], "batch_size": 2,
    "benchmark_repetitions": 3, "benchmark_warmup": 1,
}
SCOPE_OVERRIDES = {
    "NUM_QUESTIONS": "num_questions", "MAX_NEW_TOKENS": "max_new_tokens",
    "ENCODER_UPDATES": "encoder_updates", "UPDATES": "updates", "SEEDS": "seeds",
}
DATA_PARAMETERS = frozenset({
    "num_questions", "max_new_tokens", "max_context_tokens",
    "physical_block_size", "small_block_size", "threshold",
})
EXCLUDED_PARTS = frozenset({
    ".git", "__pycache__", ".cache", "hf", "hf_cache", "hub",
    "model_weights", "weights", "fast_dllm_v2_1_5b",
})
LAUNCHER_ARTIFACTS = frozenset({
    "launcher_metadata.json", "launcher_config.json", "dependency_request.json",
    "versions.json", "model_metadata.json", "launcher_error.txt",
})
OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_ref(ref):
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("SOURCE_REF must be a branch name or commit SHA")
    return ref


def build_config(scope):
    mode = scope.get("MODE", "full")
    if mode not in ("full", "smoke"):
        raise ValueError("MODE must be 'full' or 'smoke'")
    config = copy.deepcopy(DEFAULT_CONFIG)
    if mode == "smoke":
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
    data_parameters = scope.get("DATAPARAM", {})
    if not isinstance(data_parameters, dict) or set(data_parameters) - DATA_PARAMETERS:
        raise ValueError(f"DATAPARAM must be a dict with keys from {sorted(DATA_PARAMETERS)}")
    config.update(copy.deepcopy(data_parameters))
    for external, internal in SCOPE_OVERRIDES.items():
        if external in scope:
            config[internal] = copy.deepcopy(scope[external])
    for key in ("num_questions", "max_new_tokens", "max_context_tokens",
                "physical_block_size", "small_block_size", "encoder_updates", "updates"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_questions"] < 3:
        raise ValueError("Need at least three questions for nonempty train/validation/test splits")
    if config["max_new_tokens"] % 32:
        raise ValueError("max_new_tokens must be a positive multiple of the native block size 32")
    if config["physical_block_size"] != 32 or config["small_block_size"] != 8:
        raise ValueError("The native block canvas must stay 32 with active eligible sub-blocks of 8")
    if (isinstance(config["threshold"], bool) or not isinstance(config["threshold"], (int, float))
            or config["threshold"] != 0.5):
        raise ValueError("The native collector requires threshold=0.5")
    seeds = config["seeds"]
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("SEEDS must contain distinct integer seeds in [0, 2**32)")
    config["seeds"] = list(seeds)
    # Milestones refer to training questions; the runner owns question splitting.
    train_count = max(1, int(config["num_questions"] * 0.7))
    train_count = min(train_count, config["num_questions"] - 2)
    config["learning_milestones"] = sorted(set(
        min(milestone, train_count) for milestone in config["learning_milestones"]))
    return config


def run_command(command, *, cwd, env, check=True):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    return subprocess.run(command, cwd=str(cwd), env=env, check=check)


def download_source(repo, ref, env):
    ref = validate_ref(ref)
    repo = Path(repo)
    repo.mkdir(exist_ok=False)
    for command in (
        ["git", "init", repo],
        ["git", "-C", repo, "remote", "add", "origin", SOURCE_REPO],
        ["git", "-C", repo, "fetch", "--depth", "1", "origin", ref],
        ["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"],
    ):
        run_command(command, cwd=repo.parent, env=env)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], cwd=str(repo.parent), env=env, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Git did not return a resolved source commit SHA")
    return revision


def installed_versions(*, cwd, env):
    # Use a new interpreter after pip, including in already-used notebook kernels.
    code = (
        "import importlib.metadata as m,json,platform; "
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )
    return json.loads(subprocess.check_output(
        [sys.executable, "-c", code], cwd=str(cwd), env=env, text=True))


def install_dependencies(temp, output, env):
    torch_version = metadata.version("torch")
    constraint = Path(temp) / "keep_torch.txt"
    constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    write_json(Path(output) / "dependency_request.json", {
        "requirements": list(DEPENDENCIES), "preserved_torch": torch_version,
    })
    run_command([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                 "--constraint", constraint, *DEPENDENCIES], cwd=temp, env=env)
    versions = installed_versions(cwd=temp, env=env)
    write_json(Path(output) / "versions.json", versions)
    packages = {name.lower().replace("_", "-"): version
                for name, version in versions["packages"].items()}
    if packages.get("torch") != torch_version:
        raise RuntimeError("Dependency installation changed PyTorch")
    if packages.get("transformers") != "4.53.1":
        raise RuntimeError("Dependency installation did not select transformers==4.53.1")
    hub_version = packages.get("huggingface-hub", "")
    if not hub_version or int(hub_version.split(".", 1)[0]) >= 1:
        raise RuntimeError("Dependency installation did not select huggingface_hub<1")


def select_devices(scope, *, cwd, env):
    if scope.get("ALLOW_CPU") is True:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
            raise ValueError("ALLOW_CPU is for local checks only")
        return "cpu", "cpu"
    code = (
        "import json,torch; n=torch.cuda.device_count(); "
        "print(json.dumps({'available':torch.cuda.is_available(),'count':n,"
        "'names':[torch.cuda.get_device_name(i) for i in range(n)]}))"
    )
    info = json.loads(subprocess.check_output(
        [sys.executable, "-c", code], cwd=str(cwd), env=env, text=True))
    if not info["available"] or info["count"] < 1:
        raise RuntimeError("Select a GPU accelerator and enable Internet; T4 x2 is recommended")
    print("GPUs:", info["names"], flush=True)
    return "cuda:0", "cuda:1" if info["count"] >= 2 else "cuda:0"


# The installed hub library is loaded in a fresh process rather than the notebook.
# Write the resolved revision BEFORE downloading so failures retain provenance.
WEIGHT_DOWNLOAD_CODE = r'''
import hashlib, json, pathlib, shutil, sys
from huggingface_hub import HfApi, snapshot_download
model_id, requested, destination, overlay, record_path, patterns = sys.argv[1:]
destination, overlay, record_path = map(pathlib.Path, (destination, overlay, record_path))
info = HfApi().model_info(model_id, revision=requested)
revision = info.sha
if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
    raise RuntimeError("Hugging Face metadata did not resolve a model commit SHA")
record = {"model_id": model_id, "requested_revision": requested, "resolved_revision": revision,
          "allow_patterns": json.loads(patterns), "download_complete": False,
          "overlay_source": str(overlay), "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()}
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
snapshot_download(repo_id=model_id, revision=revision, local_dir=str(destination),
                  allow_patterns=record["allow_patterns"])
if not (destination / "config.json").is_file() or not any(destination.glob("*.safetensors")):
    raise RuntimeError("Model download is missing config.json or safetensors weights")
shutil.copy2(overlay, destination / "modeling.py")
record["download_complete"] = True
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
'''


def download_weights(repo, temp, output, scope, env):
    destination = Path(temp) / "hf" / "Fast_dLLM_v2_1_5B"
    overlay = Path(repo) / OVERLAY_RELATIVE
    if not overlay.is_file():
        raise FileNotFoundError(f"Selected source is missing the modeling.py overlay: {overlay}")
    requested = scope.get("MODEL_REVISION", "main")
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("MODEL_REVISION must be a nonempty model branch, tag or SHA")
    run_command([sys.executable, "-c", WEIGHT_DOWNLOAD_CODE, MODEL_ID, requested,
                 destination, overlay, Path(output) / "model_metadata.json",
                 json.dumps(WEIGHT_PATTERNS)], cwd=temp, env=env)
    return destination


def excluded_artifact(relative):
    parts = [part.lower() for part in Path(relative).parts]
    name = parts[-1]
    return (any(part in EXCLUDED_PARTS for part in parts)
            or name.endswith(".safetensors") or name.startswith("pytorch_model")
            or name.startswith("model.safetensors"))


def archive_partial(output):
    """Refresh the root ZIP, including late diagnostics after runner packaging."""
    output = Path(output)
    if not output.is_dir():
        return None
    archive = output.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.tmp")
    excluded = []
    def member_info(name):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        return info
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if path.is_symlink() or excluded_artifact(relative):
                excluded.append(relative.as_posix())
                continue
            if path.is_file():
                info = member_info(relative.as_posix())
                with path.open("rb") as source, zipped.open(
                        info, "w", force_zip64=path.stat().st_size >= zipfile.ZIP64_LIMIT) as destination:
                    shutil.copyfileobj(source, destination)
        zipped.writestr(member_info("launcher_packaging.json"), json.dumps({
            "refreshed_from_output_folder": True, "excluded": excluded,
        }, indent=2))
    temporary.replace(archive)
    return archive


def show_archive(archive):
    print("Output ZIP ready:", archive, flush=True)
    try:
        from IPython.display import FileLink, display
        display(FileLink(archive.name))
    except ImportError:
        pass


def verify_completion(output):
    summary_path = Path(output) / "summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise RuntimeError("Runner exited with code 0 but did not produce summary.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("Runner summary.json could not be read as JSON") from error
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise RuntimeError("Runner exited with code 0 but summary.json does not report status='complete'")
    return summary


def launch(scope):
    """Run one isolated experiment; exceptions stay failures after packaging."""
    working = Path(scope.get("WORKING_DIR", "/kaggle/working")).resolve()
    working.mkdir(parents=True, exist_ok=True)
    # Never chdir into a disposable checkout or delete/reuse a prior run folder.
    os.chdir(working)
    output = Path(tempfile.mkdtemp(prefix="drafter_simulator_", dir=working))
    record = {"status": "running", "stage": "configuration", "source_repo": SOURCE_REPO,
              "source_ref": scope.get("SOURCE_REF", DEFAULT_SOURCE_REF),
              "source_requested_ref": scope.get("SOURCE_REQUESTED_REF", scope.get("SOURCE_REF", DEFAULT_SOURCE_REF)),
              "mode": scope.get("MODE", "full"), "output_dir": str(output)}
    metadata_path = output / "launcher_metadata.json"
    staged_provenance = None
    try:
        write_json(metadata_path, record)
        config = build_config(scope)
        validate_ref(record["source_ref"])
        write_json(output / "launcher_config.json", config)
        temp_root = Path(scope.get("TEMP_DIR", "/kaggle/temp")).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="drafter_simulator_", dir=temp_root))
        record["temp_dir"] = str(temp)
        env = os.environ.copy()
        for key in OFFLINE_FLAGS:
            env.pop(key, None)
        env.update(PYTHONUNBUFFERED="1", GIT_LFS_SKIP_SMUDGE="1", WANDB_MODE="disabled",
                   USE_TF="0", USE_FLAX="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   HF_HOME=str(temp / "hf"), HF_HUB_CACHE=str(temp / "hf" / "hub"),
                   HF_DATASETS_CACHE=str(temp / "hf" / "datasets"),
                   HF_MODULES_CACHE=str(temp / "hf" / "modules"), TOKENIZERS_PARALLELISM="false")
        repo = temp / "repo"
        record["stage"] = "source_download"
        write_json(metadata_path, record)
        record["source_revision"] = download_source(repo, record["source_ref"], env)
        config["source_revision"] = record["source_revision"]
        write_json(output / "launcher_config.json", config)
        if not (repo / RUNNER_RELATIVE).is_file():
            raise FileNotFoundError(f"Selected source is missing {RUNNER_RELATIVE}")
        record["stage"] = "dependencies"
        write_json(metadata_path, record)
        install_dependencies(temp, output, env)
        config["collect_device"], config["train_device"] = select_devices(scope, cwd=temp, env=env)
        config_path = temp / "config.json"
        write_json(config_path, config)
        write_json(output / "launcher_config.json", config)
        record["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        record["stage"] = "tests"
        write_json(metadata_path, record)
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                     "-p", "test_drafter_simulator_*.py"], cwd=repo, env=env)
        record["stage"] = "weights_download"
        write_json(metadata_path, record)
        dllm = download_weights(repo, temp, output, scope, env)
        record["stage"] = "runner"
        command = [sys.executable, "-u", repo / RUNNER_RELATIVE,
                   "--config_json", config_path, "--output_dir", output, "--dllm_dir", dllm]
        record["runner_command"] = [str(part) for part in command]
        write_json(metadata_path, record)
        # The runner accepts an empty fresh output directory. Stage only files
        # created by this launch and restore them after its own final packaging.
        staged_provenance = temp / "launcher_provenance"
        staged_provenance.mkdir()
        for path in output.iterdir():
            if path.name not in LAUNCHER_ARTIFACTS or not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Unexpected artifact before runner launch: {path}")
        for path in list(output.iterdir()):
            path.replace(staged_provenance / path.name)
        metadata_path = staged_provenance / "launcher_metadata.json"
        print("Fresh GSM8K collection; max_new_tokens is a collection cap, not finished answers.", flush=True)
        print("Mode:", record["mode"], "config:", config, flush=True)
        result = run_command(command, cwd=repo, env=env, check=False)
        record["runner_returncode"] = result.returncode
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, record["runner_command"])
        # A zero exit alone is insufficient. Quality/efficacy still require
        # inspecting the runner's detailed audits, counts and metrics.
        summary = verify_completion(output)
        record.update(status="complete", stage="finished", runner_summary_status=summary["status"])
        write_json(metadata_path, record)
        return output
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(metadata_path, record)
        (metadata_path.parent / "launcher_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if staged_provenance is not None:
                for path in staged_provenance.iterdir():
                    shutil.copy2(path, output / path.name)
            archive = archive_partial(output)
            if archive is not None:
                show_archive(archive)
        except BaseException:
            if not active_error:
                raise
            print("Output packaging/display also failed:\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    launch(globals())
