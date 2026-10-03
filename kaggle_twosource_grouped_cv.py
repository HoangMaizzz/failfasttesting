"""Kaggle entry point for offline question-grouped 5-fold TwoSource CV.

Execute via a small notebook cell that downloads this file from the selected
GitHub branch and runs it with RUN_DIR/SOURCE_REF in globals.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


SOURCE_REF = globals().get("SOURCE_REF", "codex/persistent-wm-film-v1")
RUN_DIR = Path(globals().get("RUN_DIR", "/kaggle/input/datasets/ainzkhail/2source2"))
FOLDS = int(globals().get("FOLDS", 5))
UPDATES_PER_QUESTION = int(globals().get("UPDATES_PER_QUESTION", 16))
MILESTONES = list(globals().get("MILESTONES", [10, 20, 40, 60, 80]))
ROOTS_PER_QUESTION = int(globals().get("ROOTS_PER_QUESTION", 16))

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
WORK.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

env = os.environ.copy()
env.update(GIT_LFS_SKIP_SMUDGE="1", CUDA_VISIBLE_DEVICES="0,1",
           HF_HOME=str(TEMP / "twosource_cv_hf_cache"), PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="false",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           USE_TF="0", USE_FLAX="0", WANDB_MODE="disabled")


def run(command, cwd=None):
    command = [str(item) for item in command]
    print("\n>>>", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def locate_run(path):
    required = ("summary.json", "states.jsonl", "edges.jsonl", "labels.jsonl",
                "teacher_targets.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Kaggle input folder does not exist: {path}")
    candidates = [path] if path.is_dir() else []
    candidates += [item.parent for item in path.rglob("summary.json")]
    matches = []
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            try:
                summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if summary.get("schema") == "interactive_acceptance_two_source_v1":
                matches.append(candidate)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one extracted complete TwoSource run under {path}; found {matches}. "
            "This cell accepts the Kaggle input folder directly and does not search for an old ZIP name.")
    return matches[0]


RUN_DIR = locate_run(RUN_DIR)
run_summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
if run_summary.get("status") != "complete":
    raise RuntimeError(f"Input run is partial: {run_summary.get('status')}")
print(f"Input TwoSource run: {RUN_DIR}", flush=True)
print(f"Questions={run_summary.get('questions_completed')} states={run_summary.get('nodes')} "
      f"edges={run_summary.get('edges')}; original checkpoint will NOT be loaded.", flush=True)

import torch
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 for this cell")
for gpu in range(2):
    print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)}", flush=True)
if FOLDS != 5 or UPDATES_PER_QUESTION < 1 or MILESTONES != sorted(set(MILESTONES)):
    raise ValueError("Use 5 folds, positive updates/question and increasing unique milestones")

run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "transformers==4.53.1", "accelerate", "einops", "numpy", "huggingface_hub"])

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
repo = TEMP / f"twosource_grouped_cv_repo_{stamp}"
run(["git", "clone", "--depth", "1", "--branch", SOURCE_REF,
     "https://github.com/HoangMaizzz/failfasttesting.git", repo], cwd=WORK)
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 cwd=WORK, env=env, text=True).strip()
print("Source commit:", commit, flush=True)
run([sys.executable, str(repo / "tests/test_twosource_grouped_cv.py")], cwd=repo)

dllm = TEMP / f"twosource_cv_dllm_{stamp}"
download = ("from huggingface_hub import snapshot_download; "
            "snapshot_download('Efficient-Large-Model/Fast_dLLM_v2_1.5B', "
            f"local_dir={str(dllm)!r}, allow_patterns=['configuration.py','*.json',"
            "'*.safetensors','*.txt','*.jinja'])")
run([sys.executable, "-c", download], cwd=repo)
shutil.copy2(repo / "Fast_dLLM_v2_1_5B/modeling.py", dllm / "modeling.py")
if shutil.disk_usage(WORK).free < 1 * 1024**3:
    raise OSError("Need at least 1 GiB free under /kaggle/working for the report ZIP")

output = WORK / f"twosource_grouped_5fold_cv_gsm8k100_{stamp}"
command = [sys.executable, "-u", str(repo / "run_twosource_grouped_cv.py"),
    "--run_dir", str(RUN_DIR), "--dllm_dir", str(dllm), "--output_dir", str(output),
    "--folds", str(FOLDS), "--milestones", *map(str, MILESTONES),
    "--updates_per_question", str(UPDATES_PER_QUESTION),
    "--roots_per_question", str(ROOTS_PER_QUESTION), "--horizon", "3",
    "--batch_size", "8", "--device", "1", "--seed", "42"]
print("\nStarting true grouped 5-fold retraining + heldout validation; no verifier/drafter calls.",
      flush=True)
result = subprocess.run(command, cwd=repo, env=env, check=False)
archive = output.with_suffix(".zip")
if archive.is_file():
    print("RESULT ZIP (also visible under Kaggle Output):", archive, flush=True)
if result.returncode:
    print("Run stopped; preserve the partial result ZIP and output folder.", flush=True)
    raise subprocess.CalledProcessError(result.returncode, command)
with zipfile.ZipFile(archive) as zf:
    if zf.testzip() is not None:
        raise RuntimeError("CV report ZIP failed CRC validation")
    cv_summary = json.loads(zf.read("cv_summary.json"))
if cv_summary.get("status") != "complete" or cv_summary.get("completed_folds") != 5:
    raise RuntimeError("Not all five folds completed; keep the partial ZIP for diagnosis")
print(json.dumps({"status": cv_summary["status"],
                  "completed_folds": cv_summary["completed_folds"],
                  "pooled_oof": cv_summary["pooled_oof"]["groups"],
                  "learning_curve_mean": cv_summary["learning_curve_mean"]},
                 indent=2, ensure_ascii=False), flush=True)
from IPython.display import FileLink, display
display(FileLink(archive.name))
