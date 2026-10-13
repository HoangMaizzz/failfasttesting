"""Kaggle launcher for the no-recapture score-latent follow-up audit."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from IPython.display import FileLink, display


SOURCE_REF = globals().get("SOURCE_REF", "codex/factorized-wm-feasibility")
INPUT_DIR = Path(globals().get(
    "INPUT_DIR", "/kaggle/input/datasets/ainzkhail/default1"))
DEVICE = globals().get("DEVICE", "auto")
SAVE_POSITION_PREDICTIONS = bool(globals().get("SAVE_POSITION_PREDICTIONS", True))

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
REPO = Path(tempfile.mkdtemp(prefix="score_latent_audit_repo_", dir=TEMP))
OUT_DIR = WORK / ("score_latent_drafter_audit_" +
                  datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1", "USE_TF": "0", "USE_FLAX": "0",
    "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1",
})


def run_live(command, cwd=None):
    command = [str(value) for value in command]
    print("\n>>>", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if not INPUT_DIR.is_dir():
    raise FileNotFoundError(
        f"Input folder not mounted: {INPUT_DIR}. Add the existing score-latent result as Kaggle Input.")

run_live([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
          "transformers==4.53.1", "datasets", "accelerate>=1,<2", "einops",
          "sentencepiece", "huggingface_hub", "safetensors"])

run_live(["git", "clone", "--depth", "1", "--branch", SOURCE_REF,
          "https://github.com/HoangMaizzz/failfasttesting.git", str(REPO)], cwd=TEMP)
commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
env["WM_SCORE_AUDIT_COMMIT"] = commit
print("Audit code commit:", commit, flush=True)
print("Mounted input:", INPUT_DIR, flush=True)

command = [sys.executable, "-u", str(REPO / "score_latent_drafter_audit.py"),
           "--input-dir", str(INPUT_DIR), "--out-dir", str(OUT_DIR),
           "--device", DEVICE, "--batch-size", "32", "--bootstrap-reps", "500"]
if SAVE_POSITION_PREDICTIONS:
    command.append("--save-predictions")
run_live(command, cwd=REPO)

result_zip = OUT_DIR.with_suffix(".zip")
print("\nDOWNLOAD ZIP:", result_zip, flush=True)
display(FileLink(str(result_zip)))
print("No dLLM recapture and no verifier run were performed.", flush=True)
