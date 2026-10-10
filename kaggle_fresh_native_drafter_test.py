"""One-cell Kaggle launcher for fresh native R/E dLLM latent testing."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from IPython.display import FileLink, display


SOURCE_REF = globals().get("SOURCE_REF", "codex/factorized-wm-feasibility")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 100))
MAX_PROPOSAL_TOKENS = int(globals().get("MAX_PROPOSAL_TOKENS", 64))
MAX_REFINEMENT_STEPS = int(globals().get("MAX_REFINEMENT_STEPS", 3))
THRESHOLD = float(globals().get("THRESHOLD", 0.5))
UPDATES = int(globals().get("UPDATES", 240))
SEEDS = list(globals().get("SEEDS", [42, 43]))
EXPECTED_COMMIT = globals().get("EXPECTED_COMMIT")

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = TEMP / "fresh_native_drafter_latent_repo"
DLLM_DIR = REPO / "Fast_dLLM_v2_1_5B"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORK / f"fresh_native_latent_gsm8k_{NUM_QUESTIONS}q_{STAMP}"

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "WANDB_MODE": "disabled",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # This is a drafter-only test; the 1.5B model fits on one Kaggle T4.
    "CUDA_VISIBLE_DEVICES": "0",
})


def run_live(command, cwd=None, check=True):
    command = [str(part) for part in command]
    print("\n>>>", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if check and code:
        raise subprocess.CalledProcessError(code, command)
    return code


import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a Kaggle GPU accelerator. One T4 is sufficient; verifier is not loaded.")
print("GPU:", torch.cuda.get_device_name(0), "| GPUs visible:", torch.cuda.device_count(), flush=True)

# Only install Python packages; keep Kaggle's preinstalled torch/CUDA intact.
run_live([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
          "transformers==4.53.1", "accelerate>=1,<2", "datasets", "einops",
          "sentencepiece", "huggingface_hub", "numpy"])

TEMP.mkdir(parents=True, exist_ok=True)
if REPO.exists():
    shutil.rmtree(REPO)
run_live(["git", "clone", "--depth", "1", "--branch", SOURCE_REF,
          "https://github.com/HoangMaizzz/failfasttesting.git", str(REPO)])
commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Source commit:", commit, flush=True)
if EXPECTED_COMMIT and commit != EXPECTED_COMMIT:
    raise RuntimeError(f"Expected source commit {EXPECTED_COMMIT}, got {commit}")

run_live([sys.executable, "-m", "unittest",
          "tests.test_fresh_native_drafter_test",
          "tests.test_latent_sufficiency_audit",
          "tests.test_native_elysia_graph"], cwd=REPO)

from huggingface_hub import snapshot_download
DLLM_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading Fast-dLLM 1.5B weights; the GSM8K questions are loaded from Hugging Face.", flush=True)
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir=str(DLLM_DIR),
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

command = [
    sys.executable, "-u", str(REPO / "fresh_native_drafter_test.py"),
    "--dllm-dir", str(DLLM_DIR), "--out-dir", str(OUT_DIR),
    "--num-questions", str(NUM_QUESTIONS), "--threshold", str(THRESHOLD),
    "--extend-size", "8", "--max-proposal-tokens", str(MAX_PROPOSAL_TOKENS),
    "--max-refinement-steps", str(MAX_REFINEMENT_STEPS),
    "--physical-block-size", "32", "--small-block-size", "8",
    "--raw-top-k", "32", "--drafter-device", "0",
    "--updates", str(UPDATES), "--seeds", *map(str, SEEDS),
]
print("\nRunning fresh native dLLM collection + latent audit.", flush=True)
return_code = run_live(command, cwd=REPO, check=False)
result_zip = OUT_DIR.with_suffix(".zip")
if result_zip.exists():
    print("DOWNLOAD ZIP:", result_zip, flush=True)
    display(FileLink(str(result_zip)))
if return_code:
    raise subprocess.CalledProcessError(return_code, command)

print("\nComplete. No prior ZIP or Kaggle dataset upload is required.", flush=True)
