"""Kaggle one-cell launcher for drafter-only latent feature-dynamics test."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from IPython.display import FileLink, display


SOURCE_REF = globals().get("SOURCE_REF", "codex/factorized-wm-feasibility")
EXPECTED_COMMIT = globals().get("EXPECTED_COMMIT")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 100))
SEED = int(globals().get("SEED", 42))
UPDATES = int(globals().get("UPDATES", 160))
CURVE_UPDATES = int(globals().get("CURVE_UPDATES", 80))
SEEDS = list(globals().get("SEEDS", [42, 43]))

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = TEMP / "drafter_feature_dynamics_repo"
DLLM_DIR = REPO / "Fast_dLLM_v2_1_5B"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORK / f"drafter_feature_dynamics_gsm8k_{NUM_QUESTIONS}q_{STAMP}"

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "WANDB_MODE": "disabled",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Select Kaggle GPU T4 x2. Drafter capture uses GPU 0; latent training uses GPU 1.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

# Keep Kaggle's existing Torch/CUDA build intact; install only Python dependencies.
run_live([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
          "transformers==4.53.1", "accelerate>=1,<2", "datasets", "einops",
          "sentencepiece", "huggingface_hub", "scikit-learn"])

TEMP.mkdir(parents=True, exist_ok=True)
if REPO.exists():
    shutil.rmtree(REPO)
run_live(["git", "clone", "--depth", "1", "--branch", SOURCE_REF,
          "https://github.com/HoangMaizzz/failfasttesting.git", str(REPO)], cwd=TEMP)
commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Source commit:", commit, flush=True)
if EXPECTED_COMMIT and commit != EXPECTED_COMMIT:
    raise RuntimeError(f"Expected source commit {EXPECTED_COMMIT}, got {commit}")

# Use unittest discovery from the repository's explicit tests directory. This
# avoids Kaggle's third-party `tests` package shadowing the local test modules.
run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
          "-p", "test_drafter_feature_dynamics_test.py"], cwd=REPO)
run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
          "-p", "test_fresh_native_drafter_test.py"], cwd=REPO)
run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
          "-p", "test_native_elysia_graph.py"], cwd=REPO)

from huggingface_hub import snapshot_download
DLLM_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading Fast-dLLM 1.5B weights; GSM8K test split is fetched by the collector.", flush=True)
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir=str(DLLM_DIR),
    # Keep the repository's instrumented modeling.py; fetch model weights/config only.
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
)

command = [
    sys.executable, "-u", str(REPO / "drafter_feature_dynamics_test.py"),
    "--dllm-dir", str(DLLM_DIR), "--out-dir", str(OUT_DIR),
    "--num-questions", str(NUM_QUESTIONS), "--seed", str(SEED),
    "--split-seed", str(SEED), "--threshold", "0.5",
    "--extend-size", "8", "--max-proposal-tokens", "64",
    "--max-refinement-steps", "3", "--updates", str(UPDATES),
    "--curve-updates", str(CURVE_UPDATES), "--batch-size", "12",
    "--seeds", *map(str, SEEDS), "--drafter-device", "0",
    "--train-device", "cuda:1",
]
print("\nRunning fresh native R/E feature-dynamics training. Verifier is not loaded.", flush=True)
return_code = run_live(command, cwd=REPO, check=False)
result_zip = OUT_DIR.with_suffix(".zip")
if result_zip.exists():
    print("\nDOWNLOAD ZIP:", result_zip, flush=True)
    display(FileLink(str(result_zip)))
if return_code:
    raise subprocess.CalledProcessError(return_code, command)
print("\nComplete. No prior ZIP or Kaggle dataset upload is required.", flush=True)
