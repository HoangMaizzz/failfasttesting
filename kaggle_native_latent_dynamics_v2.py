"""Kaggle launcher for the native latent-dynamics V2 GSM8K test."""
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
UPDATES = int(globals().get("UPDATES", 300))

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = TEMP / "native_latent_dynamics_v2_repo"
DLLM_DIR = TEMP / "native_latent_dynamics_v2_fast_dllm"
OUT_DIR = WORK / (
    "native_latent_dynamics_v2_gsm8k_"
    + str(NUM_QUESTIONS) + "q_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

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
    command = [str(value) for value in command]
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
    raise RuntimeError("Select Kaggle GPU T4 x2; native dLLM capture uses GPU0 and latent training GPU1.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

# Keep Kaggle's preinstalled Torch/CUDA build intact.
run_live([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
          "transformers==4.53.1", "accelerate>=1,<2", "datasets",
          "einops", "sentencepiece", "huggingface_hub"])

TEMP.mkdir(parents=True, exist_ok=True)
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
run_live(["git", "init", str(REPO)], cwd=TEMP)
run_live(["git", "-C", str(REPO), "remote", "add", "origin",
          "https://github.com/HoangMaizzz/failfasttesting.git"], cwd=TEMP)
fetch_ref = EXPECTED_COMMIT or SOURCE_REF
run_live(["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", fetch_ref], cwd=TEMP)
run_live(["git", "-C", str(REPO), "checkout", "--detach", "FETCH_HEAD"], cwd=TEMP)
commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Source commit:", commit, flush=True)
if EXPECTED_COMMIT and commit != EXPECTED_COMMIT:
    raise RuntimeError(f"Expected commit {EXPECTED_COMMIT}, got {commit}")

run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
          "-p", "test_native_latent_dynamics_v2.py"], cwd=REPO)
run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
          "-p", "test_native_latent_assets_v2.py"], cwd=REPO)

from huggingface_hub import snapshot_download
DLLM_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading Fast-dLLM 1.5B weights; GSM8K test split will be fetched by the collector.", flush=True)
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir=str(DLLM_DIR),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "configuration.py"],
)
if not (DLLM_DIR / "configuration.py").is_file():
    raise FileNotFoundError(f"Missing custom Fast-dLLM config at {DLLM_DIR / 'configuration.py'}")

# The model directory is separate from the Git checkout. AutoModel loads its
# custom code from beside config.json, so install the pinned instrumented
# modeling.py there rather than downloading the uninstrumented Hub version.
run_live([
    sys.executable, "-c",
    "from pathlib import Path; import sys; "
    "from native_latent_assets_v2 import prepare_native_model_assets; "
    "digest = prepare_native_model_assets(Path(sys.argv[1]), Path(sys.argv[2])); "
    "print('[assets-ready] instrumented modeling.py sha256=' + digest, flush=True)",
    str(REPO), str(DLLM_DIR),
], cwd=REPO)

command = [
    sys.executable, "-u", str(REPO / "native_latent_dynamics_v2.py"),
    "--dllm-dir", str(DLLM_DIR), "--out-dir", str(OUT_DIR),
    "--num-questions", str(NUM_QUESTIONS), "--seed", str(SEED),
    "--split-seed", str(SEED), "--threshold", "0.5",
    "--extend-size", "8", "--max-proposal-tokens", "64",
    "--max-refinement-steps", "3", "--horizon", "3",
    "--latent-dim", "128", "--model-layers", "3",
    "--cut-layer", "24", "--suffix-layers", "4",
    "--updates", str(UPDATES), "--batch-size", "12",
    "--drafter-device", "0", "--train-device", "cuda:1",
]
print("\nRun V2: fresh native dLLM capture + latent feature-dynamics training; no verifier/token target.", flush=True)
env["WM_V2_SOURCE_COMMIT"] = commit
return_code = run_live(command, cwd=REPO, check=False)
result_zip = OUT_DIR.with_suffix(".zip")
if result_zip.exists():
    print("\nDOWNLOAD ZIP:", result_zip, flush=True)
    display(FileLink(str(result_zip)))
if return_code:
    raise subprocess.CalledProcessError(return_code, command)
print("\nComplete. The final ZIP contains sanitized latent/structural features, ablations, and H1-H3 metrics.", flush=True)
