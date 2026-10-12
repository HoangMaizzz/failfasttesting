"""Kaggle launcher for the score-valued native latent-drafter dynamics test."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from IPython.display import FileLink, display


SOURCE_REF = globals().get("SOURCE_REF", "codex/factorized-wm-feasibility")
EXPECTED_COMMIT = globals().get("EXPECTED_COMMIT")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 100))
SEED = int(globals().get("SEED", 20261012))
UPDATES = int(globals().get("UPDATES", 300))

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
REPO = Path(tempfile.mkdtemp(prefix="score_latent_drafter_repo_", dir=TEMP))
DLLM_DIR = TEMP / "score_latent_drafter_fast_dllm"
OUT_DIR = WORK / (
    "score_latent_drafter_gsm8k_" + str(NUM_QUESTIONS) + "q_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "WANDB_MODE": "disabled",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HOME": str(TEMP / "score_latent_drafter_hf_cache"),
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
    raise RuntimeError("Chọn GPU T4 x2; GPU0 chạy Fast-dLLM capture, GPU1 train latent drafter.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
if NUM_QUESTIONS < 20:
    raise ValueError("Cần ít nhất 20 câu để chia train/validation/test theo question 60/20/20.")

# Giữ nguyên PyTorch/CUDA Kaggle đang cài.
run_live([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
          "transformers==4.53.1", "accelerate>=1,<2", "datasets",
          "einops", "sentencepiece", "huggingface_hub", "safetensors"])

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

for pattern in ("test_score_latent_drafter.py", "test_native_latent_dynamics_v2.py",
                "test_native_latent_assets_v2.py", "test_native_latent_generator_rules_v2.py"):
    run_live([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests"),
              "-p", pattern], cwd=REPO)

from huggingface_hub import snapshot_download
DLLM_DIR.mkdir(parents=True, exist_ok=True)
print("Downloading public Fast-dLLM 1.5B weights; no Kaggle secret or dataset ZIP is needed.", flush=True)
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir=str(DLLM_DIR),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py"],
)
if not (DLLM_DIR / "configuration.py").is_file():
    raise FileNotFoundError(f"Missing Fast-dLLM configuration.py in {DLLM_DIR}")

run_live([
    sys.executable, "-c",
    "from pathlib import Path; import sys; "
    "from native_latent_assets_v2 import prepare_native_model_assets; "
    "digest = prepare_native_model_assets(Path(sys.argv[1]), Path(sys.argv[2])); "
    "print('[native-assets-ready] modeling.py sha256=' + digest, flush=True)",
    str(REPO), str(DLLM_DIR),
], cwd=REPO)

command = [
    sys.executable, "-u", str(REPO / "score_latent_drafter.py"),
    "--dllm-dir", str(DLLM_DIR), "--out-dir", str(OUT_DIR),
    "--num-questions", str(NUM_QUESTIONS), "--seed", str(SEED),
    "--split-seed", str(SEED), "--threshold", "0.5",
    "--extend-size", "8", "--max-proposal-tokens", "64",
    "--max-refinement-steps", "3", "--horizon", "3",
    "--latent-dim", "64", "--encoder-layers", "2",
    "--transition-layers", "2", "--updates", str(UPDATES),
    "--batch-size", "8", "--drafter-device", "0", "--train-device", "cuda:1",
]
env["WM_SCORE_SOURCE_COMMIT"] = commit
print("\nFresh native GSM8K capture + score-latent R/E learning; verifier is not loaded.", flush=True)
return_code = run_live(command, cwd=REPO, check=False)
result_zip = OUT_DIR.with_suffix(".zip")
if result_zip.exists():
    print("\nDOWNLOAD ZIP:", result_zip, flush=True)
    display(FileLink(str(result_zip)))
if return_code:
    raise subprocess.CalledProcessError(return_code, command)
print("\nComplete: report includes H1-H3 rollout metrics and three input-feature ablations.", flush=True)
