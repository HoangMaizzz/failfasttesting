"""Colab entry point: INT8 verifier, FP16 drafter, 100 old IDs per dataset."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["nvidia-smi"], check=True)
    model = root / "Fast_dLLM_v2_1.5B"
    snapshot_download("ruipeterpan/Fast_dLLM_v2_1.5B", local_dir=str(model),
                      ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"])
    subprocess.run([
        sys.executable, "-u", "run_u1_multiwindow_int8.py",
        "--datasets", "math", "gsm8k", "humaneval",
        "--num_questions", "100", "--id_offset", "25",
        "--dllm_dir", str(model),
        "--output_dir", str(root / "outputs_u1_multiwindow_int8_test100"),
        "--resume",
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
