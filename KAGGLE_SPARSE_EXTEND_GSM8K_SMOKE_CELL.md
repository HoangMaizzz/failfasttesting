# Kaggle smoke test: 3 GSM8K questions, proposals up to 64 tokens

Add the uploaded `gsm8k_raw` Kaggle dataset as an input. The cell clones the
source-only branch; the ZIP is read from `/kaggle/input`, not Git LFS.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!git clone --depth 1 -b codex/sparse-extend-world-model https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/sparse_extend_world_model_repo

from pathlib import Path
import os, subprocess, sys, zipfile
from huggingface_hub import snapshot_download

repo = Path("/kaggle/working/sparse_extend_world_model_repo")
data_dir = Path("/kaggle/input/datasets/ainzkhail/specworld/gsm8k_raw")
if not data_dir.exists():
    raise FileNotFoundError(f"Kaggle input folder not found: {data_dir}")
archives = [p for p in data_dir.rglob("*.zip") if zipfile.is_zipfile(p)]
if len(archives) > 1:
    raise RuntimeError(f"Expected at most one GSM8K ZIP under {data_dir}; found {archives}")
backbone_input = archives[0] if archives else data_dir
print("Using backbone input:", backbone_input)

dllm_dir = repo / "Fast_dLLM_v2_1_5B"
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir=str(dllm_dir),
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

cmd = [
    sys.executable, str(repo / "sparse_extend_world_model_collector.py"),
    "--backbone_zip", str(backbone_input), "--dataset", "gsm8k",
    "--num_questions", "3", "--max_rounds_per_question", "1",
    "--root_boundaries", "4", "--extend_size", "8",
    "--max_proposal_tokens", "64", "--max_unmask_passes", "4",
    "--branch_width", "4", "--physical_block_size", "32",
    "--small_block_size", "8", "--target_device", "0",
    "--drafter_device", "1", "--dllm_dir", str(dllm_dir),
    "--output_dir", "/kaggle/working/sparse_extend_gsm8k_64_smoke",
]
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1"
subprocess.run(cmd, cwd=repo, env=env, check=True)

out = Path("/kaggle/working/sparse_extend_gsm8k_64_smoke")
print("Manifest:", out / "graph_manifest.json")
print("Result ZIP:", out / "gsm8k_extend_graph.zip")
```

Verifier latency for proposals longer than 8 tokens is explicitly recorded as
a length-scaled estimate from the existing archive's measured 8-token timings.
```
