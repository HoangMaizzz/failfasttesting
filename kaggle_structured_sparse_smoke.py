"""Standalone Kaggle cell: dependencies, source, weights, tests and collection."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import zipfile

DATASETS = globals().get("DATASETS", ["gsm8k"])
NUM_QUESTIONS = globals().get("NUM_QUESTIONS", 3)
MAX_PROPOSAL_TOKENS = globals().get("MAX_PROPOSAL_TOKENS", 64)
ANCHORS_PER_QUESTION = globals().get("ANCHORS_PER_QUESTION", 4)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers==4.53.1", "accelerate", "datasets", "einops", "numpy", "pandas",
    "matplotlib", "scipy", "sentencepiece", "huggingface_hub"], check=True)

# Fresh Save Version runs keep model weights/source outside published Output.
repo = Path("/kaggle/temp/sparse_extend_world_model_repo")
repo.parent.mkdir(parents=True, exist_ok=True)
branch = "codex/sparse-extend-world-model"
env = os.environ.copy()
env["GIT_LFS_SKIP_SMUDGE"] = "1"
env["CUDA_VISIBLE_DEVICES"] = "0,1"
if (repo / ".git").exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only", "origin", branch], check=True, env=env)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", branch,
                    "https://github.com/HoangMaizzz/failfasttesting.git", str(repo)], check=True, env=env)
subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True)

import torch
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet before running this cell")
subprocess.run([sys.executable, str(repo / "tests/test_structured_sparse_collector.py")], cwd=repo, env=env, check=True)

from huggingface_hub import snapshot_download

# Reuse weights already downloaded in the user's active session when available.
old_dllm = Path("/kaggle/working/sparse_extend_world_model_repo/Fast_dLLM_v2_1_5B")
dllm = old_dllm if (old_dllm / "config.json").exists() and any(old_dllm.glob("*.safetensors")) else repo / "Fast_dLLM_v2_1_5B"
snapshot_download("Efficient-Large-Model/Fast_dLLM_v2_1.5B", local_dir=str(dllm),
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"])

run_dir = Path("/kaggle/working") / ("structured_sparse_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
for dataset in DATASETS:
    input_root = Path("/kaggle/input")
    candidates = [Path(f"/kaggle/input/datasets/yumesakihikari/speculativeworld/{dataset}_raw"),
                  Path(f"/kaggle/input/datasets/ainzkhail/specworld/{dataset}_raw"),
                  Path(f"/kaggle/input/specworld/{dataset}_raw")]
    # Kaggle mounts an attached dataset by its slug, which can differ from
    # the owner/slug URL. Discover raw folders under mounted dataset roots.
    if input_root.exists():
        for mount in input_root.iterdir():
            if not mount.is_dir():
                continue
            candidates.append(mount / f"{dataset}_raw")
            try:
                candidates.extend(child / f"{dataset}_raw" for child in mount.iterdir()
                                  if child.is_dir())
            except PermissionError:
                pass
    data = next((p for p in candidates if p.exists()), None)
    if data is None:
        mounted = [str(p) for p in input_root.iterdir()] if input_root.exists() else []
        raise FileNotFoundError(
            f"Cannot find {dataset}_raw in Kaggle inputs. Checked: "
            f"{[str(p) for p in candidates]}; mounted roots: {mounted}"
        )
    archives = [p for p in data.rglob("*.zip") if zipfile.is_zipfile(p)]
    if len(archives) > 1:
        raise RuntimeError(f"Ambiguous input ZIP files: {archives}")
    backbone = archives[0] if archives else data
    out = run_dir / dataset
    cmd = [sys.executable, "-u", str(repo / "structured_sparse_collector.py"),
        "--backbone_zip", str(backbone), "--dataset", dataset,
        "--num_questions", str(NUM_QUESTIONS), "--anchors_per_question", str(ANCHORS_PER_QUESTION),
        "--max_rounds_per_question", "0", "--extend_size", "8",
        "--max_proposal_tokens", str(MAX_PROPOSAL_TOKENS),
        "--max_refinement_steps", "3", "--branch_width", "2",
        "--min_expand_acceptance_ratio", "0.5", "--bad_probe_branches", "1",
        "--bad_refinement_steps", "2", "--physical_block_size", "32",
        "--small_block_size", "8", "--drafter_threshold", "0.3",
        "--target_device", "0", "--drafter_device", "1",
        "--target_gpu_memory_gib", "9",
        "--dllm_dir", str(dllm), "--output_dir", str(out),
        "--reference_cache_dir", str(Path("/kaggle/temp") /
            f"structured_reference_cache_{run_dir.name}")]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    print("DOWNLOAD:", out / f"{dataset}_structured_graph.zip", flush=True)
