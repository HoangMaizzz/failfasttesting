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
DRAFTER_THRESHOLD = globals().get("DRAFTER_THRESHOLD", 0.5)
VERIFIER_MODE = globals().get("VERIFIER_MODE", "prefix_kv_calibration")
DRAFTER_KV_MODE = globals().get("DRAFTER_KV_MODE", "none")
UNMASK_BACKEND = globals().get("UNMASK_BACKEND", "explicit_replay")
EXPECTED_COMMIT = globals().get("EXPECTED_COMMIT")


def find_dataset_input(dataset):
    input_root = Path("/kaggle/input")
    expected_name = f"{dataset}_raw"
    known = [
        Path(f"/kaggle/input/datasets/yumesakihikari/speculativeworldmodel/{expected_name}"),
        Path(f"/kaggle/input/datasets/yumesakihikari/speculativeworld/{expected_name}"),
        Path(f"/kaggle/input/datasets/ainzkhail/specworld/{expected_name}"),
        Path(f"/kaggle/input/specworld/{expected_name}"),
    ]
    for candidate in known:
        if candidate.exists():
            return candidate

    # Kaggle may mount inputs as /kaggle/input/datasets/<owner>/<slug>/...
    # or directly by slug. Search directory names only, with a shallow bound.
    frontier = [(input_root, 0)]
    discovered = []
    while frontier:
        parent, depth = frontier.pop(0)
        if depth >= 6 or not parent.is_dir():
            continue
        try:
            children = list(parent.iterdir())
        except PermissionError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            if child.name == expected_name:
                discovered.append(child)
            else:
                frontier.append((child, depth + 1))
    if discovered:
        return sorted(discovered, key=lambda p: (len(p.parts), str(p)))[0]
    mounted = [str(p) for p in input_root.iterdir()] if input_root.exists() else []
    raise FileNotFoundError(
        f"Cannot find {expected_name} in Kaggle inputs; mounted roots: {mounted}"
    )


# Fail before pip/model downloads when a requested input is not mounted.
DATASET_INPUT_DIRS = {dataset: find_dataset_input(dataset) for dataset in DATASETS}
print("Dataset inputs:", {k: str(v) for k, v in DATASET_INPUT_DIRS.items()}, flush=True)

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
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
print("Source commit:", head, flush=True)
if EXPECTED_COMMIT and head != EXPECTED_COMMIT:
    raise RuntimeError(f"Expected source commit {EXPECTED_COMMIT}, got {head}")

import torch
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet before running this cell")
subprocess.run([sys.executable, str(repo / "tests/test_structured_sparse_collector.py")], cwd=repo, env=env, check=True)
if UNMASK_BACKEND == "native_elysia":
    subprocess.run([sys.executable, str(repo / "tests/test_native_elysia_graph.py")],
                   cwd=repo, env=env, check=True)
if DRAFTER_KV_MODE == "stable_block_prefix":
    subprocess.run([sys.executable, str(repo / "tests/test_structured_native_forward.py")],
                   cwd=repo, env=env, check=True)

from huggingface_hub import snapshot_download

# Reuse weights already downloaded in the user's active session when available.
old_dllm = Path("/kaggle/working/sparse_extend_world_model_repo/Fast_dLLM_v2_1_5B")
dllm = old_dllm if (old_dllm / "config.json").exists() and any(old_dllm.glob("*.safetensors")) else repo / "Fast_dLLM_v2_1_5B"
snapshot_download("Efficient-Large-Model/Fast_dLLM_v2_1.5B", local_dir=str(dllm),
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"])

run_dir = Path("/kaggle/working") / ("structured_sparse_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
for dataset in DATASETS:
    data = DATASET_INPUT_DIRS[dataset]
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
        "--min_expand_acceptance_ratio", "0.0", "--bad_probe_branches", "1",
        "--bad_refinement_steps", "2", "--physical_block_size", "32",
        "--small_block_size", "8", "--drafter_threshold", str(DRAFTER_THRESHOLD),
        "--verifier_mode", VERIFIER_MODE,
        "--drafter_kv_mode", DRAFTER_KV_MODE,
        "--unmask_backend", UNMASK_BACKEND,
        "--target_device", "0", "--drafter_device", "1",
        "--target_gpu_memory_gib", "9",
        "--dllm_dir", str(dllm), "--output_dir", str(out),
        "--reference_cache_dir", str(Path("/kaggle/temp") /
            f"structured_reference_cache_{run_dir.name}")]
    print("Running:", " ".join(cmd), flush=True)
    run = subprocess.run(cmd, cwd=repo, env=env, check=False)
    archive = out / f"{dataset}_structured_graph.zip"
    if run.returncode:
        print("Collector failed; partial archive, if written:", archive, flush=True)
        raise subprocess.CalledProcessError(run.returncode, cmd)
    with zipfile.ZipFile(archive) as z:
        import json
        manifest = json.loads(z.read("graph_manifest.json"))
        if manifest["status"] != "complete" or z.testzip() is not None:
            raise RuntimeError(f"Incomplete or corrupt archive: {archive}")
        if UNMASK_BACKEND == "native_elysia" and manifest["schema_version"] != "structured_sparse_native_elysia_v1":
            raise RuntimeError("Unexpected collector schema; refusing a misleading smoke result")
        print("Native nodes:", manifest["nodes"], "edges:", manifest["edges"], flush=True)
    print("DOWNLOAD:", archive, flush=True)
