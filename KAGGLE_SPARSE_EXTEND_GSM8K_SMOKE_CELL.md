# Kaggle smoke test: 3 GSM8K questions, up to 4 branches

Run this as a single Kaggle notebook cell. It loads the GSM8K raw archive from
Git LFS and tests two Extend levels (proposal lengths 16 and 24), with at most
four active branches and four unmask snapshots per Extend. The production
collector supports proposals through 64 tokens.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!git lfs version || (apt-get update -qq && apt-get install -y -qq git-lfs)
%cd /kaggle/working
!GIT_LFS_SKIP_SMUDGE=1 git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git sparse_extend_failfasttesting
%cd /kaggle/working/sparse_extend_failfasttesting
!git lfs install
!git lfs pull --include="reference_data/gsm8k_raw.zip"

from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/sparse_extend_failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

!CUDA_VISIBLE_DEVICES=0,1 python sparse_extend_world_model_collector.py \
    --backbone_zip /kaggle/working/sparse_extend_failfasttesting/reference_data/gsm8k_raw.zip \
    --dataset gsm8k \
    --num_questions 3 \
    --max_rounds_per_question 1 \
    --root_boundaries 4 \
    --extend_size 8 \
    --max_proposal_tokens 24 \
    --max_unmask_passes 4 \
    --branch_width 4 \
    --physical_block_size 32 \
    --small_block_size 8 \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/sparse_extend_failfasttesting/Fast_dLLM_v2_1_5B \
    --output_dir /kaggle/working/sparse_extend_gsm8k_smoke \
    2>&1 | tee /kaggle/working/sparse_extend_gsm8k_smoke.log

from pathlib import Path
out = Path("/kaggle/working/sparse_extend_gsm8k_smoke")
print("MANIFEST:", out / "graph_manifest.json", (out / "graph_manifest.json").exists())
print("ZIP:", out / "gsm8k_extend_graph.zip", (out / "gsm8k_extend_graph.zip").exists())
print("LOG:", Path("/kaggle/working/sparse_extend_gsm8k_smoke.log"))
```

The collector reads the archived verifier timings at proposal length 8. For
longer proposals it records an explicit length-scaled verifier-time estimate;
it does not claim those per-node verifier times were measured. Greedy verifier
continuations are generated once per selected round and reused for all candidate
labels.
