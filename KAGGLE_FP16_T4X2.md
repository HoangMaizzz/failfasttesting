# Kaggle T4 x2 FP16 run

This collector keeps the verifier on GPU 0 and the Fast-dLLM drafter on GPU 1.
It uses FP16 (`target_quantization=none`) and preserves the raw-state NPZ schema
used by the existing MATH/GSM8K collection: proposal IDs/probabilities/mask,
five hidden-state layers, and top-32 token IDs/logits.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy
!git clone -b codex/kaggle-fp16-t4x2-20260917 https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/failfasttesting
```

Download the dLLM weights into the code directory without replacing its patched
Python implementation, then run the three datasets:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)
```

```python
%cd /kaggle/working/failfasttesting
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math gsm8k humaneval \
    --num_questions 100 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --drop_staging \
    --output_dir /kaggle/working/world_model_raw_fp16_t4x2 \
    2>&1 | tee /kaggle/working/world_model_raw_fp16_t4x2.log
```
