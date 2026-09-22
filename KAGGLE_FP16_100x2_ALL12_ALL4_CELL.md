# Single Kaggle cell: two tests, all lengths 12 and all lengths 4

This runs two separate tests. Each test contains 100 MATH + 100 GSM8K
problems on two T4 GPUs:

1. dLLM block length = 12 and draft length = 12.
2. dLLM block length = 4 and draft length = 4.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!rm -rf /kaggle/working/failfasttesting /kaggle/working/world_model_raw_fp16_t4x2_all12 /kaggle/working/world_model_raw_fp16_t4x2_all4
!git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/failfasttesting

from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

%cd /kaggle/working/failfasttesting

# Test 1: dLLM length = 12, draft length = 12
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math gsm8k \
    --num_questions 100 \
    --block_size 12 \
    --small_block_size 12 \
    --spec_len 12 \
    --sweep_max_spec_len 12 \
    --sweep_incr_len 12 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --output_dir /kaggle/working/world_model_raw_fp16_t4x2_all12 \
    2>&1 | tee /kaggle/working/world_model_raw_fp16_t4x2_all12.log

# Test 2: dLLM length = 4, draft length = 4
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math gsm8k \
    --num_questions 100 \
    --block_size 4 \
    --small_block_size 4 \
    --spec_len 4 \
    --sweep_max_spec_len 4 \
    --sweep_incr_len 4 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --output_dir /kaggle/working/world_model_raw_fp16_t4x2_all4 \
    2>&1 | tee /kaggle/working/world_model_raw_fp16_t4x2_all4.log

import os
print("ALL12:", os.listdir("/kaggle/working/world_model_raw_fp16_t4x2_all12"))
print("ALL4:", os.listdir("/kaggle/working/world_model_raw_fp16_t4x2_all4"))
```

Outputs:

```text
/kaggle/working/world_model_raw_fp16_t4x2_all12/math_raw.zip
/kaggle/working/world_model_raw_fp16_t4x2_all12/gsm8k_raw.zip
/kaggle/working/world_model_raw_fp16_t4x2_all4/math_raw.zip
/kaggle/working/world_model_raw_fp16_t4x2_all4/gsm8k_raw.zip
```
