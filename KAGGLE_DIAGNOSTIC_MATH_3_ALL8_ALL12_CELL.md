# Kaggle diagnostic: 3 MATH problems with all-8 and all-12 lengths

This cell runs two short diagnostic jobs. It keeps progress bars and INFO
logging visible, uses only 3 MATH problems per configuration, and limits each
generation to 256 output tokens so a bad configuration becomes observable
quickly instead of running for many hours.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!rm -rf /kaggle/working/failfasttesting /kaggle/working/world_model_diag_math_all8 /kaggle/working/world_model_diag_math_all12
!git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/failfasttesting

from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

%cd /kaggle/working/failfasttesting

# Diagnostic 1: all lengths = 8
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math \
    --num_questions 3 \
    --block_size 8 \
    --small_block_size 8 \
    --spec_len 8 \
    --sweep_max_spec_len 8 \
    --sweep_incr_len 8 \
    --max_new_tokens 256 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 32 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --show_progress \
    --log_level INFO \
    --output_dir /kaggle/working/world_model_diag_math_all8 \
    2>&1 | tee /kaggle/working/world_model_diag_math_all8.log

# Diagnostic 2: all lengths = 12
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math \
    --num_questions 3 \
    --block_size 12 \
    --small_block_size 12 \
    --spec_len 12 \
    --sweep_max_spec_len 12 \
    --sweep_incr_len 12 \
    --max_new_tokens 256 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 32 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --show_progress \
    --log_level INFO \
    --output_dir /kaggle/working/world_model_diag_math_all12 \
    2>&1 | tee /kaggle/working/world_model_diag_math_all12.log

import os
print("ALL8:", os.listdir("/kaggle/working/world_model_diag_math_all8"))
print("ALL12:", os.listdir("/kaggle/working/world_model_diag_math_all12"))
```

Expected archives:

```text
/kaggle/working/world_model_diag_math_all8/math_raw.zip
/kaggle/working/world_model_diag_math_all12/math_raw.zip
```
