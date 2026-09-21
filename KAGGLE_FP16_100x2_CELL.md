# Single Kaggle cell: 100 MATH + 100 GSM8K

Paste the following into one Kaggle notebook cell with two T4 GPUs enabled.
The collector writes each dataset directly to compressed shards, creates its ZIP
immediately after that dataset finishes, and removes the unpacked shards/staging
CSV to avoid the previous disk spike.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!rm -rf /kaggle/working/failfasttesting /kaggle/working/world_model_raw_fp16_t4x2
!git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/failfasttesting

from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

%cd /kaggle/working/failfasttesting
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math gsm8k \
    --num_questions 100 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --output_dir /kaggle/working/world_model_raw_fp16_t4x2 \
    2>&1 | tee /kaggle/working/world_model_raw_fp16_t4x2.log

import os
print(os.listdir('/kaggle/working/world_model_raw_fp16_t4x2'))
```

The final files are:

```text
/kaggle/working/world_model_raw_fp16_t4x2/math_raw.zip
/kaggle/working/world_model_raw_fp16_t4x2/gsm8k_raw.zip
```

The new ZIP schema includes `proposal_mask_before_fill`, the proposal before
and after counterfactual fill, refinement counters, prefix token IDs, explicit
`hidden_state_stage="native_pre_counterfactual_fill"`, terminal reasons, and
the one-step latency-per-output-token fields. `actual_action_taken` remains null
because the collection trajectory is intentionally forced to continue.
