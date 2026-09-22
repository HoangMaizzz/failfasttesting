# Single Kaggle cell: 30 MATH level-5 problems

Paste this into one Kaggle notebook cell with two T4 GPUs enabled. The level-5
indices are selected from the same `HuggingFaceH4/MATH-500` test split that
`failfast.py` loads, so the IDs passed to the collector remain aligned.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy huggingface_hub
!rm -rf /kaggle/working/failfasttesting /kaggle/working/world_model_raw_fp16_t4x2_math_level5_30 /kaggle/working/math_level5_ids.txt
!git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git /kaggle/working/failfasttesting

from datasets import load_dataset

math = load_dataset("HuggingFaceH4/MATH-500", split="test")
if "level" not in math.column_names:
    raise RuntimeError(f"MATH dataset has no level column: {math.column_names}")

def is_level5(value):
    text = str(value).strip().lower().replace("level", "").strip()
    return text == "5"

level5_ids = [i for i, value in enumerate(math["level"]) if is_level5(value)]
if len(level5_ids) < 30:
    raise RuntimeError(f"Only {len(level5_ids)} level-5 problems found")

selected_ids = level5_ids[:30]
with open("/kaggle/working/math_level5_ids.txt", "w") as f:
    f.write("\n".join(map(str, selected_ids)) + "\n")

print("Selected MATH level-5 problem IDs:", selected_ids)
print("Count:", len(selected_ids))

from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)

%cd /kaggle/working/failfasttesting
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math \
    --num_questions 30 \
    --problem_ids_file /kaggle/working/math_level5_ids.txt \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /kaggle/working/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --output_dir /kaggle/working/world_model_raw_fp16_t4x2_math_level5_30 \
    2>&1 | tee /kaggle/working/world_model_raw_fp16_t4x2_math_level5_30.log

import os
print(os.listdir("/kaggle/working/world_model_raw_fp16_t4x2_math_level5_30"))
```

Output ZIP:

```text
/kaggle/working/world_model_raw_fp16_t4x2_math_level5_30/math_raw.zip
```
