# Colab FP16 run on two GPUs

Enable a Colab runtime with two GPUs, then run the following cells. GPU 0 holds
the 7B verifier and GPU 1 holds the 1.5B Fast-dLLM drafter. Each dataset is
archived immediately after its 100 questions finish; the unpacked raw shards
and staging CSV are then removed to keep the runtime disk bounded.

```python
!pip install -q --no-cache-dir "transformers==4.53.1" "bitsandbytes>=0.46.1" datasets accelerate einops tqdm numpy pandas matplotlib sentencepiece scipy
!git clone -b codex/colab-fp16-math-gsm8k-zip-20260918 https://github.com/HoangMaizzz/failfasttesting.git /content/failfasttesting
```

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    local_dir="/content/failfasttesting/Fast_dLLM_v2_1_5B",
    allow_patterns=["configuration.py", "*.json", "*.safetensors", "*.txt", "*.jinja"],
)
```

```python
%cd /content/failfasttesting
!CUDA_VISIBLE_DEVICES=0,1 python full_refinement_oracle_collector.py \
    --datasets math gsm8k \
    --num_questions 100 \
    --target_quantization none \
    --target_device 0 \
    --drafter_device 1 \
    --dllm_dir /content/failfasttesting/Fast_dLLM_v2_1_5B \
    --shard_rows 128 \
    --stream_raw \
    --archive_each_dataset \
    --remove_raw_after_archive \
    --drop_staging \
    --output_dir /content/world_model_fp16_math_gsm8k \
    2>&1 | tee /content/world_model_fp16_math_gsm8k.log
```

The downloadable files are `math_raw.zip` and `gsm8k_raw.zip` in the output
directory. Each archive contains its NPZ shards, `index.jsonl`, and the small
benchmark result CSV.
