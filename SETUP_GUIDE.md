# 🚀 FailFast Complete Setup & Test Guide

## One-Command Test (Recommended)

```bash
cd "d:\LLM lab\failfast"
python setup_and_test.py
```

**This will automatically:**
1. ✅ Check Python & GPU
2. ✅ Clone/setup Fast_dLLM_v2_1.5B
3. ✅ Apply Fix 2 (RoPE KeyError patch)
4. ✅ Run 3 tests on GSM8K (3 samples each)

---

## ⚙️ Configuration

Before running, edit these paths in `setup_and_test.py`:

```python
WORKSPACE = Path("d:/LLM lab/failfast").resolve()
DLLM_DIR = Path("/data2/USERNAME/Fast_dLLM_v2_1.5B").resolve()  # ← Update USERNAME
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
```

Also verify models are available:
```python
VERIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFTER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DLLM_REPO = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
```

---

## 📊 What Gets Tested

| Mode | Drafter | Verifier | Description |
|------|---------|----------|-------------|
| **verifier_ar** | None | 7B | Baseline (no speculation) |
| **ar_ar** | 1.5B | 7B | AR drafting + verification |
| **dllm_ar** | dLLM | 7B | FailFast dynamic frequency |

Each test runs on **3 GSM8K samples** with:
- Max tokens: 256
- Speculation length: 10
- Timing tracked: `draft_time_ms` + `verify_time_ms`

---

## 📈 Expected Output

For each test:
```
Step 5: Run 3 FailFast Tests on GSM8K
════════════════════════════════════════════════════════════════════════════════

TEST: Verifier-only AR baseline (7B)
════════════════════════════════════════════════════════════════════════════════
>>> python failfast.py --mode verifier_ar ...
[Loading model Qwen/Qwen2.5-7B-Instruct]
[Processing 3 questions from gsm8k]
[Draft time: X.XXms | Verify time: X.XXms]
...
✓ verifier_ar PASSED

TEST: AR-AR mode (1.5B drafter + 7B verifier)
...
✓ ar_ar PASSED

TEST: dLLM-AR mode with FailFast
...
✓ dllm_ar PASSED

SUMMARY: Test Results
  verifier_ar     : PASSED
  ar_ar           : PASSED
  dllm_ar         : PASSED
✓ All tests passed!
```

---

## 🔍 Manual Step-by-Step (If needed)

### Step 1: Check Setup
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

### Step 2: Clone dLLM (if needed)
```bash
huggingface-cli download Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --local-dir /data2/USERNAME/Fast_dLLM_v2_1.5B
```

### Step 3: Apply Fix 2 Manually
Edit `/data2/USERNAME/Fast_dLLM_v2_1.5B/modeling.py`:

**Find line:**
```python
self.rope_init_fn = ROPE_INIT_FUNCTIONS[config.rope_scaling["type"]]
```

**Replace with:**
```python
self.rope_init_fn = None
inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

### Step 4: Test Each Mode
```bash
# Test 1: Baseline
python failfast.py --mode verifier_ar --dataset_name gsm8k --num_questions 3 --overwrite

# Test 2: AR-AR
python failfast.py --mode ar_ar --dataset_name gsm8k --num_questions 3 --overwrite

# Test 3: dLLM-AR
python failfast.py --mode dllm_ar --dataset_name gsm8k --num_questions 3 --overwrite
```

---

## 📋 Troubleshooting

### OOM Error on L4
- Reduce `MAX_NEW_TOKENS` (default 256)
- Use `device_map={"": 0}` (already set)
- Enable bfloat16 (already enabled)

### RoPE KeyError (Fast_dLLM issue)
- Apply Fix 2 manually if auto-patch fails
- Or skip dllm_ar mode and test verifier_ar + ar_ar only

### Model Download Timeout
- Pre-download models manually before running tests
- Use `CUDA_VISIBLE_DEVICES=0` to target specific GPU

### Wrong Python/Environment
- Use: `python -c "import sys; print(sys.executable)"`
- Ensure transformers >= 4.40.0

---

## 🎯 Quick Verification

After running, check output files:
```bash
ls -lah /data2/USERNAME/failfast_results/
```

Should contain:
- `gsm8k_verifier_ar_*.pkl` - Baseline results
- `gsm8k_ar_ar_*.pkl` - AR-AR results  
- `gsm8k_dllm_ar_*.pkl` - dLLM-AR results
- Acceptance rates, timing metrics in each

---

## 🏁 Ready?

```bash
cd "d:\LLM lab\failfast"
python setup_and_test.py
```

**Runtime:** ~30 minutes (3 modes × ~10min each on L4)

Good luck! 🚀
