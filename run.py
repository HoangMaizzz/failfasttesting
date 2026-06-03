#!/usr/bin/env python
"""
Run script for FailFast speculative decoding benchmark on gsm8k dataset.

I am GitHub Copilot using Claude Haiku 4.5.

This script runs 3 test modes with a small sample (3 questions) on GSM8K dataset:
1. verifier_ar  - Baseline: only verifier AR (7B)
2. ar_ar        - AR drafter (1.5B) + AR verifier (7B)
3. dllm_ar      - dLLM drafter + AR verifier (7B) with FailFast dynamic frequency

All models use bfloat16 precision and device_map={"": 0} for L4 GPU.
"""

import subprocess
import sys

def run_command(cmd):
    """Execute a command and print output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, cwd="d:\\LLM lab\\failfast")
    return result.returncode

def main():
    # ==========================================
    # Configuration (easily modifiable)
    # ==========================================
    
    DATASET = "gsm8k"
    NUM_QUESTIONS = 3  # Small sample for testing
    MAX_NEW_TOKENS = 256  # Reduced for faster testing
    SPEC_LEN = 10  # Default speculation length
    
    VERIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DRAFTER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    
    DRAFTER_THRESHOLDS = [0.05]  # dLLM confidence threshold
    SWEEP_LOWCONF_THRESHOLD = [0.45]  # FailFast confidence threshold (tau)
    SWEEP_MAX_SPEC_LEN = [60]  # N_max in FailFast Alg 1
    SWEEP_INCR_LEN = [10]  # N in FailFast Alg 1
    
    OUTPUT_DIR = "/data2/USERNAME/failfast_results"
    
    # ==========================================
    # Base command template
    # ==========================================
    base_cmd = [
        sys.executable, "failfast.py",
        f"--dataset_name", DATASET,
        f"--num_questions", str(NUM_QUESTIONS),
        f"--max_new_tokens", str(MAX_NEW_TOKENS),
        f"--spec_len", str(SPEC_LEN),
        f"--output_dir", OUTPUT_DIR,
        f"--verifier_model_name", VERIFIER_MODEL,
        f"--drafter_model_name", DRAFTER_MODEL,
        "--overwrite",
        "--log_level", "INFO",
    ]
    
    # Add drafter threshold sweep
    for threshold in DRAFTER_THRESHOLDS:
        base_cmd.extend(["--drafter_thresholds", str(threshold)])
    
    # Add FailFast sweep parameters
    for threshold in SWEEP_LOWCONF_THRESHOLD:
        base_cmd.extend(["--sweep_lowconf_threshold", str(threshold)])
    for length in SWEEP_MAX_SPEC_LEN:
        base_cmd.extend(["--sweep_max_spec_len", str(length)])
    for length in SWEEP_INCR_LEN:
        base_cmd.extend(["--sweep_incr_len", str(length)])
    
    # ==========================================
    # Test 1: Verifier-only AR baseline (7B)
    # ==========================================
    print("\n" + "="*80)
    print("TEST 1: Verifier-only AR baseline (7B Instruct)")
    print("="*80)
    cmd1 = base_cmd + ["--mode", "verifier_ar"]
    rc1 = run_command(cmd1)
    
    # ==========================================
    # Test 2: AR-AR mode (1.5B drafter + 7B verifier)
    # ==========================================
    print("\n" + "="*80)
    print("TEST 2: AR-AR mode (1.5B drafter + 7B verifier)")
    print("="*80)
    cmd2 = base_cmd + ["--mode", "ar_ar"]
    rc2 = run_command(cmd2)
    
    # ==========================================
    # Test 3: dLLM-AR mode with FailFast (dynamic frequency)
    # ==========================================
    print("\n" + "="*80)
    print("TEST 3: dLLM-AR mode with FailFast (dynamic frequency)")
    print("="*80)
    cmd3 = base_cmd + ["--mode", "dllm_ar"]
    rc3 = run_command(cmd3)
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test 1 (verifier_ar):  {'PASSED' if rc1 == 0 else 'FAILED'}")
    print(f"Test 2 (ar_ar):        {'PASSED' if rc2 == 0 else 'FAILED'}")
    print(f"Test 3 (dllm_ar):      {'PASSED' if rc3 == 0 else 'FAILED'}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nConfiguration used:")
    print(f"  Dataset: {DATASET}")
    print(f"  Num questions: {NUM_QUESTIONS}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Spec length: {SPEC_LEN}")
    print(f"  Verifier model: {VERIFIER_MODEL}")
    print(f"  Drafter model: {DRAFTER_MODEL}")
    print(f"  dLLM repo: Efficient-Large-Model/Fast_dLLM_v2_1.5B")
    print(f"  Precision: torch.bfloat16 (for L4 GPU)")
    print(f"  Device map: {{\\\"\\\" : 0}} (single GPU)")
    print("\n" + "="*80)
    
    return 0 if all(rc == 0 for rc in [rc1, rc2, rc3]) else 1

if __name__ == "__main__":
    sys.exit(main())
