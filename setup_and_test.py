#!/usr/bin/env python
"""
Master setup & test script for FailFast on GSM8K.

Runs all steps from resource loading to 3-mode testing:
1. Clone/setup Fast_dLLM_v2_1.5B if needed
2. Apply Fix 2 (RoPE KeyError patch) to modeling.py
3. Run all 3 tests (verifier_ar, ar_ar, dllm_ar)

GitHub Copilot using Claude Haiku 4.5
"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path

# ====================================================================================
# CONFIGURATION - Modify these paths as needed
# ====================================================================================

WORKSPACE = Path("/content/failfasttesting").resolve()
DLLM_DIR = Path("/content/Fast_dLLM_v2_1.5B").resolve()
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# HuggingFace model names
DLLM_REPO = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
VERIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFTER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Test parameters
DATASET = "gsm8k"
NUM_QUESTIONS = 3
MAX_NEW_TOKENS = 256
SPEC_LEN = 10

# ====================================================================================
# UTILITIES
# ====================================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, title):
    """Print step header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*80}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def run_cmd(cmd, cwd=None, check=True):
    """Run shell command."""
    print(f"{Colors.CYAN}>>> {' '.join(cmd)}{Colors.RESET}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print_error(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result.returncode

# ====================================================================================
# STEP 1: Check Python & Dependencies
# ====================================================================================

def check_dependencies():
    print_step(1, "Check Python & Dependencies")
    
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"Workspace: {WORKSPACE}")
    print_info(f"dLLM dir: {DLLM_DIR}")
    
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print_error("PyTorch not installed!")
        sys.exit(1)
    
    try:
        import transformers
        print_success(f"Transformers {transformers.__version__}")
    except ImportError:
        print_error("Transformers not installed!")
        sys.exit(1)

# ====================================================================================
# STEP 2: Setup Fast_dLLM
# ====================================================================================

def setup_dllm():
    print_step(2, "Setup Fast_dLLM_v2_1.5B")
    
    if DLLM_DIR.exists():
        print_info(f"dLLM already exists at {DLLM_DIR}")
        
        # Check if modeling.py exists
        modeling_path = DLLM_DIR / "modeling.py"
        if not modeling_path.exists():
            print_warning("modeling.py not found, may need to download the model")
        else:
            print_success("modeling.py found")
        return
    
    print_info(f"Downloading {DLLM_REPO}...")
    DLLM_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Use transformers to download
    try:
        print_info("Using huggingface_hub to download model...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=DLLM_REPO,
            repo_type="model",
            local_dir=str(DLLM_DIR),
            local_dir_use_symlinks=False
        )
        print_success(f"dLLM downloaded to {DLLM_DIR}")
    except Exception as e:
        print_error(f"Download failed: {e}")
        print_info("You can manually download from:")
        print_info(f"  https://huggingface.co/{DLLM_REPO}")
        print_warning("Skipping dLLM setup - will fail if dllm_ar mode is used")
        return

# ====================================================================================
# STEP 3: Apply Fix 2 (RoPE KeyError Patch)
# ====================================================================================

def apply_fix2_rope_patch():
    print_step(3, "Apply Fix 2: RoPE KeyError Patch to modeling.py")
    
    modeling_path = DLLM_DIR / "modeling.py"
    
    if not modeling_path.exists():
        print_error(f"modeling.py not found at {modeling_path}")
        print_warning("Skipping Fix 2 - model may not support Qwen2.5")
        return False
    
    # Read the file
    with open(modeling_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "self.rope_init_fn = None" in content:
        print_success("Fix 2 already applied!")
        return True
    
    # Pattern to find the problematic line
    pattern = r'self\.rope_init_fn\s*=\s*ROPE_INIT_FUNCTIONS\[config\.rope_scaling\["type"\]\]'
    
    if not re.search(pattern, content):
        print_warning("Could not find rope_init_fn pattern - model structure may differ")
        print_info("This might be OK if using Qwen2.5 which handles RoPE differently")
        return False
    
    # Apply the patch
    replacement = '''self.rope_init_fn = None
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)'''
    
    new_content = re.sub(pattern, replacement, content)
    
    # Backup original
    backup_path = modeling_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy2(modeling_path, backup_path)
        print_success(f"Backup created: {backup_path}")
    
    # Write patched file
    with open(modeling_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print_success("Fix 2 applied to modeling.py")
    return True

# ====================================================================================
# STEP 4: Pre-download Models to HuggingFace Cache
# ====================================================================================

def predownload_models():
    print_step(4, "Pre-download Models to HuggingFace Cache")
    
    models = [
        (VERIFIER_MODEL, "Verifier (7B)"),
        (DRAFTER_MODEL, "Drafter (1.5B)"),
    ]
    
    for model_name, label in models:
        print_info(f"Checking {label}: {model_name}")
        
        cmd = [
            sys.executable, "-m", "huggingface_hub", "scan-cache-dir",
            "--cache-dir", str(HF_CACHE)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if model_name.split('/')[-1] in result.stdout:
            print_success(f"{label} already cached")
        else:
            print_info(f"Downloading {label}... (this may take a few minutes)")
            cmd_dl = [
                sys.executable, "-c",
                f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('{model_name}')
print('Downloading model headers...')
AutoModelForCausalLM.from_pretrained('{model_name}', torch_dtype='bfloat16', device_map={{}})
"""
            ]
            subprocess.run(cmd_dl, capture_output=True)
            print_success(f"{label} downloaded")

# ====================================================================================
# STEP 5: Run 3 Tests
# ====================================================================================

def run_tests():
    print_step(5, "Run 3 FailFast Tests on GSM8K")
    
    os.chdir(WORKSPACE)
    
    # Base command
    base_cmd = [
        sys.executable, "failfast.py",
        "--dataset_name", DATASET,
        "--num_questions", str(NUM_QUESTIONS),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--spec_len", str(SPEC_LEN),
        "--verifier_model_name", VERIFIER_MODEL,
        "--drafter_model_name", DRAFTER_MODEL,
        "--overwrite",
        "--log_level", "INFO",
        "--drafter_thresholds", "0.05",
        "--sweep_lowconf_threshold", "0.45",
        "--sweep_max_spec_len", "60",
        "--sweep_incr_len", "10",
    ]
    
    tests = [
        ("verifier_ar", "Verifier-only AR baseline (7B)"),
        ("ar_ar", "AR-AR mode (1.5B drafter + 7B verifier)"),
        ("dllm_ar", "dLLM-AR mode with FailFast"),
    ]
    
    results = {}
    
    for mode, title in tests:
        print_info(f"\n{'='*80}")
        print_info(f"TEST: {title}")
        print_info(f"{'='*80}")
        
        cmd = base_cmd + ["--mode", mode]
        rc = run_cmd(cmd, cwd=WORKSPACE, check=False)
        results[mode] = rc
        
        if rc == 0:
            print_success(f"{title} PASSED")
        else:
            print_error(f"{title} FAILED (exit code {rc})")
        
        print_info("Waiting 5 seconds before next test...")
        import time
        time.sleep(5)
    
    return results

# ====================================================================================
# MAIN
# ====================================================================================

def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("="*80)
    print("FailFast Master Setup & Test Script")
    print("="*80)
    print(f"{Colors.RESET}")
    
    try:
        # Step 1: Check dependencies
        check_dependencies()
        
        # Step 2: Setup dLLM
        setup_dllm()
        
        # Step 3: Apply RoPE patch
        apply_fix2_rope_patch()
        
        # Step 4: Pre-download models
        print_info("Skipping model pre-download to save time (will auto-download during test)")
        # Uncomment to enable:
        # predownload_models()
        
        # Step 5: Run tests
        results = run_tests()
        
        # Summary
        print_step("SUMMARY", "Test Results")
        print(f"{Colors.BOLD}")
        for mode, rc in results.items():
            status = f"{Colors.GREEN}PASSED{Colors.RESET}" if rc == 0 else f"{Colors.RED}FAILED{Colors.RESET}"
            print(f"  {mode:15} : {status}")
        print(f"{Colors.RESET}")
        
        all_passed = all(rc == 0 for rc in results.values())
        if all_passed:
            print_success("All tests passed! ✓")
            return 0
        else:
            print_error("Some tests failed.")
            return 1
        
    except KeyboardInterrupt:
        print_error("\nInterrupted by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
