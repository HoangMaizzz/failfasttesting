"""Run the fixed-0.5 U1 batch-1x MATH benchmark on a Colab GPU."""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pandas as pd
from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parent
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab first.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}; free={free / 2**30:.2f} GiB", flush=True)
    if total < 22 * 2**30:
        raise RuntimeError("This unquantized two-model run requires a 24-GB-class GPU or larger.")
    model_dir = root / "Fast_dLLM_v2_1.5B"
    snapshot_download(
        "ruipeterpan/Fast_dLLM_v2_1.5B", local_dir=str(model_dir),
        ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"],
    )
    output = root / ("outputs_u1_fixed05_math100_fp16_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    command = [
        sys.executable, "-u", "run_u1_sgd_ablation.py",
        "--datasets", "math", "--methods", "failfast", "u1_batch1x",
        "--num_questions", "100", "--id_offset", "25",
        "--continue_threshold", "0.5",
        "--replay_stop_to_continue_ratio", "0",
        "--target_quantization", "none",
        "--target_dtype", "float16", "--drafter_dtype", "float16",
        "--target_device", "0", "--drafter_device", "0",
        "--drafter_threshold", "0.5", "--lowconf_threshold", "0.7",
        "--max_new_tokens", "1024",
        "--dllm_dir", str(model_dir), "--output_dir", str(output),
    ]
    subprocess.run(command, cwd=root, check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(set(manifest["problem_ids"]["math"])) == 100
    reference_dir = output / "raw" / "math" / "verifier_ar_fp16"
    reference_command = [
        sys.executable, "-u", "failfast.py", "--dataset_name", "math",
        "--num_questions", "100", "--problem_ids",
        *map(str, manifest["problem_ids"]["math"]),
        "--warmup_questions", "1", "--benchmark_modes", "verifier_ar",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--target_quantization", "none", "--target_dtype", "float16",
        "--drafter_dtype", "float16", "--target_device", "0",
        "--decoding_strategy", "greedy", "--max_new_tokens", "1024",
        "--seed", "42", "--quiet_generation", "--disable_progress",
        "--skip_artifacts", "--skip_plots", "--overwrite",
        "--output_dir", str(reference_dir),
    ]
    print("Checking lossless output against independent FP16 greedy verifier...", flush=True)
    subprocess.run(reference_command, cwd=root, check=True)
    columns = ["problem_id", "output_tokens", "output_token_hash"]
    reference = pd.read_csv(reference_dir / "benchmark_results.csv")[columns]
    comparisons = []
    method_audits = {}
    for method in ("failfast", "u1_batch1x"):
        result = pd.read_csv(output / "raw" / "math" / method / "benchmark_results.csv")[columns]
        paired = result.merge(reference, on="problem_id", how="outer",
                              suffixes=("_method", "_reference"), validate="one_to_one", indicator=True)
        paired["method"] = method
        paired["exact_match"] = (
            paired["_merge"].eq("both")
            & paired.output_tokens_method.eq(paired.output_tokens_reference)
            & paired.output_token_hash_method.eq(paired.output_token_hash_reference)
        )
        method_audits[method] = {
            "passed": len(paired) == 100 and bool(paired.exact_match.all()),
            "matched": int(paired.exact_match.sum()),
            "questions": len(paired),
            "mismatch_ids": paired.loc[~paired.exact_match, "problem_id"].tolist(),
        }
        comparisons.append(paired)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "lossless_output_comparison.csv", index=False)
    passed = all(item["passed"] for item in method_audits.values())
    audit = {
        "passed": passed, "methods": method_audits,
        "comparison": "full generated token sequence SHA256 and length, including EOS",
        "reference_time_included_in_u1_timing": False,
    }
    (output / "lossless_audit.json").write_text(json.dumps(audit, indent=2))
    archive = shutil.make_archive(str(output), "zip", output.parent, output.name)
    (root / "latest_u1_fixed05_math100_fp16_archive.txt").write_text(archive)
    print(f"ARCHIVE: {archive}", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Lossless audit FAILED; archive preserved with mismatch IDs.")


if __name__ == "__main__":
    main()
