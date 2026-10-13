"""One-cell Kaggle launcher: frozen capture, A/B/C probes, atomic result ZIP.

Set SOURCE_REF to the published commit. All pipeline explanation is in the
chat; this executable launcher accepts notebook globals for smoke/pilot/resume.
No input archive is needed for a fresh all/collect run. Train-only reuses a ZIP.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def launch(settings):
    working, temporary = Path("/kaggle/working"), Path("/kaggle/temp")
    working.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    os.chdir(working)
    mode = settings.get("MODE", "all")
    source = settings.get("SOURCE_REF", "codex/factorized-wm-feasibility")
    input_path = settings.get("INPUT_PATH")
    if mode not in ("all", "collect", "train"):
        raise ValueError("MODE must be all, collect or train")
    if input_path and not Path(input_path).exists():
        raise FileNotFoundError(input_path)
    if mode == "train" and not input_path:
        raise ValueError("Train-only requires INPUT_PATH: previous result ZIP or extracted run")
    import torch
    if mode != "train" and torch.cuda.device_count() != 2:
        raise RuntimeError("Select GPU T4 x2 and enable Internet")
    for gpu in range(torch.cuda.device_count()):
        print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)} "
              f"({torch.cuda.get_device_properties(gpu).total_memory / 2**30:.1f} GiB)", flush=True)
    env = os.environ.copy()
    env.update(GIT_LFS_SKIP_SMUDGE="1", PYTHONUNBUFFERED="1",
               HF_HOME=str(temporary / "agreement_hf_cache"),
               USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
               WANDB_MODE="disabled", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if mode != "train":
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "transformers==4.53.1", "accelerate", "datasets", "einops", "numpy",
                    "pandas", "scipy", "sentencepiece", "huggingface_hub"],
                   cwd=working, env=env, check=True)
    repo = temporary / "agreement_profile_repo"
    if not (repo / ".git").is_dir():
        if repo.exists():
            raise FileExistsError(f"Existing non-Git folder: {repo}")
        subprocess.run(["git", "clone", "--depth", "1", "--branch",
                        "codex/factorized-wm-feasibility",
                        "https://github.com/HoangMaizzz/failfasttesting.git", str(repo)],
                       cwd=working, env=env, check=True)
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("Source checkout has edits; use a fresh session/source folder")
    subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", source],
                   cwd=working, env=env, check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"],
                   cwd=working, env=env, check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if len(source) == 40 and commit != source:
        raise RuntimeError("Pinned source commit mismatch")
    print("SOURCE COMMIT:", commit, flush=True)
    for test in ("test_agreement_profile_study.py", "test_world_model_probe.py",
                 "test_native_elysia_graph.py", "test_native_unmask_contract.py"):
        subprocess.run([sys.executable, str(repo / "tests" / test)], cwd=repo, env=env, check=True)
    dllm = temporary / "agreement_fast_dllm_1_5b"
    if mode != "train":
        script = ("from huggingface_hub import snapshot_download; "
            "snapshot_download('Efficient-Large-Model/Fast_dLLM_v2_1.5B', local_dir="
            + repr(str(dllm)) + ", allow_patterns=['configuration.py','*.json','*.safetensors','*.txt','*.jinja'])")
        subprocess.run([sys.executable, "-c", script], cwd=repo, env=env, check=True)
        shutil.copy2(repo / "Fast_dLLM_v2_1_5B/modeling.py", dllm / "modeling.py")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = working / ("agreement_profile_" + stamp)
    command = [sys.executable, "-u", str(repo / "run_agreement_profile_study.py"),
               "--mode", mode, "--output_dir", str(output), "--dllm_dir", str(dllm),
               "--source_revision", commit]
    options = {
        "DATASET": ("dataset", "gsm8k"), "NUM_QUESTIONS": ("num_questions", 100),
        "VALIDATION_QUESTIONS": ("validation_questions", 20), "TEST_QUESTIONS": ("test_questions", 20),
        "COLLECTION_SEED": ("seed", 31415), "STATES_PER_QUESTION": ("states_per_question", 32),
        "MAX_ROUNDS": ("max_rounds_per_question", 2), "SEGMENTS_PER_ROUND": ("segments_per_round", 8),
        "MAX_CONTEXT_TOKENS": ("max_context_tokens", 1024),
        "MAX_PROPOSAL_TOKENS": ("max_proposal_tokens", 64),
        "DRAFTER_THRESHOLD": ("drafter_threshold", .5), "EPOCHS": ("epochs", 30),
        "PATIENCE": ("patience", 5), "BATCH_SIZE": ("batch_size", 16),
        "PROBE_WIDTH": ("width", 256), "PROBE_LAYERS": ("layers", 2),
        "OOF_FOLDS": ("oof_folds", 5), "BOOTSTRAP_DRAWS": ("bootstrap_draws", 2000),
        "SIGN_WEIGHT": ("sign_weight", 0.), "LEARNING_RATE": ("learning_rate", 3e-4)}
    for name, (flag, default) in options.items():
        command += ["--" + flag, str(settings.get(name, default))]
    command += ["--training_seeds", *map(str, settings.get("TRAINING_SEEDS", [42, 43, 44]))]
    if input_path:
        command += ["--input_path", str(input_path)]
    if settings.get("PARITY_REPEAT", True):
        command.append("--parity_repeat")
    if settings.get("DEVICE"):
        command += ["--device", settings["DEVICE"]]
    print("PIPELINE: frozen native capture + shadow labels -> prompt split -> A/B -> OOF -> C", flush=True)
    print("T4 x2: Fast-dLLM FP16 entirely GPU1; Qwen FP16 sharded GPU0+GPU1, 8 GiB weight cap/card. "
          "LLMs released before probe training on GPU0; no CPU offload/quantization fallback.", flush=True)
    if input_path:
        print("Resume: completed capture questions are reused; offline probes train again from the corpus.", flush=True)
    print("R uses native replay; replay wall time is not production action latency.", flush=True)
    print("OUTPUT:", output, flush=True)
    completed = subprocess.run(command, cwd=repo, env=env, check=False)
    archive = output.with_suffix(".zip")
    if archive.is_file():
        print("DOWNLOAD / RESUME ZIP:", archive.name, flush=True)
        from IPython.display import FileLink, display
        display(FileLink(archive.name))
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    with zipfile.ZipFile(archive) as z:
        summary = json.loads(z.read("summary.json"))
        if summary["status"] != "complete":
            raise RuntimeError("Run did not complete; inspect error.txt in result ZIP")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary.get("scientific_tests_complete") and mode != "collect":
        print("Test C lacks R coverage in at least one split; inspect audit.json before drawing conclusions.", flush=True)
    return archive


if __name__ == "__main__":
    launch(globals())
