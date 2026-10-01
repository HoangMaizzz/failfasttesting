"""Kaggle T4 x2 launcher. Execute this file in ONE fresh notebook cell.

Optional notebook globals: SOURCE_REF, NUM_QUESTIONS, VALIDATION_QUESTIONS,
DATASET, MAX_ROUNDS_PER_QUESTION, MAX_NEW_TOKENS, STOP_WEIGHT, EXTEND_WEIGHT,
REFINE_WEIGHT. No input ZIP required.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SOURCE_REF = globals().get("SOURCE_REF", "codex/sparse-extend-world-model")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 10))
VALIDATION_QUESTIONS = int(globals().get("VALIDATION_QUESTIONS", 2))
DATASET = globals().get("DATASET", "gsm8k")
MAX_ROUNDS_PER_QUESTION = int(globals().get("MAX_ROUNDS_PER_QUESTION", 2))
MAX_NEW_TOKENS = int(globals().get("MAX_NEW_TOKENS", 128))
MAX_CONTEXT_TOKENS = int(globals().get("MAX_CONTEXT_TOKENS", 768))
STOP_WEIGHT = float(globals().get("STOP_WEIGHT", 1.0))
EXTEND_WEIGHT = float(globals().get("EXTEND_WEIGHT", 1.0))
REFINE_WEIGHT = float(globals().get("REFINE_WEIGHT", 1.0))
MODEL_ARCHITECTURE = globals().get("MODEL_ARCHITECTURE", "legacy")
EPISODES_PER_QUESTION = int(globals().get("EPISODES_PER_QUESTION", 1))
UPDATES_PER_TRANSITION = int(globals().get("UPDATES_PER_TRANSITION", 1))
LATENT_DIM = int(globals().get("LATENT_DIM", 128))
REPLAY_STATES = int(globals().get("REPLAY_STATES", 4096))

working = Path("/kaggle/working")
temporary = Path("/kaggle/temp")
working.mkdir(parents=True, exist_ok=True)
temporary.mkdir(parents=True, exist_ok=True)
# A previous notebook cell may have deleted its own cwd. Always recover first.
os.chdir(working)
if DATASET not in ("gsm8k", "math") or not 0 < VALIDATION_QUESTIONS < NUM_QUESTIONS:
    raise ValueError("Choose gsm8k/math and nonempty training + validation splits")
if MODEL_ARCHITECTURE not in ("legacy","token_dual", 'two_source'):
    raise ValueError("Unknown world-model architecture")
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet before running this cell")
for index in range(2):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)

environment = os.environ.copy()
environment.update(GIT_LFS_SKIP_SMUDGE="1", CUDA_VISIBLE_DEVICES="0,1",
    HF_HOME=str(temporary / "wm_hf_cache"), PYTHONUNBUFFERED="1",
    TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    USE_TF='0', USE_FLAX='0', WANDB_MODE='disabled')
print("Installing Python dependencies; keeping Kaggle's existing PyTorch.", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers==4.53.1", "accelerate", "datasets", "einops", "numpy",
    "pandas", "matplotlib", "scipy", "sentencepiece", "huggingface_hub"],
    cwd=working, env=environment, check=True)

repo = temporary / "interactive_world_model_repo"
git_url = "https://github.com/HoangMaizzz/failfasttesting.git"
if not (repo / ".git").is_dir():
    if repo.exists():
        raise FileExistsError(f"Non-Git directory already exists: {repo}; no files deleted")
    subprocess.run(["git", "clone", "--depth", "1", "--branch",
        "codex/sparse-extend-world-model", git_url, str(repo)], cwd=working,
        env=environment, check=True)
dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
if dirty.strip():
    raise RuntimeError("Source checkout has local edits; refusing to overwrite them")
subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", SOURCE_REF],
               cwd=working, env=environment, check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"],
               cwd=working, env=environment, check=True)
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
print("Source commit:", commit, flush=True)
if len(SOURCE_REF) == 40 and commit != SOURCE_REF:
    raise RuntimeError("Pinned source revision mismatch")

subprocess.run([sys.executable, str(repo / "tests/test_world_model_pretraining.py")],
               cwd=repo, env=environment, check=True)
subprocess.run([sys.executable, str(repo / "tests/test_native_elysia_graph.py")],
               cwd=repo, env=environment, check=True)
if MODEL_ARCHITECTURE in ("token_dual", 'two_source'):
    subprocess.run([sys.executable,str(repo/"tests/test_world_model_probe.py")],
                   cwd=repo,env=environment,check=True)
if MODEL_ARCHITECTURE=='two_source':
    subprocess.run([sys.executable,str(repo/'tests/test_world_model_twosource.py')],
                   cwd=repo,env=environment,check=True)

# Weights and source are outside /kaggle/working, so they are not published Output.
# Download in a fresh Python subprocess, avoiding stale imports after pip changes.
dllm = temporary / "wm_fast_dllm_1_5b"
download_code = (
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('Efficient-Large-Model/Fast_dLLM_v2_1.5B', "
    "local_dir=" + repr(str(dllm)) + ", "
    "allow_patterns=['configuration.py','*.json','*.safetensors','*.txt','*.jinja'])"
)
subprocess.run([sys.executable, "-c", download_code], cwd=repo, env=environment, check=True)
# Only our audited native generator is used, never the upstream modeling.py.
shutil.copy2(repo / "Fast_dLLM_v2_1_5B/modeling.py", dllm / "modeling.py")
if shutil.disk_usage(working).free < 2 * 1024**3:
    raise OSError("Need at least 2 GiB free for smoke outputs/checkpoints")

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
output = working / f"wm_{MODEL_ARCHITECTURE}_{DATASET}_{NUM_QUESTIONS}q_{stamp}"
command = [sys.executable, "-u", str(repo / "pretrain_acceptance_world_model.py"),
    "--dataset", DATASET, "--num_questions", str(NUM_QUESTIONS),
    "--validation_questions", str(VALIDATION_QUESTIONS),
    "--model_architecture",MODEL_ARCHITECTURE,"--episodes_per_question",str(EPISODES_PER_QUESTION),
    "--updates_per_transition",str(UPDATES_PER_TRANSITION),"--latent_dim",str(LATENT_DIM),
    "--replay_states",str(REPLAY_STATES),
    "--max_rounds_per_question", str(MAX_ROUNDS_PER_QUESTION),
    "--max_new_tokens", str(MAX_NEW_TOKENS), "--max_proposal_tokens", "64",
    "--max_context_tokens",str(MAX_CONTEXT_TOKENS),
    "--extend_size", "8", "--max_refinement_steps", "3", "--drafter_threshold", "0.5",
    "--stop_weight", str(STOP_WEIGHT), "--extend_weight", str(EXTEND_WEIGHT),
    "--refine_weight", str(REFINE_WEIGHT),
    "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
    "--target_device", "0", "--drafter_device", "1",
    "--target_gpu_memory_gib", "8",
    "--dllm_dir", str(dllm), "--output_dir", str(output)]
command += ['--source_revision',commit]
if MODEL_ARCHITECTURE in ("token_dual", 'two_source'):
    command += ["--package_every_question","--warmup_updates","16","--horizon_warmup_updates","64"]
if MODEL_ARCHITECTURE=='two_source':
    command += ['--horizon','3','--shadow_verify_probability',str(globals().get('SHADOW_VERIFY_PROBABILITY', .15)),
        '--audit_retrain_updates',str(globals().get('AUDIT_RETRAIN_UPDATES',400)),
        '--audit_states_per_question',str(globals().get('AUDIT_STATES_PER_QUESTION',32)),
        '--audit_train_states_per_question',str(globals().get('AUDIT_TRAIN_STATES_PER_QUESTION',24)),
        '--audit_seeds',*[str(s) for s in globals().get('AUDIT_SEEDS',[42,43])]]
    if globals().get('AUDIT_VARIANTS'):
        command += ['--audit_variants',*globals()['AUDIT_VARIANTS']]
print("Running:", " ".join(command), flush=True)
print("Memory layout: verifier FP16 sharded across GPU 0+1 (8 GiB placement cap/card); "
      "dLLM FP16 + world model also on GPU 1", flush=True)
print("No pre-collected ZIP needed. Generation stops on verified EOS; "
      f"round cap={MAX_ROUNDS_PER_QUESTION or 'none'}, answer token cap={MAX_NEW_TOKENS or 'none'}.", flush=True)
print(f"Context safety guard={MAX_CONTEXT_TOKENS}; hitting it before EOS in uncapped mode is a PARTIAL run, not success.",flush=True)
print("Random legal S/E/R; E includes first unmask, max 3 extra R. "
      + ('Actual S and sampled shadow probes call verifier; shadow never changes prefix or memory.'
         if MODEL_ARCHITECTURE=='two_source' else 'Only S calls verifier.'), flush=True)
if MODEL_ARCHITECTURE=='two_source':
    print('Fixed heldout collected first; curves at 0/10/20/40/60/80 training questions. '
          'Feature sensitivity + independently retrained removals reuse saved data; no extra LLM forwards.',flush=True)
print(f"Architecture={MODEL_ARCHITECTURE}; episodes/question={EPISODES_PER_QUESTION}; updates/transition={UPDATES_PER_TRANSITION}; replay capacity={REPLAY_STATES}",flush=True)
completed = subprocess.run(command, cwd=repo, env=environment, check=False)
archive = output.with_suffix(".zip")
if archive.is_file():
    print("DOWNLOAD ZIP (already at /kaggle/working root):", archive.name, flush=True)
if completed.returncode:
    print("Run failed: keep the partial ZIP and error.txt for diagnosis.", flush=True)
    raise subprocess.CalledProcessError(completed.returncode, command)
with zipfile.ZipFile(archive) as zip_file:
    summary = json.loads(zip_file.read("summary.json"))
    if summary["status"] != "complete" or zip_file.testzip() is not None:
        raise RuntimeError("Incomplete or corrupt result archive")
print(json.dumps({key: summary.get(key) for key in (
    "status", "questions_completed", "episodes_completed", "updates", "dynamics_updates", "nodes", "edges", "parameter_l2_change",
    "environment", "label_coverage", "evaluation")}, indent=2), flush=True)
# The relative link is served by the live notebook; Save Version users can use
# the top-level ZIP shown in Output after the run completes.
from IPython.display import FileLink, display
display(FileLink(archive.name))
