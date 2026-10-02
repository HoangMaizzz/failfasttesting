"""One-cell Kaggle entry point: GSM8K-100 collection, WM V1, FiLM and real eval."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
from urllib.request import urlopen

SOURCE_REF = globals().get("SOURCE_REF", "codex/persistent-wm-film-v1")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 100))
VALIDATION_QUESTIONS = int(globals().get("VALIDATION_QUESTIONS", 20))
SHADOW_VERIFY_PROBABILITY = float(globals().get("SHADOW_VERIFY_PROBABILITY", 0.15))
WM_UPDATES_PER_QUESTION = int(globals().get("WM_UPDATES_PER_QUESTION", 4))
FILM_STEPS = int(globals().get("FILM_STEPS", 300))
EVAL_QUESTIONS = int(globals().get("EVAL_QUESTIONS", 20))

if NUM_QUESTIONS != 100 or VALIDATION_QUESTIONS >= NUM_QUESTIONS:
    raise ValueError("This V1 comparison is configured for 100 GSM8K questions (80 train / 20 heldout)")

launcher_url = (
    f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{SOURCE_REF}/"
    "kaggle_twosource_pretrain.py"
)
source = urlopen(launcher_url, timeout=90).read().decode("utf-8")
scope = dict(globals())
scope.update(SOURCE_REF=SOURCE_REF, NUM_QUESTIONS=NUM_QUESTIONS,
    VALIDATION_QUESTIONS=VALIDATION_QUESTIONS, DATASET="gsm8k",
    MAX_ROUNDS_PER_QUESTION=0, MAX_NEW_TOKENS=0, MAX_CONTEXT_TOKENS=4096,
    EPISODES_PER_QUESTION=1, UPDATES_PER_TRANSITION=1, LATENT_DIM=128,
    REPLAY_STATES=4096, SHADOW_VERIFY_PROBABILITY=SHADOW_VERIFY_PROBABILITY,
    # The V1 runner performs its own structured feature-sensitivity evaluation.
    # Keep the earlier TwoSource retraining audit to one tiny full-model sanity fit.
    AUDIT_VARIANTS=["full"], AUDIT_SEEDS=[42], AUDIT_RETRAIN_UPDATES=1,
    AUDIT_STATES_PER_QUESTION=16, AUDIT_TRAIN_STATES_PER_QUESTION=8)
exec(compile(source, "kaggle_twosource_pretrain.py", "exec"), scope)

repo = Path(scope["repo"])
run_dir = Path(scope["output"])
dllm = Path(scope["dllm"])
environment = scope["environment"]
command = [sys.executable, "-u", str(repo / "run_persistent_world_model_v1.py"),
    "--run_dir", str(run_dir), "--dllm_dir", str(dllm), "--drafter_device", "1",
    "--target_device", "0", "--target_gpu_memory_gib", "8",
    "--max_nodes_per_question", "80", "--max_edges_per_question", "24",
    "--batch_size", "16", "--updates_per_question", str(WM_UPDATES_PER_QUESTION),
    "--min_updates_per_stage", "20", "--milestones", "10", "20", "40", "60", "80",
    "--horizon", "3", "--horizon_warmup_updates", "40",
    "--film_steps", str(FILM_STEPS), "--film_batch_tokens", "32",
    "--eval_questions", str(EVAL_QUESTIONS), "--seed", "42"]
print("\nRunning offline persistent latent training and paired real-verifier FiLM test:",
      " ".join(command), flush=True)
result = subprocess.run(command, cwd=repo, env=environment, check=False)
archive = run_dir.with_suffix(".zip")
if archive.is_file():
    print("LATEST DOWNLOAD ZIP:", archive, flush=True)
if result.returncode:
    print("Phase V1 failed; keep the partial output folder/ZIP for diagnosis.", flush=True)
    raise subprocess.CalledProcessError(result.returncode, command)
with zipfile.ZipFile(archive) as zf:
    if zf.testzip() is not None:
        raise RuntimeError("Final V1 result ZIP failed integrity check")
    summary = json.loads(zf.read("summary.json"))
    if summary.get("persistent_world_model_v1", {}).get("status") != "complete":
        raise RuntimeError("V1 experiment did not finish; preserve the partial ZIP")
print("V1 report:", run_dir / "persistent_v1" / "READ_RESULTS.md", flush=True)
from IPython.display import FileLink, display
display(FileLink(archive.name))
