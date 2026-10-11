"""One-cell Kaggle launcher for the native-structured latent feasibility test."""
from urllib.request import urlopen

SOURCE_REF = globals().get("SOURCE_REF", "codex/factorized-wm-feasibility")
MODEL_ARCHITECTURE = "native_structured"
DATASET = "gsm8k"
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS", 100))
VALIDATION_QUESTIONS = int(globals().get("VALIDATION_QUESTIONS", 20))
EPISODES_PER_QUESTION = 1
MAX_ROUNDS_PER_QUESTION = 0
MAX_NEW_TOKENS = 0
MAX_CONTEXT_TOKENS = int(globals().get("MAX_CONTEXT_TOKENS", 4096))
UPDATES_PER_TRANSITION = int(globals().get("UPDATES_PER_TRANSITION", 1))
LATENT_DIM = int(globals().get("LATENT_DIM", 512))
REPLAY_STATES = int(globals().get("REPLAY_STATES", 2048))
RESUME_ARCHIVE = globals().get("RESUME_ARCHIVE")
STOP_WEIGHT = EXTEND_WEIGHT = REFINE_WEIGHT = 1.0
url = (f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/"
       f"{SOURCE_REF}/kaggle_world_model_pretrain.py")
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_world_model_pretrain.py", "exec"))
