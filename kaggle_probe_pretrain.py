"""One-cell, no-input-ZIP, dual-latent acceptance-world-model probe."""
from urllib.request import urlopen

MODEL_ARCHITECTURE = "token_dual"
SOURCE_REF = globals().get("SOURCE_REF","codex/sparse-extend-world-model")
NUM_QUESTIONS = int(globals().get("NUM_QUESTIONS",40))
VALIDATION_QUESTIONS = int(globals().get("VALIDATION_QUESTIONS",8))
DATASET = globals().get("DATASET", "gsm8k")
EPISODES_PER_QUESTION = int(globals().get("EPISODES_PER_QUESTION",1))
MAX_ROUNDS_PER_QUESTION = int(globals().get("MAX_ROUNDS_PER_QUESTION",0))
MAX_NEW_TOKENS = int(globals().get("MAX_NEW_TOKENS",0))
MAX_CONTEXT_TOKENS = int(globals().get("MAX_CONTEXT_TOKENS",4096))
UPDATES_PER_TRANSITION = int(globals().get("UPDATES_PER_TRANSITION",1))
LATENT_DIM = int(globals().get("LATENT_DIM",128))
STOP_WEIGHT = float(globals().get("STOP_WEIGHT",1.0))
EXTEND_WEIGHT = float(globals().get("EXTEND_WEIGHT",1.0))
REFINE_WEIGHT = float(globals().get("REFINE_WEIGHT",1.0))
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{SOURCE_REF}/kaggle_world_model_pretrain.py"
exec(compile(urlopen(url,timeout=60).read().decode("utf-8"),"kaggle_world_model_pretrain.py","exec"))
