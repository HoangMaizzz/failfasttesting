"""Fresh Kaggle T4 x2: 100 GSM8K questions, two-source H3 + full factor audit.

No uploaded ZIP/dataset/secret needed. Enable Internet and GPU T4 x2.
Overrides may be supplied in the exec globals for a shorter integration smoke.
"""
from urllib.request import urlopen

SOURCE_REF = globals().get('SOURCE_REF','codex/sparse-extend-world-model')
MODEL_ARCHITECTURE = 'two_source'
DATASET = 'gsm8k'
NUM_QUESTIONS = int(globals().get('NUM_QUESTIONS',100))
VALIDATION_QUESTIONS = int(globals().get('VALIDATION_QUESTIONS',20))
EPISODES_PER_QUESTION = int(globals().get('EPISODES_PER_QUESTION',1))
MAX_ROUNDS_PER_QUESTION = int(globals().get('MAX_ROUNDS_PER_QUESTION',0))
MAX_NEW_TOKENS = int(globals().get('MAX_NEW_TOKENS',0))
MAX_CONTEXT_TOKENS = int(globals().get('MAX_CONTEXT_TOKENS',4096))
UPDATES_PER_TRANSITION = int(globals().get('UPDATES_PER_TRANSITION',1))
LATENT_DIM = int(globals().get('LATENT_DIM',128))
REPLAY_STATES = int(globals().get('REPLAY_STATES',1024))
STOP_WEIGHT = EXTEND_WEIGHT = REFINE_WEIGHT = 1.
SHADOW_VERIFY_PROBABILITY = float(globals().get('SHADOW_VERIFY_PROBABILITY',.15))
AUDIT_SEEDS = globals().get('AUDIT_SEEDS',[42,43])
AUDIT_RETRAIN_UPDATES = int(globals().get('AUDIT_RETRAIN_UPDATES',400))
url=f'https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{SOURCE_REF}/kaggle_world_model_pretrain.py'
exec(compile(urlopen(url,timeout=60).read().decode('utf-8'),'kaggle_world_model_pretrain.py','exec'))
