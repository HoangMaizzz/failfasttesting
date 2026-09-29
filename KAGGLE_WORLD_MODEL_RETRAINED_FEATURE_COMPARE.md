# Retrained full-feature vs hidden-layer-28-only comparison

This is a retrained ablation, unlike the earlier frozen-checkpoint feature mask.
It fits two fresh models on the exact same 80 training questions and scores them
on the archive's same 20 held-out questions:

* `full`: all archived hidden layers `[7, 14, 28]` plus the existing features.
* `hidden28`: only raw hidden layer 28, while retaining top-K logit gaps and all
  other non-hidden inputs (token streams, candidate embeddings, prefix, history,
  and scalar features).

Both start from fresh initialization and receive the same optimizer-update
budget, replay sampling seed, train/validation split, validation states, and
validation roots. Default is 512 updates per model for each of three seeds.
The report compares current acceptance prediction, one-step R/E, and horizon-3
rollout using paired question-macro MAE. Negative `hidden28 - full` favors the
compact hidden-28 model. Raw predictions and per-seed paired bootstrap intervals
are included in the output ZIP. The parameter count is recorded for each model.

This test runs only the small world model on one GPU. It makes zero drafter and
verifier calls; it does not measure production inference latency.

## Kaggle cell

Attach the full `wm_token_dual_math_100q_*.zip` training archive under
`/kaggle/input/datasets/yumesakihikari/math100` (or set `INPUT_PATH` to the exact
ZIP path). Do not use the small feature-audit ZIP.

```python
from urllib.request import urlopen

config = {
    "SOURCE_REF": "codex/sparse-extend-world-model",  # or pin a commit SHA
    "INPUT_PATH": "/kaggle/input/datasets/yumesakihikari/math100",
    "SEEDS": "42,43,44",
    "UPDATES": 512,
    "VALIDATION_STATES_PER_QUESTION": 24,
    "VALIDATION_ROOTS_PER_QUESTION": 4,
}

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/"
    f"{config['SOURCE_REF']}/kaggle_world_model_retrained_feature_compare.py"
)
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_world_model_retrained_feature_compare.py", "exec"), config)
```
