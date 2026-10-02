# Persistent World Model + FiLM V1

V1 tests the proposed offline-first path on a fresh 100-question GSM8K run. It
keeps the existing two-source collection protocol and adds verifier
distribution targets to the verifier calls that were already required; it does
not run an extra verifier pass just to collect those targets.

## Model and training

The state latent is 128 dimensions. A drafter encoder maps the native proposal
observation to 64 dimensions; a verifier encoder maps the available verifier
observation to 64 dimensions, and a posterior combines them. The verifier
encoder is causal: when the accepted prefix is `Y`, it can use verifier rows
through `Y+1` (including the first rejected token's prediction row), never the
unobserved suffix. Shadow verifier probes are training labels only and do not
advance persistent runtime state.

Two separate transition functions model `R` (refine/unmask) and `E` (extend).
Each transition predicts a prior latent, then corrects it using the next native
drafter observation. The acceptance head predicts token-wise conditional
hazards, so the probability of accepting token *i* is conditioned on all
earlier draft tokens having been accepted. Its loss is prefix-censored: it
learns the accepted prefix and first rejection, not post-rejection positions.
Training warms up at horizon 1 and then trains/evaluates rollouts through H=3.

For Phase 2, the world model and Fast-dLLM backbone are frozen. A token-wise
gated FiLM adapter conditions final-normalized drafter representations on the
*previous* persistent latent. Its FiLM projection is zero-initialized, making
the initial logits exactly equal to the unmodified drafter. It is distilled
from the real verifier's top-32 logits plus an exact aggregate `OTHER` bucket,
with a small base-drafter preservation KL. Current-proposal verifier outputs
are targets only and are not fed into the adapter. No acceptance-surrogate
gradient, online update, or planner is used.

## Evaluation saved in the result ZIP

- `persistent_v1/learning_curve.json`: fixed held-out H1/H2/H3 errors after
  10, 20, 40, 60, and 80 training questions.
- `persistent_v1/evaluation/phase1_metrics.json`: token hazard NLL/AUC,
  expected accepted-prefix MAE/RMSE/bias, exact accepted-length rate, survival
  calibration, R/E breakdowns, and feature-lesion sensitivity.
- `persistent_v1/evaluation/heldout_grounded_state_predictions.jsonl` and
  `heldout_rollout_predictions.jsonl`: state/token-level prediction records.
- `persistent_v1/film_training.json` and `gated_film_adapter.pt`: offline
  distillation curve, held-out grouped KL before/after, and adapter checkpoint.
- `persistent_v1/evaluation/real_verifier_film_comparison.json`: paired,
  same-prefix baseline versus FiLM evaluation using a real verifier, plus
  per-question accepted lengths and draft/verifier latency.
- `persistent_v1/READ_RESULTS.md`: concise interpretation and caveats.

The real-verifier check uses at most 20 held-out questions and compares the
first 8-token native refinement snapshot after one verifier-grounded prior
round. It is a structural pilot, not a full-answer GSM8K accuracy benchmark.
The full 100-question collection and offline training run need to be executed
on Kaggle; local unit tests do not emulate T4 inference.

## One-cell Kaggle run

Use a fresh Kaggle notebook with Internet enabled and `GPU T4 x2`. No result ZIP
or uploaded dataset is required. Run this as one code cell:

```python
from urllib.request import urlopen

config = {
    "SOURCE_REF": "codex/persistent-wm-film-v1",
    "NUM_QUESTIONS": 100,
    "VALIDATION_QUESTIONS": 20,
    "WM_UPDATES_PER_QUESTION": 4,
    "FILM_STEPS": 300,
    "EVAL_QUESTIONS": 20,
}

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/"
    f"{config['SOURCE_REF']}/kaggle_persistent_wm_film_v1.py"
)
source = urlopen(url, timeout=90).read().decode("utf-8")
exec(compile(source, "kaggle_persistent_wm_film_v1.py", "exec"), config)
```

The completed result archive is emitted in `/kaggle/working` and linked at the
end of the cell. It contains the original collection artifacts and the new V1
evaluation together.
