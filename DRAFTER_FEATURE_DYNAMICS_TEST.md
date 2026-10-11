# Drafter latent feature-dynamics feasibility test

## Question being tested

Can a compact, action-conditioned latent model learn the *state dynamics* of
Fast-dLLM from native observations, without learning to generate token IDs and
without involving the verifier? This test is deliberately not an acceptance
predictor or a miniature language model.

## Fresh data protocol

- Dataset: 100 sampled GSM8K test questions by default; each question is
  generated fresh by `Efficient-Large-Model/Fast_dLLM_v2_1.5B`.
- Drafter: native greedy Fast-dLLM path, threshold 0.5, physical block 32,
  logical segment/extend size 8, maximum 64 generated proposal tokens.
- Probe policy: before each new segment, sample 0–3 native R steps uniformly
  with a deterministic per-question seed, then take E from the same-forward
  top-1-filled parent. EOS ends the trajectory.
- State: one native 8-position segment snapshot. R connects consecutive native
  unmask snapshots; E connects to the next 8-position segment under the longer
  committed prefix. The E transition is therefore **not** a single 64-slot
  in-flight proposal simulation.
- Devices: native dLLM capture on GPU 0; compact latent training on GPU 1.
  The verifier is never loaded or called.

## Feature and latent formulation

Input candidates are tested as separately retrained models:

1. `uncertainty_scalars`: confidence, top-1/top-2 margin, normalized top-k entropy.
2. `scheduler_scalars`: mask, within-segment position, segment/block index,
   native forward index, context length, feature-valid flag.
3. `unmask_event_only`: whether a position was newly unmasked in this snapshot.
4. `scalars`: all ten numeric features above.
5. `hidden_scalars`: all scalars plus the native final-layer hidden vector.
6. `hidden_scalars_history`: group 5 plus R-only deltas from the preceding
   native snapshot (confidence, margin, entropy, hidden cosine change, and a
   validity flag). History is reset at E because the proposal segment changes.

The encoder maps the 8 observed position vectors into a 64/128/256-dimensional
latent sequence. Separate R- and E-conditioned transition blocks advance that
latent. A small decoder predicts only:

- a 64-component train-only PCA projection of the next native hidden vectors;
- next confidence, margin, and entropy;
- next mask state and newly-unmasked event.

The encoder, transition, and decoder are trained jointly by multi-step feature
prediction loss. Rollout is open-loop: after the initial state, the model feeds
its own latent forward for H1, H2, and H3. The real child observation is used
only as the training/evaluation target, never fed into that rollout. Losses are
masked by destination observation validity.

Token IDs, token embeddings, candidate embeddings, vocabulary logits, verifier
outputs, accepted-token counts, and latency are excluded from training targets
and model outputs. The source collector temporarily creates richer arrays, but
the experiment strips them before feature loading/training and refuses to ZIP
any capture shard that still contains those arrays or raw prompt text/IDs.

## Comparisons and readout

- Question-disjoint split: 60% train / 20% validation / 20% test.
- Every feature group is retrained with the same architecture at latent size
  128; the full feature group is also retrained at sizes 64 and 256.
- Two seeds are averaged for validation-based family selection and final
  predictions. The test set is reserved for the declared group/dimension
  ablations and final audit.
- The learned model is compared with a train-only action-conditioned mean
  feature baseline. Reported outcomes include hidden-projection MAE/RMSE,
  confidence/margin/entropy MAE, mask and unmask-event Brier/F1, and paired
  question-bootstrap confidence intervals, split by action and horizon.
- A nested 10/20/40/all-training-question curve shows whether more questions
  improve held-out feature prediction under a fixed update budget.

Evidence for a useful latent dynamic would be lower held-out feature error than
the action-mean baseline, non-collapsing mask/event predictions, and retained
signal from H1 through H3. If hidden features help beyond scalars, or temporal
history helps beyond the current observation, the corresponding retrained
ablation should improve held-out metrics. This experiment cannot establish
verifier acceptance accuracy or deployment latency.

## Kaggle one-cell launcher

Choose GPU T4 x2 and enable Internet. No prior ZIP or input dataset is needed;
the launcher downloads the code, dLLM weights, and GSM8K test split. Paste this
as one Kaggle code cell:

```python
from urllib.request import urlopen

SOURCE_REF = "codex/factorized-wm-feasibility"
NUM_QUESTIONS = 100
UPDATES = 160
CURVE_UPDATES = 80
SEEDS = [42, 43]

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/"
    f"{SOURCE_REF}/kaggle_drafter_feature_dynamics.py"
)
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_drafter_feature_dynamics.py", "exec"), globals())
```

The launcher prints the exact source commit it cloned before starting the run.
