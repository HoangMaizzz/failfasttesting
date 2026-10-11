# Native latent-dynamics V2 feasibility test

## What this test answers

Can a compact action-conditioned model predict how Fast-dLLM's *native hidden
representation and scheduler state* change after one to three real R/E steps?
This is a drafter-dynamics experiment, not an acceptance predictor, token
generator, verifier model, or latency test.

## Native state captured

The collector runs Fast-dLLM greedily at threshold 0.5, physical block 32,
scheduler sub-block 8, E length 8, proposal cap 64, and samples 0--3 R steps
before each E. EOS ends the trajectory. Prefix KV is enabled; mutable block KV
is not used.

For each native snapshot, the capture stores:

- `U`: the full 32-position residual at `hidden_states[24]`, the input to native
  decoder layer 24 (the first of the final four layers) on the denoising forward
  that made the commit decision;
- `M`: for each of native layers 24--27, K and V means over the last 128 prefix
  positions and over the last 8 positions; this is a compact *summary* of the
  prefix cache, not a lossless/full KV cache;
- `G`: the 32-position mask, positions newly committed by that forward, the
  next active scheduler region, context/block/forward/refinement coordinates,
  and native-EOS / top-1-fill-EOS flags.

The residual `U` is the pre-commit input to the native forward; the mask and
event in `G` describe the result of that same forward. This timestamp convention
is kept explicit so a model is not trained against a falsely “post-commit”
hidden tensor. Token IDs, question text, top-k logits, confidence, margin,
entropy, verifier output, accepted-token counts, and latency are excluded from
the learner and the packaged feature archive.

## Learner

The model encodes each of the 32 native residual rows into a 128-dimensional
per-position latent with positional information. The action-conditioned
transition uses separate learned R/E scale-shift adapters before a shared
three-layer Transformer with a residual update. A small decoder predicts the next native residual,
compressed prefix-KV summary, mask, commit event, active scheduler region, and
scheduler/terminal features. The binary outputs describe structural events;
there is no vocabulary head, token-level distribution, or token-ID target.

Training uses multi-step losses through H1--H3. After encoding a real root, the
latent state is rolled forward internally; true child features are used only
for loss/evaluation and are never fed into the next prediction. The real R/E
action sequence is supplied to isolate dynamics learning from policy learning.

Three matched feature ablations are trained:

1. `native_only`: native residual and position;
2. `native_structure`: adds mask, event, active-region, scheduler and terminal
   features;
3. `full`: additionally conditions on the prefix-KV summary.

All three predict the same targets. A hold-state persistence control is also
reported. The question split is 60/20/20 and disjoint by source question.

## Readout

The ZIP reports metrics separately for R/E and H1/H2/H3: normalized and native
residual error, cosine, prediction-vs-true change error, prefix-summary error,
mask accuracy/F1, commit-event F1, active-region F1, terminal-event accuracy,
and validation curves at 25/50/100% of updates. The result archive contains
only the permitted latent/structural teacher features, action edges, scaler
statistics, model weights and reports.

## Scope and limitations

This first executable test deliberately stops at the cut-layer residual and a
small prefix-KV summary. It does **not** pass predicted residuals through the
frozen four-layer suffix, preserve the complete position-indexed KV cache, or
test end-to-end decoding. Therefore a positive result supports learnability of
these native feature dynamics, but does not yet prove exact dLLM-state
simulation or production readiness. A negative result should be interpreted
with the reported R/E, horizon and feature ablations; it does not by itself
prove that every richer native latent is insufficient.

## Kaggle run

Use a Kaggle session with Internet and GPU T4 x2. The launcher downloads the
repo, Fast-dLLM 1.5B weights and GSM8K test split, runs its focused unit tests,
captures fresh trajectories on GPU 0, trains/evaluates on GPU 1, and creates a
sanitized ZIP under `/kaggle/working`.
