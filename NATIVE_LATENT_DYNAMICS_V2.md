# Native latent-dynamics V2: native-rules revision V3

This revision changes the capture schema and learner. **Recapture fresh data**:
the older V2 ZIP omitted full physical events and upstream KV features and cannot
be safely relabeled as this experiment. The filenames stay compatible with the
existing launcher URL; output ZIPs now start with `native_latent_rules_v3_`.

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
- `M`: for each of native layers 0/7/15/23, K and V means over the last 128 prefix
  positions and over the last 8 positions; this is a compact *summary* of the
  prefix cache, not a lossless/full KV cache;
- `G`: the 32-position mask, positions newly committed by that forward, the
  next active scheduler region, context/block/forward/refinement coordinates,
  and EOS-present / top-1-fill-EOS flags. Exact absolute geometry, physical
  pre-forward mask, actual forward region and proposal membership are separate
  fields. `native_terminal` is distinct from merely containing EOS.

Physical commit event is the complete pre/post mask difference, including native
commits outside the logical 8-position proposal. A boundary-crossing proposal
may have fewer than 8 rows visible in its final physical frame; earlier positions
must already be committed. No missing hidden rows are fabricated. Masks after
terminal EOS are excluded from loss/scoring. EOS without an exposed hidden
snapshot is retained as a termination-only edge, never as a fake hidden target.

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
three-layer Transformer with a residual update. The output predicts a **change**
on top of the carried native residual, not a lossy reconstruction of all 1536
channels through 128 dimensions. Source rows are carried only when absolute
positions match; a new physical frame has no such identity baseline. The binary outputs describe structural events;
there is no vocabulary head, token-level distribution, or token-ID target.

At a frame crossing, **source conditioning is not discarded**: distinct target
position queries cross-attend to the unaligned source latent. Absolute alignment
is used for the identity skip only. This avoids a geometry fix itself collapsing
all new-frame positions to an identical source-independent representation.

`native_latent_rules_v2.py` enforces known rules outside the neural network:

- R cannot remask committed positions, and copies its prefix KV exactly;
- events are restricted to masked positions of the actual native 8-subblock,
  with at least one commit when eligible;
- next region, absolute coordinates and R counters are computed, not learned;
- E fills the previous proposal before a **fresh** native prefill, seed/bridge
  operations and first exposed unmask. It is a macro action, not the generator's
  in-place E operation. Its first usable pre-mask is fixed by native seed and
  absolute geometry (and audited); its micro-forward count still depends on
  bridge passes and is predicted. Actual counts are also recorded;
- the event readout uses the native right-shift alignment `[row0,row0,row1,...]`.

The event head predicts *structural commit likelihood*. It does **not** reproduce
the native `token confidence > .5` calculation. Enforcing region/progress makes
predictions legal, not automatically correct. This distinction is explicit in
the report. Native EOS readiness is learned separately from EOS presence.

Training starts with H1, adds H2 after one third of updates, and H3 after two
thirds. Loss weights count actual valid transitions, so a rare H3 subset does
not get the same total weight as the full H1 batch. After encoding a real root,
latent and predicted U/M/G roll forward internally; true child features are used only
for loss/evaluation and are never fed into the next prediction. The real R/E
action sequence is supplied to isolate dynamics learning from policy learning.

Three matched feature ablations are trained:

1. `native_only`: learned dynamics consume native residual and position;
2. `native_structure`: adds mask, event, active-region, scheduler and terminal
   features;
3. `full`: additionally conditions on the prefix-KV summary.

All three share the same deterministic rule executor/absolute geometry: the
ablation concerns what learned dynamics sees, not whether native rules apply.
They use identical initialization/minibatch seeds and question splits. All
three predict the same targets. Validation selects a checkpoint using H1
residual error plus proposal mask-count error; test is untouched until selection.
A hold-state persistence control is also
reported. The question split is 60/20/20 and disjoint by source question.

## Readout

The ZIP reports metrics separately for R/E and H1/H2/H3: normalized and native
residual error, cosine, prediction-vs-true change error, prefix-summary error,
physical and proposal-scoped mask accuracy, wrong-position count/exact mask rate,
eligible-event precision/recall/F1, commit-count error, active-region F1, terminal-event accuracy,
and validation curves at 25/50/100% of updates. The result archive contains
only the permitted latent/structural teacher features, action edges, scaler
statistics, model weights and reports.

Reports separate action **sequences** (e.g. RER versus RRR), E frame crossings,
and EOS-positive coverage. No EOS-positive samples means “not tested,” not
proof that 100% majority-class EOS accuracy is informative. Delta metrics use
only mapped overlapping positions; new-frame prediction uses its own absolute
error. A native-contract audit fails on corrupt capture before training.

Sanitized features are saved **before** training. Partial failures retain them
and full traceback. A corrected saved folder can be retrained without dLLM via
`--capture-input <folder> --train-device cpu` (or a CUDA device). This does not
silently convert an old V2 archive. Checkpoints include config, scalers and the
selected update.

## Scope and limitations

This first executable test deliberately stops at the cut-layer residual and a
small prefix-KV summary. It does **not** pass predicted residuals through the
frozen four-layer suffix, preserve the complete position-indexed KV cache, or
test end-to-end decoding. Therefore a positive result supports learnability of
these native feature dynamics, but does not yet prove exact dLLM-state
simulation or production readiness. A negative result should be interpreted
with the reported R/E, horizon and feature ablations; it does not by itself
prove that every richer native latent is insufficient.

The boundary pairs pre-commit U with post-commit structure. It is **partially
observed**: replacement-token representations and the full position-indexed
prefix cache are absent. Correct scheduler rules cannot recover that missing
semantic information. They remove avoidable protocol/geometry errors; accuracy
must still be measured on fresh held-out native trajectories.

## Kaggle run

Use a Kaggle session with Internet and GPU T4 x2. The launcher downloads the
repo, Fast-dLLM 1.5B weights and GSM8K test split, runs its focused unit tests,
captures fresh trajectories on GPU 0, trains/evaluates on GPU 1, and creates a
sanitized ZIP under `/kaggle/working`.

The weights directory is separate from the Git checkout. After downloading
weights/config, the launcher copies the pinned checkout's instrumented
`modeling.py` beside `config.json`, checks its SHA-256 and all custom modules
referenced by `auto_map`, then starts capture. This also replaces stale model
code on a notebook rerun. Missing custom-code files fail before GPU model load.
