# Persistent behavioral V2 + Gated FiLM

This extends the existing TwoSource loader, hazard utilities, native Elysia
runner, actual/shadow verifier and generator. It does not replace native
unmasking, invent a new remask policy, or implement memory-token approach A.

## Input and question split

Use the **original completed GSM8K-100 TwoSource run**, including
`experience/shard_*.npz`, `states.jsonl`, `edges.jsonl`, `labels.jsonl`,
`teacher_targets.jsonl`, `config.json`, `summary.json`. The existing
`/kaggle/input/datasets/ainzkhail/2source2` folder is the default. Both already
extracted folders and a ZIP (any filename, possibly nested) are supported.
A report-only CV ZIP does not contain the required experiences.

No old checkpoint is loaded into V2. The saved experiences can be reused
because behavior supervision comes from recorded real-verifier targets.
New native-tail examples and FiLM paired comparisons run the verifier again.

Five question-grouped outer folds: 20 test questions/fold, 10 inner-dev and
70 train questions. Thus all 100 questions receive out-of-fold predictions.
No question appears in its own training set. Dev alone selects checkpoints.
Test learning curves never influence selection. Identical question order,
seeds, length-balanced sampling and limits are used for retrained variants;
early stopping can give different actual update counts (these are logged).

## Information flow

```text
native drafter hidden + minimal structure ── D encoder (64) ─┐
previous ACTUAL STOP behavior ───────────── V memory  (64) ─┴─ z_pre (128)
                                                               │
                                                    behavioral decoder G
                                                               │
                                                current acceptance prediction

z_pre + current verifier observation ── U ── z_post
                       ↑ only AFTER actual verifier submission

available z + R/E descriptor ── F_R/F_E ── imagined z_next_pre ── G
                                            │
                                  direct future b_GT loss (main)
                                  frozen future z_post (weak reference)
```

`z_post` is a learned corrected belief, **not physical ground truth**.
The principal ground truth `b_GT` is direct verifier output: accepted K,
prefix survival labels, candidate probability, greedy logit gap, known L.
`K/L` and reject position are deterministic views of K, not independent labels.
Conditional hazard q is trained with censored likelihood through the accepted
prefix and first rejection. A deterministic sample of K is not itself a
probability q. Suffix survival is zero after first rejection, but local suffix
token agreement is unknown/censored and is never mislabeled as a rejection.
Probability and gap targets use only positions through first rejection.
There is no full-vocabulary or raw-hidden reconstruction objective.

Pre never consumes current K, p_V, h_V, margin or teacher distributions. Its
verifier memory is reconstructed from **previous actual STOPs** only (last 8),
bounded by saved causal-history availability. Shadow verification is target-only
and does not update runtime belief. After actual STOP the prefix changes, so an
R/E rollout without a verifier call normally starts from pre, not an imaginary
current posterior. Using a posterior from every shadow node as a rollout input
would leak information unavailable at the decision point.

## Phase A: representation first

Raw cached drafter hidden layers [7,14,28] project into a 64-dimensional
per-token Transformer and pooled D summary. Input scalars are limited to
proposal L, extra-refine index, mask ratio, segment start/depth, physical phase,
position and hidden validity. Confidence/entropy/rank/stability/token embeddings
are not WM inputs. V contains Y/L, reject-position/L, mean candidate p/gap on
the clean prefix, a 32-channel fixed projected last clean causal hidden, and
minimal structure. Hidden after rejection is excluded.

The decoder receives z(128), normalized i/L, L/64 and position Fourier features;
it outputs conditional hazard q_i, candidate probability and nonnegative gap.
Expected K = sum of cumulative products of q_i. This is not sum of verifier
softmax probabilities. Current and privileged-posterior metrics are separate.

Loss = hazard NLL + prefix-survival Brier + candidate-probability MSE +
scaled-gap Huber; posterior behavioral supervision has weight .25.
Representation checkpoints are chosen using inner-dev **pre** macro MAE.
Encoder, grounding module and decoder are then frozen.

## Phase B: fixed-coordinate dynamics

R is `z + delta_R` with the last delta projection zero-initialized: initial R
is exactly identity. E has a separate residual map and an explicit source-only
descriptor [old L, new L, delta L, new segment start/depth, refine index, mask
ratio, action]. Mask ratio at the next state is predicted, never borrowed from
the future observation. Both maps are two-layer MLPs, width 256.

All valid within-round subsequences H1/H2/H3 are enumerated. The same endpoint
supervises paths from up to three past starts. Intermediate predictions are
**never reset to observed truth**. Curriculum H1→H2→H3 chooses best checkpoints
on inner-dev behavioral MAE and supports early stopping.

Default loss at each predicted depth:

```
1.0 hazard + 1.0 (survival Brier + p_V MSE + gap Huber)
+ 0.5 Huber(delta_K / 8)
+ 0.1 normalized frozen-posterior latent consistency
+ 0.05 mask-ratio MSE + 0.001 R-residual regularization
```

The learned future posterior is detached and only a weak coordinate reference.
Real behavior dominates. Report probability/gap error and token calibration,
not merely latent MSE. Neither latent nor behavioral targets enter the earlier
imagined state. Dynamics batches balance starting lengths AND R/E sequences.

Retrained variants: full, D only, D+verifier outcome, no verifier logits,
no verifier hidden, no drafter hidden. Additional full-model diagnostics:
generic vs residual dynamics, latent weight 1.0 vs 0.1, no dynamics, direct
endpoint hazard predictor, and oracle K persistence. Oracle persistence is
explicitly privileged; no hazard scores are attributed to this scalar baseline.

## Phase C: frozen-backbone Gated FiLM

Native generation is instrumented immediately **before the last decoder
block**. Its exact input hidden, attention mask, RoPE positions and immutable
last-layer prefix KV are captured on CPU. Training replays this one frozen
block→final norm→head, with gradients only into FiLM. It does not substitute
post-commit inputs for the pre-unmask forward. An LM-head parity check fails
closed if the native tail cannot be replayed correctly.

Each bounded per-question native example is freshly verified without verifier
KV. First-mismatch correction loss applies only if that position was actually
masked before this forward: already-committed mismatches are not remasked.
Training tails live in a temporary directory and are not included in the ZIP.
Memory is not accumulated as GPU graphs; all LLM/WM parameters remain frozen.

FiLM uses a scalar per-token gate conditioned on LN(h), **previous available**
z, i/L and active/new segment membership. Gamma/beta contribution is zero-init.
Only active eight-token segment rows are changed; prefix/cache-update forwards
are not changed. Four separately trained arms form a 2×2 comparison:

- late-block acceptance objective;
- late-block old grouped-KL objective;
- before-head acceptance objective;
- before-head old grouped-KL objective.

Acceptance objective: protect accepted draft tokens, correct the mutable first
rejection with CE and margin, .05 auxiliary KL, .01 relative hidden-shift penalty.
Full vocabulary allocation is restricted to captured active rows, avoiding a
prefix-length × vocabulary buffer. Teacher argmax is mapped through **vocabulary
IDs**, not mistaken for an index into top-K support (also fixed in the old audit).

Proxy-best checkpoints use inner-dev loss. A separate small real-dev audit
chooses between that checkpoint and zero residual for the `*_real_best.pt`
safety checkpoint. This is a limited two-candidate guard, not a broad
real-verifier hyperparameter search. Raw trained arms remain in the held-out
report so the guard cannot hide intervention failures.

Paired native tests run 4 outer-test questions/fold (20 distinct held-out
questions total), plus inner-dev checkpoint guards. Fixed schedule is root 8,
R/E alternating to at most 64 or EOS; maximum 3 extra unmask is retained in
the native environment, though this structural test uses one scheduled R per
segment. Planner is OFF. Each node is freshly verified, but shadow labels do
not ground the next proposal. Only pairs with the same executed action trace,
proposal length and native pass count are compared. Unavailable R/E and EOS
without snapshots are exclusions, not negatives. Cross-arm token transition
counts use only their shared causal prefix; post-divergence teacher-forced
suffixes are not treated as common verifier ground truth.

This is a native generator replay experiment, not suspended/resumed production
Python execution. Logged drafter wall time includes replay/hook overhead and
must **not** be used as production action latency. FiLM overhead is the paired
replay-time difference, not an isolated kernel microbenchmark.

## Hardware, artifacts and limits

Kaggle T4×2, Internet ON. WM + dLLM1.5B run on GPU1. Qwen2.5-7B-Instruct is
FP16/no KV and memory-sharded over both GPUs (8 GiB cap/card), preserving the
existing memory-optimized path; it is not forced to fit wholly on GPU0.
Model weights/repo/native tails are outside `/kaggle/working`, not in output.

Result ZIP is directly in `/kaggle/working`. It is atomically refreshed after
each completed offline variant/fold and FiLM fold, and on catchable failure;
hard VM termination may leave only the last packaged phase. Saved checkpoints
and errors remain separately in the output folder. Inputs are never modified.

Outputs: fold splits, config/source revision, behavior/latent/rollout metrics,
current/H1/H2/H3 predictions, R/E/action-sequence and L groups, learning curves,
question-bootstrap factor impact, FiLM curves/checkpoints/token diagnostics,
real pairs/exclusions, `READ_RESULTS.md`. Ablations are factor-level tests,
not a complete feature-interaction factorial study. Five folds and multiple
retrained arms take longer than a single training run; no T4 time guarantee is
claimed before measurement. CPU unit/integration tests and a real-archive
schema audit do not substitute for a GPU smoke run.

## Kaggle cell

```python
from urllib.request import urlopen
REF = "codex/persistent-wm-film-v1"
config = {
    "__name__": "__main__",
    "SOURCE_REF": REF,
    "RUN_DIR": "/kaggle/input/datasets/ainzkhail/2source2",
    "UPDATES_PER_QUESTION": 16,
    "DYNAMICS_UPDATES": 300,
    "FILM_STEPS": 200,
    "REAL_QUESTIONS_PER_FOLD": 4,
}
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{REF}/kaggle_persistent_wm_v2.py"
source = urlopen(url, timeout=90).read().decode("utf-8")
exec(compile(source, "kaggle_persistent_wm_v2.py", "exec"), config)
```

To run only offline phases, explicitly set `FILM_STEPS=0`; this skips the
drafter/verifier download and is labeled as disabled, not a completed FiLM test.
