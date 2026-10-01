# GSM8K 100-question random-action H3 training test

Execute `kaggle_twosource_pretrain.py` from this revision with GPU T4 x2 and
Internet enabled. No ZIP, mounted dataset, secret or prior session is needed.

## Execution and architecture

100 seeded GSM8K training-split questions are selected: 80 for weight updates,
20 for fixed heldout evaluation. The 20 heldout trajectories are collected first,
with zero optimizer updates. They never enter the training replay. This order
makes curves at 0/10/20/40/60/80 training questions comparable on identical roots.
The audit saves the exact sampled root IDs and real descendants; sampling includes
a quarter teacher-labelled roots, so it is not a natural-frequency prevalence survey.

The frozen 1.5B Fast-dLLM uses native Elysia snapshots and prefix KV inside each
generator invocation. R replays the segment to the next native snapshot and
asserts the predecessor; replay wall time is not deployment action latency.
R commits threshold-0.5 positions monotonically. E first fixes all remaining masks
using the current native forward's top-1 STOP candidate and opens 8 more tokens,
including its first native unmask. Up to 3 additional R steps per segment, proposal
cap 64. Random S/R/E have equal weights among legal actions. Proposal EOS forces
S; only verified emitted EOS ends an episode. No round/answer cap by default;
4096-token context guard reports partial on exhaustion, rather than silently
claiming a completed answer.

Actual S runs Qwen2.5-7B-Instruct, appends accepted tokens plus correction/bonus
and starts a new round. Hindsight labels visited STOP candidates against the
actual verified stream; censored candidates remain missing, never K=0. A separate
15% random shadow probe scores a state without submitting it or updating history.
This supplies additional current-state teacher targets; it adds verifier forwards.

The student is a 2-layer, 4-head Transformer, width 128, with native per-token
hidden layers [7,14,28], native/fill token embeddings, weighted top-K identity and
logit gaps, confidence, masks, positions, feature provenance/ages, prefix pooling
and native feature changes. The last 32 latent channels represent predicted
verifier agreement; the other 96 preserve native information. These are learned
roles, not guarantees of uniquely identifiable/disentangled factors.

A width-128 GRU re-encodes a bounded window of the preceding 8 ACTUAL STOP
summaries. Histories reset every episode, are copied into each observation before
its current verifier call, and are serialized for replay. Each summary has
8 outcome/context statistics, 32 projected pre-reject/pre-bonus verifier hidden
features and 8 projected emitted correction/bonus embedding features. Training
backpropagates through the GRU over this window; it is not an unbounded RSSM.

The teacher observes shifted candidate-versus-best-rival margin, actual candidate
probability, local teacher-forced agreement and 32 projected final-normalization
verifier hidden features. Projection is deterministic Rademacher seed 901; raw
full-dimensional verifier hidden states are NOT archived. A one-layer Transformer
encodes these features into a 32D teacher latent, trained with margin/agreement
targets; an EMA copy supplies student targets. Forward hooks restrict stored
rows to L+1 causal positions; no full-layer hidden tuple, KV or GPU graph is kept.
The verifier is FP16 sharded over both T4s at an 8-GiB/card weight placement cap,
leaving space for activations and the drafter/world model on GPU 1. Verifier KV
is disabled and its LM head produces only L+1 rows.

Only frozen LLM inference produces real experiences. Weight updates during
pretraining train the small model, using bounded replay and action-balanced root
edges. Current K NLL, imagined K NLL, privileged teacher loss, EMA latent dynamics
and structural mask/candidate losses are separate. The current encoder has an
explicit observation allowlist; current K, margin and verifier hidden never enter
it. H1 is used during warmup, then sequences up to H3. Each horizon step is ONE
R or E action, not an entire speculation round. True children are targets only;
steps 2/3 consume predicted latent states. `sampled_path_depth_counts` reports
actually sampled sequence depths, rather than only the configured upper limit.

## Factor experiment

Online inference lesions measure sensitivity of the trained model. Independently
retrained ablations use the same saved, bounded train subset, initialization seeds
[42,43], minibatch draws, update budget 400 and heldout roots. No LLM rerun occurs
during this stage; large models are released before fitting the small variants.
These are single-factor removals, not exhaustive factorial interactions.

Variants: full; no native hidden; no paired token IDs; no top-K identities/gaps;
no confidence; no mask features; no prefix embeddings; no native temporal feature
changes; no provenance/age indicators; no prior verifier memory; no teacher loss;
no verifier hidden in teacher/memory; no latent loss; no structure loss; no rollout
acceptance loss; no residual dynamics; omit each of layers 7/14/28; no physical
block alignment features; no explicit agreement-channel acceptance readout.
Exact lengths/masks remain legal transition metadata even in mask-feature ablation.
Loss/architecture removals are retrained rather than treated as inference lesions.
The offline fit has equal compute across variants and is distinct from the online
80-question weights; compare each variant to its own same-seed offline full.

## Outputs and interpretation

* `checkpoint.pt`, real experience NPZ shards, labels, actions, question IDs and
  actual/shadow teacher provenance: enough for later audits without LLM reruns.
* `evaluation/learning_*.json` and per-state predictions: H0/H1/H2/H3 MAE, RMSE,
  signed bias, p90/p95, within-1/2, K-mode accuracy, NLL, length/action/mask groups,
  latent cosine distance and mask Brier. Compare paired per-question change from
  initialization; heldout rows remain fixed across stages.
* Per-token prefix survival Brier/AUC/calibration, and separately local agreement
  Brier/AUC, scaled margin error and first-mismatch weakness rank. A local match
  after rejection is not an accepted token or a guaranteed corrected suffix.
* `factor_audit/comparison.csv`, per-variant JSON/predictions/checkpoints and
  paired question-bootstrap CIs; positive removal-minus-full MAE means the factor
  helped. Small sample counts and seed disagreement remain visible.
* `learning_and_factor_impact.png`, `learning_curve.json`, `READ_RESULTS.md`.

ZIP refresh is atomic after every completed question and each finished ablation,
then at final success or caught failure. ZIP is at `/kaggle/working` root and
contains this run's data/reports/small models only; large LLM weights/cache/source
stay in `/kaggle/temp`. A hard timeout can preserve the last completed ZIP, not
necessarily the current unfinished question. This measures acceptance/dynamics,
not production scheduler speedup or answer accuracy.

Local validation uses explicit fake LLM integration and tiny random Qwen forward
tests, including causal indexing, no current-teacher leakage, history reset and
padding, EOS without raw state, H3 prediction chaining, all ablation gradients and
artifact packaging. Full pretrained two-T4 execution must be tested on Kaggle.
