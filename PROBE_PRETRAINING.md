# Native random-probe pretraining: 10 GSM8K questions

This extends the existing interactive pretraining pipeline, not a replacement
collector or a rewrite of the native dLLM generator. No input ZIP is required.
Use `kaggle_probe_pretrain.py` on Kaggle with Internet and T4 x2 enabled.

## Exact interaction protocol

1. Seed 42 selects 10 questions from GSM8K train. Eight train the world model;
   two are held out. Run two independently sampled action trajectories per question.
2. GPU 1's frozen Fast_dLLM_v2_1.5B generates the first 8-token segment with
   native threshold 0.5. Capture the native snapshot and its same-forward top-1
   STOP candidate; do not run a separate STOP-fill forward.
3. Sample uniformly among legal S/R/E. R is at most **3 extra native unmask steps
   per segment**. E commits the current filled candidate and opens 8 new tokens,
   including their first native unmask; it resets the extra-R counter. E does not
   wait for all tokens to exceed threshold. Proposal length is at most **64**.
4. S alone calls GPU 0's frozen Qwen2.5-7B-Instruct verifier (FP16, no KV).
   Append accepted prefix plus correction/bonus to the verified answer. Start a
   new round; **64 limits one proposal, not the complete answer**.
5. Continue until the verifier emits EOS. A draft EOS only forces submission;
   it is not itself proof that the answer is done. Defaults have no round or
   answer-token cap (`MAX_ROUNDS_PER_QUESTION=MAX_NEW_TOKENS=0`).
6. Each visited state and executed R/E edge enters replay. Actual S supplies exact
   K; its emitted greedy stream may resolve earlier hypothetical STOP candidates.
   Unresolved labels remain unknown, not zero. Up to four minibatch updates run
   after each transition/STOP when supervised signal exists. No pre-collected
   dataset, controller training, latency prediction, or full branching tree.

The context safety guard defaults to 4096 total tokens. Reaching it before EOS,
an OOM, or a disk error produces a **partial/error** run, never a success falsely
labelled EOS. This is a resource guard, not an answer-length target. FP16/no-KV
verification can run out of T4 memory before this limit on long contexts.

Native R still uses the previous pipeline's bounded deterministic segment replay
and predecessor assertion, because the generator has no resumable iterator. KV
is used inside each drafter invocation, not carried between replay invocations.
These execution times are not production action-latency measurements.

## Trainable latent and objective

- Per-token latent width 128: 64 dimensions assigned to dynamics/mask targets,
  64 to verifier-agreement targets, with shared attention and a global latent.
  This is an inductive bias, not guaranteed independent/disentangled factors.
- Inputs: native hidden layers 7/14/28, frozen pretrained embeddings of native
  and STOP tokens, probability-weighted top-32 identity embeddings (normalized
  within top-k), logit gaps, masks, provenance/age, position, context, confidence
  changes, and prefix embeddings pooled into 8 memory slots. No current verifier
  answer or acceptance label is an input.
- Conditional prefix hazards predict the distribution over K=0..L. K excludes
  correction/bonus tokens. Report MAP exact-K, expected-K MAE and NLL.
- Action-conditioned R/E dynamics predict the next token/global latent. Training
  starts with current-K supervision, enables dynamics after 16 updates, then
  increases the maximum rollout horizon from 1 to 3 after 64 updates. Each rollout
  feeds its own predicted latent onward; real children supply targets only.
- Auxiliary targets: EMA-encoder latent consistency, native masks, projected
  frozen STOP-token embeddings, and candidate-vs-best-rival verifier margins.
  Margins come from the already executed actual-S forward; no extra verifier
  forward. After first mismatch these margins are conditional on the submitted
  draft, NOT ground-truth greedy continuation or independent acceptance labels.
- Active/new slots receive their own loss average so 56 unchanged old slots do
  not swamp an 8-token extension. This does not invent positive examples or
  classify refine as good/bad. Prefix hazards known structurally unchanged are
  carried across the imagined action.

Uniform S/R/E biases visits toward short proposals. Length 64 is legal, not
guaranteed coverage. Ten questions test mechanics; they cannot establish an
optimal latent, reliable generalization, or a converged controller. The prefix
memory currently uses token embeddings, not verifier hidden states. Next-state
candidate reconstruction predicts an embedding, not exact future token IDs.

## Saved outputs

The result ZIP is directly under `/kaggle/working`. It is atomically refreshed
after each completed question (both episodes) and finalized at the end. Hard
termination may lose work after the last completed-question ZIP; before the
first completed question no checkpoint ZIP is guaranteed.

- `checkpoint.pt`: model, EMA, optimizer and RNG state; no LLM weights. Reload
  model with `ProbeWorldModel(**checkpoint['model_config'])`. Automatic full-run
  resume is not implemented; replay can be rebuilt from artifacts.
- `states/edges/actions/rounds/questions.jsonl`, `labels.jsonl`, compact NPZ feature
  shards, `teacher_targets.jsonl`, training metrics and question split/config.
- `online_predictions.jsonl`: predictions recorded before label/update.
- `validation_predictions.jsonl`: held-out current-K and multi-step predictions,
  including predicted masks. Held-out states never update weights.
- `summary.json`: completion/error, counts, label coverage and evaluation. Replay
  is bounded to 512 states per split; evaluation covers retained holdout states.

Source and downloaded frozen models remain in `/kaggle/temp`, not the result ZIP.
Tests with fake LLMs check contracts only; an actual T4 x2 run is still required
to validate full-model memory, throughput and learning quality.
