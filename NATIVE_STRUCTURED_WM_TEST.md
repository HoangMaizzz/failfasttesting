# Native-structured latent feasibility test (100 GSM8K)

This test answers a narrower and falsifiable question than “can a small model
replace Fast-dLLM?”: do Fast-dLLM's own intermediate states and native scheduler
state provide enough signal for a learned latent to predict verifier acceptance
now and after a short R/E rollout?

## What runs

1. The original Fast-dLLM 1.5B remains frozen and executes its native generator:
   threshold 0.5, logical extend 8, proposal cap 64, physical block 32, at most
   three extra R steps per segment, and native EOS behavior. The verifier is the
   original Qwen2.5-7B-Instruct greedy verifier. They stay on the prior two-GPU
   layout (dLLM on GPU 1; verifier sharded over both cards).
2. At each visited boundary, the collector records native hidden layers 7/14/28,
   the pretrained dLLM token embeddings for native and same-forward STOP content,
   top-k/logit-gap summaries, masks/commits, newly-unmasked positions, block
   geometry, prefix token embeddings, and provenance/validity. It does not use
   random token identity codes and does not feed verifier answers into the input.
3. A seeded policy samples uniformly among currently legal STOP, REFINE and
   EXTEND actions. STOP submits the current same-forward candidate to the real
   verifier. R/E execute the native shell and yield real child observations.
   The run follows one trajectory per question; it does not construct a tree.
4. A structured model fuses native layers by a learned layer mixer, attends to
   the prefix, and uses separate R and E transition stacks. E marks the previous
   top-1 materialized candidate committed and creates eight new latent slots;
   R keeps proposal length and can only reduce its predicted mask probability.
   The transition also predicts which positions unmask on the next action. A
   normalized distribution over K=0..L yields each position's prefix-acceptance
   probability P(K >= i). No exact token generator or latency head is trained.
5. Training updates happen online from real native transitions and verified
   labels. The final twenty questions are held out from optimizer updates. A
   fixed set of validation roots is evaluated at zero and at configured training
   milestones; H1/H2/H3 denote individual real action edges unrolled by the
   learned latent. After training, matched inference-only lesions remove hidden,
   content, candidate, prefix, logit, geometry or history inputs, and compare
   shared action-agnostic dynamics. These lesions use saved validation states and
   make no extra LLM calls. They measure sensitivity, not causal feature value.

## Labels and reported results

The primary target is the verifier's accepted draft-prefix length K, excluding
the correction/bonus token. It is derived from the real verifier's greedy output
and checked against actual submitted STOPs. A position target is 1 for each
position before K and 0 after K only when an exact mismatch is observed; labels
that are not resolved by the verified stream remain censored, never relabeled as
zero. Predictions are a monotone prefix survival curve derived from P(K).

The ZIP includes current-state and H1-H3 K MAE/RMSE/bias, within-one/two accuracy,
K NLL, per-token prefix Brier/log-loss/AUC/ECE, next-unmask Brier, confidence
and persistence baselines, error by proposal length/action sequence/mask count, learning curves,
paired question-bootstrap deltas, feature-sensitivity reports, state/edge/action/
label ledgers, model checkpoints at milestones and source hashes. Question-level
resampling avoids treating correlated token positions as independent examples.

Interpretation gates:

- If the native-feature reader cannot beat length-only/current-persistence
  baselines on held-out H0, improve state capture/readout before blaming dynamics.
- If H0 works but H1-H3 collapse, the action-conditioned latent transition is
  the bottleneck; compare R and E separately and inspect mask-state and next-unmask prediction.
- If full model works but a lesion barely changes held-out scores, that input
  group may be redundant in this setup. Lesions alone do not establish causality.
- This run is a feasibility experiment, not a production speedup, a full dLLM
  emulator, or final statistical evidence. It uses 80 training and 20 validation
  questions (one seeded split); repeat with other question seeds before claiming
  generalization.

## Kaggle setup

Enable Internet and GPU T4 x2. No data ZIP or GitHub secret is required for a
fresh run. It downloads the source, GSM8K, and model checkpoints, runs CPU unit
tests, executes the experiment and puts a verified ZIP in `/kaggle/working` after
each completed question. The 100-question uncapped-to-EOS run can take many hours
and may exceed one Kaggle session; a question-boundary partial ZIP is kept.

To resume a partial run, attach its ZIP as a Kaggle input or keep it at its
original `/kaggle/working` path, then set `RESUME_ARCHIVE` to that ZIP's path in
the notebook cell. Resume restores the last atomic model/optimizer checkpoint
and completed-question ledger, then restarts the interrupted question. Runs with
no completed-question ledger cannot be resumed safely.
