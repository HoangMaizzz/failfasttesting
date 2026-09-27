# Interactive world-model pretraining (Kaggle T4 x2)

This is **interact → real verifier labels → replay minibatches → gradient update**,
not a launcher for collecting an entire dataset before training. No old ZIP input
is required. Exploration is a fixed random S/E/R mixture, not an actor/RL algorithm
and not learning from deployed inference traffic.

## Models and objective

- GPU 0: Qwen/Qwen2.5-7B-Instruct FP16 verifier, full-context forward, KV disabled.
- GPU 1: Fast_dLLM_v2_1.5B FP16 native drafter, and the trainable world model.
- Drafter and verifier weights are frozen. Only the world model changes.
- Threshold 0.5, physical block 32, native segment/extension 8, proposal cap 64.
- STOP fills remaining masks with the native snapshot's same-forward top-1.
- E commits that entire candidate before generating the new segment.
- E includes the first native unmask of its new segment; R is one additional unmask.
- A candidate containing EOS must STOP; the question ends only if verified output
  actually emits EOS. Confidence/acceptance is not an E eligibility gate.
- Target K is accepted prefix length, NOT emitted length including bonus/correction.
- There is no latency/policy/value head. Profile times are not training features.

An important execution limitation is explicit: the existing generator is not a
resumable Python iterator. For R we replay the frozen native segment only through
the requested next snapshot, check the previous snapshot's token IDs and forward
index, and keep only the genuine next edge. No future snapshot is supplied to the
learner. This costs extra native computation and is NOT a production-latency test.
Prefix KV is used inside each native invocation; it is not preserved across these
replay invocations. Direct incremental resume is a future performance improvement.

## Default 10-question smoke

- Random seeded GSM8K **train split** questions, IDs recorded in question_split.json.
- First 8 selected questions train; last 2 validate without weight updates.
- Up to 2 verification rounds per question, at most 128 emitted tokens per question.
- Within a round, E can grow to 64; up to 3 extra R steps per segment.
- One actual trajectory, no side branches or full tree. At each decision, random
  S/E/R have weight 1 each, renormalized over legal actions. All three legal means
  1/3 each; exhausted R leaves S/E at 1/2 each. Weights are configurable and positive.
- R is legal only with unresolved masks and fewer than 3 extra R steps since the
  segment started. E resets that counter. Missing native next snapshot disables R
  and resamples S/E; it does not create a zero-yield edge or force extension.
- STOP is legal at every boundary. At the cap E disappears; remaining legal R/S
  still compete. Reaching 64 is permitted, NOT guaranteed by random exploration.
- Capture native features and a same-forward STOP candidate at each visited state.
  **Only actual STOP calls the verifier.** Backfill earlier candidates from its
  emitted greedy stream, carrying pending candidates across subsequent real rounds.
- Train after each real R/E transition when a loss is available, and after each S
  backfill. Replay includes root-only STOP states, not just endpoints of edges.
- 8 updates warm up current K, then one-step dynamics; after 24 updates allow h=1..3.
- Warmup with no exact labels skips the optimizer (not a fake zero-label update).
  After warmup, unlabeled R/E edges can still train latent consistency; their
  acceptance losses stay masked. Validation questions never update weights.
- Short/no-child paths have masked/truncated losses, never fabricated negative labels.

## Hindsight labels and STOP semantics

Each state records a filled candidate D at a **verified root prefix** P. After real
STOP, append only the verifier's emitted accepted-prefix + correction/bonus tokens
to the question's confirmed stream G. A previous candidate's exact target is
`K = longest_common_prefix(D, G[len(P):])` only if a mismatch is already observed
or the reference covers all of D. Otherwise the matching reference length is only
a lower bound. More real rounds can resolve that label later. EOS, the output-token
budget or the smoke round cap can leave unresolved states; those remain `None` /
`-1` with `label_valid=False`, not zero or a fabricated exact K. Directly submitted
states retain the verifier's exact K even if emitted output is budget/EOS-clipped.

Never compare to the dataset answer or decoded answer string. Never use argmax
logits *after* the verifier's first mismatch as a greedy reference: those logits
were conditioned on rejected draft tokens. Only actual emitted tokens are safe.
Greedy decoding, the same frozen verifier/tokenizer, and the exact same prefix are
required. No stochastic acceptance replay is claimed. Direct and inferred exact
labels are checked for contradictions. No additional verifier call fills a gap.

Sampling S is **not** a supervised "optimal STOP" label. It diversifies actual
stopping depths and supplies K labels. The world model still learns current K and
R/E dynamics, with no policy head. A future planner will compare predicted yields
with external costs to decide when stopping is best. S ends the round; no latent
S transition is trained and sequences never cross a verification boundary.

Uniform S/E/R strongly favors short episodes, so a 10-question, 2-round smoke may
contain few long proposals or length-3 paths. Inspect action/length coverage; do
not equate a passing smoke with a trained 64-token planner. More rounds/questions
and an explicitly chosen exploration mixture are needed for substantial training.

This is an end-to-end pipeline smoke, not evidence of generalization, convergence,
full-answer accuracy or improved speculative-decoding speed. Increase independent
question coverage before making those claims. MATH mode uses MATH-500's test split
for local experiments; do not report trained-on questions as benchmark holdout.

## Representation

Native layers 14/28 (not diagnostic full-proposal re-encoding), native confidence,
top-32 logit gaps, current/STOP token IDs, mask/commit state and position/provenance.
Old native rows are cached, with missing validity and transition-age indicators.
Hidden and logits use separate native offsets; boundary captures may be partial.
Top-k logits are not renormalized into fake full-vocabulary confidence.

Frozen drafter embeddings are looked up at training time; they are not copied into
the world-model checkpoint. The encoder projects observations into L x 128 token
slots plus a global 128-vector. Encoder and shared R/E dynamics each have two
Transformer layers / four attention heads. E adds new query slots without true
child token features. A conditional prefix-match head defines a normalized P(K).

Loss = current-K NLL + imagined-K NLL + 0.1 * EMA latent consistency. Multistep
rollouts feed predicted latents back into dynamics, never true intermediate inputs.
Optimizer AdamW 3e-4, clipping 1, EMA 0.99. These are starting hyperparameters.
The current implementation has no uncertainty ensemble, anti-collapse guarantee,
future-action-legality head, deployed planner or automatic resumed-run command.

## Run

In Kaggle enable Internet and **GPU T4 x2**. Use a fresh cell:

```python
from urllib.request import urlopen

SOURCE_REF = "codex/sparse-extend-world-model"  # pin the delivered commit for reproducibility
NUM_QUESTIONS = 10
VALIDATION_QUESTIONS = 2
DATASET = "gsm8k"
MAX_ROUNDS_PER_QUESTION = 2
MAX_NEW_TOKENS = 128
STOP_WEIGHT = EXTEND_WEIGHT = REFINE_WEIGHT = 1.0
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{SOURCE_REF}/kaggle_world_model_pretrain.py"
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_world_model_pretrain.py", "exec"))
```

The launcher installs dependencies, fetches source, runs CPU tests, downloads model
weights outside `/kaggle/working`, and starts the real 2-GPU process. It deliberately
does not fall back to quantization, CPU offload or sharing verifier layers on GPU 1.
If GPU 0 lacks memory, use a clean session and inspect the partial error output.

## Artifacts

`wm_pretrain_gsm8k_10q_<timestamp>.zip` is placed directly at the working root.
It includes:

- checkpoint.pt: world-model weights, EMA encoder, optimizer and RNG states;
- config.json, question_split.json, summary.json and source hashes;
- training_metrics.jsonl, states.jsonl, edges.jsonl, rounds.jsonl, questions.jsonl;
- actions.jsonl: selected S/E/R, legal-action probabilities and whether executed;
- labels.jsonl: authoritative exact/censored label records joined by state_id;
- compact per-question experience NPZ files with native hidden layers 14/28,
  proposal IDs, features, offsets and acceptance labels;
- error.txt on failure.

`states.jsonl` is an immutable capture log, so its accepted_len is null at capture.
Join the latest `labels.jsonl` record to obtain the final target; derive edge yield
deltas only if **both** endpoint labels are valid. Per-question NPZ shards are
flushed after backfilling and include `accepted`, `label_valid` and lower bounds.
Unknown labels in a partial archive must also stay masked. `rounds.jsonl` records
the actual submitted candidate's K, emitted tokens and verifier profiling time.

No verifier/drafter weights are packaged. Each completed question flushes a shard
and checkpoint. Python exceptions trigger a partial archive; an uncatchable runtime
kill cannot execute finally. The launcher does not delete any old output folder.
Checkpoint includes optimizer/RNG, but automated resume/replay reconstruction is
not implemented yet and is not claimed to work by rerunning this cell.

summary.json reports exact/unresolved label coverage, actual verifier calls,
optimizer updates, parameter L2 change, held-out current
K NLL/MAE and imagined K MAE at depths 1..3 versus an unchanged-current-prediction
baseline. Most transitions keeping K constant can make overall errors misleading;
check gain-path counts and collect more examples before evaluating usefulness.

## Local verification

```text
python tests/test_world_model_pretraining.py
python tests/test_native_elysia_graph.py
```

The first suite includes a complete **10 synthetic-question** interaction/training/
packaging smoke on CPU. Fake LLMs are explicitly named; it is not a substitute for
running the real Qwen/dLLM smoke on Kaggle.
