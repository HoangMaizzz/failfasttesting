# Interactive world-model pretraining (Kaggle T4 x2)

This is **interact → real verifier labels → replay minibatches → gradient update**,
not a launcher for collecting an entire dataset before training. No old ZIP input
is required. Exploration is a fixed random R/E mixture, not an actor/RL algorithm
and not learning from deployed inference traffic.

## Models and objective

- GPU 0: Qwen/Qwen2.5-7B-Instruct FP16 verifier, full-context forward, KV disabled.
- GPU 1: Fast_dLLM_v2_1.5B FP16 native drafter, and the trainable world model.
- Drafter and verifier weights are frozen. Only the world model changes.
- Threshold 0.5, physical block 32, native segment/extension 8, proposal cap 64.
- STOP fills remaining masks with the native snapshot's same-forward top-1.
- E commits that entire candidate before generating the new segment.
- EOS ends a branch; confidence/acceptance is not an E eligibility gate.
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
- 50% R/E preference where both are legal; at 20% of those states try the other
  one-step action as an extra branch. Branches are restored by deterministic replay.
- Every new observable state receives a real verifier label; no greedy archive
  reference or world-model prediction is substituted for ground truth.
- Update after each training transition using replay minibatches of 8 sequences.
- 8 updates warm up current K, then one-step dynamics; after 24 updates allow h=1..3.
- Short/no-child paths have masked/truncated losses, never fabricated negative labels.

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
- compact per-question experience NPZ files with native hidden layers 14/28,
  proposal IDs, features, offsets and acceptance labels;
- error.txt on failure.

No verifier/drafter weights are packaged. Each completed question flushes a shard
and checkpoint. Python exceptions trigger a partial archive; an uncatchable runtime
kill cannot execute finally. The launcher does not delete any old output folder.
Checkpoint includes optimizer/RNG, but automated resume/replay reconstruction is
not implemented yet and is not claimed to work by rerunning this cell.

summary.json reports actual optimizer updates, parameter L2 change, held-out current
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
