# Retrained representation/dynamics study

Run this against the **original 100-question training ZIP**, not a compact
feature-audit or learning-curve ZIP. The existing `math100` Kaggle mount works.
The archive's 80 train / 20 validation split is preserved by question. All arms
start fresh; the original checkpoint supplies architecture, never trained weights.
No new question generation or LLM inference is performed.

## Experimental arms

The default runs 15 arms, each on seeds 42/43/44 with updates 0/64/128/256/512:

| Arm | Question tested |
|---|---|
| baseline | Previous token-dual model and offline losses, without teacher supervision |
| teacher_only | What does restoring verifier teacher targets add? |
| attention_only | Does attention over individual top-K token/logit pairs beat weighted pooling? |
| residual_only | Does a gated residual transition with mask conditioning help? |
| delta_only | Does an extra Smooth-L1 loss on predicted change in K help? |
| improved | Combine the four changes above; this name is a hypothesis, not a claim of superiority |
| no_attention / no_residual / no_delta / no_teacher | Remove one change from the combined arm |
| no_latent_loss / no_structure_loss | Remove one auxiliary loss from the combined arm |
| no_hidden / no_gaps / no_prefix_history | Retrain combined arm with that input channel removed |

All common tensors are initialized identically for each seed; replay sampling,
data, batch size and optimizer steps match. New modules have deterministically
seeded initial weights. Architectural parameter counts and training/evaluation
times are recorded because equal steps do not mean identical compute.

The old offline loader did not restore teacher margins. The new loader joins
late `teacher_targets.jsonl` by state ID over the shard snapshot, counts coverage,
and validates shapes. They are targets only, never observations. Baseline sets
teacher weight to zero to retain the previous offline setting; teacher-only
isolates the effect of restoring this supervision.

## Evaluation

- Every checkpoint uses the same current-state and R/E edge panels, sampled
  deterministically per question. Edge panels can start anywhere in a trajectory.
- Every checkpoint also evaluates ALL changed-K validation edges as a separate
  diagnostic, and fixed real 2/3-step paths. Child observations never enter an
  imagined rollout. The trajectory graph retains all outgoing branches.
- The final checkpoint additionally evaluates ALL labeled validation states and
  ALL labeled R/E edges. Final full-cohort metrics are distinct from panel curves.
- Report MAE, question-macro MAE, P90, within-1/2 tokens, bias, acceptance NLL,
  delta-K MAE, prediction-persistence MAE, active-token latent cosine, mask Brier,
  proposal-length/change groups, and coverage counts.
- A separate fixed training-state panel shows train/validation divergence.
- Factor comparisons pair identical questions and seeds. Confidence intervals
  bootstrap questions after averaging seeds; per-seed differences are also saved.
  This does not estimate all interactions or correct multiple comparisons.
- Observe validation error and paired learning progress, not train loss alone.
  Flat R persistence deltas should not be interpreted as strong refinement dynamics.

The previous validation set has already informed design choices. These results
are exploratory development comparisons, not a new independent test-set claim.
Run a new held-out question set before claiming production improvement.

## Kaggle cell

Enable GPU and Internet, attach original training data, and run:

```python
from urllib.request import urlopen
REF = "codex/sparse-extend-world-model"  # Pin the published commit for reproducibility.
config = {
    "SOURCE_REF": REF,
    "INPUT_PATH": "/kaggle/input/datasets/yumesakihikari/math100",
    "VARIANTS": "all",
    "SEEDS": "42,43,44",
    "STEPS": "0,64,128,256,512",
    "BATCH_SIZE": 8,
    "EVAL_BATCH_SIZE": 8,
}
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{REF}/kaggle_world_model_improvement_test.py"
exec(compile(urlopen(url, timeout=60).read().decode(), "kaggle_world_model_improvement_test.py", "exec"), config)
```

45 fits x 512 updates = 23,040 small-model updates. Compared with the earlier
9,600-update run, this is more work and performs substantially broader evaluation;
use printed train/eval times to estimate completion instead of assuming 25 minutes.
One GPU is used; no 7B verifier is loaded. A frozen drafter embedding weight file
may be downloaded (and tensor hash checked); it is not included in output.

For a pipeline smoke only, set `VARIANTS="baseline,improved"`, `SEEDS="42"`,
`STEPS="0,16,32"`. That does not evaluate factor effects or convergence.

## Outputs and recovery

`world_model_improvement_<timestamp>.zip` is at `/kaggle/working` root. It is updated
atomically after every completed evaluation and on Python exceptions, including
partial status/error. A hard container kill can preserve only the last packaged
checkpoint if Kaggle retains outputs; the code cannot guarantee platform retention.

Outputs include `report.md`, `curves.json`, `paired_factor_impacts.json`,
`paired_learning_progress.json`, charts, evaluation plan/hash, every component
loss by update, compressed individual predictions, and final baseline/combined
weights for each seed. No original raw shards, embedding weights, or LLM cache
are copied into the ZIP. Intermediate models are not saved; this runner does
not resume interrupted optimizer state.
