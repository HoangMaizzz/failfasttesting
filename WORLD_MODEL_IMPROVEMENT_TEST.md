# Retrained representation/dynamics study

Run this against the **original 100-question training ZIP**, not a compact
feature-audit or learning-curve ZIP. The existing `math100` Kaggle mount works.
The archive's 80 train / 20 validation split is preserved by question. All arms
start fresh; the original checkpoint supplies architecture, never trained weights.
No new question generation or LLM inference is performed.

## Experimental arms

The full `all` run includes 18 arms, each on seeds 42/43/44 with updates 0/64/128/256/512:

| Arm | Question tested |
|---|---|
| baseline | Previous token-dual model and offline losses, without teacher supervision |
| teacher_only | What does restoring verifier teacher targets add? |
| attention_only | Does attention over individual top-K token/logit pairs beat weighted pooling? |
| residual_only | Does a gated residual transition with mask conditioning help? |
| delta_only | Does an extra Smooth-L1 loss on predicted change in K help? |
| improved | Combine the four changes above; this name is a hypothesis, not a claim of superiority |
| action_balanced | Keep the improved model/losses, but sample root R/E edges equally |
| change_balanced | Also sample K-gain / unchanged / K-loss root edges equally within each action |
| change_balanced_delta | Same balanced sampler, with the ΔK loss weight raised from 0.1 to 0.3 |
| no_attention / no_residual / no_delta / no_teacher | Remove one change from the combined arm |
| no_latent_loss / no_structure_loss | Remove one auxiliary loss from the combined arm |
| no_hidden / no_gaps / no_prefix_history | Retrain combined arm with that input channel removed |

All model arms use matched initialization and optimizer budgets. New modules
have deterministically seeded initial weights. The targeted sampler arms
deliberately vary root-edge sampling as described below. Architectural parameter
counts and training/evaluation times are recorded because equal steps do not
mean identical compute.

## Targeted follow-up: does balanced replay teach K changes better?

Use the same 100-question MATH archive. Run only `improved,action_balanced,change_balanced,change_balanced_delta`, seeds `42,43,44`, and updates `0,64,128,256,512`. This is 6,144 small-model optimizer updates and does not run the drafter or verifier.

All arms share the same architecture, question split, initialization, and update budget. Only the root transition sampler changes, apart from the explicitly named 0.3 ΔK loss arm. Current-state examples use a separate seeded RNG, so changing edge sampling does not alter their sampling stream. Balanced modes choose an action uniformly among available actions; `change_balanced` then chooses among available K-change classes uniformly within that action. Later rollout steps follow actual outgoing graph edges, preserving a real trajectory. Verifier labels are used only to stratify training root edges; no validation label enters training or sampling.

The output report compares each sampler with `improved` on the unchanged validation cohort and separately reports R/E gain, same, and loss groups. It also records sampled root-edge counts so balancing can be verified. Look for lower gain-group MAE across seeds without worsening natural full-cohort or current-state error. This remains exploratory because the same 20-question validation split has already been inspected.

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
    "VARIANTS": "improved,action_balanced,change_balanced,change_balanced_delta",
    "SEEDS": "42,43,44",
    "STEPS": "0,64,128,256,512",
    "BATCH_SIZE": 8,
    "EVAL_BATCH_SIZE": 8,
}
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{REF}/kaggle_world_model_improvement_test.py"
exec(compile(urlopen(url, timeout=60).read().decode(), "kaggle_world_model_improvement_test.py", "exec"), config)
```

This targeted comparison is 12 fits x 512 updates = 6,144 small-model updates.
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
