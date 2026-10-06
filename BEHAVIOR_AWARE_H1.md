# Paired behavior-aware H1 latent dynamics

This study reuses the full100 Phase 0 question split and its frozen
128-dimensional encoder, verifier `G`, reconstruction head, and latent cache.
It trains independent small dynamics/readout models with paired initialization
and batches. There are **no LLM downloads or LLM forwards**, no encoder
retraining, and no token-generation objective. Source-code checks and a small
train-only encoder/cache verification are correctness checks, not retraining.

## Required inputs

Keep the original experience mounted at
`/kaggle/input/datasets/ainzkhail/2source2` (or set `RUN_DIR`). Also upload the
**new approximately 176 MB Phase 0 result ZIP**, or use **Add Notebook Output**
for the matching completed Phase 0 run. The original `2source2` upload alone
does not contain the required frozen source.

The checked source result was named
`latent_wm_phase0_full_20261006_095849_320025_ICT_1zjkyml6.zip` and had Stage A
128 passing, H1 failing, and a complete frozen latent cache. This filename is
an example, not a required name. A failed Phase 0 H1 gate is the motivation for
this new study and does not invalidate its frozen Stage A source.

`PHASE0_INPUT` defaults to `/kaggle/input`. The new
`behavior_aware_source.resolve_phase0_input` scans by contents and requires
exactly one matching result: summary schema `latent_world_model_phase0_v1`,
`config.num_questions=100`, pretrained input embeddings, a frozen latent128
checkpoint/cache, preprocessing, provenance, and the exact question split
70/15/15. Arbitrarily named ZIPs, wrapped ZIP roots, and extracted output
folders work. If multiple matching results are mounted, set `PHASE0_INPUT` to
the exact ZIP or extracted result. Do not substitute an older report-only ZIP
or a dynamics checkpoint for the complete Phase 0 source.

`RUN_DIR` independently accepts an original extracted experience folder or ZIP
by content. Its filename and extension need not match a notebook example.
The runner checks raw-trace content, source code, frozen weights, and cached
latents against the supplied Phase 0 provenance; input discovery alone is not
an integrity guarantee.

## Fresh Kaggle cell

Choose **T4 x2** and enable Internet to fetch source and install `numpy`.
PyTorch is preinstalled; the report metrics need no scikit-learn. Transformers, Hugging Face packages,
and model downloads are unnecessary. Publish the source branch/commit before
running this cell; for a reproducible study, replace `SOURCE_REF` with the
published commit SHA. The launcher fetches that exact ref and records the
resolved commit. Resume with the same commit.

```python
import os
import sys
import types
from urllib.parse import quote
from urllib.request import urlopen

os.chdir('/kaggle/working')
RUN_DIR = globals().get('RUN_DIR', '/kaggle/input/datasets/ainzkhail/2source2')
PHASE0_INPUT = globals().get('PHASE0_INPUT', '/kaggle/input')
SOURCE_REF = globals().get('SOURCE_REF', 'codex/behavior-aware-latent-dynamics')
MODE = globals().get('MODE', 'full')
# PHASE0_INPUT = '/kaggle/input/my-phase0-output/result.zip'  # or extracted folder
# RESUME_INPUT = '/kaggle/input/my-behavior-results/result.zip'  # optional

base = ('https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/'
        + quote(SOURCE_REF, safe='/') + '/')
def read_source(filename):
    with urlopen(base + filename, timeout=120) as response:
        return response.read().decode('utf-8')

# Import-safe Phase 0 helpers are needed even in an otherwise fresh notebook.
helpers = types.ModuleType('kaggle_latent_wm_phase0')
exec(compile(read_source('kaggle_latent_wm_phase0.py'),
             'kaggle_latent_wm_phase0.py', 'exec'), helpers.__dict__)
sys.modules[helpers.__name__] = helpers
launcher = {'__name__': '_behavior_aware_notebook_launcher'}
exec(compile(read_source('kaggle_behavior_aware_h1.py'),
             'kaggle_behavior_aware_h1.py', 'exec'), launcher)
OUTPUT = launcher['launch'](globals())
```

The two bootstrap files come from the same requested ref; neither import
starts Phase 0 training. The behavior launcher uses the existing import-safe
helpers for source fetching, preflight, content discovery, GPU checks, safe
resume extraction, and partial packaging. It never invokes the old study.
It fetches source into a unique directory under `/kaggle/temp`, changes to
`/kaggle/working` first, and does not delete an existing working directory.

## Configuration and paired comparisons

`configs/behavior_aware_h1.json` defines the full study. The effective JSON is
written to `<OUTPUT>/launcher_config.json`, and the only study CLI invoked is:

```text
python -u run_behavior_aware_h1.py --input ORIGINAL_TRACE --phase0_input SELECTED_PHASE0_SOURCE --output OUTPUT --config EFFECTIVE_JSON [--resume]
```

| Setting | Full | Smoke |
| --- | --- | --- |
| Frozen source / question split | latent128; 100 questions, 70/15/15 | unchanged |
| `seeds` | `[42,43,44]` | `[42]` |
| `max_updates` | 2400 | 8 |
| `pilot_updates` | 1200 | 4 |
| `eval_every` | 100 | 4 |
| `lambda_grid` | six pairs below | first two pairs |
| `batch_size`, `workers` | 16, 2 | unchanged |
| `bootstrap_samples` | 2000 | unchanged |

The six `(lambda_q, lambda_K)` pairs are `(0.1,0.1)`, `(0.3,0.1)`, `(0.1,0.3)`,
`(0.3,0.3)`, `(1.0,0.3)`, and `(0.3,1.0)`. Smoke reduces the training and pilot
budget only; it does not switch to 20 questions, latent64, learned embeddings,
or a freshly trained encoder. Smoke results support pipeline checks and no
feasibility verdict.

The runner schedules at most two standalone training jobs at once, one on
`cuda:0`, one on `cuda:1`. Each owns its small trainable model and frozen
verifier/reconstruction modules. Worker placement is not an edit to the
frozen Phase 0 configuration or its source hashes. There is no automatic
single-GPU fallback. Local launcher tests can explicitly use `ALLOW_CPU=True`
with `WORKING_DIR`/`TEMP_DIR`; full study placement remains the runner's job.

A uses the unchanged Phase 0 native/latent state loss. B1 adds behavioral
consistency with detached `G(real child)` survival/K targets. B2 adds observed
Qwen verifier truth. C is a direct source/action-to-future verifier challenge
trained separately. The imagined-input path through frozen `G` must retain
input gradients for B1/B2 while `G` parameters and reconstruction weights
remain frozen. Missing labels are masked, not filled with negatives. True
parent K is loss/evaluation metadata, never a future-truth model input.

Train H1 only. Validate on fixed question groups and challenging R-nonzero and
E-full-prefix cohorts. B1/B2 pilots select lambda pairs using validation and
the state-fidelity restriction relative to A; if no pilot is eligible, report
that result without relaxing the gate. Lock choices before opening test.
Final A/B1/B2 seeds must share initialization and batch schedules with audit
artifacts, not just a common seed number. Report H1 and free H2/H3 rollout
generalization where applicable; C is an H1 challenge. Distinguish real truth,
real-child oracle `G`, learned imagined behavior, and direct prediction.
Low-budget or reused-question results do not prove convergence, impossibility,
or a controller speedup.

## Runtime and storage estimates

The previous full Phase 0 run took about **532 seconds (9 minutes)**. That is
context, not the duration of this new study. Full behavior-aware training can
run **12 pilots** and **12 final jobs** (four systems × three seeds), up to
**43,200 optimizer steps**: `12×1200 + 12×2400`. Two workers run in parallel.

The **planning estimate is 1–2 hours on Kaggle T4 x2**, not a measured
Kaggle runtime. A local RTX 3060 laptop check using the actual 100-question
traces ran 100 updates plus complete H1 validation in about 24–26 seconds
for A/B1/B2, and 9 seconds for C. This is a pipeline/timing check, not a
converged scientific result or a T4 benchmark. Validation, checkpoint compression, report generation,
ineligible pilot families, and hardware affect wall time. This is substantially
more work than the earlier Phase 0 run; a smoke timing is not a full-study
runtime measurement.

Phase 0 extraction, native assets, raw-input caches, frozen latent worker
payloads, and source code stay under temporary/input storage, outside published
working outputs. Results contain new worker checkpoints/optimizer/RNG states,
learning curves, selections, predictions, provenance, and reports. Retaining
checkpoints at every 100 full-mode updates can yield a **rough storage estimate
of 600 MB–2 GB** for the result ZIP; this is not a measured final artifact size.
The original trace, Phase 0 cache, and native embedding table are not copied
into it.

## Checkpoints, errors, and resume

Every matching `tests/test_behavior_aware_*.py` in the fetched ref is executed
as a standalone script before the experiment, including the required launcher
tests. This preserves the tests-directory import path for shared fixtures.
Launcher tests use local fixtures/mocks and perform no training or downloads.

The output is a unique folder directly beneath `/kaggle/working`. Its sibling
ZIP is at the working root; download links use only the ZIP basename. The
runner saves worker model snapshots and `resume.pt` with optimizer, sampler,
CPU/CUDA RNG, and progress state every evaluation interval (100 updates in
full mode). The runner packages completed jobs/stages and errors. The launcher
uses that ZIP when available and atomically packages compact outputs if it is
missing or unreadable, including preflight/test errors. Fallback packaging
excludes raw and Phase 0 assets, latent caches, NPZs, temporary files, and
individual files above 2 GiB; exclusions are recorded. It permits optimizer
checkpoints larger than the old Phase 0 fallback's 128 MiB cutoff.

Set `RESUME_INPUT` to the prior **behavior-aware** result ZIP or extracted
result. Keep both `RUN_DIR` and `PHASE0_INPUT` pointing at the original inputs,
and retain the same effective configuration and source commit. The launcher
validates the result manifest/config/checkpoint structure, safely restores it
to a fresh output folder, and passes `--resume`. The runner verifies the strict
content-based source/config/code/input fingerprint, reuses completed jobs, and
resumes interrupted workers from their last saved optimizer/RNG checkpoint.
The launcher does not validate optimizer payloads or make an independent
training-correctness claim; that behavior is the runner's responsibility.

Inspect `study_manifest.json`, `source_checkpoint_manifest.json`,
`split_manifest.json`, `selection_protocol.json`, `lambda_selection.json`,
`paired_training_audit.json`, `test_access_manifest.json`, and `summary.json`.
The method folders (`A_pure`, `B1_consistency`, `B2_qwen`, `C_direct`) contain
per-job selections, checkpoint histories, learning curves, and predictions.
A downloadable partial ZIP is not a completed study.
