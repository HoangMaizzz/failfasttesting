# Paired native latent feasibility: projected32 pilot

This is a **projected32 Qwen teacher + STOP embedding native pilot**, using
saved paired traces and the frozen Phase 0 latent128 source. Run native probes
first, then bridge, direct and distillation tests. The optional existing frozen
Phase 0 dynamics H1–H3 rollout is a diagnostic. It does not train new dynamics.
The pilot does not establish a full Qwen hidden-layer ceiling, actual planner
regret, or partial-real timing. Smoke results check the pipeline only.
There are no LLM downloads or LLM forwards.

## Required Kaggle inputs

Mount **both** the original trace (approximately 1.4 GB) and the original
completed **Phase 0 result ZIP (approximately 176 MB)**:

- `RUN_DIR=/kaggle/input/datasets/ainzkhail/2source2`
- `PHASE0_INPUT=/kaggle/input/datasets/ainzkhail/phase0`

Do **not** use the latest behavior-aware result ZIP as `PHASE0_INPUT`.
The original trace alone also cannot supply the frozen source. Both parameters
accept arbitrarily named ZIP files, dataset mount folders containing a unique
matching ZIP, or extracted folders, including wrapped result roots. Selection
uses contents rather than a hardcoded filename. Set an exact path if multiple
matching inputs are mounted.

The Phase 0 reader is loaded from `behavior_aware_source.py` in the newly
downloaded source checkout. It requires the pretrained full100 Phase 0 schema,
latent128 Stage A checkpoint, preprocessing, frozen latents, provenance, and an
exact 70/15/15 question split. The runner verifies source, input and frozen-cache
signatures; file discovery alone does not certify them.

## Copy cell

Choose Kaggle T4 x2 and enable Internet for source fetching and `numpy`.
PyTorch is preinstalled. Publish the source ref before running; pin
`SOURCE_REF` to the published commit SHA for a reproducible run and resume.
The cell reads only the launcher URL. On a fresh runtime, the launcher fetches
one checkout under `/kaggle/temp`, imports its sibling helpers, and reuses that
checkout and verified Git HEAD for the run. The requested ref and resolved
revision are recorded in `launcher_metadata.json`. No initial Git cell or
separate helper downloads are needed. Ordinary imports defer missing helpers
until `launch()` and never fetch source or query GPUs.

```python
from urllib.parse import quote
from urllib.request import urlopen

SOURCE_REF = globals().get('SOURCE_REF', 'codex/paired-native-latent-feasibility')
launcher = dict(globals(), __name__='__main__', SOURCE_REF=SOURCE_REF,
    RUN_DIR=globals().get('RUN_DIR', '/kaggle/input/datasets/ainzkhail/2source2'),
    PHASE0_INPUT=globals().get('PHASE0_INPUT', '/kaggle/input/datasets/ainzkhail/phase0'),
    MODE=globals().get('MODE', 'full'), NUM_QUESTIONS=globals().get('NUM_QUESTIONS', 100),
    DEVICES=globals().get('DEVICES', ['cuda:0', 'cuda:1']),
    SMOKE_CONFIG=globals().get('SMOKE_CONFIG', {}),
    RESUME_INPUT=globals().get('RESUME_INPUT', None))
url = ('https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/'
       + quote(SOURCE_REF, safe='/') + '/kaggle_paired_native_latent.py')
with urlopen(url, timeout=120) as response:
    exec(compile(response.read().decode('utf-8'), 'kaggle_paired_native_latent.py', 'exec'), launcher)
OUTPUT = launcher['OUTPUT']
```

Helper imports start no study and query no GPUs. The launcher changes to
`/kaggle/working` before cloning into a unique `/kaggle/temp` directory. It
fetches the requested branch or commit into a detached checkout, installs only
`numpy`, discovers and runs `tests/test_paired_latent*.py` plus the focused
launcher tests from that checkout, and invokes:

```text
python -u run_paired_native_latent.py --input ORIGINAL_TRACE --phase0_input RESOLVED_PHASE0 --output OUTPUT --config OUTPUT/launcher_config.json [--resume]
```

## Configuration

`configs/paired_native_latent.json` and the runner are owned by the main study.
The launcher saves its effective configuration as `launcher_config.json`.
Full mode preserves existing source config fields and budgets; the runner
validates them. Missing standard fields use these defaults:

| Field | Default full | Default smoke |
| --- | --- | --- |
| `schema`, `latent_dim` | `paired_native_latent_v1`, 128 | unchanged |
| `num_questions`, question split | 100, exact 70/15/15 | unchanged |
| `seeds`, `workers` | `[42,43,44]`, 2 | `[42]`, 2 |
| `oracle_updates`, `max_updates`, `eval_every` | 1000, 1500, 100 | 8, 12, 4 |
| `batch_size`, `learning_rate` | 32, 0.0003 | source values |
| `bootstrap_samples` | 2000 | 100 |
| `encoder_verification_samples` | 32 | source value |
| `rollout_existing_dynamics` | true | source value |
| `bridge_state_weight`, `bridge_behavior_weight`, `distill_weight` | 1.0, 0.5, 0.3 | source values |
| `pipeline_check_only` | false | true |

`SMOKE_CONFIG` may customize smoke updates, evaluation frequency, seeds and
bootstrap count. Every smoke run keeps `pipeline_check_only=true` and retains
the full question population and split. `NUM_QUESTIONS` is exposed for clarity
and must remain 100. No launcher equality check freezes the main study's full
training budgets to the values in this table.

At the default full budgets, each seed trains three native oracle/probe models
for 1,000 updates each and four students for 1,500 updates each: 9,000 updates
per seed, **27,000 updates across three seeds**. This is a budget count, not a
runtime or convergence estimate. The paired source audit reported 4,966
teacher-covered states: 3,474 train, 786 validation, and 706 test. State counts
do not change the exact 100-question 70/15/15 split.

Typically two GPU workers execute separate seed jobs; three seeds do not
require three GPUs. Optional `DEVICES='cpu'` or `DEVICES=['cpu']` assigns every worker to CPU, or
use `['cuda:0','cuda:1']` (also accepted as `'cuda:0,cuda:1'`). The runner
validates device availability. Worker placement is saved in the effective
configuration, so keep it unchanged for an exact resume.
Device lists are limited to the configured worker count. A one-worker source
configuration uses one device; a single CPU entry is repeated for a two-worker
configuration. The launcher writes the selected list into `config['devices']`.

## Results and resume

The download ZIP is written atomically in the `/kaggle/working` root alongside
the fresh output folder. It includes configuration, source revision metadata,
study/split manifests, reports, predictions JSONL, logs, job `best.pt` selected
models, and `last.pt` model/optimizer/RNG resume checkpoints produced by the runner. Caches,
input/source trees, `preprocessing.pt`, `frozen_latents.pt`, original raw JSONL,
experience NPZ, nested ZIPs, and model embedding assets are excluded. The
original 1.4 GB trace is never copied into the result ZIP. Frozen source assets
remain in their input mounts or temporary cache.

On failure the launcher saves `launcher_error.txt`, merges available partial
results with any earlier runner ZIP, filters source assets from both, and
re-raises the error. The runner's exact output-folder prefix is stripped before
merging so current files replace earlier copies and the ZIP has one result
root for resume; unrelated archived reports retain their paths. Packaging uses
a temporary ZIP and atomic replacement; a
packaging failure leaves the previous ZIP available.

Set `RESUME_INPUT` to a previous paired native result ZIP or extracted folder
(a mount containing one result ZIP also works). It must contain `config.json`,
`study_manifest.json` with a source/config/input signature, and saved job
checkpoints under `jobs/`. The launcher restores compact files to a fresh
output, checks effective configuration equality, and passes `--resume`.
The runner enforces the exact signature and checks optimizer/RNG payloads.
Keep the same published commit, original trace, Phase 0 source, mode, smoke
overrides and worker devices. Resume does not replace either required input.

The main study's measured local GPU full-population smoke completed seven
models and 72 updates, produced 10,114 prediction rows, and took approximately
151 seconds; its result ZIP was 62.36 MB. These are observed pipeline-check
measurements, not full-run results or convergence evidence. Use the main
study's final measured benchmark for full runtime planning; no full runtime
estimate is asserted here.

Local sidecar verification (no downloads, GPU queries or training):

```text
python tests/test_paired_native_launcher.py
```

Also run the main study's new data/model/metrics/runner tests as they arrive:

```text
python -m unittest discover -s tests -p "test_paired_latent*.py"
```
