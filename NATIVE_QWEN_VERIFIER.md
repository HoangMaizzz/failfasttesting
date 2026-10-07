# Native Qwen verifier: ceiling and compression study

This study captures uncompressed, causal hidden states from the actual frozen
Qwen verifier, compares small survival probes across native layers, and learns
64/128/256D verifier latents on the two layers selected using validation.
The direct baseline uses the existing frozen 128D Drafter representation on
the same captured states. This task trains no Drafter dynamics, D-to-V bridge,
controller or planner. It does not reuse the previous projected32 Qwen teacher.

The runner reads the historical model ID and revision from the actual original
input summary and validates them. The launcher supplies no assumed model ID.
Capture must reproduce every saved accepted-prefix length K, with mismatch
tolerance 0.0, and pass the causal hidden-to-logit alignment check before probe
training. Candidate embeddings come from Qwen's own frozen embedding table;
final logits and future labels must never become features.

## Fresh Kaggle Safe Version cell

Mount both required datasets, select **T4 x2**, enable Internet, and paste this
single cell in a fresh notebook. Source must be published before running.
`SOURCE_REF` accepts the branch below or a published commit SHA. Pin a SHA for
a reproducible run and exact resume. This cell downloads one launcher URL;
the launcher creates one detached Git checkout under `/kaggle/temp` and reuses
it for helpers, tests, configuration and the runner. There is no separate clone
cell or helper-file download. An ordinary import performs no downloads, GPU
queries, directory changes or experiment execution.

```python
from urllib.parse import quote
from urllib.request import urlopen

SOURCE_REF = globals().get('SOURCE_REF', 'codex/native-qwen-verifier-latent')
launcher = dict(globals(), __name__='__main__', SOURCE_REF=SOURCE_REF,
    RUN_DIR=globals().get('RUN_DIR', '/kaggle/input/datasets/ainzkhail/2source2'),
    PHASE0_INPUT=globals().get('PHASE0_INPUT', '/kaggle/input/datasets/ainzkhail/phase0'),
    MODE=globals().get('MODE', 'full'),
    NUM_QUESTIONS=globals().get('NUM_QUESTIONS', 100),
    CAPTURE_INPUT=globals().get('CAPTURE_INPUT', None),
    RESUME_INPUT=globals().get('RESUME_INPUT', None))
url = ('https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/'
       + quote(SOURCE_REF, safe='/') + '/kaggle_native_qwen_verifier.py')
with urlopen(url, timeout=120) as response:
    exec(compile(response.read().decode('utf-8'),
                 'kaggle_native_qwen_verifier.py', 'exec'), launcher)
OUTPUT = launcher['OUTPUT']
```

`RUN_DIR` is the original full100 trajectory/state input.
`PHASE0_INPUT` is the original full100 pretrained Phase 0 latent128 result,
including its frozen source encoder, latents, provenance and exact question
split. Content resolvers from the selected checkout accept arbitrarily named
ZIPs, dataset mount folders, and extracted/wrapped roots. They do not require
particular upload filenames. Use an explicit path if multiple matching inputs
are mounted.

The launcher changes its working directory to `/kaggle/working` before work.
Original input extraction, Hugging Face/model downloads, temporary files and
caches belong under `/kaggle/temp`. It clears inherited `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` flags. An explicit capture or
resume path does not automatically enable offline model loading. PyTorch is
provided by Kaggle; the install request contains only:

```text
transformers==4.53.1 accelerate>=1,<2 huggingface_hub<1 safetensors numpy matplotlib
```

The launcher discovers and runs all `tests/test_native_qwen_*.py` before heavy
runner work, then checks that at least two GPUs are visible for full mode.
The GPU check loads no language model. It invokes only:

```text
python -u run_native_qwen_verifier.py --input ORIGINAL --phase0_input PHASE0 --output OUTPUT --config OUTPUT/launcher_config.json [--resume] [--capture_input CAPTURE_PATH]
```

## Full and smoke modes

The main study owns `configs/native_qwen_verifier.json` and
`run_native_qwen_verifier.py`. Both modes use all 100 questions and the existing
70/15/15 question split; smoke limits captured states within each question.
Neither mode creates a new state-level split. Source configuration is copied
without changing the checked-in JSON. The launcher enforces these budgets:

| Setting | Full | Smoke |
| --- | --- | --- |
| Questions / train-val-test | 100 / 70-15-15 | same |
| `capture_states_per_question` | all eligible states | 2 |
| `probe_updates`, `latent_updates` | 1000, 1000 | 8, 8 |
| `eval_every` | source configuration | 4 |
| `selected_layers`, `latent_dims` | 2 / 64, 128, 256 | same |
| `seeds` | 42, 43, 44 | 42 |
| `benchmark_repetitions`, `benchmark_warmup` | 100, 10 | 3, 1 |
| `bootstrap_samples` | 2000 | 2000 |
| `package_raw_hidden` | true | true |
| `max_reproduction_mismatch_rate` | 0.0 | 0.0 |
| `pipeline_check_only` | false | true |

Set `MODE='smoke'` before the cell for a pipeline check. Smoke's small training
and timing budgets do not establish representation quality, convergence,
statistical stability or a feasibility verdict. Full training uses seeds
42/43/44, selects layers on validation, then evaluates all three preregistered
dimensions on test using 2000 question-level paired bootstrap resamples.

Full mode runs 75 training jobs: 51 for the direct baseline and raw layer/loss
variants, 18 for compression, and 6 for the same-structural-input raw control.
Checkpoint choice uses validation only. `jobs/` owns checkpoint files; the
requested layer/dimension folders contain metrics and checkpoint indexes,
avoiding duplicated model files in the ZIP.

The checked source has **10,318 labeled states / 160,536 positions**. Four
3584D FP16 layers require **4.29 GiB** of raw hidden, plus experiment
checkpoints and frozen candidate embeddings. Qwen's full model weights require
additional temporary space outside Output. A **1–3 hour** full-run planning
range is tentative and must be replaced by measurements on the actual Kaggle
runtime; no fixed runtime or convergence promise is made.

Input preparation finishes in a temporary folder which is removed afterward;
it does not duplicate the old 1.4 GB trace in Output. A conservative disk check
reserves raw hidden, resumable Adam checkpoints and the final ZIP. Failed runs
package available native observations and diagnostics as partial results,
not as completed feasibility results.

## Capture replay and exact resume

`CAPTURE_INPUT=None` starts fresh capture. To replay an earlier native capture,
set `CAPTURE_INPUT` to a mounted native capture/result ZIP or unpacked folder
before running the same cell in a fresh runtime. The explicit path is passed
as `--capture_input`. The runner resolves the contents and validates the exact
capture identity, model, source, split and relevant configuration. Both original
inputs remain required for a matched direct baseline. A supplied capture is
not a reason to assume the model is locally cached.

`RESUME_INPUT=None` starts new training. To continue a previous native study,
set it to that study's result ZIP, unpacked root, or mount with one matching
result ZIP. The launcher restores permitted result files into a **new** Output
folder and passes `--resume`; it never writes into the old mount. The previous
result must include `config.json`, `study_manifest.json` with its exact study
signature, and `native_hidden/`. Preserve the raw capture and resumable
`last.pt` checkpoints. The launcher checks effective config equality and the
previous Git revision when recorded; the runner checks the current model,
source, capture and study signatures before reuse. Keep the original commit
SHA, mode, inputs and configuration. A result with only metrics is insufficient
for training resume. `CAPTURE_INPUT` and `RESUME_INPUT` can both be supplied
when the runner needs a separate explicit capture.

## Results, audit and packaging

The runner owns atomic success/failure packaging. The download ZIP sits at the
`/kaggle/working` root alongside the fresh result folder and contains root-level
result members. The launcher prefers a valid runner ZIP. If it needs a fallback,
it atomically merges available outputs with prior runner reports, strips only
the exact `OUTPUT.name/` prefix, and replaces current config/checkpoint copies
without duplicate members. Packaging failure preserves the previous ZIP.
Experiment failures are saved as `launcher_error.txt` and re-raised.

Fallback excludes original input trees, raw trajectories, Phase 0 source
assets, Hugging Face/model caches, full language-model weights and nested input
ZIPs. It **keeps `native_hidden/*.npy`, Qwen's frozen candidate embeddings, and
experiment checkpoints**, with no file-size cutoff. Raw native capture is a
deliberate result, not a model cache. Do not discard it when uploading a replay
or resume dataset.

The main runner's result contract includes `config.json`, `split_manifest.json`,
`capture_audit.json`, `verifier_reproduction_check.json`,
`alignment_unit_test.json`, raw `native_hidden/`, per-seed layer-probe metrics,
the matched direct baseline, verifier-latent checkpoints/compression comparisons,
measured partial-Qwen timing, question bootstrap results, plots and
`FINAL_REPORT.md`. Per-state/per-position predictions must preserve accepted
lengths and survival outputs, question/state IDs, seed, action, candidate
positions/tokens, parent K/length, full-prefix status and new-block masks.

Report question-macro K-MAE, survival Brier/AUC and K-bias for `ALL`, `R_ALL`,
`R_GAIN`, `R_LOSS`, `E_ALL`, `E_PARENT_FULL_PREFIX`, and `E_NEW_BLOCK`, always
with state, position and question counts. Compare H and HC probes using
identical architecture across layers; retain the BCE-only ablation. Use the
same captured states for the direct comparison and dimension sweep. Plots must
cover layer quality, compression loss, measured layer latency and quality versus
latency fraction. Partial-Qwen timing is a separate measured forward benchmark;
it establishes no end-to-end scheduler or speculative-decoding speedup.

Local launcher checks need no network, GPU or model loading:

```text
python tests/test_native_qwen_launcher.py
```

Run the complete study suite:

```text
python -m unittest discover -s tests -p "test_native_qwen_*.py"
```

Local verification passed 145 tests under both the installed Transformers
4.57.3 and an isolated pinned 4.53.1 runtime. The integration fixture uses a
real randomly initialized tiny Qwen on CPU and synthetic 100-question inputs;
it validates capture, training, locked test evaluation, packaging and resume.
The original real-source schema, checkpoint identity and 70/15/15 split were
also checked locally. No real 7B capture or two-T4 feasibility measurement has
been run locally; those results must come from the Kaggle experiment.
