# Offline latent world model: Phase 0

Phase 0 asks whether a compact native drafter state can preserve verifier
information and predict the result of saved R/E actions. It uses the original
100-question GSM8K experience and teacher trace, with **zero LLM forwards**.
It does not generate tokens, decode a full vocabulary, or decode random token
identity codes. The pretrained input embedding table is an observation feature;
there is no language model or token generator attached to the latent dynamics.

## One Kaggle cell

Select **GPU T4 x2** and enable Internet for the source, four Python dependencies,
and the pinned embedding checkpoint if it is not supplied locally. PyTorch is
already installed. The launcher installs `numpy`, `scikit-learn`,
`huggingface_hub`, and `safetensors`; it does not install Transformers.

The following cell loads the launcher from the same branch or SHA that it will
fetch for the experiment. That ref must already exist on the remote repository.

```python
import os
from urllib.parse import quote
from urllib.request import urlopen

os.chdir('/kaggle/working')
RUN_DIR = globals().get('RUN_DIR', '/kaggle/input/datasets/ainzkhail/2source2')
SOURCE_REF = globals().get('SOURCE_REF', 'codex/latent-wm-phase0')  # or published SHA
MODE = globals().get('MODE', 'full')  # 'smoke' checks the pipeline
# TOKEN_EMBEDDING = 'learned'  # explicit, weaker semantic ablation; no download
# EMBEDDING_PATH = '/kaggle/input/my-embedding/model.safetensors'  # optional
# RESUME_INPUT = '/kaggle/input/my-results/previous_result.zip'  # optional
# CONTINUE_DIAGNOSTICS = True  # explicit continuation after failed validation gates

url = ('https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/'
       + quote(SOURCE_REF, safe='/') + '/kaggle_latent_wm_phase0.py')
with urlopen(url, timeout=120) as response:
    source = response.read().decode('utf-8')
launcher = {'__name__': '_phase0_notebook_launcher'}
exec(compile(source, 'kaggle_latent_wm_phase0.py', 'exec'), launcher)
OUTPUT = launcher['launch'](globals())
```

If the launcher file is already available locally, import it and call
`launch(globals())`, or execute that file as a notebook cell. Importing it alone
does not download anything, change the current directory, or start training.

`RUN_DIR` accepts an original extracted trace folder, any ZIP regardless of its
name or suffix, or a mount containing one original trace. Discovery uses
`factorized_wm_data.resolve_input` and the original summary schema and members,
including nested ZIP prefixes. The launcher also discovers ZIP content in files
without a `.zip` suffix. Original inputs need no model checkpoint. A report ZIP
is not an original trace. If there are multiple original runs, point `RUN_DIR`
at the specific folder or ZIP. Missing mounts print available dataset roots
before any downloads. Original ZIPs are read in place.

The launcher changes directory to `/kaggle/working` before launching and never
deletes an existing working directory. It fetches
`https://github.com/HoangMaizzz/failfasttesting.git` into a unique directory under
`/kaggle/temp`, with a detached checkout of `SOURCE_REF`. Both branch names and
commit SHAs work; `launcher_metadata.json` records the resolved commit. Pin a SHA
to reproduce a run after the branch changes.

## Effective configuration

`configs/latent_wm_phase0.json` controls the full training configuration. The
launcher writes the effective configuration to `<OUTPUT>/launcher_config.json`
and invokes only the Phase 0 runner:

```text
python -u run_latent_wm_phase0.py --input ORIGINAL_TRACE --output OUTPUT --config EFFECTIVE_JSON [--resume]
```

| Setting | Full | Smoke |
| --- | --- | --- |
| Questions | 100 | 20 |
| Question train/validation/test split | 70/15/15 | 14/3/3 |
| `latent_dims` | `[64, 128]` | `[64]` |
| `stage_a_updates` | 1600 | 10 |
| `raw_updates` | 1200 | 20 |
| `dynamics_updates_per_horizon` | 800 | 20 |
| `direct_updates` | 1000 | 20 |
| `eval_every`, `minimum_updates` | 100, 400 | 5, 5 |
| `patience`, `bootstrap_samples` | 5, 500 | 2, 50 |
| `batch_size` | 16 | 16 |
| Encoder, verifier, reconstruction | `device_encoder='cuda:0'` | same |
| Independent small R/E dynamics | `device_dynamics='cuda:1'` | same |

The full budgets above describe the checked-in JSON; subsequent changes to that
JSON remain authoritative. Full mode requires 100 questions, latent dimensions
64/128, and the question split 70/15/15. Smoke caps update budgets and explicitly
reduces the dataset and dimension grid. It retains the pretrained embedding
mode and the gate logic, so smoke can still require the large checkpoint and can
legitimately stop at a failed gate. Smoke is a pipeline check, not a full study.

Supported notebook overrides are `RUN_DIR`, `SOURCE_REF`, `MODE`,
`TOKEN_EMBEDDING`, `EMBEDDING_PATH`, `RESUME_INPUT`, and `CONTINUE_DIAGNOSTICS`.
The embedding and diagnostic overrides appear in the effective JSON. Kaggle
requires two visible GPUs; there is no automatic single-GPU or CPU fallback.
For an explicit local check only, `ALLOW_CPU=True` permits CPU and
`WORKING_DIR`/`TEMP_DIR` select local folders.

## Local timing and runtime estimates

A local GPU run on **20 real-trace questions completed in 87.15 seconds**.
That timing used explicit learned input embeddings, latent width 64,
`raw_updates=100`, `stage_a_updates=100`,
`dynamics_updates_per_horizon=40`, `direct_updates=40`, and
`continue_diagnostics=true`. This was a separate timing configuration, not the
launcher's default smoke budgets. Learned embeddings avoided the pretrained
checkpoint download. This low-budget run checks execution and produces no
feasibility verdict; diagnostic continuation does not convert failed gates
into passing evidence.

The full default budget totals up to **11,200 optimizer steps**, with both
latent widths, 100-question validation and reports, and native pretrained
embedding download/preparation. The **planning estimate for Kaggle T4 x2 is
30–60 minutes** when all stages run, or **10–25 minutes with an earlier gate
stop**. These are estimates, not measured Kaggle runtimes. Gate behavior,
checkpoint/download speed, and validation/report costs affect elapsed time;
the 87.15-second local run does not directly establish full-run duration.

## Embeddings and temporary storage

The default is `token_embedding='pretrained'`, with:

```text
embedding_repo_id: Efficient-Large-Model/Fast_dLLM_v2_1.5B
embedding_revision: cd3af22d326325d015267a7845b5cd5e91a28fa7
embedding_dim: 64
```

The repository ID is the one used by `prepare_fixed05_assets.py` and the original
Kaggle launchers. The runner owns embedding preparation. It uses the native
`Fast_dLLM_v2_1_5B` checkpoint locally when available, an explicit
`EMBEDDING_PATH`, or the pinned Hugging Face checkpoint. A supplied path may be a
safetensors file or its containing model directory. Pretrained mode must fail
clearly if the file or required embedding tensor is unavailable; it must never
silently switch to learned embeddings.

The checkpoint `model.safetensors` is about 3.09 GB. The runner reads only
`model.embed_tokens.weight` through safetensors memory mapping; it does not
construct the LLM or execute its forward pass. A 64-dimensional PCA projection
and normalization are fit using token IDs from train questions only, then
frozen. Held-out token IDs can be looked up in the pretrained table without
fitting the projection on held-out data. `TOKEN_EMBEDDING='learned'` explicitly
selects a train-vocabulary input embedding with an UNK input for unseen IDs;
it avoids the checkpoint download but weakens the semantic representation.
Neither mode introduces a token decoder.

Large checkpoints, embedding assets, and caches belong in `/kaggle/temp`.
The launcher places source and Hugging Face, Torch, and temporary-file caches
there. A model already mounted under `/kaggle/input` is read-only input, not a
new output copy. Results in `/kaggle/working` should contain compact latent
checkpoints, predictions, configuration, provenance, and reports. Launcher
metadata records the requested embedding mode/ref/path; the runner's provenance
report records what it actually loaded and the frozen projection. Read that
report before interpreting results.

## Stages and evidence

Stage A jointly fits the native observation encoder, verifier `G`, and native
reconstruction head on train questions. It compares the latent verifier to a
raw-input verifier on validation. The encoder, `G`, and reconstruction head
are frozen before fitting R/E dynamics. Checkpoint and gate choices use
validation; test questions are reserved for the final challenge reports.

R and E are independent dynamics modules. Their losses supervise the frozen
real-child latent plus native drafter auxiliary/reconstruction targets. Verifier
truth is not a dynamics input or a verifier-loss shortcut into the dynamics.
The frozen `G` evaluates the imagined result. A direct source/actions-to-future
verifier baseline is a separate diagnostic, not the learned latent transition.

The validation gate after Stage A checks whether the bottleneck preserves
enough verifier information relative to the raw baseline. H1 then has its own
validation gate before H2/H3. Default `continue_diagnostics=false` skips later
stages when a required gate fails. `CONTINUE_DIAGNOSTICS=True` permits explicitly
flagged diagnostic continuation; a failed gate remains failed and must not be
presented as positive feasibility evidence. Gate criteria and thresholds are
controlled by the JSON and reported by the runner.

Reports must compare real truth with four aligned predictions: `G` on the real
future latent (oracle), `G` on learned imagined futures, `G` on copy/prior
futures, and direct prediction from the source and actions. Separate the real
truth error from the imagined-versus-oracle gap. Report accepted length `K`
(excluding correction/bonus), token acceptance/hazards, teacher-forced
compatibility and calibration where labels exist, action R/E and sequence,
horizon, and proposal region. Use paired question-level comparisons and
bootstrap intervals. A good aggregate score dominated by an immutable prefix
does not establish good frontier or newly extended-block dynamics.

The encoder consumes cached raw native hidden observations with validity and
age information. Hidden PCA is a fixed **reconstruction target** fit on train,
not a substitution for the raw encoder input. Reconstruction masks exclude
unavailable native fields. Whole-question grouping keeps every state, edge,
and action path from a question in one split. Child latent, hidden, masks,
context, token IDs, and teacher values are targets or evaluation truth, never
inputs to imagined free rollout. Initial parent information and known actions
are the permitted inputs; future structure and masks must be advanced from
predictions and known R/E rules. Missing labels must not become negatives.

These are exploratory results on reused questions. A failed gate diagnoses the
current representation/training setup; it is not an impossibility proof or a
controller speedup claim. Saved native top-k support does not provide a
full-vocabulary generation or distribution claim.

## Results, errors, and resume

Before running the experiment, the launcher checks the small-model suites:

```text
tests/test_phase0_wm_models.py
tests/test_phase0_wm_metrics.py
tests/test_phase0_kaggle_launcher.py
tests/test_phase0_wm_runner.py
```

The launcher tests are offline and mocked: content discovery, source-ref
handling, GPU requirements, effective smoke config, resume extraction, CLI
wiring, partial ZIPs, and basename links. They do not train models or download
embeddings. The selected source ref must include all four test files.

Each run creates a unique result folder directly beneath `/kaggle/working`
with an Asia/Bangkok timestamp (`ICT`, or `UTC` if timezone data is unavailable).
The ZIP is a sibling at the working root, `OUTPUT.with_suffix('.zip')`.
`FileLink` uses just its basename, so notebook download links resolve properly.

The runner is responsible for checkpointing and packaging after every completed
stage and on errors. The launcher prefers that runner ZIP. If it is missing,
the launcher packages available compact outputs, including errors from
preflight, dependency installation, and tests. Its fallback excludes raw
experience, NPZs, caches, embedding assets, and files larger than 128 MiB, and
records exclusions in `launcher_packaging.json`. If a runner stage ZIP already
exists when the launcher fails, it retains those results and adds the launcher
error. A downloadable ZIP alone does not mean the study completed: check the
runner summary, completed-stage manifest, and gate reports.

Start with `summary.json`, `embedding_provenance.json`,
`latent_<dim>/stage_A_gate.json`, and `latent_<dim>/dynamics/H1/complete.json`.
`raw_reference_test.json` and `latent_<dim>/real_latent_test.json` distinguish
the raw-observation reference from the frozen latent verifier.
`latent_<dim>/test/phase0_comparison.json`, `challenge_cohorts.json`, and
`composition_predictions.jsonl` contain the aligned composition diagnostics
when gates permit those stages. Checkpoints use stage folders such as
`raw_reference/best.pt` and `latent_<dim>/stage_A/best.pt`.

Set `RESUME_INPUT` to one prior Phase 0 result ZIP or extracted result folder.
Keep `RUN_DIR` pointing at the **original trace**, with the same source ref,
mode, and embedding/diagnostic settings as the prior run. Resume does not
replace original experience with result artifacts. The launcher restores into
a fresh working result folder, leaving uploaded inputs intact, and passes
`--resume` to the runner. It requires `config.json`, the Phase 0 runner's
`study_manifest.json`, and nonempty compact model checkpoints; the old factorized study's `jobs`
schema is not sufficient. The runner performs the authoritative content/code/
configuration and checkpoint checks and determines which completed stages can
be reused. Large embedding/cache assets are prepared again under temporary
storage as needed; they are intentionally absent from the result ZIP.

Resume extraction validates every member before writing, rejecting traversal,
absolute/drive paths, links, duplicate normalized paths, and encrypted members.
Wrapped result folders are supported, but multiple valid result roots are
ambiguous. Config mismatches fail with a partial-result ZIP instead of silently
changing the experiment.
