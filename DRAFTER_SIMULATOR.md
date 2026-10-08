# Drafter-only latent simulator

This experiment collects fresh native Fast-dLLM refinement trajectories from 100 GSM8K questions, then trains small models to predict the next refinement state. It uses `Efficient-Large-Model/Fast_dLLM_v2_1.5B` alone: there is no target model or verifier. Native decoding is greedy (`temperature=0`), with the native confidence threshold of 0.5 and forced argmax progress.

The default 256 `max_new_tokens` is a **collection cap**, not a promise of 100 finished answers. Natural EOS stops generation according to the native sub-block behavior. A naturally completed active sub-block does not imply a completed answer; capped or EOS-interrupted groups can be incomplete. Report completion, truncation, valid transition counts and skipped horizons from the collector/runner manifests. Do not treat missing future states as successful predictions.

## One Kaggle cell: fresh Save & Run All

Enable **Internet**, select **GPU T4 x2** (recommended), and remove obsolete attached Kaggle datasets before saving. No input ZIP, previous trace archive, attached model dataset, or new Kaggle dataset is needed. A stale attachment can cause a mount failure before any notebook cell executes; remove it in the notebook's input panel.

Paste this entire cell into a fresh notebook. The selected source branch/SHA must exist on the public remote and contain the simulator runner, collector, models and launcher. Local uncommitted changes are not downloaded. This cell does not publish them.

```python
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

SOURCE_REF = "codex/drafter-latent-simulator"  # Use an exact published SHA for a repeat run.
MODE = "full"                              # "smoke" uses 3 prompts, with 1/1/1 splits.

# Optional budget overrides; uncomment only the settings you want to change.
# NUM_QUESTIONS = 100
# MAX_NEW_TOKENS = 256                      # Collection cap, not finished answers.
# ENCODER_UPDATES = 400
# UPDATES = 600
# SEEDS = [42, 43, 44]
# DATAPARAM = {"max_context_tokens": 4096}
# MODEL_REVISION = "<resolved model SHA from model_metadata.json>"

# Resolve before fetching the launcher, so its code and checkout use the same SHA.
requested_ref = SOURCE_REF
headers = {"User-Agent": "drafter-simulator-kaggle", "Accept": "application/vnd.github+json"}
commit_url = "https://api.github.com/repos/HoangMaizzz/failfasttesting/commits/" + quote(SOURCE_REF, safe="")
with urlopen(Request(commit_url, headers=headers), timeout=60) as response:
    source_sha = json.load(response)["sha"]
if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
    raise RuntimeError("GitHub did not return a source commit SHA")
launcher_url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{source_sha}/kaggle_drafter_simulator.py"
with urlopen(Request(launcher_url, headers=headers), timeout=60) as response:
    launcher_source = response.read().decode("utf-8")
scope = {"__name__": "__main__", "SOURCE_REF": source_sha, "MODE": MODE,
         "SOURCE_REQUESTED_REF": requested_ref}
for name in ("NUM_QUESTIONS", "MAX_NEW_TOKENS", "ENCODER_UPDATES", "UPDATES", "SEEDS", "DATAPARAM", "MODEL_REVISION"):
    if name in globals():
        scope[name] = globals()[name]
print("Requested source:", requested_ref, "resolved SHA:", source_sha)
exec(compile(launcher_source, launcher_url, "exec"), scope)
```

The launcher is import-safe. Importing `kaggle_drafter_simulator` starts nothing; call `launch(scope)` explicitly. Executing its source with a notebook configuration requires `scope["__name__"] = "__main__"`, as in the cell above. Bootstrap download failures raise directly. No setup/download failure is reported as a completed experiment.

Each invocation fetches the resolved branch/SHA into a new detached checkout under a unique `/kaggle/temp/drafter_simulator_*` directory. It keeps the notebook's current directory at `/kaggle/working`; it never deletes or reuses a prior checkout/output folder. A fresh Save & Run All session repeats setup and collection without relying on files from an interactive session.

Dependencies include `transformers==4.53.1`, `datasets`, `scipy`, `numpy`, `matplotlib`, `accelerate`, `einops`, `huggingface_hub<1`, `safetensors` and `sentencepiece`. Installation constrains the existing PyTorch version and checks that it remains unchanged. It does not install a replacement PyTorch/CUDA build. Child processes use `USE_TF=0`, `USE_FLAX=0` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Inherited `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` flags are cleared for those processes so a prior session's offline setting cannot disable fresh downloads. Effective package versions are saved in the output; the packages other than Transformers/Hub are resolved by pip, so their versions can change across fresh sessions. An exact source SHA and model SHA fix the code/weights, while `versions.json` records the environment needed for a stricter replay. GPU arithmetic need not be bit-for-bit identical across environments.

The launcher resolves the requested Hugging Face model revision (default `main`) through `HfApi.model_info(...).sha` before calling `snapshot_download` with that exact SHA. It records the requested/resolved revisions before downloading. The download lives under the unique temporary directory's `hf/` folder and allows JSON configuration/tokenizer files, safetensors, Python custom code, text/SentencePiece tokenizer assets and Jinja templates. It then overlays the checkout's `Fast_dLLM_v2_1_5B/modeling.py`, recording its SHA-256. No model weights are written to notebook Output. See the [Hub metadata API](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/package_reference/hf_api) for the revision metadata contract.

GPU 0 collects the 1.5B drafter traces. The runner waits for the collector process to exit and release its model before training the small simulator on GPU 1. After training, a simulator copy is benchmarked on GPU 0 for a same-device comparison with the logging-disabled native forward. The report records both dtypes and excludes observation preparation and the chosen real action from simulator core timing; it is not an end-to-end speedup claim. A single visible GPU is supported by selecting `cuda:0` for both sequential stages; T4 x2 is recommended. No simultaneous collector/trainer residency is assumed. CPU operation is intended for synthetic checks, not the default 1.5B collection run.

## Runner and configuration contract

The launcher writes an effective JSON config in its temporary directory and keeps a copy as `launcher_config.json` in the result folder. It invokes exactly:

```text
python -u run_drafter_simulator.py --config_json /kaggle/temp/<unique>/config.json --output_dir /kaggle/working/drafter_simulator_<unique> --dllm_dir /kaggle/temp/<unique>/hf/Fast_dLLM_v2_1_5B
```

The runner owns collection, splitting, training, evaluation, completion gates and its own final packaging. Its subprocess runs with `check=False`; the launcher records the return code, packages diagnostics and raises on a nonzero exit. Even with code 0, a readable `summary.json` with `status="complete"` is required; a missing, malformed or partial summary fails the launch.

| Config keys | Full defaults |
| --- | --- |
| `num_questions`, `max_new_tokens`, `max_context_tokens` | `100`, `256` collection cap, `4096` |
| `physical_block_size`, `small_block_size`, `threshold` | `32`, `8`, `0.5` |
| `collect_device`, `train_device` | `cuda:0`, `cuda:1` (both `cuda:0` with one GPU) |
| `seeds`, `latent_dim`, `horizon` | `[42,43,44]`, `128`, `3` |
| `encoder_updates`, `updates`, `eval_every` | `400`, `600`, `100` |
| `learning_milestones`, `batch_size` | `[20,40,70]`, `32` |
| `benchmark_repetitions`, `benchmark_warmup` | `100`, `20` |

The 100 questions split by question into 70 training, 15 validation and 15 test questions. Trajectories from one question stay together. Data order/split seeds stay fixed across model comparisons; `[42,43,44]` are training seeds. Learning milestones refer to training-question counts. The collector accepts `max_context_tokens` and `threshold` as aliases for its `max_context` and `drafter_threshold` fields. The effective config also records `source_revision` for the runner's provenance.

Notebook scope overrides are `NUM_QUESTIONS`, `MAX_NEW_TOKENS`, `ENCODER_UPDATES`, `UPDATES`, `SEEDS` and `SOURCE_REF`. `DATAPARAM` accepts the lowercase collection config keys in the first two rows of the table; explicit uppercase overrides take precedence. Keep native geometry at 32/8 and threshold at 0.5. Collection token caps must be positive multiples of 32. Changing the question count changes the question split and clamps/deduplicates training milestones to the available training-question count. `MODEL_REVISION` optionally pins a recorded model SHA. Paths `WORKING_DIR`/`TEMP_DIR` and `ALLOW_CPU` are local launcher-check hooks, not required Kaggle settings.

`MODE="smoke"` defaults to 3 prompts, one training seed, a 32-token cap, 2 encoder updates, 2 transition updates, evaluation every update, a `[1]` learning milestone, batch size 2, and 3 benchmark repetitions after 1 warmup. It permits the runner's 1/1/1 question split. Sparse traces can lack valid horizon-2 or horizon-3 transitions, especially in smoke mode. Such jobs must be marked skipped with their valid horizon counts; empty metrics are not evidence of efficacy. Explicit overrides still apply after the smoke defaults. Smoke mode checks the pipeline only.

## What the simulator observes and predicts

The physical canvas always contains all 32 native block positions. Only the active eligible 8-token sub-block can commit tokens during that native refinement step. Eligible generated positions, already committed positions, and positions outside the active span retain distinct roles; the simulator does not reduce the physical canvas to an independent 8-token forward.

A snapshot is taken **after the native commit**, with post-commit tokens/masks and a record of eligible positions. Its late hidden state and logits come from the **same forward before that commit**. Hidden provenance is the final normalized layer with the native alignment/right shift, rather than a recomputation on the newly committed tokens. This timing matters: the hidden/logit observation describes the computation that caused the commit. Only this late hidden observation is captured currently. Fusion with earlier layers is future work; no layer ablations are implemented or implied here.

Current-state encoding and step predictions cover per-token masks, confidence/entropy/margin, hidden-state observations `H`, semantic candidate embeddings, temporal candidate stability and latent state. Candidate embeddings use native token embeddings rather than arbitrary numeric token-ID distances. Ground-truth future hidden/behavior states may supervise losses and offline metrics but must not be fed into a teacher-free rollout.

The observation encoder is fit on training questions and frozen before transition training. PCA is a possible frozen-encoder baseline; the current learned path uses behavior pretraining and reconstruction. An encoder's usefulness must be judged by behavior reconstruction and future behavior prediction, not latent MSE alone. Do not assume an EMA encoder unless the runner actually implements and reports it.

The runner comparisons are `linear_h1`, `mlp_h1`, `transformer_h1`, `transformer_h3`, and `raw_direct`, a matched-input direct behavior predictor. Feature-group ablations are `no_hidden`, `no_content`, `no_confidence`, `no_temporal` and `no_structure`. Leave-one-group-out comparisons quantify dependence on those observations; they are not layer ablations or causal guarantees.

Evaluate horizons 1–3 on valid consecutive native refinement transitions within the same active group. Teacher-free rollouts feed predicted latent states back into the transition; an oracle-input baseline receives the true intermediate state and measures the easier conditional problem. Report these separately, alongside raw direct prediction and current-state reconstruction. If a baseline, ablation or horizon is unavailable in the selected source revision, report it as unavailable rather than infer its result.

Natural EOS and collection caps retain incomplete groups for diagnostics; they do not manufacture future targets. Native commits are irreversible and forced progress is part of the native rule. Reporting projections that enforce these constraints must be distinguished from raw prediction quality.

## Outputs, failures and claim gates

Results go to a unique `/kaggle/working/drafter_simulator_*` folder. Launcher provenance is temporarily staged outside that folder while the runner starts, so its fresh-output check receives an empty directory. The runner packages its own outputs in `finally`; the launcher restores provenance and refreshes a ZIP beside the result folder, including late launcher diagnostics. The fallback ZIP uses sorted members and fixed ZIP timestamps/permissions, making unchanged output content package deterministically. It excludes model-weight/cache folders and safetensors while retaining small simulator checkpoints. The displayed download uses `FileLink(archive.name)` from `/kaggle/working`, avoiding an absolute-path notebook link that returns 404. Saved notebook versions also expose the root ZIP through Output.

Launcher provenance includes `launcher_metadata.json` (source repository/ref/SHA, stage, status, command and return code), `launcher_config.json`, `dependency_request.json`, `versions.json`, and `model_metadata.json` (resolved model SHA and overlay hash). Failures add `launcher_error.txt`. Files appear only after their stages are reached; a setup failure does not invent downstream manifests. The temporary source, weights, caches and config are not archived; the output retains the effective config and revision records.

A successful launcher return requires code 0 and the runner's `summary.json` reporting `status="complete"`. The launcher then records `status="complete"` and prints `Output ZIP ready:` with the archive path. These are pipeline completion checks, not efficacy checks. Inspect native reproduction/alignment audits, valid data counts, split integrity, per-seed metrics and skipped jobs before drawing conclusions. A ZIP-ready message also appears on failures to make partial diagnostics downloadable; consult the status inside it. Downloading a ZIP or passing a smoke run does not establish success on quality gates.

This study predicts drafter refinement behavior. It does **not** establish GSM8K answer accuracy, target-verifier acceptance, speculative-decoding correctness, or end-to-end generation speedup. Small-model timing is an isolated benchmark; collection is instrumented and includes observation overhead. Any measured simulator latency must be reported with devices, repetitions, warmups and synchronization. Avoid end-to-end claims without a separate integrated generation experiment.

## Local checks

From the repository root, run the focused suite:

```text
python -m unittest discover -s tests -p test_drafter_simulator_*.py
```

`tests/test_drafter_simulator_launcher.py` uses stdlib mocks and fixture files; it never downloads weights, calls a GPU or trains models. Discovery also includes the matching collector/model/runner tests supplied by their owners, while excluding older unrelated suites.

The training test creates a synthetic CPU study, optimizes the encoder and transition, evaluates free-running horizons 1–3, and packages the resulting reports. Synthetic fixtures validate plumbing and invariants only; they provide no evidence about native model efficacy. The collection tests additionally execute the real native generator with a tiny randomly initialized model on CUDA when pinned Transformers and CUDA are available.
