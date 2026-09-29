# Learning curve using an existing 100-question archive

This experiment answers two separate questions: whether more examples help and
whether the world model improves as optimizer training continues. It uses the
archived train/validation split and raw cached states/labels. It does not invoke
the 1.5B drafter or 7B verifier. It trains fresh small world-model weights; the
saved checkpoint supplies architecture/configuration, not pretrained weights.

Four nested train-set sizes (10, 20, 40, 80 questions) and three seeds are run
under two schedules:

* `fixed_updates`: reset to the same random initialization for each data size and
  run 512 optimizer updates. This is the main apples-to-apples data-volume test.
* `proportional`: add questions incrementally, run 8 optimizer updates per added
  question, and retain the learner between sizes. This represents scaling data
  and training work together.

A third `training_progress` schedule fixes the data at the largest selected
training subset and evaluates fresh models at updates `0, 16, 64, 128, 256, 512`.
That curve isolates training time from data volume. Every checkpoint uses the
same held-out questions and the same deterministic state/root sample. Read
validation MAE (lower is better), not training loss alone, to decide whether it
actually generalizes. R/E one-step and horizon-3 MAE reveal if state dynamics
improve alongside current-state acceptance prediction.

Unlike the original online replay buffer, the offline experiment retains all
selected training observations at every size. Otherwise a 4096-state cap could
silently discard data and hide the learning-curve effect we want to measure.

Both schedules score a deterministic stratified sample of up to 24 current states
and 4 real R/E paths per held-out question, identical at every checkpoint. It
reports current acceptance MAE, R/E one-step MAE, horizon-3 MAE, the R/E changed
label subset, and a model-prediction persistence baseline. The original ZIP has
20 held-out questions; three seeds quantify initialization and sampling
variation, but this is still a preliminary curve, not a scaling law.

The source archive lists 80 train and 20 validation questions. The Hugging Face
embedding is pinned to the revision and tensor hash verified in the previous
audit. The check stops the run if the embedding differs. If the cache from that
audit is present at `/kaggle/temp/wm_audit_hf`, it is reused; otherwise only the
embedding weight file is downloaded, and no LLM forward is run.

Default work is about 9,600 small-model optimizer updates across all schedules
and three seeds. Lower it with `UPDATES_PER_QUESTION`, `FIXED_UPDATES`, and
`PROGRESS_UPDATES` for a quick feasibility pass; report those settings so
results are not mistaken for the default curve. One GPU is sufficient.

## Kaggle cell

Add the training ZIP (or extracted training run) as a Kaggle Dataset. Set the
path below to the dataset mount root. Do not provide the compact feature-audit
ZIP; it does not include training states.

```python
from urllib.request import urlopen

REF = "codex/sparse-extend-world-model"  # replace with the commit printed by Codex
config = {
    "SOURCE_REF": REF,
    "INPUT_PATH": "/kaggle/input/datasets/yumesakihikari/math100",
    "QUESTION_SIZES": "10,20,40,80",
    "SEEDS": "42,43,44",
    "UPDATES_PER_QUESTION": 8,
    "FIXED_UPDATES": 512,
    "PROGRESS_UPDATES": "0,16,64,128,256,512",
    "VALIDATION_STATES_PER_QUESTION": 24,
    "VALIDATION_ROOTS_PER_QUESTION": 4,
}
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{REF}/kaggle_world_model_learning_curve.py"
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_world_model_learning_curve.py", "exec"), config)
```

The report ZIP is written to `/kaggle/working/world_model_learning_curve_<timestamp>.zip`.
It contains every seed/size metric and selected-question prediction rows.
