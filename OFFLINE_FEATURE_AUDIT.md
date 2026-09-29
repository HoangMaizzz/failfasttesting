# Frozen world-model audit (no LLM rerun)

Input: the **training run** ZIP containing `checkpoint.pt`, `states.jsonl`,
`labels.jsonl`, `edges.jsonl`, and `experience/*.npz`. The older `math_raw` /
`gsm8k_raw` archives do not contain the trained model and cannot replace it.
The MATH 100-question run has 80 training and 20 held-out questions. Evaluating
training questions does not turn them into independent test questions.

## Computation

* No retraining, no drafter forward, no verifier forward. Reuse exact saved labels.
* Reconstruct the frozen small world model and input features from the archive.
* Read only the drafter input embedding tensor, not the full model into GPU RAM.
  An existing `model.safetensors` can be reused. Otherwise Hugging Face downloads
  the weight file containing that tensor (~3.1 GB for the current unsharded model).
  Downloads/cache stay in `/kaggle/temp`, not the output ZIP.
* Stream raw features by question and pack embeddings once per minibatch, reuse
  them across perturbations. Baseline on all questions; perturbations on validation
  only by default. Only the initial observation is encoded for each rollout;
  horizons 1..3 use predicted latents, never true child features as model inputs.
* True child encodings/masks are used solely as evaluation targets. `R=refine`,
  `E=extend including its first native unmask`. Report action paths as well as the
  last-action groups `h1_R`, `h1_E`, `h2_R`, etc.
* Compare unperturbed predictions with saved predictions. If maximum discrepancy
  exceeds 0.05 tokens, exit with error after saving the report. Investigate source,
  embedding identity and numerical differences before interpreting importance.

## Feature groups

`hidden_all`, `hidden_layer_0/1/2` (actual layer IDs in summary), `token_native`,
`token_stop`, `topk` (gaps + candidate weighted embedding), `prefix_content`,
`history`, `confidence`, `position_age`, `mask_encoder`, `encoder_context`.

Perturbation zeros each group's input. For mask/context, known metadata needed
by dynamics/frontier/carry is retained; only the encoder signal is removed.
Lengths, validity and legal trajectories are always preserved. Prefix perturbation
removes content, not prefix length or positional encoding. These are **sensitivity
tests**, not retrained ablations. Redundant signals can conceal a group's value;
zeroing can also create out-of-distribution inputs. Do not sum contributions or
claim a group is inherently useless. Fixed threshold/block-size settings cannot
be causally compared with this dataset alone.

## Outputs

`summary.json`: state-weighted MAE, RMSE, median, P90/P95, bias, within 1/2/4,
large over/underestimates, NLL, rollout persistence baseline, active-mask Brier,
latent cosine/MSE; paired delta MAE/P90/within2 against full inputs. Question-macro
delta MAE has a paired question-cluster bootstrap 95% interval (1000 draws).
Rows within the same question are **not** assumed independent.

`predictions.jsonl`: per-state and per-path predictions for every evaluated arm.
`report.md`: readable metrics table. ZIP is placed directly in `/kaggle/working`.
Latent similarity is diagnostic, not a claim of perfect semantic state recovery.

Use the saved validation baseline and held-out results for model conclusions.
All archived states are evaluated, so counts can exceed the original replay cap
(4114 vs 4096 validation states in the MATH run). Baseline parity checks their
overlapping observations. Longer-horizon cohorts are different subsets.

## Kaggle cell

Upload the training ZIP as a Kaggle Dataset and Add Input. Kaggle may extract it;
both a ZIP path and its extracted run directory work. One T4 is enough; T4 x2
also works but this job intentionally uses only one GPU. Enable Internet if source
or the embedding file must be downloaded. The launcher loads a trusted PyTorch
checkpoint: use only your own training archive.

```python
from urllib.request import urlopen
REF = "codex/sparse-extend-world-model"  # pin to the provided audit commit
config = {
    "SOURCE_REF": REF,
    "INPUT_PATH": "",  # auto-detect ONE MATH 100q ZIP / extracted checkpoint
    "BATCH_SIZE": 4,
    "ABLATE_TRAIN": False,  # saves work; full baseline still covers 100 questions
    # "EMBEDDING_FILE": "/kaggle/temp/wm_fast_dllm_1_5b/model.safetensors",
}
url = f"https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/{REF}/kaggle_offline_feature_audit.py"
exec(compile(urlopen(url, timeout=60).read().decode(), "kaggle_offline_feature_audit.py", "exec"), config)
```

This evaluator has CPU synthetic end-to-end tests (including ZIP packaging,
input immutability, teacher leakage guards and paired metrics) and has been
checked against the real MATH archive's schema/edge lengths/checkpoint. A full
real-feature GPU evaluation must still be run; synthetic tests do not establish
feature importance, T4 runtime or numerical parity on that GPU.

## Follow-up suite: top-K separation and compact-latent candidates

Set `SUITE="followup"` in the same launcher. Default `VALIDATION_ONLY=True`
avoids recomputing 80 in-sample questions. Keep the original training ZIP/folder,
not the small audit output ZIP. For the current dataset use:
`INPUT_PATH="/kaggle/input/datasets/yumesakihikari/math100"`.

Eight arms (one reference and seven perturbations), no retraining:

| Arm | Modification |
|---|---|
| full | Original checkpoint and inputs |
| topk_gaps_channel | Zero explicit gaps; retain weighted candidate embedding |
| topk_candidate_channel | Zero weighted candidate embedding; retain explicit gaps |
| topk_no_probabilities | Zero gaps AND replace weighted summary with uniform mean of original candidates |
| topk | Zero both top-K channels, as in first audit |
| hidden_last_only | Keep last stored hidden layer (28 in this run), zero layers 7 and 14 |
| prefix_history | Zero separate prefix content and history together |
| compact_last | Last hidden layer only plus zero prefix/history |

Uniform candidate averaging still reveals which tokens belonged to top-K;
confidence and other features remain available. This is not a claim that ALL
drafter probability information has been removed. Zeroing hidden input layers
does not physically shrink or retrain the checkpoint. Compact arms screen
candidates for a later smaller-model training experiment, not certify their
retrained performance or speedup.

Additional metrics (available in both suites):

* `h1_R_gain/loss/same/changed` and equivalent E groups, based on exact labels.
* `h1_R_source8/source_over8` separates initial proposal from extended ones.
* `delta_K_mae` evaluates the change in predicted acceptance versus true change.
* `dynamics_minus_persistence_macro_ci95` compares learned transition with keeping
  the MODEL's source prediction (never use true source K as a deployment baseline).
* `h1_common3/h2_common3/h3_common3` score the same sources whose recorded paths
  reach horizon 3. Actual recorded actions are followed, not optimized by labels.
* `current_L8/L16/...` provides error by proposal length.

All group labels are attached AFTER predictions, not fed as features or used to
choose actions. Small gain/loss subsets should be interpreted using question
counts and uncertainty intervals, not just the mean MAE. Confidence intervals are
exploratory paired question-bootstrap intervals, without multiple-testing adjustment.

ZIP output: `/kaggle/working/offline_feature_followup_<timestamp>.zip`.
No large model weights or original raw shards are packaged.
