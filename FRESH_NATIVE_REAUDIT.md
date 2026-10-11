# Corrected native-drafter audit

`fresh_native_reaudit.py` trains from either an existing fresh-native capture ZIP
or its unpacked directory. It does not load or run Fast-dLLM or a verifier.

```console
python fresh_native_reaudit.py --source CAPTURE.zip --output NEW_OUTPUT --updates 240 --seeds 42 43 --device cuda
```

Use a new output directory. Dependencies are those of the fresh-native test.

## Corrections

- Capture writes candidate-change labels after constructing the incoming R edge,
  including when a shard flushes immediately.
- Loading old captures rebuilds R change labels from raw candidate token IDs.
  The input archive is not edited; repaired labels are saved separately.
- Rich/direct and latent/direct predictors receive the R/E action, just as
  latent dynamics does.
- Content supervision uses 0.3 times cosine distance rather than underweighted
  regression on small-magnitude native embeddings. This is a direction objective,
  not exact-token prediction, and does not guarantee embedding norm fidelity.
- Validation checkpoint selection includes content and candidate-change metrics.
  Dynamics is trained through H3 but checkpoint selection still uses H1.

## Outputs and interpretation

The runner saves checkpoints, train-only normalization, question split, logs,
per-position predictions, H1-H3 metrics, matched three-step path results by full
action sequence, and test-time feature sensitivity. The result ZIP excludes the
temporary extracted copy of the input capture.

Baselines fit only training questions: position/refine-index mask probability and
confidence median, mean target embedding, unchanged previous candidate, and
candidate-change prior. Compare H1 directly with these baselines. Later-horizon
prediction rows use the true edge's source for the diagnostic baseline fields;
that is not a deployable open-loop baseline at H2/H3.

Feature sensitivity replaces one normalized group with its training mean at
inference, without retraining. It is not a causal importance or feature-removal
experiment. H1-H3 matched results use the same set of three-step paths, but those
paths overlap within questions. Split and uncertainty units must be questions,
not positions or paths.

This small audit uses two seeds and 15 held-out questions. It tests learnable
drafter observables, not verifier acceptance, controller quality, or speedup.
Direct models train H1; dynamics trains H1-H3. Equal update count is not equal
training compute.

## Local run, 2026-10-11

On the existing 100-question GSM8K capture, 1,036 positive change labels were
recovered from 3,102 eligible R positions. Six models (three arms, two seeds)
trained/evaluated in approximately 87 seconds on an RTX 3060 Laptop GPU,
excluding environment preparation. Content cosine rose substantially after the
loss correction, but did not beat simple mean/copy baselines convincingly.
Next-mask and confidence carry some learnable signal; robust semantic latent
drafter simulation is not established by this run.
