# Latent drafter sufficiency audit

This is a targeted offline experiment using the already collected observation ZIPs. It does not load Fast-dLLM, the verifier, or any model weights, and it does not rerun generation.

## What it distinguishes

On the 100-question Refine capture, questions are split once into 70/15/15 train, validation, and test sets. Each seed uses that same split and compares:

1. `rich_direct_h1`: current full captured observation to the next observed behavior, with a wider 256D contextual predictor.
2. `latent_direct_h1`: a trainable 128D per-position latent, decoded directly into next-step behavior.
3. `latent_dynamics_h1_h3`: the same 128D observation encoder plus an action-conditioned transition; the predicted latent is fed back to itself for free-running H1, H2, and H3.

The Refine targets are next mask status, confidence, candidate-change event, and the native candidate embedding. The scored positions are masks eligible at the source state. Metrics include mask Brier/AUROC/average precision, confidence MAE, candidate-change Brier/AUROC/AP, and candidate-embedding cosine. Persistence and the native confidence-threshold rule are included as controls. The Extend archive has no semantic candidate embedding, so its content target is the recorded top-k probability vector; that is not a token-generation target.

Interpretation:

- Rich direct beats latent direct: the smaller representation or its capacity may be discarding useful signal. This is diagnostic, not a mathematically pure compression ablation because the rich reference is wider.
- Latent direct beats latent dynamics at H1: the learned transition is the main gap after encoding the current observation.
- H1 works but H2/H3 degrade: free-running state error accumulates; more one-step training alone is unlikely to solve that.
- All arms are weak: inspect observation/target alignment and whether the captured features predict the next native event before increasing rollout horizon.

The Extend ZIP is evaluated separately with five-fold leave-one-problem-out H1. It has only five problems and uses its recorded legacy protocol; it is an exploratory check, not a generalization result. Refine and Extend metrics are never pooled.

## Kaggle inputs and limits

Create or use one Kaggle dataset input containing:

- `drafter_simulator_qyx296iq.zip` (100-question Refine capture)
- `structured_sparse_20260925_015533_gsm8k.zip` (small Extend graph)

Select a GPU accelerator. One GPU is enough. The run needs no 1.5B model, verifier, Hugging Face download, or GitHub secret; it downloads only this small runner. The output ZIP is written to `/kaggle/working`.

Expected runtime: roughly 10–30 minutes on a Kaggle T4, depending on GPU load. This excludes data upload and mount time.
