# FP16 no-KV single-pool positive control

One 180-problem MATH discovery stream, followed by CPU selection/reordering of
50 observed problem traces and up to five fresh deterministic U1 executions.
This is a selected positive control, NOT an exhaustive or globally optimal oracle.

- FP16 target and drafter, no quantization. The Kaggle launcher shards Qwen2's
  verifier across both T4 GPUs and places the drafter on GPU 1.
- No verifier KV cache and no reusable drafter KV. Both target devices are
  synchronized by the existing two-GPU timing path.
- Fixed threshold .5, drafter/outer thresholds .5/.7, raw-absolute utility
  weighting, uniform replay B16/K100, one SGD update per non-tie label.
- A deterministic tape replaces only legally available random probes. Learned
  actions and cold-start are never overridden. The tape includes warm-up ID 0.
- Replay matches every recorded decision's action and candidate/context signature,
  not only F2 at selected probes. Missing or unused scheduled probes invalidate
  the witness. Actual state traces are saved in probe_tape_trace.jsonl.
- PASS requires actual learned-C benefit >= 1, at least five non-tie learned-C
  observations, a score above .5 and zero schedule discrepancies. Resolved ties
  are included in the utility sum but are not used for training/class counts.
- PASS does NOT require or measure a win over Always-STOP; output equality is
  not a gate. Local utility sums are not additive milliseconds saved.
- Probe z scores are descriptive, soft search preferences, not significance
  guarantees for adaptively visited states.
- Resume reuses complete streams only. Incomplete streams are preserved and
  restarted from zero; configuration/code changes require a new output folder.
- Files are written locally; no W&B. The launcher packages discovery AND witness
  reports in finally. Forced kernel/VM termination may prevent ZIP creation.

CPU replay generates candidates only: actual execution is authoritative. An empty
search result is a valid negative outcome. This implementation cannot promise a
beneficial witness, speedup, bitwise determinism, or absence of GPU OOM.
