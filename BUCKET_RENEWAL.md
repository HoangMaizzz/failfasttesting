# Bucket-Renewal Adaptive Unmasking

The repository keeps two runtime policies:

- `disabled`: the original FailFast drafting policy.
- `bucket_renewal`: adaptive denoising inside each Fast-dLLM block.

Proposal extension remains unchanged. FailFast extends only when every committed
draft confidence passes `--sweep_lowconf_threshold`; the bucket-renewal policy
does not control proposal length.

For a token committed at refinement step `s` and proposal position `i`, the
controller estimates verifier acceptance with the backoff hierarchy:

1. step, position, confidence, and margin bucket;
2. step, confidence, and margin bucket;
3. step, position, and confidence bucket;
4. step and confidence bucket;
5. position, confidence, and margin bucket;
6. confidence and margin bucket;
7. position and confidence bucket;
8. confidence bucket;
9. raw drafter confidence.

For a selected bucket with `A_b` accepted observations and `N_b` total
observations, the calibrated probability is

```text
p_i = clip((A_b + prior_strength * c_i) / (N_b + prior_strength), 0.02, 0.98)
```

where `c_i` is the raw drafter confidence and `prior_strength=8` by default. A
bucket is authoritative after eight observations. Before that, the most-observed
available bucket supplies a smoothed fallback; raw confidence is used when no
history exists.

The probability is computed once when a token is committed and stored with its
confidence, margin, position, and commit step. Later denoising steps reuse that
stored probability exactly. They only estimate new probabilities for positions
that remain masked, so previously committed token probabilities cannot drift.
Verifier outcomes update the step-conditioned calibration tables after the
round and affect future proposals only.

The expected accepted prefix is

```text
E_t = p_1 + p_1 p_2 + ... + p_1 ... p_L
```

and the expected emitted output is `Y_t = 1 + E_t`, including the verifier
replacement or bonus token. The first-step decision uses an online gain bucket
conditioned on refinement step, proposal length, expected-prefix ratio, and
remaining-mask ratio. Until eight valid transitions have bootstrapped that
bucket hierarchy, the controller continues to step two. Every later transition
also updates and queries its own step-conditioned gain buckets. When both a
matching bucket and the current trajectory are available, their estimates are
blended with weight `count / (count + prior_strength)`. The trajectory resets
whenever FailFast starts a new block, so block-start gain is never mixed with
refinement gain.

Acceptance position buckets are `0-1`, `2-3`, `4-7`, `8-15`, `16-23`,
`24-31`, and `32+`.

The decision compares renewal costs:

```text
J_stop     = (draft_elapsed + verify_round + post_verify) / Y_t
J_continue = (draft_elapsed + next_draft_pass + verify_round + post_verify)
             / (Y_t + predicted_gain)
```

The controller continues only when `J_continue < J_stop`. A stop reuses top-1
tokens already computed by the current dLLM pass and therefore adds no fill
forward pass.

Run the paired benchmark with:

```bash
python run_bucket_renewal_benchmark.py \
  --num_questions 15 \
  --max_new_tokens 1024 \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --dllm_dir /content/failfasttesting/Fast_dLLM_v2_1.5B
```

The command runs only bucket-renewal `N=8`. Its deterministic problem IDs match
the saved FailFast `N=8` benchmark generated with sample seed 2026. Pass that
report's `per_observation.csv` through `--reference_csv` to produce paired
comparisons without rerunning FailFast.

Validate gain and stop/continue cost predictions against measured
counterfactuals with:

```bash
python run_bucket_renewal_oracle_validation.py \
  --num_questions 15 \
  --max_new_tokens 1024 \
  --target_model_name Qwen/Qwen2.5-7B-Instruct \
  --dllm_dir /content/failfasttesting/Fast_dLLM_v2_1.5B
```

The shadow-oracle run evaluates the proposal produced by every denoising step
with greedy verification, including steps where the controller predicted
`continue`. It then forces refinement to continue so the next-step outcome is
also observed. Diagnostic verifier calls are excluded from the method timing.
