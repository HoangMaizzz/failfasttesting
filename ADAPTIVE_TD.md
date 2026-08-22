# Online TD Adaptive Refinement

## Control flow

The original path runs Fast-dLLM denoising with its confidence threshold and
forced-progress rule, finalizes a proposal, applies FailFast's outer confidence
gate and length extension, and verifies the resulting proposal greedily with the
AR target model.

With `--adaptive-td`, the generation loop remains unchanged until a Fast-dLLM
forward and the original unmask operation have completed. The controller then
observes the current refinement state and chooses `STOP` or `CONTINUE`.
`CONTINUE` resumes from the current `x_t`, masks, KV state, and block state.
`STOP` is eligible when all remaining proposal positions have current top-1
candidates. Remaining masks are filled without another forward pass, then the
unchanged outer FailFast gate independently chooses verification or length
extension.

## State

The controller uses 13 bounded features:

1. bias;
2. remaining-mask ratio;
3. newly-unmasked ratio;
4. mean recoverable confidence;
5. minimum recoverable confidence;
6. maximum recoverable confidence;
7. recoverable-confidence standard deviation;
8. mean recoverable top-1/top-2 margin;
9. normalized first remaining-mask position;
10. normalized FailFast confidence-frontier length;
11. proposal-change ratio;
12. recoverable-position change ratio;
13. optional normalized `log(1 + refinement_step)`.

The step feature is disabled by default. Margin and stability features can be
disabled independently.

## Learning

Two shared linear action values are maintained for all refinement depths:

```text
Q_STOP(s) = theta_STOP dot phi(s)
Q_CONTINUE(s) = theta_CONTINUE dot phi(s)
```

The throughput reference is the ratio of online EMAs of emitted tokens and
measured round latency. A factual `CONTINUE` transition receives immediate cost
`-rho * next_forward_latency_ms` and a one-step TD bootstrap from the observed
next state. After normal verification, the selected `STOP` state receives
`emitted_tokens - rho * verifier_latency_ms`. Reverse factual returns propagate
the terminal result through only the actions that were actually executed.

No oracle value updates the controller. Oracle collection remains a separate
diagnostic path and `--bucket_oracle_force_continue` is rejected with adaptive
TD because it would invalidate the factual trajectory.

## Risk and exploration

Each action tracks residual variance and diagonal feature precision. Mean-value
uncertainty is estimated as residual variance multiplied by diagonal leverage,
so it contracts as matching observations accumulate. A stop is selected only
when its lower bound exceeds the continue upper bound plus the configured
margin. Overlapping intervals default to continue, except for optional low-rate
factual stop exploration. Mixed TD/return updates count each executed
state-action pair once in precision and residual statistics. Parameters reset
for every `failfast.py` process and persist across samples within that dataset
run.

The default online schedule starts stop exploration at `0.10`, decays it by
`0.998` per eligible decision, and retains a `0.01` floor. This supplies early
STOP outcomes during a 50-sample run while keeping exploration explicitly
separate from learned `stop_interval_dominates` actions in the decision log.

## Baseline preservation

Adaptive TD is disabled by default. `--adaptive-force-continue` executes the
feature and learning hooks but never requests an early stop; it is intended for
GPU token-hash equivalence testing. The paired benchmark runner also supports
fixed-depth methods `fixed_r1`, `fixed_r2`, and `fixed_r3`.

## Benchmark

```bash
python run_adaptive_td_benchmark.py \
  --datasets math aime gsm8k humaneval \
  --methods failfast adaptive_td \
  --num_questions 15 \
  --max_new_tokens 1024 \
  --spec_len 8 \
  --incr_len 8
```

The primary comparison reuses the existing matched FailFast8 baseline and uses
the same `spec_len=8`, `incr_len=8`, and sampled problem IDs. A `10/10` run is a
separate sensitivity experiment against the original outer-length setting.

Primary timing runs leave decision logging and detailed profiling disabled.
For learning curves and controller component percentiles, add
`--adaptive-log-decisions --adaptive-profile-overhead` and treat that run as
instrumented rather than the primary throughput measurement.

For a baseline-equivalence run:

```bash
python run_adaptive_td_benchmark.py \
  --datasets gsm8k \
  --methods failfast adaptive_force_continue \
  --num_questions 3 \
  --max_new_tokens 256
```

The runner writes per-problem results, paired speedups, output-token hash
matches, dataset summaries, decision traces, online learning curves, and the
final controller state with overhead percentiles.

## Architectural constraints

The default model path computes a full 32-token block and applies unmasking to
the active 8-token small block. Adaptive TD retains top-1 candidates only for
proposal positions already covered by that forward. Positions outside the
computed block still force `CONTINUE`; this avoids inventing candidates or
adding a dLLM forward to `STOP`.

The controller hook is implemented at the existing inner-loop iteration
boundary instead of exposing the model's cache-rich refinement state through a
new public API. This preserves the repository's sampling and cache behavior,
but full token-hash equivalence still requires a GPU model run.

## Known limitations

- `STOP` is exposed only when current top-1 values cover every remaining
  proposal mask. The outer FailFast extension policy runs afterward and is not
  part of inner refinement-stop availability.
- The maximum refinement depth is a guardrail only when zero-forward
  finalization is available. A proposal spanning an unseen small block may run
  past the nominal guard until that block has current candidates.
- Updates are factual and on-policy. Low-rate stop exploration supplies some
  STOP outcomes, but cannot remove all selection bias without counterfactual
  verifier calls.
- The online throughput reference changes as the policy changes, so the TD
  target is mildly nonstationary. Reset the controller for every dataset,
  model pair, hardware setup, or major decoding configuration.
- The repository's dLLM timing already synchronizes CUDA after each forward.
  The controller adds no production synchronization, but this pre-existing
  synchronization remains part of measured draft latency.
- Top-1/top-2 margin requires a vocabulary `topk`. It is optional and should be
  disabled with `--no-adaptive-use-margin-feature` when its measured gain does
  not justify its cost.
