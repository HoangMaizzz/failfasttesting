# Native-rule audit and fixes

Scope: the executable V2 capture → normalization → learner → H1–H3 rollout →
archive → Kaggle launcher path. Legacy verifier/controller experiments are not
redefined by this revision. The instrumentation is optional; default native
token selection is unchanged.

## Why the old run could violate native rules

Mask, commit event, active region, context and KV were independent neural
readouts. Binary losses encouraged agreement but did not enforce the native
state machine. A latent residual skip also did not preserve the original
1536-channel residual at the decoder. Consequently, legal structure, hidden
reconstruction, and unknown dynamics were conflated in the evaluation.

## Corrections

| Area | Correction |
|---|---|
| Physical event | Full32 same-forward pre/post mask delta; logical proposal span kept separately |
| Forced progress | At least one masked position in the whole native8 region, including positions outside the logical proposal |
| R | No remask, exactly adjacent native unmask forward, same geometry and exact prefix-KV carry |
| E start | Parent same-forward top1 fill → fresh native segment; first pre-mask computed from seed/absolute geometry |
| Seed | Inserted by prefill only at aligned prefixes longer than32; cache-update seeds a crossing frame; no artificial seed for unaligned prefixes |
| Native region | Full next native8 region, empty if logical proposal resolved; not a separately learned binary head |
| Coordinates | Exact `[proposal_start, frame_start, KV_length, length8, segment]`; identity carry maps absolute positions |
| Cross-frame conditioning | New positional queries attend to all source latent rows; source semantics are not zeroed when the identity frame changes |
| Event alignment | Explicit native right-shift `[row0,row0,row1,...]`, without a vocabulary target |
| Residual forecast | Native U identity + learned change, rather than reconstruction through a 128D bottleneck |
| Prefix memory | Upstream layers0/7/15/23 for the pre-layer24 target, not exclusively the discarded suffix layers24–27 |
| EOS | Presence differs from terminal readiness; continue native R while masks remain before EOS; no E after EOS in the filled prefix |
| Unobserved EOS | Retain termination-only transitions; never invent hidden tensors for them |
| Post-EOS tail | Excluded from supervised loss/scoring once native EOS is terminal |
| Known counters | R increment, E first frontier-step, same-frame E denoising count and physical phase are computed; only bridge denoising count is unknown |
| Loss | Event supervision on native eligible positions; H1-first curriculum; weight by valid transition count |
| Feature comparison | Same initialization and minibatch seeds, same question split; checkpoint selection uses validation only |
| Readout | Proposal wrong-position/exact-mask measures, full event FP including illegal positions, sequences and frame crossings, EOS-positive coverage |
| Delta evaluation | Score mapped overlaps only; new-frame residual error reported separately |
| Normalization | Train-only mean/std; do not clip target tails and then call their error native-unit RMSE |
| Data integrity | Missing endpoints/unknown actions fail; native contract checked before training |
| Recovery | Save sanitized reusable capture before training; partial ZIP includes features/status/full traceback |
| Packaging | Self-contained checkpoint config/scalers; corrected-folder replay; old V2 schema rejected |
| Launcher | Unique checkout directory, pinned source module next to weights, native integration tests before capture |

## Verification

Focused tests exercise the learner/gradients/multi-step feedback, corrected
capture/writer, native runner, asset installation, data reload, terminal-edge
sanitization and preservation on a forced training failure.

`test_native_latent_generator_rules_v2.py` executes the actual native generator
method with synthetic forward outputs. It covers prefix phases0/1/7/25/31,
one-commit progress, R adjacency, KV invariance, and EOS with unresolved earlier
masks. CUDA is used only for the existing native timing events. This is **not**
a pretrained1.5B accuracy measurement. Kaggle performs the fresh real-weight
capture and held-out learning evaluation.

## Remaining intentional limitations

1. E is the user-specified fill-then-fresh-segment macro, not in-place native
   speculative extension. It may contain several real model forwards.
2. U is pre-commit; G is post-commit. This compressed observation omits actual
   replacement-token representations and full position-indexed KV. Its Markov
   sufficiency is unproven.
3. Structural event scores are **not** native vocabulary confidence. Enforcing
   legal eligibility/progress cannot guarantee the same selected positions.
4. No token IDs, token probabilities, verifier labels, accepted-token counts or
   latency are learned. No frozen native suffix is executed inside the learner.
5. Matched single-seed ablations reduce confounding but do not replace repeated
   seeds or broader task/model validation. Sparse EOS positives are reported as
   insufficient coverage rather than evidence of perfect EOS prediction.

Thus this revision removes avoidable protocol and structural errors. It does
not claim that the latent simulator is already accurate enough; that conclusion
requires the new held-out R/E results.
