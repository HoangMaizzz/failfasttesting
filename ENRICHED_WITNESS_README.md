# INT8 no-KV enriched-good-probe frozen witness test

This bundle does **not** search a new trajectory. It replays the precomputed 50-problem enriched-good-probe positive-control candidate from zero learner weights.

## Frozen candidate

Offline oracle-guided simulation source: `outputs_int8_nokv_math_pool`.

- 50 fixed MATH problem IDs.
- 214 fixed probe states, mapped to exact `(problem_id, decision_ordinal)` keys from the physical discovery log.
- 192 structural probes and 22 floor probes.
- 38/214 = 17.76% of the offline scheduled probes were beneficial-C states.
- Offline simulation: 6 learned CONTINUE, 6 TP / 0 FP, aggregate learned-C benefit +178.345 ms/token-equivalent.
- Offline total probe count was 214 versus 223.94 nominal expected (95.56% of expectation).

The offline result is **not** treated as the final result. This patch resets U1 and physically reruns the trajectory.

## Runtime setting

- verifier: Qwen2.5-7B-Instruct, INT8 weight quantization, FP16 compute
- verifier speculative KV cache: OFF (`use_cache=False`)
- drafter: Fast_dLLM_v2_1.5B FP16
- reusable drafter KV: OFF
- U1 learner starts from zero weights and uses the original LR / replay / threshold rules
- random probe draws are disabled only for this existence-test replay; probe execution comes from the frozen schedule

## Strong replay audit

A run can PASS only if:

1. at least `min_learned_c` actual learned CONTINUE transitions are resolved (default 3);
2. actual aggregate learned-C benefit is positive (`-sum(delta_J) >= 1.0` by default);
3. the learner actually crosses score 0.5;
4. zero frozen-state mismatches occur;
5. **all 214 frozen schedule rows are encountered**, and every one is either executed as a probe or shadowed by an autonomous learned CONTINUE at that same state.

AlwaysSTOP speed is not a criterion.

## Kaggle

Upload `math50_int8_nokv_enriched_goodprobe_witness_test.zip` and run `KAGGLE_CELL_INT8_NOKV_ENRICHED_WITNESS.py`.

The result ZIP will contain `FINAL_WITNESS_SUMMARY.json`, `VERDICT.txt`, `ACTUAL_LEARNED_CONTINUES.csv`, `ACTUAL_PROBE_TRANSITIONS.csv`, the full decision/transition logs, and benchmark results.
