# Structured sparse S/R/E collection (v2)

Run `structured_sparse_collector.py`. The old `sparse_extend_world_model_collector.py`
CLI delegates to this implementation. Do not reuse the legacy K=4 command: the
new protocol accepts K=1 or K=2 and defaults to K=2.

## Graph and collection policy

For each independently sampled anchor, collect up to three additional R actions.
Probe E at every available R boundary, including R0. Store every E child and its
Submit label. Across ALL parents at the next length, select at most two children:
best accepted prefix, then a deterministic diverse child (different source
refinement depth, mask pattern, acceptance). No random branch sampling.

Each selected E child starts a fresh three-R budget. Repeat through L=64 by
default: starting at L=8 this is seven E actions. An all-resolved or EOS state
does not get fictitious R edges. EOS also disables E. At maximum length selected
children may still have R probes, but no more E edges.

Default deep-expansion eligibility is accepted_len / L >= 0.5. Every bad E child
is stored. One deterministic bad child per level gets up to two extra R probes,
but those probes never recursively extend. Parameters expose both the threshold
and negative-probe budget. This is a collection heuristic, not a proof that
low-yield states cannot recover; retain negative probes when evaluating it.

Choose up to four L=8 anchors per question round-robin across full, near-full,
mid, and early-mismatch regimes, using all archived rounds by default. Record
missing regimes. Different anchors run sequentially; K=2 is global across parents
inside one anchor search, not two children per parent. An archived anchor may be
an intermediate native boundary: its local R budget is reset at the anchor, and
the original boundary index is retained rather than described as a new E.

## Exact state/action conventions

State consists of the exact verifier prefix, current proposal tokens/mask, and
the collector's local R budget. E appends `extend_size` masks, then performs one
unmask forward in the first unresolved logical frame of the NEW segment. Parent
proposal tokens are unchanged by E. R performs one unmask forward in the earliest
unresolved proposal-relative logical frame. Threshold defaults to 0.3, with a
forced highest-confidence masked position only if none exceeds threshold.

This protocol uses the existing dLLM weights and block-causal forward with a
physical block size of 32. It is an explicit-state replay protocol, NOT a claim
of bitwise equivalence to the old cached arbitrary-length generation loop.
It does not insert an implicit autoregressive token at a physical block boundary.
Input is prefix + exact proposal + mask padding to the physical block boundary;
absolute positional indices are preserved. Logical unmask frames default to 8.

Every node is re-encoded in its entirety before counterfactual fill. Hidden
states come from positions `prefix_length + i`; prediction logits come from
`prefix_length + i - 1`. All layers/features belong to the same exact native
input, whose hash is stored. No inherited hidden-state rows or fallback merges.
Submit fills only masked positions using this observation's predictions and
compares with a cached greedy target continuation. It never mutates native state.
This is a documented v2 observation protocol; do not silently mix its features
with the legacy archives' features from forwards preceding native commits.

## Files and labels

- `nodes.jsonl`: canonical node metadata including all roots and Submit labels.
- `edges.jsonl`: true one-step R/E edges, action cost, extension delta and source.
- `raw/<dataset>_structured/*.npz`: compressed raw features, tokens and prefixes.
- `raw/<dataset>_structured/index.jsonl`: same final metadata plus shard/row.
- `reference_cache/*.json`: greedy tokens, prefix, verifier/tokenizer identity.
- `verifier_calibration.jsonl`: actual timing samples by prefix and proposal length.
- `graph_manifest.json`: configuration, protocol, missing regimes and completion.
- `<dataset>_structured_graph.zip`: all artifacts, also produced on caught failure.

There is no duplicate raw Stop state: every node has `submit_accepted_len`,
`submit_emitted_len`, emitted token IDs, and calibrated verifier cost.
`state_masks_resolved` describes the current state; collection pruning/selection
are separate fields. The collector has no misleading per-state rollout
termination_reason. Node/edge IDs are independent of selection decisions.

## Timing and limitations

Greedy continuation is cached per exact prefix and target identity, to Lmax+1 or
EOS. Original archived root Submit acceptance is checked against it; references
are generated once when not already available, not magically recovered from old
ZIPs that do not contain them.
Use `--reference_cache_dir` to reuse references between runs. The output archive
also includes every reference used in that run. Lmax is a proposal limit:
full acceptance may emit Lmax+1 including the verifier bonus token. Set
`--remaining_output_budget` explicitly to model a smaller remaining output budget.

For each selected context, verifier calibration measures L=8,16,...,Lmax on the
actual target GPU with prefix KV reuse (prefix excluding its last token, L+1
query tokens). One warm-up is discarded and three repetitions are stored. Each
probe gets an isolated prefix cache. Per-node latency is explicitly a calibrated
estimate, not a per-node measurement or final deployment reward ground truth.
No linear L8 scaling is used.

Drafter edge cost measures its full-context replay forward. R reuses the exact
source observation's measured forward; E measures the appended-mask input
forward. Destination re-encoding and raw export are collection overhead and are
reported separately. These costs are NOT the latency of an optimized cached
drafter, and exclude unmask bookkeeping, transfer and controller overhead.
`stop_latency_per_output_token_from_anchor` includes only draft action costs
after the sampled anchor plus calibrated verifier cost. It excludes old archive
elapsed costs; do not interpret it as an optimal Continue label. No oracle policy
label is manufactured from the calibrated estimates.

Test with 3 questions before a large run. Full-context re-encoding improves
state/feature consistency but takes more collection time than the old merged
feature implementation. The test suite checks graph actions, branch limits,
position alignment through L=64, NPZ round trips and verifier cache isolation
with small deterministic models; it is not a pretrained T4x2 benchmark.
