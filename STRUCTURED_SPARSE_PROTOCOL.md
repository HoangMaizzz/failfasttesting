# Structured sparse S/R/E collection (v3)

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
the collector's local R budget. Before E, all remaining parent masks are
committed using top-1 tokens from the already-available parent observation;
this fill adds no forward. E then appends `extend_size` fresh masks and performs
one unmask forward in the first absolute-position logical frame of the NEW
segment. R commits in the earliest unresolved absolute-position logical frame.
Frame alignment uses `prefix_length + proposal_index`, not proposal-relative
index. Threshold defaults to 0.5, with a forced highest-confidence masked
position only if none exceeds threshold. E prefill is exempt from the threshold:
it must resolve every remaining parent mask before the extension is created.
The exact prefill positions and token IDs are stored on each E edge.

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
This is a documented structured-sparse observation protocol; do not silently mix its features
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

Schema v3 distinguishes `backbone_acceptance_regime` (the source label used to
stratify initial anchors) from `current_submit_regime` (computed from the
current state's Submit accepted length and proposal length). Expansion ranking,
eligibility, and reported current-regime statistics use the current Submit
outcome. The manifest reports source and current anchor regimes separately.
Each node also has `observation_hash`, computed from prefix, native proposal,
and the exact stored drafter observation (excluding timing noise), plus duplicate
group size/flag. Duplicate observations remain separate graph records so their
distinct paths/edges are preserved; training samplers should deduplicate or
downweight by `observation_hash`, and train/test splits must be by problem or
prefix rather than random node.

Nodes and calibration rows share `verifier_calibration_key` (target identity
plus exact prefix); `context_hash` separately identifies the exact prefix.
Join calibration by `verifier_calibration_key` and `proposal_length`, never by
context length alone. The reference-cache JSON stores both keys as well.

## Timing and limitations

Greedy continuation is cached per exact prefix and target identity, to Lmax+1 or
EOS. Original archived root Submit acceptance is compared against it; any
mismatch is retained as provenance while current labels use the current greedy
reference. References are generated once when not already available, not magically recovered from old
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
The FP16 target verifier is automatically sharded across both selected GPUs,
with a default 9 GiB weight budget per GPU, leaving memory for the drafter on
GPU 1 and verifier KV calibration. Target weights remain FP16, preserving the
verifier's greedy-label definition. The shard map is printed at startup.

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

## Optional measured no-KV verifier and stable-block drafter KV (v4)

Use `--verifier_mode full_context_no_kv --drafter_kv_mode stable_block_prefix`
for the new measured run. Its unmask threshold defaults to 0.5; explicitly pass
`--drafter_threshold 0.3` when reproducing the older collection.
The updated E prefill and absolute-frame semantics write schema
`structured_sparse_sre_v5` and must use a new output
directory. Every Submit candidate is sent through an actual full-prefix target
forward with `use_cache=False`, including the correction/bonus logit. Its
`submit_verifier_latency_ms` is a node measurement, not a prefix-KV estimate.
Archived anchor acceptance is independently rechecked with that same verifier.
`verifier_calibration.jsonl` contains direct per-proposal timing records in
this mode, not calibrated median rows.

The drafter KV mode caches only complete physical blocks strictly before the
first mask, keyed by the exact token prefix. Mutable blocks are recomputed.
The cache holds at most two prefixes; a cache miss includes the prefill cost.
At a physical block's first token it uses that position's logit, matching the
production block-shift rule; other positions use the preceding logit. Hence
KV observations are deliberately **not** bitwise comparable with the older
all-positions `p-1` logits. In this mode an R action charges the destination
observation forward. E charges both the appended-mask and destination forwards.
The raw feature export and CPU transfer remain collection overhead. This is
stable-block-prefix KV reuse, not the mutable denoising-block cache path of
`generate_draft_tokens_arbitrary_length`; a T4x2 smoke test must compare it
against production before interpreting latency as deployment ground truth.

## Native Elysia unmask backend (smoke)

Select `--unmask_backend native_elysia --drafter_threshold 0.5
--verifier_mode full_context_no_kv`. The source archive supplies verifier
prefixes and IDs only. Each eight-token segment is regenerated with the same
`generate_draft_tokens_arbitrary_length` entrypoint used in `failfast.py`.
Successive native oracle snapshots form R edges. For E, all parent masks are
filled from that parent's same-forward native top-1 snapshot; the filled
proposal is appended to the verifier prefix before calling the native generator
for the next eight-token segment. A fresh native call is made for each E branch.
If a segment crosses a physical-block boundary, early forwards may not yet
have a complete same-forward top-1 fill for all eight positions. Those bridge
forwards are charged in native elapsed latency but are not labeled as S/R
states. Collection stops after the first four *observable* native snapshots
(one initial state plus three R states), or when the segment resolves earlier.
No archived draft tokens or logits become graph nodes. The raw archive can have
a different threshold without silently mixing trajectories.

Submit proposals come only from native snapshot `proposal_token_ids_after_fill`.
Each is sent to the real verifier. Full-state re-encoded hidden/top-k features
remain in the standard NPZ fields and are explicitly *diagnostic*, not the
action-producing forward. The extra `native_active_*` NPZ fields contain raw
hidden/top-k rows from the native forward; their valid masks and start offsets
in the index identify which part of the current eight-token segment they cover.
The manifest uses `structured_sparse_native_elysia_v1` to prevent these data
from being confused with explicit-state replay archives.

The native generator call matches the actual `failfast.py` call's default
`use_block_cache=False`; it uses prefix KV within each call. A separate E call
starts a fresh prefix prefill, so its measured cost can differ from deployment
which may reuse an already available prefill output. This is a latency caveat,
not a different unmask/STOP token rule. Check a five-question GPU smoke before
running larger collections.
