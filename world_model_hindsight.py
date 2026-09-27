"""Exact greedy-prefix labels from actual emitted tokens; never from unaccepted logits.

One instance per question. Candidates are captured at visited decision boundaries,
not generated again for labeling. Insufficient reference is censored, not K=0.
"""
from __future__ import annotations


class HindsightLabeler:
    def __init__(self, prompt, on_label=None):
        self.verified = list(prompt)
        self.records = {}
        self.on_label = on_label or (lambda record: None)

    def register(self, state):
        if state.prefix != self.verified:
            raise ValueError("State must start at the currently verified prefix")
        o = state.observation
        if o.uid in self.records:
            raise ValueError("Duplicate state ID")
        self.records[o.uid] = dict(observation=o, offset=len(state.prefix), lower_bound=0,
                                   source=None)

    def _resolve(self, record, accepted, source):
        o = record["observation"]
        if o.accepted is not None and o.accepted != accepted:
            raise RuntimeError("Direct verifier and hindsight labels disagree")
        o.accepted = int(accepted)
        if record["source"] is None:
            record["source"] = source
            self.on_label(dict(state_id=o.uid, accepted_len=o.accepted, label_valid=True,
                lower_bound=o.accepted, source=source, status="exact"))

    def after_stop(self, state):
        if not state.submitted or state.observation.accepted is None:
            raise ValueError("Submit STOP to the verifier first")
        if state.prefix != self.verified:
            raise ValueError("STOP prefix differs from the verified stream")
        record = self.records[state.observation.uid]
        if record["source"] is not None:
            raise ValueError("STOP already processed")
        self._resolve(record, state.observation.accepted, "direct_stop_verifier")
        # Only emitted accepted-prefix + correction/bonus tokens are on-policy.
        # Full verifier argmax logits after a mismatch are NOT a reference stream.
        self.verified.extend(state.emitted)
        for record in self.records.values():
            o = record["observation"]
            candidate = o.ids[:, 1].tolist()
            reference = self.verified[record["offset"]:record["offset"]+o.length]
            mismatch = next((i for i, (a, b) in enumerate(zip(candidate, reference)) if a != b), None)
            if mismatch is not None:
                self._resolve(record, mismatch, "hindsight_verified_stream")
            elif len(reference) == o.length:
                self._resolve(record, o.length, "hindsight_verified_stream")
            else:
                record["lower_bound"] = len(reference)

    def finish(self, reason):
        for record in self.records.values():
            if record["source"] is None:
                self.on_label(dict(state_id=record["observation"].uid, accepted_len=None,
                    label_valid=False, lower_bound=record["lower_bound"], source=None,
                    status="insufficient_verified_suffix", collection_end_reason=reason))
        exact = sum(r["source"] is not None for r in self.records.values())
        return dict(states=len(self.records), exact=exact, unresolved=len(self.records)-exact)
