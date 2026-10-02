"""Exact greedy-prefix labels from actual emitted tokens; never from unaccepted logits.

One instance per question. Candidates are captured at visited decision boundaries,
not generated again for labeling. Insufficient reference is censored, not K=0.
"""
from __future__ import annotations


class HindsightLabeler:
    def __init__(self, prompt, on_label=None, on_conflict=None):
        self.verified = list(prompt)
        self.records = {}
        self.on_label = on_label or (lambda record: None)
        self.on_conflict = on_conflict or (lambda record: None)
        self.conflict_count = 0

    @staticmethod
    def _priority(source):
        return {
            "direct_stop_verifier": 3,
            "direct_shadow_verifier": 2,
            "shadow_verifier": 2,
            "hindsight_verified_stream": 1,
        }.get(source, 0)

    def register(self, state):
        if state.prefix != self.verified:
            raise ValueError("State must start at the currently verified prefix")
        o = state.observation
        if o.uid in self.records:
            raise ValueError("Duplicate state ID")
        self.records[o.uid] = dict(observation=o, offset=len(state.prefix), lower_bound=0,
                                   source=None, accepted_emitted=None)

    def _resolve(self, record, accepted, source):
        o = record["observation"]
        if o.accepted is not None and o.accepted != accepted:
            conflict = dict(state_id=o.uid, accepted_existing=int(o.accepted),
                existing_source=record["source"], accepted_new=int(accepted),
                new_source=source, action="preserve_higher_priority_label")
            self.conflict_count += 1
            self.on_conflict(conflict)
            if self._priority(source) <= self._priority(record["source"]):
                return
        o.accepted = int(accepted)
        should_emit = (record["source"] is None or
                       self._priority(source) > self._priority(record["source"]) or
                       record["accepted_emitted"] != int(accepted))
        if should_emit:
            record["source"] = source
            record["accepted_emitted"] = int(accepted)
            self.on_label(dict(state_id=o.uid, accepted_len=o.accepted, label_valid=True,
                lower_bound=o.accepted, source=source, status="exact"))

    def mark_direct(self, state, source="shadow_verifier"):
        """Mark a shadow-scored state so a later branch cannot relabel it."""
        if state.observation.accepted is None:
            raise ValueError("Direct verifier result is missing")
        record = self.records[state.observation.uid]
        self._resolve(record, state.observation.accepted, source)

    def after_stop(self, state):
        if not state.submitted or state.observation.accepted is None:
            raise ValueError("Submit STOP to the verifier first")
        if state.prefix != self.verified:
            raise ValueError("STOP prefix differs from the verified stream")
        record = self.records[state.observation.uid]
        self._resolve(record, state.observation.accepted, "direct_stop_verifier")
        # Only emitted accepted-prefix + correction/bonus tokens are on-policy.
        # Full verifier argmax logits after a mismatch are NOT a reference stream.
        self.after_emitted(state.prefix, state.emitted)

    def after_emitted(self, prefix, emitted):
        """Advance even when native EOS had no trainable raw state."""
        if list(prefix) != self.verified:
            raise ValueError('Emitted stream must extend the verified prefix')
        self.verified.extend(emitted)
        for record in self.records.values():
            # Resolved direct labels belong to their own candidate. A later
            # refinement/extension can follow a different token path, so its
            # emitted stream must not overwrite those exact labels.
            if record["source"] is not None:
                continue
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
            if record['source'] is None and record['observation'].accepted is not None:
                self._resolve(record, record['observation'].accepted, 'direct_shadow_verifier')
            if record["source"] is None:
                self.on_label(dict(state_id=record["observation"].uid, accepted_len=None,
                    label_valid=False, lower_bound=record["lower_bound"], source=None,
                    status="insufficient_verified_suffix", collection_end_reason=reason))
        exact = sum(r["source"] is not None for r in self.records.values())
        return dict(states=len(self.records), exact=exact, unresolved=len(self.records)-exact,
                    label_conflicts=self.conflict_count)
