"""Experimental fixed random-outcome tape. Never overrides learned actions."""
import csv
import hashlib
import json
import math
from pathlib import Path


class ProbeTape:
    @staticmethod
    def fingerprint(snapshot, committed, step):
        snapshot = snapshot or {}
        physical = {'committed': committed, 'candidate': snapshot.get('candidate_prefix'),
                    'start': snapshot.get('active_block_start_relative'),
                    'end': snapshot.get('active_block_end_relative'),
                    'features': snapshot.get('features', []), 'step': step,
                    'proposal_length': snapshot.get('proposal_length')}
        return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()

    def __init__(self, path, trace_path):
        self.rows = {}
        with Path(path).open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                key = (int(row['problem_id']), int(row['decision_ordinal']))
                if key in self.rows:
                    raise ValueError(f'Duplicate probe tape key: {key}')
                row['probe'] = row['probe'].lower() in ('1', 'true')
                for name in ('mask', 'pos'):
                    row[name] = float(row[name])
                    if not math.isfinite(row[name]):
                        raise ValueError('Tape features must be finite')
                self.rows[key] = row
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text('', encoding='utf-8')
        self.ordinals = {}

    def begin(self, pid, snapshot, committed, step):
        ordinal = self.ordinals.get(pid, 0)
        self.ordinals[pid] = ordinal + 1
        row = self.rows.get((pid, ordinal))
        features = (snapshot or {}).get('features', [])
        match = bool(row is not None and len(features) >= 3
                     and abs(features[1] - row['mask']) <= 1e-6
                     and abs(features[2] - row['pos']) <= 1e-6)
        digest = self.fingerprint(snapshot, committed, step)
        if row and row.get('state_hash'):
            match = match and digest == row['state_hash']
        return {'problem_id': pid, 'decision_ordinal': ordinal, 'state_hash': digest,
                'mask': features[1] if len(features) > 1 else None,
                'pos': features[2] if len(features) > 2 else None,
                'scheduled': row is not None, 'state_match': match,
                'requested_probe': bool(row and row['probe']),
                'expected_action': row.get('action') if row else None,
                'probe_eligible': False, 'nominal_probability': 0., 'probe_selected': False}

    def select(self, event, probability):
        event['probe_eligible'] = True
        event['nominal_probability'] = probability
        selected = bool(event['requested_probe'] and event['state_match'])
        if selected and probability <= 0:
            raise RuntimeError('Tape requests an impossible random outcome')
        if not selected and probability >= 1:
            raise RuntimeError('Tape suppresses a mandatory probability-one probe')
        event['probe_selected'] = selected
        return selected

    def finish(self, event, action, source):
        event.update(action=action, action_source=source)
        event['action_match'] = event['expected_action'] == action
        with self.trace_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
