#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--decisions',required=True,help='BEST_DECISIONS_REPLAY.csv from closed-loop search')
    p.add_argument('--trajectory',required=True,help='BEST_TRAJECTORY_50.csv')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    d=pd.read_csv(a.decisions); t=pd.read_csv(a.trajectory)
    # Keep the logged warm-up too: it can change the initial learner state.
    d['decision_ordinal']=d.groupby(d.problem_id.astype(int),sort=False).cumcount()
    psel=d.copy()
    out=pd.DataFrame({
        'problem_id':psel.problem_id.astype(int),
        'decision_ordinal':psel.decision_ordinal.astype(int),
        'probe_type':psel.simulated_source.astype(str),
        'expected_mask':psel['mask'].astype(float),
        'expected_pos':psel['pos'].astype(float),
        'expected_structural_eligible':psel.structural_probe_eligible.astype(bool),
        'source_snapshot_id':psel.snapshot_id.astype(int),
        'source_boundary':psel.boundary.astype(int),
        'offline_score':psel.score.astype(float),
        'probe':psel.simulated_source.astype(str).isin(['structural_probe','floor_probe']),
        'mask':psel['mask'].astype(float),
        'pos':psel['pos'].astype(float),
        'action':psel.recorded_action.astype(str),
        'state_hash':psel.state_hash.astype(str),
    })
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(f'wrote {len(out)} deterministic probes to {a.output}')
    print(out.probe_type.value_counts().to_string())

if __name__=='__main__': main()
