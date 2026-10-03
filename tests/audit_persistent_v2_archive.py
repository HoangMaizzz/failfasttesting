"""Read-only compatibility audit against a real TwoSource ZIP; no GPU/LLM calls."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from world_model_core import Observation
from persistent_world_model_v2 import pack_v2,BehavioralWorldModelV2
from run_persistent_world_model_v2 import enumerate_paths


def audit(path):
    torch.set_num_threads(2)
    with zipfile.ZipFile(path) as zf:
        summary=json.loads(zf.read('summary.json'));cfg=json.loads(zf.read('config.json'))
        def lines(name):return [json.loads(row) for row in zf.read(name).splitlines() if row.strip()]
        states=lines('states.jsonl');labels={r['state_id']:r for r in lines('labels.jsonl')}
        teachers={r['state_id']:r for r in lines('teacher_targets.jsonl')}
        edges=[(r['parent'],r['child'],r['action']) for r in lines('edges.jsonl')]
        questions=sorted({r['question'] for r in states})
        shells={r['state_id']:SimpleNamespace(length=r['length'],question=r['question'],round_id=r['round_id'],
            accepted=labels.get(r['state_id'],{}).get('accepted_len')) for r in states}
        paths=enumerate_paths(shells,edges,questions)
        chosen={}
        for r in states:
            if r['state_id'] in labels and r['state_id'] in teachers:
                chosen.setdefault(r['question'],r)
        by_shard=defaultdict(list)
        for r in chosen.values():by_shard[r['shard']].append(r)
        observations=[]
        for shard,rows in by_shard.items():
            with zf.open(shard) as stream,np.load(stream,allow_pickle=False) as a:
                for r in rows:
                    row=int(r['row']);lo,hi=map(int,a['offsets'][row:row+2]);uid=r['state_id']
                    t=teachers[uid];label=labels[uid]
                    o=Observation(uid,r['question'],r['round_id'],torch.tensor(a['ids'][lo:hi]).long(),
                        torch.tensor(a['hidden'][lo:hi]).half(),torch.tensor(a['gaps'][lo:hi]).half(),
                        torch.tensor(a['scalars'][lo:hi]).float(),torch.tensor(a['context'][row]).float(),
                        int(label['accepted_len']),teacher_margin=torch.tensor(t['margin']).float(),
                        teacher_features=torch.tensor(t['features']).float())
                    observations.append(o)
        first=observations[0];model=BehavioralWorldModelV2(first.hidden.shape[-1],first.hidden.shape[1]).eval()
        with torch.no_grad():
            b=pack_v2(observations[:8],{},'cpu');loss,terms=model.representation_loss(b)
        report=dict(schema=summary.get('schema'),status=summary.get('status'),questions=len(questions),
            states=len(states),edges=len(edges),paths_by_horizon=dict(Counter(len(p[1]) for p in paths)),
            sampled_real_observations=len(observations),hidden_shape=list(first.hidden.shape),
            hidden_layers=cfg.get('hidden_layers'),finite_behavior_loss=bool(torch.isfinite(loss)),terms=terms)
        print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);audit(p.parse_args().zip)
