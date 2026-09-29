"""Tiny CPU fixtures: verify experiment isolation, real graph paths and packaging."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from world_model_core import Observation, pack_observations, ExperienceReplay
from world_model_probe import ProbeWorldModel
from world_model_probe_v2 import ImprovedProbeWorldModel, experiment_variants
from world_model_improvement_test import build_plan, make_learner, run
from offline_feature_audit import Archive, load_observations

torch.set_num_threads(1)


def observation(uid,q,length,accepted,segment=0):
    scalars=torch.zeros(length,16); scalars[:,3:5]=1; scalars[-1,0]=1
    context=torch.zeros(8); context[1]=length/64; context[2]=segment/64
    return Observation(uid,q,0,torch.tensor([[1,2]]*length),torch.randn(length,2,4),
        torch.tensor([[0.,-1.]]*length),scalars,context,accepted,
        torch.tensor([3,4]),torch.tensor([[2,5]]*length),torch.zeros(length,4),torch.ones(length))


def fixture(root):
    data=root/'data'; data.mkdir(); (data/'experience').mkdir()
    states=[]; labels=[]; edges=[]; teacher=[]
    for i,split in enumerate(('train','validation')):
        q=f'q{i}'
        nodes=[observation(q+'a',q,2,1),observation(q+'b',q,2,2),
               observation(q+'c',q,4,3,2),observation(q+'d',q,4,4,2),
               observation(q+'e',q,4,1,2)]
        lengths=np.array([n.length for n in nodes]); shard=f'experience/{i}.npz'
        arrays={k:torch.cat([getattr(n,k) for n in nodes]).numpy() for k in ('ids','hidden','gaps','scalars','history')}
        arrays.update(offsets=np.r_[0,lengths.cumsum()],context=torch.stack([n.context for n in nodes]).numpy(),
            aligned_topk_token_ids=torch.cat([n.topk_ids for n in nodes]).numpy(),
            teacher_margin=np.zeros(lengths.sum()),teacher_valid=np.zeros(lengths.sum(),dtype=bool))
        np.savez_compressed(data/shard,**arrays)
        for j,n in enumerate(nodes):
            states.append(dict(state_id=n.uid,question=q,round_id=0,split=split,shard=shard,row=j,
                               prefix_token_ids=n.prefix_ids.tolist()))
            labels.append(dict(state_id=n.uid,accepted_len=n.accepted,label_valid=True))
            teacher.append(dict(state_id=n.uid,margin=n.teacher_margin.tolist()))
        edges += [dict(parent=q+'a',child=q+'b',action='R'),dict(parent=q+'b',child=q+'c',action='E'),
                  dict(parent=q+'c',child=q+'d',action='R'),dict(parent=q+'a',child=q+'e',action='E')]
    for name,rows in [('states',states),('labels',labels),('edges',edges),('teacher_targets',teacher)]:
        (data/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    model=ProbeWorldModel(4,4,top_k=2,dim=16,num_hidden_layers=2,dropout=0)
    torch.save(dict(model_config=model.config,args={'extend_size':2}),data/'checkpoint.pt')
    table=torch.randn(16,4); emb=root/'embedding.safetensors'
    save_file({'model.embed_tokens.weight':table},str(emb))
    return data,states,labels,edges,emb


class ImprovementTests(unittest.TestCase):
    def test_shared_initialization_baseline_and_oracle_exclusion(self):
        config=ProbeWorldModel(4,4,top_k=2,dim=16,num_hidden_layers=2,dropout=0).config
        table=torch.randn(16,4)
        learner=make_learner(config,experiment_variants()['baseline'],table,'cpu',42,2)
        torch.manual_seed(42); reference=ProbeWorldModel(**config).eval()
        learner.model.eval()
        o=observation('a','q',2,1)
        b=pack_observations([o],table,'cpu')
        self.assertTrue(torch.equal(reference.encoder(b).tokens,learner.model.encoder(b).tokens))
        improved=make_learner(config,experiment_variants()['improved'],table,'cpu',42,2).model.eval()
        b=pack_observations([o],table,'cpu',include_candidates=True)
        before=improved.encoder(b)
        b['labels'].fill_(0); b['teacher_margin'].fill_(999); b['teacher_valid'].fill_(True)
        after=improved.encoder(b)
        self.assertTrue(torch.equal(before.tokens,after.tokens))
        e=improved.transition(before,torch.tensor([1]),2)
        self.assertTrue(torch.equal(improved.acceptance(e)[:,:2],improved.acceptance(before)))
        self.assertEqual(float(e.mask_probs[:,:2].sum()),0.)

    def test_attention_gradient_and_multistep_delta_loss(self):
        cfg=ProbeWorldModel(4,4,top_k=2,dim=16,num_hidden_layers=2,dropout=0).config
        learner=make_learner(cfg,experiment_variants()['improved'],torch.randn(16,4),'cpu',7,2)
        learner.warmup_updates=0; learner.horizon_warmup=0
        a,b,c=observation('a','q',2,1),observation('b','q',2,2),observation('c','q',4,3,2)
        replay=ExperienceReplay(); replay.add(a,b,'R'); replay.add(b,c,'E')
        replay.sample=lambda *args:[([a,b,c],['R','E'])]
        weights=copy.deepcopy(learner.model.state_dict())
        m=learner.update(replay,1,3)
        self.assertGreater(m['teacher_loss'],0); self.assertGreater(m['delta_loss'],0)
        self.assertTrue(np.isfinite(m['loss']))
        for key in ('encoder.candidate_query.weight','dynamics.gate'):
            self.assertFalse(torch.equal(weights[key],learner.model.state_dict()[key]))

    def test_all_arms_use_the_requested_loss_weights(self):
        config=ProbeWorldModel(4,4,top_k=2,dim=16,num_hidden_layers=2,dropout=0).config
        table=torch.randn(16,4)
        a,b=observation('a','q',2,1),observation('b','q',2,2)
        for name,variant in experiment_variants().items():
            with self.subTest(name=name):
                learner=make_learner(config,variant,table,'cpu',8,2)
                learner.warmup_updates=0
                replay=ExperienceReplay(); replay.add(a,b,'R')
                m=learner.update(replay,1,1)
                expected=(m['current_nll']+m['rollout_nll']+variant['latent_weight']*m['latent_loss']
                    +variant['teacher_weight']*m['teacher_loss']+variant['structure_weight']*m['structural_loss']
                    +variant['delta_weight']*m['delta_loss'])
                self.assertAlmostEqual(m['loss'],expected,places=5)

    def test_late_teacher_join_graph_and_split_protection(self):
        with tempfile.TemporaryDirectory() as d:
            data,states,labels,edges,emb=fixture(Path(d))
            archive=Archive(data); label_map={r['state_id']:r for r in labels}
            old=load_observations(archive,states,label_map)
            self.assertTrue(all(o.teacher_margin is None for o in old.values()))
            obs=load_observations(archive,states,label_map,include_teacher=True)
            self.assertTrue(all(o.teacher_margin is not None for o in obs.values()))
            plan=build_plan(states,obs,edges)
            self.assertEqual(len(plan['full_edges']),4)
            self.assertEqual({e[1] for e in plan['full_edges'] if e[0]=='q1a'},{'q1b','q1e'})
            self.assertIn([['q1a','q1b','q1c','q1d'],['R','E','R']],plan['paths'])
            self.assertFalse(set(plan['train_ids'])&set(plan['full_current']))
            broken=copy.deepcopy(states); broken[-1]['split']='train'
            with self.assertRaisesRegex(ValueError,'leakage'): build_plan(broken,obs,edges)

    def test_complete_small_experiment_and_error_zip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data,states,labels,edges,emb=fixture(root)
            args=SimpleNamespace(input=str(data),output=str(root/'result'),device='cpu',
                variants='baseline,teacher_only,improved,no_teacher',seeds='42',steps='0,16,17',
                panel_states=2,panel_edges=1,panel_paths=2,batch_size=2,eval_batch_size=2,
                embeddings=str(emb),embedding_repo='unused',embedding_revision='unused',cache=str(root),embedding_sha256='')
            run(args)
            summary=json.loads((root/'result/summary.json').read_text())
            self.assertEqual(summary['status'],'complete'); self.assertEqual(summary['llm_forward_calls'],0)
            self.assertEqual(len(summary['points']),12); self.assertEqual(summary['teacher_train'],5)
            self.assertEqual(summary['full_validation_edges'],4)
            effects=json.loads((root/'result/paired_factor_impacts.json').read_text())
            self.assertIn('full/h1_R',effects['improved_minus_baseline'])
            with zipfile.ZipFile(root/'result.zip') as z:
                self.assertIsNone(z.testzip()); self.assertIn('learning_curves.png',z.namelist())
                self.assertNotIn('embedding.safetensors',z.namelist())
            args.output=str(root/'failed')
            with patch('world_model_improvement_test.load_embeddings',side_effect=RuntimeError('test fail')):
                with self.assertRaisesRegex(RuntimeError,'test fail'): run(args)
            with zipfile.ZipFile(root/'failed.zip') as z:
                self.assertEqual(json.loads(z.read('summary.json'))['status'],'partial_error')
                self.assertIn('test fail',z.read('error.txt').decode())


if __name__=='__main__': unittest.main()
