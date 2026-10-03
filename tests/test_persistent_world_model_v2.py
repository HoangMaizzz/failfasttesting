import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from world_model_core import Observation,expected_acceptance
from persistent_world_model_v2 import (BehavioralWorldModelV2,FixedDynamicsV2,StateV2,
    GatedFiLMV2,pack_v2,behavioral_targets,behavioral_loss,teacher_token_ids,film_acceptance_loss)
from run_persistent_world_model_v2 import (enumerate_paths,causal_memory,rollout_loss,
    metrics,run,parser,Balanced)
from world_model_film_v2 import replay_tail,NativeTailHook,token_transition
from kaggle_persistent_wm_v2 import locate_input

torch.set_num_threads(1)


def observation(uid='s',qid='q',length=8,y=3,round_id=0,refine=0):
    scalars=torch.zeros(length,16);scalars[:,0]=.5;scalars[:,3]=1
    context=torch.tensor([.1,length/64,max(0,length-8)/64,refine/3,.5,.5,.125,1.])
    o=Observation(uid,qid,round_id,torch.ones(length,2,dtype=torch.long),
        torch.randn(length,3,4).half(),torch.zeros(length,3).half(),scalars,context,y,
        prefix_ids=torch.tensor([1,2]),topk_ids=torch.ones(length,3,dtype=torch.long))
    o.teacher_features=torch.randn(length,35);o.teacher_features[:,1]=.5
    o.teacher_margin=-torch.ones(length);o.verifier_history=torch.empty(0,48)
    return o


class V2Tests(unittest.TestCase):
    def test_pre_cannot_read_current_verifier_labels(self):
        model=BehavioralWorldModelV2(4,3,dropout=0).eval();o=observation()
        b=pack_v2([o],{},'cpu');z=model.pre(b).z.detach()
        for key in ('labels','teacher_features','teacher_margin','current_verifier','teacher_actual'):
            b[key]=torch.zeros_like(b[key])+123
        self.assertTrue(torch.equal(z,model.pre(b).z.detach()))

    def test_shadow_does_not_enter_prior_memory(self):
        a=observation('a',round_id=0);a.teacher_is_actual=True
        shadow=observation('shadow',round_id=0);shadow.teacher_is_actual=False
        b=observation('b',round_id=1);b.verifier_history=torch.zeros(1,48)
        memory=causal_memory({o.uid:o for o in (a,shadow,b)})
        self.assertEqual(memory['a'].shape,(0,40));self.assertEqual(memory['b'].shape,(1,40))

    def test_behavior_gt_censors_local_suffix_but_labels_prefix_survival(self):
        b=pack_v2([observation(y=2)],{},'cpu');t=behavioral_targets(b)
        self.assertEqual(t['clean'].sum().item(),3)
        self.assertEqual(t['survival'][0].tolist(),[1,1,0,0,0,0,0,0])
        model=BehavioralWorldModelV2(4,3,dropout=0).eval();heads=model.heads(model.pre(b))
        a,_=behavioral_loss(heads,b);b['teacher_features'][:,3:]=999;b['teacher_margin'][:,3:]=-999
        c,_=behavioral_loss(heads,b);self.assertTrue(torch.equal(a,c))

    def test_residual_refine_is_identity_and_extend_is_structural(self):
        s=StateV2(torch.randn(2,128),torch.tensor([[.125,0,.5,0,0,0],[.5,0,.5,.375,.375,0]]))
        dynamics=FixedDynamicsV2();n,_,_=dynamics(s,torch.zeros(2,dtype=torch.long))
        self.assertTrue(torch.equal(s.z,n.z));self.assertEqual(n.lengths.tolist(),[8,32])
        e,_,_=dynamics(s,torch.ones(2,dtype=torch.long));self.assertEqual(e.lengths.tolist(),[16,40])
        self.assertEqual(e.structure[:,3].tolist(),s.structure[:,0].tolist())

    def test_all_past_starts_supervise_same_endpoint(self):
        obs={f's{i}':observation(f's{i}',length=(8 if i<2 else 16),refine=(1 if i in (1,3) else 0)) for i in range(4)}
        paths=enumerate_paths(obs,[('s0','s1','R'),('s1','s2','E'),('s2','s3','R')],['q'])
        ending=[p for p in paths if p[0][-1]=='s3']
        self.assertEqual(sorted(len(p[1]) for p in ending),[1,2,3])

    def test_rollout_does_not_reset_intermediate_predictions(self):
        class AddOne(nn.Module):
            def __init__(self):super().__init__();self.delta=nn.Parameter(torch.tensor(1.));self.seen=[]
            def forward(self,s,actions):
                self.seen.append(s.z.detach().clone());z=s.z+self.delta
                return StateV2(z,s.structure),self.delta.expand(len(z)),z-s.z
        obs={str(i):observation(str(i)) for i in range(4)}
        cache={str(i):dict(pre=torch.full((128,),float(i*10)),post=torch.full((128,),float(i*10)),
                          structure=pack_v2([obs[str(i)]],{},'cpu')['structure'][0]) for i in range(4)}
        model=BehavioralWorldModelV2(4,3,dropout=0);model.freeze_representation();d=AddOne()
        loss,_=rollout_loss(model,d,cache,obs,[(('0','1','2','3'),(0,0,0))],'cpu')
        self.assertEqual([float(x[0,0]) for x in d.seen],[0,1,2]);loss.backward()
        self.assertIsNotNone(d.delta.grad)

    def test_teacher_argmax_is_vocab_id_not_support_index(self):
        ids=torch.tensor([[91,4,72],[8,44,2]]);logits=torch.tensor([[1.,4.,0.],[8.,2.,1.]])
        self.assertEqual(teacher_token_ids(ids,logits).tolist(),[4,8])

    def test_zero_film_and_regularization_keep_gradients(self):
        adapter=GatedFiLMV2(4);h=torch.randn(1,5,4);z=torch.randn(1,128)
        out=adapter(h,z,torch.ones(1,5),torch.ones(1,5),torch.tensor([[1,1,0,0,0]]))
        self.assertTrue(torch.equal(h,out));self.assertTrue(adapter.regularization.requires_grad)
        out.sum().backward();self.assertGreater(float(adapter.projection.weight.grad.abs().sum()),0)

    def test_late_tail_gradients_use_frozen_native_block(self):
        class Layer(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,4)
            def forward(self,h,**kwargs):return h+self.linear(h)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([Layer()])
                self.model.norm=nn.LayerNorm(4);self.lm_head=nn.Linear(4,10)
        model=Tiny().eval().requires_grad_(False);original=copy.deepcopy(model.state_dict())
        adapter=GatedFiLMV2(4)
        frame=dict(hidden=torch.randn(1,5,4),positions=torch.arange(5),active=torch.tensor([0,0,1,1,1]).bool(),
            proposal_start=2,proposal_length=3,selected=torch.tensor([2,3]),kwargs={},kv=None)
        with torch.no_grad():frame['head_hidden']=model.model.norm(model.model.layers[-1](frame['hidden']))
        base=replay_tail(model,frame,None,None,'late','cpu')
        filmed=replay_tail(model,frame,adapter,torch.randn(1,128),'late','cpu')
        self.assertTrue(torch.equal(base,filmed));filmed.sum().backward()
        self.assertGreater(float(adapter.projection.weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()))

    def test_native_capture_counts_direct_forward_and_resets_on_retry(self):
        class Layer(nn.Module):
            def forward(self,h,**kwargs):return h+1
        class Inner(nn.Module):
            def __init__(self):
                super().__init__();self.layers=nn.ModuleList([Layer()]);self.norm=nn.Identity()
            def forward(self,h,positions):
                return self.norm(self.layers[0](h,cache_position=positions,update_past_key_values=False))
        class Outer(nn.Module):
            def __init__(self):super().__init__();self.model=Inner()
            def forward(self,h,positions):return self.model(h,positions)
        m=Outer();hook=NativeTailHook(m);hook.capture=True;hook.begin(2,2,3)
        m._wm_v2_reset_capture()
        m.forward(torch.zeros(1,5,4),torch.arange(5))
        self.assertEqual(hook.counter,1);self.assertIn(1,hook.frames)
        m._wm_v2_reset_capture();self.assertEqual(hook.counter,0);self.assertEqual(hook.frames,{})
        hook.close();self.assertFalse(hasattr(m,'_wm_v2_reset_capture'))

    def test_zip_and_extracted_inputs_both_work_without_old_name(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.mkdir();(source/'experience').mkdir()
            for name in ('states.jsonl','edges.jsonl','labels.jsonl','teacher_targets.jsonl'):(source/name).write_text('')
            (source/'config.json').write_text('{}')
            (source/'summary.json').write_text(json.dumps(dict(schema='interactive_acceptance_two_source_v1',status='complete')))
            (source/'experience/shard_0000.npz').write_bytes(b'fixture')
            self.assertEqual(locate_input(source,root),source)
            archive=root/'any_new_name.zip'
            with zipfile.ZipFile(archive,'w') as zf:
                for file in source.rglob('*'):
                    if file.is_file():zf.write(file,'nested/'+file.relative_to(source).as_posix())
            self.assertEqual(locate_input(archive,root),root/'two_source_input/nested')

    def test_training_and_five_fold_packaging_cpu_smoke(self):
        observations={};edges=[]
        for q in range(5):
            for i in range(4):
                uid=f'q{q}s{i}';observations[uid]=observation(uid,f'q{q}',8 if i<2 else 16,2+i,refine=i%2)
                if i:edges.append((f'q{q}s{i-1}',uid,'E' if i==2 else 'R'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.json').write_text('{}')
            args=parser().parse_args(['--run_dir',str(root),'--output_dir',str(root/'out'),
                '--num_questions','5','--device','cpu','--variants','full','--film_steps','0',
                '--updates_per_question','1','--dynamics_updates','1','--eval_every','1',
                '--batch_size','2','--milestones','1','2','3'])
            with patch('run_persistent_world_model_v2.load_experiences',return_value=(dict(dataset='gsm8k'),observations,edges,[f'q{q}' for q in range(5)])):
                run(args)
            with zipfile.ZipFile(root/'out.zip') as zf:
                self.assertIsNone(zf.testzip());summary=json.loads(zf.read('summary.json'))
                self.assertEqual(summary['status'],'complete');self.assertEqual(summary['completed_folds'],5)
                self.assertIn('fold_0/full_current_predictions.jsonl',zf.namelist())


if __name__=='__main__':unittest.main()
