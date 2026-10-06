"""CPU-only contract tests; no LLM/embedding downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from behavior_aware_losses import stage_a_targets, log_survival, behavior_losses, objective
from behavior_aware_source import split_check, weight_hash
from run_behavior_aware_h1 import (BatchSampler, select_checkpoint, study_fingerprint,
    validation_score, frozen_modules, oracle_predictions, worker, make_job)
from phase0_wm_models import (VerifierReadout, NativeReconstruction, DynamicsPair,
                             verifier_loss, valid_positions, masked_mean)
from run_latent_wm_phase0 import seed_all, cpu_weights, save_torch


def synthetic_payload():
    seed_all(14);width=6
    rows={};cache={};paths=[]
    for question in ('train','val'):
        for action in (0,1):
            names=[]
            for child in (0,1):
                uid=f'{question}_{action}_{child}';n=8+child*action*8
                c=torch.zeros(n,20);c[:,0]=1;c[:,12:14]=1
                context=torch.tensor([.3,n/64,8/64 if child and action else 0.,(1+child)/3,.5,.5,.125,1.])
                ids=torch.zeros(n,2,dtype=torch.long)
                rows[uid]=dict(uid=uid,question=question,length=n,c=c,context=context,
                    accepted=8 if not child else n-2,ids=ids,
                    native_target=torch.randn(n,width),native_mask=torch.ones(n,width,dtype=torch.bool))
                cache[uid]=torch.randn(64,128);names.append(uid)
            paths.append((names,[action]))
    g=VerifierReadout(128,1,.05);recon=NativeReconstruction(128,width)
    payload=dict(source_config=dict(dropout=.05,dynamics_layers=2),rows=rows,cachez=cache,
        frozen=dict(readout=cpu_weights(g),reconstruction=cpu_weights(recon)),
        split=dict(train=['train'],val=['val'],test=['test']),
        paths={s:[p for p in paths if rows[p[0][0]]['question']==s] for s in ('train','val')})
    payload['oracle']=oracle_predictions(payload,'cpu',4)
    return payload


class LossTests(unittest.TestCase):
    def test_targets_identical_to_stage_a_and_boundary_cases(self):
        rows=[dict(length=8,accepted=k,teacher=None) for k in list(range(9))+[None]]
        h=torch.randn(len(rows),64,requires_grad=True)
        heads=dict(hazard=h,tf=torch.zeros_like(h),probability=torch.zeros_like(h),margin=torch.zeros_like(h))
        t=stage_a_targets(rows,'cpu')
        exact=masked_mean(F.binary_cross_entropy_with_logits(h,t['survival'],reduction='none'),t['risk'])
        exact+=.5*masked_mean((log_survival(heads).exp()-t['survival']).square(),t['observed'])
        self.assertTrue(torch.equal(exact,verifier_loss(heads,rows,'cpu')))
        self.assertEqual(int(t['survival'][0].sum()),0)
        self.assertEqual(int(t['risk'][0].sum()),1)
        self.assertEqual(int(t['survival'][8].sum()),8)
        self.assertFalse(bool(t['observed'][-1].any()))

    def test_frozen_readout_preserves_imagined_input_gradients(self):
        g=VerifierReadout(128,1,0).eval().requires_grad_(False)
        z=torch.randn(2,64,128,requires_grad=True);n=torch.tensor([8,16])
        rows=[dict(length=8,accepted=3),dict(length=16,accepted=10)]
        oracle=log_survival(g(z.detach()+.2,n)).exp().detach()
        q,k=behavior_losses(g(z,n),rows,rows,'B1',oracle)
        (q+k).backward()
        self.assertGreater(float(z.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in g.parameters()))
        with self.assertRaises(ValueError):behavior_losses(g(z,n),rows,rows,'B1',oracle.requires_grad_())

    def test_tiny_survival_gradient_not_clamped_away(self):
        h=torch.full((1,64),-10.,requires_grad=True)
        rows=[dict(length=64,accepted=64)]
        q,k=behavior_losses(dict(hazard=h),rows,rows,'B2')
        self.assertTrue(bool(torch.isfinite(q+k)))
        q.backward();self.assertGreater(float(h.grad[0,-1].abs()),0)

    def test_near_one_survival_can_learn_rejection(self):
        h=torch.full((1,64),20.,requires_grad=True)
        rows=[dict(length=8,accepted=0)]
        q,_=behavior_losses(dict(hazard=h),rows,rows,'B2')
        self.assertTrue(bool(torch.isfinite(q)))
        q.backward();self.assertGreater(float(h.grad[0,0].abs()),0)

    def test_missing_labels_and_parent_truth_loss_only(self):
        h=torch.randn(2,64,requires_grad=True)
        rows=[dict(length=8,accepted=None),dict(length=8,accepted=4)]
        parents=[dict(accepted=3),dict(accepted=None)]
        q,k=behavior_losses(dict(hazard=h),rows,parents,'B2')
        self.assertEqual(float(k.detach()),0.)
        q.backward();self.assertEqual(float(h.grad[0].abs().sum()),0.)
        self.assertGreater(float(h.grad[1].abs().sum()),0.)

    def test_zero_lambda_is_identical_A_objective(self):
        state=torch.tensor(2.,requires_grad=True)
        for mode in ('A','B1','B2'):
            loss,q,k=objective(state,None,[],[],mode,0,0)
            self.assertIs(loss,state);self.assertIsNone(q);self.assertIsNone(k)


class SelectionTests(unittest.TestCase):
    def test_gate_uses_A_minimum_no_relaxed_fallback(self):
        curve=[dict(step=100,validation_score=3.,validation_state_loss=.8),
               dict(step=200,validation_score=1.,validation_state_loss=1.1)]
        best=select_checkpoint(curve,'B1',.75,1.15)
        self.assertEqual(best['step'],100);self.assertEqual(best['status'],'selected')
        diagnostic=select_checkpoint(curve,'B2',.1,1.15)
        self.assertEqual(diagnostic['status'],'diagnostic_only')
        self.assertFalse(diagnostic['state_fidelity_passed'])

    def test_validation_missing_challenge_is_not_zero_error(self):
        score,_=validation_score([]);self.assertIsNone(score)

    def test_resume_fingerprint_ignores_mount_path_not_content(self):
        a=dict(source='/old.zip',fingerprint='content',artifact_hashes={'x':'1'})
        b=dict(a,source='/new/mount/folder')
        self.assertEqual(study_fingerprint({},a,{}),study_fingerprint({},b,{}))
        b['fingerprint']='different'
        self.assertNotEqual(study_fingerprint({},a,{}),study_fingerprint({},b,{}))

    def test_split_overlap_rejected(self):
        split=dict(train=list(range(70)),val=list(range(70,85)),test=list(range(85,100)))
        split_check(split);split['test'][0]=0
        with self.assertRaises(ValueError):split_check(split)


class WorkerTests(unittest.TestCase):
    def test_sampler_and_zero_lambda_and_exact_resume(self):
        payload=synthetic_payload()
        cfg=dict(learning_rate=.0003,batch_size=2,gradient_diagnostic_updates=2,
            eval_every=2,state_fidelity_ratio=1.15,max_updates=4,pilot_updates=2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'payload.pt';save_torch(data,payload)
            def run(method,name,updates=4,lam=(0,0),reference=None):
                job=make_job(method,42,root/name,cfg,data,lam=lam,reference=reference)
                job.update(device='cpu',updates=updates)
                path=root/(name+'.json');path.write_text(json.dumps(job));return worker(path)
            a=run('A','A');b=run('B1','B1zero',reference=a['min_validation_state_loss'])
            self.assertEqual(a['initial_hash'],b['initial_hash'])
            self.assertEqual(a['batch_digest'],b['batch_digest'])
            wa=torch.load(root/'A/resume.pt',weights_only=True)['model']
            wb=torch.load(root/'B1zero/resume.pt',weights_only=True)['model']
            self.assertEqual(weight_hash(wa),weight_hash(wb))
            run('A','resumed',2);resumed=run('A','resumed',4)
            wr=torch.load(root/'resumed/resume.pt',weights_only=True)['model']
            self.assertEqual(a['batch_digest'],resumed['batch_digest'])
            self.assertEqual(weight_hash(wa),weight_hash(wr))
            c=run('C','C',lam=(1,1))
            self.assertIsNone(c['validation_state_loss'])
            self.assertNotEqual(a['initial_hash'],c['initial_hash'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
