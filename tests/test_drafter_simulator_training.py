"""Behavioral and leakage checks using actual tiny-model optimization."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drafter_simulator_models import ObservationEncoder,Transition,BehaviorHeads,project_masks
from drafter_simulator_train import (StateStore,fit_encoder,fit_transition,evaluate,
    binary_metrics,train_study,benchmark)
from run_drafter_simulator import validate_config,package


def synthetic(output):
    output=Path(output);(output/'capture').mkdir(exist_ok=True)
    rng=np.random.default_rng(2);states=[]
    for q in range(6):
        n,width=5,32
        mask=np.zeros((n,width),bool)
        for t in range(n):mask[t,t+1:8]=True
        eligible=np.zeros_like(mask);eligible[:,:8]=True
        hidden=rng.normal(size=(n,width,24)).astype(np.float16)
        ids=np.tile(np.arange(width),(n,1))+3
        ids[1:,3]=11
        data=dict(hidden=hidden,token_emb=rng.normal(size=(n,width,64)).astype(np.float16),
            candidate_emb=rng.normal(size=(n,width,64)).astype(np.float16),mask=mask,eligible=eligible,
            confidence=np.where(mask,.3,.8).astype(np.float32),entropy=np.where(mask,.7,.1).astype(np.float32),
            margin=np.where(mask,.1,.7).astype(np.float32),candidate_ids=ids)
        np.savez_compressed(output/'capture'/f'q{q}.npz',**data)
        for t in range(n):states.append(dict(uid=f'{q}_{t}',question_id=f'q{q}',group_id=f'g{q}',
            step=t,forward_id=t+1,npz=f'capture/q{q}.npz',row=t,block_start=32,context_len=41,small_block_index=0))
    split=dict(train=['q0','q1','q2','q3'],validation=['q4'],test=['q5'])
    (output/'capture_manifest.json').write_text(json.dumps(dict(states=states,split=split)))


class TrainingTests(unittest.TestCase):
    def test_core_benchmark_native_comparison_and_model_ownership(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            synthetic(temp);store=StateStore(temp);store.fit_normalization(list(range(20)))
            store.manifest['native_benchmarks']=[dict(device='cpu',native_ms_p50=2.,
                dtype='torch.float16',uid='native_reference')]
            enc=ObservationEncoder(store.features(0).shape[-1],16)
            heads=BehaviorHeads(16,64);model=Transition(16,'mlp')
            result=benchmark(store,enc,heads,model,(),torch.device('cpu'),
                dict(benchmark_warmup=0,benchmark_repetitions=1),0)
            comparison=result['same_device_native_comparison']
            self.assertAlmostEqual(comparison['horizon3_core_over_one_native_forward'],
                result['complete_horizon3']['gpu_or_cpu_ms']['p50']/2.)
            self.assertTrue(enc.training)
            self.assertEqual(next(enc.parameters()).device.type,'cpu')

    def test_semantic_constraints_and_gradient(self):
        prior=torch.tensor([[True,True,False]])
        nextmask=project_masks(torch.tensor([[.9,.8,.99]]),prior,torch.ones_like(prior))
        self.assertEqual(nextmask.tolist(),[[True,False,False]])
        model=Transition(16,'transformer');x=torch.randn(2,8,16,requires_grad=True)
        model(x).square().mean().backward();self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_constant_prediction_ap_is_prevalence(self):
        result=binary_metrics([.5,.5,.5,.5],[1,0,0,0])
        self.assertAlmostEqual(result['average_precision'],.25)

    def test_train_only_stats_and_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            synthetic(temp);store=StateStore(temp)
            store.fit_normalization([0,1,2,3,4]);original=store.mean.copy()
            # Test prompt observations are never consulted by normalization.
            test=Path(temp)/'capture/q5.npz'
            with np.load(test) as f:data={k:f[k] for k in f.files}
            data['hidden'][:]=10000;np.savez_compressed(test,**data)
            store.fit_normalization([0,1,2,3,4]);np.testing.assert_array_equal(original,store.mean)
            paths=store.paths(['q0'],3)
            self.assertEqual(len(paths[0]),4)
            for path in paths:self.assertEqual(len({store.states[i]['question_id'] for i in path}),1)

    def test_frozen_encoder_transition_and_free_rollout(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            synthetic(temp);store=StateStore(temp)
            trainids=list(range(20));valids=list(range(20,25));store.fit_normalization(trainids)
            cfg=dict(latent_dim=16,encoder_updates=3,batch_size=2,eval_every=1,updates=3,evaluation_paths=8)
            enc,heads,_=fit_encoder(store,trainids,valids,cfg,42,(),torch.device('cpu'))
            before={k:v.clone() for k,v in enc.state_dict().items()}
            model,heads,_=fit_transition(store,store.paths(['q0','q1','q2','q3'],3),
                store.paths(['q4'],3),enc,heads,cfg,42,'mlp',3,(),torch.device('cpu'),temp,'test')
            for key,value in enc.state_dict().items():torch.testing.assert_close(value,before[key])
            result=evaluate(store,store.paths(['q5'],3),enc,heads,model,(),torch.device('cpu'),3)
            self.assertEqual(set(result),{'1','2','3'})
            self.assertEqual(result['3']['paths'],2)
            self.assertTrue(np.isfinite(result['1']['macro']['commit_brier']))

    def test_complete_tiny_study_packages_reports(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run';output.mkdir();synthetic(output)
            result=train_study(dict(latent_dim=16,encoder_updates=2,updates=2,eval_every=1,
                batch_size=2,seeds=[42],variants=['transformer_h3'],learning_milestones=[2],
                train_device='cpu',benchmark_warmup=1,benchmark_repetitions=2),output)
            self.assertEqual(result['states'],30)
            self.assertTrue((output/'learning_curve.json').is_file())
            self.assertTrue((output/'diagnosis.json').is_file())
            archive=package(output);self.assertTrue(archive.is_file())

    def test_protocol_rejects_empty_splits_or_changed_native_config(self):
        with self.assertRaises(ValueError):validate_config(dict(num_questions=2))
        with self.assertRaises(ValueError):validate_config(dict(small_block_size=16))


if __name__=='__main__':unittest.main()
