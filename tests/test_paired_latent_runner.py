"""Interruption/resume equivalence, held-out quarantine and free-rollout isolation."""
from pathlib import Path
import json
import copy
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paired_native_latent as runner
from paired_latent_models import NativeVerifier
from phase0_wm_models import VerifierReadout


def config():
    return dict(schema='paired_native_latent_v1', latent_dim=128, seeds=[42], workers=1,
        devices=['cpu'], oracle_updates=6,max_updates=6,eval_every=2,batch_size=2,
        bootstrap_samples=10,learning_rate=.0003, bridge_state_weight=1.,
        bridge_behavior_weight=.5,distill_weight=.3,dropout=0.,student_layers=1,
        encoder_verification_samples=0,rollout_existing_dynamics=True)


def source():
    torch.manual_seed(1234)
    rows = {}; cache = {}; split = dict(train=['train'],val=['val'],test=['test'])
    for q in split:
        for i in range(2):
            uid=q+str(i)
            rows[uid]=dict(uid=uid,question=q,length=8,accepted=i*4,
                c=torch.randn(8,20),context=torch.randn(8),ids=torch.ones(8,2,dtype=torch.long),
                teacher_hidden=torch.randn(8,32),candidate_embedding=torch.randn(8,64),
                action='R' if i else 'E',parent_uid=None,parent_length=8,
                parent_accepted=8,parent_token_changed=True)
            cache[uid]=torch.randn(64,128)
    return dict(rows=rows,cachez=cache,split=split,
        observation_ids={q:[q+'0',q+'1'] for q in split},paths={},
        source_config=dict(dynamics_layers=1,dropout=0.),
        frozen=dict(readout=runner.cpu_weights(VerifierReadout(128,dropout=0.))))


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_interrupted_training_resumes_same_best_weights_and_batches(self):
        s=source();cfg=config()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            full,record=runner.train_one('Direct',42,s,cfg,root/'full','cpu','same')
            real_save=runner.save_torch
            def stop_after_checkpoint(path,value):
                real_save(path,value)
                if Path(path).name=='last.pt' and value['step']==2:
                    raise RuntimeError('simulated runtime interruption')
            with patch.object(runner,'save_torch',side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    runner.train_one('Direct',42,s,cfg,root/'resume','cpu','same')
            resumed,second=runner.train_one('Direct',42,s,cfg,root/'resume','cpu','same')
            self.assertEqual(record['batch_digest'],second['batch_digest'])
            self.assertEqual(record['selected_step'],second['selected_step'])
            self.assertEqual(record['validation_score'],second['validation_score'])
            for key,value in full.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
            with self.assertRaisesRegex(ValueError,'signature'):
                runner.train_one('Direct',42,s,cfg,root/'resume','cpu','different')
            with self.assertRaisesRegex(ValueError,'signature'):
                runner.train_one('Direct',43,s,cfg,root/'resume','cpu','same')
            best_file=root/'resume'/'best.pt'
            chosen=torch.load(best_file,map_location='cpu',weights_only=True)
            chosen['signature']='other'
            runner.save_torch(best_file,chosen)
            with self.assertRaisesRegex(ValueError,'checkpoint signature'):
                runner.train_one('Direct',42,s,cfg,root/'resume','cpu','same')

    def test_fingerprint_binds_frozen_cache_checkpoint_and_teacher(self):
        s=source()
        s.update(normalization={},provenance=dict(fingerprint='base',
            artifact_hashes={'cache':'original','checkpoint':'original'},
            original_data_digest='raw',teacher_hidden_checksum='hidden'))
        original=runner.study_fingerprint(config(),s)
        for field in ('cache','checkpoint'):
            altered=copy.deepcopy(s)
            altered['provenance']['artifact_hashes'][field]='changed'
            self.assertNotEqual(original,runner.study_fingerprint(config(),altered))
        for field in ('original_data_digest','teacher_hidden_checksum'):
            altered=copy.deepcopy(s)
            altered['provenance'][field]='changed'
            self.assertNotEqual(original,runner.study_fingerprint(config(),altered))

    def test_training_worker_rejects_test_state_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.pt';runner.save_torch(p,source())
            with patch.object(runner,'train_one') as train:
                with self.assertRaisesRegex(ValueError,'Test states'):
                    runner.worker(dict(payload=str(p),config=config(),seed=42,device='cpu',
                                       folder=td,signature='test'))
                train.assert_not_called()

    def test_free_rollout_reads_only_root_D_latent(self):
        s=source();s['dynamics_weights']={}
        s['paths']={'test':[(('test0','test1'),(0,)),(('test0','val0','test1'),(0,0))]}
        s['observation_ids']['test']=['test1']
        s['cachez']['test0'].zero_();s['cachez']['val0'].fill_(999.)
        s['cachez']['test1'].fill_(-999.)
        for row in s['rows'].values():row['question']='test'
        native=NativeVerifier('V_joint',dropout=0).eval()
        captured=[]
        class Recorder(torch.nn.Module):
            def __init__(self,*args):super().__init__()
            def forward(self,state,actions):
                return type(state)(
                    state.z+1,state.lengths,state.c,state.context)
        class Student(torch.nn.Module):
            def forward(self,z,c,context,lengths):
                captured.append(z.detach().clone())
                return dict(hazard=torch.zeros(len(z),64),z_V=torch.zeros_like(z))
        with patch.object(runner,'DynamicsPair',Recorder):
            rows,status=runner.evaluate_rollouts({'V_joint':native,'Direct':Student()},s,config(),'cpu',42)
        self.assertEqual(status['real_intermediate_state_injections'],0)
        self.assertEqual(len(captured),2)
        torch.testing.assert_close(captured[0],torch.ones(1,64,128))
        torch.testing.assert_close(captured[1],torch.full((1,64,128),2.))
        self.assertTrue(any(r['horizon']==2 for r in rows))

    def test_failure_packages_partial_without_input_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/'result';output.mkdir()
            cache=output/'_cache';cache.mkdir();(cache/'raw_large.pt').write_bytes(b'private input')
            cfg=root/'config.json';cfg.write_text(json.dumps(config()))
            args=SimpleNamespace(output=str(output),config=str(cfg),input='bad',phase0_input='bad',resume=False)
            with patch.object(runner,'prepare_paired_source',side_effect=ValueError('expected failure')):
                with self.assertRaisesRegex(ValueError,'expected failure'):runner.run(args)
            with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                self.assertTrue(any(n.endswith('/error.txt') for n in z.namelist()))
                self.assertFalse(any('_cache' in n for n in z.namelist()))
                d=json.loads(z.read('result/summary.json'))
                self.assertEqual(d['status'],'partial')


if __name__=='__main__':
    unittest.main()
