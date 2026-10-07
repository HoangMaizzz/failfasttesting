"""Offline end-to-end wiring with real tiny Qwen; not a 7B quality test."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_native_qwen_verifier as runner
from native_qwen_capture import load_verifier, MODEL_ID, REVISION


class RunnerTests(unittest.TestCase):
    def config(self):
        cfg=json.loads((Path(runner.__file__).parent/'configs/native_qwen_verifier.json').read_text())
        cfg.update(pipeline_check_only=True,workers=1,devices=['cpu'],seeds=[42],
            probe_updates=2,latent_updates=2,eval_every=2,batch_size=4,
            benchmark_repetitions=3,benchmark_warmup=1,bootstrap_samples=10)
        return cfg

    def test_job_count_and_features(self):
        cfg=self.config();depths=[7,14,21,28]
        raw=runner.raw_specs(depths,cfg)
        compact=runner.latent_specs([14,21],cfg)
        self.assertEqual(len(raw),17)
        self.assertEqual(len(compact),8)
        self.assertEqual(3*(len(raw)+len(compact)),75)
        self.assertEqual(len({runner.method_name(s) for s in raw+compact}),25)

    def test_tiny_teacher_is_pipeline_only(self):
        cfg=self.config();cfg['tiny_config']={'hidden_size':16}
        self.assertEqual(runner.verifier_options(cfg)['device'],'cpu')
        cfg['pipeline_check_only']=False
        with self.assertRaisesRegex(ValueError,'tiny random teacher'):runner.config_check(cfg)

    def test_forced_packaging_replaces_same_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'result';output.mkdir()
            report=output/'summary.json';report.write_text('{"iteration":1}')
            runner.archive_partial(output,force=True)
            report.write_text('{"iteration":2}')
            archive=runner.archive_partial(output,force=True)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(json.loads(zipped.read('summary.json')),{'iteration':2})

    def fixture(self):
        identity=dict(model_id=MODEL_ID,revision=REVISION,dtype='float16',quantization='none')
        torch.manual_seed(19)
        model=load_verifier(identity,dict(device='cpu',tiny_config=dict(vocab_size=43,
            hidden_size=16,intermediate_size=32,num_hidden_layers=4,
            num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128)))
        prefix=[1,2,3];sequence=list(prefix)
        with torch.inference_mode():
            for _ in range(16):
                ids=torch.tensor([sequence],dtype=torch.long)
                sequence.append(int(model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].argmax()))
        greedy=sequence[len(prefix):]
        rows={};cachez={}
        for i in range(100):
            question=f'gsm8k:{i}';root=f'q{i:03d}_root'
            for action,n,K,segment in [('root',8,8,0),('R',8,4,0),('E',16,14,8)]:
                uid=root if action=='root' else f'q{i:03d}_{action}'
                candidate=greedy[:n]
                if K<n:candidate[K]=(candidate[K]+1)%43
                rows[uid]=dict(uid=uid,question=question,round_id=0,prefix=list(prefix),
                    candidate=candidate,length=n,accepted=K,action=action,
                    parent_uid=None if action=='root' else root,
                    parent_length=None if action=='root' else 8,parent_K=None if action=='root' else 8,
                    segment_start=segment,c=torch.randn(n,20),context=torch.randn(8),
                    is_parent_full_prefix=action!='root')
                cachez[uid]=torch.randn(n,128)
        source=dict(rows=rows,cachez=cachez,identity=identity,
            split=dict(train=[f'gsm8k:{i}' for i in range(70)],
                       val=[f'gsm8k:{i}' for i in range(70,85)],
                       test=[f'gsm8k:{i}' for i in range(85,100)]),
            provenance=dict(original_data_digest='tiny-source',artifact_hashes={'encoder':'tiny'}))
        return model,source

    @staticmethod
    def inline_stage(specs,source,payload,cfg,output,stage):
        records=[]
        for seed in cfg['seeds']:
            runner.worker(dict(seed=seed,device='cpu',specs=specs,payload=str(payload),
                               config=cfg,output=str(output),stage=stage))
            records+=json.loads((Path(output)/'_cache'/f'completed_{stage}_{seed}.json').read_text())
        return records

    def test_real_tiny_qwen_capture_train_evaluate_package_and_resume(self):
        model,original=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg_path=root/'config.json';cfg_path.write_text(json.dumps(self.config()))
            output=root/'result'
            args=argparse.Namespace(input='fixture',phase0_input='fixture',output=str(output),
                                    config=str(cfg_path),resume=False,capture_input=None)
            with patch.object(runner,'prepare_native_source',side_effect=lambda *a,**kw:copy.deepcopy(original)), \
                 patch.object(runner,'load_verifier',return_value=model), \
                 patch.object(runner,'execute_stage',side_effect=self.inline_stage), \
                 patch.object(runner,'log'),patch('builtins.print'):
                result=runner.run(args)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['fresh_capture_calls'],300)
                self.assertEqual(result['training_jobs'],25)
                self.assertEqual(result['test_prediction_rows'],25*45)
                self.assertTrue(all(v=='PIPELINE_ONLY' for v in result['validation_gate'].values()))
                payload=torch.load(output/'_cache/train_val.pt',map_location='cpu',weights_only=True)
                self.assertFalse(set(payload['rows']) & {u for u,r in original['rows'].items()
                                                        if r['question'] in original['split']['test']})
                self.assertTrue(all(set(v)=={'length'} for v in payload['capture_geometry'].values()))
                with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                    self.assertIsNone(archive.testzip())
                    names=archive.namelist()
                    self.assertEqual(len(names),len(set(names)))
                    self.assertIn('FINAL_REPORT.md',names)
                    self.assertIn('native_hidden/layer100/hidden.npy',names)
                    self.assertFalse(any(n.startswith('_cache/') for n in names))
                before=json.loads((output/'all_per_seed_metrics.json').read_text())
                args.resume=True
                again=runner.run(args)
                self.assertEqual(again['fresh_capture_calls'],0)
                self.assertEqual(before,json.loads((output/'all_per_seed_metrics.json').read_text()))
                with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                    self.assertEqual(json.loads(archive.read('summary.json'))['fresh_capture_calls'],0)

    def test_reproduction_mismatch_refuses_any_probe_training(self):
        model,source=self.fixture()
        for row in source['rows'].values():row['accepted']=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg_path=root/'cfg.json';cfg_path.write_text(json.dumps(self.config()))
            output=root/'bad'
            args=argparse.Namespace(input='fixture',phase0_input='fixture',output=str(output),
                                    config=str(cfg_path),resume=False,capture_input=None)
            with patch.object(runner,'prepare_native_source',return_value=source), \
                 patch.object(runner,'load_verifier',return_value=model), \
                 patch.object(runner,'execute_stage') as train,patch.object(runner,'log'):
                with self.assertRaisesRegex(RuntimeError,'not reproduced'):runner.run(args)
                train.assert_not_called()
                self.assertTrue((output/'error.txt').exists())
                self.assertTrue(output.with_suffix('.zip').exists())
                report=json.loads((output/'verifier_reproduction_check.json').read_text())
                self.assertEqual(report['mismatches'],1)


if __name__=='__main__':unittest.main()
