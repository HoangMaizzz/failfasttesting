"""Small end-to-end runner checks, no LLM or external dataset needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorized_wm_models import HiddenCompressor, Preprocessor, Layout, StructuredDynamics
from run_factorized_wm_feasibility import (v_rows, experiment_plan, package, train_d,
                                          train_v, apply_operating_points, composition, PriorDynamics)


class FakeV(torch.nn.Module):
    def forward(self, state):
        p = state.x[..., 0] * 0
        return dict(hazard=p, tf=p, probability=p+.25, margin=p)


def config():
    return dict(seed=42,seeds=[42],hidden_dims=[2],representations=['SHT'],scaling_questions=[4],
        capacities={'medium':dict(width=16,layers=1)},batch_size=2,h1_min_updates=1,
        drafter_updates=3,verifier_updates=2,direct_updates=2,eval_every=1,early_patience=2)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def rows(self):
        l=Layout(2); rows={}
        for i,n in enumerate((8,16,16,24)):
            x=torch.zeros(n,l.size);x[:,l.structure][:,0]=1;x[:,l.structure][:,3:5]=1
            context=torch.tensor([.1,n/64,(0 if i==0 else 8 if i==1 else 16)/64,0,.5,.5,.125,1])
            rows[str(i)]=dict(uid=str(i),question='q',x=x,length=n,context=context,prefix=torch.zeros(64),
                token_targets=torch.ones(n,dtype=torch.long),raw_stop_ids=torch.ones(n,dtype=torch.long),
                accepted=min(i+2,n),teacher=None,margin=None)
        return l,rows

    def prep(self,l):
        return Preprocessor(l,[-1,1,151665],torch.randn(3,32),HiddenCompressor(8,3,2))

    def test_uniform_hazards_survival_and_pmf(self):
        _,rows=self.rows();out=v_rows(FakeV(),[rows['0']],'cpu')[0]
        self.assertAlmostEqual(out['expected_yield'],1-2**-8)
        self.assertEqual(out['mode'],0);self.assertIsNone(out['tf_truth'])
        self.assertEqual(len(out['hazards']),8)

    def test_model_forward_never_receives_teacher(self):
        _,rows=self.rows();changed=dict(rows['0'],teacher=torch.zeros(8,35),accepted=7)
        a=v_rows(FakeV(),[rows['0']],'cpu')[0];b=v_rows(FakeV(),[changed],'cpu')[0]
        self.assertEqual(a['hazards'],b['hazards']);self.assertEqual(a['expected_yield'],b['expected_yield'])
        self.assertNotEqual(a['accepted'],b['accepted'])

    def test_plan_is_unique_and_scaling_cannot_touch_test(self):
        cfg=config();plan=experiment_plan(cfg,4)
        self.assertEqual(len(plan),len({p['name'] for p in plan}))
        self.assertEqual({p['kind'] for p in plan},{'D0','D1','D2','V0','V1','V2'})
        cfg['scaling_questions']=[5]
        with self.assertRaises(ValueError):experiment_plan(cfg,4)

    def test_full_grid_three_seeds_and_fixed_scaling(self):
        path=Path(__file__).resolve().parents[1]/'configs/factorized_wm_feasibility.json'
        plan=experiment_plan(json.loads(path.read_text()),70)
        for kind in ('D2','V2'):
            self.assertEqual({j['seed'] for j in plan if j['kind']==kind},{42,43,44})
            self.assertEqual({j['ntrain'] for j in plan if j['kind']==kind},{20,40,60,70})

    def test_package_excludes_raw_cache_and_is_atomic(self):
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'result';out.mkdir();(out/'report.json').write_text('{}')
            cache=out/'_cache';cache.mkdir();(cache/'raw.pt').write_text('raw')
            archive=package(out)
            with zipfile.ZipFile(archive) as z:self.assertEqual(z.namelist(),['result/report.json'])
            self.assertFalse(archive.with_suffix('.zip.tmp').exists())

    def test_threshold_selection_uses_only_validation(self):
        val=[dict(horizon=1,action='R',real_useful=True,learnedD_score=2),
             dict(horizon=1,action='R',real_useful=False,learnedD_score=1)]
        test=[dict(horizon=1,action='R',real_useful=False,learnedD_score=3)]
        result,_=apply_operating_points(val,test,'learnedD','R')
        self.assertEqual(result['test_operating_points']['balanced']['threshold'],2)
        self.assertEqual(result['test_operating_points']['balanced']['precision'],0)

    def test_curriculum_runs_h1_h2_h3_and_checkpoints_are_finite(self):
        l,rows=self.rows();prep=self.prep(l);prior=torch.zeros(8,l.size)
        model=StructuredDynamics(prep,prior,width=16,layers=1,dropout=0)
        paths=[(('0','1'),(1,)),(('0','1','2'),(1,0)),(('0','1','2','3'),(1,0,1))]
        with tempfile.TemporaryDirectory() as td:
            curve=train_d(model,rows,paths,paths,config(),'cpu',Path(td))
            self.assertEqual({r['horizon'] for r in curve},{1,2,3})
            payload=torch.load(Path(td)/'best.pt',weights_only=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in payload['weights'].values()))

    def test_verifier_training_marks_complete_only_after_training(self):
        from factorized_wm_models import VerifierTransformer
        l,rows=self.rows();model=VerifierTransformer(l,width=16,layers=1,dropout=0)
        with tempfile.TemporaryDirectory() as td:
            train_v(model,list(rows.values()),list(rows.values()),config(),'cpu',Path(td))
            self.assertTrue((Path(td)/'training_complete.json').exists())
            self.assertTrue((Path(td)/'best.pt').exists())

    def test_composition_chunks_above_sixteen_without_reusing_first_rows(self):
        from factorized_wm_models import DirectOutcome
        from run_factorized_wm_feasibility import pack
        l,rows=self.rows();prep=self.prep(l);prior=torch.zeros(8,l.size)
        prepared={};paths=[]
        for i in range(32):
            a=dict(rows['0'],uid=f'a{i}',x=rows['0']['x'].clone());a['x'][:,0]=i/20
            b=dict(rows['1'],uid=f'b{i}',x=rows['1']['x'].clone())
            prepared[a['uid']]=a;prepared[b['uid']]=b;paths.append(((a['uid'],b['uid']),(1,)))
        class SensitiveV(FakeV):
            def forward(self,state):
                result=super().forward(state)
                result['hazard']=state.x[:,0,0,None].expand(-1,64)
                return result
        model=PriorDynamics(prep,prior)
        out=composition(model,SensitiveV(),DirectOutcome(l,16),model,prepared,paths,
            dict(config(),batch_size=32),'cpu','cpu','test')
        self.assertEqual(len(out),32)
        self.assertGreater(out[31]['learnedD_K'],out[0]['learnedD_K'])

    def test_missing_hazard_training_raises_clear_error(self):
        from factorized_wm_models import VerifierTransformer
        l,rows=self.rows();missing=[dict(r,accepted=None,teacher=None) for r in rows.values()]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'hazard'):
                train_v(VerifierTransformer(l,width=16,layers=1),missing,missing,config(),'cpu',Path(td))


if __name__=='__main__':unittest.main()
