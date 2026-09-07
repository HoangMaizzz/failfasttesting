import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from oracle_probe_schedule import ProbeTape
from collect_fp16_nokv_discovery import cmd_for, read_pool_ids
from run_deterministic_witness import u1_cmd, complete
from run_oracle_total_benefit_control import actual_pass
from test_math50_witness_online import WitnessTest


class PoolTest(unittest.TestCase):
    def test_fp16_two_gpu_commands(self):
        a=SimpleNamespace(dllm_dir='draft',target_model_name='target',target_device=0,
            drafter_device=1,target_quantization='none',target_dtype='fp16',drafter_dtype='fp16',
            two_gpu=True,seed=42,max_new_tokens=1024,drafter_threshold=.5,
            lowconf_threshold=.7,log_level='INFO',probe_schedule_csv='tape.csv')
        for cmd in (cmd_for(a,[1,2],Path('out')), u1_cmd(a,[1,2],Path('out'))):
            self.assertIn('--target_two_gpu_fp16',cmd)
            self.assertIn('--disable_reusing_drafter_kvs',cmd)
            self.assertNotIn('--verifier_kv_cache',cmd)
            self.assertEqual(cmd[cmd.index('--unquantized_dtype')+1],'float16')
            self.assertEqual(cmd[cmd.index('--target_quantization')+1],'none')
            self.assertEqual(cmd[cmd.index('--target_model_label')+1],'Qwen2.5-7B-Instruct')

    def test_pool_and_completion(self):
        a=SimpleNamespace(pool_size=180,candidate_id_min=1,candidate_id_max=499,
                          pool_seed=123,pool_ids_file=None)
        self.assertEqual(read_pool_ids(a),read_pool_ids(a))
        self.assertEqual(len(set(read_pool_ids(a))),180)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            pd.DataFrame({'mode':['dllm_ar']*2,'problem_id':[2,1]}).to_csv(p/'benchmark_results.csv',index=False)
            self.assertTrue(complete(p,[2,1]))
            self.assertFalse(complete(p,[1,2]))

    def test_int8_single_gpu_commands(self):
        a=SimpleNamespace(dllm_dir='draft',target_model_name='target',target_device=0,
            drafter_device=0,target_quantization='int8',target_dtype='fp16',drafter_dtype='fp16',
            two_gpu=False,seed=42,max_new_tokens=1024,drafter_threshold=.5,
            lowconf_threshold=.7,log_level='INFO',probe_schedule_csv='tape.csv')
        for cmd in (cmd_for(a,[1,2],Path('out')),u1_cmd(a,[1,2],Path('out'))):
            self.assertEqual(cmd[cmd.index('--target_quantization')+1],'int8')
            self.assertEqual(cmd[cmd.index('--drafter_device')+1],'0')
            self.assertIn('--disable_reusing_drafter_kvs',cmd)
            self.assertNotIn('--target_two_gpu_fp16',cmd)
            self.assertNotIn('--verifier_kv_cache',cmd)
        a.two_gpu=True
        with self.assertRaises(ValueError):u1_cmd(a,[1,2],Path('out'))

    def test_tape_never_overrides_learned(self):
        t=WitnessTest()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'schedule.csv'
            path.write_text('problem_id,decision_ordinal,probe,mask,pos,action\n299,0,1,0.5,0,continue\n')
            item=t.controller()
            item.hindsight_probe_tape=ProbeTape(path,Path(td)/'trace.jsonl')
            decision=t.decide(item)
            self.assertEqual(decision.diagnostics['action_source'],'learned_continue')
            trace=pd.read_json(Path(td)/'trace.jsonl',lines=True)
            self.assertFalse(bool(trace.probe_selected.iloc[0]))
            item=t.controller(probe=.08)
            item.hindsight_logistic_model.weights[0]=-10
            item.hindsight_current_snapshot['continue_score']=.001
            item.hindsight_current_snapshot['logistic_logit']=-10
            item.hindsight_probe_tape=ProbeTape(path,Path(td)/'trace2.jsonl')
            self.assertEqual(t.decide(item).diagnostics['action_source'],'structural_probe')

    def test_empty_tape_disables_random(self):
        t=WitnessTest()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'empty.csv'
            path.write_text('problem_id,decision_ordinal,probe,mask,pos,action\n')
            item=t.controller(probe=.08)
            item.hindsight_logistic_model.weights[0]=-10
            item.hindsight_current_snapshot['continue_score']=.001
            item.hindsight_current_snapshot['logistic_logit']=-10
            item.hindsight_probe_tape=ProbeTape(path,Path(td)/'trace.jsonl')
            self.assertEqual(t.decide(item).action,'stop')

    def test_fail_on_schedule_mismatch(self):
        a=SimpleNamespace(min_total_learned_benefit=1,min_learned_c=5)
        s={'sum_delta_j_learned_continue':-10,'learned_continue_non_tie':6,
           'max_continue_score':.8,'schedule_state_mismatch_count':1}
        self.assertFalse(actual_pass(a,s)[-1])
        s['schedule_state_mismatch_count']=0
        self.assertTrue(actual_pass(a,s)[-1])


if __name__=='__main__':unittest.main()
