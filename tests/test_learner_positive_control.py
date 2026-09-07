import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import run_math_learner_positive_control as runner


class LearnerPCTest(unittest.TestCase):
    def test_frozen_sequence_and_commands(self):
        self.assertEqual(len(set(runner.FULL_IDS)),50)
        self.assertTrue(all(0 <= i < 500 for i in runner.FULL_IDS))
        with patch('sys.argv',['runner']): a=runner.parse_args()
        for method in ['u1','always_stop','probe_only']:
            cmd=runner.method_cmd(a,method,runner.FULL_IDS,Path('out'),44,.35)
            self.assertNotIn('--verifier_kv_cache',cmd)
            self.assertIn('--no-adaptive-hindsight-logistic-balance-utility-mass',cmd)
            if method=='u1':
                self.assertEqual(cmd[cmd.index('--adaptive-hindsight-logistic-continue-threshold')+1],'0.35')

    def test_summary_empty_feedback_and_tie_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame({'problem_id':[1], 'mode':['dllm_ar'],'output_tokens':[10],
                          'actual_e2e_time_excluding_transfer':[1.]}).to_csv(root/'benchmark_results.csv',index=False)
            self.assertEqual(runner.summary(root,[1])['learned_c'],0)
            tr=pd.DataFrame({'pair_resolved':[True,True],'problem_id':[1,1],
                'binary_label_C':[1,None],'delta_J_ms_per_token':[-2,.8],
                'action_source':['learned_continue']*2,'continue_score_before_update':[.6,.7]})
            tr.to_csv(root/'adaptive_full_stream_transitions.csv',index=False)
            s=runner.summary(root,[1])
            self.assertAlmostEqual(s['sum_delta_j_learned_c'],-1.2)
            self.assertEqual(s['learned_c_ties'],1)
            self.assertEqual(s['learned_c_tp'],1)
