import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import run_math_stable_crich50_pipeline as runner
from adaptive_td import AdaptiveTDConfig, _OnlineWeightedLogistic


class StableCrichTest(unittest.TestCase):
    def args(self, *extra):
        with patch('sys.argv', ['runner', *extra]):
            return runner.parse_args()

    def test_range_and_seed_guards(self):
        args = self.args()
        self.assertEqual((args.screen_start_id, args.screen_end_id), (0,500))
        with self.assertRaises(SystemExit):
            self.args('--screen_start_id','500')
        with self.assertRaises(SystemExit):
            self.args('--confirm_seeds','42','42')

    def test_collector_zero_lr_keeps_weights(self):
        AdaptiveTDConfig(hindsight_logistic_learning_rate=0.)
        model = _OnlineWeightedLogistic(3, 0.)
        before = list(model.weights)
        model.update_batch([([1.,.5,.2],1,20.),([1.,.1,.8],0,1.)])
        self.assertEqual(model.weights, before)

    def test_command_semantics(self):
        args = self.args()
        for method in [runner.METHOD_ALWAYS,runner.METHOD_U1,runner.METHOD_SCREEN]:
            cmd = runner.command_for_method(args,method,[1,2],Path('out'),42)
            self.assertNotIn('--verifier_kv_cache', cmd)
            self.assertIn('--no-adaptive-hindsight-logistic-balance-utility-mass',cmd)
            if method == runner.METHOD_SCREEN:
                self.assertEqual(cmd[cmd.index('--adaptive-hindsight-logistic-learning-rate')+1],'0.0')
                self.assertEqual(cmd[cmd.index('--adaptive-hindsight-delta-j-floor-probe')+1],'1.0')

    def test_completion_preserves_order_and_ignores_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); case = root/'batch_0001_0002'; case.mkdir()
            pd.DataFrame({'mode':['dllm_ar']*2,'problem_id':[1,2]}).to_csv(case/'benchmark_results.csv',index=False)
            self.assertTrue(runner.case_complete(case,[1,2]))
            self.assertFalse(runner.case_complete(case,[2,1]))
            self.assertEqual(runner.screened_ids_from_dirs(root), set())
            runner.json_dump(case/'SCREEN_COMPLETE.json',{'ids':[1,2]})
            self.assertEqual(runner.screened_ids_from_dirs(root), {1,2})
