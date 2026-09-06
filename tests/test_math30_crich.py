import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import run_math30_crich_u1current_tau05_vs_alwaysstop as runner


class Math30Test(unittest.TestCase):
    def test_frozen_ids_and_command(self):
        ids, meta = runner.load_frozen_ids()
        self.assertEqual(len(set(ids)), 30)
        self.assertEqual(ids[:3], [308, 315, 332])
        self.assertEqual(ids[10], 442)
        self.assertEqual(meta.role.tolist(), ['adaptation'] * 10 + ['evaluation'] * 20)
        args = SimpleNamespace(max_new_tokens=1024, dllm_dir='drafter', target_device=0,
            drafter_device=0, target_quantization='int8', drafter_threshold=.5,
            lowconf_threshold=.7, seed=42, log_level='INFO')
        for method in runner.METHODS:
            cmd = runner.common_command(args, method, Path('output'), ids)
            self.assertNotIn('--verifier_kv_cache', cmd)
            self.assertEqual(cmd[cmd.index('--unquantized_dtype')+1], 'float16')
            if method == 'u1_current_tau05':
                self.assertEqual(cmd[cmd.index('--adaptive-hindsight-logistic-continue-threshold')+1], '0.5')
                self.assertIn('--no-adaptive-hindsight-logistic-balance-utility-mass', cmd)

    def test_mode_filter_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'problem_id':[308,315,999],
                          'mode':['dllm_ar','dllm_ar','verifier_ar']}).to_csv(
                              root/'benchmark_results.csv', index=False)
            self.assertTrue(runner.case_complete(root, [308,315]))
            self.assertFalse(runner.case_complete(root, [308]))

    def test_paired_identical_results(self):
        frame = pd.DataFrame({'problem_id':[1,2], 'output_tokens':[10,20],
            'actual_algorithm_time':[1.,2.], 'output_token_hash':['a','b']})
        result = runner.paired_comparison(frame, frame, [1,2], 20, 42)
        self.assertEqual(result['speedup_vs_always_stop'], 1.)
        self.assertEqual(result['exact_hash_matches'], 2)
