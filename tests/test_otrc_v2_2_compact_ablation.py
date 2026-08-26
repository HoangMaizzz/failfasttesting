import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from run_otrc_v2_2_compact_ablation import case_command, paired_comparison


class OTRCV22CompactAblationTests(unittest.TestCase):
    def args(self):
        return Namespace(
            datasets=["math", "gsm8k"],
            num_questions=25,
            rho_warmup_boundaries=32,
            warmup_questions=1,
            max_new_tokens=1024,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            resume=False,
            log_level="INFO",
        )

    def test_case_command_changes_only_rho_warmup(self):
        args = self.args()
        compact = case_command(args, Path("/tmp/compact"), 0)
        warmup = case_command(args, Path("/tmp/warmup"), 32)
        compact_text = " ".join(compact)
        warmup_text = " ".join(warmup)

        self.assertIn("--feature_schema otrc_v2_2_compact_td", compact_text)
        self.assertIn(
            "--credit_assignment verifier_boundary_factual_no_bootstrap",
            compact_text,
        )
        self.assertIn("--rho_warmup_boundaries 0", compact_text)
        self.assertIn("--rho_warmup_boundaries 32", warmup_text)

    def test_paired_comparison_uses_identical_problem_ids(self):
        compact = pd.DataFrame({
            "problem_id": [1, 2],
            "actual_algorithm_ms_per_output_token": [10.0, 20.0],
            "actual_algorithm_time": [1.0, 4.0],
            "output_tokens": [100, 200],
            "output_token_hash": ["a", "b"],
        })
        warmup = pd.DataFrame({
            "problem_id": [1, 2],
            "actual_algorithm_ms_per_output_token": [8.0, 25.0],
            "actual_algorithm_time": [0.8, 5.0],
            "output_tokens": [100, 200],
            "output_token_hash": ["a", "b"],
        })

        paired, summary = paired_comparison(
            {"math": compact},
            {"math": warmup},
        )

        self.assertEqual(len(paired), 2)
        self.assertEqual(summary.iloc[0].num_questions, 2)
        self.assertEqual(summary.iloc[0].rho_warmup_win_rate_percent, 50.0)
        self.assertEqual(summary.iloc[0].output_match_rate_percent, 100.0)


if __name__ == "__main__":
    unittest.main()
