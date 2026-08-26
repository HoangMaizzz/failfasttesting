import unittest
from argparse import Namespace

from run_no_weight_ema_aime_humaneval import (
    DATASET_COUNTS,
    benchmark_command,
    configure_dataset_counts,
    failfast_command,
    validate_args,
)
from run_otrc_v2_td_benchmark import PROBLEM_IDS


class NoWeightEMAAimeHumanEvalTests(unittest.TestCase):
    def args(self):
        return Namespace(
            warmup_questions=1,
            max_new_tokens=1024,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/no_weight_ema",
            resume=True,
            log_level="INFO",
        )

    def test_fixed_problem_ids_have_requested_sizes_and_reserve_warmup(self):
        self.assertEqual(DATASET_COUNTS, {"aime": 29, "humaneval": 29})
        for dataset, count in DATASET_COUNTS.items():
            selected = PROBLEM_IDS[dataset][:count]
            self.assertEqual(len(selected), count)
            self.assertEqual(len(set(selected)), count)
            self.assertNotIn(0, selected)

    def test_command_disables_weight_ema_and_runs_no_baseline(self):
        command = " ".join(
            benchmark_command(self.args(), "humaneval", 29, "/tmp/out")
        )
        self.assertIn("--policy_weight_ema_beta 0.0", command)
        self.assertIn(
            "--credit_assignment verifier_boundary_factual_no_bootstrap",
            command,
        )
        self.assertIn("--feature_schema otrc_v2_2_compact_td", command)
        self.assertNotIn("oracle", command)
        self.assertNotIn("--methods failfast", command)
        self.assertIn("--resume", command)

    def test_failfast_command_uses_identical_ids_and_greedy_decoding(self):
        command = failfast_command(
            self.args(),
            "aime",
            29,
            "/tmp/failfast",
        )
        joined = " ".join(command)
        problem_index = command.index("--problem_ids") + 1
        warmup_index = command.index("--warmup_questions")
        measured_ids = [int(value) for value in command[problem_index:warmup_index]]

        self.assertEqual(measured_ids, PROBLEM_IDS["aime"][:29])
        self.assertIn("--decoding_strategy greedy", joined)
        self.assertIn("--dllm_variant failfast", joined)
        self.assertIn("--spec_len 8", joined)
        self.assertIn("--sweep_incr_len 8", joined)
        self.assertNotIn("--adaptive-td", command)

    def test_benchmark_requires_one_warmup_question(self):
        args = self.args()
        args.warmup_questions = 0
        with self.assertRaisesRegex(ValueError, "one warmup"):
            validate_args(args)

    def test_custom_dataset_counts_support_fifty_math_and_gsm8k(self):
        args = self.args()
        args.datasets = ["math", "gsm8k"]
        args.num_questions = 50

        self.assertEqual(
            configure_dataset_counts(args),
            {"math": 50, "gsm8k": 50},
        )
        self.assertGreaterEqual(len(PROBLEM_IDS["math"]), 50)
        self.assertGreaterEqual(len(PROBLEM_IDS["gsm8k"]), 50)


if __name__ == "__main__":
    unittest.main()
