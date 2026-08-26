import unittest
from argparse import Namespace

from run_otrc_v2_2_weight_ema_benchmark import (
    benchmark_command,
    validate_args,
)


class OTRCV22WeightEMABenchmarkTests(unittest.TestCase):
    def args(self):
        return Namespace(
            datasets=["math", "gsm8k"],
            num_questions=25,
            policy_weight_ema_beta=0.99,
            policy_weight_ema_mode="global_step",
            warmup_questions=1,
            max_new_tokens=1024,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/weight_ema",
            resume=False,
            log_level="INFO",
        )

    def test_command_runs_only_compact_no_bootstrap_weight_ema(self):
        args = self.args()
        command = " ".join(benchmark_command(args))

        self.assertIn("--feature_schema otrc_v2_2_compact_td", command)
        self.assertIn(
            "--credit_assignment verifier_boundary_factual_no_bootstrap",
            command,
        )
        self.assertIn("--rho_warmup_boundaries 0", command)
        self.assertIn("--policy_weight_ema_beta 0.99", command)
        self.assertIn("--policy_weight_ema_mode global_step", command)
        self.assertNotIn("oracle", command)
        self.assertNotIn("failfast_spec", command)

    def test_beta_is_predeclared(self):
        args = self.args()
        args.policy_weight_ema_beta = 0.98
        with self.assertRaisesRegex(ValueError, "beta=0.99"):
            validate_args(args)

    def test_global_step_clock_is_predeclared(self):
        args = self.args()
        args.policy_weight_ema_mode = "action_step"
        with self.assertRaisesRegex(ValueError, "global_step"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
