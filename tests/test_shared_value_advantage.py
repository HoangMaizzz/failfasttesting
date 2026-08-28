import unittest
from argparse import Namespace

from adaptive_td import (
    CONTINUE,
    STOP,
    AdaptiveTDConfig,
    OnlineTDRefinementController,
)
from run_shared_value_advantage_benchmark import (
    ADVANTAGE_LEARNING_RATE,
    DATASETS,
    VALUE_LEARNING_RATE,
    benchmark_command,
    failfast_baseline_command,
    validate_args,
)
from run_otrc_v2_td_benchmark import method_name


def shared_config(**overrides):
    values = {
        "feature_dim": 6,
        "feature_schema": "otrc_v2_2_compact_td",
        "feature_version": 226,
        "credit_assignment": "verifier_boundary_factual_no_bootstrap",
        "value_parameterization": "shared_value_advantage",
        "shared_value_learning_rate": 0.015,
        "shared_advantage_learning_rate": 0.02,
        "policy_mode": "symmetric",
    }
    values.update(overrides)
    return AdaptiveTDConfig(**values)


class SharedValueAdvantageTests(unittest.TestCase):
    def args(self):
        return Namespace(
            datasets=list(DATASETS),
            num_questions=25,
            warmup_questions=1,
            max_new_tokens=1024,
            drafter_threshold=0.05,
            lowconf_threshold=0.45,
            target_device=0,
            drafter_device=0,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/shared_value_advantage",
            resume=True,
            include_failfast_baseline=False,
            skip_archive=False,
            log_level="INFO",
        )

    def test_coupled_update_preserves_selected_effective_learning_rate(self):
        features = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        stop_controller = OnlineTDRefinementController(shared_config())
        residual = stop_controller._update_factual_action_value(
            STOP,
            features,
            1.0,
            observation_weight=1.0,
        )
        self.assertEqual(residual, 1.0)
        self.assertAlmostEqual(stop_controller.shared_value_theta[0], 0.015)
        self.assertAlmostEqual(stop_controller.shared_advantage_theta[0], 0.01)
        self.assertAlmostEqual(stop_controller.values[STOP].theta[0], 0.02)
        self.assertAlmostEqual(stop_controller.values[CONTINUE].theta[0], 0.01)
        self.assertEqual(stop_controller.values[STOP].sample_count, 1)
        self.assertEqual(stop_controller.values[CONTINUE].sample_count, 0)

        continue_controller = OnlineTDRefinementController(shared_config())
        continue_controller._update_factual_action_value(
            CONTINUE,
            features,
            1.0,
            observation_weight=1.0,
        )
        self.assertAlmostEqual(continue_controller.values[CONTINUE].theta[0], 0.02)
        self.assertAlmostEqual(continue_controller.values[STOP].theta[0], 0.01)

    def test_equivalent_control_rates_reproduce_independent_head_update(self):
        controller = OnlineTDRefinementController(shared_config(
            shared_value_learning_rate=0.01,
            shared_advantage_learning_rate=0.04,
        ))
        features = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        controller._update_factual_action_value(
            STOP,
            features,
            1.0,
            observation_weight=1.0,
        )
        self.assertAlmostEqual(controller.values[STOP].theta[0], 0.02)
        self.assertAlmostEqual(controller.values[CONTINUE].theta[0], 0.0)

    def test_shared_snapshot_round_trip_restores_value_and_advantage(self):
        original = OnlineTDRefinementController(shared_config())
        features = (1.0, 0.5, 0.0, 0.0, 0.0, 0.0)
        original._update_factual_action_value(
            STOP,
            features,
            2.0,
            observation_weight=1.0,
        )
        restored = OnlineTDRefinementController(shared_config())
        restored.load_snapshot(original.snapshot())
        self.assertEqual(
            restored.shared_value_theta,
            original.shared_value_theta,
        )
        self.assertEqual(
            restored.shared_advantage_theta,
            original.shared_advantage_theta,
        )
        self.assertAlmostEqual(
            restored.evaluate(STOP, features).mean,
            original.evaluate(STOP, features).mean,
        )

    def test_shared_mode_rejects_bootstrap_and_policy_weight_ema(self):
        with self.assertRaisesRegex(ValueError, "no-bootstrap"):
            AdaptiveTDConfig(value_parameterization="shared_value_advantage")
        with self.assertRaisesRegex(ValueError, "does not support"):
            shared_config(policy_weight_ema_beta=0.9)

    def test_runner_uses_four_datasets_and_matched_configuration(self):
        args = self.args()
        validate_args(args)
        command = benchmark_command(args)
        joined = " ".join(command)
        dataset_start = command.index("--datasets") + 1
        question_index = command.index("--num_questions")
        self.assertEqual(tuple(command[dataset_start:question_index]), DATASETS)
        self.assertIn(
            "--value_parameterization shared_value_advantage",
            joined,
        )
        self.assertIn(
            f"--shared_value_learning_rate {VALUE_LEARNING_RATE}",
            joined,
        )
        self.assertIn(
            f"--shared_advantage_learning_rate {ADVANTAGE_LEARNING_RATE}",
            joined,
        )
        self.assertIn("--policy_weight_ema_beta 0.0", joined)
        self.assertIn("--max_spec_len 64", joined)
        self.assertIn("--resume", command)

    def test_runner_can_limit_the_matched_run_to_math_and_gsm8k(self):
        args = self.args()
        args.datasets = ["math", "gsm8k"]
        validate_args(args)
        command = benchmark_command(args)
        dataset_start = command.index("--datasets") + 1
        question_index = command.index("--num_questions")
        self.assertEqual(
            command[dataset_start:question_index],
            ["math", "gsm8k"],
        )

    def test_runner_forwards_custom_confidence_thresholds(self):
        args = self.args()
        args.drafter_threshold = 0.30
        args.lowconf_threshold = 0.50
        validate_args(args)
        command = benchmark_command(args)
        self.assertEqual(
            command[command.index("--drafter_threshold") + 1],
            "0.3",
        )
        self.assertEqual(
            command[command.index("--lowconf_threshold") + 1],
            "0.5",
        )

    def test_runner_forwards_split_gpu_devices(self):
        args = self.args()
        args.target_device = 0
        args.drafter_device = 1
        command = benchmark_command(args)
        self.assertEqual(command[command.index("--target_device") + 1], "0")
        self.assertEqual(command[command.index("--drafter_device") + 1], "1")

    def test_integrated_failfast_command_is_matched(self):
        args = self.args()
        args.datasets = ["math", "gsm8k"]
        args.include_failfast_baseline = True
        args.drafter_threshold = 0.30
        args.lowconf_threshold = 0.50
        command = failfast_baseline_command(args)
        joined = " ".join(command)
        self.assertIn("run_matched_failfast_baseline.py", command)
        self.assertIn("--datasets math gsm8k", joined)
        self.assertIn("--drafter_threshold 0.3", joined)
        self.assertIn("--lowconf_threshold 0.5", joined)
        self.assertIn("--target_device 0", joined)
        self.assertIn("--drafter_device 0", joined)
        self.assertIn("--skip_archive", command)
        self.assertIn("--resume", command)

    def test_integrated_baseline_rejects_unsupported_dataset(self):
        args = self.args()
        args.datasets = ["aime"]
        args.include_failfast_baseline = True
        with self.assertRaisesRegex(ValueError, "math and gsm8k"):
            validate_args(args)

    def test_runner_rejects_invalid_confidence_thresholds(self):
        args = self.args()
        args.drafter_threshold = 0.0
        with self.assertRaisesRegex(ValueError, "drafter_threshold"):
            validate_args(args)

        args = self.args()
        args.lowconf_threshold = 1.1
        with self.assertRaisesRegex(ValueError, "lowconf_threshold"):
            validate_args(args)

    def test_shared_parameterization_has_distinct_method_name(self):
        args = Namespace(
            credit_assignment="verifier_boundary_factual_no_bootstrap",
            feature_schema="otrc_v2_2_compact_td",
            rho_warmup_boundaries=0,
            policy_weight_ema_beta=0.0,
            policy_weight_ema_mode="global_step",
            value_parameterization="shared_value_advantage",
        )
        self.assertEqual(method_name(args), (
            "otrc_v2_2_compact_factual_no_bootstrap_"
            "shared_value_advantage"
        ))


if __name__ == "__main__":
    unittest.main()
