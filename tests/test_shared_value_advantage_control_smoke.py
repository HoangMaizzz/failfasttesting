import unittest
from types import SimpleNamespace

import pandas as pd

from run_shared_value_advantage_control_smoke import (
    CONTROL_ADVANTAGE_LEARNING_RATE,
    CONTROL_VALUE_LEARNING_RATE,
    INDEPENDENT_LEARNING_RATE,
    replay_control_equivalence,
    smoke_command,
)


class SharedValueAdvantageControlSmokeTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            max_new_tokens=1024,
            target_model_name="target",
            dllm_dir="drafter",
            log_level="INFO",
        )

    def test_control_command_uses_eta_value_001_and_eta_advantage_004(self):
        command = smoke_command(
            self.args(),
            "math",
            [2, 6, 42],
            "out",
            "shared_value_advantage",
        )
        joined = " ".join(command)
        self.assertIn("--adaptive-learning-rate 0.02", joined)
        self.assertIn("--adaptive-shared-value-learning-rate 0.01", joined)
        self.assertIn("--adaptive-shared-advantage-learning-rate 0.04", joined)
        self.assertIn(
            "--adaptive-value-parameterization shared_value_advantage",
            joined,
        )

    def test_control_rates_have_expected_effective_head_rates(self):
        selected = (
            CONTROL_VALUE_LEARNING_RATE
            + 0.25 * CONTROL_ADVANTAGE_LEARNING_RATE
        )
        unselected = (
            CONTROL_VALUE_LEARNING_RATE
            - 0.25 * CONTROL_ADVANTAGE_LEARNING_RATE
        )
        self.assertAlmostEqual(selected, INDEPENDENT_LEARNING_RATE)
        self.assertAlmostEqual(unselected, 0.0)

    def test_realistic_same_stream_replay_matches_both_q_heads(self):
        decisions = pd.DataFrame([
            {
                "problem_id": 2,
                "round_id": 0,
                "decision_id": 0,
                "action": "stop",
                "executed_action": "stop",
                "features": "[1.0, 0.25, -0.5, 0.13, 0.8, 0.05]",
                "factual_target": 2.5,
                "importance_weight": 1.7,
                "factual_update_applied": True,
            },
            {
                "problem_id": 2,
                "round_id": 1,
                "decision_id": 0,
                "action": "continue",
                "executed_action": "continue",
                "features": "[1.0, 0.5, 0.2, 0.27, 0.9, 0.08]",
                "factual_target": -1.25,
                "importance_weight": 0.8,
                "factual_update_applied": True,
            },
            {
                "problem_id": 6,
                "round_id": 0,
                "decision_id": 0,
                "action": "stop",
                "executed_action": "stop",
                "features": "[1.0, 0.0, -0.8, 0.13, 1.1, 0.03]",
                "factual_target": 0.4,
                "importance_weight": 2.0,
                "factual_update_applied": True,
            },
        ])
        result = replay_control_equivalence(decisions)
        self.assertEqual(len(result), 3)
        self.assertLessEqual(
            result[
                [
                    "residual_absolute_error",
                    "stop_head_max_absolute_error",
                    "continue_head_max_absolute_error",
                ]
            ].to_numpy().max(),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
