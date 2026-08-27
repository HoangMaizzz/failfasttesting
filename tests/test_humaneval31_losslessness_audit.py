import unittest
from types import SimpleNamespace

import pandas as pd

from run_humaneval31_losslessness_audit import (
    PROBLEM_ID,
    SHARED_HISTORY_IDS,
    classify_divergence,
    shared_command,
)


class HumanEval31LosslessnessAuditTests(unittest.TestCase):
    def test_shared_history_reproduces_online_state_before_problem_31(self):
        self.assertEqual(SHARED_HISTORY_IDS[-1], PROBLEM_ID)
        self.assertEqual(
            SHARED_HISTORY_IDS,
            [1, 3, 4, 19, 20, 21, 23, 26, 27, 29, 31],
        )

    def test_shared_command_audits_only_problem_31(self):
        args = SimpleNamespace(
            max_new_tokens=1024,
            target_model_name="target",
            dllm_dir="drafter",
            log_level="INFO",
        )
        command = shared_command(args, "out")
        problem_index = command.index("--problem_ids") + 1
        warmup_index = command.index("--warmup_questions")
        self.assertEqual(
            command[problem_index:warmup_index],
            [str(value) for value in SHARED_HISTORY_IDS],
        )
        audit_index = command.index("--audit_greedy_problem_ids") + 1
        self.assertEqual(command[audit_index], "31")
        self.assertIn("shared_value_advantage", command)

    def test_classifies_emitted_batched_mismatch_as_implementation_bug(self):
        result = classify_divergence(
            {"first_different_position": 12},
            {
                "failfast": {
                    "emitted_matches_batched": 0,
                    "batched_matches_prefix": 1,
                },
                "shared_value_advantage": {
                    "emitted_matches_batched": 1,
                    "batched_matches_prefix": 1,
                },
            },
            1e-3,
        )
        self.assertEqual(result["classification"], "commit_or_indexing_mismatch")
        self.assertEqual(result["severity"], "implementation_bug")

    def test_classifies_small_margin_batched_prefix_mismatch(self):
        result = classify_divergence(
            {"first_different_position": 12},
            {
                "failfast": {
                    "emitted_matches_batched": 1,
                    "batched_matches_prefix": 0,
                    "batched_margin": 0.0002,
                    "prefix_margin": 0.0001,
                },
            },
            1e-3,
        )
        self.assertEqual(
            result["classification"],
            "near_tie_batched_prefix_mismatch",
        )
        self.assertEqual(result["severity"], "numerical_instability")

    def test_classifies_identical_trace_as_lossless(self):
        result = classify_divergence(
            {"first_different_position": float("nan")},
            {},
            1e-3,
        )
        self.assertEqual(result["classification"], "lossless_match")


if __name__ == "__main__":
    unittest.main()
