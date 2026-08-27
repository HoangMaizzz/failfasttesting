import unittest

from capacity_audit import COMPACT6
from run_compact6_capacity_audit import VERSION
from run_corrected_greedy_oracle_two_datasets import (
    gsm8k_configuration,
    math_configuration,
)


class Compact6CapacityRunnerTests(unittest.TestCase):
    def test_probe_uses_pre_registered_compact6(self):
        self.assertEqual(COMPACT6, (
            "bias",
            "prefix_advance_ratio",
            "failfast_margin",
            "accumulated_spec_ratio",
            "draft_verify_latency_ratio",
            "ema_tokens_per_verifier_ratio",
        ))
        self.assertIn("failfast_support", VERSION)

    def test_thirty_question_ids_are_deterministic(self):
        _, math_ids = math_configuration(30)
        _, gsm_ids = gsm8k_configuration(30, 2026, 50)
        self.assertEqual(len(math_ids), 30)
        self.assertEqual(len(gsm_ids), 30)
        self.assertEqual(len(set(math_ids)), 30)
        self.assertEqual(len(set(gsm_ids)), 30)


if __name__ == "__main__":
    unittest.main()
