import tempfile
import unittest
from pathlib import Path

import pandas as pd

from capacity_audit import (
    action_regret_ms,
    assign_native_oracle_label,
    local_regret_capture,
)
from run_strict_greedy_math50 import build_verifier_profile
from strict_greedy_oracle import predict_verifier_latency_ms


class CapacityAuditTests(unittest.TestCase):
    def test_native_labels_use_inclusive_one_ms_tie(self):
        self.assertEqual(assign_native_oracle_label(1.0), "tie")
        self.assertEqual(assign_native_oracle_label(-1.0), "tie")
        self.assertEqual(assign_native_oracle_label(1.001), "stop")
        self.assertEqual(assign_native_oracle_label(-1.001), "continue")

    def test_regret_and_capture_use_continue_as_failfast_action(self):
        self.assertEqual(action_regret_ms(3.0, "continue"), 3.0)
        self.assertEqual(action_regret_ms(3.0, "stop"), 0.0)
        self.assertAlmostEqual(local_regret_capture(1.0, 4.0), 0.75)

    def test_frozen_profile_uses_nearest_context_proposal_bucket(self):
        profile = {
            "mean_verify_latency_ms": 20.0,
            "mean_tokens_per_verify": 4.0,
            "context_bucket_size": 256,
            "proposal_bucket_size": 8,
            "latency_bins": [
                {
                    "context_bucket": 1,
                    "proposal_bucket": 1,
                    "mean_verify_latency_ms": 11.0,
                    "observations": 4,
                },
                {
                    "context_bucket": 3,
                    "proposal_bucket": 2,
                    "mean_verify_latency_ms": 17.0,
                    "observations": 8,
                },
            ],
        }
        self.assertEqual(predict_verifier_latency_ms(profile, 800, 14), 17.0)

    def test_profile_builder_freezes_binned_prepass_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({
                "context_length": [300, 310, 800],
                "proposal_length": [8, 8, 16],
                "verify_latency_ms": [10.0, 14.0, 20.0],
                "emitted_tokens": [4, 5, 7],
            }).to_csv(root / "verifier_calls.csv", index=False)
            profile = build_verifier_profile(root, root / "profile.json")
        self.assertEqual(len(profile["latency_bins"]), 2)
        first = profile["latency_bins"][0]
        self.assertEqual(first["observations"], 2)
        self.assertEqual(first["mean_verify_latency_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()
