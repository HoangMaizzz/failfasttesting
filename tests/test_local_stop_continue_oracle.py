import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from run_local_stop_continue_oracle import (
    build_verifier_profile,
    common_command,
    selected_problem_ids,
)


class LocalStopContinueOracleTests(unittest.TestCase):
    def test_matched_problem_counts_preserve_prior_prefix(self):
        self.assertEqual(len(selected_problem_ids("math")), 50)
        self.assertEqual(len(selected_problem_ids("gsm8k")), 50)
        self.assertEqual(len(selected_problem_ids("humaneval")), 50)
        self.assertEqual(len(selected_problem_ids("aime")), 25)

    def test_command_is_failfast_not_global_oracle(self):
        args = SimpleNamespace(
            max_new_tokens=1024,
            target_model_name="target",
            dllm_dir="drafter",
            target_device=0,
            drafter_device=0,
            target_quantization="int8",
            drafter_threshold=0.30,
            lowconf_threshold=0.50,
            log_level="INFO",
        )
        command = common_command(args, "math", [2], Path("out"))
        self.assertIn("--log_verifier_calls", command)
        self.assertNotIn("--global_oracle_graph", command)
        self.assertIn("int8", command)

    def test_profile_contains_latency_bins_and_future_call_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame([
                {
                    "problem_id": 1,
                    "context_length": 300,
                    "proposal_length": 8,
                    "emitted_tokens": 5,
                    "verify_latency_ms": 10.0,
                },
                {
                    "problem_id": 1,
                    "context_length": 320,
                    "proposal_length": 8,
                    "emitted_tokens": 7,
                    "verify_latency_ms": 14.0,
                },
            ]).to_csv(root / "verifier_calls.csv", index=False)
            destination = root / "profile.json"
            profile = build_verifier_profile(root, destination)
            self.assertEqual(profile["mean_verify_latency_ms"], 12.0)
            self.assertEqual(profile["mean_tokens_per_verify"], 6.0)
            self.assertEqual(profile["latency_bins"][0]["observations"], 2)
            self.assertEqual(json.loads(destination.read_text())["version"],
                             "frozen_failfast_hardware_profile_v2")


if __name__ == "__main__":
    unittest.main()
