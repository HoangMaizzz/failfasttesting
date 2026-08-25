import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_strict_greedy_math50 import (
    build_verifier_profile,
    common_command,
    decision_summary,
)


class StrictGreedyMath50Tests(unittest.TestCase):
    def test_common_command_accepts_an_explicit_dataset(self):
        source = {
            "dataset": "gsm8k",
            "max_new_tokens": 1024,
            "block_size": 32,
            "small_block_size": 8,
            "target_model_name": "target",
            "drafter_threshold": 0.05,
            "lowconf_threshold": 0.45,
            "max_spec_len": 60,
            "seed": 42,
        }
        command = common_command(source, [1, 2], "dllm", "output", "INFO")
        dataset_index = command.index("--dataset_name") + 1
        self.assertEqual(command[dataset_index], "gsm8k")

    def test_prepass_profile_uses_real_arithmetic_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame({
                "verify_latency_ms": [100.0, 140.0, 180.0],
                "emitted_tokens": [4, 5, 9],
            }).to_csv(directory / "verifier_calls.csv", index=False)
            path = directory / "profile.json"
            profile = build_verifier_profile(directory, path)
            self.assertAlmostEqual(profile["mean_verify_latency_ms"], 140.0)
            self.assertAlmostEqual(profile["mean_tokens_per_verify"], 6.0)
            self.assertEqual(json.loads(path.read_text())["verifier_calls"], 3)

    def test_predicted_call_change_is_summed_per_run_then_averaged(self):
        rows = pd.DataFrame({
            "phase": ["oracle_1", "oracle_1", "oracle_2", "oracle_2"],
            "sample_id": [2, 2, 2, 2],
            "decision_id": [0, 1, 0, 1],
            "chosen_action": ["stop", "continue", "stop", "continue"],
            "predicted_extra_calls_stop": [0.5, 0.0, 0.7, 0.0],
            "predicted_extra_calls_continue": [0.0, 0.2, 0.0, 0.1],
            "differs_from_baseline": [1, 0, 1, 0],
            "changed_by_verifier_penalty": [0, 1, 0, 1],
            "verify_to_extend_flip": [1, 0, 1, 0],
            "stop_outer_path": ["VERIFY", "EXTEND -> VERIFY"] * 2,
            "continue_outer_path": ["EXTEND -> VERIFY"] * 4,
        })
        summary = decision_summary(rows).iloc[0]
        self.assertAlmostEqual(summary["predicted_net_verifier_call_change"], 0.6)
        self.assertEqual(summary["total_greedy_decisions"], 2)
        self.assertEqual(summary["stop_count"], 1)
        self.assertEqual(summary["verify_to_extend_flips"], 1)


if __name__ == "__main__":
    unittest.main()
