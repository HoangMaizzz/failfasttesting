import unittest
from argparse import Namespace
from pathlib import Path

from run_matched_failfast_baseline import command_for, validate_args


class MatchedFailFastBaselineTests(unittest.TestCase):
    def test_transfer_metrics_are_part_of_benchmark_csv_schema(self):
        expected = {
            "device_transfer_time",
            "device_transfer_ms_per_output_token",
            "actual_e2e_time_excluding_transfer",
            "e2e_ms_per_output_token_excluding_transfer",
        }
        source = (Path(__file__).resolve().parents[1] / "failfast.py").read_text(
            encoding="utf-8"
        )
        schema = source.split("BENCHMARK_CSV_COLUMNS = [", 1)[1].split("]", 1)[0]
        self.assertTrue(all(f'"{field}"' in schema for field in expected))

    def args(self):
        return Namespace(
            datasets=["math", "gsm8k"],
            num_questions=25,
            warmup_questions=1,
            max_new_tokens=1024,
            drafter_threshold=0.30,
            lowconf_threshold=0.50,
            target_device=0,
            drafter_device=1,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/out",
            resume=False,
            skip_archive=False,
            log_level="INFO",
        )

    def test_command_uses_matched_thresholds_ids_and_devices(self):
        args = self.args()
        validate_args(args)
        command = command_for(args, "math", Path("/tmp/out"))
        self.assertEqual(command[command.index("--drafter_thresholds") + 1], "0.3")
        self.assertEqual(command[command.index("--sweep_lowconf_threshold") + 1], "0.5")
        self.assertEqual(command[command.index("--target_device") + 1], "0")
        self.assertEqual(command[command.index("--drafter_device") + 1], "1")
        self.assertNotIn("--adaptive-td", command)


if __name__ == "__main__":
    unittest.main()
