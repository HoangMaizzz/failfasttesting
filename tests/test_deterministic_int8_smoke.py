import unittest
from pathlib import Path
from types import SimpleNamespace

from run_deterministic_int8_gsm8k_smoke import PROBLEM_IDS_SMOKE, command_args


class DeterministicInt8SmokeTests(unittest.TestCase):
    def test_smoke_uses_five_fixed_gsm8k_ids(self):
        self.assertEqual(len(PROBLEM_IDS_SMOKE), 5)
        self.assertEqual(len(set(PROBLEM_IDS_SMOKE)), 5)

    def test_command_selects_deterministic_int8(self):
        args = SimpleNamespace(
            dllm_dir="/tmp/dllm", target_device=0, drafter_device=0,
            log_level="INFO",
        )
        self.assertEqual(command_args(args).target_quantization, "int8_deterministic")

    def test_deterministic_int8_load_config(self):
        source = (Path(__file__).resolve().parents[1] / "failfast.py").read_text(
            encoding="utf-8",
        )
        self.assertIn('"eager" if deterministic_int8 else "sdpa"', source)
        self.assertIn("llm_int8_threshold=0.0 if deterministic_int8 else 6.0", source)


if __name__ == "__main__":
    unittest.main()
