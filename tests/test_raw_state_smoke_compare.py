import unittest
from pathlib import Path

from run_raw_state_smoke_compare5 import DATASETS, METHODS


ROOT = Path(__file__).resolve().parents[1]


class RawStateSmokeCompareTests(unittest.TestCase):
    def test_protocol_has_two_new_models_and_three_reused_controls(self):
        self.assertEqual(DATASETS, ("math", "gsm8k"))
        self.assertEqual(
            METHODS,
            ("failfast", "always_stop", "compact6_annealed", "raw_linear", "raw_mlp"),
        )

    def test_runner_uses_aligned_schema_and_five_questions(self):
        source = (ROOT / "run_raw_state_smoke_compare5.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--num_questions", "5"', source)
        self.assertIn('"feature_version": 302', source)
        self.assertIn('"--allow_output_mismatch"', source)


if __name__ == "__main__":
    unittest.main()
