import unittest
from types import SimpleNamespace

from run_compact6_nam_ga2m_benchmark import DATASETS, command
from run_otrc_v2_td_benchmark import PROBLEM_IDS


class Compact6NamGa2mBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            num_questions=25,
            target_quantization="int8",
            dllm_dir="/tmp/dllm",
            output_dir="/tmp/out",
            resume=False,
        )

    def test_commands_differ_only_by_value_model_destination(self):
        nam = command(self.args, "nam")
        ga2m = command(self.args, "ga2m")
        self.assertIn("nam", nam)
        self.assertIn("ga2m", ga2m)
        for required in (
            "otrc_v2_2_compact_td",
            "shared_value_advantage",
            "verifier_boundary_factual_no_bootstrap",
            "symmetric_annealed",
            "int8",
        ):
            self.assertIn(required, nam)
            self.assertIn(required, ga2m)

    def test_matched_id_sets_have_25_unique_questions(self):
        for dataset in DATASETS:
            selected = PROBLEM_IDS[dataset][:25]
            self.assertEqual(len(selected), 25)
            self.assertEqual(len(set(selected)), 25)


if __name__ == "__main__":
    unittest.main()
