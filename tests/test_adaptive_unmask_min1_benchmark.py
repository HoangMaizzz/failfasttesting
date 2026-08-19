import sys
import unittest
from unittest.mock import patch

import run_adaptive_unmask_min1_benchmark as benchmark


class AdaptiveUnmaskMin1BenchmarkTests(unittest.TestCase):
    def test_method_is_refinement_only_min1_variant(self):
        self.assertEqual(benchmark.benchmark.METHOD["name"], "adaptive_unmask_only_min1")
        self.assertEqual(
            benchmark.benchmark.METHOD["frontier_mode"],
            "cost_aware_v2_refinement_only",
        )

    def test_default_min_steps_is_one(self):
        with patch.object(sys, "argv", ["run_adaptive_unmask_min1_benchmark.py"]):
            args = benchmark.parse_args()

        self.assertEqual(args.frontier_min_steps, 1)
        self.assertEqual(args.output_dir, benchmark.DEFAULT_OUTPUT_DIR)
        self.assertEqual(args.frontier_v2_first_step_prior_shrink, 1.0)
        self.assertEqual(args.frontier_v2_first_step_prior_floor, 0.6)
        self.assertEqual(args.frontier_v2_min_refinement_gain_observations, 4)

    def test_explicit_min_steps_is_preserved(self):
        with patch.object(
            sys,
            "argv",
            ["run_adaptive_unmask_min1_benchmark.py", "--frontier_min_steps", "3"],
        ):
            args = benchmark.parse_args()

        self.assertEqual(args.frontier_min_steps, 3)


if __name__ == "__main__":
    unittest.main()
