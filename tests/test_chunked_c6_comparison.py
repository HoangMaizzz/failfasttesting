import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from run_chunked_c6_comparison_test50 import (
    adaptive_flags,
    base_command,
    candidate_pool,
    chunks,
    failure_category,
    RunFailure,
    summarize,
)


class ChunkedC6ComparisonTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            dllm_dir="/tmp/dllm",
            target_device=0,
            drafter_device=0,
            target_quantization="int8",
            log_level="INFO",
        )

    def test_chunks_preserve_order_without_overlap(self):
        values = list(range(50))
        observed = list(chunks(values, 2))
        self.assertEqual([value for _, part in observed for value in part], values)
        self.assertEqual(len(observed), 25)

    def test_second_chunk_loads_prior_adaptive_state(self):
        flags = adaptive_flags("c6_annealed", "/tmp/state.json")
        self.assertIn("--adaptive-state-path", flags)
        self.assertIn("/tmp/state.json", flags)
        self.assertIn("symmetric_annealed", flags)

    def test_failfast_base_command_is_matched_int8(self):
        command = base_command(self.args(), "math", [2, 6], "/tmp/out", 1)
        joined = " ".join(map(str, command))
        self.assertIn("--problem_ids 2 6", joined)
        self.assertIn("--target_quantization int8", joined)
        self.assertIn("--drafter_thresholds 0.3", joined)
        self.assertIn("--sweep_lowconf_threshold 0.5", joined)

    def test_candidate_pool_keeps_preferred_ids_then_unique_backups(self):
        pool = candidate_pool("math", 50)
        self.assertEqual(pool[:50], [int(value) for value in __import__(
            "run_otrc_v2_td_benchmark"
        ).PROBLEM_IDS["math"][:50]])
        self.assertEqual(len(pool), len(set(pool)))
        self.assertGreater(len(pool), 50)

    def test_oom_failure_is_classified(self):
        error = RunFailure(1, ["python"], ["torch.OutOfMemoryError: CUDA out of memory\n"])
        self.assertEqual(failure_category(error), "cuda_oom")

    def test_summary_builds_failfast_latency_before_speedups(self):
        with TemporaryDirectory() as temporary:
            runs = {}
            for dataset in ("math", "gsm8k"):
                for index, method in enumerate(("c6_annealed", "always_stop", "failfast")):
                    directory = Path(temporary) / dataset / method
                    directory.mkdir(parents=True)
                    pd.DataFrame([{
                        "problem_id": 1,
                        "actual_algorithm_time": 1.0 + index,
                        "output_tokens": 10,
                        "drafted_tokens": 12,
                        "output_token_hash": "same",
                        "accepted_tokens": 8,
                        "actual_draft_time": 0.4,
                        "actual_verify_time": 0.6,
                        "actual_post_verify_time": 0.0,
                        "total_num_forward_passes": 2,
                        "num_speculation_rounds": 1,
                        "is_correct": True,
                    }]).to_csv(directory / "benchmark_results.csv", index=False)
                    runs[(dataset, method)] = [directory]
            args = SimpleNamespace(output_dir=temporary)
            summarize(args, runs)
            paired = pd.read_csv(Path(temporary) / "paired_problem_comparison.csv")
            self.assertIn("failfast_ms_per_output_token", paired.columns)
            self.assertIn("c6_annealed_speedup_vs_failfast", paired.columns)


if __name__ == "__main__":
    unittest.main()
