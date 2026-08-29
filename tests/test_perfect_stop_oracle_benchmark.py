import unittest
from argparse import Namespace
from pathlib import Path

from run_otrc_v2_td_benchmark import PROBLEM_IDS
from global_oracle import solve_canonical_oracle_graph
from run_perfect_stop_oracle_benchmark import (
    DATASET_COUNTS,
    command_for,
    selected_problem_ids,
    validate_args,
)


class PerfectStopOracleBenchmarkTests(unittest.TestCase):
    def args(self):
        return Namespace(
            batch_size=5,
            max_new_tokens=1024,
            drafter_threshold=0.30,
            lowconf_threshold=0.50,
            target_device=0,
            drafter_device=0,
            target_quantization="int8",
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/oracle",
            global_oracle_max_states=0,
            resume=True,
            skip_archive=False,
            log_level="INFO",
        )

    def test_selected_ids_include_the_prior_25(self):
        ids = selected_problem_ids()
        self.assertEqual({key: len(value) for key, value in ids.items()}, DATASET_COUNTS)
        for dataset in DATASET_COUNTS:
            self.assertEqual(ids[dataset][:25], PROBLEM_IDS[dataset][:25])

    def test_command_runs_exact_unbounded_greedy_oracle(self):
        args = self.args()
        validate_args(args)
        command = command_for(args, "math", PROBLEM_IDS["math"][:5], Path("/tmp/out"))
        joined = " ".join(command)
        self.assertIn("--global_oracle_graph", command)
        self.assertIn("--global_oracle_max_states 0", joined)
        self.assertIn("--disable_reusing_drafter_kvs", command)
        self.assertIn("--decoding_strategy greedy", joined)
        self.assertIn("--drafter_thresholds 0.3", joined)
        self.assertIn("--sweep_lowconf_threshold 0.5", joined)
        self.assertIn("--sweep_max_spec_len 64", joined)
        self.assertNotIn("--adaptive-td", command)
        self.assertNotIn("--collect_bucket_oracle", command)

        source = (Path(__file__).resolve().parents[1] / "failfast.py").read_text(
            encoding="utf-8"
        )
        frontier_return = source.split("return_frontier_stats = (", 1)[1][:300]
        self.assertIn('getattr(args, "global_oracle_graph", False)', frontier_return)

    def test_global_solver_uses_future_cost_not_immediate_ratio(self):
        edges = {
            0: [
                {
                    "candidate_index": 0,
                    "draft_passes": 1,
                    "step": 1,
                    "edge_latency_ms": 2.0,
                    "emitted_len": 1,
                    "child_state": 1,
                    "is_failfast": 0,
                },
                {
                    "candidate_index": 1,
                    "draft_passes": 2,
                    "step": 2,
                    "edge_latency_ms": 5.0,
                    "emitted_len": 2,
                    "child_state": 2,
                    "is_failfast": 1,
                },
            ],
            1: [
                {
                    "candidate_index": 0,
                    "draft_passes": 1,
                    "step": 1,
                    "edge_latency_ms": 10.0,
                    "emitted_len": 1,
                    "child_state": 2,
                    "is_failfast": 1,
                }
            ],
        }
        solved = solve_canonical_oracle_graph(edges, terminal_state=2)
        self.assertEqual(solved["policies"]["global"][0]["child_state"], 2)
        self.assertEqual(solved["values"]["global"][0], 5.0)
        self.assertLessEqual(
            solved["values"]["global"][0],
            solved["values"]["failfast"][0],
        )


if __name__ == "__main__":
    unittest.main()
