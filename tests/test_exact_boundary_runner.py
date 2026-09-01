import unittest

import pandas as pd

from run_exact_boundary_oracle import (
    build_exact_behavior_policy,
    evenly_spaced,
    run_method,
)


class ExactBoundaryRunnerTests(unittest.TestCase):
    def test_evenly_spaced_rounds_include_trajectory_ends(self):
        self.assertEqual(evenly_spaced(range(10), 5), [0, 2, 4, 7, 9])

    def test_policy_marks_five_rounds_and_preserves_actions(self):
        rows = []
        for round_id in range(10):
            rows.append({
                "problem_id": 7,
                "round_id": round_id,
                "decision_id": 99,
                "action": "stop" if round_id % 2 else "continue",
                "stop_available": True,
            })
        policy, selected, selected_rounds = build_exact_behavior_policy(
            pd.DataFrame(rows),
            pd.DataFrame([{"problem_id": 7, "num_speculation_rounds": 10}]),
            5,
        )
        rounds = policy["policies"]["7"]
        self.assertEqual(sum(row["exact_boundary_probe"] for row in rounds), 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(selected_rounds), 5)
        self.assertEqual(rounds[0]["actions"], ["continue"])

    def test_policy_replays_forced_decisions_but_probes_only_legal_rounds(self):
        rows = pd.DataFrame([
            {
                "problem_id": 7,
                "round_id": 0,
                "decision_id": 0,
                "action": "continue",
                "executed_action": "continue",
                "stop_available": False,
            },
            {
                "problem_id": 7,
                "round_id": 0,
                "decision_id": 1,
                "action": "stop",
                "executed_action": "stop",
                "stop_available": True,
            },
        ])
        policy, selected, selected_rounds = build_exact_behavior_policy(
            rows,
            pd.DataFrame([{"problem_id": 7, "num_speculation_rounds": 1}]),
            1,
        )
        round_zero = policy["policies"]["7"][0]
        self.assertEqual(round_zero["actions"], ["continue", "stop"])
        self.assertTrue(round_zero["exact_boundary_probe"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected_rounds), 1)

    def test_behavior_collection_disables_kv_reuse_like_exact_replay(self):
        source = __import__("inspect").getsource(run_method)
        self.assertIn('command.append("--disable_reusing_drafter_kvs")', source)
        self.assertIn('command.append("--adaptive-collect-raw-state")', source)


if __name__ == "__main__":
    unittest.main()
