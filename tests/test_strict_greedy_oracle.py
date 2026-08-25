import unittest

from adaptive_td import CONTINUE, STOP
from strict_greedy_oracle import (
    GreedyBranch,
    build_oracle_state_key,
    choose_strict_greedy_action,
    format_outer_path,
    one_action_rollout_scripts,
)


class StrictGreedyOracleTests(unittest.TestCase):
    def test_one_action_rollout_does_not_force_stop_at_next_state(self):
        stop_script, continue_script = one_action_rollout_scripts([CONTINUE])
        self.assertEqual(stop_script, (CONTINUE, STOP))
        self.assertEqual(continue_script, (CONTINUE, CONTINUE))

    def test_state_key_covers_prefix_and_provisional_proposal(self):
        base = build_oracle_state_key(2, 300, 8, 1, [1, None, 3])
        self.assertEqual(
            base,
            build_oracle_state_key(2, 300, 8, 1, [1, None, 3]),
        )
        self.assertNotEqual(
            base,
            build_oracle_state_key(2, 301, 8, 1, [1, None, 3]),
        )
        self.assertNotEqual(
            base,
            build_oracle_state_key(2, 300, 8, 1, [1, 4, 3]),
        )

    def test_sequential_greedy_continue_then_stop(self):
        first = choose_strict_greedy_action(
            GreedyBranch(100.0, 5),
            GreedyBranch(90.0, 5),
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=0.0,
        )
        second = choose_strict_greedy_action(
            GreedyBranch(80.0, 5),
            GreedyBranch(85.0, 5),
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=0.0,
        )
        self.assertEqual(first.action, CONTINUE)
        self.assertEqual(second.action, STOP)

    def test_myopic_limit_does_not_observe_step_three(self):
        decision = choose_strict_greedy_action(
            GreedyBranch(100.0, 5),
            GreedyBranch(105.0, 5),
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=0.0,
        )
        unreachable_step_three_cost = 70.0
        self.assertEqual(decision.action, STOP)
        self.assertLess(unreachable_step_three_cost, decision.stop_score_ms)

    def test_verifier_penalty_reverses_immediate_stop(self):
        decision = choose_strict_greedy_action(
            GreedyBranch(170.0, 4),
            GreedyBranch(210.0, 7),
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=0.0,
        )
        self.assertEqual(decision.immediate_action, STOP)
        self.assertEqual(decision.action, CONTINUE)
        self.assertAlmostEqual(decision.stop_extra_calls, 0.6)
        self.assertAlmostEqual(decision.stop_penalty_ms, 84.0)
        self.assertAlmostEqual(decision.stop_score_ms, 254.0)
        self.assertAlmostEqual(decision.continue_score_ms, 210.0)

    def test_tie_falls_back_to_failfast_continue(self):
        decision = choose_strict_greedy_action(
            GreedyBranch(100.0, 5),
            GreedyBranch(100.5, 5),
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=1.0,
            baseline_action=CONTINUE,
        )
        self.assertTrue(decision.tie_fallback_used)
        self.assertEqual(decision.action, CONTINUE)

    def test_outer_extend_path_is_preserved_in_local_branch(self):
        stop = GreedyBranch(170.0, 4)
        continue_ = GreedyBranch(210.0, 7)
        decision = choose_strict_greedy_action(
            stop,
            continue_,
            mean_verify_latency_ms=140.0,
            mean_tokens_per_verify=5.0,
            epsilon_ms=0.0,
        )
        self.assertEqual(format_outer_path(0), "VERIFY")
        self.assertEqual(format_outer_path(2), "EXTEND -> EXTEND -> VERIFY")
        self.assertEqual(decision.action, CONTINUE)


if __name__ == "__main__":
    unittest.main()
