import math
import time
import unittest

import pandas as pd

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController, STOP
from distributional_controller import (
    DistributionalControllerConfig,
    DistributionalTimeTokenController,
)
from run_adaptive_td_benchmark import paired_comparison


FEATURES = (
    1.0,
    0.5,
    0.25,
    0.6,
    0.4,
    0.8,
    0.1,
    0.2,
    0.25,
    0.5,
    0.0,
    0.0,
    0.2,
)


class FullStreamAverageRewardTests(unittest.TestCase):
    def test_stop_update_is_deferred_until_next_adaptive_state(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                full_stream_bootstrap=True,
                reverse_backup=False,
                learning_rate=0.01,
            )
        )
        controller.observe_round(4, 20.0)
        trajectory = [{
            "action": STOP,
            "features": FEATURES,
            "post_stop_outer_action": "verify",
            "importance_weight": 1.0,
            "decision_monotonic_s": time.perf_counter(),
        }]

        controller.complete_trajectory(
            trajectory,
            emitted_tokens=3,
            verifier_latency_ms=2.0,
            terminal=False,
        )

        self.assertIsNotNone(controller.pending_stop)
        self.assertEqual(controller.values[STOP].sample_count, 0)
        update = controller.resolve_pending_stop(
            FEATURES,
            next_stop_available=True,
        )
        self.assertIsNotNone(update)
        self.assertIsNone(controller.pending_stop)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        self.assertFalse(update["terminal"])
        self.assertIn("bootstrap_value", update)

    def test_terminal_stop_has_no_bootstrap(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(full_stream_bootstrap=True)
        )
        trajectory = [{
            "action": STOP,
            "features": FEATURES,
            "post_stop_outer_action": "verify",
            "importance_weight": 1.0,
            "decision_monotonic_s": time.perf_counter(),
        }]
        controller.complete_trajectory(
            trajectory,
            emitted_tokens=2,
            verifier_latency_ms=1.0,
            terminal=True,
        )
        self.assertIsNone(controller.pending_stop)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        self.assertEqual(controller.full_stream_transitions[-1]["bootstrap_value"], 0.0)


class DistributionalControllerTests(unittest.TestCase):
    def test_symmetric_expected_regret_difference_equals_mean_difference(self):
        controller = DistributionalTimeTokenController(
            DistributionalControllerConfig()
        )
        for mean, std in [(-3.0, 0.5), (-0.2, 2.0), (0.0, 1.0), (4.0, 3.0)]:
            regret_continue = controller._expected_positive_normal(mean, std)
            regret_stop = controller._expected_positive_normal(-mean, std)
            self.assertAlmostEqual(regret_continue - regret_stop, mean, places=10)

    def test_ratio_distribution_is_finite_and_positive(self):
        mean, std = DistributionalTimeTokenController._ratio_distribution(
            20.0,
            4.0,
            5.0,
            1.0,
        )
        self.assertTrue(math.isfinite(mean))
        self.assertTrue(math.isfinite(std))
        self.assertGreater(mean, 0.0)
        self.assertGreater(std, 0.0)

    def test_factual_stop_updates_only_stop_output_model(self):
        controller = DistributionalTimeTokenController(
            DistributionalControllerConfig(warmup_rounds=0)
        )
        trajectory = [{
            "action": STOP,
            "features": FEATURES,
            "t_elapsed_ms": 5.0,
        }]
        controller.complete_trajectory(
            trajectory,
            emitted_tokens=4,
            verifier_latency_ms=10.0,
            round_latency_ms=18.0,
        )
        self.assertEqual(controller.stop_output.sample_count, 1)
        self.assertEqual(controller.continue_output.sample_count, 0)
        self.assertEqual(controller.verify_latency.count, 1)
        self.assertEqual(controller.stop_path_extra_latency.count, 1)

    def test_decision_logs_distributional_components(self):
        controller = DistributionalTimeTokenController(
            DistributionalControllerConfig(
                warmup_rounds=0,
                explore_epsilon=0.0,
                explore_min=0.0,
            )
        )
        decision = controller.choose(
            FEATURES,
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
            elapsed_draft_ms=6.0,
            proposal_length=8,
            frontier_length=3,
        )
        self.assertIn(decision.action, {"stop", "continue"})
        for key in (
            "y_stop_mean",
            "y_continue_mean",
            "j_stop_mean",
            "j_continue_mean",
            "difference_mean",
            "expected_regret_stop",
            "expected_regret_continue",
        ):
            self.assertIn(key, decision.diagnostics)
            self.assertTrue(math.isfinite(float(decision.diagnostics[key])))


class AdaptiveControllerBenchmarkTests(unittest.TestCase):
    def test_pairing_compares_both_controllers_to_same_failfast_rows(self):
        rows = pd.DataFrame({
            "dataset": ["gsm8k"] * 3,
            "problem_id": [7, 7, 7],
            "method": ["failfast", "avg_td", "dist_time_token"],
            "measured_ms_per_output_token": [10.0, 8.0, 12.0],
            "e2e_ms_per_output_token": [11.0, 9.0, 13.0],
            "output_token_hash": ["a", "a", "a"],
            "num_speculation_rounds": [10, 9, 11],
            "total_num_forward_passes": [20, 18, 22],
            "output_tokens": [100, 100, 100],
        })
        paired = paired_comparison(rows)
        self.assertEqual(set(paired["method"]), {"avg_td", "dist_time_token"})
        speedups = paired.set_index("method")["measured_speedup_vs_failfast"]
        self.assertAlmostEqual(speedups["avg_td"], 1.25)
        self.assertAlmostEqual(speedups["dist_time_token"], 10.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
