import unittest

from adaptive_td import (
    AdaptiveTDConfig,
    CONTINUE,
    OnlineTDRefinementController,
    STOP,
)


FEATURES = (1.0,) + (0.0,) * 12


def factual_controller():
    controller = OnlineTDRefinementController(
        AdaptiveTDConfig(
            credit_assignment="verifier_boundary_factual",
            full_stream_bootstrap=True,
            learning_rate=0.01,
            profile_overhead=False,
        )
    )
    controller.observe_round(1, 10.0)
    controller.algorithm_latency_ms = 0.0
    controller.algorithm_latency_components_ms.clear()
    return controller


def record(action, anchor_ms, rho=0.1):
    return {
        "action": action,
        "features": FEATURES,
        "algorithm_latency_anchor_ms": anchor_ms,
        "rho_tokens_per_ms": rho,
        "selected_action_probability": 1.0,
        "importance_weight": 1.0,
    }


class VerifierBoundaryFactualTests(unittest.TestCase):
    def test_per_step_transition_is_disabled(self):
        controller = factual_controller()

        residual = controller.observe_continue_transition(
            FEATURES,
            FEATURES,
            4.0,
        )

        self.assertIsNone(residual)
        self.assertEqual(controller.values[CONTINUE].sample_count, 0)

    def test_boundary_waits_for_next_state_and_includes_its_creation_cost(self):
        controller = factual_controller()
        controller.values[STOP].theta[0] = 1.0
        controller.values[CONTINUE].theta[0] = 2.0
        trajectory = [
            record(CONTINUE, 0.0),
            record(STOP, 2.0),
        ]
        controller.advance_algorithm_latency(5.0, component="draft_forward")

        controller.complete_trajectory(
            trajectory,
            emitted_tokens=3,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=2.0,
            terminal=False,
        )

        self.assertIsNotNone(controller.pending_factual_boundary)
        self.assertEqual(controller.values[CONTINUE].sample_count, 0)
        self.assertEqual(controller.values[STOP].sample_count, 0)

        controller.advance_algorithm_latency(
            4.0,
            component="next_state_draft_forward",
        )
        transitions = controller.resolve_pending_factual_boundary(
            FEATURES,
            next_stop_available=True,
        )

        self.assertEqual(len(transitions), 2)
        self.assertEqual(controller.values[CONTINUE].sample_count, 1)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        self.assertAlmostEqual(transitions[0]["bootstrap_value"], 2.0)
        self.assertAlmostEqual(transitions[0]["delta_time_ms"], 14.0)
        self.assertAlmostEqual(transitions[0]["post_boundary_latency_ms"], 4.0)
        self.assertAlmostEqual(transitions[0]["td_target"], 3.6)
        self.assertAlmostEqual(transitions[1]["delta_time_ms"], 12.0)
        self.assertAlmostEqual(transitions[1]["td_target"], 3.8)

    def test_terminal_boundary_has_no_bootstrap(self):
        controller = factual_controller()
        trajectory = [record(STOP, 0.0)]
        controller.advance_algorithm_latency(4.0, component="draft_forward")

        controller.complete_trajectory(
            trajectory,
            emitted_tokens=2,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=True,
        )

        self.assertIsNone(controller.pending_factual_boundary)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        transition = controller.full_stream_transitions[-1]
        self.assertTrue(transition["terminal"])
        self.assertEqual(transition["bootstrap_value"], 0.0)
        self.assertAlmostEqual(transition["delta_time_ms"], 8.0)
        self.assertAlmostEqual(transition["td_target"], 1.2)

    def test_pending_return_spans_boundary_without_adaptive_state(self):
        controller = factual_controller()
        controller.advance_algorithm_latency(2.0, component="draft_forward")
        controller.complete_trajectory(
            [record(CONTINUE, 0.0)],
            emitted_tokens=2,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=False,
        )
        controller.advance_algorithm_latency(2.0, component="draft_forward")

        controller.complete_trajectory(
            [],
            emitted_tokens=3,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=False,
        )
        controller.advance_algorithm_latency(2.0, component="next_state_forward")
        transitions = controller.resolve_pending_factual_boundary(FEATURES)

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["emitted_tokens"], 5)
        self.assertEqual(transitions[0]["verifier_boundaries_spanned"], 2)
        self.assertAlmostEqual(transitions[0]["delta_time_ms"], 14.0)

    def test_snapshot_identifies_factual_credit_assignment(self):
        controller = factual_controller()
        snapshot = controller.snapshot()

        self.assertEqual(
            snapshot["credit_assignment"],
            "verifier_boundary_factual",
        )
        self.assertIn("algorithm_latency_components_ms", snapshot)


if __name__ == "__main__":
    unittest.main()
