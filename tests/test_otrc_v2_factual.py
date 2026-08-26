import unittest

from adaptive_td import (
    AdaptiveTDConfig,
    CONTINUE,
    OnlineTDRefinementController,
    STOP,
)


FEATURES = (1.0,) + (0.0,) * 12


def factual_controller(
    credit_assignment="verifier_boundary_factual",
    rho_warmup_boundaries=0,
    policy_weight_ema_beta=0.0,
):
    controller = OnlineTDRefinementController(
        AdaptiveTDConfig(
            credit_assignment=credit_assignment,
            full_stream_bootstrap=True,
            learning_rate=0.01,
            profile_overhead=False,
            rho_warmup_boundaries=rho_warmup_boundaries,
            policy_weight_ema_beta=policy_weight_ema_beta,
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
    def test_policy_weight_ema_rejects_bootstrapped_credit(self):
        with self.assertRaisesRegex(ValueError, "no-bootstrap"):
            AdaptiveTDConfig(
                credit_assignment="verifier_boundary_factual",
                policy_weight_ema_beta=0.99,
            )

    def test_policy_weight_ema_updates_only_the_factual_action(self):
        controller = factual_controller(
            "verifier_boundary_factual_no_bootstrap",
            policy_weight_ema_beta=0.5,
        )

        controller.complete_trajectory(
            [record(STOP, 0.0, rho=0.0)],
            emitted_tokens=2,
            verifier_latency_ms=1.0,
            post_verify_latency_ms=0.0,
            terminal=False,
        )
        first_online = list(controller.values[STOP].theta)
        self.assertEqual(controller.policy_ema_theta[STOP], first_online)
        self.assertTrue(controller.policy_ema_initialized[STOP])
        self.assertFalse(controller.policy_ema_initialized[CONTINUE])

        anchor = controller.algorithm_latency_ms
        controller.complete_trajectory(
            [record(STOP, anchor, rho=0.0)],
            emitted_tokens=4,
            verifier_latency_ms=1.0,
            post_verify_latency_ms=0.0,
            terminal=False,
        )
        second_online = controller.values[STOP].theta
        expected = [
            0.5 * old + 0.5 * new
            for old, new in zip(first_online, second_online)
        ]
        for actual, wanted in zip(controller.policy_ema_theta[STOP], expected):
            self.assertAlmostEqual(actual, wanted)

        stop_ema = list(controller.policy_ema_theta[STOP])
        anchor = controller.algorithm_latency_ms
        controller.complete_trajectory(
            [record(CONTINUE, anchor, rho=0.0)],
            emitted_tokens=3,
            verifier_latency_ms=1.0,
            post_verify_latency_ms=0.0,
            terminal=False,
        )
        self.assertEqual(controller.policy_ema_theta[STOP], stop_ema)
        self.assertEqual(
            controller.policy_ema_theta[CONTINUE],
            controller.values[CONTINUE].theta,
        )
        self.assertEqual(controller.policy_ema_update_count[STOP], 2)
        self.assertEqual(controller.policy_ema_update_count[CONTINUE], 1)

    def test_policy_uses_ema_mean_while_logging_raw_shadow_policy(self):
        controller = OnlineTDRefinementController(AdaptiveTDConfig(
            credit_assignment="verifier_boundary_factual_no_bootstrap",
            policy_weight_ema_beta=0.99,
            policy_mode="symmetric",
        ))
        controller.values[STOP].theta[0] = 5.0
        controller.policy_ema_theta[STOP][0] = -1.0
        controller.policy_ema_initialized[STOP] = True
        controller.policy_ema_theta[CONTINUE][0] = 0.0
        controller.policy_ema_initialized[CONTINUE] = True

        decision = controller.choose(
            FEATURES,
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
        )

        self.assertEqual(decision.action, CONTINUE)
        self.assertEqual(decision.diagnostics["raw_greedy_action"], STOP)
        self.assertEqual(decision.diagnostics["ema_greedy_action"], CONTINUE)
        self.assertTrue(
            decision.diagnostics["raw_ema_greedy_disagreement"]
        )

    def test_policy_weight_ema_round_trips_through_snapshot(self):
        original = factual_controller(
            "verifier_boundary_factual_no_bootstrap",
            policy_weight_ema_beta=0.99,
        )
        original.complete_trajectory(
            [record(STOP, 0.0, rho=0.0)],
            emitted_tokens=2,
            verifier_latency_ms=1.0,
            post_verify_latency_ms=0.0,
            terminal=False,
        )
        restored = factual_controller(
            "verifier_boundary_factual_no_bootstrap",
            policy_weight_ema_beta=0.99,
        )
        restored.load_snapshot(original.snapshot())

        self.assertEqual(restored.policy_ema_theta, original.policy_ema_theta)
        self.assertEqual(
            restored.policy_ema_initialized,
            original.policy_ema_initialized,
        )
        self.assertEqual(
            restored.policy_ema_update_count,
            original.policy_ema_update_count,
        )

    def test_rho_warmup_rejects_bootstrapped_credit(self):
        with self.assertRaisesRegex(ValueError, "no-bootstrap"):
            AdaptiveTDConfig(
                credit_assignment="verifier_boundary_factual",
                rho_warmup_boundaries=1,
            )

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

    def test_no_bootstrap_resolves_immediately_at_verifier_boundary(self):
        controller = factual_controller(
            "verifier_boundary_factual_no_bootstrap"
        )
        controller.values[STOP].theta[0] = 100.0
        controller.values[CONTINUE].theta[0] = 200.0
        controller.advance_algorithm_latency(5.0, component="draft_forward")

        controller.complete_trajectory(
            [record(CONTINUE, 0.0)],
            emitted_tokens=3,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=2.0,
            terminal=False,
        )

        self.assertIsNone(controller.pending_factual_boundary)
        self.assertEqual(controller.values[CONTINUE].sample_count, 1)
        self.assertEqual(controller.values[STOP].sample_count, 0)
        transition = controller.full_stream_transitions[-1]
        self.assertFalse(transition["terminal"])
        self.assertEqual(transition["bootstrap_value"], 0.0)
        self.assertAlmostEqual(transition["delta_time_ms"], 10.0)
        self.assertEqual(transition["post_boundary_latency_ms"], 0.0)
        self.assertAlmostEqual(transition["td_target"], 2.0)

    def test_no_bootstrap_does_not_wait_for_next_adaptive_state(self):
        controller = factual_controller(
            "verifier_boundary_factual_no_bootstrap"
        )
        controller.advance_algorithm_latency(2.0, component="draft_forward")
        controller.complete_trajectory(
            [record(STOP, 0.0)],
            emitted_tokens=2,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=False,
        )

        transition_count = len(controller.full_stream_transitions)
        controller.advance_algorithm_latency(
            7.0,
            component="next_state_draft_forward",
        )
        resolved = controller.resolve_pending_factual_boundary(FEATURES)

        self.assertEqual(resolved, [])
        self.assertEqual(len(controller.full_stream_transitions), transition_count)

    def test_rho_warmup_collects_targets_without_updating_q_heads(self):
        controller = factual_controller(
            "verifier_boundary_factual_no_bootstrap",
            rho_warmup_boundaries=2,
        )
        actions = [STOP, CONTINUE, STOP]
        for action in actions:
            anchor = controller.algorithm_latency_ms
            controller.advance_algorithm_latency(2.0, component="draft_forward")
            controller.complete_trajectory(
                [record(action, anchor)],
                emitted_tokens=2,
                verifier_latency_ms=3.0,
                post_verify_latency_ms=1.0,
                terminal=False,
            )

        transitions = controller.full_stream_transitions
        self.assertEqual(
            [item["update_applied"] for item in transitions],
            [False, False, True],
        )
        self.assertEqual(controller.factual_warmup_transition_count, 2)
        self.assertEqual(controller.factual_learning_update_count, 1)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        self.assertEqual(controller.values[CONTINUE].sample_count, 0)

    def test_rho_warmup_counts_verifier_boundaries_without_decisions(self):
        controller = factual_controller(
            "verifier_boundary_factual_no_bootstrap",
            rho_warmup_boundaries=1,
        )
        controller.complete_trajectory(
            [],
            emitted_tokens=1,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=False,
        )
        anchor = controller.algorithm_latency_ms
        controller.complete_trajectory(
            [record(STOP, anchor)],
            emitted_tokens=2,
            verifier_latency_ms=3.0,
            post_verify_latency_ms=1.0,
            terminal=False,
        )

        self.assertEqual(controller.factual_observed_verifier_boundaries, 2)
        self.assertTrue(
            controller.full_stream_transitions[-1]["update_applied"]
        )


if __name__ == "__main__":
    unittest.main()
