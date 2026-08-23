import unittest

from adaptive_td import (
    CONTINUE,
    STOP,
    AdaptiveTDConfig,
    OnlineTDRefinementController,
    build_state_features,
    trajectory_forward_latency,
)


def synthetic_features(step, preferred_depth):
    confidence = min(0.95, 0.1 + 0.85 * step / preferred_depth)
    return build_state_features(
        proposal_length=8,
        remaining_masks=max(1, 8 - min(7, int(7 * step / preferred_depth))),
        newly_unmasked=1,
        recoverable_confidences=[confidence] * 4,
        recoverable_margins=[confidence * 0.5] * 4,
        first_remaining_position=min(7, step),
        frontier_length=min(7, int(7 * step / preferred_depth)),
        proposal_change_ratio=max(0.0, 0.5 - step * 0.03),
        recoverable_change_ratio=max(0.0, 0.6 - step * 0.04),
        refinement_step=step,
        max_refinement_steps=8,
        use_step=False,
    )


class AdaptiveTDTests(unittest.TestCase):
    def test_default_exploration_schedule_supports_online_stop_learning(self):
        config = AdaptiveTDConfig()
        self.assertEqual(config.explore_epsilon, 0.10)
        self.assertEqual(config.explore_min, 0.01)
        self.assertEqual(config.explore_decay, 0.998)
        self.assertEqual(config.early_stop_min_observations, 32)

    def test_calibration_blocks_interval_exploitation_until_early_stop_data_exists(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                warmup_rounds=0,
                early_stop_min_observations=1,
                explore_epsilon=0.0,
                explore_min=0.0,
            )
        )
        features = synthetic_features(1, 3)
        controller.values[STOP].theta[0] = 100.0
        decision = controller.choose(
            features,
            allow_stop=True,
            refinement_step=1,
        )
        self.assertEqual(decision.action, CONTINUE)
        self.assertEqual(decision.reason, "early_stop_calibration_continue")
        self.assertTrue(decision.calibration_active)

        controller.early_stop_observations = 1
        decision = controller.choose(
            features,
            allow_stop=True,
            refinement_step=1,
        )
        self.assertEqual(decision.action, STOP)
        self.assertEqual(decision.reason, "stop_interval_dominates")
        self.assertFalse(decision.calibration_active)

    def test_only_factual_early_stop_increments_calibration_count(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="factual_return")
        )
        features = synthetic_features(1, 3)
        controller.complete_trajectory(
            [{
                "features": features,
                "action": STOP,
                "reason": "factual_terminal_verification",
                "remaining_masks": 0,
            }],
            emitted_tokens=1,
            verifier_latency_ms=1.0,
        )
        self.assertEqual(controller.early_stop_observations, 0)

        controller.complete_trajectory(
            [{
                "features": features,
                "action": STOP,
                "reason": "early_stop_calibration_exploration",
                "remaining_masks": 2,
            }],
            emitted_tokens=1,
            verifier_latency_ms=1.0,
        )
        self.assertEqual(controller.early_stop_observations, 1)

    def test_features_are_fixed_width_and_normalized(self):
        features = synthetic_features(2, 3)
        self.assertEqual(len(features), 13)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features))

    def test_cold_start_and_force_continue_preserve_baseline_action(self):
        features = synthetic_features(1, 3)
        cold = OnlineTDRefinementController(
            AdaptiveTDConfig(warmup_rounds=10, explore_epsilon=0.0, explore_min=0.0)
        )
        forced = OnlineTDRefinementController(
            AdaptiveTDConfig(
                warmup_rounds=0,
                force_continue=True,
                explore_epsilon=0.0,
                explore_min=0.0,
            )
        )
        self.assertEqual(
            cold.choose(features, allow_stop=True, refinement_step=1).action,
            CONTINUE,
        )
        self.assertEqual(
            forced.choose(features, allow_stop=True, refinement_step=1).action,
            CONTINUE,
        )

    def test_unavailable_provisional_proposal_forces_continue(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(warmup_rounds=0, explore_epsilon=1.0, explore_min=1.0)
        )
        decision = controller.choose(
            synthetic_features(1, 1),
            allow_stop=False,
            refinement_step=1,
        )
        self.assertEqual(decision.action, CONTINUE)
        self.assertEqual(decision.reason, "provisional_proposal_unavailable")

    def test_fixed_depth_baseline_uses_same_step_hook(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                fixed_refinement_steps=3,
                warmup_rounds=100,
                explore_epsilon=0.0,
                explore_min=0.0,
            )
        )
        actions = [
            controller.choose(
                synthetic_features(step, 3),
                allow_stop=True,
                refinement_step=step,
            ).action
            for step in (1, 2, 3)
        ]
        self.assertEqual(actions, [CONTINUE, CONTINUE, STOP])

    def test_factual_update_never_updates_unexecuted_actions(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="factual_return", warmup_rounds=0)
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        first = synthetic_features(1, 3)
        second = synthetic_features(2, 3)
        controller.complete_trajectory(
            [
                {"features": first, "action": CONTINUE, "next_forward_latency_ms": 1.0},
                {"features": second, "action": STOP, "next_forward_latency_ms": 0.0},
            ],
            emitted_tokens=4,
            verifier_latency_ms=2.0,
        )
        self.assertEqual(controller.values[CONTINUE].sample_count, 1)
        self.assertEqual(controller.values[STOP].sample_count, 1)

    def test_decision_does_not_update_before_factual_feedback(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(warmup_rounds=0, explore_epsilon=0.0, explore_min=0.0)
        )
        features = synthetic_features(1, 3)
        controller.choose(features, allow_stop=True, refinement_step=1)
        self.assertEqual(controller.values[STOP].sample_count, 0)
        self.assertEqual(controller.values[CONTINUE].sample_count, 0)

    def test_mean_uncertainty_shrinks_with_repeated_observations(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                warmup_rounds=0,
                epistemic_scale=1.0,
                explore_epsilon=0.0,
                explore_min=0.0,
            )
        )
        features = synthetic_features(1, 3)
        value = controller.values[STOP]
        initial_risk = value.risk(features)
        for index in range(200):
            value.update(features, float(index % 2), rate=0.0)
        self.assertGreater(value.residual_variance(), 0.1)
        self.assertLess(value.risk(features), initial_risk)

    def test_full_covariance_preserves_uncertainty_for_unseen_direction(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(epistemic_scale=1.0)
        )
        value = controller.values[STOP]
        seen = [1.0] + [0.0] * 12
        unseen = [0.0, 1.0] + [0.0] * 11
        for _ in range(100):
            value.update(seen, 0.0, rate=0.0)
        self.assertLess(value.risk(seen), value.risk(unseen))

    def test_stop_to_extension_transition_charges_next_forward(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="td", learning_rate=0.1)
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        current = synthetic_features(1, 3)
        following = synthetic_features(1, 4)
        residual = controller.observe_transition(
            STOP,
            current,
            following,
            2.0,
        )
        self.assertIsNotNone(residual)
        self.assertEqual(controller.values[STOP].sample_count, 1)
        self.assertLess(controller.values[STOP].mean(current), 0.0)

    def test_trajectory_latency_includes_stop_extension_forward(self):
        trajectory = [
            {"action": STOP, "next_forward_latency_ms": 2.0},
            {"action": CONTINUE, "next_forward_latency_ms": 3.0},
            {"action": STOP, "next_forward_latency_ms": 0.0},
        ]
        self.assertEqual(trajectory_forward_latency(trajectory), 5.0)

    def test_mixed_terminal_update_counts_one_factual_observation(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="mixed", warmup_rounds=0)
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        controller.complete_trajectory(
            [{
                "features": synthetic_features(1, 1),
                "action": STOP,
                "next_forward_latency_ms": 0.0,
            }],
            emitted_tokens=2,
            verifier_latency_ms=1.0,
        )
        self.assertEqual(controller.values[STOP].sample_count, 1)

    def test_mixed_fallback_counts_continue_when_td_was_unavailable(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="mixed", warmup_rounds=0)
        )
        controller.complete_trajectory(
            [
                {
                    "features": synthetic_features(1, 2),
                    "action": CONTINUE,
                    "next_forward_latency_ms": 1.0,
                    "td_observation_counted": False,
                },
                {
                    "features": synthetic_features(2, 2),
                    "action": STOP,
                    "next_forward_latency_ms": 0.0,
                },
            ],
            emitted_tokens=3,
            verifier_latency_ms=1.0,
        )
        self.assertEqual(controller.values[CONTINUE].sample_count, 1)
        self.assertEqual(controller.values[STOP].sample_count, 1)

    def test_td_target_masks_unavailable_stop_action(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(warmup_rounds=0, learning_rate=0.1)
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        current = synthetic_features(1, 3)
        following = synthetic_features(2, 3)
        controller.values[STOP].theta[0] = 100.0
        controller.values[CONTINUE].theta[0] = 2.0
        controller.observe_continue_transition(
            current,
            following,
            1.0,
            next_stop_available=False,
        )
        self.assertLess(controller.values[CONTINUE].theta[0], 3.0)

    def test_invalid_action_value_forces_safe_stop(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(warmup_rounds=0, explore_epsilon=0.0, explore_min=0.0)
        )
        controller.values[STOP].theta[0] = float("nan")
        decision = controller.choose(
            synthetic_features(1, 3),
            allow_stop=True,
            refinement_step=1,
        )
        self.assertEqual(decision.action, STOP)
        self.assertEqual(decision.reason, "invalid_numeric_state")

    def test_ratio_of_emas_is_used_for_throughput(self):
        controller = OnlineTDRefinementController(AdaptiveTDConfig(rho_alpha=0.5))
        controller.observe_round(4, 100.0)
        controller.observe_round(8, 200.0)
        self.assertAlmostEqual(controller.y_ema, 6.0)
        self.assertAlmostEqual(controller.t_ema_ms, 150.0)
        self.assertAlmostEqual(controller.rho, 0.04)

    def test_same_controller_learns_depths_one_three_and_six(self):
        for preferred_depth in (1, 3, 6):
            config = AdaptiveTDConfig(
                learning_rate=0.02,
                mc_learning_rate=0.02,
                mc_mix=1.0,
                update_mode="factual_return",
                risk_beta=0.0,
                explore_epsilon=0.2,
                explore_min=0.05,
                explore_decay=0.999,
                warmup_rounds=0,
                early_stop_min_observations=0,
                max_refinement_steps=8,
                seed=preferred_depth,
            )
            controller = OnlineTDRefinementController(config)
            controller.y_ema = 1.0
            controller.t_ema_ms = 1.0
            for _ in range(2000):
                trajectory = []
                for step in range(1, 9):
                    features = synthetic_features(step, preferred_depth)
                    action = controller.choose(
                        features,
                        allow_stop=True,
                        refinement_step=step,
                    ).action
                    if step == 8:
                        action = STOP
                    trajectory.append({
                        "features": features,
                        "action": action,
                        "next_forward_latency_ms": 1.0 if action == CONTINUE else 0.0,
                    })
                    if action == STOP:
                        break
                emitted = 1 + 4 * min(step, preferred_depth)
                controller.complete_trajectory(
                    trajectory,
                    emitted_tokens=emitted,
                    verifier_latency_ms=0.0,
                )
            learned_depth = next(
                (
                    step
                    for step in range(1, 9)
                    if controller.evaluate(STOP, synthetic_features(step, preferred_depth)).mean
                    > controller.evaluate(
                        CONTINUE,
                        synthetic_features(step, preferred_depth),
                    ).mean
                ),
                8,
            )
            self.assertEqual(learned_depth, preferred_depth)


if __name__ == "__main__":
    unittest.main()
