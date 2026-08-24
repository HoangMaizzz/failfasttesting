import unittest

from adaptive_td import (
    FEATURE_NAMES,
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
        self.assertEqual(config.stop_probability_threshold, 0.75)

    def test_calibration_blocks_policy_exploitation_until_early_stop_data_exists(self):
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
        self.assertEqual(decision.reason, "stop_probability_threshold")
        self.assertFalse(decision.calibration_active)
        self.assertGreater(decision.stop_probability, 0.75)

    def test_frozen_decision_can_disable_exploration(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                warmup_rounds=0,
                early_stop_min_observations=10,
                explore_epsilon=1.0,
                explore_min=1.0,
            )
        )
        decision = controller.choose(
            synthetic_features(1, 3),
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
        )
        self.assertEqual(decision.action, CONTINUE)
        self.assertFalse(decision.exploration_used)

    def test_snapshot_round_trip_restores_predictions_and_runtime_state(self):
        original = OnlineTDRefinementController(AdaptiveTDConfig())
        features = synthetic_features(2, 3)
        original.values[STOP].update(features, 3.0, rate=0.1)
        original.values[CONTINUE].update(features, 1.0, rate=0.1)
        original.early_stop_uncertainty.update(features, 2.0, rate=0.0)
        original.y_ema = 4.0
        original.t_ema_ms = 20.0
        original.completed_rounds = 12
        original.early_stop_observations = 7

        restored = OnlineTDRefinementController(AdaptiveTDConfig())
        restored.load_snapshot(original.snapshot())

        self.assertAlmostEqual(
            restored.values[STOP].mean(features),
            original.values[STOP].mean(features),
        )
        self.assertAlmostEqual(restored.rho, original.rho)
        self.assertEqual(restored.completed_rounds, 12)
        self.assertEqual(restored.early_stop_observations, 7)

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
        self.assertEqual(controller.early_stop_uncertainty.sample_count, 0)

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
        self.assertEqual(controller.early_stop_uncertainty.sample_count, 1)

    def test_invalid_stop_probability_threshold_is_rejected(self):
        for threshold in (0.5, 1.0):
            with self.assertRaises(ValueError):
                AdaptiveTDConfig(stop_probability_threshold=threshold)

    def test_features_are_fixed_width_and_normalized(self):
        features = synthetic_features(2, 3)
        self.assertEqual(len(features), 13)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features))

    def test_disabled_features_are_zeroed_without_changing_width(self):
        disabled = ("mean_confidence", "frontier_ratio", "refinement_step")
        features = build_state_features(
            proposal_length=8,
            remaining_masks=4,
            newly_unmasked=4,
            recoverable_confidences=[0.2, 0.4, 0.6, 0.8],
            recoverable_margins=[0.1, 0.2, 0.3, 0.4],
            first_remaining_position=4,
            frontier_length=5,
            proposal_change_ratio=0.25,
            recoverable_change_ratio=0.5,
            refinement_step=2,
            max_refinement_steps=16,
            disabled_features=disabled,
        )
        self.assertEqual(len(features), len(FEATURE_NAMES))
        for name in disabled:
            self.assertEqual(features[FEATURE_NAMES.index(name)], 0.0)
        self.assertGreater(features[FEATURE_NAMES.index("mean_margin")], 0.0)

    def test_bias_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            AdaptiveTDConfig(disabled_features=("bias",))

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

    def test_terminal_stop_residuals_do_not_control_early_stop_risk(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(epistemic_scale=1.0)
        )
        features = synthetic_features(1, 3)
        for index in range(100):
            controller.values[STOP].update(
                features,
                100.0 if index % 2 else -100.0,
                rate=0.0,
            )
        decision = controller.choose(
            features,
            allow_stop=False,
            refinement_step=1,
        )
        self.assertAlmostEqual(
            decision.stop.risk,
            controller.early_stop_uncertainty.risk(features),
        )
        self.assertEqual(controller.early_stop_uncertainty.sample_count, 0)

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

    def test_terminal_verification_does_not_train_early_stop_value(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(update_mode="mixed", warmup_rounds=0)
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        controller.complete_trajectory(
            [{
                "features": synthetic_features(1, 1),
                "action": STOP,
                "reason": "factual_terminal_verification",
                "next_forward_latency_ms": 0.0,
            }],
            emitted_tokens=2,
            verifier_latency_ms=1.0,
        )
        self.assertEqual(controller.values[STOP].sample_count, 0)

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

    def test_symmetric_cold_start_samples_both_actions_without_default(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                policy_mode="symmetric",
                min_action_probability=0.1,
                warmup_rounds=100,
                early_stop_min_observations=100,
                seed=42,
            )
        )
        decisions = [
            controller.choose(
                synthetic_features(1, 3),
                allow_stop=True,
                refinement_step=1,
            )
            for _ in range(1000)
        ]
        stop_rate = sum(item.action == STOP for item in decisions) / len(decisions)
        self.assertGreater(stop_rate, 0.45)
        self.assertLess(stop_rate, 0.55)
        self.assertTrue(
            all(item.reason == "symmetric_posterior_sample" for item in decisions)
        )
        self.assertTrue(
            all(item.behavior_stop_probability == 0.5 for item in decisions)
        )

    def test_symmetric_frozen_policy_is_greedy_and_deterministic(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(policy_mode="symmetric")
        )
        features = synthetic_features(1, 3)
        controller.values[STOP].theta[0] = 2.0
        controller.values[CONTINUE].theta[0] = 1.0
        decisions = [
            controller.choose(
                features,
                allow_stop=True,
                refinement_step=1,
                allow_exploration=False,
            )
            for _ in range(10)
        ]
        self.assertTrue(all(item.action == STOP for item in decisions))
        self.assertTrue(all(item.selected_action_probability == 1.0 for item in decisions))
        self.assertTrue(all(not item.exploration_used for item in decisions))

    def test_symmetric_policy_mirrors_stop_and_continue_values(self):
        features = synthetic_features(1, 3)
        first = OnlineTDRefinementController(
            AdaptiveTDConfig(policy_mode="symmetric")
        )
        second = OnlineTDRefinementController(
            AdaptiveTDConfig(policy_mode="symmetric")
        )
        first.values[STOP].theta[0] = 1.0
        first.values[CONTINUE].theta[0] = -1.0
        second.values[STOP].theta[0] = -1.0
        second.values[CONTINUE].theta[0] = 1.0
        stop_preferred = first.choose(
            features,
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
        )
        continue_preferred = second.choose(
            features,
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
        )
        self.assertEqual(stop_preferred.action, STOP)
        self.assertEqual(continue_preferred.action, CONTINUE)
        self.assertAlmostEqual(
            stop_preferred.stop_probability,
            1.0 - continue_preferred.stop_probability,
        )

    def test_symmetric_importance_weight_is_capped(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(
                policy_mode="symmetric",
                min_action_probability=0.1,
                max_importance_weight=5.0,
                seed=31,
            )
        )
        features = synthetic_features(1, 3)
        controller.values[STOP].theta[0] = -100.0
        controller.values[CONTINUE].theta[0] = 100.0
        decision = controller.choose(
            features,
            allow_stop=True,
            refinement_step=1,
        )
        self.assertEqual(decision.action, STOP)
        self.assertEqual(decision.selected_action_probability, 0.1)
        self.assertEqual(decision.importance_weight, 5.0)

    def test_weighted_feedback_updates_effective_sample_mass(self):
        controller = OnlineTDRefinementController(
            AdaptiveTDConfig(policy_mode="symmetric", update_mode="td")
        )
        controller.y_ema = 1.0
        controller.t_ema_ms = 1.0
        features = synthetic_features(1, 3)
        controller.observe_transition(
            CONTINUE,
            features,
            synthetic_features(2, 3),
            1.0,
            action_probability=0.2,
        )
        value = controller.values[CONTINUE]
        self.assertEqual(value.sample_count, 1)
        self.assertEqual(value.sample_weight_sum, 5.0)


if __name__ == "__main__":
    unittest.main()
