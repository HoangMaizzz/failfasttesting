import unittest

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController


def make_controller(**overrides):
    values = dict(
        feature_dim=6,
        feature_schema="otrc_v2_2_compact_td",
        credit_assignment="hindsight_delta_j_f5",
        policy_mode="hindsight_delta_j_f5",
        hindsight_delta_j_min_pairs=0,
        hindsight_delta_j_min_continue_pairs=0,
        hindsight_delta_j_structural_probe_probability=0.0,
        hindsight_delta_j_floor_probe_probability=0.0,
    )
    values.update(overrides)
    return OnlineTDRefinementController(AdaptiveTDConfig(**values))


def f5(mask_count=3):
    return {
        "active_span_size": 8,
        "current_mask_count": mask_count,
        "masked_entropy_std": 0.2,
        "resolved_margin_mean": 0.4,
        "resolved_entropy_max": 0.6,
    }


class HindsightDeltaJF5Test(unittest.TestCase):
    def prepare(self, item, candidate, step, eligible=True, mask_count=3):
        return item.prepare_hindsight_snapshot(
            draft_proposal=candidate,
            context_len=100,
            active_block_start=100,
            active_block_end=108,
            raw_current_state=None,
            f5_state=f5(mask_count),
            proposal_length=8,
            max_spec_len=64,
            refinement_step=step,
            next_forward_latency_ms=4.0,
            forward_pass_index=step,
            decision_eligible=eligible,
        )

    def pair(self, item, before, after, latency_ms=4.0):
        item.begin_hindsight_problem(1)
        item.hindsight_gain_model.weights[0] = -10.0
        self.prepare(item, before, 1)
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        self.prepare(item, after, 2, eligible=False)
        item.hindsight_pending_pairs[-1]["next_forward_latency_ms"] = latency_ms

    def test_f5_exact_values(self):
        item = make_controller()
        item.factual_tokens_per_verifier_ema = 7.5
        features = item._hindsight_delta_j_f5_features(f5(mask_count=4))
        self.assertEqual(features, (1.0, 0.5, 0.2, 0.4, 0.6, 7.5))

    def test_yield_gain_can_still_be_stop_better(self):
        item = make_controller()
        target = list(range(8))
        before = [0, 1, 99, 3, 4, 5, 6, 7]  # Y_S=3
        after = [0, 1, 2, 99, 4, 5, 6, 7]   # Y_C=4
        self.pair(item, before, after, latency_ms=20.0)
        item.observe_hindsight_verifier_boundary(
            target, verifier_latency_ms=10.0, post_verify_latency_ms=0.0,
            terminal=False,
        )
        row = item.full_stream_transitions[-1]
        self.assertGreater(row["delta_J_ms_per_token"], 0.0)
        self.assertEqual(row["true_action_from_delta_J"], "stop")

    def test_clearly_beneficial_continue(self):
        item = make_controller()
        target = list(range(8))
        before = [99] + target[1:]                 # Y_S=1
        after = target[:5] + [99] + target[6:]    # Y_C=6
        self.pair(item, before, after, latency_ms=2.0)
        item.observe_hindsight_verifier_boundary(
            target, verifier_latency_ms=20.0, post_verify_latency_ms=5.0,
            terminal=False,
        )
        row = item.full_stream_transitions[-1]
        self.assertEqual(row["verifier_boundary_latency_ms_T_B"], 25.0)
        self.assertLess(row["delta_J_ms_per_token"], 0.0)
        self.assertEqual(row["true_action_from_delta_J"], "continue")

    def test_continue_pass_lower_bound_is_censored_when_not_enough(self):
        item = make_controller()
        target = list(range(8))
        before = target[:7] + [99]  # Y_S=8, same capped yield as pass
        after = target
        self.pair(item, before, after, latency_ms=2.0)
        item.observe_hindsight_verifier_boundary(
            target, verifier_latency_ms=20.0, terminal=False,
        )
        self.assertEqual(item.hindsight_resolved_count, 0)
        self.assertEqual(item.hindsight_censored_count, 1)

    def test_both_pass_is_structural_stop(self):
        item = make_controller()
        target = list(range(8))
        self.pair(item, target, target, latency_ms=2.0)
        item.observe_hindsight_verifier_boundary(
            target, verifier_latency_ms=20.0, terminal=False,
        )
        row = item.full_stream_transitions[-1]
        self.assertEqual(row["label_reason"], "both_pass_active_block")
        self.assertGreater(row["normalized_delta_J"], 0.0)

    def test_allow_exploration_false_disables_probe(self):
        item = make_controller(
            hindsight_delta_j_structural_probe_probability=1.0,
        )
        item.hindsight_gain_model.weights[0] = 1.0
        item.begin_hindsight_problem(2)
        self.prepare(item, list(range(8)), 1)
        decision = item.choose(
            (1.0,) * 6,
            allow_stop=True,
            refinement_step=1,
            allow_exploration=False,
        )
        self.assertEqual(decision.action, "stop")
        self.assertFalse(decision.exploration_used)
        self.assertEqual(item.hindsight_probe_count, 0)

    def test_f2_uses_only_mask_ratio_and_first_unresolved_position(self):
        item = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_delta_j_f2",
            policy_mode="hindsight_delta_j_f2",
        ))
        features = item._hindsight_delta_j_f2_features({
            "active_span_size": 8,
            "current_mask_count": 4,
            "first_unresolved_position": 3 / 7,
        })
        self.assertEqual(features, (1.0, 0.5, 3 / 7))
        self.assertEqual(
            item.hindsight_feature_names,
            ("bias", "current_mask_ratio", "first_unresolved_position"),
        )

    def test_logistic_f2_uses_global_proposal_position(self):
        item = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_delta_j_logistic_f2",
            policy_mode="hindsight_delta_j_logistic_f2",
        ))
        features = item._hindsight_delta_j_logistic_f2_features(
            {
                "active_span_size": 8,
                "current_mask_count": 4,
            },
            active_block_start_relative=8,
            proposal_length=32,
        )
        self.assertEqual(features, (1.0, 0.5, 0.25))
        self.assertEqual(
            item.hindsight_feature_names,
            ("bias", "current_mask_ratio", "global_proposal_position"),
        )

    def test_logistic_update_moves_continue_score_in_label_direction(self):
        item = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_delta_j_logistic_f2",
            policy_mode="hindsight_delta_j_logistic_f2",
        ))
        features = (1.0, 0.5, 0.25)
        before = item.hindsight_logistic_model.predict(features)[1]
        item.hindsight_logistic_model.update(features, 1, 1.0)
        after = item.hindsight_logistic_model.predict(features)[1]
        self.assertGreater(after, before)

    def test_logistic_tie_does_not_update(self):
        item = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_delta_j_logistic_f2",
            policy_mode="hindsight_delta_j_logistic_f2",
            hindsight_delta_j_min_pairs=30,
            hindsight_delta_j_min_continue_pairs=3,
            hindsight_logistic_tie_ms_per_token=1.0,
        ))
        state = {"active_span_size": 8, "current_mask_count": 4}
        target = list(range(8))
        item.begin_hindsight_problem(7)
        item.prepare_hindsight_snapshot(
            draft_proposal=target, context_len=100,
            active_block_start=100, active_block_end=108,
            raw_current_state=None, f2_state=state,
            proposal_length=8, max_spec_len=64, refinement_step=1,
            next_forward_latency_ms=2.0, forward_pass_index=1,
            decision_eligible=True,
        )
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        item.prepare_hindsight_snapshot(
            draft_proposal=target, context_len=100,
            active_block_start=100, active_block_end=108,
            raw_current_state=None, f2_state=state,
            proposal_length=8, max_spec_len=64, refinement_step=2,
            next_forward_latency_ms=2.0, forward_pass_index=2,
            decision_eligible=False,
        )
        item.hindsight_pending_pairs[-1]["next_forward_latency_ms"] = 2.0
        item.observe_hindsight_verifier_boundary(
            target, verifier_latency_ms=20.0, terminal=False,
        )
        row = item.full_stream_transitions[-1]
        self.assertTrue(row["is_tie"])
        self.assertFalse(row["update_applied"])
        self.assertEqual(item.hindsight_logistic_model.sample_count, 0)


if __name__ == "__main__":
    unittest.main()
