import unittest

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController


def raw_state(masked=8, top1=0.6, top2=0.2, entropy=0.5, span=8, offset=0):
    return [
        [
            float(index < masked),
            1.0,
            float(top1),
            float(top2),
            float(entropy),
            (offset + index) / 7.0,
        ]
        for index in range(span)
    ]


def controller():
    item = OnlineTDRefinementController(
        AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_block_gain",
            policy_mode="hindsight_gain",
            early_stop_min_observations=2,
            hindsight_confidence_kappa=0.0,
        )
    )
    item.hindsight_gain_model.weights[0] = 1.0
    return item


def probe_controller(probability=1.0):
    return OnlineTDRefinementController(
        AdaptiveTDConfig(
            feature_dim=6,
            feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_block_gain",
            policy_mode="hindsight_gain",
            early_stop_min_observations=32,
            hindsight_probe_initial=probability,
            hindsight_probe_floor=probability,
            hindsight_probe_max_fraction=1.0,
        )
    )


class HindsightBlockGainTest(unittest.TestCase):
    def prepare(self, item, proposal, state, step, eligible=True):
        return item.prepare_hindsight_snapshot(
            draft_proposal=proposal,
            context_len=100,
            active_block_start=100,
            active_block_end=108,
            raw_current_state=state,
            proposal_length=8,
            max_spec_len=64,
            refinement_step=step,
            next_forward_latency_ms=2.0,
            forward_pass_index=step,
            decision_eligible=eligible,
        )

    def test_delayed_lcp_labels_one_token_gain(self):
        item = controller()
        item.begin_hindsight_problem(7)
        self.prepare(item, list(range(1, 9)), raw_state(masked=5), 1)
        decision = item.choose(
            (1.0,) * 6, allow_stop=True, refinement_step=1
        )
        self.assertEqual(decision.action, "continue")
        self.prepare(item, list(range(1, 8)) + [99], raw_state(masked=2), 2)

        item.observe_hindsight_verifier_boundary(
            list(range(1, 8)) + [99], terminal=False
        )

        self.assertEqual(item.hindsight_resolved_count, 1)
        row = item.full_stream_transitions[-1]
        self.assertEqual(row["before_active_yield"], 7)
        self.assertEqual(row["after_active_yield"], 8)
        self.assertEqual(row["gain_tokens"], 1)
        self.assertEqual(item.hindsight_gain_model.sample_count, 1)

    def test_partial_physical_span_uses_its_own_gain_scale(self):
        item = controller()
        item.begin_hindsight_problem(70)
        before = [1, 2, 3, 4, 9]
        after = [1, 2, 3, 4, 5]

        item.prepare_hindsight_snapshot(
            draft_proposal=before,
            context_len=103,
            active_block_start=103,
            active_block_end=108,
            raw_current_state=raw_state(masked=3, span=5, offset=3),
            proposal_length=5,
            max_spec_len=64,
            refinement_step=1,
            next_forward_latency_ms=2.0,
            forward_pass_index=1,
            decision_eligible=True,
        )
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        item.prepare_hindsight_snapshot(
            draft_proposal=after,
            context_len=103,
            active_block_start=103,
            active_block_end=108,
            raw_current_state=raw_state(masked=1, span=5, offset=3),
            proposal_length=5,
            max_spec_len=64,
            refinement_step=2,
            next_forward_latency_ms=2.0,
            forward_pass_index=2,
            decision_eligible=False,
        )
        item.observe_hindsight_verifier_boundary(after, terminal=False)

        row = item.full_stream_transitions[-1]
        self.assertEqual(row["active_span_size"], 5)
        self.assertEqual(row["before_active_yield"], 4)
        self.assertEqual(row["after_active_yield"], 5)
        self.assertAlmostEqual(row["normalized_target"], 0.2)
        self.assertAlmostEqual(
            row["active_span_ratio"], 5 / 8
        )

    def test_equal_full_yield_has_zero_gain(self):
        item = controller()
        item.begin_hindsight_problem(8)
        proposal = list(range(10, 18))
        self.prepare(item, proposal, raw_state(masked=4), 1)
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        self.prepare(item, proposal, raw_state(masked=1), 2)
        item.observe_hindsight_verifier_boundary(proposal, terminal=False)

        row = item.full_stream_transitions[-1]
        self.assertEqual(row["before_active_yield"], 8)
        self.assertEqual(row["after_active_yield"], 8)
        self.assertEqual(row["gain_tokens"], 0)

    def test_candidate_waits_across_verifier_boundaries(self):
        item = controller()
        item.begin_hindsight_problem(9)
        before = list(range(20, 28))
        after = list(range(20, 27)) + [88]
        self.prepare(item, before, raw_state(masked=5), 1)
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        self.prepare(item, after, raw_state(masked=2), 2)

        item.observe_hindsight_verifier_boundary(after[:4], terminal=False)
        self.assertEqual(item.hindsight_resolved_count, 0)
        self.assertEqual(len(item.hindsight_pending_pairs), 1)
        item.observe_hindsight_verifier_boundary(after[4:], terminal=False)
        self.assertEqual(item.hindsight_resolved_count, 1)

    def test_interleaved_blocks_keep_independent_pending_pairs(self):
        item = controller()
        item.begin_hindsight_problem(10)
        proposal = list(range(1, 17))

        def prepare_block(start, step, forward):
            item.prepare_hindsight_snapshot(
                draft_proposal=proposal,
                context_len=100,
                active_block_start=100 + start,
                active_block_end=108 + start,
                raw_current_state=raw_state(masked=4),
                proposal_length=16,
                max_spec_len=64,
                refinement_step=step,
                next_forward_latency_ms=2.0,
                forward_pass_index=forward,
                decision_eligible=True,
            )

        prepare_block(0, 1, 1)
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        prepare_block(8, 1, 2)
        item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        prepare_block(0, 2, 3)
        self.assertEqual(len(item.hindsight_pending_pairs), 1)
        self.assertEqual(len(item.hindsight_pending_sources), 1)
        prepare_block(8, 2, 4)
        self.assertEqual(len(item.hindsight_pending_pairs), 2)
        self.assertEqual(len(item.hindsight_pending_sources), 0)

    def test_cold_start_no_longer_forces_continue(self):
        item = probe_controller(probability=0.0)
        item.begin_hindsight_problem(11)
        self.prepare(item, list(range(1, 9)), raw_state(masked=4), 1)

        decision = item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)

        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.reason, "hindsight_gain_not_worth_cost")
        self.assertTrue(decision.calibration_active)
        self.assertFalse(decision.exploration_used)

    def test_uncertain_stop_can_become_forced_continue(self):
        item = probe_controller(probability=1.0)
        item.begin_hindsight_problem(12)
        self.prepare(item, list(range(1, 9)), raw_state(masked=4), 1)

        decision = item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)

        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.reason, "hindsight_uncertainty_probe")
        self.assertTrue(decision.exploration_used)
        self.assertEqual(decision.diagnostics["greedy_action"], "stop")
        self.assertEqual(decision.selected_action_probability, 1.0)
        self.assertEqual(item.hindsight_probe_count, 1)
        self.assertTrue(item.hindsight_probe_outstanding)

    def test_probe_cannot_repeat_before_feedback(self):
        item = probe_controller(probability=1.0)
        item.begin_hindsight_problem(13)
        proposal = list(range(1, 9))
        self.prepare(item, proposal, raw_state(masked=4), 1)
        first = item.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        self.assertEqual(first.reason, "hindsight_uncertainty_probe")

        self.prepare(item, proposal, raw_state(masked=2), 2)
        second = item.choose((1.0,) * 6, allow_stop=True, refinement_step=2)

        self.assertEqual(second.action, "stop")
        self.assertEqual(second.reason, "hindsight_gain_not_worth_cost")
        self.assertFalse(second.exploration_used)
        self.assertEqual(item.hindsight_probe_count, 1)


if __name__ == "__main__":
    unittest.main()
