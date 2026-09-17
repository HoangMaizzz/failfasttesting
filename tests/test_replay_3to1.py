import unittest

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController


class ReplayRatioTest(unittest.TestCase):
    def resolve(self, ratio, initial_labels):
        controller = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6, feature_schema="otrc_v2_2_compact_td",
            credit_assignment="hindsight_delta_j_logistic_f2",
            policy_mode="hindsight_delta_j_logistic_f2",
            hindsight_logistic_utility_weighting="raw_abs",
            hindsight_logistic_replay_batch_size=16,
            hindsight_logistic_replay_buffer_size=100,
            hindsight_logistic_replay_stop_to_continue_ratio=ratio,
        ))
        controller.hindsight_logistic_replay_buffer = [
            ([1.0, 0.5, 0.0], label, 2.0) for label in initial_labels
        ]
        controller.begin_hindsight_problem(1)
        target = list(range(8))
        for step, candidate in enumerate([
            [99] + target[1:], target[:5] + [99] + target[6:],
        ], start=1):
            controller.prepare_hindsight_snapshot(
                draft_proposal=candidate, context_len=100,
                active_block_start=100, active_block_end=108,
                raw_current_state=None, proposal_length=8, max_spec_len=64,
                refinement_step=step, next_forward_latency_ms=2.0,
                forward_pass_index=step, decision_eligible=step == 1,
                remaining_masks=4,
            )
            if step == 1:
                controller.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
        controller.observe_hindsight_verifier_boundary(target, verifier_latency_ms=20.0, terminal=False)
        return controller, controller.full_stream_transitions[-1]

    def test_ratio_and_single_update_with_rare_continue(self):
        controller, row = self.resolve(3.0, [0] * 100)
        self.assertEqual(row["replay_stop_count_used"], 12)
        self.assertEqual(row["replay_continue_count_used"], 4)
        self.assertEqual(controller.hindsight_logistic_model.sample_count, 1)
        self.assertEqual(len(controller.hindsight_logistic_replay_buffer), 100)
        self.assertAlmostEqual(row["sample_weight"], abs(row["delta_J_ms_per_token"]))

    def test_single_class_falls_back_without_fabricating_labels(self):
        controller, row = self.resolve(3.0, [1] * 20)
        self.assertEqual(row["replay_stop_count_used"], 0)
        self.assertEqual(row["replay_continue_count_used"], 16)
        self.assertEqual(controller.hindsight_logistic_model.sample_count, 1)

    def test_zero_ratio_keeps_uniform_sampling_without_repetition(self):
        _, row = self.resolve(0.0, [0] * 20)
        self.assertLessEqual(row["replay_continue_count_used"], 1)
        self.assertEqual(row["replay_batch_size_used"], 16)


if __name__ == "__main__":
    unittest.main()
