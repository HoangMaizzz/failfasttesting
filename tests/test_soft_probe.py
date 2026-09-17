import unittest

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController
from hindsight_soft_probe import soft_probe_probability


class SoftProbeTest(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(soft_probe_probability(0, 7, 8)[0], 0.24)
        self.assertAlmostEqual(soft_probe_probability(1, 7, 8)[0], 0.162146739, places=8)
        self.assertGreater(soft_probe_probability(1, 6, 8)[0], soft_probe_probability(2, 6, 8)[0])
        self.assertGreater(soft_probe_probability(1, 6, 8)[0], soft_probe_probability(1, 2, 8)[0])
        self.assertEqual(soft_probe_probability(1, 0, 1)[0], 0.08)

    def test_decision_uses_soft_probability_and_keeps_floor(self):
        for active_masks, expected in [(2, 0.24), (1, 0.02)]:
            controller = OnlineTDRefinementController(AdaptiveTDConfig(
                feature_dim=6, feature_schema="otrc_v2_2_compact_td",
                credit_assignment="hindsight_delta_j_logistic_f2",
                policy_mode="hindsight_delta_j_logistic_f2",
                hindsight_soft_probe=True,
                hindsight_logistic_continue_threshold=0.53,
                hindsight_delta_j_min_pairs=0,
                hindsight_delta_j_min_continue_pairs=0,
                hindsight_logistic_min_positive_problems=0,
            ))
            controller.begin_hindsight_problem(1)
            controller.prepare_hindsight_snapshot(
                draft_proposal=list(range(8)), context_len=100,
                active_block_start=100, active_block_end=108,
                raw_current_state=None, proposal_length=8,
                max_spec_len=64, refinement_step=1,
                next_forward_latency_ms=2, forward_pass_index=1,
                decision_eligible=True, remaining_masks=active_masks,
                probe_state=dict(prefix_length=0, remaining_masks=7, proposal_length=8),
            )
            decision = controller.choose((1.0,) * 6, allow_stop=True, refinement_step=1)
            self.assertAlmostEqual(decision.diagnostics["probe_probability"], expected)
            self.assertEqual(decision.diagnostics["model_action"], "stop")
            self.assertEqual(len(controller.hindsight_logistic_model.weights), 3)


if __name__ == "__main__":
    unittest.main()
