import unittest

from bucket_renewal import (
    compare_renewal_costs,
    expected_accepted_prefix,
    predict_next_gain,
)


class BucketRenewalTests(unittest.TestCase):
    def test_expected_prefix_uses_left_to_right_survival(self):
        self.assertAlmostEqual(
            expected_accepted_prefix([0.5, 0.5, 0.5]),
            0.5 + 0.25 + 0.125,
        )

    def test_gain_forecast_decays_when_recent_improvement_shrinks(self):
        self.assertAlmostEqual(predict_next_gain([2.0, 4.0, 5.0]), 0.5)

    def test_first_step_uses_online_bucket_estimate(self):
        self.assertAlmostEqual(
            predict_next_gain([2.0], first_step_estimate=0.4),
            0.4,
        )

    def test_renewal_cost_continues_when_gain_avoids_verifier_rounds(self):
        decision = compare_renewal_costs(
            elapsed_draft_ms=12.0,
            next_draft_ms=4.0,
            verify_round_ms=20.0,
            controller_ms=1.0,
            expected_prefix=3.0,
            predicted_gain=1.0,
        )
        self.assertTrue(decision.should_continue)
        self.assertLess(decision.continue_ms_per_output, decision.stop_ms_per_output)

    def test_renewal_cost_stops_when_gain_does_not_pay_for_pass(self):
        decision = compare_renewal_costs(
            elapsed_draft_ms=12.0,
            next_draft_ms=8.0,
            verify_round_ms=8.0,
            controller_ms=1.0,
            expected_prefix=5.0,
            predicted_gain=0.1,
        )
        self.assertFalse(decision.should_continue)

    def test_first_step_bootstraps_when_gain_bucket_is_empty(self):
        decision = compare_renewal_costs(
            elapsed_draft_ms=5.0,
            next_draft_ms=5.0,
            verify_round_ms=10.0,
            controller_ms=0.0,
            expected_prefix=2.0,
            predicted_gain=None,
        )
        self.assertTrue(decision.should_continue)
        self.assertIsNone(decision.continue_ms_per_output)


if __name__ == "__main__":
    unittest.main()
