import unittest

from frontier_latency import (
    create_verify_latency_model,
    estimate_verify_latency_ms,
    update_verify_latency_model,
)


class VerifyLatencyModelTest(unittest.TestCase):
    def test_uses_exact_context_and_proposal_bucket(self):
        model = create_verify_latency_model(256, 8)
        update_verify_latency_model(model, 300, 10, 21.0)
        estimate = estimate_verify_latency_ms(model, 400, 15, 13.5)
        self.assertEqual(estimate["source"], "joint_exact")
        self.assertEqual(estimate["samples"], 1)
        self.assertAlmostEqual(estimate["latency_ms"], 21.0)

    def test_interpolates_proposal_latency_without_linear_token_cost(self):
        model = create_verify_latency_model(256, 8)
        update_verify_latency_model(model, 300, 8, 20.0)
        update_verify_latency_model(model, 300, 24, 24.0)
        estimate = estimate_verify_latency_ms(model, 300, 16, 13.5)
        self.assertEqual(estimate["source"], "context_interpolation")
        self.assertAlmostEqual(estimate["latency_ms"], 22.0)

    def test_updates_bucket_with_ema(self):
        model = create_verify_latency_model(256, 8)
        update_verify_latency_model(model, 300, 8, 20.0, alpha=0.2)
        update_verify_latency_model(model, 300, 8, 30.0, alpha=0.2)
        estimate = estimate_verify_latency_ms(model, 300, 8, 13.5)
        self.assertEqual(estimate["samples"], 2)
        self.assertAlmostEqual(estimate["latency_ms"], 22.0)

    def test_uses_proposal_profile_when_context_has_only_one_length(self):
        model = create_verify_latency_model(256, 8)
        update_verify_latency_model(model, 300, 8, 20.0)
        update_verify_latency_model(model, 700, 16, 24.0)
        estimate = estimate_verify_latency_ms(model, 300, 16, 13.5)
        self.assertEqual(estimate["source"], "proposal_interpolation")
        self.assertAlmostEqual(estimate["latency_ms"], 24.0)

    def test_uses_fallback_before_observations(self):
        model = create_verify_latency_model(256, 8)
        estimate = estimate_verify_latency_ms(model, 300, 8, 17.0)
        self.assertEqual(estimate["source"], "fallback")
        self.assertEqual(estimate["latency_ms"], 17.0)


if __name__ == "__main__":
    unittest.main()
