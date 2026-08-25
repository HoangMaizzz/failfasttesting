import unittest

from adaptive_td import (
    AdaptiveTDConfig,
    OnlineTDRefinementController,
    V21_FEATURE_NAMES,
    V2_FEATURE_NAMES,
    build_v21_state_features,
    build_v2_state_features,
)


class OTRCV2TDTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "feature_dim": len(V2_FEATURE_NAMES),
            "feature_schema": "otrc_v2_td",
            "feature_version": 2,
            "factual_ema_alpha": 0.2,
        }
        values.update(overrides)
        return AdaptiveTDConfig(**values)

    def test_compact_features_use_local_and_global_normalization(self):
        features = build_v2_state_features(
            proposal_length=16,
            max_spec_len=60,
            active_span_length=8,
            active_remaining_masks=4,
            active_newly_unmasked=4,
            prefix_length=10,
            prefix_advance=2,
            newly_unmasked_prefix=1,
            active_remaining_confidences=[0.01, 0.02, 0.03, 0.04],
            failfast_candidate_min_confidence=0.30,
            drafter_threshold=0.05,
            failfast_threshold=0.45,
            factual_draft_latency_ema_ms=None,
            factual_verifier_latency_ema_ms=None,
            factual_tokens_per_verifier_ema=None,
        )
        self.assertEqual(len(features), len(V2_FEATURE_NAMES))
        expected = [
            1.0,
            0.5,
            10.0 / 16.0,
            2.0 / 16.0,
            0.5,
            0.25,
            0.8,
            (0.30 - 0.45) / 0.45,
            16.0 / 60.0,
            1.0,
            1.0 / 61.0,
        ]
        for actual, wanted in zip(features, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_factual_emas_use_every_observation_without_changing_rho(self):
        controller = OnlineTDRefinementController(self.config())
        controller.observe_factual_draft_forward(10.0)
        controller.observe_factual_draft_forward(20.0)
        controller.observe_factual_verifier_call(5, 40.0)
        controller.observe_factual_verifier_call(9, 60.0)
        self.assertAlmostEqual(controller.factual_draft_latency_ema_ms, 12.0)
        self.assertAlmostEqual(controller.factual_verifier_latency_ema_ms, 44.0)
        self.assertAlmostEqual(controller.factual_tokens_per_verifier_ema, 5.8)
        self.assertEqual(controller.rho, 0.0)

    def test_v21_uses_post_commit_proposal_scope_and_signed_gap(self):
        features = build_v21_state_features(
            proposal_length=16,
            max_spec_len=60,
            proposal_remaining_masks=6,
            proposal_remaining_confidences=[0.01, 0.08],
            prefix_length=10,
            prefix_advance=2,
            failfast_candidate_min_confidence=0.30,
            drafter_threshold=0.05,
            failfast_threshold=0.45,
            factual_draft_latency_ema_ms=10.0,
            factual_verifier_latency_ema_ms=40.0,
            factual_tokens_per_verifier_ema=8.0,
        )
        self.assertEqual(len(features), len(V21_FEATURE_NAMES))
        expected = [
            1.0,
            6.0 / 16.0,
            10.0 / 16.0,
            2.0 / 16.0,
            0.8,
            (0.30 - 0.45) / 0.45,
            16.0 / 60.0,
            0.25,
            8.0 / 61.0,
        ]
        for actual, wanted in zip(features, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_v21_confidence_gap_preserves_above_threshold_sign(self):
        features = build_v21_state_features(
            proposal_length=8,
            max_spec_len=60,
            proposal_remaining_masks=1,
            proposal_remaining_confidences=[0.075],
            prefix_length=7,
            prefix_advance=1,
            failfast_candidate_min_confidence=0.45,
            drafter_threshold=0.05,
            failfast_threshold=0.45,
            factual_draft_latency_ema_ms=None,
            factual_verifier_latency_ema_ms=None,
            factual_tokens_per_verifier_ema=None,
        )
        gap_index = V21_FEATURE_NAMES.index("min_confidence_gap")
        self.assertAlmostEqual(features[gap_index], -0.5)

    def test_snapshot_rejects_cross_schema_loading(self):
        v2 = OnlineTDRefinementController(self.config())
        snapshot = v2.snapshot()
        v1 = OnlineTDRefinementController(AdaptiveTDConfig())
        with self.assertRaisesRegex(ValueError, "feature schema"):
            v1.load_snapshot(snapshot)

    def test_v21_snapshot_cannot_load_into_v2(self):
        v21 = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=len(V21_FEATURE_NAMES),
            feature_schema="otrc_v2_1_td",
            feature_version=21,
        ))
        v2 = OnlineTDRefinementController(self.config())
        with self.assertRaisesRegex(ValueError, "feature schema"):
            v2.load_snapshot(v21.snapshot())

    def test_snapshot_restores_v2_factual_state(self):
        original = OnlineTDRefinementController(self.config())
        original.observe_factual_draft_forward(7.0)
        original.observe_factual_verifier_call(4, 21.0)
        restored = OnlineTDRefinementController(self.config())
        restored.load_snapshot(original.snapshot())
        self.assertEqual(restored.feature_names, V2_FEATURE_NAMES)
        self.assertAlmostEqual(restored.factual_draft_latency_ema_ms, 7.0)
        self.assertAlmostEqual(restored.factual_verifier_latency_ema_ms, 21.0)
        self.assertAlmostEqual(restored.factual_tokens_per_verifier_ema, 4.0)


if __name__ == "__main__":
    unittest.main()
