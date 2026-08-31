import math
import unittest

from adaptive_nonlinear import OnlineNonlinearVA
from adaptive_td import (
    CONTINUE,
    RAW_STATE_FEATURE_NAMES,
    STOP,
    build_raw_state_features,
)


def snapshot(masked=1.0, shift=0.0):
    return [
        [
            masked if index >= 4 else 0.0,
            1.0,
            0.6 + shift,
            0.2,
            0.5,
            index / 7.0,
        ]
        for index in range(8)
    ]


class RawStateControllerTests(unittest.TestCase):
    def test_raw_feature_shape_and_global_state(self):
        current = snapshot(shift=0.05)
        features = build_raw_state_features(
            raw_previous_state=snapshot(),
            raw_current_state=current,
            has_previous_state=True,
            proposal_length=16,
            max_spec_len=64,
            refinement_step=2,
            max_refinement_steps=16,
            factual_draft_latency_ema_ms=4.0,
            factual_verifier_latency_ema_ms=10.0,
            factual_tokens_per_verifier_ema=8.0,
        )
        self.assertEqual(len(features), len(RAW_STATE_FEATURE_NAMES))
        self.assertEqual(len(features), 101)
        self.assertEqual(features[1], 1.0)
        self.assertEqual(features[-5], 0.25)
        self.assertEqual(features[-4], 0.4)
        self.assertEqual(features[-1], 1.0)
        self.assertTrue(all(math.isfinite(value) for value in features))

    def test_raw_feature_rejects_partial_block(self):
        with self.assertRaisesRegex(ValueError, "8 positions"):
            build_raw_state_features(
                raw_previous_state=snapshot()[:-1],
                raw_current_state=snapshot(),
                has_previous_state=False,
                proposal_length=8,
                max_spec_len=64,
                refinement_step=1,
                max_refinement_steps=16,
            )

    def test_raw_models_start_neutral_and_update_selected_action(self):
        features = build_raw_state_features(
            raw_previous_state=snapshot(),
            raw_current_state=snapshot(shift=0.05),
            has_previous_state=True,
            proposal_length=8,
            max_spec_len=64,
            refinement_step=1,
            max_refinement_steps=16,
        )
        for model_type in ("raw_linear", "raw_mlp"):
            learner = OnlineNonlinearVA(
                model_type,
                learning_rate=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                huber_delta=32.0,
                seed=42,
                feature_dim=len(features),
            )
            value, advantage, q_stop, q_continue = learner.predict(features)
            self.assertEqual((value, advantage, q_stop, q_continue), (0.0,) * 4)
            update = learner.update(STOP, features, target=100.0, weight=2.0)
            self.assertEqual(update["clipped_residual"], 32.0)
            self.assertLessEqual(update["gradient_norm"], 1000.0)
            _, advantage, q_stop, q_continue = learner.predict(features)
            self.assertGreater(advantage, 0.0)
            self.assertGreater(q_stop, q_continue)
            snapshot_state = learner.snapshot()
            restored = OnlineNonlinearVA(
                model_type,
                learning_rate=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                huber_delta=32.0,
                seed=7,
                feature_dim=len(features),
            )
            restored.load_snapshot(snapshot_state)
            self.assertEqual(restored.predict(features), learner.predict(features))


if __name__ == "__main__":
    unittest.main()
