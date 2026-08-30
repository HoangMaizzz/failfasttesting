import unittest

import numpy as np
import pandas as pd

from adaptive_td import V22_COMPACT_FEATURE_NAMES
from run_paired_boundary_feature_probe import (
    build_behavior_policy,
    classification_metrics,
    evenly_spaced_positions,
    grouped_predictions,
)


class PairedBoundaryFeatureProbeTests(unittest.TestCase):
    def test_evenly_spaced_positions_cover_trajectory(self):
        positions = evenly_spaced_positions(21, 5)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 20)
        self.assertEqual(len(positions), 5)

    def test_behavior_policy_preserves_actions_and_marks_selected_states(self):
        decisions = pd.DataFrame([
            {"problem_id": 7, "round_id": 0, "decision_id": 0, "action": "continue", "stop_available": False},
            {"problem_id": 7, "round_id": 0, "decision_id": 1, "action": "continue", "stop_available": True},
            {"problem_id": 7, "round_id": 0, "decision_id": 2, "action": "stop", "stop_available": True},
            {"problem_id": 7, "round_id": 1, "decision_id": 0, "action": "continue", "stop_available": True},
        ])
        results = pd.DataFrame([
            {"problem_id": 7, "num_speculation_rounds": 2},
        ])
        policy, selected = build_behavior_policy(decisions, results, 2)
        rounds = policy["policies"]["7"]
        self.assertEqual(rounds[0]["actions"], ["continue", "stop"])
        self.assertEqual(rounds[1]["actions"], ["continue"])
        self.assertEqual(len(selected), 2)
        selected_keys = set(zip(selected.round_id, selected.decision_id))
        marked_keys = {
            (row["round_id"], decision_id)
            for row in rounds
            for decision_id in row["probe_decision_ids"]
        }
        self.assertEqual(selected_keys, marked_keys)

    def test_metrics_use_stop_as_positive_class(self):
        metrics = classification_metrics(
            labels=[1, 1, 0, 0],
            scores=[2.0, -1.0, -2.0, 1.0],
            advantage=[2.0, 1.0, -2.0, -1.0],
        )
        self.assertEqual(metrics["tp_stop"], 1)
        self.assertEqual(metrics["fp_stop"], 1)
        self.assertEqual(metrics["fn_stop"], 1)
        self.assertEqual(metrics["tn_continue"], 1)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_grouped_ridge_recovers_held_out_advantage_signal(self):
        rows = []
        for problem_id in range(10):
            for offset in range(4):
                signal = -1.0 + 2.0 * offset / 3.0 + problem_id * 0.01
                row = {
                    "problem_id": problem_id,
                    "oracle_advantage_tokens": signal,
                }
                for name in V22_COMPACT_FEATURE_NAMES:
                    row[name] = 1.0 if name == "bias" else 0.0
                row["prefix_advance_ratio"] = signal
                rows.append(row)
        frame = pd.DataFrame(rows)
        predictions = grouped_predictions(frame, list(V22_COMPACT_FEATURE_NAMES))
        correlation = np.corrcoef(
            predictions, frame.oracle_advantage_tokens.to_numpy()
        )[0, 1]
        self.assertGreater(correlation, 0.95)


if __name__ == "__main__":
    unittest.main()
