import unittest

from causal_oracle_utils import prepare_causal_oracle_snapshots


class CausalOracleUtilsTests(unittest.TestCase):
    def test_preserves_recorded_refinement_snapshots(self):
        recorded = {
            "step": 2,
            "target_len": 8,
            "draft_passes_elapsed": 2,
            "draft_latency_elapsed_ms": 7.5,
            "draft_proposal": [1, 2],
        }
        snapshots, fallback_used = prepare_causal_oracle_snapshots(
            {"oracle_refinement_snapshots": [recorded]},
            [3, 4],
            3,
            10.0,
        )

        self.assertFalse(fallback_used)
        self.assertEqual(snapshots[0]["draft_proposal"], [1, 2])
        self.assertEqual(snapshots[0]["candidate_source"], "refinement_snapshot")

    def test_uses_measured_terminal_proposal_when_snapshots_are_unavailable(self):
        snapshots, fallback_used = prepare_causal_oracle_snapshots(
            {
                "oracle_refinement_snapshots": [],
                "steps": [{"step": 1}, {"step": 2}],
            },
            [11, 12, 13],
            4,
            18.25,
        )

        self.assertTrue(fallback_used)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["draft_proposal"], [11, 12, 13])
        self.assertEqual(snapshots[0]["step"], 2)
        self.assertEqual(snapshots[0]["draft_passes_elapsed"], 4)
        self.assertEqual(snapshots[0]["draft_latency_elapsed_ms"], 18.25)
        self.assertEqual(snapshots[0]["candidate_source"], "factual_terminal_fallback")

    def test_rejects_round_without_any_valid_proposal(self):
        with self.assertRaisesRegex(RuntimeError, "neither refinement snapshots"):
            prepare_causal_oracle_snapshots({}, [], 0, 0.0)


if __name__ == "__main__":
    unittest.main()
