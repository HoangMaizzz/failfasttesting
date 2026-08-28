import unittest
from pathlib import Path

from adaptive_td import active_refinement_positions, logical_refinement_span


ROOT = Path(__file__).resolve().parents[1]


class ActiveBlockRefinementTests(unittest.TestCase):
    def test_logical_span_starts_at_the_verifier_boundary(self):
        self.assertEqual(
            logical_refinement_span(118, 126, 112, 120, 8),
            (0, 118, 126),
        )

    def test_extension_uses_the_next_proposal_relative_span(self):
        self.assertEqual(
            logical_refinement_span(118, 134, 126, 134, 8),
            (1, 126, 134),
        )

    def test_physical_padding_outside_the_proposal_has_no_logical_span(self):
        self.assertIsNone(
            logical_refinement_span(118, 126, 128, 136, 8)
        )

    def test_future_small_block_masks_are_not_refinement_candidates(self):
        proposal_positions = tuple(range(118, 126))
        active = active_refinement_positions(
            proposal_positions,
            active_start=112,
            active_end=120,
        )
        self.assertEqual(active, (118, 119))

    def test_only_unresolved_positions_in_the_active_block_are_selected(self):
        active = active_refinement_positions(
            (119, 120, 123, 125),
            active_start=120,
            active_end=128,
        )
        self.assertEqual(active, (120, 123, 125))

    def test_modeling_gates_decisions_and_fill_to_active_positions(self):
        source = (ROOT / "Fast_dLLM_v2_1_5B" / "modeling.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "if active_remaining_positions and stop_available:",
            source,
        )
        self.assertIn(
            "remaining_absolute_positions = list(\n"
            "                                        active_remaining_positions",
            source,
        )
        self.assertIn('"unprocessed_future_masks"', source)
        self.assertIn("logical_refinement_span(", source)
        self.assertIn("if logical_span is None:", source)
        self.assertIn(
            "+ small_block_start_idx\n"
            "                                < draft_token_end_idx",
            source,
        )
        self.assertIn('"logical_block_index"', source)
        self.assertIn('"physical_small_block_index"', source)


if __name__ == "__main__":
    unittest.main()
