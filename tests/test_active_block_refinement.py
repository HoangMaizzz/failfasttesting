import unittest
from pathlib import Path

from adaptive_td import (
    active_refinement_positions,
    complete_raw_probability_frame,
    logical_refinement_span,
)


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

    def test_raw_frame_waits_for_both_physical_segments(self):
        cache = {
            position: (float(position), 0.0, 0.0)
            for position in range(112, 120)
        }
        self.assertIsNone(
            complete_raw_probability_frame(cache, 118, 126)
        )

        cache.update({
            position: (float(position), 0.0, 0.0)
            for position in range(120, 128)
        })
        frame = complete_raw_probability_frame(cache, 118, 126)
        self.assertEqual(
            [values[0] for values in frame],
            [float(position) for position in range(118, 126)],
        )

    def test_raw_frame_rejects_non_eight_token_span(self):
        with self.assertRaisesRegex(ValueError, "span 8 positions"):
            complete_raw_probability_frame({}, 118, 125)

    def test_every_verifier_prefix_offset_keeps_relative_eight_token_frames(self):
        for offset in range(8):
            draft_start = 112 + offset
            draft_end = draft_start + 24
            expected = [
                (index, draft_start + index * 8, draft_start + (index + 1) * 8)
                for index in range(3)
            ]
            observed = []
            physical_start = 112
            while physical_start < draft_end:
                span = logical_refinement_span(
                    draft_start,
                    draft_end,
                    physical_start,
                    physical_start + 8,
                    8,
                )
                if span is not None and span not in observed:
                    observed.append(span)
                physical_start += 8
            self.assertEqual(observed, expected)

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
        self.assertIn(
            "adaptive logical-frame extension must add",
            source,
        )
        self.assertIn("adaptive_raw_probability_cache", source)
        self.assertIn("raw_state_incomplete", source)
        self.assertIn(
            "not adaptive_enabled\n"
            "                                        and num_salvagable_tokens",
            source,
        )

    def test_long_runs_release_per_problem_gpu_cache(self):
        source = (ROOT / "failfast.py").read_text(encoding="utf-8")
        self.assertIn("orig_model_inputs = None", source)
        self.assertIn("prefill_output = None", source)
        self.assertIn("torch.cuda.empty_cache()", source)


if __name__ == "__main__":
    unittest.main()
