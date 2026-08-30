import unittest

from exact_boundary_oracle import (
    BoundaryLeaf,
    CONTINUE,
    STOP,
    solve_exact_boundary_tree,
)


def leaf(script, emitted, passes):
    return BoundaryLeaf(
        script=tuple(script),
        emitted_tokens=emitted,
        draft_passes=passes,
        predicted_verify_ms=10.0,
        measured_verify_ms=999.0,
        proposal_length=8,
        proposal_hash="-".join(script),
    )


class ExactBoundaryOracleTests(unittest.TestCase):
    def test_continue_uses_optimally_timed_future_stop(self):
        leaves = [
            leaf([STOP], 4, 1),
            leaf([CONTINUE, STOP], 8, 2),
            leaf([CONTINUE, CONTINUE], 5, 3),
        ]
        rows = solve_exact_boundary_tree(
            [(), (CONTINUE,)],
            leaves,
            rho_tokens_per_ms=0.1,
            mean_draft_forward_ms=1.0,
            mean_post_verify_ms=0.0,
            epsilon_ms=0.0,
        )
        by_prefix = {tuple(row["action_prefix"]): row for row in rows}
        self.assertEqual(by_prefix[(CONTINUE,)]["oracle_action"], STOP)
        self.assertEqual(by_prefix[()]["oracle_action"], CONTINUE)
        self.assertEqual(
            by_prefix[()]["best_continue_leaf_script"],
            [CONTINUE, STOP],
        )

    def test_measured_verifier_latency_does_not_affect_winner(self):
        leaves = [
            BoundaryLeaf((STOP,), 4, 1, 10.0, 1_000.0, 8, "s"),
            BoundaryLeaf((CONTINUE,), 4, 2, 10.0, 1.0, 8, "c"),
        ]
        row = solve_exact_boundary_tree(
            [()],
            leaves,
            rho_tokens_per_ms=0.1,
            mean_draft_forward_ms=2.0,
            mean_post_verify_ms=0.0,
            epsilon_ms=0.0,
        )[0]
        self.assertEqual(row["oracle_action"], STOP)

    def test_one_millisecond_equivalent_is_a_tie(self):
        leaves = [leaf([STOP], 4, 1), leaf([CONTINUE], 4, 2)]
        row = solve_exact_boundary_tree(
            [()],
            leaves,
            rho_tokens_per_ms=0.1,
            mean_draft_forward_ms=1.0,
            mean_post_verify_ms=0.0,
            epsilon_ms=1.0,
        )[0]
        self.assertEqual(row["oracle_label"], "tie")
        self.assertEqual(row["oracle_action"], CONTINUE)


if __name__ == "__main__":
    unittest.main()
