import unittest
from types import SimpleNamespace

import pandas as pd

from run_failfast_counterfactual_oracle import decision_state_key
from run_online_oracle_matched_benchmark import (
    match_oracle,
    oracle_problem_ids,
    problem_orders,
)


class OnlineOracleMatchedBenchmarkTests(unittest.TestCase):
    def test_problem_ids_match_existing_oracle_sampling(self):
        args = SimpleNamespace(
            dataset="gsm8k",
            oracle_sample_seed=2026,
            num_questions=50,
        )
        problem_ids = oracle_problem_ids(args)
        self.assertEqual(problem_ids[:5], [6, 24, 51, 157, 166])
        self.assertEqual(problem_ids[-5:], [1202, 1231, 1258, 1273, 1307])

    def test_orders_are_deterministic_and_preserve_question_set(self):
        problem_ids = [1, 2, 3, 4]
        first = problem_orders(problem_ids, [7, 19])
        second = problem_orders(problem_ids, [7, 19])
        self.assertEqual(first, second)
        self.assertEqual(first["canonical"], problem_ids)
        for order in first.values():
            self.assertEqual(set(order), set(problem_ids))

    def test_oracle_matching_requires_exact_proposal_state(self):
        key = decision_state_key(3, 100, 8, 1, [1, 2, 3])
        decisions = pd.DataFrame([{
            "state_key": key,
            "state_occurrence": 0,
            "action": "stop",
        }, {
            "state_key": decision_state_key(3, 100, 8, 1, [1, 2, 4]),
            "state_occurrence": 0,
            "action": "continue",
        }])
        oracle = pd.DataFrame([{
            "state_key": key,
            "oracle_action": "stop",
            "stop_ms_per_output_token": 2.0,
            "continue_ms_per_output_token": 3.0,
            "actual_next_gain_tokens": 0.0,
            "next_draft_latency_ms": 1.0,
        }])
        matched, evaluated = match_oracle(decisions, oracle)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(evaluated.iloc[0]["decision_correct"], 1)


if __name__ == "__main__":
    unittest.main()
