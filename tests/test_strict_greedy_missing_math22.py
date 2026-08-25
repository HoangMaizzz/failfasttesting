import json
import unittest

from run_strict_greedy_missing_math22 import (
    MISSING_PROBLEM_IDS,
    ORACLE_REFERENCE_DIR,
)


class StrictGreedyMissingMath22Tests(unittest.TestCase):
    def test_missing_problem_ids_are_unique(self):
        self.assertEqual(len(MISSING_PROBLEM_IDS), 22)
        self.assertEqual(len(set(MISSING_PROBLEM_IDS)), 22)

    def test_bundled_policy_covers_every_missing_problem(self):
        policy = json.loads(
            (ORACLE_REFERENCE_DIR / "strict_greedy_policy.json").read_text(
                encoding="utf-8"
            )
        )
        policy_ids = {int(value) for value in policy["policies"]}
        self.assertTrue(set(MISSING_PROBLEM_IDS).issubset(policy_ids))


if __name__ == "__main__":
    unittest.main()
