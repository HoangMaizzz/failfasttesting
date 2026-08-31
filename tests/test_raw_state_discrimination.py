import unittest

import numpy as np

from run_raw_checkpoint_oracle_test import metrics


class RawStateDiscriminationTests(unittest.TestCase):
    def test_perfect_advantage_reports_perfect_confusion_matrix(self):
        result = metrics(
            np.asarray([3.0, 1.0, 4.0, 2.0]),
            np.asarray([1.0, 2.0, 2.0, 3.0]),
            np.asarray([2.0, -1.0, 2.0, -1.0]),
        )
        self.assertEqual(result["sign_accuracy"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["advantage_auc"], 1.0)
        self.assertEqual(result["true_stop"], 2)
        self.assertEqual(result["true_continue"], 2)
        self.assertEqual(result["false_stop"], 0)
        self.assertEqual(result["false_continue"], 0)

    def test_oracle_ties_are_excluded(self):
        result = metrics(
            np.asarray([1.0, 2.0, 4.0]),
            np.asarray([1.0, 3.0, 2.0]),
            np.asarray([99.0, -1.0, 1.0]),
        )
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["ties_excluded"], 1)
        self.assertEqual(result["sign_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
