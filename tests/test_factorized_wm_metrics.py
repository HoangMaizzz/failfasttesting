"""Scientific reporting checks, runnable with unittest and no model dependencies."""

import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorized_wm_metrics import (
    binary_metrics, calibration, delta_report, dynamics_report,
    paired_question_bootstrap, select_operating_points, threshold_sweep,
    verifier_report, write_csv, write_json, write_jsonl,
)


def verifier_row(**overrides):
    row = {
        "question": "q1", "uid": "s1", "length": 8, "accepted": 2,
        "expected_yield": 3.0, "mode": 2,
        "hazards": [.9, .8, .2, .9, .9, .9, .9, .9],
        "tf_probs": [.9, .8, .2, .9, .1, .8, .2, .7],
        "tf_truth": [1, 1, 0, 1, 0, 1, 0, 1],
        "probability_pred": [.5] * 8, "probability_truth": [.4] * 8,
        "margin_pred": [-1.] * 8, "margin_truth": [-2.] * 8,
    }
    row.update(overrides)
    return row


class BinaryMetricsTests(unittest.TestCase):
    def test_precision_and_recall_are_not_confused(self):
        result = binary_metrics([1, 0, 0, 1], [.8, .7, .6, .1])
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], .5)
        self.assertAlmostEqual(result["f1"], .4)
        self.assertEqual(result["accuracy"], .25)
        self.assertEqual(result["balanced_accuracy"], .25)
        self.assertEqual((result["positives"], result["count"]), (2, 4))
        self.assertEqual(result["fpr"], 1.)
        self.assertEqual(result["fnr"], .5)

    def test_probability_statistics(self):
        result = binary_metrics([0, 1], [.2, .8])
        self.assertAlmostEqual(result["brier"], .04)
        self.assertAlmostEqual(result["nll"], -math.log(.8))
        self.assertAlmostEqual(result["ece"], .2)
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(result["average_precision"], 1.)
        endpoints = binary_metrics([1, 0], [0., 1.])
        self.assertEqual(endpoints["brier"], 1.)
        self.assertTrue(math.isfinite(endpoints["nll"]))

    def test_unbounded_scores_do_not_become_probabilities(self):
        result = binary_metrics([0, 1, 1], [-4., 1., 8.], threshold=0.)
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["roc_auc"], 1.)
        for name in ("brier", "nll", "ece"):
            self.assertIsNone(result[name])
        self.assertEqual(result["probability_status"], "scores_not_probabilities")

    def test_tied_ranking_has_half_credit_and_grouped_ap(self):
        tied = binary_metrics([0, 1, 0, 1], [2., 2., 2., 2.])
        self.assertEqual(tied["roc_auc"], .5)
        self.assertEqual(tied["average_precision"], .5)
        partial = binary_metrics([1, 0, 1, 0], [.8, .8, .2, .1])
        self.assertEqual(partial["roc_auc"], .625)
        self.assertAlmostEqual(partial["average_precision"], 7 / 12)
        self.assertEqual(partial, binary_metrics([0, 1, 0, 1], [.1, .2, .8, .8]))

    def test_empty_single_class_and_missing_pairs(self):
        empty = binary_metrics([], [])
        self.assertEqual(empty["status"], "empty")
        for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                     "roc_auc", "average_precision", "brier", "nll", "ece"):
            self.assertIsNone(empty[name])
        negative = binary_metrics([0, 0], [.1, .2])
        self.assertIsNone(negative["precision"])
        self.assertIsNone(negative["recall"])
        self.assertIsNone(negative["roc_auc"])
        self.assertIsNone(negative["average_precision"])
        self.assertEqual(negative["balanced_accuracy"], 1.)
        positive = binary_metrics([1, 1], [.1, .2])
        self.assertEqual(positive["f1"], 0.)
        self.assertEqual(positive["average_precision"], 1.)
        missing = binary_metrics([1, None, 0, 1], [.8, .4, math.nan, math.inf])
        self.assertEqual((missing["count"], missing["missing_count"]), (1, 3))

    def test_bad_binary_inputs_raise(self):
        with self.assertRaises(ValueError):
            binary_metrics([1], [])
        with self.assertRaises(ValueError):
            binary_metrics([2], [.2])
        with self.assertRaises(ValueError):
            binary_metrics([1], [.2], threshold=math.nan)

    def test_calibration_boundaries_and_empty_bins(self):
        bins = calibration([0, 1, 1, 0], [0., .25, .5, 1.], bins=4)
        self.assertEqual([b["count"] for b in bins], [1, 1, 1, 1])
        self.assertEqual(bins[-1], {
            "mean_probability": 1., "empirical_rate": 0., "count": 1,
            "lo": .75, "hi": 1.,
        })
        bins = calibration([0, 1], [.1, .9], bins=4)
        self.assertEqual(len(bins), 4)
        self.assertEqual(bins[1]["count"], 0)
        self.assertIsNone(bins[1]["mean_probability"])
        self.assertIsNone(bins[1]["empirical_rate"])
        self.assertEqual(len(calibration([], [])), 10)
        with self.assertRaises(ValueError):
            calibration([1], [1.1])
        with self.assertRaises(ValueError):
            calibration([1], [.5], bins=0)

    def test_optional_numpy_sklearn_agreement(self):
        try:
            import numpy as np
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            self.skipTest("optional numpy/sklearn unavailable")
        truth = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        scores = np.array([.8, .8, .2, .1, .4, .4, .9, .2])
        result = binary_metrics(truth, scores)
        self.assertAlmostEqual(result["roc_auc"], roc_auc_score(truth, scores))
        self.assertAlmostEqual(result["average_precision"], average_precision_score(truth, scores))


class VerifierTests(unittest.TestCase):
    def test_hazard_censoring_and_tf_suffix_are_separate(self):
        row = verifier_row()
        result = verifier_report([row])["all"]
        self.assertEqual(result["hazard"]["metrics"]["count"], 3)
        self.assertEqual(result["hazard"]["metrics"]["positives"], 2)
        self.assertEqual(result["survival"]["metrics"]["count"], 8)
        self.assertEqual(result["survival"]["metrics"]["positives"], 2)
        self.assertEqual(result["full_tf"]["metrics"]["count"], 8)
        self.assertEqual(result["full_tf"]["metrics"]["positives"], 5)
        changed = verifier_row(hazards=[.9, .8, .2] + [.01] * 5,
                               tf_truth=[1, 1, 0, 0, 0, 0, 0, 0])
        other = verifier_report([changed])["all"]
        self.assertEqual(result["hazard"], other["hazard"])
        self.assertNotEqual(result["survival"], other["survival"])
        self.assertNotEqual(result["full_tf"], other["full_tf"])
        survival = [.9, .72, .144, .1296, .11664, .104976, .0944784, .08503056]
        expected = binary_metrics([1, 1, 0, 0, 0, 0, 0, 0], survival)
        self.assertAlmostEqual(result["survival"]["metrics"]["brier"], expected["brier"])

    def test_zero_and_full_acceptance_mask_boundaries(self):
        zero = verifier_report([verifier_row(accepted=0)])["all"]
        full = verifier_report([verifier_row(accepted=8)])["all"]
        self.assertEqual(zero["hazard"]["metrics"]["count"], 1)
        self.assertEqual(zero["survival"]["metrics"]["positives"], 0)
        self.assertEqual(full["hazard"]["metrics"]["count"], 8)
        self.assertEqual(full["survival"]["metrics"]["positives"], 8)

    def test_missing_teachers_are_not_zero_labels(self):
        result = verifier_report([verifier_row(tf_truth=None, probability_truth=None,
                                               margin_truth=None)])["all"]
        self.assertEqual(result["hazard"]["metrics"]["count"], 3)
        self.assertEqual(result["full_tf"]["metrics"]["count"], 0)
        self.assertEqual(result["full_tf"]["metrics"]["missing_count"], 8)
        self.assertIsNone(result["full_tf"]["metrics"]["brier"])
        self.assertEqual(result["probability"], {"count": 0, "missing_count": 8,
                                                 "mae": None, "status": "missing"})
        self.assertIsNone(result["margin"]["mae"])

    def test_partial_teacher_validity_and_padding(self):
        result = verifier_report([verifier_row(
            tf_truth=[1, None, 0, math.nan, 0, 1, None, 1, 1],
            probability_truth=[.4, None, .4], margin_truth=[-2., math.nan],
            hazards=[.9, None, .2, .9, .9, .9, .9, .9, 7.],
        )])["all"]
        self.assertEqual(result["full_tf"]["metrics"]["count"], 5)
        self.assertEqual(result["hazard"]["metrics"]["count"], 2)
        self.assertEqual(result["hazard"]["metrics"]["missing_count"], 1)
        self.assertEqual(result["survival"]["metrics"]["count"], 1)
        self.assertEqual(result["survival"]["metrics"]["missing_count"], 7)
        self.assertEqual(result["probability"]["count"], 2)
        self.assertAlmostEqual(result["probability"]["mae"], .1)
        self.assertEqual(result["margin"]["count"], 1)
        self.assertEqual(result["margin"]["mae"], 1.)

    def test_k_metrics_and_equal_question_weight(self):
        rows = [verifier_row(expected_yield=3., mode=2),
                verifier_row(uid="s2", expected_yield=4., mode=3),
                verifier_row(question="q2", uid="s3", expected_yield=7., mode=2)]
        result = verifier_report(rows)["all"]["k"]
        self.assertAlmostEqual(result["mae"], 8 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(10))
        self.assertEqual(result["median"], 2.)
        self.assertAlmostEqual(result["p90"], 4.4)
        self.assertAlmostEqual(result["within1"], 1 / 3)
        self.assertAlmostEqual(result["within2"], 2 / 3)
        self.assertAlmostEqual(result["within4"], 2 / 3)
        self.assertAlmostEqual(result["bias"], 8 / 3)
        self.assertAlmostEqual(result["exact_mode"], 2 / 3)
        self.assertEqual(result["question_macro_mae"], 3.25)
        self.assertEqual(result["question_count"], 2)

    def test_modes_and_expected_k_have_independent_validity(self):
        result = verifier_report([verifier_row(expected_yield=None),
                                  verifier_row(expected_yield=0., mode=None)])["all"]["k"]
        self.assertEqual((result["count"], result["mode_count"]), (1, 1))
        self.assertEqual(result["mae"], 2.)
        self.assertEqual(result["bias"], -2.)
        self.assertEqual(result["exact_mode"], 1.)

    def test_length_and_relative_position_groups_and_purity(self):
        rows = [verifier_row(), verifier_row(question="q2", length=48, accepted=8)]
        before = copy.deepcopy(rows)
        result = verifier_report(rows)
        self.assertEqual(rows, before)
        self.assertEqual(result["by_length"]["8"]["count"], 1)
        self.assertEqual(result["by_length"]["40+"]["count"], 1)
        self.assertEqual(result["by_length"]["16"]["status"], "empty")
        self.assertEqual(list(result["by_relative_position"]), ["Q1", "Q2", "Q3", "Q4"])
        for group in result["by_relative_position"].values():
            self.assertEqual(group["positions"], 14)
        single = verifier_report([verifier_row()])["by_relative_position"]
        self.assertEqual([single[q]["hazard"]["metrics"]["count"] for q in single], [2, 1, 0, 0])
        self.assertEqual([single[q]["full_tf"]["metrics"]["count"] for q in single], [2, 2, 2, 2])

    def test_verifier_empty_and_invalid_inputs(self):
        empty = verifier_report([])
        self.assertIsNone(empty["all"]["k"]["mae"])
        self.assertEqual(len(empty["by_length"]), 5)
        for changes in ({"accepted": 9}, {"accepted": -1}, {"mode": .5},
                        {"length": 0}, {"hazards": [1.1]}, {"question": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verifier_report([verifier_row(**changes)])


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"question": "q1", "action": "R", "score": .9, "real_useful": 1},
            {"question": "q1", "action": "E", "score": .6, "real_useful": 0},
            {"question": "q2", "action": "R", "score": .6, "real_useful": 1},
            {"question": "q2", "action": "E", "score": .1, "real_useful": 0},
        ]

    def test_sweep_unique_thresholds_and_safe_infinities(self):
        result = threshold_sweep(self.rows)
        self.assertEqual([r["threshold"] for r in result], ["-inf", .1, .6, .9, "+inf"])
        self.assertEqual([r["selected_count"] for r in result], [4, 4, 3, 1, 0])
        self.assertAlmostEqual(result[2]["precision"], 2 / 3)
        self.assertEqual(result[2]["recall"], 1.)
        self.assertEqual(result[2]["fpr"], .5)
        self.assertEqual(result[2]["fnr"], 0.)
        self.assertIsNone(result[-1]["precision"])
        json.dumps(result, allow_nan=False)
        for record in result:
            metrics = binary_metrics([r["real_useful"] for r in self.rows],
                                     [r["score"] for r in self.rows], record["threshold"])
            for key in ("tp", "fp", "tn", "fn", "f1", "selected_count"):
                self.assertEqual(record[key], metrics[key])

    def test_selection_is_pure_and_validation_only(self):
        rows = [dict(row, split="validation") for row in self.rows]
        before = copy.deepcopy(rows)
        result = select_operating_points(rows)
        self.assertEqual(rows, before)
        self.assertEqual(result["high_precision"]["threshold"], .9)
        self.assertEqual(result["balanced"]["threshold"], .6)
        self.assertEqual(result["high_recall"]["threshold"], .6)
        test_rows = [dict(row, score=1 - row["score"], split="test") for row in rows]
        threshold_sweep(test_rows)  # evaluation is permitted, selection is not
        self.assertEqual(result, select_operating_points(rows))
        for split in ("test", "train"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                select_operating_points([dict(self.rows[0], split=split)])

    def test_infeasible_constraints_and_absent_data(self):
        tied = [dict(row, score=.5) for row in self.rows]
        result = select_operating_points(tied)
        self.assertEqual(result["base_rate"], .5)
        self.assertIsNone(result["high_recall"])
        self.assertIsNotNone(result["balanced"])
        negative = select_operating_points([dict(row, real_useful=0) for row in self.rows])
        self.assertIsNone(negative["high_precision"])
        self.assertIsNone(negative["balanced"])
        self.assertIsNone(negative["high_recall"])
        empty = select_operating_points([])
        self.assertIsNone(empty["base_rate"])
        self.assertIsNone(empty["balanced"])
        self.assertEqual(len(threshold_sweep([])), 2)

    def test_custom_keys_and_one_row_per_action(self):
        rows = [{"question": "q", "action": "E", "gain": 7., "good": 1},
                {"question": "q", "action": "R", "gain": -2., "good": 0},
                {"question": "q", "action": "R", "gain": None, "good": 1}]
        result = threshold_sweep(rows, "gain", "good")
        self.assertEqual(result[0]["count"], 2)
        self.assertEqual(result[0]["missing_count"], 1)
        self.assertEqual(result[-2]["precision"], 1.)


class DeltaTests(unittest.TestCase):
    def test_action_reports_correlations_and_sign_tolerance(self):
        rows = [{"question": "q", "action": "R", "delta_true": t, "delta_pred": p}
                for t, p in [(-2., -1.), (0., .5), (.25, -.25), (2., 3.)]]
        result = delta_report(rows)
        all_metrics = result["all"]
        self.assertEqual(all_metrics["sign_accuracy"], 1.)
        self.assertEqual(all_metrics["zero_tolerance_tokens"], .5)
        self.assertEqual(all_metrics["mae"], .75)
        self.assertEqual(all_metrics["useful"]["positives"], 2)
        self.assertEqual(all_metrics["useful"]["precision"], .5)
        self.assertEqual(all_metrics["useful"]["recall"], .5)
        self.assertEqual(all_metrics["spearman"], .8)
        self.assertGreater(all_metrics["pearson"], .9)
        self.assertEqual(result["by_action"]["R"], all_metrics)
        self.assertEqual(result["by_action"]["E"]["status"], "missing")
        self.assertIsNone(result["by_action"]["E"]["pearson"])

    def test_delta_never_calibrates_even_if_in_unit_interval(self):
        rows = [{"question": "q", "action": "E", "delta_true": t, "delta_pred": p}
                for t, p in [(-1, .1), (1, .9)]]
        result = delta_report(rows, threshold=.5)["all"]["useful"]
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(result["average_precision"], 1.)
        self.assertEqual(result["precision"], 1.)
        for key in ("brier", "nll", "ece"):
            self.assertIsNone(result[key])

    def test_tied_spearman_constants_and_missing(self):
        rows = [{"question": "q", "action": "R", "delta_true": t, "delta_pred": p}
                for t, p in [(0, 1), (0, 1), (2, 3)]]
        self.assertEqual(delta_report(rows)["all"]["spearman"], 1.)
        constant = [dict(row, delta_pred=0.) for row in rows]
        result = delta_report(constant)["all"]
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])
        missing = delta_report([dict(rows[0], delta_true=None)])["all"]
        self.assertEqual(missing["count"], 0)
        self.assertEqual(missing["missing_count"], 1)
        with self.assertRaises(ValueError):
            delta_report([dict(rows[0], action="X")])


class BootstrapTests(unittest.TestCase):
    def test_question_macro_pairing_not_row_bootstrap(self):
        rows = [{"question": "large", "accepted": 0., "a": 0., "b": 1.}] * 20
        rows += [{"question": "small", "accepted": 0., "a": 3., "b": 0.}]
        rows += [{"question": "missing", "accepted": 0., "a": None, "b": 0.}]
        result = paired_question_bootstrap(rows, "a", "b", samples=2000, seed=42)
        self.assertEqual(result["delta_macro_mae"], 1.)
        self.assertEqual(result["ci95"], [-1., 3.])
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["count"], 21)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result, paired_question_bootstrap(reversed(rows), "a", "b", samples=2000, seed=42))
        duplicated = rows[:20] * 5 + rows[20:]
        extra = paired_question_bootstrap(duplicated, "a", "b", samples=2000, seed=42)
        self.assertEqual(extra["delta_macro_mae"], result["delta_macro_mae"])
        self.assertEqual(extra["ci95"], result["ci95"])

    def test_bootstrap_direction_reproducibility_and_identical_predictions(self):
        rows = [{"question": str(i), "accepted": 1., "a": 1., "b": float(i)} for i in range(5)]
        result = paired_question_bootstrap(rows, "a", "b", samples=100, seed=7)
        self.assertLess(result["delta_macro_mae"], 0.)
        self.assertEqual(result, paired_question_bootstrap(rows, "a", "b", samples=100, seed=7))
        same = paired_question_bootstrap(rows, "a", "a")
        self.assertEqual(same["delta_macro_mae"], 0.)
        self.assertEqual(same["ci95"], [0., 0.])
        one = paired_question_bootstrap(rows[:1], "a", "b")
        self.assertEqual(one["ci95"], [-1., -1.])

    def test_bootstrap_empty_and_invalid(self):
        empty = paired_question_bootstrap([], "a", "b")
        self.assertIsNone(empty["delta_macro_mae"])
        self.assertIsNone(empty["ci95"])
        with self.assertRaises(ValueError):
            paired_question_bootstrap([], "a", "b", samples=0)
        with self.assertRaises(ValueError):
            paired_question_bootstrap([{"a": 1, "b": 2, "accepted": 0}], "a", "b")


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"horizon": 1, "actions": "E", "question": "q1", "length": 16,
             "change": "changed", "regions": {"new_block": {
                 "mse_hidden": 1., "token_agreement": .75, "correct_tokens": 6,
                 "token_count": 8, "top5_recall": .8}}},
            {"horizon": 2, "actions": "RE", "question": "q2", "length": 24,
             "tags": ["unchanged"], "regions": {"new_block": {
                 "mse_hidden": 3., "token_agreement": 1., "correct_tokens": 8,
                 "token_count": 8, "conditional_topk_kl": .2}}},
            {"horizon": 3, "actions": "EER", "question": "q1", "changed": False,
             "regions": {"new_block": {"mse_hidden": None, "correct_tokens": 4,
                                      "token_count": 8}, "prefix": {"mask_brier": .1}}},
        ]

    def test_grouping_statuses_and_row_means(self):
        before = copy.deepcopy(self.rows)
        result = dynamics_report(self.rows)
        self.assertEqual(before, self.rows)
        self.assertEqual(result["all"]["count"], 3)
        self.assertEqual(result["all"]["question_count"], 2)
        self.assertEqual(result["by_horizon"]["H2"]["count"], 1)
        self.assertEqual(result["by_action_sequence"]["RE"]["count"], 1)
        self.assertEqual(result["by_action"]["E"]["count"], 2)
        self.assertEqual(result["by_action"]["R"]["count"], 1)
        self.assertEqual(result["by_length"]["24"]["count"], 1)
        self.assertEqual(result["by_change"]["unchanged"]["count"], 2)
        region = result["all"]["regions"]["new_block"]
        self.assertEqual(region["mse_hidden"], 2.)
        self.assertEqual(region["metric_counts"]["mse_hidden"], 2)
        self.assertEqual(region["top5_recall"], .8)
        self.assertEqual(region["conditional_topk_kl"], .2)
        self.assertEqual(region["metric_status"]["cosine_hidden"], "missing")
        self.assertIsNone(region["cosine_hidden"])
        absent = result["by_horizon"]["H1"]["regions"]["prefix"]
        self.assertIsNone(absent["mask_brier"])
        self.assertEqual(absent["status"], "missing")
        self.assertEqual(region["token_count"], 24)
        self.assertEqual(region["correct_tokens"], 18)
        self.assertEqual(region["pooled_token_agreement"], .75)
        self.assertEqual(region["token_agreement"], .875)

    def test_new_diagnostics_survive_aggregation_and_grouping(self):
        diagnostics = {
            "native_top1_agreement": (0., .8, .4),
            "conditional_topk_coverage": (.25, .75, .5),
            "token_in_vocabulary_coverage": (.6, .8, .7),
            "linear_CKA_hidden": (-.25, .5, .125),
            "jaccard10": (.2, .8, .5),
            "fresh_hidden_tokens": (8, 4, 6.),
            "fresh_logits_tokens": (0, 8, 4.),
        }
        rows = copy.deepcopy(self.rows)
        for key, (first, second, _) in diagnostics.items():
            rows[0]["regions"]["new_block"][key] = first
            rows[1]["regions"]["new_block"][key] = second
            rows[2]["regions"]["new_block"][key] = math.nan
        # Exercise the serialized report consumers use, not just raw input rows.
        report = json.loads(json.dumps(dynamics_report(rows), allow_nan=False))
        for key, (first, second, expected) in diagnostics.items():
            with self.subTest(metric=key):
                for group in (report["all"], report["by_action"]["E"]):
                    region = group["regions"]["new_block"]
                    self.assertAlmostEqual(region[key], expected)
                    self.assertEqual(region["metric_counts"][key], 2)
                    self.assertEqual(region["metric_status"][key], "ok")
                for group, value in ((report["by_horizon"]["H1"], first),
                                     (report["by_action_sequence"]["RE"], second),
                                     (report["by_length"]["16"], first),
                                     (report["by_change"]["changed"], first)):
                    region = group["regions"]["new_block"]
                    self.assertEqual(region[key], value)
                    self.assertEqual(region["metric_counts"][key], 1)
                missing = report["by_horizon"]["H3"]["regions"]["new_block"]
                self.assertIsNone(missing[key])
                self.assertEqual(missing["metric_counts"][key], 0)
                self.assertEqual(missing["metric_status"][key], "missing")

    def test_new_diagnostics_are_explicitly_missing_without_observations(self):
        report = dynamics_report(self.rows)
        keys = ("native_top1_agreement", "conditional_topk_coverage",
                "token_in_vocabulary_coverage", "linear_CKA_hidden", "jaccard10",
                "fresh_hidden_tokens", "fresh_logits_tokens")
        for group in (report["all"], report["by_horizon"]["H1"], report["by_length"]["8"]):
            for name in ("new_block", "prefix"):
                region = group["regions"][name]
                for key in keys:
                    with self.subTest(region=name, metric=key, group_count=group["count"]):
                        self.assertIn(key, region)
                        self.assertIsNone(region[key])
                        self.assertEqual(region["metric_counts"][key], 0)
                        self.assertEqual(region["metric_status"][key], "missing")

    def test_eight_token_e_blocks_only_with_explicit_counters(self):
        result = dynamics_report(self.rows)["all"]["regions"]["new_block"]["e_block"]
        self.assertEqual(result, {"count": 2, "status": "ok", "ge4_of8": 1.,
                                  "ge6_of8": 1., "ge7_of8": .5, "eq8_of8": .5})
        rows = [dict(self.rows[0], regions={"new_block": {"token_agreement": 1.}}),
                dict(self.rows[0], regions={"new_block": {"correct_tokens": 8, "token_count": 16}}),
                dict(self.rows[0], regions={"new_block": {"correct_tokens": None, "token_count": 8}})]
        missing = dynamics_report(rows)["all"]["regions"]["new_block"]["e_block"]
        self.assertEqual(missing["count"], 0)
        self.assertIsNone(missing["eq8_of8"])
        self.assertEqual(missing["status"], "missing")

    def test_e_block_success_rates_at_every_requested_boundary(self):
        rows = [dict(self.rows[0], regions={"new_block": {"correct_tokens": correct,
                                                         "token_count": 8}})
                for correct in (0, 4, 6, 7, 8)]
        result = dynamics_report(rows)["all"]["regions"]["new_block"]["e_block"]
        self.assertEqual(result["count"], 5)
        self.assertEqual([result[key] for key in ("ge4_of8", "ge6_of8", "ge7_of8", "eq8_of8")],
                         [.8, .6, .4, .2])

    def test_empty_missing_region_and_no_inferred_change(self):
        empty = dynamics_report([])
        self.assertEqual(empty["all"]["status"], "empty")
        self.assertEqual(set(empty["by_horizon"]), {"H1", "H2", "H3"})
        rows = [{"horizon": 1, "actions": "R", "question": "q", "regions": {"x": None}}]
        result = dynamics_report(rows)
        self.assertEqual(result["by_change"], {})
        self.assertEqual(result["by_length"], {})
        self.assertEqual(result["all"]["regions"]["x"]["metric_status"]["mse_hidden"], "missing")
        self.assertIsNone(result["all"]["regions"]["x"]["token_count"])

    def test_dynamics_invalid_inputs_and_conflicting_tags(self):
        for changes in ({"horizon": 2}, {"actions": "X"}, {"changed": False},
                        {"regions": {"x": {"correct_tokens": 9, "token_count": 8}}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dynamics_report([dict(self.rows[0], **changes)])


class WriterTests(unittest.TestCase):
    def test_json_and_jsonl_are_strict_and_preserve_unicode(self):
        data = {"nan": math.nan, "inf": math.inf, "minus_inf": -math.inf,
                "nested": [1, None, {"text": "คำถาม"}], "threshold": "+inf"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.assertEqual(write_json(path, data), path)
            result = json.loads(path.read_text(encoding="utf-8"))
            for key in ("nan", "inf", "minus_inf"):
                self.assertIsNone(result[key])
            self.assertEqual(result["nested"][2]["text"], "คำถาม")
            self.assertNotIn("NaN", path.read_text(encoding="utf-8"))
            lines = Path(directory) / "report.jsonl"
            write_jsonl(lines, iter([data, {"value": 2}]))
            records = [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0], result)
            self.assertEqual(records[1], {"value": 2})

    def test_csv_union_nested_values_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            write_csv(path, [{"a": 1, "b": math.nan}, {"a": 2, "c": [1, None]}])
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["a", "b", "c"])
                rows = list(reader)
            self.assertEqual(rows[0], {"a": "1", "b": "", "c": ""})
            self.assertEqual(json.loads(rows[1]["c"]), [1, None])
            write_csv(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_optional_numpy_serialization(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numpy.json"
            write_json(path, {"array": np.array([1., np.nan, np.inf]),
                              "int": np.int64(4), "bool": np.bool_(True),
                              "extended": np.longdouble(.25),
                              "extended_array": np.array([.5, np.nan], dtype=np.longdouble)})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"array": [1., None, None], "int": 4, "bool": True,
                              "extended": .25, "extended_array": [.5, None]})

    def test_failed_write_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            write_jsonl(path, [{"original": True}])
            original = path.read_text(encoding="utf-8")
            with self.assertRaises(TypeError):
                write_jsonl(path, [{"ok": True}, {"unsupported": object()}])
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_every_report_is_strict_json_serializable(self):
        reports = [verifier_report([verifier_row()]), verifier_report([]),
                   delta_report([]), dynamics_report([]), select_operating_points([]),
                   paired_question_bootstrap([], "a", "b"), binary_metrics([], [])]
        for report in reports:
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
