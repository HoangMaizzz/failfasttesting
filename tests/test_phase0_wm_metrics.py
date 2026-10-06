"""True-label, paired-output, and question-bootstrap checks without model dependencies."""

import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase0_wm_metrics import phase0_report, write_reports


def action_row(question="q1", actions="R", accepted=1, parent_accepted=1,
               changed=False, predictions=(1.5, 0.5, 2.0, 1.0), sources=(1., 1., 1., 1.)):
    row = dict(question=question, parent=question + "-parent", child=question + "-" + actions,
               horizon=len(actions), actions=actions, action=actions[-1], parent_length=2,
               length=2, parent_accepted=parent_accepted, accepted=accepted,
               parent_token_changed=changed,
               delta_true=None if accepted is None or parent_accepted is None else accepted - parent_accepted)
    for model, prediction, source in zip(("oracle", "imagined", "prior", "direct"), predictions, sources):
        row[model + "_source_K"] = source
        row[model + "_verifier"] = dict(
            question=question, uid=row["child"], length=2, accepted=accepted,
            expected_yield=prediction, mode=1, hazards=[.8, .25], tf_probs=[.7, .6],
            tf_truth=[1, 0], probability_pred=[.6, .3], probability_truth=[.4, .8],
            margin_pred=[.2, -.3], margin_truth=[.5, -.1])
    return row


def three_questions():
    first = action_row()
    first.update(latent_cosine=.8, latent_mse=.2)
    imagined = first["imagined_verifier"]
    imagined.update(hazards=[.6, .5], tf_probs=[.4, .8], probability_pred=[.5, .9],
                    margin_pred=[-.2, .5])
    second = action_row("q2", accepted=2, parent_accepted=1, changed=True,
                        predictions=(1., 2., 0., 1.5), sources=(1., 1.5, 0., 1.))
    second.update(latent_cosine=.4, latent_mse=.6)
    third = action_row("q3", "E", accepted=0, parent_accepted=2,
                       predictions=(0., .5, 1., None), sources=(2., 2., 2., 2.))
    for model in ("oracle", "imagined", "prior", "direct"):
        third[model + "_verifier"].update(tf_truth=None, probability_truth=None,
                                           margin_truth=[None, math.nan])
    return [first, second, third]


def block_row(question="q1", actions="E", accepted=10, parent_accepted=8,
              parent_length=8, changed=False):
    """Native eight-token geometry, with teacher error concentrated on new tokens."""
    row = action_row(question, actions, accepted, parent_accepted, changed)
    length = parent_length + 8 * actions.count("E")
    row.update(parent_length=parent_length, length=length)
    for model in ("oracle", "imagined", "prior", "direct"):
        imagined = model == "imagined"
        hazards = ([.5 if imagined else .8] + [1.] * 7
                   + [.5 if imagined else .75] * (length - 8))
        row[model + "_source_K"] = 6.4
        row[model + "_verifier"].update(
            length=length, mode=0,
            expected_yield=sum(math.prod(hazards[:i + 1]) for i in range(length)), hazards=hazards,
            tf_probs=[1.] * 8 + [.9 if imagined else .1] * (length - 8),
            tf_truth=[1] * 8 + [0] * (length - 8),
            probability_pred=[.9] * 8 + [.8 if imagined else .2] * (length - 8),
            probability_truth=[.9] * 8 + [.1] * (length - 8),
            margin_pred=[.1] * 8 + [1. if imagined else -1.] * (length - 8),
            margin_truth=[.1] * 8 + [-1.] * (length - 8))
    return row


class Phase0MetricsTests(unittest.TestCase):
    def report(self, rows):
        return phase0_report(rows, bootstrap_samples=120, seed=17)

    def test_true_labels_censoring_and_model_specific_delta_sources(self):
        rows = three_questions()
        before = copy.deepcopy(rows)
        report = self.report(iter(rows))
        # Check input preservation without NaN equality semantics.
        self.assertEqual(json.dumps(rows, sort_keys=True), json.dumps(before, sort_keys=True))
        overall = report["overall"]
        self.assertEqual((overall["count"], overall["question_count"]), (3, 3))
        self.assertAlmostEqual(overall["latent"]["latent_cosine"]["mean"], .6)
        self.assertAlmostEqual(overall["latent"]["latent_mse"]["mean"], .4)
        self.assertEqual(overall["latent"]["latent_cosine"]["missing_count"], 1)
        imagined = overall["models"]["imagined"]
        self.assertAlmostEqual(imagined["verifier"]["k"]["mae"], 1 / 3)
        self.assertAlmostEqual(imagined["verifier"]["k"]["bias"], 0.)
        self.assertAlmostEqual(imagined["delta"]["mae"], .5)
        self.assertAlmostEqual(imagined["delta"]["sign_accuracy"], 2 / 3)
        self.assertEqual(imagined["delta"]["zero_tolerance_tokens"], .5)
        self.assertEqual(imagined["verifier"]["hazard"]["metrics"]["count"], 5)
        self.assertEqual(imagined["verifier"]["survival"]["metrics"]["count"], 6)
        # Imagined can score better than oracle against real endpoint K.
        self.assertEqual(overall["models"]["oracle"]["verifier"]["k"]["mae"], .5)
        first = self.report(rows[:1])["overall"]["models"]["imagined"]["verifier"]
        self.assertAlmostEqual(first["hazard"]["metrics"]["brier"], (.4**2 + .5**2) / 2)
        self.assertAlmostEqual(first["survival"]["metrics"]["brier"], (.4**2 + .3**2) / 2)
        self.assertAlmostEqual(first["full_tf"]["metrics"]["brier"], (.6**2 + .8**2) / 2)
        self.assertAlmostEqual(first["probability"]["mae"], .1)
        self.assertAlmostEqual(first["margin"]["mae"], .65)

    def test_paired_outputs_detect_differences_despite_equal_endpoint_mae(self):
        row = three_questions()[0]
        overall = self.report([row])["overall"]
        self.assertEqual(overall["models"]["imagined"]["verifier"]["k"]["mae"], .5)
        self.assertEqual(overall["models"]["oracle"]["verifier"]["k"]["mae"], .5)
        gaps = overall["paired_prediction_gaps"]
        self.assertEqual((gaps["K"]["bias"], gaps["K"]["mae"]), (-1., 1.))
        expected = {"hazards": .225, "survival": .15, "TF": .25, "pV": .35, "margin": .6}
        for head, mae in expected.items():
            with self.subTest(head=head):
                self.assertEqual(gaps[head]["count"], 2)
                self.assertAlmostEqual(gaps[head]["mae"], mae)
                for boundary in gaps[head]["question_macro_mae_ci95"]:
                    self.assertAlmostEqual(boundary, mae)
        self.assertAlmostEqual(gaps["hazards"]["bias"], .025)
        self.assertAlmostEqual(gaps["survival"]["bias"], -.05)
        # Survival is cumulative .6,.3 versus .8,.2, not the hazard vector.
        self.assertNotEqual(gaps["hazards"]["mae"], gaps["survival"]["mae"])

    def test_single_class_and_h1_native_token_and_extension_cohorts(self):
        rows = three_questions()
        rows += [action_row("q1", accepted=0, parent_accepted=1, changed=True),
                 action_row("q2", "E", accepted=1, parent_accepted=1)]
        cohorts = self.report(rows)["cohorts"]
        self.assertEqual([cohorts["R_H1"][name]["count"] for name in
                          ("unchanged", "changed", "gain", "loss")], [1, 2, 1, 1])
        self.assertEqual(cohorts["E_H1"]["parent_K_eq_L"]["count"], 1)
        self.assertEqual(cohorts["E_H1"]["parent_K_lt_L"]["count"], 1)
        gain = cohorts["R_H1"]["gain"]["models"]["imagined"]
        self.assertIsNone(gain["delta"]["useful"]["roc_auc"])
        self.assertEqual(gain["delta"]["useful"]["ranking_status"], "single_class")
        self.assertIsNone(gain["verifier"]["hazard"]["metrics"]["roc_auc"])
        self.assertEqual(gain["verifier"]["hazard"]["metrics"]["ranking_status"], "single_class")
        # This is a supplied native-token flag, independent of whether K changed.
        self.assertEqual(cohorts["R_H1"]["unchanged"]["models"]["imagined"]["delta"]["count"], 1)

    def test_h2_h3_preserve_full_sequences_and_exclude_h1_cohorts(self):
        rows = [action_row("q1", "RR"), action_row("q2", "ER"),
                action_row("q3", "RER"), action_row("q1", "EER")]
        report = self.report(rows)
        self.assertEqual([report["by_horizon"][h]["count"] for h in ("H1", "H2", "H3")], [0, 2, 2])
        self.assertEqual(set(report["by_sequence"]), {"RR", "ER", "RER", "EER"})
        self.assertTrue(all(group["count"] == 1 for group in report["by_sequence"].values()))
        self.assertTrue(all(group["count"] == 0 for family in report["cohorts"].values()
                            for group in family.values()))

    def test_missing_teachers_targets_sources_and_latents_remain_unobserved(self):
        rows = three_questions()
        unknown = action_row("q3", "E", accepted=None, parent_accepted=None, changed=None)
        unknown["imagined_source_K"] = None
        unknown["latent_cosine"], unknown["latent_mse"] = math.nan, math.inf
        for model in ("oracle", "imagined", "prior", "direct"):
            # Outer truth must prevent stale nested labels from becoming zero targets.
            unknown[model + "_verifier"].update(accepted=0, tf_truth=None,
                                                probability_truth=None, margin_truth=None)
        report = self.report([unknown])["overall"]
        imagined = report["models"]["imagined"]
        self.assertEqual(imagined["verifier"]["k"]["count"], 0)
        self.assertIsNone(imagined["verifier"]["k"]["mae"])
        self.assertEqual(imagined["delta"]["count"], 0)
        for head in ("hazard", "survival", "full_tf"):
            self.assertEqual(imagined["verifier"][head]["metrics"]["count"], 0)
            self.assertIsNone(imagined["verifier"][head]["metrics"]["brier"])
        for head in ("probability", "margin"):
            self.assertEqual(imagined["verifier"][head]["count"], 0)
            self.assertIsNone(imagined["verifier"][head]["mae"])
        for scalar in report["latent"].values():
            self.assertEqual((scalar["count"], scalar["status"], scalar["mean"]), (0, "missing", None))
        for target in report["paired_mae_differences"].values():
            for comparison in target.values():
                self.assertEqual(comparison["question_count"], 0)
                self.assertIsNone(comparison["ci95"])
        # Output differences remain observable even when no teacher exists.
        self.assertEqual(report["paired_prediction_gaps"]["TF"]["count"], 2)
        mixed = self.report(rows)["overall"]["models"]["imagined"]["verifier"]
        self.assertEqual(mixed["full_tf"]["metrics"]["count"], 4)
        self.assertEqual(mixed["full_tf"]["metrics"]["missing_count"], 2)
        self.assertEqual(mixed["probability"]["missing_count"], 2)
        self.assertEqual(mixed["margin"]["missing_count"], 2)
        source_missing = self.report([dict(rows[0], imagined_source_K=None)])["overall"]
        self.assertEqual(source_missing["models"]["imagined"]["verifier"]["k"]["count"], 1)
        self.assertEqual(source_missing["models"]["imagined"]["delta"]["count"], 0)
        delta_missing = self.report([dict(rows[0], delta_true=None)])["overall"]
        self.assertEqual(delta_missing["models"]["imagined"]["delta"]["count"], 0)
        self.assertIsNone(delta_missing["paired_mae_differences"]["delta_K"]
                          ["imagined_minus_prior"]["delta_macro_mae"])
        self.assertEqual(self.report([unknown])["cohorts"]["E_H1"]["parent_K_lt_L"]["count"], 0)

    def test_question_bootstrap_pairing_direction_reproducibility_and_weighting(self):
        rows = three_questions()
        pairs = self.report(rows)["overall"]["paired_mae_differences"]
        self.assertAlmostEqual(pairs["endpoint_K"]["imagined_minus_prior"]["delta_macro_mae"], -1.)
        self.assertEqual(pairs["endpoint_K"]["imagined_minus_direct"]["count"], 2)
        self.assertEqual(pairs["endpoint_K"]["imagined_minus_direct"]["missing_count"], 1)
        self.assertEqual(pairs["endpoint_K"]["imagined_minus_direct"]["delta_macro_mae"], 0.)
        self.assertEqual(pairs["delta_K"]["imagined_minus_prior"]["delta_macro_mae"], -.5)
        self.assertEqual(pairs["delta_K"]["imagined_minus_direct"]["delta_macro_mae"], .25)
        duplicated = rows[:1] * 20 + rows[1:]
        repeat = self.report(duplicated)["overall"]["paired_mae_differences"]
        for target in pairs:
            for baseline in pairs[target]:
                expected, actual = pairs[target][baseline], repeat[target][baseline]
                self.assertEqual(actual["delta_macro_mae"], expected["delta_macro_mae"])
                self.assertEqual(actual["ci95"], expected["ci95"])
                self.assertEqual(actual["question_count"], expected["question_count"])
        reversed_pairs = self.report(reversed(duplicated))["overall"]["paired_mae_differences"]
        self.assertEqual(repeat, reversed_pairs)
        self.assertEqual(pairs["endpoint_K"]["imagined_minus_prior"]["ci95"], [-2., -.5])
        gaps = self.report(rows)["overall"]["paired_prediction_gaps"]
        extra_gaps = self.report(duplicated)["overall"]["paired_prediction_gaps"]
        self.assertNotEqual(gaps["K"]["mae"], extra_gaps["K"]["mae"])
        for head in gaps:
            self.assertAlmostEqual(gaps[head]["question_macro_mae"], extra_gaps[head]["question_macro_mae"])
            for expected, actual in zip(gaps[head]["question_macro_mae_ci95"],
                                        extra_gaps[head]["question_macro_mae_ci95"]):
                self.assertAlmostEqual(expected, actual)

    def test_missing_vector_positions_do_not_restart_survival_or_count_as_zero(self):
        row = action_row()
        row["imagined_verifier"].update(hazards=[None, .5], tf_probs=[.4],
                                       probability_pred=None, margin_pred=[math.nan, .5])
        gaps = self.report([row])["overall"]["paired_prediction_gaps"]
        self.assertEqual((gaps["hazards"]["count"], gaps["hazards"]["missing_count"]), (1, 1))
        self.assertEqual(gaps["survival"]["count"], 0)
        self.assertIsNone(gaps["survival"]["mae"])
        self.assertEqual(gaps["TF"]["count"], 1)
        self.assertEqual(gaps["pV"]["count"], 0)
        self.assertEqual(gaps["margin"]["count"], 1)
        self.assertEqual(gaps["K"]["count"], 1)

    def test_empty_reports_and_invalid_contracts(self):
        empty = self.report([])
        self.assertEqual(empty["overall"]["status"], "empty")
        self.assertEqual(set(empty["by_horizon"]), {"H1", "H2", "H3"})
        self.assertEqual(empty["by_sequence"], {})
        json.dumps(empty, allow_nan=False)
        json.dumps(self.report(three_questions()), allow_nan=False)
        for overrides in ({"horizon": 2}, {"actions": "X"}, {"action": "E"},
                          {"accepted": 3}, {"parent_accepted": -1}, {"question": None},
                          {"parent_token_changed": 0}, {"imagined_verifier": []},
                          {"native_active_start": 3}, {"changed_positions": [2]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.report([dict(action_row(), **overrides)])
        with self.assertRaises(ValueError):
            phase0_report([], bootstrap_samples=0)

    def test_E_new_block_true_teachers_and_survival_keep_global_prefix(self):
        overall = self.report([block_row()])["overall"]
        regions = overall["regions"]
        frontier = regions["E_new_block"]
        self.assertEqual(frontier["position_count"], 8)
        self.assertEqual(regions["E_old_prefix"]["position_count"], 8)
        imagined = frontier["models"]["imagined"]
        self.assertEqual(imagined["hazard"]["metrics"]["count"], 3)
        self.assertEqual(imagined["hazard"]["metrics"]["positives"], 2)
        self.assertEqual(imagined["survival"]["metrics"]["count"], 8)
        survival = [.5 * .5**i for i in range(1, 9)]
        expected_brier = sum((p - int(i < 2))**2 for i, p in enumerate(survival)) / 8
        self.assertAlmostEqual(imagined["survival"]["metrics"]["brier"], expected_brier)
        self.assertAlmostEqual(imagined["full_tf"]["metrics"]["brier"], .81)
        self.assertEqual(imagined["full_tf"]["metrics"]["positives"], 0)
        self.assertEqual(imagined["full_tf"]["metrics"]["ranking_status"], "single_class")
        self.assertIsNone(imagined["full_tf"]["metrics"]["roc_auc"])
        self.assertAlmostEqual(imagined["probability"]["mae"], .7)
        self.assertEqual(imagined["margin"]["mae"], 2.)
        self.assertEqual(regions["E_old_prefix"]["models"]["imagined"]["full_tf"]["metrics"]["brier"], 0.)
        self.assertAlmostEqual(overall["models"]["imagined"]["verifier"]["full_tf"]["metrics"]["brier"], .405)
        gaps = frontier["paired_prediction_gaps"]
        expected_gap = sum(abs(a - .8 * .75**i) for i, a in enumerate(survival, 1)) / 8
        self.assertAlmostEqual(gaps["survival"]["mae"], expected_gap)
        self.assertAlmostEqual(gaps["hazards"]["mae"], .25)
        self.assertAlmostEqual(gaps["TF"]["mae"], .8)
        self.assertAlmostEqual(gaps["pV"]["mae"], .6)
        self.assertEqual(gaps["margin"]["mae"], 2.)
        for boundary in gaps["survival"]["question_macro_mae_ci95"]:
            self.assertAlmostEqual(boundary, expected_gap)

    def test_frontier_censoring_and_missing_truth_are_not_zero_filled(self):
        rejected = block_row("q1", accepted=3, parent_accepted=3)
        for model in ("oracle", "imagined", "prior", "direct"):
            rejected[model + "_verifier"]["tf_truth"] = [1] * 16
        unknown = block_row("q2", accepted=None, parent_accepted=None)
        for model in ("oracle", "imagined", "prior", "direct"):
            unknown[model + "_verifier"].update(tf_truth=None, probability_truth=None, margin_truth=None)
        broken_prefix = block_row("q3")
        broken_prefix["imagined_verifier"]["hazards"][0] = None
        report = self.report([rejected, unknown, broken_prefix])
        # First rejection in the prefix leaves no observed hazard labels on E's frontier.
        cohort = report["cohorts"]["E_H1"]["parent_K_lt_L"]["regions"]["E_new_block"]
        verifier = cohort["models"]["imagined"]
        self.assertEqual(verifier["hazard"]["metrics"]["count"], 0)
        self.assertEqual(verifier["hazard"]["metrics"]["missing_count"], 0)
        self.assertEqual(verifier["survival"]["metrics"]["count"], 8)
        self.assertEqual(verifier["survival"]["metrics"]["positives"], 0)
        # TF suffix labels remain observed and positive despite accepted K=3.
        self.assertEqual(verifier["full_tf"]["metrics"]["positives"], 8)
        self.assertAlmostEqual(verifier["full_tf"]["metrics"]["brier"], .01)
        full_prefix = report["cohorts"]["E_H1"]["parent_K_eq_L"]["regions"]["E_new_block"]
        self.assertEqual(full_prefix["models"]["imagined"]["hazard"]["metrics"]["count"], 3)
        self.assertEqual(full_prefix["models"]["imagined"]["survival"]["metrics"]["count"], 0)
        self.assertEqual(full_prefix["paired_prediction_gaps"]["survival"]["missing_count"], 8)
        self.assertEqual(full_prefix["paired_prediction_gaps"]["TF"]["count"], 8)
        overall = report["overall"]["regions"]["E_new_block"]["models"]["imagined"]
        self.assertEqual(overall["full_tf"]["metrics"]["count"], 16)
        self.assertEqual(overall["full_tf"]["metrics"]["missing_count"], 8)
        self.assertEqual(overall["probability"]["missing_count"], 8)
        self.assertEqual(overall["margin"]["missing_count"], 8)

    def test_composed_root_parent_and_R_changed_frontier_are_distinct_regions(self):
        rows = [block_row("q1", "EE"), block_row("q2", "EER"),
                block_row("q3", "R", accepted=12, parent_accepted=12, parent_length=16, changed=True)]
        report = self.report(rows)
        extension = report["by_sequence"]["EE"]["regions"]
        self.assertEqual(extension["E_new_block"]["position_count"], 8)
        self.assertEqual(extension["E_old_prefix"]["position_count"], 16)
        self.assertEqual(extension["new_since_parent"]["position_count"], 16)
        refinement = report["by_sequence"]["EER"]["regions"]
        self.assertEqual(refinement["R_frontier"]["position_count"], 8)
        self.assertEqual(refinement["R_old_prefix"]["position_count"], 16)
        self.assertEqual(refinement["new_since_parent"]["position_count"], 16)
        # Runner's change flag observes only the original 8-token source prefix.
        # It cannot identify changes at positions 16:24 after EER.
        self.assertEqual(refinement["R_changed_frontier"]["eligible_row_count"], 0)
        changed = report["cohorts"]["R_H1"]["changed"]["regions"]["R_changed_frontier"]
        self.assertEqual(changed["position_count"], 8)
        self.assertAlmostEqual(changed["models"]["imagined"]["full_tf"]["metrics"]["brier"], .81)
        self.assertAlmostEqual(changed["paired_prediction_gaps"]["TF"]["mae"], .8)
        # True K did not change; R's changed cohort must follow the native-token flag.
        self.assertEqual(report["cohorts"]["R_H1"]["gain"]["count"], 0)
        self.assertEqual(report["cohorts"]["R_H1"]["loss"]["count"], 0)
        json.dumps(report, allow_nan=False)

    def test_missing_or_inconsistent_region_geometry_is_explicitly_unavailable(self):
        for changes in ({"parent_length": None}, {"parent_length": 12}):
            with self.subTest(changes=changes):
                row = dict(block_row(), **changes)
                group = self.report([row])["overall"]
                self.assertEqual(group["models"]["imagined"]["verifier"]["k"]["count"], 1)
                region = group["regions"]["E_new_block"]
                self.assertEqual((region["status"], region["unavailable_count"]), ("unavailable", 1))
                self.assertEqual(region["position_count"], 0)
                self.assertIsNone(region["models"]["imagined"]["full_tf"]["metrics"]["brier"])
                self.assertIsNone(region["paired_prediction_gaps"]["survival"]["mae"])

    def test_exact_R_changed_positions_and_native_active_boundary_are_evaluation_only(self):
        changed = block_row("q1", "R", 12, 12, parent_length=16, changed=True)
        changed.update(changed_positions=[9, 12, 14, 12], native_active_start=8)
        unchanged = block_row("q2", "R", 12, 12, parent_length=16)
        unchanged.update(changed_positions=[], native_active_start=10)
        unknown = block_row("q3", "R", 12, 12, parent_length=16, changed=True)
        unknown["native_active_start"] = 8
        report = self.report([changed, unchanged, unknown])
        region = report["overall"]["regions"]["R_changed_positions"]
        self.assertEqual((region["position_count"], region["count"], region["unavailable_count"]), (3, 2, 1))
        verifier = region["models"]["imagined"]
        self.assertEqual(verifier["hazard"]["metrics"]["count"], 2)
        self.assertEqual(verifier["survival"]["metrics"]["count"], 3)
        probabilities = [.5 * .5**(i - 7) for i in (9, 12, 14)]
        expected_brier = ((probabilities[0] - 1)**2 + sum(p*p for p in probabilities[1:])) / 3
        self.assertAlmostEqual(verifier["survival"]["metrics"]["brier"], expected_brier)
        self.assertEqual(verifier["full_tf"]["metrics"]["count"], 3)
        self.assertAlmostEqual(verifier["full_tf"]["metrics"]["brier"], .81)
        gap = sum(abs(p - .8 * .75**(i - 7)) for i, p in zip((9, 12, 14), probabilities)) / 3
        self.assertAlmostEqual(region["paired_prediction_gaps"]["survival"]["mae"], gap)
        native = report["cohorts"]["R_H1"]["unchanged"]["regions"]
        self.assertEqual(native["R_frontier"]["position_count"], 6)
        self.assertEqual(native["R_old_prefix"]["position_count"], 10)
        self.assertEqual(native["R_changed_positions"]["status"], "empty_region")
        self.assertEqual(native["R_changed_positions"]["unavailable_count"], 0)
        self.assertIsNone(native["R_changed_positions"]["models"]["imagined"]["full_tf"]["metrics"]["brier"])
        composed = block_row("q1", "EER")
        composed.update(changed_positions=[], native_active_start=16)
        self.assertEqual(self.report([composed])["overall"]["regions"]["R_changed_positions"]["eligible_row_count"], 0)

    def test_writer_consumes_iterator_once_and_preserves_missing_csv_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports"
            inputs = three_questions()
            inputs[2] = block_row("q3", accepted=0, parent_accepted=0)
            inputs[2]["direct_verifier"]["expected_yield"] = None
            for model in ("oracle", "imagined", "prior", "direct"):
                inputs[2][model + "_verifier"]["tf_truth"] = None
            report = write_reports(iter(inputs), output, bootstrap_samples=120, seed=17)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "phase0_comparison.json", "challenge_cohorts.json",
                "paired_question_bootstrap.json", "phase0_comparison.csv"})
            comparison = json.loads((output / "phase0_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison, report)
            challenge = json.loads((output / "challenge_cohorts.json").read_text(encoding="utf-8"))
            self.assertEqual(challenge["cohorts"], report["cohorts"])
            bootstrap = json.loads((output / "paired_question_bootstrap.json").read_text(encoding="utf-8"))
            self.assertEqual(bootstrap["overall"], report["overall"]["paired_mae_differences"])
            self.assertEqual(bootstrap["by_sequence"]["E"], report["by_sequence"]["E"]["paired_mae_differences"])
            self.assertEqual(bootstrap["paired_prediction_gaps"]["overall"],
                             report["overall"]["paired_prediction_gaps"])
            self.assertEqual(bootstrap["regional_prediction_gaps"]["overall"]["E_new_block"],
                             report["overall"]["regions"]["E_new_block"]["paired_prediction_gaps"])
            with (output / "phase0_comparison.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            missing = next(row for row in rows if row["scope"] == "by_sequence"
                           and row["group"] == "E" and row["model"] == "direct")
            self.assertEqual(missing["K_mae"], "")
            self.assertEqual(missing["TF_brier"], "")
            frontier = next(row for row in rows if row["scope"] == "by_sequence" and row["group"] == "E"
                            and row["region"] == "E_new_block" and row["model"] == "imagined")
            self.assertEqual(frontier["position_count"], "8")
            self.assertEqual(frontier["TF_count"], "0")
            self.assertEqual(frontier["TF_brier"], "")
            self.assertEqual(frontier["survival_count"], "8")


if __name__ == "__main__":
    unittest.main()
