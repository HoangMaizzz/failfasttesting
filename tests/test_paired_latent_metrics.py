"""Scientific semantics and file-contract checks; no model dependencies."""

import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paired_latent_metrics import build_report, write_reports


def row(**overrides):
    result = dict(uid="u1", question="q1", seed=1, method="Direct", action="R",
                  length=8, parent_length=8, parent_accepted=2, accepted=2,
                  parent_token_changed=False, K_pred=3., q_pred=[.5] * 8,
                  hazard_pred=[.5] * 8, horizon=1, actions="R")
    result.update(overrides)
    return result


def cohort(report, name="all", group="H1", actions=None, length=None):
    node = report["groups"][group]
    if actions is not None:
        node = node["by_actions"][actions]
    if length is not None:
        node = node["by_length"][str(length)]
    return node["cohorts"][name]


def metrics(report, method="Direct", **kwargs):
    return cohort(report, **kwargs)["by_method"][method]


class PairedLatentMetricsTests(unittest.TestCase):
    def report(self, rows, samples=100, seed=42):
        return build_report(rows, bootstrap_samples=samples, seed=seed)

    def test_current_and_rollout_paths_and_exact_length_groups_are_separate(self):
        rows = [row(uid="current", horizon=0, actions="", action=""),
                row(uid="one"), row(uid="two", horizon=2, actions="ER"),
                row(uid="three", horizon=3, actions="RER"),
                row(uid="long", length=64, q_pred=[.5] * 64, hazard_pred=[.5] * 64)]
        report = self.report(rows)
        self.assertEqual(list(report["groups"]), ["current", "H1", "H2", "H3"])
        self.assertEqual([report["groups"][g]["row_count"] for g in report["groups"]], [1, 2, 1, 1])
        lengths = report["groups"]["H1"]["by_actions"]["R"]["by_length"]
        self.assertEqual(list(lengths), [str(n) for n in range(8, 65, 8)])
        self.assertEqual(lengths["64"]["row_count"], 1)
        self.assertEqual(lengths["16"]["status"], "empty")
        self.assertEqual(cohort(report, group="current")["observation_count"], 1)
        self.assertEqual(cohort(report, group="H2", actions="ER", length=8)["question_count"], 1)
        short = self.report([row(length=3, q_pred=[.5] * 3, hazard_pred=[.5] * 3)])
        self.assertIn("3", short["groups"]["H1"]["by_actions"]["R"]["by_length"])

    def test_counts_do_not_multiply_observations_by_methods_or_seeds(self):
        rows = [row(method=m, seed=s) for m in ("Direct", "Bridge_state") for s in (1, 2)]
        report = self.report(rows)
        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["observation_count"], 1)
        self.assertEqual(report["observation_seed_count"], 2)
        self.assertEqual(report["question_count"], 1)
        self.assertEqual(metrics(report)["observation_seed_count"], 2)

    def test_endpoint_error_statistics_and_question_macro(self):
        rows = [row(uid="a", accepted=0, K_pred=0),
                row(uid="b", accepted=0, K_pred=2),
                row(uid="c", question="q2", accepted=8, K_pred=2)]
        result = metrics(self.report(rows))["k"]
        self.assertAlmostEqual(result["row_mean_mae"], 8 / 3)
        self.assertEqual(result["question_macro_mae"], 3.5)
        self.assertEqual(result["median"], 2)
        self.assertAlmostEqual(result["p90"], 5.2)
        self.assertEqual(result["within1"], 1 / 3)
        self.assertEqual(result["within2"], 2 / 3)
        self.assertEqual(result["within4"], 2 / 3)
        self.assertAlmostEqual(result["bias"], -4 / 3)
        self.assertEqual(result["under_count"], 1)
        self.assertEqual(result["over_count"], 1)
        self.assertEqual(result["under_rate"], 1 / 3)
        self.assertEqual(result["over_rate"], 1 / 3)

    def test_cumulative_survival_is_not_multiplied_again_and_hazards_are_censored(self):
        result = metrics(self.report([row(q_pred=[.9, .8, .7, .6, .5, .4, .3, .2],
                                         hazard_pred=[.9, .8, .1, .99, .99, .99, .99, .99])]))
        expected = (.1 ** 2 + .2 ** 2 + sum(p ** 2 for p in [.7, .6, .5, .4, .3, .2])) / 8
        self.assertAlmostEqual(result["survival"]["metrics"]["brier"], expected)
        self.assertEqual(result["survival"]["metrics"]["roc_auc"], 1.)
        self.assertEqual(sum(b["count"] for b in result["survival"]["calibration"]), 8)
        hazard = result["hazard"]
        self.assertEqual(hazard["risk_position_count"], 3)
        self.assertEqual(hazard["censored_position_count"], 5)
        self.assertEqual(hazard["metrics"]["positives"], 2)
        self.assertAlmostEqual(hazard["metrics"]["brier"], .02)
        full = metrics(self.report([row(accepted=8)]))["hazard"]
        self.assertEqual(full["risk_position_count"], 8)
        self.assertEqual(full["censored_position_count"], 0)

    def test_new_block_uses_global_survival_and_global_hazard_risk(self):
        item = row(action="E", actions="E", length=16, parent_length=8,
                   parent_accepted=8, accepted=10, K_pred=10,
                   q_pred=[.9] * 8 + [.2] * 8, hazard_pred=[.9] * 8 + [.8] * 8)
        result = metrics(self.report([item]), name="E_new_block")
        self.assertEqual(result["position_count"], 8)
        # Global labels: two accepted positions, six rejected positions.
        self.assertAlmostEqual(result["survival"]["metrics"]["brier"], .19)
        self.assertEqual(result["survival"]["metrics"]["positives"], 2)
        self.assertEqual(result["hazard"]["risk_position_count"], 3)
        self.assertEqual(result["hazard"]["censored_position_count"], 5)
        self.assertAlmostEqual(result["hazard"]["metrics"]["brier"], .24)
        # A rejected old prefix does not restart the conditional risk set.
        item.update(accepted=3, parent_accepted=3)
        rejected = metrics(self.report([item]), name="E_new_block")
        self.assertAlmostEqual(rejected["survival"]["metrics"]["brier"], .04)
        self.assertEqual(rejected["hazard"]["risk_position_count"], 0)
        self.assertEqual(rejected["hazard"]["censored_position_count"], 8)
        # The supplied span, not a hardcoded final eight-token block, is scored.
        item.update(length=24, q_pred=[.2] * 24, hazard_pred=[.8] * 24)
        self.assertEqual(metrics(self.report([item]), name="E_new_block")["position_count"], 16)

    def test_E_cohorts_overlap_and_missing_parent_truth_stays_unclassified(self):
        base = row(action="E", actions="E", parent_length=4)
        rows = [dict(base, uid="full", parent_accepted=4),
                dict(base, uid="rejected", parent_accepted=3),
                dict(base, uid="unknown", parent_accepted=None),
                dict(base, uid="no_span", parent_length=None, parent_accepted=None)]
        report = self.report(rows)
        self.assertEqual(cohort(report)["row_count"], 4)
        self.assertEqual(cohort(report, "E_all")["row_count"], 4)
        self.assertEqual(cohort(report, "E_new_block")["row_count"], 3)
        self.assertEqual(cohort(report, "E_full_prefix")["row_count"], 1)
        self.assertEqual(cohort(report, "E_rejected_prefix")["row_count"], 1)
        self.assertEqual(cohort(report, "R")["status"], "empty")
        self.assertIsNone(metrics(report, name="R")["k"]["mae"])
        alias = dict(base, parent_K=4)
        del alias["parent_accepted"]
        self.assertEqual(cohort(self.report([alias]), "E_full_prefix")["row_count"], 1)
        self.assertIn("evaluate-only", report["semantics"]["E_full_prefix"])

    def test_oracle_disagreement_and_truth_error_are_nonadditive(self):
        report = self.report([row(accepted=4, K_pred=3, oracle_K_pred=6, latent_mse=.25)])
        result = metrics(report)
        self.assertEqual(result["k"]["mae"], 1)
        self.assertEqual(result["oracle"]["output_disagreement"]["mae"], 3)
        self.assertEqual(result["oracle"]["true_k"]["mae"], 2)
        self.assertFalse(report["semantics"]["decomposition_additive"])
        self.assertFalse(result["oracle"]["decomposition_additive"])
        self.assertEqual(result["latent_mse"]["mean"], .25)
        # Disagreement has paired prediction support even without true labels.
        missing = metrics(self.report([row(accepted=None, K_pred=3, oracle_K_pred=6)]))
        self.assertEqual(missing["oracle"]["output_disagreement"]["count"], 1)
        self.assertEqual(missing["oracle"]["true_k"]["count"], 0)
        self.assertIsNone(missing["k"]["mae"])

    def test_missing_and_nonfinite_labels_are_never_zero(self):
        rows = [row(uid="none", accepted=None, parent_accepted=None),
                row(uid="nan", accepted=math.nan, parent_accepted=math.inf),
                row(uid="zero", accepted=0, K_pred=None, q_pred=None, hazard_pred=None)]
        report = self.report(rows)
        result = metrics(report)
        self.assertEqual(result["k"]["status"], "missing")
        self.assertEqual(result["k"]["count"], 0)
        self.assertEqual(result["survival"]["metrics"]["count"], 0)
        self.assertEqual(result["survival"]["metrics"]["missing_count"], 24)
        self.assertEqual(result["survival"]["metrics"]["status"], "missing")
        self.assertEqual(result["hazard"]["unknown_risk_position_count"], 16)
        self.assertEqual(result["hazard"]["risk_position_count"], 1)
        self.assertEqual(result["hazard"]["metrics"]["missing_count"], 1)
        json.dumps(report, allow_nan=False)

    def test_paired_bootstrap_averages_seeds_then_paths_then_questions(self):
        rows = []
        for uid, question, seeds, a, b in (("many", "q1", (1, 2, 3), 3, 0),
                                           ("few", "q1", (1,), 0, 3),
                                           ("other", "q2", (1,), 2, 0)):
            for seed in seeds:
                rows += [row(uid=uid, question=question, seed=seed, method="Bridge_state", accepted=0, K_pred=a),
                         row(uid=uid, question=question, seed=seed, method="Direct", accepted=0, K_pred=b)]
        report = build_report(rows)
        paired = cohort(report)["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(paired["matched_count"], 5)
        self.assertEqual(paired["scored_count"], 5)
        self.assertEqual(paired["path_count"], 3)
        self.assertEqual(paired["question_count"], 2)
        self.assertEqual(paired["delta_abs_error"], 1)
        self.assertEqual([u["delta_abs_error"] for u in paired["question_means"]], [0, 2])
        self.assertEqual(paired["ci95"], [0, 2])
        self.assertEqual(paired["bootstrap_samples"], 2000)
        self.assertEqual(paired["seed"], 42)
        # Duplicate seed support of just one path must not change its weight.
        rows += [row(uid="many", seed=4, method=m, accepted=0, K_pred=k)
                 for m, k in (("Bridge_state", 3), ("Direct", 0))]
        repeated = build_report(reversed(rows))
        repeated_pair = cohort(repeated)["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(repeated_pair["delta_abs_error"], paired["delta_abs_error"])
        self.assertEqual(repeated_pair["ci95"], paired["ci95"])
        self.assertEqual(repeated_pair["question_count"], 2)
        self.assertEqual(build_report(rows), repeated)

    def test_question_percentile_CI_uses_same_random_seed_for_comparisons(self):
        rows = []
        for i, prediction in enumerate((1., 2., 5.)):
            rows += [row(uid=f"u{i}", question=f"q{i}", method=m, accepted=0, K_pred=k)
                     for m, k in (("Direct", 0.), ("Bridge_state", prediction),
                                  ("Bridge_behavior", prediction))]
        comparisons = cohort(self.report(rows, samples=201, seed=19))["paired_comparisons"]
        a, b = (comparisons[name] for name in ("Bridge_state_vs_Direct", "Bridge_behavior_vs_Direct"))
        self.assertEqual(a["ci95"], b["ci95"])
        rng = random.Random(19)
        deltas = [1., 2., 5.]
        draws = sorted(sum(deltas[rng.randrange(3)] for _ in range(3)) / 3 for _ in range(201))
        self.assertEqual(a["ci95"], [draws[5], draws[195]])

    def test_negative_paired_delta_and_seed_specific_truth(self):
        rows = [row(seed=s, method=m, accepted=k, K_pred=p)
                for s, k, predictions in ((1, 4, (4, 7)), (2, 2, (2, 3)))
                for m, p in zip(("Bridge_state", "Direct"), predictions)]
        paired = cohort(self.report(rows))["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(paired["delta_abs_error"], -2)
        self.assertEqual(paired["ci95"], [-2, -2])
        self.assertEqual(paired["question_count"], 1)
        self.assertEqual(paired["path_count"], 1)
        self.assertEqual(paired["path_means"][0]["matched_seed_count"], 2)

    def test_missing_label_on_one_method_is_not_borrowed_and_partial_q_stays_global(self):
        rows = [row(method="Bridge_state", accepted=None), row()]
        paired = cohort(self.report(rows))["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(paired["matched_count"], 1)
        self.assertEqual(paired["scored_count"], 0)
        self.assertEqual(paired["missing_label_count"], 1)
        self.assertEqual(paired["status"], "missing")
        self.assertIsNone(paired["ci95"])
        result = metrics(self.report([row(accepted=0, q_pred=[None, math.nan] + [.2] * 6,
                                         hazard_pred=[math.inf] + [.5] * 7)]))
        self.assertEqual(result["survival"]["metrics"]["count"], 6)
        self.assertEqual(result["survival"]["metrics"]["missing_count"], 2)
        self.assertAlmostEqual(result["survival"]["metrics"]["brier"], .04)
        self.assertEqual(result["hazard"]["metrics"]["count"], 0)

    def test_matching_uses_uid_seed_horizon_actions_and_reports_missing_support(self):
        rows = [row(method="Bridge_state", uid="shared", K_pred=3),
                row(method="Direct", uid="shared", K_pred=1),
                row(method="Bridge_state", uid="a_only"), row(uid="b_only"),
                row(method="Bridge_state", uid="seed", seed=1), row(uid="seed", seed=2),
                row(method="Bridge_state", uid="missing", accepted=None),
                row(uid="missing", accepted=None),
                row(method="Bridge_state", uid="pred", K_pred=None), row(uid="pred"),
                row(method="Bridge_state", uid="shared", horizon=0, actions="", action=""),
                row(method="Direct", uid="shared", horizon=2, actions="RR")]
        report = self.report(rows)
        paired = cohort(report)["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(paired["matched_count"], 3)
        self.assertEqual(paired["scored_count"], 1)
        self.assertEqual(paired["unmatched_a_count"], 2)
        self.assertEqual(paired["unmatched_b_count"], 2)
        self.assertEqual(paired["missing_label_count"], 1)
        self.assertEqual(paired["missing_prediction_count"], 1)
        # Both methods have error 1, despite their output disagreement of 2.
        self.assertEqual(paired["delta_abs_error"], 0)
        self.assertEqual(paired["ci95"], [0, 0])
        current = cohort(report, group="current")["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(current["matched_count"], 0)
        self.assertEqual(current["status"], "missing_method")
        paths = [row(method="Bridge_state", uid="path", horizon=2, actions="ER"),
                 row(uid="path", horizon=2, actions="RR")]
        path_pair = cohort(self.report(paths), group="H2")["paired_comparisons"]["Bridge_state_vs_Direct"]
        self.assertEqual(path_pair["status"], "unmatched")
        self.assertEqual(path_pair["matched_count"], 0)

    def test_all_required_and_optional_comparisons_and_seed_variation(self):
        methods = ("Direct", "Direct_distill", "Bridge_state", "Bridge_behavior",
                   "V_joint", "V_hidden_only", "V_joint_probe")
        rows = [row(method=m, seed=s, K_pred=k) for m in methods for s, k in ((1, 3), (2, 5))]
        report = self.report(rows)
        comparisons = cohort(report)["paired_comparisons"]
        self.assertEqual(set(comparisons), {
            "Bridge_state_vs_Direct", "Bridge_behavior_vs_Direct", "Direct_distill_vs_Direct",
            "Bridge_behavior_vs_Bridge_state", "V_joint_vs_V_hidden_only", "V_joint_probe_vs_V_joint"})
        self.assertTrue(all(v["seed"] == 42 for v in comparisons.values()))
        variation = report["seed_variation"]["Direct"]["H1"]
        self.assertEqual(variation["observed_seed_count"], 2)
        self.assertEqual(variation["mean_question_macro_mae"], 2)
        self.assertAlmostEqual(variation["std_question_macro_mae"], math.sqrt(2))
        single = metrics(self.report([row()]))["seed_variation"]
        self.assertIsNone(single["std_question_macro_mae"])
        self.assertNotIn("V_joint_vs_V_hidden_only", cohort(self.report([row()]))["paired_comparisons"])

    def test_empty_reports_and_missing_metrics_have_explicit_statuses(self):
        report = self.report([])
        self.assertEqual(report["status"], "empty")
        for node in report["groups"].values():
            self.assertEqual(node["status"], "empty")
            for values in node["cohorts"].values():
                self.assertEqual(values["status"], "empty")
                self.assertTrue(all(p["status"] == "empty" and p["ci95"] is None
                                    for p in values["paired_comparisons"].values()))
        missing = metrics(self.report([row()]))
        self.assertEqual(missing["oracle"]["true_k"]["status"], "missing")
        self.assertEqual(missing["latent_mse"]["status"], "missing")
        self.assertEqual(metrics(self.report([row()]), name="E_all")["oracle"]["true_k"]["status"], "empty")

    def test_writer_roundtrip_generator_and_input_preservation(self):
        rows = [row(), row(method="Bridge_state")]
        before = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as directory:
            report = write_reports((r for r in rows), directory, bootstrap_samples=11, seed=7)
            self.assertEqual(rows, before)
            self.assertEqual(set(report["paths"]), {
                "comparison.csv", "metrics.json", "paired_question_bootstrap.json"})
            self.assertEqual({p.name for p in Path(directory).iterdir()}, set(report["paths"]))
            for path in report["paths"].values():
                self.assertTrue(Path(path).is_absolute())
            saved = json.loads(Path(report["paths"]["metrics.json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            bootstrap = json.loads(Path(report["paths"]["paired_question_bootstrap.json"]).read_text(encoding="utf-8"))
            main = next(g for g in bootstrap["groups"] if g["group"] == "H1" and g["cohort"] == "all"
                        and g["actions"] is None and g["length"] is None)
            self.assertEqual(main["comparisons"], cohort(report)["paired_comparisons"])
            with Path(report["paths"]["comparison.csv"]).open(encoding="utf-8", newline="") as stream:
                table = list(csv.DictReader(stream))
            method = next(r for r in table if r["group"] == "H1" and r["cohort"] == "all"
                          and r["method"] == "Direct" and r["actions"] == "" and r["length"] == "")
            self.assertEqual(method["K_mae"], "1.0")
            self.assertEqual(method["decomposition_additive"], "False")
            self.assertTrue(any(r["kind"] == "paired" for r in table))
            self.assertFalse(any("teacherforcer" in key for r in table for key in r))

    def test_invalid_contract_and_duplicate_pair_keys_fail_explicitly(self):
        invalid = ({"uid": None}, {"question": None}, {"seed": None}, {"method": ""},
                   {"horizon": 4}, {"horizon": 0}, {"actions": "RR"}, {"action": "E"},
                   {"length": 0}, {"accepted": 9}, {"accepted": -1}, {"accepted": .5},
                   {"parent_accepted": 9}, {"parent_K": 3}, {"parent_token_changed": 1},
                   {"q_pred": [.5]}, {"hazard_pred": [.5] * 9},
                   {"q_pred": [.5, .6] + [.4] * 6}, {"hazard_pred": [1.1] * 8},
                   {"latent_mse": -1})
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.report([row(**overrides)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.report([row(), row()])
        for overrides in ({"question": "different"}, {"accepted": 3}, {"length": 16, "q_pred": [.5] * 16, "hazard_pred": [.5] * 16}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "inconsistent"):
                self.report([row(), row(method="Bridge_state", **overrides)])
        with self.assertRaises(ValueError):
            self.report([], samples=0)


if __name__ == "__main__":
    unittest.main()
