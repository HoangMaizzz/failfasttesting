"""Paired, seed-clustered behavior diagnostics; no inference dependencies."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_aware_metrics import behavior_report, write_behavior_reports


def predictions():
    rows = []
    for seed in (42, 43, 44):
        for question, action, parent, child in (("q1", "R", 4, 6), ("q2", "R", 6, 4), ("q3", "E", 8, 10)):
            length = 16 if action == "E" else 8
            oracle = child - 2
            for method, gap, profile in (("A", 2., .6), ("B1", 1., .7), ("B2", 2., .75), ("C", .5, .65)):
                rows.append(dict(method=method, seed=seed, question_id=question, edge_id=question + action,
                    action=action, actions=action, horizon=1, parent_length=8, child_length=length,
                    K_parent_true=parent, K_child_true=child, delta_K_true=child-parent,
                    K_parent_emulator=float(parent)-.5, K_oracle_latent=float(oracle), K_pred=oracle+gap,
                    q_oracle=[.8]*length, q_pred=[profile]*length,
                    q_qwen_truth=None if question == "q3" else [int(i < child) for i in range(length)],
                    hazards_oracle=[.9]*length, hazards_pred=[.9]*length,
                    state_mse=None if method == "C" else .2, state_cos=None if method == "C" else .8,
                    R_changed=action == "R", R_gain=child > parent, R_loss=child < parent,
                    E_full_prefix=parent == 8 if action == "E" else None,
                    region_mask_new_block=[False]*8 + ([True]*8 if action == "E" else []),
                    state_fidelity_passed=True, selection_status="selected"))
    return rows


def config(**changes):
    return dict(seeds=[42, 43, 44], bootstrap_samples=200, bootstrap_seed=19,
                selection={str(seed): {"state_gate_passed": True, "selection_split": "validation"}
                           for seed in (42, 43, 44)}, **changes)


class BehaviorMetricsTests(unittest.TestCase):
    def test_smoke_cannot_claim_primary_success(self):
        report=behavior_report(predictions(),config(pipeline_check_only=True))
        self.assertFalse(report['gates']['B1']['passed'])
        self.assertTrue(report['gates']['B1']['pipeline_check_only'])

    def test_true_endpoint_delta_identity_and_emulator_fidelity_are_separate(self):
        rows = predictions()
        before = copy.deepcopy(rows)
        report = behavior_report(iter(rows), config())
        self.assertEqual(rows, before)
        methods = report["H1_ALL"]["methods"]
        self.assertEqual(methods["A"]["metrics"]["losses"]["K_child_truth_mae"]["question_macro_mean"], 0.)
        self.assertEqual(methods["B1"]["metrics"]["losses"]["K_child_truth_mae"]["question_macro_mean"], 1.)
        self.assertEqual(methods["A"]["metrics"]["losses"]["emulator_K_gap"]["question_macro_mean"], 2.)
        self.assertEqual(methods["B1"]["metrics"]["losses"]["emulator_K_gap"]["question_macro_mean"], 1.)
        for method in methods.values():
            metrics = method["metrics"]
            self.assertEqual(metrics["anchored_delta"]["mae"], metrics["endpoint_K"]["mae"])
        self.assertTrue(report["semantics"]["anchored_delta_is_not_independent_endpoint_evidence"])
        first = behavior_report(rows[:4], config())["H1_ALL"]["methods"]["A"]["metrics"]
        self.assertEqual(first["anchored_delta"]["mae"], 0.)
        self.assertEqual(first["deployed_delta"]["mae"], .5)

    def test_supplied_survival_is_not_recumulated_and_missing_Qwen_is_missing(self):
        report = behavior_report(predictions(), config())
        fidelity = report["H1_QWEN_FIDELITY"]["methods"]["A"]["metrics"]
        self.assertEqual(fidelity["qwen_survival"]["metrics"]["count"], 48)
        self.assertEqual(fidelity["qwen_survival"]["metrics"]["missing_count"], 48)
        self.assertAlmostEqual(fidelity["losses"]["survival_profile_gap"]["mean"], .2)
        new = report["H1_E_COHORTS"]["new_block"]["methods"]["A"]["metrics"]
        self.assertEqual(new["qwen_survival"]["metrics"]["count"], 0)
        self.assertIsNone(new["qwen_survival"]["metrics"]["brier"])
        self.assertAlmostEqual(new["losses"]["new_block_survival_gap"]["mean"], .2)
        self.assertIsNone(new["losses"]["new_block_Qwen_brier"]["mean"])
        self.assertEqual(report["H1_R_COHORTS"]["nonzero"]["count"], 24)
        self.assertEqual(report["H1_R_COHORTS"]["gain"]["count"], 12)
        self.assertEqual(report["H1_R_COHORTS"]["loss"]["count"], 12)

    def test_bootstrap_averages_edges_then_matched_seeds_then_questions(self):
        rows = []
        for seed, value in zip((42, 43, 44), (1., 5., 9.)):
            for question, gap in (("q1", value), ("q2", 2.)):
                for method in ("A", "B1"):
                    row = next(r for r in predictions() if r["seed"] == seed and r["question_id"] == question and r["method"] == method).copy()
                    row.update(K_parent_true=0, K_child_true=0, delta_K_true=0, K_oracle_latent=0.,
                               K_pred=0. if method == "A" else gap, R_gain=False, R_loss=False)
                    rows.append(row)
        for index in range(20):
            for row in rows[:2]:
                rows.append(dict(row, edge_id="extra" + str(index)))
        report = behavior_report(rows, config())
        result = report["QUESTION_BOOTSTRAP"]["primary_selected"]["H1_ALL"]["B1_minus_A"]["K_child_truth_mae"]
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["paired_seed_count"], 3)
        self.assertEqual(result["mean_difference"], 3.5)
        self.assertEqual(result["ci95"], [2., 5.])
        self.assertEqual(result["per_question_seed_counts"], {"q1": 3, "q2": 3})
        reverse = behavior_report(reversed(rows), config())["QUESTION_BOOTSTRAP"]
        self.assertEqual(reverse, report["QUESTION_BOOTSTRAP"])
        variation = report["SEED_VARIATION"]["H1_ALL"]["B1"]["K_child_truth_mae"]
        self.assertEqual(variation["by_seed"], {"42": 1.5, "43": 3.5, "44": 5.5})
        # Above seed 42 includes q1 extra edges plus its original edge, still mean=1.
        self.assertAlmostEqual(variation["mean"], 3.5)
        self.assertAlmostEqual(variation["std"], 2.)

    def test_failed_state_gate_remains_diagnostic_even_if_marked_selected(self):
        rows = predictions()
        for row in rows:
            if row["method"] == "B1":
                row["state_fidelity_passed"] = False
        report = behavior_report(rows, config())
        method = report["H1_ALL"]["methods"]["B1"]
        self.assertEqual(method["selection"]["selected_count"], 0)
        self.assertEqual(method["selection"]["diagnostic_only_count"], 9)
        self.assertFalse(report["gates"]["B1"]["passed"])
        primary = report["QUESTION_BOOTSTRAP"]["primary_selected"]["H1_ALL"]["B1_minus_A"]["emulator_K_gap"]
        self.assertEqual(primary["question_count"], 0)
        diagnostic = report["QUESTION_BOOTSTRAP"]["diagnostic_all_predictions"]["H1_ALL"]["B1_minus_A"]["emulator_K_gap"]
        self.assertLess(diagnostic["mean_difference"], 0.)
        for row in rows:
            if row["method"] == "B1":
                row.update(state_fidelity_passed=True, selection_status="diagnostic_only")
        self.assertFalse(behavior_report(rows, config())["gates"]["B1"]["passed"])
        self.assertTrue(behavior_report(predictions(), config())["gates"]["B1"]["passed"])

    def test_free_rollout_keeps_sequences_and_C_is_H1_reference(self):
        rows = predictions()
        for sequence in ("RE", "ER", "RER"):
            for row in predictions()[:3]:
                copy_row = dict(row, actions=sequence, action=sequence[-1], horizon=len(sequence),
                                edge_id=row["edge_id"] + sequence)
                rows.append(copy_row)
        report = behavior_report(rows, config())
        rollout = report["FREE_ROLLOUT_H1_H2_H3"]
        self.assertEqual(set(rollout["by_sequence"]), {"R", "E", "RE", "ER", "RER"})
        self.assertEqual(rollout["by_horizon"]["H2"]["methods"]["C"]["count"], 0)
        self.assertIn("reference", report["semantics"]["C_role"])
        self.assertNotIn("ceiling", report["semantics"]["C_role"])

    def test_runner_selection_metadata_cannot_override_failed_or_unsupported_flags(self):
        entries = [dict(method=method, seed=seed, status="selected", checkpoint="chosen.pt",
                        state_fidelity_passed=True) for method in ("A", "B1", "B2", "C") for seed in (42, 43, 44)]
        cfg = config()
        cfg["selection"] = dict(A=[e for e in entries if e["method"] == "A"], final=entries,
                                lambda_selection={"B1": {"status": "selected", "state_fidelity_passed": True},
                                                  "B2": {"status": "selected", "state_fidelity_passed": True}})
        self.assertTrue(behavior_report(predictions(), cfg)["gates"]["B1"]["passed"])
        for changes in ({"status": "diagnostic_only"}, {"status": "unsupported"},
                        {"state_fidelity_passed": False}, {"state_fidelity_passed": None}):
            with self.subTest(changes=changes):
                bad = copy.deepcopy(cfg)
                for entry in bad["selection"]["final"]:
                    if entry["method"] == "B1":
                        entry.update(changes)
                report = behavior_report(predictions(), bad)
                self.assertFalse(report["gates"]["B1"]["passed"])
                self.assertEqual(report["H1_ALL"]["methods"]["B1"]["selection"]["selected_count"], 0)
        for value in (None, "unsupported", {"status": "diagnostic_only"}):
            bad = copy.deepcopy(cfg)
            bad["selection"]["lambda_selection"]["B1"] = value
            self.assertFalse(behavior_report(predictions(), bad)["gates"]["B1"]["passed"])
        missing = copy.deepcopy(cfg)
        missing["selection"]["final"] = [e for e in entries if e["method"] != "B1"]
        self.assertFalse(behavior_report(predictions(), missing)["gates"]["B1"]["passed"])

    def test_writer_joint_predictions_and_strict_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = write_behavior_reports(iter(predictions()), directory, config())
            output = Path(directory)
            expected = {key + ".json" for key in report if key.startswith("H1_") or key in
                        ("QUESTION_BOOTSTRAP", "SEED_VARIATION", "FREE_ROLLOUT_H1_H2_H3")}
            self.assertEqual({p.name for p in output.iterdir()}, expected | {"FINAL_REPORT.md", "joint_predictions.jsonl"})
            joint = [json.loads(line) for line in (output / "joint_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(joint), 9)
            self.assertTrue(all(all("K_" + method in row for method in ("A", "B1", "B2", "C")) for row in joint))
            missing = next(row for row in joint if row["question_id"] == "q3")
            self.assertIsNone(missing["predictions"]["A"]["q_qwen_truth"])
            text = (output / "FINAL_REPORT.md").read_text()
            self.assertIn("exploratory", text.lower())
            self.assertIn("not independent", text.lower())
            json.dumps(report, allow_nan=False)

    def test_bad_pair_contracts_and_missing_endpoint_labels(self):
        rows = predictions()[:4]
        with self.assertRaises(ValueError):
            behavior_report(rows + rows[:1], config())
        with self.assertRaises(ValueError):
            behavior_report([dict(rows[0], method="C", horizon=2, actions="RR")], config())
        with self.assertRaises(ValueError):
            behavior_report([dict(rows[0], region_mask_new_block=[True])], config())
        for row in rows:
            row.update(K_parent_true=None, K_child_true=None, delta_K_true=None, q_qwen_truth=None,
                       R_gain=None, R_loss=None)
        metrics = behavior_report(rows, config())["H1_ALL"]["methods"]["A"]["metrics"]
        self.assertEqual(metrics["endpoint_K"]["count"], 0)
        self.assertIsNone(metrics["endpoint_K"]["mae"])
        self.assertEqual(metrics["anchored_delta"]["count"], 0)
        self.assertEqual(metrics["qwen_survival"]["metrics"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
