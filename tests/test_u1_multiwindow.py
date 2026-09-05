import json
import ast
import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController, FEATURE_SCHEMAS
from run_u1_multiwindow_int8 import parse_args, problem_command
from run_u1_sgd_ablation import selected_ids


def controller(**kwargs):
    return OnlineTDRefinementController(AdaptiveTDConfig(
        feature_dim=6, feature_schema="otrc_v2_2_compact_td",
        credit_assignment="hindsight_delta_j_logistic_f2",
        policy_mode="hindsight_delta_j_logistic_f2",
        hindsight_logistic_replay_stop_to_continue_ratio=0.0, **kwargs))


def history(delta, n=100):
    return [{"features": [1., 0.5, 0.], "delta_j": delta,
             "behavior_continue_probability": .02} for _ in range(n)]


class MultiwindowTest(unittest.TestCase):
    def test_empty_or_harmful_history_stops(self):
        c = controller()
        self.assertEqual(c._get_dynamic_logistic_threshold()[0], 1.)
        c.hindsight_logistic_calibration_history = history(2.)
        self.assertEqual(c._get_dynamic_logistic_threshold()[0], 1.)

    def test_profitable_tied_scores_all_selected(self):
        c = controller()
        c.hindsight_logistic_calibration_history = history(-2.)
        threshold, diag = c._get_dynamic_logistic_threshold()
        self.assertEqual(threshold, 0.)
        self.assertEqual(diag["dynamic_threshold_w50_selected_count"], 50)
        self.assertAlmostEqual(diag["dynamic_threshold_w50_effective_n"], 50.)

    def test_one_harmful_window_vetoes_continue(self):
        c = controller()
        c.hindsight_logistic_calibration_history = history(-10., 50) + history(1., 50)
        threshold, diag = c._get_dynamic_logistic_threshold()
        self.assertEqual(threshold, 1.)
        self.assertTrue(diag["dynamic_threshold_w100_valid"])
        self.assertFalse(diag["dynamic_threshold_w50_valid"])

    def test_rescores_using_current_weights(self):
        c = controller(hindsight_logistic_dynamic_min_selected=7)
        rows = history(-4., 7) + [dict(r, features=[1., 0., 0.]) for r in history(5., 7)]
        c.hindsight_logistic_model.weights = [0., 4., 0.]
        good = c._estimate_dynamic_logistic_threshold_window(rows)[1]
        self.assertTrue(good["valid"])
        c.hindsight_logistic_model.weights = [0., -4., 0.]
        self.assertFalse(c._estimate_dynamic_logistic_threshold_window(rows)[1]["valid"])

    def test_json_checkpoint_preserves_learning_and_rng(self):
        c = controller()
        c.hindsight_logistic_model.update([1., .5, 0.], 1, 2.)
        c.hindsight_logistic_calibration_history = history(-2.)
        c.hindsight_logistic_replay_buffer = [([1., .5, 0.], 1, 2.)]
        c.hindsight_logistic_positive_problem_ids = {2, 42}
        c.hindsight_delta_j_continue_count = 13
        c.rng.random()
        c.hindsight_logistic_replay_rng.random()
        snapshot = c.snapshot()
        snapshot["logistic_boundary_checkpoint"] = c.save_logistic_boundary_checkpoint()
        restored = controller()
        restored.load_snapshot(json.loads(json.dumps(snapshot)))
        self.assertEqual(c._get_dynamic_logistic_threshold(), restored._get_dynamic_logistic_threshold())
        self.assertEqual(c.rng.random(), restored.rng.random())
        self.assertEqual(c.hindsight_logistic_replay_rng.random(), restored.hindsight_logistic_replay_rng.random())
        for model in (c.hindsight_logistic_model, restored.hindsight_logistic_model):
            model.update_batch([([1., .5, 0.], 0, 3.)])
        self.assertEqual(vars(c.hindsight_logistic_model), vars(restored.hindsight_logistic_model))
        self.assertEqual(restored.hindsight_logistic_positive_problem_ids, {2, 42})
        self.assertEqual(restored.hindsight_delta_j_continue_count, 13)

    def test_runner_int8_fp16_drafter_same_ids(self):
        with patch.object(sys, "argv", ["test"]):
            args = parse_args()
        for dataset in args.datasets:
            ids = selected_ids(args, dataset)
            self.assertEqual(len(set(ids)), 100)
            cmd = problem_command(args, dataset, ids[0], Path("out"), None)
            self.assertEqual(cmd[cmd.index("--target_quantization") + 1], "int8")
            self.assertEqual(cmd[cmd.index("--unquantized_dtype") + 1], "float16")
            self.assertNotIn("--adaptive-hindsight-logistic-threshold-mode", cmd)
            self.assertIn("--adaptive-hindsight-logistic-dynamic-threshold", cmd)
            parser = argparse.ArgumentParser()
            tree = ast.parse(Path("failfast.py").read_text(encoding="utf-8"))
            for node in tree.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "parser"
                    and node.value.func.attr == "add_argument"):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "failfast.py", "exec"),
                         {"parser": parser, "argparse": argparse, "FEATURE_SCHEMAS": FEATURE_SCHEMAS})
            parsed = parser.parse_args(cmd[3:])
            self.assertTrue(parsed.adaptive_hindsight_logistic_dynamic_threshold)


if __name__ == "__main__":
    unittest.main()
