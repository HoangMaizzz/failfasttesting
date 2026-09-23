"""CPU-only regression tests for raw feature composition (no model download)."""
import ast
import copy
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "sparse_extend_world_model_collector.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
HELPERS = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
           and node.name in {"_merge_feature_rows", "_merge_layer_features"}]
EXTEND = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
              and node.name == "_extend_one")
# Execute the actual fallback block as well as its helpers, without importing
# torch/transformers or allocating GPUs. This catches concatenating layer lists.
FALLBACK = next(node for node in EXTEND.body if isinstance(node, ast.For))
FALLBACK = next(node for node in FALLBACK.body if isinstance(node, ast.If)
                and ast.unparse(node.test) ==
                "feature_merge_mode == 'append_extension_fallback'")


class FeatureMergeTest(unittest.TestCase):
    def namespace(self):
        ns = {}
        exec(compile(ast.Module(body=HELPERS, type_ignores=[]), str(SOURCE), "exec"), ns)
        return ns

    def test_append_token_axis_through_64(self):
        for parent_length in range(8, 64, 8):
            with self.subTest(parent_length=parent_length):
                parent = {
                    "hidden_states": [[[layer, token] for token in range(parent_length)]
                                      for layer in range(5)],
                    "topk_token_ids": [[token] for token in range(parent_length)],
                    "topk_logits": [[float(token)] for token in range(parent_length)],
                }
                snap = {
                    "hidden_states": [[[layer, token] for token in range(parent_length, parent_length + 8)]
                                      for layer in range(5)],
                    "topk_token_ids": [[token] for token in range(parent_length, parent_length + 8)],
                    "topk_logits": [[float(token)] for token in range(parent_length, parent_length + 8)],
                }
                original = copy.deepcopy(parent)
                ns = self.namespace()
                ns.update(parent=parent, snap=snap, full_native=[0] * parent_length,
                          native=[0] * (parent_length + 8),
                          feature_merge_mode="append_extension_fallback")
                exec(compile(ast.Module(body=[FALLBACK], type_ignores=[]), str(SOURCE), "exec"), ns)
                self.assertEqual(len(ns["hidden_states"]), 5)
                for layer, rows in enumerate(ns["hidden_states"]):
                    self.assertEqual(rows, [[layer, token] for token in range(parent_length + 8)])
                self.assertEqual(len(ns["topk_token_ids"]), parent_length + 8)
                self.assertEqual(len(ns["topk_logits"]), parent_length + 8)
                self.assertEqual(parent, original)

    def test_invalid_overlay_is_detected(self):
        ns = self.namespace()
        with self.assertRaisesRegex(RuntimeError, "offset 25"):
            ns["_merge_layer_features"]([[[0]] * 24] * 5, [[[1]] * 7] * 5, 25, 32)

    def test_missing_layer_is_rejected(self):
        ns = self.namespace()
        with self.assertRaisesRegex(RuntimeError, "layer count"):
            ns["_merge_layer_features"]([[[0]] * 24] * 5, [[[1]] * 8] * 4, 24, 32)


if __name__ == "__main__":
    unittest.main()
