import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from audit_verifier_kv import audit_prompt
from verifier_kv_cache import VerifierKVCache
from verifier_diagnostic import install_replay, trace_verifier


class VerifierCacheTest(unittest.TestCase):
    def test_diagnostic_ties_use_production_argmax(self):
        cached = torch.full((1, 2, 8), 27.5, dtype=torch.float16)
        reference = cached.clone()
        reference[0, 1, 4] = 27.515625
        model = lambda **kwargs: SimpleNamespace(logits=reference)
        full = torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            trace_verifier(model, full, torch.ones_like(full), cached, [3], path, 394, 0)
            row = json.loads(path.read_text())
            self.assertEqual(row["cached_predictions"], [0, 0])
            self.assertEqual(row["full_predictions"], [0, 4])
            self.assertEqual(row["different_prediction_positions"], [1])
            self.assertEqual(row["prediction_rule"], "argmax")

    def test_strict_diagnostic_replay(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "replay.json"
            expected = dict(context_len=10, refinement_step=1, active_block_start=10,
                            active_block_end=16, draft_proposal=[1, 2], action="stop")
            path.write_text(json.dumps({"problems": {"355": {"decisions": [expected]}}}))
            controller = SimpleNamespace(hindsight_problem_id=355, decision_count=0)
            _, offsets = install_replay(controller, path)
            kwargs = {k: v for k, v in expected.items() if k != "action"}
            kwargs["allow_stop"] = True
            wrong = dict(kwargs, draft_proposal=[1, 3])
            with self.assertRaises(RuntimeError):
                controller.choose([], **wrong)
            self.assertEqual(offsets, {})
            self.assertEqual(controller.choose([], **kwargs).action, "stop")
            self.assertEqual(offsets, {"355": 1})
            with self.assertRaises(RuntimeError):
                controller.choose([], **kwargs)

    def test_diagnostic_trace_records_logits(self):
        full = torch.randint(0, 97, (1, 16))
        cache = VerifierKVCache()
        output = cache.verify(self.model, full, torch.ones_like(full), 8)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            trace_verifier(self.model, full, torch.ones_like(full), output.logits,
                           full[0, -8:].tolist(), path, 355, 0)
            row = json.loads(path.read_text())
            self.assertEqual(row["different_prediction_positions"], [])
            self.assertEqual(len(row["cached_top2_logits"]), 9)
            self.assertEqual(len(row["full_input_ids"]), 16)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=2048,
                             attention_dropout=0.0, tie_word_embeddings=False)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).eval()

    def test_real_attention_matches_full_prefix_and_greedy(self):
        for length in (7, 33, 129):
            rows = audit_prompt(self.model, torch.randint(0, 97, (1, length)))
            self.assertEqual([r["accepted"] for r in rows], [0, 3, 8, 4, 8])
            self.assertEqual(rows[-1]["emitted"], 2)
            self.assertLess(rows[-1]["verifier_input_tokens"],
                            rows[-1]["full_prefix_input_tokens_without_cache"])

    def test_commit_crop_and_alignment_guards(self):
        cache = VerifierKVCache()
        full = torch.randint(0, 97, (1, 16))
        with self.assertRaises(RuntimeError):
            cache.commit(10)
        cache.verify(self.model, full, torch.ones_like(full), 8)
        with self.assertRaises(RuntimeError):
            cache.verify(self.model, full, torch.ones_like(full), 8)
        # Terminal EOS inside the accepted proposal uses the truncated committed length.
        cache.commit(10)
        self.assertEqual(cache.cache.get_seq_length(), 9)
        with self.assertRaises(RuntimeError):
            cache.verify(self.model, full, torch.ones_like(full), 8)
        next_full = torch.cat([full[:, :10], full[:, :4]], dim=1)
        actual = cache.verify(self.model, next_full, torch.ones_like(next_full), 4)
        with torch.inference_mode():
            expected = self.model(input_ids=next_full, attention_mask=torch.ones_like(next_full),
                                  use_cache=False, logits_to_keep=5)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-4)

    def test_maximum_proposal_length_and_new_problem(self):
        for _ in range(2):
            cache = VerifierKVCache()
            context = torch.randint(0, 97, (1, 31))
            for size in (64, 8, 32, 64):
                full = torch.cat([context, torch.randint(0, 97, (1, size))], dim=1)
                actual = cache.verify(self.model, full, torch.ones_like(full), size)
                with torch.inference_mode():
                    expected = self.model(input_ids=full, attention_mask=torch.ones_like(full),
                                          use_cache=False, logits_to_keep=size + 1)
                torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-4)
                correction = expected.logits[:, :1].argmax(-1)
                context = torch.cat([context, correction], dim=1)
                cache.commit(context.shape[1])

    def test_float16_attention(self):
        self.model.half()
        try:
            rows = audit_prompt(self.model, torch.randint(0, 97, (1, 19)))
            self.assertTrue(all(row["greedy_output_match"] for row in rows))
        finally:
            self.model.float()


if __name__ == "__main__":
    unittest.main()
