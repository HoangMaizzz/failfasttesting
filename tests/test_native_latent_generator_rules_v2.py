"""Execute the real native generator with a deterministic synthetic forward.

This validates scheduler/capture integration, not pretrained dLLM accuracy.
No downloaded checkpoint, token-target learner, or verifier is involved.
"""
import ast
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from native_elysia_graph import NativeElysiaRunner
from native_latent_rules_v2 import proposal_span, scheduler_region
from bucket_renewal import (position_bucket, expected_accepted_prefix,
                            predict_next_gain, compare_renewal_costs, calibrated_acceptance_probability)


class Cache:
    def __init__(self, length):
        self.length = length
        self.kv = torch.zeros(1, 2, length, 128)

    def get_seq_length(self):
        return self.length

    def __getitem__(self, index):
        return self.kv, self.kv


class SyntheticForward(torch.nn.Module):
    def __init__(self, eos_row=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(1, 8)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(hidden_size=8, num_key_value_heads=2,
                                      num_attention_heads=1, head_dim=128)
        self.model = SimpleNamespace(layers=[None] * 28)
        self.eos_row = eos_row

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, update_past_key_values=True, **kwargs):
        length = input_ids.shape[1]
        # Probability of argmax stays below .5; native forced-progress rule
        # therefore commits exactly one eligible position per unmask pass.
        logits = torch.zeros(1, length, 151666)
        logits[..., 10] = 4.
        if self.eos_row is not None and self.eos_row < length:
            logits[:, self.eos_row, 151645] = 15.
        cached = 0 if past_key_values is None else past_key_values.length
        cache = Cache(cached + length) if update_past_key_values else past_key_values
        hidden = tuple(torch.zeros(1, length, 8) for _ in range(29))
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden)

    def sample_with_top_p(self, logits, **kwargs):
        return logits.argmax(-1), logits.softmax(-1)


def instrument_generator():
    source = Path(__file__).resolve().parents[1] / "Fast_dLLM_v2_1_5B/modeling.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"Colors", "materialize_stop_candidate", "native_v2_physical_structure"}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Fast_dLLM_QwenForCausalLM")
    generator = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate_draft_tokens_arbitrary_length")
    selected.append(generator)
    namespace = {"torch": torch, "logger": logging.getLogger("native_synthetic"), "position_bucket": position_bucket}
    namespace.update(expected_accepted_prefix=expected_accepted_prefix,
                     predict_next_gain=predict_next_gain, compare_renewal_costs=compare_renewal_costs,
                     calibrated_acceptance_probability=calibrated_acceptance_probability, math=math, time=time)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    SyntheticForward.generate_draft_tokens_arbitrary_length = namespace["generate_draft_tokens_arbitrary_length"]


class NativeGeneratorRulesTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "native timing uses CUDA events; no checkpoint is loaded")
    def test_actual_generator_masks_seed_boundary_and_R(self):
        torch.set_num_threads(2)
        instrument_generator()
        tokenizer = SimpleNamespace(eos_token_id=151645, decode=lambda _, **kwargs: "x")
        args = SimpleNamespace(extend_size=8, max_refinement_steps=2,
            physical_block_size=32, small_block_size=8, raw_top_k=1,
            drafter_threshold=.5, wm_v2_capture_native_latent=True)
        runner = NativeElysiaRunner(SyntheticForward(), tokenizer, args)
        for phase in (0, 1, 7, 25, 31):
            start = 96 + phase
            with self.subTest(phase=phase):
                snapshots = runner.segment([10] * start, max_snapshots=3)
                frame = (start + 7) // 32 * 32
                seeded = phase == 0 or frame > start
                first_mask = max(start, frame) + 1 if seeded else start
                expected = torch.arange(32) + frame >= first_mask
                first = snapshots[0]
                self.assertTrue(torch.equal(torch.tensor(first["wm_v2_mask_before_forward"]), expected))
                self.assertEqual(first["wm_v2_physical_block_start"], frame)
                for parent, child in zip(snapshots, snapshots[1:]):
                    self.assertEqual(child["unmask_forward_index"], parent["unmask_forward_index"] + 1)
                    self.assertEqual(child["wm_v2_mask_before_forward"], parent["wm_v2_physical_mask"])
                    self.assertEqual(child["wm_v2_prefix_kv_summary"], parent["wm_v2_prefix_kv_summary"])

    @unittest.skipUnless(torch.cuda.is_available(), "native timing uses CUDA events")
    def test_EOS_does_not_finish_while_previous_masks_remain(self):
        torch.set_num_threads(2)
        instrument_generator()
        args = SimpleNamespace(extend_size=8, max_refinement_steps=6,
            physical_block_size=32, small_block_size=8, raw_top_k=1,
            drafter_threshold=.5, wm_v2_capture_native_latent=True)
        tokenizer = SimpleNamespace(eos_token_id=151645, decode=lambda _, **kw: "x")
        snapshots = NativeElysiaRunner(SyntheticForward(eos_row=4), tokenizer, args).segment([10]*96, max_snapshots=7)
        self.assertTrue(snapshots[0]["wm_v2_eos_present"])
        self.assertFalse(snapshots[0]["wm_v2_native_terminal"])
        self.assertGreater(snapshots[0]["wm_v2_masks_before_eos"], 0)
        self.assertGreater(len(snapshots), 1)
        self.assertTrue(snapshots[-1]["wm_v2_native_terminal"])
        self.assertEqual(snapshots[-1]["wm_v2_masks_before_eos"], 0)
        self.assertFalse(any(snapshots[-1]["wm_v2_valid_positions"][6:]))


if __name__ == "__main__":
    unittest.main()
