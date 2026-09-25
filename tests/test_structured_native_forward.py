"""Exercise the repository's real dLLM forward with tiny random weights.

This validates the adapter/position contract, not pretrained model quality.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / '.test-deps').exists():
    sys.path.insert(0, str(ROOT / '.test-deps'))

import torch
from transformers import Qwen2Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from structured_sparse_collector import ExplicitDrafter, KVExplicitDrafter, MASK_ID
from native_elysia_graph import NativeElysiaRunner


class NativeForwardTest(unittest.TestCase):
    def test_full_state_adapter_on_real_forward(self):
        package = ModuleType('tiny_dllm_test')
        package.__path__ = []
        sys.modules[package.__name__] = package
        configuration = ModuleType('tiny_dllm_test.configuration')
        configuration.Fast_dLLM_QwenConfig = Qwen2Config
        sys.modules[configuration.__name__] = configuration
        spec = importlib.util.spec_from_file_location('tiny_dllm_test.modeling', ROOT / 'Fast_dLLM_v2_1_5B/modeling.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if 'default' not in ROPE_INIT_FUNCTIONS:
            def rope(config, device, **kwargs):
                dim = config.hidden_size // config.num_attention_heads
                return 1 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)), 1.0
            ROPE_INIT_FUNCTIONS['default'] = rope
        config = Qwen2Config(vocab_size=151680, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=4, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=512)
        config.bd_size = 32
        config._attn_implementation = 'sdpa'
        model = module.Fast_dLLM_QwenForCausalLM(config).eval()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            model = model.to(device=device, dtype=torch.float16 if device == 'cuda' else torch.float32)
            engine = ExplicitDrafter(model, SimpleNamespace(physical_block_size=32, raw_top_k=32))
            for prefix_length, length in [(7, 8), (8, 24), (8, 32), (8, 64)]:
                with self.subTest(device=device, prefix=prefix_length, length=length):
                    obs = engine.observe([1] * prefix_length, [MASK_ID] * length)
                    self.assertEqual(len(obs['hidden_states']), 5)
                    self.assertTrue(all(len(layer) == length for layer in obs['hidden_states']))
                    self.assertEqual(len(obs['filled']), length)
                    self.assertEqual(len(obs['topk_logits']), length)

            # Complete cached blocks must reproduce a full-context forward.
            cached = KVExplicitDrafter(model, SimpleNamespace(physical_block_size=32, raw_top_k=32))
            full = ExplicitDrafter(model, SimpleNamespace(physical_block_size=32, raw_top_k=32))
            prefix = [1] * 40
            for native in ([MASK_ID] * 8,
                           [1] * 24 + [MASK_ID] * 8):
                expected = full.observe(prefix, native)
                observed = cached.observe(prefix, native)
                self.assertEqual(observed['kv_cached_prefix_len'] % 32, 0)
                for layer_expected, layer_observed in zip(expected['hidden_states'], observed['hidden_states']):
                    self.assertTrue(torch.allclose(torch.tensor(layer_expected),
                                                    torch.tensor(layer_observed), atol=0.03, rtol=0.01))
                # Only first-of-block logits intentionally differ from v3.
                for index, (a, b) in enumerate(zip(expected['predictions'], observed['predictions'])):
                    if (len(prefix) + index) % 32:
                        self.assertEqual(a, b)

            if device == 'cpu':
                class CpuEvent:
                    def __init__(self, **kwargs):
                        self.timestamp = 0.0

                    def record(self):
                        self.timestamp = time.perf_counter()

                    def elapsed_time(self, later):
                        return max(0.0, (later.timestamp - self.timestamp) * 1000.0)

                runner_args = SimpleNamespace(raw_top_k=8, physical_block_size=32,
                    small_block_size=8, extend_size=8, drafter_threshold=0.5,
                    max_refinement_steps=3)
                tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: '')
                native = NativeElysiaRunner(model, tokenizer, runner_args)
                with patch.object(torch.cuda, 'Event', CpuEvent), \
                     patch.object(torch.cuda, 'synchronize', lambda: None):
                    for prefix_length in (29, 32):
                        snapshots = native.segment([1] * prefix_length)
                        self.assertTrue(snapshots)
                        self.assertEqual(len(snapshots[0]['proposal_token_ids_after_fill']), 8)
                        self.assertTrue(snapshots[0]['hidden_states'])


if __name__ == '__main__':
    unittest.main()
