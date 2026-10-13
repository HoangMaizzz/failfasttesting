"""Execute the actual native generator with deterministic synthetic forwards.

Only model forwards and CUDA timers are replaced. This checks control flow,
not real checkpoint numerics or GPU memory usage.
"""
import ast
import logging
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import bucket_renewal
import adaptive_td

torch.set_num_threads(1)


def native_methods():
    path = Path(__file__).resolve().parents[1] / "Fast_dLLM_v2_1_5B" / "modeling.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"generate_draft_tokens_arbitrary_length", "sample_with_top_p"}
    nodes = [n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name in wanted]
    nodes += [n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Colors"]
    namespace = dict(vars(bucket_renewal), **{k: v for k, v in vars(adaptive_td).items()
                                           if k not in vars(bucket_renewal)})
    namespace.update(torch=torch, math=math, time=time, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Timer:
    def __init__(self, **kwargs): pass
    def record(self): pass
    def elapsed_time(self, other): return 1.


class NativeUnmaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_methods()

    def run_native(self, prefix_length, threshold, snapshot_limit=3, predict_eos=False):
        native = self.native
        class Model:
            device = torch.device("cpu")
            generate = native["generate_draft_tokens_arbitrary_length"]
            sample_with_top_p = native["sample_with_top_p"]
            def __init__(self): self.calls = []
            def forward(self, input_ids, **kwargs):
                self.calls.append((input_ids.clone(), dict(kwargs)))
                n = input_ids.shape[1]
                logits = torch.zeros(1, n, 64)
                # Changing predicted IDs exposes accidental STOP writes to x_t.
                token = 62 if predict_eos else 1 + len(self.calls) % 10
                logits[:, :, token] = torch.arange(n).remainder(8).float() * .4 + 3
                hidden = tuple(torch.full((1, n, 4), float(i)) for i in range(29))
                return SimpleNamespace(logits=logits, hidden_states=hidden, past_key_values=None)
        args = SimpleNamespace(full_refinement_oracle=True, raw_top_k=2,
            target_tokenizer=SimpleNamespace(decode=lambda *a, **kw: "synthetic"),
            collect_bucket_oracle=True, bucket_oracle_force_continue=True,
            frontier_stop_mode="disabled", adaptive_td=False,
            collector_max_oracle_snapshots=snapshot_limit)
        model = Model()
        with patch("torch.cuda.Event", Timer), patch("torch.cuda.synchronize"):
            result = model.generate(torch.full((1, prefix_length), 11),
                max_new_tokens=96, mask_id=63, stop_token=62, threshold=threshold,
                small_block_size=8, block_size=32, is_drafter=True, spec_len=8,
                args=args, lowconf_threshold=0., max_spec_len=8, incr_len=8,
                return_frontier_stats=True, top_p=1., temperature=0.,
                max_denoising_passes=16)
        return model, result[-1]["oracle_refinement_snapshots"]

    def test_replay_snapshot_limit_does_not_change_predecessor(self):
        for prefix in (3, 29, 32, 61, 64):
            _, one = self.run_native(prefix, 1., snapshot_limit=1)
            _, three = self.run_native(prefix, 1., snapshot_limit=3)
            for key in ("proposal_token_ids_before_fill", "proposal_token_ids_after_fill",
                        "unmask_forward_index", "hidden_states", "topk_logits",
                        "topk_token_ids", "confidences"):
                self.assertEqual(one[0][key], three[0][key])

    def test_native_EOS_does_not_stop_before_earlier_masks_are_committed(self):
        _, snapshots = self.run_native(3, 1., predict_eos=True)
        self.assertEqual(len(snapshots), 3)
        for snap in snapshots:
            live = snap["proposal_token_ids_before_fill"]
            self.assertIn(62, live)
            self.assertIn(63, live[:live.index(62)])
            self.assertNotIn(63, snap["proposal_token_ids_after_fill"])

    def test_threshold_fallback_live_state_and_block_boundaries(self):
        for prefix in (3, 24, 29, 32, 35, 61, 64):
            for threshold in (0., .5, 1.):
                with self.subTest(prefix=prefix, threshold=threshold):
                    model, snapshots = self.run_native(prefix, threshold)
                    self.assertTrue(snapshots)
                    previous = None
                    for snap in snapshots:
                        live = snap["proposal_token_ids_before_fill"]
                        stop = snap["proposal_token_ids_after_fill"]
                        self.assertEqual(len(live), 8)
                        self.assertNotIn(63, stop)
                        self.assertEqual(snap["masks_remaining"], live.count(63))
                        self.assertTrue(all(a == 63 or a == b for a, b in zip(live, stop)))
                        if previous:
                            self.assertTrue(all(a == 63 or a == b for a, b in zip(previous, live)))
                        if threshold == 1.:
                            self.assertLessEqual(len(snap["newly_unmasked_positions"]), 1)
                        previous = live
                    for call_index, ((before, kw1), (after, kw2)) in enumerate(
                            zip(model.calls, model.calls[1:])):
                        if (not kw1.get("update_past_key_values")
                                and not kw2.get("update_past_key_values")
                                and before.shape == after.shape):
                            committed = before != 63
                            self.assertTrue(torch.equal(before[committed], after[committed]))
                            first_mask = int((before[0] == 63).nonzero()[0])
                            lo = first_mask // 8 * 8
                            hi = lo + 8
                            positions = torch.arange(lo, hi)
                            shifted = (positions - 1).clamp_min(0)
                            strength = shifted.remainder(8).float() * .4 + 3
                            confidence = strength.exp() / (strength.exp() + 63)
                            masked = before[0, lo:hi] == 63
                            confidence[~masked] = -torch.inf
                            expected = confidence > threshold
                            expected[confidence.argmax()] = True
                            expected &= masked
                            actual = (before[0] == 63) & (after[0] != 63)
                            wanted = torch.zeros_like(actual)
                            wanted[lo:hi] = expected
                            self.assertTrue(torch.equal(actual, wanted))
                            self.assertTrue(bool((after[0, actual] == 1 + (call_index + 1) % 10).all()))
                            if threshold == 1.:
                                self.assertEqual(int(((before == 63) & (after != 63)).sum()), 1)


if __name__ == "__main__": unittest.main()
