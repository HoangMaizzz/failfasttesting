"""Offline native teacher checks: no Hub model download, both Qwen2 APIs."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

# Transformers can replace its lazy root module during model imports.
transformers = sys.modules["transformers"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_qwen_capture as native


IDENTITY = dict(model_id=native.MODEL_ID, revision=native.REVISION)
TINY = dict(vocab_size=37, hidden_size=16, intermediate_size=24,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=512, attention_dropout=0.0,
            tie_word_embeddings=False)


def hook_counts(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


class NativeCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(10)
        self.model = native.load_verifier(IDENTITY, dict(device="cpu", tiny_config=TINY))
        self.prefix = [1, 3, 5, 7]
        self.candidate = [9, 11, 13]

    def test_depths_use_ceiling_from_config_and_tiny_deduplication(self):
        self.assertEqual(native.depth_indices(28), [7, 14, 21, 28])
        self.assertEqual(native.depth_indices(7), [2, 4, 6, 7])
        self.assertEqual(native.depth_indices(2), [1, 2])
        self.assertEqual(native.depth_indices(1), [1])
        for bad in (0, -1, 2.5, True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                native.depth_indices(bad)

    def test_offline_loader_freezes_fp16_sdpa_eval(self):
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad and p.dtype == torch.float16
                            for p in self.model.parameters()))
        self.assertEqual(self.model.config._attn_implementation, "sdpa")
        self.assertFalse(self.model.config.use_cache)
        self.assertEqual(self.model.native_verifier_identity, IDENTITY)
        with patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                          side_effect=AssertionError("must not download")):
            native.load_verifier(IDENTITY, dict(device="cpu", tiny_config=TINY))

    def test_invalid_identity_quantization_and_unfrozen_model(self):
        for identity in ({}, dict(model_id=native.MODEL_ID), dict(model_id="", revision="main")):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                native.load_verifier(identity, dict(device="cpu", tiny_config=TINY))
        for field in ("quantization_config", "load_in_4bit", "load_in_8bit"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "quantization"):
                native.load_verifier(IDENTITY, dict(device="cpu", tiny_config=TINY, **{field: {}}))
        with self.assertRaisesRegex(ValueError, "historical"):
            native.load_verifier(dict(IDENTITY, revision="main"), {})
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            native.capture_one(self.model, self.prefix, self.candidate, [4])
        self.model.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            native.capture_one(self.model, self.prefix, self.candidate, [4])
        self.model.requires_grad_(False)
        self.model.is_loaded_in_4bit = True
        with self.assertRaisesRegex(ValueError, "quantization"):
            native.capture_one(self.model, self.prefix, self.candidate, [4])

    def test_production_load_kwargs_exact_pin_and_contiguous_map(self):
        config = Qwen2Config(**TINY)
        model = Qwen2ForCausalLM(config).half()
        with patch.object(transformers, "__version__", "4.53.1"), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(transformers.AutoConfig, "from_pretrained", return_value=config) as ac, \
             patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model) as am:
            # The mocked CPU model must fail the real post-load placement check.
            with self.assertRaisesRegex(RuntimeError, "unexpected device"):
                native.load_verifier(IDENTITY, {})
            self.assertEqual(ac.call_args.kwargs["revision"], native.REVISION)
            kw = am.call_args.kwargs
            self.assertEqual(kw["revision"], native.REVISION)
            self.assertEqual(kw["torch_dtype"], torch.float16)
            self.assertEqual(kw["attn_implementation"], "sdpa")
            self.assertEqual(kw["device_map"], {
                "model.embed_tokens": 0, "model.rotary_emb": 0,
                "model.layers.0": 0, "model.layers.1": 0,
                "model.layers.2": 1, "model.layers.3": 1,
                "model.norm": 1, "lm_head": 1})
            self.assertFalse(any(v in ("cpu", "disk") for v in kw["device_map"].values()))
        self.assertEqual(native._device_map(7)["model.layers.3"], 0)
        self.assertEqual(native._device_map(7)["model.layers.4"], 1)

    def test_production_requires_historical_transformers_and_two_gpus(self):
        with patch.object(transformers, "__version__", "4.57.3"):
            with self.assertRaisesRegex(RuntimeError, "4.53.1"):
                native.load_verifier(IDENTITY, {})
        with patch.object(transformers, "__version__", "4.53.1"), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(transformers.AutoConfig, "from_pretrained") as download:
            with self.assertRaisesRegex(RuntimeError, "GPU 0 and GPU 1"):
                native.load_verifier(IDENTITY, {})
            download.assert_not_called()

    def test_invalid_tokens_depths_and_context_raise_before_forward(self):
        before = hook_counts(self.model)
        bad_pairs = [([], [1]), ([1], []), ([37], [1]), ([1], [-1]),
                     ([True], [1]), ([1.0], [1]), ([1], (2,)),
                     ([1] * 512, [1])]
        with patch.object(self.model, "forward", side_effect=AssertionError("unexpected forward")):
            for prefix, candidate in bad_pairs:
                with self.subTest(prefix=prefix[:3], candidate=candidate), self.assertRaises(ValueError):
                    native.capture_one(self.model, prefix, candidate, [4])
            for depths in ([], [0], [5], [1, 1], [1.5], [True]):
                with self.subTest(depths=depths), self.assertRaises(ValueError):
                    native.capture_one(self.model, self.prefix, self.candidate, depths)
        self.assertEqual(before, hook_counts(self.model))

    def test_final_hidden_l_plus_one_replays_actual_candidate_logits(self):
        observed = {}
        def observe(module, args, output):
            if "actual" not in observed:
                observed["actual"] = output.detach().clone()
        handles = [self.model.lm_head.register_forward_hook(observe)]
        try:
            result = native.capture_one(self.model, self.prefix, self.candidate, [1, 2, 3, 4])
        finally:
            for h in handles:
                h.remove()
        final = result["final_hidden_with_bonus"]
        self.assertEqual(tuple(final.shape), (4, 16))
        with torch.inference_mode():
            replay = self.model.lm_head(final.unsqueeze(0))
        torch.testing.assert_close(replay, observed["actual"], rtol=0, atol=0)
        for h in result["hidden"].values():
            self.assertEqual(tuple(h.shape), (3, 16))
            self.assertEqual(h.dtype, torch.float16)
            self.assertEqual(h.device.type, "cpu")
            self.assertFalse(h.requires_grad)
        torch.testing.assert_close(result["hidden"][4], final[:3], rtol=0, atol=0)
        self.assertEqual(result["predictions"], observed["actual"][0].argmax(-1).tolist())
        self.assertTrue(result["alignment"]["passed"])
        self.assertEqual(result["alignment"]["causal_indices"], [3, 4, 5])
        self.assertNotIn("logits", result)
        json.dumps(result["alignment"])

    def test_intermediate_raw_and_final_normalized_positions(self):
        observed = {}
        def save(name):
            def hook(module, args, output):
                observed[name] = native._hidden(output).detach().clone()
            return hook
        handles = [self.model.model.layers[1].register_forward_hook(save("raw2")),
                   self.model.model.layers[3].register_forward_hook(save("raw4")),
                   self.model.model.norm.register_forward_hook(save("norm"))]
        try:
            result = native.capture_one(self.model, self.prefix, self.candidate, [2, 4])
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(result["hidden"][2], observed["raw2"][0, 3:6], atol=0, rtol=0)
        torch.testing.assert_close(result["hidden"][4], observed["norm"][0, 3:6], atol=0, rtol=0)
        self.assertFalse(torch.equal(result["hidden"][4], observed["raw4"][0, 3:6]))
        self.assertEqual(set(native.capture_one(self.model, self.prefix, self.candidate, [2])["hidden"]), {2})

    def test_no_future_or_current_candidate_leakage_at_every_depth(self):
        base = native.capture_one(self.model, self.prefix, self.candidate, [1, 2, 3, 4])
        for i in range(len(self.candidate)):
            changed = self.candidate[:i] + [30] * (len(self.candidate) - i)
            altered = native.capture_one(self.model, self.prefix, changed, [1, 2, 3, 4])
            for d in base["hidden"]:
                torch.testing.assert_close(base["hidden"][d][:i + 1], altered["hidden"][d][:i + 1],
                                           atol=0, rtol=0)
            self.assertEqual(base["predictions"][:i + 1], altered["predictions"][:i + 1])
        # Independent prefix-only forward predicts the first proposed token.
        with torch.inference_mode():
            first = self.model(input_ids=torch.tensor([self.prefix]),
                               attention_mask=torch.ones(1, len(self.prefix), dtype=torch.long),
                               use_cache=False, logits_to_keep=1).logits.argmax(-1).item()
        self.assertEqual(first, base["predictions"][0])

    def test_k_counts_contiguous_prefix_not_total_matches(self):
        prefix, candidate = [1, 2], []
        for _ in range(3):
            with torch.inference_mode():
                token = self.model(input_ids=torch.tensor([prefix + candidate]), use_cache=False,
                                   logits_to_keep=1).logits.argmax(-1).item()
            candidate.append(token)
        self.assertEqual(native.capture_one(self.model, prefix, candidate, [4])["K"], 3)
        candidate[1] = (candidate[1] + 1) % 37
        self.assertEqual(native.capture_one(self.model, prefix, candidate, [4])["K"], 1)

    def test_only_l_plus_one_head_positions_no_all_hidden_no_cache(self):
        head_sizes, calls = [], []
        def head_hook(module, args):
            head_sizes.append(args[0].shape[1])
        def model_hook(module, args, kwargs):
            calls.append(kwargs.copy())
        handles = [self.model.lm_head.register_forward_pre_hook(head_hook),
                   self.model.register_forward_pre_hook(model_hook, with_kwargs=True)]
        before = hook_counts(self.model)
        try:
            native.capture_one(self.model, self.prefix, self.candidate, [1, 4])
            self.assertEqual(before, hook_counts(self.model))
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(head_sizes, [4, 4, 4])
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["use_cache"])
        self.assertFalse(calls[0]["output_hidden_states"])
        self.assertFalse(calls[0]["output_attentions"])
        self.assertTrue(torch.equal(calls[0]["input_ids"], torch.tensor([self.prefix + self.candidate])))
        self.assertTrue(torch.equal(calls[0]["attention_mask"], torch.ones(1, 7, dtype=torch.long)))

    def test_capture_cleanup_on_forward_exception_and_alignment_failure(self):
        before = hook_counts(self.model)
        def fail(module, args, output):
            raise RuntimeError("injected failure")
        h = self.model.model.layers[2].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                native.capture_one(self.model, self.prefix, self.candidate, [1, 4])
        finally:
            h.remove()
        self.assertEqual(before, hook_counts(self.model))
        original = self.model.forward
        def shifted_logits(**kwargs):
            output = original(**kwargs)
            output.logits = output.logits + 0.1
            return output
        with patch.object(self.model, "forward", side_effect=shifted_logits):
            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                native.capture_one(self.model, self.prefix, self.candidate, [1, 4])
        self.assertEqual(before, hook_counts(self.model))

    def test_benchmark_executes_true_cuts_identical_inputs_and_weighted_timings(self):
        rows = [dict(uid="a", prefix=[1, 2], candidate=[3, 4]),
                dict(uid="b", prefix=[1] * 129, candidate=[5])]
        counters = dict(block0=0, block1=0, block2=0, block3=0, norm=0, head=0)
        calls = []
        def count(name):
            def hook(module, args, output):
                counters[name] += 1
            return hook
        def record(module, args, kwargs):
            calls.append(kwargs["input_ids"].clone())
            self.assertFalse(kwargs["use_cache"])
            self.assertFalse(kwargs["output_hidden_states"])
            self.assertEqual(kwargs["logits_to_keep"], kwargs["input_ids"].shape[1] -
                             (2 if kwargs["input_ids"].shape[1] == 4 else 129) + 1)
        handles = [m.register_forward_hook(count(name)) for name, m in
                   [(f"block{i}", m) for i, m in enumerate(self.model.model.layers)] +
                   [("norm", self.model.model.norm), ("head", self.model.lm_head)]]
        handles.append(self.model.register_forward_pre_hook(record, with_kwargs=True))
        before = hook_counts(self.model)
        try:
            # A deterministic clock makes aggregation verifiable without speed assertions.
            with patch.object(native.time, "perf_counter_ns", side_effect=range(0, 10**10, 1000000)):
                report = native.benchmark_partial(self.model, rows, [2, 4], {})
            self.assertEqual(before, hook_counts(self.model))
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(counters, dict(block0=660, block1=660, block2=440,
                                        block3=440, norm=440, head=220))
        self.assertEqual(report["full_weighted_mean_ms"], 1.0)
        self.assertEqual(report["by_depth"]["2"]["latency_fraction"], 1.0)
        self.assertEqual(report["device_map"]["model.norm"], "cpu")
        self.assertEqual(report["synchronized_cuda_devices"], [])
        self.assertTrue(report["full_protocol"])
        self.assertFalse(report["pipeline_check_only"])
        self.assertTrue(report["matched_sequence_distribution"])
        self.assertTrue(report["within_bin_population_distribution_exact"])
        self.assertFalse(report["device_audit"]["gpu_model_no_cpu_offload_validated"])
        for offset in (0, 330):
            self.assertTrue(all(torch.equal(calls[offset], c) for c in calls[offset:offset + 330]))
        for b in report["bins"].values():
            self.assertEqual(b["weight"], 0.5)
            self.assertEqual(len(b["full"]["raw_measurements"]), 100)
            for cut in b["cuts"].values():
                self.assertEqual(cut["raw_measurements"], b["full"]["raw_measurements"])
        json.dumps(report)

    def test_benchmark_deterministic_cap_bin_weights_and_round_robin(self):
        rows = [dict(uid=f"row{i}", prefix=[1] * (130 if i == 10 else i + 1), candidate=[2])
                for i in range(11)]
        def fake_measure(model, prepared, depth, warmups, repetitions, devices):
            return dict(mean_ms=10.0 if depth is None else 4.0,
                        raw_measurements=[dict(uid=row["uid"]) for row, _ in prepared])
        with patch.object(native, "_measure", side_effect=fake_measure):
            a = native.benchmark_partial(self.model, rows, [2], {})
            b = native.benchmark_partial(self.model, list(reversed(rows)), [2], {})
        self.assertEqual(a, b)
        self.assertEqual(len(a["bins"]["128"]["sampled_uids"]), 8)
        self.assertEqual(a["bins"]["128"]["weight"], 10 / 11)
        self.assertEqual(a["bins"]["256"]["weight"], 1 / 11)
        self.assertAlmostEqual(a["by_depth"]["2"]["latency_fraction"], 0.4)
        self.assertTrue(a["matched_sequence_distribution"])
        self.assertFalse(a["within_bin_population_distribution_exact"])
        self.assertFalse(a["bins"]["128"]["within_bin_population_distribution_exact"])

    def test_benchmark_smoke_requires_explicit_flag_and_audits_matched_schedule(self):
        rows = [dict(uid=f"smoke{i}", prefix=[1] * (i + 1), candidate=[2]) for i in range(2)]
        options = dict(repetitions=3, warmups=1, max_rows_per_bin=8, seed=42,
                       pipeline_check_only=True)
        before = hook_counts(self.model)
        report = native.benchmark_partial(self.model, rows, [2, 4], options)
        self.assertEqual(before, hook_counts(self.model))
        self.assertFalse(report["full_protocol"])
        self.assertTrue(report["pipeline_check_only"])
        self.assertTrue(report["matched_sequence_distribution"])
        # All rows were sampled, but 3 repetitions cannot give two lengths equal weight.
        self.assertFalse(report["within_bin_population_distribution_exact"])
        b = report["bins"]["128"]
        self.assertEqual(b["population_sequence_length_counts"], {2: 1, 3: 1})
        self.assertEqual(sum(b["timed_sequence_length_counts"].values()), 3)
        full_trace = [(r["uid"], r["sequence_length"]) for r in b["full"]["raw_measurements"]]
        for run in [b["full"], *b["cuts"].values()]:
            self.assertEqual(run["warmups"], 1)
            self.assertEqual(run["repetitions"], 3)
            self.assertEqual([(r["uid"], r["sequence_length"]) for r in run["raw_measurements"]], full_trace)
        for flag in (False, 1, "True", None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                native.benchmark_partial(self.model, rows, [2], dict(options, pipeline_check_only=flag))
        for invalid in (dict(warmups=0), dict(repetitions=0), dict(max_rows_per_bin=9)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                native.benchmark_partial(self.model, rows, [2], dict(options, **invalid))
        json.dumps(report)

    def test_population_distribution_exact_requires_all_rows_and_length_weights(self):
        rows = [dict(uid=f"row{i}", prefix=[1] * (i + 1), candidate=[2]) for i in range(2)]
        def fake_measure(model, prepared, depth, warmups, repetitions, devices):
            return dict(mean_ms=1.0, raw_measurements=[])
        with patch.object(native, "_measure", side_effect=fake_measure):
            exact = native.benchmark_partial(self.model, rows, [2], {})
            uneven = native.benchmark_partial(self.model, rows, [2], dict(repetitions=101))
            smoke = native.benchmark_partial(self.model, rows, [2], dict(pipeline_check_only=True))
        self.assertTrue(exact["within_bin_population_distribution_exact"])
        self.assertFalse(uneven["within_bin_population_distribution_exact"])
        self.assertFalse(smoke["full_protocol"])
        self.assertTrue(smoke["within_bin_population_distribution_exact"])

    def test_gpu_device_audit_rejects_cpu_and_disk_offload_before_timing(self):
        row = dict(uid="train", prefix=[1], candidate=[2])
        with patch.object(native, "_cuda_devices", return_value=[torch.device("cuda", 0)]), \
             patch.object(native, "_measure") as measure:
            with self.assertRaisesRegex(RuntimeError, "offload"):
                native.benchmark_partial(self.model, [row], [2], {})
            measure.assert_not_called()

    def test_benchmark_invalid_config_rows_and_hook_cleanup(self):
        row = dict(uid="train", prefix=[1], candidate=[2])
        for cfg in (dict(warmups=9), dict(repetitions=99), dict(max_rows_per_bin=9),
                    dict(repetitions=True)):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                native.benchmark_partial(self.model, [row], [2], cfg)
        for rows in ([], [dict(row, split="val")], [dict(row, split="test")],
                     [row, row], [dict(row, prefix=[])], [dict(row, candidate=[37])], [{}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                native.benchmark_partial(self.model, rows, [2], {})
        before = hook_counts(self.model)
        with patch.object(self.model.model.layers[0], "forward", side_effect=RuntimeError("broken")):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                native._measure(self.model, [(row, native._tokens(self.model, [1], [2]))],
                                2, 10, 100, [])
        self.assertEqual(before, hook_counts(self.model))

    def test_sync_visits_every_participating_cuda_device(self):
        devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
        with patch.object(torch.cuda, "synchronize") as sync:
            native._sync(devices)
        self.assertEqual([call.args[0] for call in sync.call_args_list], devices)


if __name__ == "__main__":
    unittest.main()
