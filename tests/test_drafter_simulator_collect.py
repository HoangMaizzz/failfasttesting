"""Execute the repository native loop, using a deterministic forward core.

The loop is extracted with AST so these CPU tests do not import Transformers,
download weights, or replace native commit/EOS/cache behavior with a simulator.
"""
import ast
import contextlib
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import drafter_simulator_collect as collector


def native_methods(original=False):
    path = ROOT / "Fast_dLLM_v2_1_5B/modeling.py"
    if original:
        source = subprocess.check_output([
            "git", "-c", f"safe.directory={ROOT.as_posix()}", "show",
            "HEAD:Fast_dLLM_v2_1_5B/modeling.py"], cwd=ROOT, text=True)
    else:
        source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Fast_dLLM_QwenForCausalLM")
    methods = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("generate_draft_tokens_arbitrary_length", "sample_with_top_p")]
    for method in methods:
        method.decorator_list = []
    namespace = dict(torch=torch, time=time,
                     logger=SimpleNamespace(debug=lambda *a: None),
                     Colors=SimpleNamespace(**{x: "" for x in ("RED", "GREEN", "CYAN", "YELLOW", "RESET", "MAGENTA")}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class CpuEvent:
    def __init__(self, **kwargs):
        self.value = 0.0

    def record(self):
        self.value = time.perf_counter()

    def elapsed_time(self, later):
        return max(0.001, (later.value - self.value) * 1000)


class FakeCore:
    device = torch.device("cpu")

    def __init__(self, eos=False, original=False):
        methods = native_methods(original)
        self.generate_draft_tokens_arbitrary_length = methods["generate_draft_tokens_arbitrary_length"].__get__(self)
        self.sample_with_top_p = methods["sample_with_top_p"].__get__(self)
        self.embedding = torch.nn.Embedding(collector.MASK_ID + 1, 16)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(self.embedding.weight.numel()).reshape_as(self.embedding.weight) % 19 / 19)
        self.head = torch.nn.Identity()
        self.lm_head = self.head
        self.eos = eos
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, *args):
        return self

    def forward(self, input_ids, use_cache, update_past_key_values, past_key_values=None,
                output_hidden_states=False, **kwargs):
        if kwargs.get("use_block_cache"):
            raise AssertionError("collector used mutable block cache")
        self.calls.append(dict(input=input_ids.clone(), hidden_requested=output_hidden_states,
                               cache=update_past_key_values,
                               prefix=0 if past_key_values is None else past_key_values.length))
        width = input_ids.shape[1]
        hidden = torch.zeros(1, width, 16)
        # Every unresolved position remains below .5: forced argmax must commit
        # one position per true denoising forward, with no artificial pass cap.
        positions = torch.arange(width)
        ids = (positions % 13) + 1
        hidden[0, positions, ids] = 0.9 + (positions % 3) * 0.01
        # Outside-active hidden features vary with the whole physical canvas.
        hidden[..., 0] = float(input_ids.eq(collector.MASK_ID).sum()) / 1000
        if self.eos and not update_past_key_values:
            hidden[0, :, 15] = 20.0
            # Only one active position crosses threshold, producing incomplete
            # EOS groups. Logit index 3 maps to native prediction index 4.
            hidden[0, :, 15] = 0.0
            hidden[0, 3, 15] = 20.0
        cache = SimpleNamespace(length=(0 if past_key_values is None else past_key_values.length) + width)
        return SimpleNamespace(logits=hidden, hidden_states=(hidden,) if output_hidden_states else None,
                               past_key_values=cache)


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # PyTorch lazily imports Dynamo for AdamW and torch.compile. Initialize
        # it while real import metadata and real CUDA Event types are present;
        # partially initializing it under mocks poisons later training tests.
        import torch._dynamo

    def setUp(self):
        self.timer = patch.object(torch.cuda, "Event", CpuEvent)
        self.sync = patch.object(torch.cuda, "synchronize", lambda: None)
        self.timer.start()
        self.sync.start()
        self.addCleanup(self.timer.stop)
        self.addCleanup(self.sync.stop)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.tokenizer = SimpleNamespace(eos_token_id=15, decode=lambda *a, **k: "")
        self.config = dict(max_new_tokens=32, max_context=4096, benchmark_native_states=0)

    def test_import_without_transformers_or_datasets(self):
        script = """
import sys
class Reject:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('transformers', 'datasets'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
import drafter_simulator_collect
"""
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True)

    def test_strict_threshold_forced_argmax_and_frozen_prefix(self):
        pre = torch.tensor([1, collector.MASK_ID, collector.MASK_ID, collector.MASK_ID])
        logits = torch.tensor([[20., 0.], [0., 0.], [0., 0.], [0., 0.]])
        eligible = torch.tensor([True, True, True, False])
        post, selected = collector.native_commit(pre, logits, eligible)
        self.assertEqual(selected.tolist(), [False, True, False, False])
        self.assertEqual(post.tolist(), [1, 0, collector.MASK_ID, collector.MASK_ID])

    def test_hook_off_matches_original_native_loop(self):
        # Compare the actual unmodified HEAD method to the instrumented method.
        original, changed = FakeCore(original=True), FakeCore()
        baseline, reason_a, stats_a = collector.run_native(original, self.tokenizer, [2] * 35, self.config)
        observed, reason_b, stats_b = collector.run_native(changed, self.tokenizer, [2] * 35, self.config)
        self.assertEqual(baseline, observed)
        self.assertEqual(reason_a, reason_b)
        self.assertEqual(stats_a, stats_b)
        for a, b in zip(original.calls, changed.calls):
            self.assertTrue(torch.equal(a["input"], b["input"]))
            self.assertEqual(a["hidden_requested"], b["hidden_requested"])
            self.assertEqual(a["prefix"], b["prefix"])
        self.assertFalse(any(call["hidden_requested"] for call in changed.calls))

    def test_full_natural_trajectories_alignment_schema_and_cpu_ownership(self):
        core = FakeCore()
        prompt = [2] * 35
        observer, tokens, reason, audit = collector.audit_replay(core, self.tokenizer, prompt, self.config, "17")
        self.assertTrue(audit["tokens_equal"])
        self.assertGreater(len(observer.rows), 20)
        self.assertGreater(max(len(group["state_uids"]) for group in observer.groups.values()), 3)
        self.assertTrue(all(row["hidden"].shape == (32, 16) for row in observer.rows))
        first = observer.rows[0]
        self.assertEqual(first["positions"].tolist(), list(range(32, 64)))
        self.assertFalse(first["eligible"][:3].any())
        self.assertTrue(first["eligible"][3:8].all())
        self.assertEqual(first["tokens"][:3].tolist(), [2, 2, 2])
        self.assertTrue(np.array_equal(first["hidden"][0], first["hidden"][1]))
        self.assertTrue(np.array_equal(first["candidate_ids"], first["hidden"].argmax(-1)))
        self.assertFalse(np.array_equal(first["hidden"][20], observer.rows[1]["hidden"][20]))
        for row in observer.rows:
            self.assertTrue(all(not isinstance(value, torch.Tensor) for value in row.values()))
        for edge in observer.edges:
            self.assertEqual(edge["target_forward_id"], edge["source_forward_id"] + 1)
        with tempfile.TemporaryDirectory() as directory:
            record = observer.save(directory, tokens, reason)
            with np.load(Path(directory) / record["npz"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["hidden"].dtype, np.float16)
                self.assertEqual(arrays["hidden"].shape, (len(observer.rows), 32, 16))
                self.assertEqual(arrays["candidate_emb"].shape, (len(observer.rows), 32, 64))
                self.assertEqual(arrays["token_emb"].shape, (len(observer.rows), 32, 64))
                self.assertEqual(arrays["mask"].dtype, bool)
                self.assertTrue(np.isfinite(arrays["entropy"]).all())
                self.assertTrue((arrays["forward_ms"] > 0).all())
                self.assertTrue((arrays["capture_ms"] > 0).all())
                for index in range(len(observer.rows)):
                    if arrays["prev_valid"][index]:
                        np.testing.assert_array_equal(arrays["prev_confidence"][index], arrays["confidence"][index - 1])
                    else:
                        self.assertFalse(arrays["prev_confidence"][index].any())
            self.assertTrue(all(not group["incomplete"] for group in record["groups"]))
            self.assertEqual(len(record["states"]), len(observer.rows))

    def test_eos_incomplete_group_uses_actual_commit_not_candidate_fill(self):
        core = FakeCore(eos=True)
        observer, tokens, reason, _ = collector.audit_replay(core, self.tokenizer, [2] * 3, self.config, "eos")
        self.assertEqual(reason, "eos")
        self.assertEqual(tokens[-1], self.tokenizer.eos_token_id)
        self.assertLess(len(observer.rows), 8)
        with tempfile.TemporaryDirectory() as directory:
            record = observer.save(directory, tokens, reason)
            group = record["groups"][0]
            self.assertTrue(group["eos_terminated_incomplete"])
            self.assertTrue(group["incomplete"])
            self.assertIn(collector.MASK_ID, group["terminal_tokens"])
            self.assertEqual(group["terminal_tokens"], observer.rows[-1]["tokens"].tolist())

    def test_context_cap_does_not_append_bonus_outside_canvas(self):
        core = FakeCore()
        config = {**self.config, "max_context": 64, "max_new_tokens": 64}
        observer, tokens, reason, _ = collector.audit_replay(core, self.tokenizer, [2] * 35, config, "cap")
        self.assertEqual(reason, "context_cap")
        self.assertEqual(len(tokens), 64)
        self.assertTrue(all(state["context_len"] <= 64 for state in observer.states))
        self.assertFalse(any(collector.MASK_ID == token for token in tokens))
        _, capped, cap_reason, _ = collector.audit_replay(FakeCore(), self.tokenizer, [2] * 64, config, "aligned_cap")
        self.assertEqual(cap_reason, "context_cap")
        self.assertEqual(len(capped), 64)

    def test_no_future_temporal_or_edge_across_forward_gap(self):
        core = FakeCore()
        observer = collector.QuestionObserver(core, "gap", [2] * 3, self.config)
        events = []
        collector.run_native(core, self.tokenizer, [2] * 3, self.config,
                             lambda event: events.append({key: value.clone() if isinstance(value, torch.Tensor) else value
                                                           for key, value in event.items()}))
        observer(events[0])
        observer(events[2])
        self.assertEqual(len(observer.edges), 0)
        self.assertFalse(observer.rows[1]["prev_valid"])
        self.assertFalse(observer.rows[1]["prev_confidence"].any())

    def test_projection_fixed_and_question_splits(self):
        a = collector.semantic_projection(16)
        b = collector.semantic_projection(16)
        self.assertTrue(torch.equal(a, b))
        splits = collector.question_splits(list(map(str, range(100))))
        self.assertEqual([sum(split == name for split in splits.values())
                          for name in ("train", "validation", "test")], [70, 15, 15])
        self.assertEqual(splits, collector.question_splits(list(map(str, range(100)))))
        self.assertEqual(set(collector.question_splits(["1", "2", "3"]).values()),
                         {"train", "validation", "test"})
        config = collector.normalize_config(dict(max_context_tokens=64, threshold=.5))
        self.assertEqual(config["max_context"], 64)
        with self.assertRaises(ValueError):
            collector.normalize_config(dict(threshold=.6))

    def test_reject_misaligned_hidden(self):
        core = FakeCore()
        events = []
        collector.run_native(core, self.tokenizer, [2] * 3, self.config,
                             lambda event: events.append({key: value.clone() if isinstance(value, torch.Tensor) else value
                                                           for key, value in event.items()}))
        event = events[0]
        event["hidden"] = event["hidden"].roll(1, 1)
        observer = collector.QuestionObserver(core, "wrong", [2] * 3, self.config)
        with self.assertRaisesRegex(AssertionError, "LM-head"):
            observer(event)

    def test_failed_question_preserves_partial_npz(self):
        core = FakeCore()
        observer = collector.QuestionObserver(core, "partial", [2] * 3, self.config)
        def fail_after_one(event):
            if observer.rows:
                raise RuntimeError("test failure")
            observer(event)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            collector.run_native(core, self.tokenizer, [2] * 3, self.config, fail_after_one)
        with tempfile.TemporaryDirectory() as directory:
            record = observer.save(directory, [2] * 3, "error", "test failure")
            self.assertEqual(record["num_states"], 1)
            self.assertEqual(record["error"], "test failure")
            self.assertTrue(record["groups"][0]["incomplete"])

    def test_collect_api_manifest_trainer_contract_and_explicit_prompt_skip(self):
        class Dataset:
            _fingerprint = "test-fingerprint"
            def __init__(self, rows):
                self.rows = rows
            def __len__(self):
                return len(self.rows)
            def __iter__(self):
                return iter(self.rows)
            def add_column(self, name, values):
                return Dataset([{**row, name: value} for row, value in zip(self.rows, values)])
            def shuffle(self, seed):
                return Dataset([self.rows[index] for index in np.random.default_rng(seed).permutation(len(self.rows))])
            def select(self, indices):
                return Dataset([self.rows[index] for index in indices])
        dataset = Dataset([dict(question="normal one"), dict(question="normal two"), dict(question="oversize")])
        datasets = ModuleType("datasets")
        datasets.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
        datasets.load_dataset = lambda *a, **k: dataset
        transformers = ModuleType("transformers")
        transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)
        core = FakeCore()
        load_calls = []
        def load_model(*args, **kwargs):
            load_calls.append(kwargs)
            return core
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
        transformers.AutoModel = SimpleNamespace(from_pretrained=lambda *a, **kw:
            self.fail("AutoModel resolves the headless backbone; use AutoModelForCausalLM"))
        tokenizer = SimpleNamespace(eos_token_id=15, decode=lambda *a, **k: "",
            apply_chat_template=lambda messages, **kw: [2] * (65 if "oversize" in messages[0]["content"] else 35))
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"datasets": datasets, "transformers": transformers}), \
             patch.object(torch.cuda, "is_available", lambda: True):
            path = collector.collect(dict(num_questions=3, max_new_tokens=32, max_context_tokens=64,
                                          threshold=.5), directory, directory)
            self.assertEqual(path.name, "capture_manifest.json")
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["model"]["loading"], "AutoModelForCausalLM trust_remote_code=True")
            self.assertEqual(manifest["model"]["resolved_class"], type(core).__name__)
            self.assertEqual(len(manifest["audits"]), 2)
            self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})
            self.assertTrue(all(len(ids) == 1 for ids in manifest["split"].values()))
            self.assertTrue(load_calls[0]["trust_remote_code"])
            self.assertEqual(load_calls[0]["torch_dtype"], torch.float16)
            required = {"uid", "question_id", "group_id", "step", "forward_id", "npz", "row",
                        "block_start", "context_len", "small_block_index"}
            self.assertTrue(all(required <= set(state) for state in manifest["states"]))
            skips = [row for row in manifest["questions"] if row["terminal_reason"] == "prompt_exceeds_context_cap"]
            self.assertEqual(len(skips), 1)
            self.assertEqual(skips[0]["num_states"], 0)
            self.assertEqual(len(skips[0]["prompt_tokens"]), 65)
            from drafter_simulator_train import StateStore
            store = StateStore(directory)
            self.assertEqual(store.raw_features(0).shape, (32, 16 + 128 + 14))
            self.assertTrue(store.paths([manifest["states"][0]["question_id"]], horizon=3))
            # The first audit's failure must retain the observer's already
            # captured rows and all previously flushed question records.
            original_run = collector.run_native
            def failing_run(model, tokenizer, prompt, config, observer=None):
                if observer is None:
                    return original_run(model, tokenizer, prompt, config)
                def fail_after_one(event):
                    if observer.rows:
                        raise RuntimeError("injected collection failure")
                    observer(event)
                return original_run(model, tokenizer, prompt, config, fail_after_one)
            failed_dir = Path(directory) / "failed"
            with patch.object(collector, "run_native", failing_run), self.assertRaisesRegex(RuntimeError, "injected"):
                collector.collect(dict(num_questions=3, max_new_tokens=32, max_context=64), failed_dir, directory)
            failed = json.loads((failed_dir / "capture_manifest.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(len(failed["questions"]), 2)
            self.assertEqual(len(failed["states"]), 1)
            self.assertFalse(failed["questions"][-1]["terminal_tokens_complete"])
            with np.load(failed_dir / failed["states"][0]["npz"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["tokens_pre"].shape, (1, 32))


class CpuBackedCudaTensor(torch.Tensor):
    """CPU data with CUDA metadata, used only with mocked CUDA runtime tools."""
    @property
    def device(self):
        return torch.device("cuda:0")


class BenchmarkRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch._dynamo

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def fixture(self):
        core = FakeCore()
        core.embedding.to(dtype=torch.float16)
        class Cache:
            length = 32
            def __init__(self):
                self.values = [(torch.ones(1, 1, 32, 2), torch.zeros(1, 1, 32, 2))]
            def get_seq_length(self):
                return self.length
            def __iter__(self):
                return iter(self.values)
        pre = torch.full((1, 32), collector.MASK_ID, dtype=torch.long)
        pre[:, :3] = 2
        output = core.forward(pre, use_cache=True, update_past_key_values=False, output_hidden_states=True)
        hidden = torch.cat([output.hidden_states[-1][:, :1], output.hidden_states[-1][:, :-1]], 1)
        eligible = torch.zeros(32, dtype=torch.bool)
        eligible[3:8] = True
        post, committed = collector.native_commit(pre[0], hidden[0], eligible)
        event = dict(phase="post_native_commit", hidden=hidden, logits=hidden, tokens_pre=pre,
            tokens_post=post[None].as_subclass(CpuBackedCudaTensor), committed=committed[None, :8],
            past_key_values=Cache(), mask_id=collector.MASK_ID, block_start=32, small_block_idx=0,
            context_len=64, forward_id=2, forward_ms=7.0)
        core.calls.clear()
        return core, event, eligible

    @contextlib.contextmanager
    def runtime(self, latencies=(1., 2., 3.)):
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): pass
            def elapsed_time(self, other): return next(samples)
        samples = iter(latencies)
        with patch.object(torch.cuda, "device", lambda *a: contextlib.nullcontext()), \
             patch.object(torch.cuda, "Event", Event), \
             patch.object(torch.cuda, "synchronize") as synchronize, \
             patch.object(torch.cuda, "get_device_name", lambda *a: "mock CUDA runtime"):
            yield synchronize

    def test_runtime_counts_logging_off_and_cache_identity(self):
        core, event, eligible = self.fixture()
        with self.runtime(), patch.object(core, "forward", wraps=core.forward) as forward:
            record = collector.benchmark_native_state(core, event, eligible,
                dict(benchmark_warmup=2, benchmark_repetitions=3))
        self.assertEqual(forward.call_count, 5)
        self.assertEqual(record["extra_forward_passes"], 5)
        for call in forward.call_args_list:
            self.assertIs(call.kwargs["past_key_values"], event["past_key_values"])
            self.assertFalse(call.kwargs["output_hidden_states"])
            self.assertFalse(call.kwargs["update_past_key_values"])
        self.assertEqual(record["dtype"], "torch.float16")
        self.assertFalse(record["generator_forward_count_includes_benchmark"])
        self.assertTrue(record["input_unchanged"])

    def test_cuda_event_percentiles_and_synchronized_wall_samples(self):
        core, event, eligible = self.fixture()
        with self.runtime((1., 2., 10.)) as synchronize:
            record = collector.benchmark_native_state(core, event, eligible,
                dict(benchmark_warmup=0, benchmark_repetitions=3))
        self.assertEqual(synchronize.call_count, 6)
        self.assertEqual(record["native_ms_p50"], 2.)
        self.assertAlmostEqual(record["native_ms_p90"], 8.4)
        self.assertAlmostEqual(record["native_ms_p95"], 9.2)
        self.assertEqual(set(record["wall_ms"]), {"p50", "p90", "p95"})
        self.assertEqual(len(record["samples_wall_ms"]), 3)
        self.assertTrue(all(ms > 0 for ms in record["samples_wall_ms"]))

    def test_benchmark_rejects_hidden_logging_or_mutated_input(self):
        core, event, eligible = self.fixture()
        original = core.forward
        def incorrect_logging(**kwargs):
            output = original(**kwargs)
            output.hidden_states = (output.logits,)
            return output
        with self.runtime(), patch.object(core, "forward", incorrect_logging), \
             self.assertRaisesRegex(AssertionError, "returned hidden states"):
            collector.benchmark_native_state(core, event, eligible, dict(benchmark_warmup=0, benchmark_repetitions=3))
        core, event, eligible = self.fixture()
        original = core.forward
        def mutating(**kwargs):
            output = original(**kwargs)
            kwargs["input_ids"][0, 0] += 1
            return output
        with self.runtime(), patch.object(core, "forward", mutating), \
             self.assertRaisesRegex(AssertionError, "changed|mutated"):
            collector.benchmark_native_state(core, event, eligible, dict(benchmark_warmup=0, benchmark_repetitions=3))

    def test_serializable_results_drop_ephemeral_cache(self):
        core, event, eligible = self.fixture()
        cache_ref = weakref.ref(event["past_key_values"])
        with self.runtime():
            record = collector.benchmark_native_state(core, event, eligible,
                dict(benchmark_warmup=0, benchmark_repetitions=3))
        self.assertEqual(json.loads(json.dumps(record))["cached_prefix_len"], 32)
        self.assertNotIn("past_key_values", record)
        del event
        self.assertIsNone(cache_ref())

    def test_capture_and_benchmark_accounting_saved_separately(self):
        core, event, eligible = self.fixture()
        observer = collector.QuestionObserver(core, "timing", [2] * 35, dict(benchmark_native_states=1))
        zeros, arange = torch.zeros, torch.arange
        def cpu_zeros(*args, **kwargs):
            kwargs.pop("device", None)
            return zeros(*args, **kwargs)
        def cpu_arange(*args, **kwargs):
            kwargs.pop("device", None)
            return arange(*args, **kwargs)
        clock = iter((10., 10.001, 20., 70.))
        benchmark = dict(extra_forward_passes=120)
        with patch.object(torch, "zeros", cpu_zeros), patch.object(torch, "arange", cpu_arange), \
             patch.object(collector.time, "perf_counter", lambda: next(clock)), \
             patch.object(collector, "benchmark_native_state", return_value=benchmark):
            observer(event)
        with tempfile.TemporaryDirectory() as directory:
            record = observer.save(directory, [2] * 35, "error", "timing fixture")
            self.assertAlmostEqual(record["capture_ms"], 1.)
            self.assertEqual(record["benchmark_ms"], 50000.)
            self.assertEqual(record["extra_benchmark_forwards"], 120)
            with np.load(Path(directory) / record["npz"]) as arrays:
                self.assertAlmostEqual(float(arrays["feature_capture_ms"][0]), 1.)
                self.assertEqual(float(arrays["benchmark_ms"][0]), 50000.)
                self.assertEqual(float(arrays["forward_ms"][0]), 7.)

    def test_dataset_budget_first_three_questions_and_zero_disables(self):
        class Dataset:
            _fingerprint = "benchmark-fixture"
            def __init__(self, rows): self.rows = rows
            def __len__(self): return len(self.rows)
            def __iter__(self): return iter(self.rows)
            def add_column(self, key, values):
                return Dataset([{**row, key: value} for row, value in zip(self.rows, values)])
            def shuffle(self, seed): return self
            def select(self, indices): return Dataset([self.rows[i] for i in indices])
        modules = {name: ModuleType(name) for name in ("datasets", "transformers")}
        for name, module in modules.items():
            module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        modules["datasets"].load_dataset = lambda *a, **kw: Dataset([dict(question=str(i)) for i in range(5)])
        core = FakeCore()
        modules["transformers"].AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *a, **kw: core)
        tokenizer = SimpleNamespace(eos_token_id=15, decode=lambda *a, **kw: "",
                                   apply_chat_template=lambda *a, **kw: [2] * 35)
        modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        budgets = []
        class MockBenchmarkObserver(collector.QuestionObserver):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                budgets.append(self.config["benchmark_native_states"])
            def __call__(self, event):
                super().__call__(event)
                if self.config["benchmark_native_states"] and not self.native_benchmarks:
                    self.native_benchmarks.append(dict(question_id=self.question_id,
                        extra_forward_passes=120, benchmark_overhead_ms=100.))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.object(torch.cuda, "is_available", lambda: True), \
             patch.object(torch.cuda, "Event", CpuEvent), patch.object(torch.cuda, "synchronize", lambda: None), \
             patch.object(collector, "QuestionObserver", MockBenchmarkObserver):
            cfg = dict(num_questions=5, max_new_tokens=32, max_context=64)
            manifest = json.loads(collector.collect(cfg, Path(directory)/"enabled", directory).read_text())
            self.assertEqual(budgets, [1, 1, 1, 0, 0])
            self.assertEqual([b["question_id"] for b in manifest["native_benchmarks"]], ["0", "1", "2"])
            self.assertEqual(manifest["extra_benchmark_forwards"], 360)
            budgets.clear()
            manifest = json.loads(collector.collect({**cfg, "benchmark_native_states": 0},
                                                    Path(directory)/"disabled", directory).read_text())
            self.assertEqual(budgets, [0] * 5)
            self.assertEqual(manifest["native_benchmarks"], [])
            self.assertEqual(manifest["extra_benchmark_forwards"], 0)


class NativeTinyModelTest(unittest.TestCase):
    def test_pinned_native_cuda_forward_hidden_logits_cache_and_replay(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        try:
            import transformers
            if transformers.__version__ != "4.53.1":
                self.skipTest("real native integration requires pinned transformers==4.53.1")
            from transformers import Qwen2Config
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if (ROOT / ".test-deps").exists():
                sys.path.insert(0, str(ROOT / ".test-deps"))
            import einops  # required by native model
        except ImportError as exc:
            self.skipTest(f"native integration dependency unavailable: {exc}")
        package = ModuleType("drafter_simulator_tiny")
        package.__path__ = []
        package.__spec__ = importlib.util.spec_from_loader(package.__name__, loader=None, is_package=True)
        configuration = ModuleType("drafter_simulator_tiny.configuration")
        configuration.__spec__ = importlib.util.spec_from_loader(configuration.__name__, loader=None)
        configuration.Fast_dLLM_QwenConfig = Qwen2Config
        spec = importlib.util.spec_from_file_location("drafter_simulator_tiny.modeling",
                                                     ROOT / "Fast_dLLM_v2_1_5B/modeling.py")
        module = importlib.util.module_from_spec(spec)
        def rope(config, device, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            return 1 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)), 1.0
        with patch.dict(sys.modules, {package.__name__: package, configuration.__name__: configuration,
                                      spec.name: module}), \
             patch.dict(ROPE_INIT_FUNCTIONS, {"default": ROPE_INIT_FUNCTIONS.get("default", rope)}):
            spec.loader.exec_module(module)
            torch.manual_seed(42)
            config = Qwen2Config(vocab_size=151680, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=128)
            config.bd_size = 32
            config._attn_implementation = "sdpa"
            model = module.Fast_dLLM_QwenForCausalLM(config).to(device="cuda:0", dtype=torch.float16).eval()
            # Exercise the same local AutoModelForCausalLM path used in Kaggle,
            # with the real checkpoint's distinct backbone/generation mappings.
            with tempfile.TemporaryDirectory(prefix="drafter_native_loader_") as local_model:
                local_model = Path(local_model)
                config.auto_map = {
                    "AutoConfig": "configuration.Fast_dLLM_QwenConfig",
                    "AutoModel": "modeling.Fast_dLLM_QwenModel",
                    "AutoModelForCausalLM": "modeling.Fast_dLLM_QwenForCausalLM",
                }
                model.save_pretrained(local_model)
                saved_config = json.loads((local_model / "config.json").read_text())
                saved_config["model_type"] = "Fast_dLLM_Qwen"
                (local_model / "config.json").write_text(json.dumps(saved_config), encoding="utf-8")
                (local_model / "configuration.py").write_text(
                    'from transformers import Qwen2Config\n'
                    'class Fast_dLLM_QwenConfig(Qwen2Config):\n'
                    '    model_type = "Fast_dLLM_Qwen"\n', encoding="utf-8")
                shutil.copy2(ROOT / "Fast_dLLM_v2_1_5B/modeling.py", local_model / "modeling.py")
                loaded = collector.load_native_drafter(local_model, "cuda:0")
                self.assertEqual(type(loaded).__name__, "Fast_dLLM_QwenForCausalLM")
                self.assertTrue(callable(loaded.generate_draft_tokens_arbitrary_length))
                self.assertTrue(callable(loaded.lm_head))
                self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
                torch.testing.assert_close(loaded.get_input_embeddings().weight,
                                           model.get_input_embeddings().weight)
                model = loaded
            tokenizer = SimpleNamespace(eos_token_id=151645, decode=lambda *a, **kw: "")
            cfg = dict(max_new_tokens=32, max_context=96, benchmark_native_states=1,
                       benchmark_warmup=1, benchmark_repetitions=3)
            native_benchmark = collector.benchmark_native_state
            def checked_benchmark(model, event, eligible, options):
                cache = event["past_key_values"]
                cache_tensors = [(key.clone(), value.clone()) for key, value in cache]
                factual_tokens = event["tokens_post"].clone()
                with patch.object(model, "forward", wraps=model.forward) as forward:
                    result = native_benchmark(model, event, eligible, options)
                self.assertEqual(forward.call_count, options["benchmark_warmup"] + options["benchmark_repetitions"])
                for call in forward.call_args_list:
                    self.assertIs(call.kwargs["past_key_values"], cache)
                    self.assertFalse(call.kwargs["update_past_key_values"])
                    self.assertFalse(call.kwargs["output_hidden_states"])
                self.assertTrue(torch.equal(event["tokens_post"], factual_tokens))
                for (before_key, before_value), (key, value) in zip(cache_tensors, cache):
                    self.assertTrue(torch.equal(before_key, key))
                    self.assertTrue(torch.equal(before_value, value))
                return result
            for prefix in (35, 64):
                with self.subTest(prefix=prefix):
                    with patch.object(collector, "benchmark_native_state", checked_benchmark):
                        observer, tokens, reason, audit = collector.audit_replay(model, tokenizer, [2] * prefix,
                                                                                 cfg, str(prefix))
                    self.assertTrue(audit["tokens_equal"])
                    self.assertTrue(observer.rows)
                    self.assertEqual(observer.lm_head_checks, len(observer.rows))
                    self.assertEqual(tokens[:prefix], [2] * prefix)
                    self.assertTrue(all(state["block_start"] % 32 == 0 for state in observer.states))
                    self.assertEqual(observer.rows[0]["hidden"].dtype, np.float16)
                    self.assertEqual(len(observer.native_benchmarks), 1)
                    benchmark = observer.native_benchmarks[0]
                    self.assertEqual(benchmark["extra_forward_passes"], 4)
                    self.assertEqual(benchmark["device"], "cuda:0")
                    self.assertEqual(benchmark["dtype"], "torch.float16")
                    self.assertFalse(benchmark["output_hidden_states"])
                    self.assertEqual(benchmark["cached_prefix_len"], prefix // 32 * 32)
                    self.assertTrue(all(value > 0 for value in benchmark["native_ms"].values()))
                    self.assertEqual(len(benchmark["samples_native_ms"]), 3)
                    self.assertEqual(observer.states[0]["capture_ms"], observer.states[0]["feature_capture_ms"])
                    self.assertGreater(observer.states[0]["benchmark_ms"], 0)
                    self.assertTrue(all(not isinstance(value, torch.Tensor) for value in benchmark.values()))
            del model, observer


if __name__ == "__main__":
    unittest.main()
