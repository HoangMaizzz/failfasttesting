"""Native drafter-only, post-commit trajectory collection (no verifier).

Importing this module needs NumPy and PyTorch, never Transformers/datasets.
The expensive optional dependencies are loaded only by ``collect``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from collections.abc import Mapping

import numpy as np
import torch

SCHEMA_VERSION = "drafter_simulator_v1"
MASK_ID = 151665
PROJECTION_SEED = 1729
DEFAULTS = dict(num_questions=100, max_new_tokens=256, max_context=4096,
                physical_block_size=32, small_block_size=8,
                drafter_threshold=0.5, collect_device="cuda:0",
                embedding_dim=64, projection_seed=PROJECTION_SEED,
                dataset_seed=42, split_seed=42, audit_prompts=2,
                benchmark_native_states=3, benchmark_warmup=20, benchmark_repetitions=100)


def normalize_config(config):
    values = dict(config) if isinstance(config, Mapping) else vars(config).copy()
    for alias, canonical in (("max_context_tokens", "max_context"), ("threshold", "drafter_threshold")):
        if alias in values:
            if canonical in values and values[canonical] != values[alias]:
                raise ValueError(f"conflicting {alias} and {canonical}")
            values[canonical] = values[alias]
    result = {**DEFAULTS, **values}
    if result["physical_block_size"] != 32 or result["small_block_size"] != 8:
        raise ValueError("native collector requires physical_block_size=32, small_block_size=8")
    if float(result["drafter_threshold"]) != 0.5:
        raise ValueError("native collector requires drafter_threshold=0.5")
    if result["embedding_dim"] not in (64, 128):
        raise ValueError("embedding_dim must be 64 or 128")
    for name in ("num_questions", "max_new_tokens", "max_context"):
        if int(result[name]) < 1:
            raise ValueError(f"{name} must be positive")
        result[name] = int(result[name])
    if result["max_new_tokens"] < 32 or result["max_new_tokens"] % 32:
        raise ValueError("max_new_tokens must be a positive multiple of native physical block size 32")
    for name in ("benchmark_native_states", "benchmark_warmup", "benchmark_repetitions"):
        if type(result[name]) is not int or result[name] < (1 if name == "benchmark_repetitions" else 0):
            raise ValueError(f"{name} must be an integer with valid nonnegative/positive count")
    return result


def question_splits(question_ids, seed=42):
    """Disjoint question-level 70/15/15 split; never split trajectories."""
    ids = list(question_ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    train_count = int(len(ids) * 0.70)
    val_count = int(len(ids) * 0.15)
    if len(ids) >= 3:
        train_count = max(1, min(train_count, len(ids) - 2))
        val_count = max(1, min(val_count, len(ids) - train_count - 1))
    train_end = train_count
    val_end = train_end + val_count
    return {ids[int(index)]: split for split, indices in
            (("train", order[:train_end]), ("validation", order[train_end:val_end]),
             ("test", order[val_end:])) for index in indices}


def semantic_projection(hidden_dim, output_dim=64, seed=PROJECTION_SEED):
    """Fixed Gaussian projection, independent of data and train/test splits."""
    matrix = np.random.default_rng(seed).standard_normal((hidden_dim, output_dim))
    return torch.from_numpy((matrix / math.sqrt(output_dim)).astype(np.float32))


def native_commit(tokens_pre, logits, eligible, mask_id=MASK_ID, threshold=0.5):
    """Audit the native strict-threshold + forced-argmax greedy commit rule.

Softmax intentionally uses the logits' native dtype, matching generation.
"""
    probs = logits.softmax(-1)
    candidates = probs.argmax(-1)
    confidence = probs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
    active = eligible & tokens_pre.eq(mask_id)
    scores = confidence.masked_fill(~active, -torch.inf)
    selected = (scores > threshold) & active
    if active.any():
        selected[scores.argmax()] = True
    tokens = tokens_pre.clone()
    tokens[selected] = candidates[selected]
    return tokens, selected


def _cpu(tensor, dtype=None):
    value = tensor.detach().to(device="cpu", dtype=dtype).contiguous().numpy()
    return value.copy()


def _cache_signature(cache):
    """CPU metadata only; no cache tensor is retained or copied for timing."""
    if cache is None:
        return (0, ())
    length = int(cache.get_seq_length())
    return length, tuple((tuple(tensor.shape), tensor.data_ptr())
                         for key, value in cache for tensor in (key, value))


@torch.inference_mode()
def benchmark_native_state(model, event, eligible, config):
    """Time a logging-disabled native Refine action at a fixed real post state.

The prefix cache belongs to the active callback. Direct forwards never update
it; only scalar/CPU benchmark results survive this function's return.
"""
    config = normalize_config(config)
    post = event["tokens_post"]
    device = post.device
    if device.type != "cuda":
        raise ValueError("native benchmark requires CUDA events on the drafter device")
    cache = event["past_key_values"]
    cache_before = _cache_signature(cache)
    input_before = post.clone()
    lo = int(event["small_block_idx"]) * config["small_block_size"]
    hi = lo + config["small_block_size"]
    active = post[:, lo:hi].eq(int(event["mask_id"]))
    if not bool((eligible & post[0].eq(int(event["mask_id"]))).any()):
        raise ValueError("native benchmark needs remaining masks in the eligible span")

    def step():
        output = model.forward(input_ids=post, use_cache=True, past_key_values=cache,
                               update_past_key_values=False, output_hidden_states=False)
        if getattr(output, "hidden_states", None) is not None:
            raise AssertionError("logging-disabled benchmark returned hidden states")
        shifted = torch.cat([output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1)
        logits = shifted[:, lo:hi]
        # Same eight-position softmax, strict threshold, and forced argmax as
        # native generation; cloning prevents any factual token modification.
        candidates, probabilities = model.sample_with_top_p(logits, top_p=1.0, temperature=0.0)
        confidence = probabilities.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
        scores = torch.where(active, confidence, -torch.inf)
        selected = scores > config["drafter_threshold"]
        selected[torch.arange(candidates.shape[0]), scores.argmax(-1)] = True
        selected &= active
        tokens = post.clone()
        tokens[:, lo:hi][selected] = candidates[selected]
        return tokens, selected, shifted

    warmup = config["benchmark_warmup"]
    repeats = config["benchmark_repetitions"]
    first = None
    for _ in range(warmup):
        result = step()
        if first is None:
            first = _cpu(result[0], torch.int64)
    values, wall = [], []
    started = time.perf_counter()
    with torch.cuda.device(device):
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            result = step()
            end.record()
            torch.cuda.synchronize(device)
            values.append(float(start.elapsed_time(end)))
            wall.append((time.perf_counter() - before) * 1000)
            if first is None:
                first = _cpu(result[0], torch.int64)
    repetitions_wall_ms = (time.perf_counter() - started) * 1000
    # Untimed validation uses the independently tested full-canvas commit rule.
    expected, selected = native_commit(post[0], result[2][0], eligible,
                                        int(event["mask_id"]), config["drafter_threshold"])
    if not torch.equal(result[0][0], expected) or not torch.equal(result[1][0], selected[lo:hi]):
        raise AssertionError("benchmark action differs from native greedy commit")
    if not np.array_equal(first, _cpu(result[0], torch.int64)):
        raise AssertionError("benchmark repeated action changed at a fixed native state")
    if not torch.equal(post, input_before) or _cache_signature(cache) != cache_before:
        raise AssertionError("benchmark mutated factual tokens or prefix cache")
    percentiles = lambda samples: {f"p{p}": float(np.percentile(samples, p)) for p in (50, 90, 95)}
    native_ms = percentiles(values)
    return dict(context_len=int(event["context_len"]), block_start=int(event["block_start"]),
        small_block_index=int(event["small_block_idx"]), forward_id=int(event["forward_id"]),
        physical_block_size=32, small_block_size=8, batch_size=1, device=str(device),
        device_name=torch.cuda.get_device_name(device), dtype=str(model.get_input_embeddings().weight.dtype),
        output_hidden_states=False, observer_enabled=False, use_cache=True, use_block_cache=False,
        update_past_key_values=False, cached_prefix_len=cache_before[0], post_commit_state=True,
        remaining_eligible_masks=int((eligible & post[0].eq(int(event["mask_id"]))).sum().item()),
        workload="native_full32_forward_right_shift_greedy_small8_commit_on_clone",
        warmup=warmup, repetitions=repeats, extra_forward_passes=warmup + repeats,
        generator_forward_count_includes_benchmark=False, native_ms=native_ms,
        native_ms_p50=native_ms["p50"], native_ms_p90=native_ms["p90"], native_ms_p95=native_ms["p95"],
        gpu_or_cpu_ms=native_ms, wall_ms=percentiles(wall),
        samples_native_ms=values, samples_wall_ms=wall,
        timed_repetitions_wall_ms=repetitions_wall_ms,
        input_unchanged=True, prefix_cache_metadata_unchanged=True, native_rule_equivalent=True,
        note="Direct native action only; excludes observation capture and end-to-end decoding overhead.")


class QuestionObserver:
    """Synchronously consume ephemeral GPU views; retain CPU arrays only."""

    def __init__(self, model, question_id, prompt_ids, config):
        self.model = model
        self.question_id = str(question_id)
        self.prompt_ids = list(prompt_ids)
        self.config = normalize_config(config)
        weight = model.get_input_embeddings().weight
        self.projection = semantic_projection(weight.shape[1], self.config["embedding_dim"],
                                              self.config["projection_seed"])
        self.rows = []
        self.states = []
        self.edges = []
        self.groups = {}
        self.native_rule_checks = 0
        self.lm_head_checks = 0
        self.native_stats = None
        self.native_benchmarks = []

    @torch.inference_mode()
    def __call__(self, event):
        started = event.get("observation_started", time.perf_counter())
        if event.get("phase") != "post_native_commit":
            raise ValueError("observer requires post-native-commit events")
        width = self.config["physical_block_size"]
        hidden = event["hidden"]
        logits = event["logits"]
        pre = event["tokens_pre"]
        post = event["tokens_post"]
        if hidden.ndim != 3 or hidden.shape[:2] != (1, width):
            raise ValueError("hidden must include all 32 physical canvas positions")
        if logits.shape[:2] != (1, width) or pre.shape != (1, width) or post.shape != (1, width):
            raise ValueError("observer must capture a batch-one full physical canvas")
        block_start = int(event["block_start"])
        small_index = int(event["small_block_idx"])
        context_len = int(event["context_len"])
        forward_id = int(event["forward_id"])
        eligible = torch.zeros(width, dtype=torch.bool, device=post.device)
        lo = small_index * self.config["small_block_size"]
        hi = lo + self.config["small_block_size"]
        eligible[lo:hi] = True
        # The group span stays fixed across forwards; committed partial-prefix
        # positions belong to the canvas but are not eligible generated tokens.
        positions = torch.arange(width, device=post.device) + block_start
        eligible &= positions >= len(self.prompt_ids)
        expected, selected = native_commit(pre[0], logits[0], eligible,
                                           int(event["mask_id"]), self.config["drafter_threshold"])
        if not torch.equal(expected, post[0]):
            raise AssertionError("observed commit differs from native .5 + forced argmax rule")
        if not torch.equal(selected[lo:hi], event["committed"][0]):
            raise AssertionError("native committed mask disagrees with replay")
        self.native_rule_checks += 1
        mask = post[0].eq(int(event["mask_id"]))
        # Use full-vocabulary FP32 probabilities for features, rather than top-k
        # renormalization; commit audit above still uses native FP16 arithmetic.
        log_probs = logits[0].float().log_softmax(-1)
        probs = log_probs.exp()
        top = probs.topk(2, -1)
        candidates = logits[0].softmax(-1).argmax(-1)
        head_candidates = self.model.get_output_embeddings()(hidden).softmax(-1).argmax(-1)
        if not torch.equal(head_candidates[0], candidates):
            raise AssertionError("aligned hidden LM-head top1 differs from native shifted logits")
        self.lm_head_checks += 1
        confidence = probs.gather(-1, candidates[:, None]).squeeze(-1)
        entropy = -(probs * log_probs).sum(-1) / math.log(logits.shape[-1])
        margin = top.values[:, 0] - top.values[:, 1]
        weight = self.model.get_input_embeddings().weight
        # The projection is CPU-owned. Only transient projection/embedding
        # tensors exist on the accelerator during this callback.
        projection = self.projection.to(weight.device)
        token_emb = weight[post[0]].float() @ projection
        candidate_emb = weight[candidates].float() @ projection
        group_id = f"{self.question_id}/{block_start}/{small_index}"
        previous = self.rows[-1] if self.rows else None
        consecutive = bool(previous is not None and previous["group_id"] == group_id
                           and previous["forward_id"] + 1 == forward_id
                           and np.array_equal(previous["eligible"], _cpu(eligible)))
        current = dict(hidden=_cpu(hidden[0], torch.float16),
                       token_emb=_cpu(token_emb, torch.float16),
                       candidate_emb=_cpu(candidate_emb, torch.float16),
                       mask=_cpu(mask), eligible=_cpu(eligible),
                       confidence=_cpu(confidence, torch.float32),
                       entropy=_cpu(entropy, torch.float32), margin=_cpu(margin, torch.float32),
                       candidate_ids=_cpu(candidates, torch.int64), tokens=_cpu(post[0], torch.int64),
                       tokens_pre=_cpu(pre[0], torch.int64), mask_pre=_cpu(pre[0].eq(int(event["mask_id"]))),
                       committed=_cpu(selected),
                       positions=_cpu(positions, torch.int64), forward_ms=float(event["forward_ms"]),
                       forward_id=forward_id, group_id=group_id,
                       prev_valid=consecutive)
        current["prev_confidence"] = previous["confidence"].copy() if consecutive else np.zeros(width, np.float32)
        current["prev_entropy"] = previous["entropy"].copy() if consecutive else np.zeros(width, np.float32)
        current["prev_margin"] = previous["margin"].copy() if consecutive else np.zeros(width, np.float32)
        current["prev_candidate_ids"] = previous["candidate_ids"].copy() if consecutive else np.full(width, -1, np.int64)
        current["prev_mask"] = previous["mask"].copy() if consecutive else np.zeros(width, bool)
        current["candidate_changed"] = ((current["candidate_ids"] != previous["candidate_ids"])
                                          if consecutive else np.zeros(width, bool))
        group = self.groups.setdefault(group_id, dict(group_id=group_id, question_id=self.question_id,
            block_start=block_start, small_block_idx=small_index, state_uids=[]))
        step = len(group["state_uids"])
        row = len(self.rows)
        uid = f"{group_id}/{step}"
        if consecutive:
            if np.any(current["mask"] & ~previous["mask"]):
                raise AssertionError("post-commit mask sequence is not monotonic")
            frozen = ~previous["mask"]
            if not np.array_equal(current["tokens"][frozen], previous["tokens"][frozen]):
                raise AssertionError("native generation modified committed physical positions")
            self.edges.append(dict(source=self.states[-1]["uid"], target=uid,
                                   source_row=row - 1, target_row=row, group_id=group_id,
                                   source_forward_id=previous["forward_id"], target_forward_id=forward_id))
        group["state_uids"].append(uid)
        group["terminal_row"] = row
        state = dict(uid=uid, question_id=self.question_id, row=row, group_id=group_id,
            step=step, local_step=step, forward_id=forward_id, context_len=context_len,
            block_start=block_start, blockstart=block_start, small_block_idx=small_index,
            small_block_index=small_index, smallidx=small_index,
            eligible_span=[max(len(self.prompt_ids), block_start + lo), block_start + hi],
            phase="post_native_commit", hidden_provenance="same_forward_precommit_final_norm_native_right_shift",
            logits_provenance="same_forward_precommit_native_right_shift",
            tokens_provenance="actual_native_postcommit", prev_valid=consecutive)
        current["capture_ms"] = (time.perf_counter() - started) * 1000
        current["feature_capture_ms"] = current["capture_ms"]
        current["benchmark_ms"] = 0.0
        state["forward_ms"] = current["forward_ms"]
        state["capture_ms"] = current["capture_ms"]
        state["feature_capture_ms"] = current["capture_ms"]
        state["benchmark_ms"] = 0.0
        self.rows.append(current)
        self.states.append(state)
        if (post.device.type == "cuda"
                and len(self.native_benchmarks) < self.config["benchmark_native_states"]
                and bool((mask & eligible).any())):
            benchmark_started = time.perf_counter()
            benchmark = benchmark_native_state(self.model, event, eligible, self.config)
            benchmark.update(uid=uid, question_id=self.question_id, group_id=group_id, row=row,
                             benchmark_overhead_ms=(time.perf_counter() - benchmark_started) * 1000)
            self.native_benchmarks.append(benchmark)
            state["benchmark_overhead_ms"] = benchmark["benchmark_overhead_ms"]
            state["benchmark_ms"] = benchmark["benchmark_overhead_ms"]
            current["benchmark_ms"] = benchmark["benchmark_overhead_ms"]

    def save(self, output, terminal_tokens, reason, error=None):
        """Flush a question even after failure; terminal labels never fill masks."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"question_{self.question_id}.npz"
        width = self.config["physical_block_size"]
        dim = self.model.get_input_embeddings().weight.shape[1]
        shape = dict(hidden=(width, dim), token_emb=(width, self.config["embedding_dim"]),
                     candidate_emb=(width, self.config["embedding_dim"]))
        float_keys = ("hidden", "token_emb", "candidate_emb", "confidence", "entropy", "margin",
                      "prev_confidence", "prev_entropy", "prev_margin")
        bool_keys = ("mask", "eligible", "prev_mask", "candidate_changed", "mask_pre", "committed")
        int_keys = ("candidate_ids", "tokens", "positions", "prev_candidate_ids", "tokens_pre")
        arrays = {}
        for key in (*float_keys, *bool_keys, *int_keys):
            dtype = (np.float16 if key in ("hidden", "token_emb", "candidate_emb") else
                     np.float32 if key in float_keys else bool if key in bool_keys else np.int64)
            arrays[key] = (np.stack([row[key] for row in self.rows]) if self.rows else
                           np.empty((0, *shape.get(key, (width,))), dtype=dtype))
        for key, dtype in (("forward_ms", np.float32), ("capture_ms", np.float32),
                           ("feature_capture_ms", np.float32), ("benchmark_ms", np.float32),
                           ("forward_id", np.int64), ("prev_valid", bool), ("group_id", str)):
            arrays[key] = np.asarray([row[key] for row in self.rows], dtype=dtype)
        # Aliases make both provenance and consumer terminology explicit.
        arrays["absolute_positions"] = arrays["positions"]
        arrays["forward_index"] = arrays["forward_id"]
        arrays["group_ids"] = arrays["group_id"]
        np.savez_compressed(path, **arrays)
        for state in self.states:
            state["npz"] = path.name
            state["npz_path"] = path.name
            state["terminal"] = self.groups[state["group_id"]]["terminal_row"] == state["row"]
        for edge in self.edges:
            edge.update(source_npz=path.name, target_npz=path.name)
        groups = []
        for group in self.groups.values():
            group = dict(group)
            last = self.rows[group["terminal_row"]]
            incomplete = bool(np.any(last["mask"] & last["eligible"]))
            group.update(terminal_uid=group["state_uids"][-1],
                         terminal_tokens=last["tokens"].tolist(), terminal_mask=last["mask"].tolist(),
                         terminal_reason=reason if incomplete else "all_masks_resolved",
                         generation_terminal_reason=reason,
                         incomplete=incomplete, eos_terminated_incomplete=(reason == "eos" and incomplete))
            groups.append(group)
            for state in self.states:
                if state["group_id"] == group["group_id"]:
                    state.update(terminal_uid=group["terminal_uid"], terminal_row=group["terminal_row"],
                                 terminal_reason=group["terminal_reason"], group_incomplete=incomplete,
                                 eos_terminated_incomplete=group["eos_terminated_incomplete"])
        return dict(question_id=self.question_id, npz=path.name, states=self.states,
                    edges=self.edges, groups=groups, terminal_tokens=list(terminal_tokens),
                    terminal_reason=reason, error=error, num_states=len(self.rows),
                    native_rule_checks=self.native_rule_checks,
                    native_benchmarks=self.native_benchmarks,
                    extra_benchmark_forwards=sum(row["extra_forward_passes"] for row in self.native_benchmarks),
                    benchmark_ms=sum(row["benchmark_overhead_ms"] for row in self.native_benchmarks),
                    lm_head_checks=self.lm_head_checks,
                    native_forward_pass_breakdown=(self.native_stats or {}).get("forward_pass_breakdown"),
                    terminal_tokens_complete=error is None,
                    forward_ms=sum(row["forward_ms"] for row in self.rows),
                    capture_ms=sum(row["capture_ms"] for row in self.rows))


def run_native(model, tokenizer, prompt_ids, config, observer=None):
    """Natural native generation: no oracle, controller, artificial pass cap."""
    config = normalize_config(config)
    method = model.generate_draft_tokens_arbitrary_length
    parameters = inspect.signature(method).parameters
    if "is_drafter" not in parameters and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        raise TypeError("native generator must support is_drafter=False")
    args = SimpleNamespace(target_tokenizer=tokenizer, drafter_simulator_observer=observer,
                           drafter_simulator_max_context=config["max_context"],
                           full_refinement_oracle=False, adaptive_td=False,
                           global_oracle_graph=False, strict_greedy_local_oracle=False,
                           collect_bucket_oracle=False, frontier_stop_mode="disabled")
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        result = method(inputs, max_new_tokens=config["max_new_tokens"],
            mask_id=MASK_ID, threshold=config["drafter_threshold"], small_block_size=8,
            block_size=32, stop_token=tokenizer.eos_token_id, temperature=0.0, top_p=1.0,
            use_block_cache=False, is_drafter=False, spec_len=8, max_spec_len=8, incr_len=8,
            return_prefill_kvs=False, return_frontier_stats=True, args=args)
    tokens = result[0][0].detach().cpu().tolist()
    stats = result[-1]
    if observer is not None and hasattr(observer, "native_stats"):
        observer.native_stats = stats
    generated = tokens[len(prompt_ids):]
    if tokenizer.eos_token_id in generated:
        reason = "eos"
    elif stats.get("drafter_simulator_context_cap_reached"):
        reason = "context_cap"
    else:
        reason = "native_max_new_tokens"
    return tokens, reason, stats


def audit_replay(model, tokenizer, prompt_ids, config, question_id="audit", observer=None):
    """Replay selected real prompts; observer must preserve exact native output."""
    if observer is None:
        observer = QuestionObserver(model, question_id, prompt_ids, config)
    plain, plain_reason, plain_stats = run_native(model, tokenizer, prompt_ids, config)
    captured, captured_reason, captured_stats = run_native(model, tokenizer, prompt_ids, config, observer)
    if plain != captured or plain_reason != captured_reason:
        raise AssertionError("observer-on/off replay changed native generated tokens or termination")
    if plain_stats["forward_pass_breakdown"] != captured_stats["forward_pass_breakdown"]:
        raise AssertionError("observer changed native forward count")
    return observer, captured, captured_reason, dict(question_id=str(question_id),
        tokens_equal=True, termination_equal=True, forward_counts_equal=True,
        native_rule_checks=observer.native_rule_checks, lm_head_checks=observer.lm_head_checks,
        forward_pass_breakdown=captured_stats["forward_pass_breakdown"])


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_native_drafter(dllm_dir, device):
    """Resolve the generation class, not AutoModel's headless backbone."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(dllm_dir), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation="sdpa")
    if not callable(getattr(model, "generate_draft_tokens_arbitrary_length", None)):
        raise RuntimeError(f"Loaded {type(model).__name__}, but the native drafter generator is missing")
    if not callable(getattr(model, "lm_head", None)):
        raise RuntimeError(f"Loaded {type(model).__name__}, but the native drafter LM head is missing")
    return model.to(device).eval().requires_grad_(False)


def collect(config, output, dllm_dir):
    """Collect GSM8K train natural trajectories and return the manifest Path.

The launcher supplies downloaded local weights with the instrumented modeling
file. Each completed/failed question is flushed before the next one starts.
"""
    config = normalize_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "capture_manifest.json"
    manifest = dict(schema_version=SCHEMA_VERSION, config=config,
        model=dict(local_path=str(Path(dllm_dir).resolve()), dtype="float16", device=config["collect_device"],
                   loading="AutoModelForCausalLM trust_remote_code=True", verifier=None),
        dataset=dict(id="openai/gsm8k", name="main", source_split="train", shuffle_seed=config["dataset_seed"]),
        semantic_projection=dict(kind="fixed_gaussian_native_input_embeddings", dim=config["embedding_dim"],
                                 seed=config["projection_seed"], train_independent=True),
        provenance=dict(phase="post_native_commit", hidden="same_forward_precommit_final_norm_native_right_shift",
                        physical_canvas=32, native_logits="right_shift_[0,0,1,...,30]",
                        forward_ms="native CUDA events including native logit alignment; excludes callback",
                        capture_ms="synchronous callback wall time including feature computation and CPU copies"),
        questions=[], states=[], edges=[], groups=[], audits=[], native_benchmarks=[],
        extra_benchmark_forwards=0, status="initializing")
    _write_manifest(manifest_path, manifest)
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        if config["collect_device"] != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("production collection requires FP16 single GPU cuda:0")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        if config["num_questions"] > len(dataset):
            raise ValueError("num_questions exceeds GSM8K train size")
        dataset = dataset.add_column("drafter_source_index", list(range(len(dataset))))
        dataset = dataset.shuffle(seed=config["dataset_seed"]).select(range(config["num_questions"]))
        ids = [str(row["drafter_source_index"]) for row in dataset]
        splits = question_splits(ids, config["split_seed"])
        manifest["splits"] = {name: [qid for qid in ids if splits[qid] == name]
                              for name in ("train", "validation", "test")}
        manifest["split"] = manifest["splits"]
        manifest["dataset"]["fingerprint"] = getattr(dataset, "_fingerprint", None)
        tokenizer = AutoTokenizer.from_pretrained(str(dllm_dir), trust_remote_code=True, local_files_only=True)
        model = load_native_drafter(dllm_dir, "cuda:0")
        manifest["model"]["resolved_class"] = type(model).__name__
        method_source = inspect.getsource(model.generate_draft_tokens_arbitrary_length)
        if "drafter_simulator_observer" not in method_source:
            raise RuntimeError("local model code lacks native drafter_simulator_observer hook")
        manifest["model"]["native_source_sha256"] = hashlib.sha256(method_source.encode()).hexdigest()
        manifest["model"]["hidden_dim"] = int(model.get_input_embeddings().weight.shape[1])
        manifest["model"]["embedding_vocabulary_size"] = int(model.get_input_embeddings().weight.shape[0])
        manifest["semantic_projection"]["sha256"] = hashlib.sha256(
            semantic_projection(manifest["model"]["hidden_dim"], config["embedding_dim"],
                                config["projection_seed"]).numpy().tobytes()).hexdigest()
        manifest["status"] = "collecting"
        _write_manifest(manifest_path, manifest)
        audit_count = 0
        for row, question_id in zip(dataset, ids):
            messages = [{"role": "user", "content": f"Solve this math question and give the answer.\n\n{row['question']}"}]
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            # Bound total benchmark work across the dataset, with at most one
            # eligible post state per question. Other callbacks only capture.
            observer_config = {**config, "benchmark_native_states": min(1, max(0,
                config["benchmark_native_states"] - len(manifest["native_benchmarks"])))}
            observer = QuestionObserver(model, question_id, prompt_ids, observer_config)
            tokens, reason, error = prompt_ids, "error", None
            try:
                if len(prompt_ids) > config["max_context"]:
                    reason = "prompt_exceeds_context_cap"
                elif audit_count < config["audit_prompts"]:
                    observer, tokens, reason, audit = audit_replay(model, tokenizer, prompt_ids, config,
                                                                  question_id, observer=observer)
                    manifest["audits"].append(audit)
                    if observer.rows:
                        audit_count += 1
                else:
                    tokens, reason, _ = run_native(model, tokenizer, prompt_ids, config, observer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                reason = "error"
            record = observer.save(output, tokens, reason, error)
            record.update(source_index=int(question_id), dataset_id="openai/gsm8k", split=splits[question_id],
                          prompt_len=len(prompt_ids), prompt_tokens=prompt_ids, question=row["question"])
            manifest["questions"].append({k: v for k, v in record.items() if k not in ("states", "edges", "groups")})
            for name in ("states", "edges", "groups", "native_benchmarks"):
                manifest[name].extend(record[name])
            manifest["extra_benchmark_forwards"] += record["extra_benchmark_forwards"]
            _write_manifest(manifest_path, manifest)
            print(json.dumps(dict(question_id=question_id, states=record["num_states"], reason=reason, error=error)), flush=True)
            del observer
            if error:
                raise RuntimeError(error)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise
    _write_manifest(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dllm-dir", required=True, type=Path)
    args = parser.parse_args()
    print(collect(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.dllm_dir))


if __name__ == "__main__":
    main()
