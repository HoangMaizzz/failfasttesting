"""Frozen, uncompressed Qwen teacher capture and measured early-exit latency.

Depths are one-based block counts. The final depth means post-final-norm
hidden (the LM head input); other depths mean raw decoder block output.
Production loading is pinned to the historical checkpoint and Transformers.
``tiny_config`` is an explicit offline test path, never a teacher substitute.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from numbers import Integral
import time
from typing import Mapping

import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TRANSFORMERS_VERSION = "4.53.1"
LENGTH_BINS = (128, 256, 512, 1024, 2048, 4096, 8192)


def _cfg(cfg):
    return dict(cfg) if isinstance(cfg, Mapping) else vars(cfg)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def depth_indices(num_layers: int) -> list[int]:
    """Ceiling of 25/50/75/100% depth, deduplicated for tiny models."""
    n = _positive_int(num_layers, "num_layers")
    return sorted({(n * quarter + 3) // 4 for quarter in (1, 2, 3, 4)})


def _device_map(num_layers):
    split = (num_layers + 1) // 2
    result = {"model.embed_tokens": 0, "model.rotary_emb": 0}
    result.update({f"model.layers.{i}": 0 if i < split else 1
                   for i in range(num_layers)})
    result.update({"model.norm": 1, "lm_head": 1})
    return result


def _reject_quantization(obj):
    for name in ("quantization_config", "load_in_4bit", "load_in_8bit",
                 "is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized"):
        value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
        if value is not None and value is not False:
            raise ValueError("quantization is forbidden for native verifier capture")


def load_verifier(identity: dict, cfg):
    """Load historical FP16 SDPA teacher, with a contiguous two-GPU map.

    cfg: device='cpu' for local cached checkpoints, or tiny_config={Qwen2
    config fields} for randomly initialized offline tests. Production defaults
    to exactly GPU 0/1 and Transformers 4.53.1. No automatic CPU offload.
    """
    options = _cfg(cfg)
    _reject_quantization(options)
    if not isinstance(identity, Mapping):
        raise ValueError("identity must contain model_id and revision")
    if not all(isinstance(identity.get(k), str) and identity[k].strip()
               for k in ("model_id", "revision")):
        raise ValueError("identity must contain explicit model_id and revision")
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

    tiny = options.get("tiny_config")
    cpu = options.get("device") == "cpu"
    if tiny is not None:
        if not cpu:
            raise ValueError("tiny_config requires device='cpu'")
        _reject_quantization(tiny)
        config = Qwen2Config(**tiny)
        config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(config).to(device="cpu", dtype=torch.float16)
    else:
        if dict(identity).get("model_id") != MODEL_ID or identity["revision"] != REVISION:
            raise ValueError("verifier identity differs from the exact historical checkpoint")
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"historical verifier requires Transformers {TRANSFORMERS_VERSION}")
        if not cpu and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
            raise RuntimeError("native verifier requires GPU 0 and GPU 1; use tiny_config for CPU tests")
        local_only = bool(options.get("local_files_only", cpu))
        config = AutoConfig.from_pretrained(identity["model_id"], revision=identity["revision"],
                                            local_files_only=local_only, trust_remote_code=False)
        _reject_quantization(config)
        if config.model_type != "qwen2" or config.tie_word_embeddings:
            raise ValueError("expected Qwen2 with untied embedding/head for the device map")
        mapping = {"": "cpu"} if cpu else _device_map(config.num_hidden_layers)
        model = AutoModelForCausalLM.from_pretrained(
            identity["model_id"], revision=identity["revision"], config=config,
            torch_dtype=torch.float16, attn_implementation="sdpa", device_map=mapping,
            low_cpu_mem_usage=True, local_files_only=local_only, trust_remote_code=False)
    _reject_quantization(model)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    model.config.output_hidden_states = False
    model.config.output_attentions = False
    model.native_verifier_identity = dict(identity)
    _validate_model(model)
    if not cpu:
        expected = _device_map(model.config.num_hidden_layers)
        for name, target in expected.items():
            module = model.get_submodule(name)
            devices = {p.device for p in module.parameters()} | {b.device for b in module.buffers()}
            if any(d != torch.device("cuda", target) for d in devices):
                raise RuntimeError(f"unexpected device or offload for {name}: {devices}")
    return model


def _validate_model(model):
    _reject_quantization(model)
    _reject_quantization(model.config)
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("native capture requires Qwen2ForCausalLM")
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("verifier must be in eval mode")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("verifier must be frozen")
    if any(not p.is_floating_point() or p.device.type == "meta" for p in model.parameters()):
        raise ValueError("verifier parameters must be floating point and materialized")
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise ValueError("block count differs from config.num_hidden_layers")


def _depths(model, depths):
    if not isinstance(depths, (list, tuple)) or not depths:
        raise ValueError("depths must be a nonempty list of block counts")
    n = model.config.num_hidden_layers
    result = [_positive_int(d, "depth") for d in depths]
    if max(result) > n or len(result) != len(set(result)):
        raise ValueError(f"depths must be unique and in range 1..{n}")
    return result


def _tokens(model, prefix, candidate):
    vocab = model.get_input_embeddings().num_embeddings
    for name, values in (("prefix", prefix), ("candidate", candidate)):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty token list")
        if any(isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < vocab
               for t in values):
            raise ValueError(f"{name} token IDs must be integers in vocabulary range")
    if len(prefix) + len(candidate) > model.config.max_position_embeddings:
        raise ValueError("prefix plus candidate exceeds max_position_embeddings")
    ids = torch.tensor([prefix + candidate], dtype=torch.long,
                       device=model.get_input_embeddings().weight.device)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                output_hidden_states=False, output_attentions=False,
                return_dict=True, logits_to_keep=len(candidate) + 1)


def _hidden(output):
    # Qwen2 4.53.1 returns a tuple; 4.57.3 returns the tensor directly.
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.inference_mode()
def capture_one(model, prefix: list, candidate: list, depths) -> dict:
    """Capture only causal proposal positions; return no logits as features.

    ``hidden[depth]`` is CPU FP16 [L,d]. ``final_hidden_with_bonus`` is CPU
    FP16 [L+1,d] for auditing the bonus prediction. predictions has L+1 IDs.
    Alignment failure raises immediately, with hooks removed in all cases.
    """
    _validate_model(model)
    selected = _depths(model, depths)
    inputs = _tokens(model, prefix, candidate)
    start, length = len(prefix) - 1, len(candidate)
    n = model.config.num_hidden_layers
    captured, handles = {}, []

    def hook(depth):
        def collect(module, args, output):
            h = _hidden(output)
            count = length + 1 if depth == n else length
            captured[depth] = h[0, start:start + count].detach().clone()
        return collect

    try:
        for depth in selected:
            if depth != n:
                handles.append(model.model.layers[depth - 1].register_forward_hook(hook(depth)))
        handles.append(model.model.norm.register_forward_hook(hook(n)))
        actual = model(**inputs).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    if actual.shape != (length + 1, model.config.vocab_size):
        raise RuntimeError("verifier did not honor logits_to_keep=L+1")
    final = captured[n]
    head = model.get_output_embeddings()
    head_device = head.weight.device
    replay = head(final.to(head_device).unsqueeze(0))[0]
    actual = actual.to(head_device)
    saved_final = final.to(dtype=torch.float16, device="cpu")
    replay_fp16 = head(saved_final.to(device=head_device, dtype=head.weight.dtype).unsqueeze(0))[0]
    delta = (replay.float() - actual.float()).abs()
    delta_fp16 = (replay_fp16.float() - actual.float()).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(replay_fp16).all()
                  and all(torch.isfinite(h).all() for h in captured.values()))
    passed = finite and torch.allclose(replay, actual, atol=1e-3, rtol=1e-3)
    passed = passed and torch.allclose(replay_fp16.float(), actual.float(), atol=2e-3, rtol=2e-3)
    audit = dict(passed=bool(passed), finite=finite, max_abs_error=float(delta.max()),
                 max_relative_error=float((delta / actual.float().abs().clamp_min(1e-6)).max()),
                 fp16_max_abs_error=float(delta_fp16.max()),
                 compared_positions=length + 1, candidate_positions=length,
                 causal_indices=list(range(start, start + length)), bonus_index=start + length,
                 final_representation="post_final_norm_lm_head_input",
                 intermediate_representation="raw_block_output", atol=1e-3, rtol=1e-3,
                 fp16_atol=2e-3, fp16_rtol=2e-3, no_logits_as_features=True)
    if not passed:
        raise RuntimeError(f"hidden-to-logit alignment failed: {audit}")
    predictions = actual.argmax(-1).cpu().tolist()
    accepted = 0
    for prediction, token in zip(predictions, candidate):
        if prediction != token:
            break
        accepted += 1
    return dict(hidden={d: captured[d][:length].to(device="cpu", dtype=torch.float16).clone()
                        for d in selected}, predictions=predictions, K=accepted,
                alignment=audit, final_hidden_with_bonus=saved_final,
                prefix_length=len(prefix), proposal_length=length)


class _ReachedCut(Exception):
    """Private control flow exception: no hidden tensor retained or copied."""


def _stop_at_cut(module, args, output):
    raise _ReachedCut()


def _cuda_devices(model):
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    return sorted((d for d in devices if d.type == "cuda"), key=lambda d: d.index)


def _sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def _measure(model, prepared, depth, warmups, repetitions, devices):
    handle = None
    if depth is not None:
        module = model.model.norm if depth == model.config.num_hidden_layers else model.model.layers[depth - 1]
        handle = module.register_forward_hook(_stop_at_cut)

    def run(inputs):
        try:
            result = model(**inputs)
        except _ReachedCut:
            if depth is None:
                raise
        else:
            if depth is not None:
                raise RuntimeError("partial verifier failed to exit at the requested cut")
            del result

    measurements = []
    try:
        for i in range(warmups):
            run(prepared[i % len(prepared)][1])
        _sync(devices)
        for i in range(repetitions):
            row, inputs = prepared[i % len(prepared)]
            _sync(devices)
            before = time.perf_counter_ns()
            run(inputs)
            _sync(devices)
            elapsed_ms = (time.perf_counter_ns() - before) / 1e6
            measurements.append(dict(repetition=i, uid=row["uid"],
                                     sequence_length=inputs["input_ids"].shape[1],
                                     proposal_length=len(row["candidate"]), elapsed_ms=elapsed_ms))
    finally:
        if handle is not None:
            handle.remove()
    return dict(mean_ms=sum(m["elapsed_ms"] for m in measurements) / repetitions,
                raw_measurements=measurements, warmups=warmups, repetitions=repetitions)


@torch.inference_mode()
def benchmark_partial(model, rows: list[dict], depths, cfg) -> dict:
    """True cut latency, weighted by occupied bins of the supplied train rows.

    cfg: warmups>=10 (default 10), repetitions>=100 (default 100),
    max_rows_per_bin<=8 (default 8), seed (default 42). Inputs are batch=1,
    unpadded, identical for every cut/full run; preparation is outside timing.
    Only explicit pipeline_check_only=True permits smaller positive counts;
    those runs are smoke checks and are marked full_protocol=False.
    The caller must supply train-only rows; explicit held-out splits fail.
    """
    _validate_model(model)
    selected = _depths(model, depths)
    options = _cfg(cfg)
    warmups = _positive_int(options.get("warmups", 10), "warmups")
    repetitions = _positive_int(options.get("repetitions", 100), "repetitions")
    cap = _positive_int(options.get("max_rows_per_bin", 8), "max_rows_per_bin")
    pipeline_check_only = options.get("pipeline_check_only", False)
    if not isinstance(pipeline_check_only, bool):
        raise ValueError("pipeline_check_only must be an explicit boolean")
    if cap > 8 or (not pipeline_check_only and (warmups < 10 or repetitions < 100)):
        raise ValueError("full benchmark requires warmups>=10, repetitions>=100, max_rows_per_bin<=8; "
                         "smaller counts require pipeline_check_only=True")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark requires nonempty train-only rows")
    grouped, seen = {}, set()
    for row in rows:
        if not isinstance(row, Mapping) or not all(k in row for k in ("uid", "prefix", "candidate")):
            raise ValueError("benchmark row requires uid/prefix/candidate")
        if not isinstance(row["uid"], str) or not row["uid"] or row["uid"] in seen:
            raise ValueError("benchmark requires unique nonempty string uid values")
        seen.add(row["uid"])
        if row.get("split", "train") != "train":
            raise ValueError("benchmark rows must be train-only")
        # Validate before starting any measurements, then discard preparation.
        _tokens(model, row["prefix"], row["candidate"])
        seq_len = len(row["prefix"]) + len(row["candidate"])
        upper = next((b for b in LENGTH_BINS if seq_len <= b), None)
        if upper is None:
            raise ValueError("benchmark sequence length exceeds the 8192 bin")
        grouped.setdefault(upper, []).append(row)
    devices = _cuda_devices(model)
    parameter_devices = {p.device for p in model.parameters()}
    buffer_devices = {b.device for b in model.buffers()}
    all_devices = parameter_devices | buffer_devices
    no_cpu_offload = bool(devices) and all(d.type == "cuda" for d in all_devices)
    mapping = getattr(model, "hf_device_map", None)
    if devices and (not no_cpu_offload or (mapping and any(str(v) in ("cpu", "disk")
                                                         for v in mapping.values()))):
        raise RuntimeError("GPU verifier benchmark forbids CPU or disk offload")
    device_audit = dict(parameter_devices=sorted(str(d) for d in parameter_devices),
                        buffer_devices=sorted(str(d) for d in buffer_devices),
                        gpu_model_no_cpu_offload_validated=no_cpu_offload)
    bins = {}
    seed = options.get("seed", 42)
    for upper, population in sorted(grouped.items()):
        sampled = sorted(population, key=lambda row: (
            hashlib.sha256(f"{seed}:{row['uid']}".encode()).hexdigest(), row["uid"]))[:cap]
        prepared = [(row, _tokens(model, row["prefix"], row["candidate"])) for row in sampled]
        full = _measure(model, prepared, None, warmups, repetitions, devices)
        cuts = {str(depth): _measure(model, prepared, depth, warmups, repetitions, devices)
                for depth in selected}
        for cut in cuts.values():
            cut["latency_fraction"] = cut["mean_ms"] / full["mean_ms"]
        population_lengths = Counter(len(r["prefix"]) + len(r["candidate"]) for r in population)
        timed_lengths = Counter(len(sampled[i % len(sampled)]["prefix"]) +
                                len(sampled[i % len(sampled)]["candidate"]) for i in range(repetitions))
        population_exact = len(sampled) == len(population) and all(
            timed_lengths[length] * len(population) == count * repetitions
            for length, count in population_lengths.items())
        bins[str(upper)] = dict(population_rows=len(population), weight=len(population) / len(rows),
                                sampled_uids=[r["uid"] for r in sampled],
                                sequence_lengths=[len(r["prefix"]) + len(r["candidate"]) for r in sampled],
                                population_sequence_length_counts=dict(sorted(population_lengths.items())),
                                timed_sequence_length_counts=dict(sorted(timed_lengths.items())),
                                matched_sequence_distribution=True,
                                within_bin_population_distribution_exact=population_exact,
                                full=full, cuts=cuts)
        del prepared
    full_mean = sum(b["weight"] * b["full"]["mean_ms"] for b in bins.values())
    by_depth = {}
    for depth in selected:
        mean = sum(b["weight"] * b["cuts"][str(depth)]["mean_ms"] for b in bins.values())
        by_depth[str(depth)] = dict(weighted_mean_ms=mean, latency_fraction=mean / full_mean)
    if mapping is None:
        mapping = {name: str(next(module.parameters()).device)
                   for name, module in model.named_modules() if any(True for _ in module.parameters(recurse=False))}
    return dict(bins=bins, by_depth=by_depth, full_weighted_mean_ms=full_mean,
                population_rows=len(rows), length_bins=list(LENGTH_BINS), batch_size=1,
                seed=seed, warmups=warmups, repetitions=repetitions,
                pipeline_check_only=pipeline_check_only, full_protocol=not pipeline_check_only,
                matched_sequence_distribution=True,
                within_bin_population_distribution_exact=all(
                    b["within_bin_population_distribution_exact"] for b in bins.values()),
                device_map={k: str(v) for k, v in mapping.items()},
                device_audit=device_audit,
                synchronized_cuda_devices=[str(d) for d in devices],
                model_identity=getattr(model, "native_verifier_identity", None),
                attention_implementation=model.config._attn_implementation,
                dtype=str(model.get_input_embeddings().weight.dtype),
                notes=["Supplied rows are train-only; unlabeled split provenance is the caller's responsibility.",
                       "Weights use all supplied rows per bin; at most eight rows per bin are sampled deterministically.",
                       "Every cut and full run uses the same prepared inputs and repetition schedule.",
                       "Within-bin population distribution is approximate unless all rows are sampled with exact length weights.",
                       "pipeline_check_only runs are smoke checks and never count as the full latency protocol.",
                       "Cuts stop in a forward hook via an exception before any later block or LM head.",
                       "Final-depth cut includes final norm; full baseline includes L+1 LM-head logits.",
                       "Input creation, hook setup, and CPU hidden transfers are outside timing; no hidden is copied.",
                       "All participating CUDA devices are synchronized around each timed batch=1 forward.",
                       "Latency is model-forward-only; no end-to-end speedup claim."])
