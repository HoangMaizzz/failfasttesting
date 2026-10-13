"""Frozen native R pairs plus shadow Qwen labels, independent of teacher gains."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from agreement_profile_data import (SCHEMA, Record, atomic_json, identity,
    load_corpus, split_questions, validate_edge, validate_record, write_question)


def features_from_observation(observation, token_table):
    s = observation.scalars.detach().cpu().float().numpy()
    n = observation.length
    context = observation.context.detach().cpu().float().numpy()
    history = (observation.history.detach().cpu().float().numpy()
               if observation.history is not None else np.zeros((n, 4), np.float32))
    metadata = np.concatenate([s[:, :7], s[:, 9:16],
                               np.broadcast_to(context, (n, len(context))),
                               history[:, 1:2]], 1)
    gaps = observation.gaps.detach().cpu().float().numpy()
    weights = np.exp(gaps - gaps.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    entropy = -(weights * np.log(weights.clip(1e-12))).sum(1, keepdims=True)
    conditional_margin = (weights[:, :1] - weights[:, 1:2])
    topk = np.concatenate([gaps.clip(-40, 0) / 10, entropy, conditional_margin], 1) * s[:, 4:5]
    # Confidence availability is distinct from top-K availability, especially
    # when the proposal spans a physical boundary. Do not erase valid confidence.
    confidence = np.concatenate([s[:, 7:8] * s[:, 8:9], s[:, 8:9],
                                 history[:, :1] * s[:, 8:9]], 1)
    distribution = np.concatenate([topk, confidence], 1)
    # Top-K entropy/margin are CONDITIONAL summaries, not full-softmax entropy.
    hidden = observation.hidden.detach().cpu().float().flatten(1).numpy()
    hidden *= s[:, 3:4]
    content = F.embedding(observation.ids[:, 1].to(token_table.device),
                          token_table).detach().cpu().float().numpy()
    return dict(metadata=metadata.astype(np.float32),
                distribution=distribution.astype(np.float32), content=content,
                hidden=np.concatenate([hidden, history[:, 2:4]], 1))


def make_record(state, teacher, result, table, atol=0.01):
    o = state.observation
    candidate = o.ids[:, 1].cpu().numpy().astype(np.int64)
    argmax = np.asarray(teacher["argmax_ids"], dtype=np.int64)
    if len(argmax) != o.length:
        raise ValueError("Teacher argmax/candidate alignment mismatch")
    matches = candidate == argmax
    uid = identity(o.question)[:12] + "_" + o.uid
    meta = dict(round=o.round_id, context_hash=identity(state.prefix),
                context_token_ids=list(state.prefix),
                native_forward_index=int(state.snapshot["unmask_forward_index"]),
                refinement_index=state.snapshot_index,
                segment_start=len(state.prior),
                hidden_stage="same_forward_input_pre_commit_or_cached",
                mask_stage="post_native_commit_pre_stop_fill",
                argmax_ids=argmax.tolist(),
                candidate_logit=teacher["candidate_logit"],
                rival_logit=teacher["rival_logit"],
                label_source="shadow_full_context_greedy_same_verifier",
                emitted=list(result[2]),
                verifier_forward_ms=float(result[3]),
                native_terminal_reason=state.terminal_reason)
    record = Record(uid, o.question, state.study_split, meta,
                    features_from_observation(o, table),
                    np.asarray(teacher["margin"], dtype=np.float32), matches,
                    candidate, o.ids[:, 0].cpu().numpy().astype(np.int64), int(result[0]))
    validate_record(record)
    return record


def collect_question(question, split, tokenizer, environment, verifier, table, args):
    """Finite metadata-driven capture; labels never choose R/E or retained states."""
    from native_elysia_graph import NativeEosWithoutSnapshot
    records, edges, exclusions = [], [], []
    label_cache = {}
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question["prompt"]}],
                                           tokenize=True, add_generation_prompt=True)
    prefix = list(prompt)
    rng = random.Random(args.seed + int(identity(question["question_id"])[:8], 16))
    termination = "round_sampling_budget"
    for round_index in range(args.max_rounds_per_question):
        if len(prefix) + args.max_proposal_tokens > args.max_context_tokens:
            termination = "context_sampling_guard"
            break
        try:
            state = environment.start(question["question_id"], round_index,
                                      prefix, args.max_proposal_tokens + 1)
        except NativeEosWithoutSnapshot:
            exclusions.append("EOS_before_observable_boundary")
            termination = "native_eos_without_snapshot"
            break
        current_record = None
        for segment in range(args.segments_per_round):
            # At least one R attempt per segment; subsequent count is seeded and
            # independent of Qwen margins/K. Illegal/exhausted R is not a self-loop.
            wanted_r = rng.randint(1, max(1, args.max_refinement_steps))
            for extra_r in range(wanted_r + 1):
                if len(records) >= args.states_per_question:
                    termination = "state_sampling_budget"
                    break
                state.study_split = split
                before_native = state.observation.ids.clone()
                key = identity([state.prefix, state.observation.ids[:, 1].tolist(),
                                args.max_proposal_tokens + 1])
                reused = key in label_cache
                if reused:
                    result, teacher = copy.deepcopy(label_cache[key])
                else:
                    result = verifier.score(state.prefix,
                                            state.observation.ids[:, 1].tolist(),
                                            args.max_proposal_tokens + 1)
                    teacher = copy.deepcopy(verifier.last_teacher)
                    label_cache[key] = copy.deepcopy((result, teacher))
                if not torch.equal(before_native, state.observation.ids):
                    raise RuntimeError("Shadow verifier changed live state")
                record = make_record(state, teacher, result, table)
                record.meta["label_reused_same_context_candidate"] = reused
                if reused:
                    record.meta["verifier_forward_ms"] = None
                if current_record is not None:
                    validate_edge(current_record, record, args.causal_atol)
                    edges.append(dict(source=current_record.uid, child=record.uid,
                                      action="R", question=record.question, split=split))
                records.append(record)
                current_record = record
                if args.parity_repeat and len(records) == 1:
                    repeat = verifier.score(state.prefix, record.candidate.tolist(),
                                            args.max_proposal_tokens + 1)
                    repeat_teacher = verifier.last_teacher
                    if (repeat[0] != result[0]
                            or repeat_teacher["argmax_ids"] != teacher["argmax_ids"]
                            or not np.allclose(repeat_teacher["margin"], record.margin,
                                               atol=args.causal_atol, rtol=0)):
                        raise RuntimeError("Repeated production-verifier parity failed")
                if (len(records) >= args.states_per_question or extra_r == wanted_r
                        or "R" not in environment.actions(state)):
                    break
                child = environment.step(state, "R", args.max_proposal_tokens + 1)
                if child is None:
                    exclusions.append("R_unavailable_no_child")
                    break
                state = child
            if len(records) >= args.states_per_question:
                break
            if segment + 1 >= args.segments_per_round or "E" not in environment.actions(state):
                break
            child = environment.step(state, "E", args.max_proposal_tokens + 1)
            if child is None:
                exclusions.append("E_unavailable_no_child")
                break
            state = child
            current_record = None  # E is coverage, not a one-step R study edge.
        if not records or current_record is None:
            termination = "no_labelled_boundary"
            break
        # Follow the production emitted stream only at this scheduled round end.
        # Shadow labels at intermediate states have not changed the real prefix.
        emitted = current_record.meta["emitted"]
        prefix += emitted
        if tokenizer.eos_token_id in emitted:
            termination = "verified_eos"
            break
        if len(records) >= args.states_per_question:
            break
    ledger = dict(termination=termination, states=len(records), R_edges=len(edges),
                  unique_label_candidates=len(label_cache),
                  exclusions=exclusions, generation_to_eos_required=False)
    return records, edges, ledger


def collection_config(args):
    return {key: getattr(args, key) for key in (
        "dataset", "num_questions", "validation_questions", "test_questions", "seed",
        "max_rounds_per_question", "segments_per_round", "states_per_question",
        "max_context_tokens", "max_proposal_tokens", "extend_size", "max_refinement_steps",
        "physical_block_size", "small_block_size", "drafter_threshold", "hidden_layers",
        "raw_top_k", "target_model_name", "causal_atol", "parity_repeat")}


def weight_fingerprint(folder):
    """Bind resume to actual local weights, even when HF commit metadata is absent."""
    folder = Path(folder)
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("No local drafter safetensors")
    files += [folder / name for name in ("config.json", "configuration.py", "tokenizer.json",
                                         "tokenizer_config.json") if (folder / name).is_file()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
    return digest.hexdigest()


def collect_real(args, corpus, on_checkpoint=lambda: None):
    from pretrain_acceptance_world_model import load_questions
    from sparse_extend_world_model_collector import _load_models
    from structured_sparse_collector import FullContextVerifier
    from native_elysia_graph import NativeElysiaRunner
    from world_model_environment import NativeTrainingEnvironment, MASK_ID
    from transformers import AutoTokenizer
    corpus = Path(corpus)
    manifest_path = corpus / "manifest.json"
    config = collection_config(args)
    native_hash = identity((Path(__file__).parent / "Fast_dLLM_v2_1_5B/modeling.py").read_text(encoding="utf-8"))
    capture_hash = identity([(Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in ("collect_agreement_profiles.py", "world_model_environment.py",
                     "native_elysia_graph.py", "structured_sparse_collector.py")])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["collection_config"] != config
                or manifest["native_source_hash"] != native_hash
                or manifest.get("capture_source_hash") != capture_hash):
            raise ValueError("Resume capture/config mismatch; train-only mode can reuse old corpus")
        questions, split = manifest["questions"], manifest["split"]
    else:
        questions = load_questions(args)
        split = split_questions(questions, args.validation_questions, args.test_questions)
        manifest = dict(schema=SCHEMA, collection_config=config, questions=questions,
                        split=split, native_source_hash=native_hash,
                        capture_source_hash=capture_hash,
                        source_revision=args.source_revision,
                        targets="teacher_forced_candidate_margin_and_exact_argmax",
                        distribution_entropy="conditional_topk_not_full_vocabulary",
                        current_boundary="post_native_commit_pre_stop_materialization",
                        collection_policy="metadata_seeded_R_count_E_coverage_no_teacher_pruning")
        atomic_json(manifest_path, manifest)
    _, _, completed = load_corpus(corpus)
    if len(completed) == len(questions):
        return manifest
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Real collection requires Kaggle T4 x2; use --mode train for cached corpus")
    args.target_model_revision = manifest.get("runtime", {}).get("target_revision")
    tokenizer, target, drafter = _load_models(args)
    target.eval().requires_grad_(False)
    drafter.eval().requires_grad_(False)
    expected_target = {f"cuda:{args.target_device}", f"cuda:{args.drafter_device}"}
    actual_target = {str(p.device) for p in target.parameters()}
    if actual_target != expected_target:
        raise RuntimeError(f"Expected Qwen FP16 sharded on both GPUs, got {actual_target}; CPU/disk offload is forbidden")
    if {str(p.device) for p in drafter.parameters()} != {f"cuda:{args.drafter_device}"}:
        raise RuntimeError("Drafter must be entirely on GPU 1 in the T4 x2 layout")
    if target.dtype != torch.float16 or drafter.dtype != torch.float16:
        raise RuntimeError("Expected FP16 models; quantization/precision changes require a new protocol")
    draft_tokenizer = AutoTokenizer.from_pretrained(args.dllm_dir, local_files_only=True,
                                                   trust_remote_code=True)
    a = {k: v for k, v in tokenizer.get_vocab().items() if v != MASK_ID}
    b = {k: v for k, v in draft_tokenizer.get_vocab().items() if v != MASK_ID}
    if a != b or tokenizer.eos_token_id != draft_tokenizer.eos_token_id:
        raise RuntimeError("Drafter/verifier tokenizer mapping mismatch")
    runtime = dict(target_revision=getattr(target.config, "_commit_hash", None),
                   drafter_revision=getattr(drafter.config, "_commit_hash", None),
                   drafter_weights_sha256=weight_fingerprint(args.dllm_dir),
                   target_dtype=str(target.dtype), drafter_dtype=str(drafter.dtype),
                   tokenizer_hash=identity(a), chat_template_hash=identity(tokenizer.chat_template),
                   torch=str(torch.__version__), target_device_map=getattr(target, "hf_device_map", {}))
    memory = []
    for gpu in range(2):
        free, total = torch.cuda.mem_get_info(gpu)
        memory.append(dict(gpu=gpu, name=torch.cuda.get_device_name(gpu),
                           total_GiB=total / 2**30, free_after_models_GiB=free / 2**30,
                           allocated_GiB=torch.cuda.memory_allocated(gpu) / 2**30,
                           reserved_GiB=torch.cuda.memory_reserved(gpu) / 2**30))
        if free < .5 * 2**30:
            raise RuntimeError(f"GPU {gpu} has <0.5 GiB headroom before collection; refusing likely OOM")
    print("[T4 x2 memory]", json.dumps(memory), flush=True)
    if "runtime" in manifest and manifest["runtime"] != runtime:
        raise RuntimeError("Resume model/tokenizer/runtime changed; refusing mixed labels")
    manifest["runtime"] = runtime
    manifest["memory_after_loading"] = memory
    manifest["placement"] = dict(drafter="FP16 entirely GPU1", verifier="FP16 sharded GPU0+GPU1",
                                  verifier_weight_placement_cap_GiB=args.target_gpu_memory_gib,
                                  CPU_offload=False, collection_sequential=True)
    atomic_json(manifest_path, manifest)
    verifier = FullContextVerifier(target, tokenizer, args)
    runner = NativeElysiaRunner(drafter, tokenizer, args)
    environment = NativeTrainingEnvironment(runner, verifier, tokenizer.eos_token_id,
        drafter.config.hidden_size, args, lambda *unused: None)
    table = drafter.get_input_embeddings().weight.detach()
    try:
        for question in questions:
            qid = question["question_id"]
            if qid in completed:
                continue
            print(f"[collect] {qid} split={split[qid]}", flush=True)
            started = time.perf_counter()
            rows, edges, ledger = collect_question(question, split[qid], tokenizer,
                                                   environment, verifier, table, args)
            ledger["capture_wall_seconds_including_replay_and_shadow"] = time.perf_counter() - started
            write_question(corpus / "questions", qid, rows, edges, ledger)
            completed.append(qid)
            atomic_json(corpus / "progress.json", dict(completed=completed,
                        total=len(questions), status="complete" if len(completed) == len(questions) else "partial"))
            on_checkpoint()
            del rows
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del table, environment, runner, verifier, target, drafter
        gc.collect()
        torch.cuda.empty_cache()
    return manifest
