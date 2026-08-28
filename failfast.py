import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import copy
import csv
import hashlib
import json
import math
import re
import time
import torch
import pickle
import pprint
import logging
import argparse
import transformers
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


def _synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _record_transfer(args, elapsed):
    args.device_transfer_time_total = (
        getattr(args, "device_transfer_time_total", 0.0) + elapsed
    )


def timed_device_copy(tensor, device, args):
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    _synchronize_device(tensor.device)
    started = time.perf_counter()
    copied = tensor.to(device)
    _synchronize_device(device)
    _record_transfer(args, time.perf_counter() - started)
    return copied


def timed_tensor_to_list(tensor, args):
    if args.target_device == args.drafter_device:
        return tensor.tolist()
    _synchronize_device(tensor.device)
    started = time.perf_counter()
    values = tensor.tolist()
    _record_transfer(args, time.perf_counter() - started)
    return values


def timed_token_tensor(values, device, args):
    if args.target_device == args.drafter_device:
        return torch.tensor([values], device=device, dtype=torch.long)
    _synchronize_device(device)
    started = time.perf_counter()
    tensor = torch.tensor([values], device=device, dtype=torch.long)
    _synchronize_device(device)
    _record_transfer(args, time.perf_counter() - started)
    return tensor

from adaptive_td import (
    FEATURE_NAMES,
    FEATURE_SCHEMAS,
    AdaptiveTDConfig,
    OnlineTDRefinementController,
)
from distributional_controller import (
    DistributionalControllerConfig,
    DistributionalTimeTokenController,
)
from bucket_renewal import position_bucket
from causal_oracle_utils import prepare_causal_oracle_snapshots
from future_oracle import (
    load_future_cost_profile,
    select_greedy_future_adjusted_candidate,
    stats_for_problem,
)
from global_oracle import (
    CONTINUE,
    STOP,
    OracleBranchRequired,
    ScriptedOracleRefinementController,
    analyze_stop_depth_curves,
    solve_canonical_oracle_graph,
    summarize_policy_path,
)
from truncated_global_oracle import (
    VerifierLatencyProfile,
    estimate_cache_bytes,
    greedy_lcp_verification,
    solve_truncated_horizon,
)
from strict_greedy_oracle import (
    GreedyBranch,
    build_oracle_state_key,
    choose_strict_greedy_action,
    one_action_rollout_scripts,
    format_outer_path,
    load_verifier_profile,
    predict_verifier_latency_ms,
)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

transformers.logging.set_verbosity_error()

TEMPERATURE = 0.6

sys.path.insert(1, os.path.dirname(os.getcwd()))
from plotting import (
    visualize_acc_rate_over_time,
)
from utils import (
    Colors, is_interactive,
    populate_dataset, get_first_user_msg, 
    format_problem_and_options, format_drafter_name, get_proposal_str, get_output_tokens,
    get_output_dir,
    print_sd_trajectory,
)

def get_target_token_ids(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    num_input_tokens = model_inputs.input_ids.shape[1]
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, # 🚀 Đã đổi sang Sampling
        temperature=TEMPERATURE,
        top_k=0,        # 🚀 Tắt ép top_k mặc định của HF
        top_p=1.0,      # 🚀 Tắt ép top_p mặc định của HF
        pad_token_id=model.config.eos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs

def get_next_n_tokens_ar(model, orig_model_inputs, token_ids_so_far, n, temperature=TEMPERATURE, do_sample=False):
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }
    generation_kwargs = {
        "max_new_tokens": n,
        "do_sample": do_sample,
        "output_scores": True,
        "return_dict_in_generate": True,
        "pad_token_id": model.config.eos_token_id,
        "eos_token_id": model.config.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
    generate_output = model.generate(**new_model_inputs, **generation_kwargs)
    generated_ids = generate_output.sequences[0][len(new_model_inputs["input_ids"][0]):].tolist()
    
    # 🚀 LẤY BẢNG XÁC SUẤT CỦA AR DRAFTER ĐỂ LÀM RESIDUAL SAMPLING
    drafter_probs = []
    for scores in generate_output.scores:
        # scores đã được chia cho temperature từ LogitsWarper của Generate, nên chỉ cần Softmax
        probs = torch.softmax(scores[0], dim=-1)
        drafter_probs.append(probs)
        
    return generated_ids, drafter_probs

def get_next_tokens_ar(
    model,
    orig_model_inputs,
    token_ids_so_far,
    n,
    lowconf_threshold,
    max_spec_len,
    incr_len,
    temperature=TEMPERATURE,
    do_sample=False,
):
    if incr_len is None or incr_len <= 0:
        raise ValueError(f"incr_len must be a positive int, got {incr_len}")
    if max_spec_len is not None and max_spec_len <= 0:
        raise ValueError(f"max_spec_len must be a positive int or None, got {max_spec_len}")
    if lowconf_threshold is None:
        raise ValueError("lowconf_threshold must not be None for get_next_tokens_ar")

    cap = n if max_spec_len is None else max_spec_len
    if cap <= 0:
        return [], []

    device = orig_model_inputs["input_ids"].device
    drafted = []
    confidences = []
    drafter_probs_list = []

    current_tokens = torch.tensor(token_ids_so_far, device=device, dtype=torch.long).unsqueeze(0)
    current_mask = torch.ones_like(current_tokens, dtype=torch.long)
    current_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], current_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], current_mask], dim=1)
    }

    with torch.no_grad():
        while len(drafted) < cap:
            chunk_size = min(incr_len, cap - len(drafted))
            
            generation_kwargs = {
                "max_new_tokens": chunk_size,
                "do_sample": do_sample,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": model.config.eos_token_id,
                "eos_token_id": model.config.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
            generate_output = model.generate(**current_inputs, **generation_kwargs)
            
            generated_ids = generate_output.sequences[0][len(current_inputs["input_ids"][0]):].tolist()
            scores = generate_output.scores
            
            found_lowconf = False
            for i, (token_id, score_logits) in enumerate(zip(generated_ids, scores)):
                probs = torch.softmax(score_logits[0], dim=-1)
                drafter_probs_list.append(probs)
                
                conf = probs[token_id].item()
                drafted.append(token_id)
                confidences.append(conf)
                
                if conf < lowconf_threshold:
                    found_lowconf = True
            
            if found_lowconf:
                return drafted, confidences, drafter_probs_list
            
            if len(drafted) < cap:
                new_tokens = torch.tensor(generated_ids, device=device, dtype=torch.long).unsqueeze(0)
                new_mask = torch.ones_like(new_tokens, dtype=torch.long)
                current_inputs = {
                    'input_ids': torch.cat([current_inputs['input_ids'], new_tokens], dim=1),
                    'attention_mask': torch.cat([current_inputs['attention_mask'], new_mask], dim=1)
                }

    return drafted, confidences, drafter_probs_list

def get_next_n_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
    num_tokens_in_prompt = orig_model_inputs['input_ids'].shape[1]
    prompt_ids = timed_device_copy(orig_model_inputs['input_ids'], dllm.device, args)
    prompt_mask = timed_device_copy(orig_model_inputs['attention_mask'], dllm.device, args)
    new_tokens = torch.tensor(token_ids_so_far, device=dllm.device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([prompt_ids, new_tokens], dim=1),
        'attention_mask': torch.cat([prompt_mask, new_mask], dim=1)
    }
    return_frontier_stats = False
    frontier_stats = None

    if args.disable_reusing_drafter_kvs:
        draft_result = dllm.generate_draft_tokens(
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            block_size=args.block_size,
            threshold=threshold,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
            return_frontier_stats=return_frontier_stats,
        )
        if return_frontier_stats:
            generated_ids, num_forward_passes, forward_pass_latencies, frontier_stats = draft_result
        else:
            generated_ids, num_forward_passes, forward_pass_latencies = draft_result
    else:
        draft_result = dllm.generate_draft_tokens(
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            block_size=args.block_size,
            threshold=threshold,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
            return_frontier_stats=return_frontier_stats,
        )
        if return_frontier_stats:
            generated_ids, prefill_output, num_forward_passes, forward_pass_latencies, frontier_stats = draft_result
        else:
            generated_ids, prefill_output, num_forward_passes, forward_pass_latencies = draft_result
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = timed_tensor_to_list(generated_ids, args)[:spec_len]
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, num_forward_passes, forward_pass_latencies

def get_next_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None,
                        lowconf_threshold=None,
                        max_spec_len=None,
        incr_len=None,
        last_round_rejected=None,
    ):
    controller_enabled = frontier_stop_enabled(args)
    if controller_enabled:
        ensure_frontier_runtime_state(args)
    num_tokens_in_prompt = orig_model_inputs['input_ids'].shape[1]
    prompt_ids = timed_device_copy(orig_model_inputs['input_ids'], dllm.device, args)
    prompt_mask = timed_device_copy(orig_model_inputs['attention_mask'], dllm.device, args)
    new_tokens = torch.tensor(token_ids_so_far, device=dllm.device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([prompt_ids, new_tokens], dim=1),
        'attention_mask': torch.cat([prompt_mask, new_mask], dim=1)
    }
    if controller_enabled:
        args.bucket_current_context_len = int(new_model_inputs['input_ids'].shape[1])
    return_frontier_stats = (
        controller_enabled
        or getattr(args, "collect_draft_diagnostics", False)
        or getattr(args, "collect_bucket_oracle", False)
    )
    frontier_stats = None

    if args.disable_reusing_drafter_kvs:
        draft_result = dllm.generate_draft_tokens_arbitrary_length(
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            block_size=args.block_size,
            threshold=threshold,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
            return_frontier_stats=return_frontier_stats,
        )
        if return_frontier_stats:
            generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies, frontier_stats = draft_result
        else:
            generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies = draft_result
    else:
        draft_result = dllm.generate_draft_tokens_arbitrary_length(
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
            return_frontier_stats=return_frontier_stats,
        )
        if return_frontier_stats:
            generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies, frontier_stats = draft_result
        else:
            generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = draft_result
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = timed_tensor_to_list(generated_ids, args)[:actual_spec_len]
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    args.last_frontier_stats = frontier_stats
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies

def construct_drafter_configs(args):
    drafter_configs = []
    if args.mode == "verifier_ar":
        drafter_configs.append(("verifier_ar", None, "none", None, None, None))
        args.drafter_configs = drafter_configs
        return
    if args.run_ar:
        drafter_configs.extend([("ar", None, "sf", None, None, None)])
        if args.ar_dynamic:
            drafter_configs.extend([
                ("ar", None, "df", lowconf_threshold, max_spec_len, incr_len)
                for lowconf_threshold in args.sweep_lowconf_threshold
                for max_spec_len in args.sweep_max_spec_len
                for incr_len in args.sweep_incr_len
            ])
    if args.run_dllm_sf:
        drafter_configs.extend([("dllm", thr, "sf", None, None, None) for thr in args.drafter_thresholds])
    if not args.baseline_sweep:
        drafter_configs.extend([("dllm", thr, "df", lowconf_threshold, max_spec_len, incr_len) 
                                for thr in args.drafter_thresholds
                                for lowconf_threshold in args.sweep_lowconf_threshold
                                for max_spec_len in args.sweep_max_spec_len
                                for incr_len in args.sweep_incr_len
                                ])
    args.drafter_configs = drafter_configs

parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gsm8k", "gpqa", "humaneval", "mt_bench"], default="math")
parser.add_argument("--output_dir", type=str, default="/data2/USERNAME/failfast")
parser.add_argument("--mode", type=str, choices=["verifier_ar", "ar_ar", "dllm_ar"], default="dllm_ar")
parser.add_argument("--benchmark_modes", type=str, nargs="+", choices=["verifier_ar", "ar_ar", "dllm_ar"], default=["verifier_ar", "ar_ar", "dllm_ar"])
parser.add_argument("--dllm_variant", type=str, choices=["failfast", "fixed"], default="failfast")
parser.add_argument("--decoding_strategy", type=str, choices=["greedy", "sampling"], default="greedy")
parser.add_argument("--target_model_name", type=str, default=None)
parser.add_argument("--verifier_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--drafter_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--dllm_dir", type=str, default=None)
parser.add_argument("--target_device", type=int, default=0)
parser.add_argument("--drafter_device", type=int, default=0)
parser.add_argument("--num_questions", type=int, default=1)
parser.add_argument("--problem_ids", type=int, nargs="+")
parser.add_argument("--warmup_questions", type=int, default=0)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--small_block_size", type=int, default=8)
parser.add_argument("--spec_len", type=int, default=10)
parser.add_argument("--drafter_thresholds", type=float, nargs="+", default=[0.05])
parser.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument("--sweep_lowconf_threshold", type=float, nargs="+", default=[0.45])
parser.add_argument("--sweep_max_spec_len", type=int, nargs="+", default=[60])
parser.add_argument("--sweep_incr_len", type=int, nargs="+", default=[10])
parser.add_argument(
    "--frontier_stop_mode",
    type=str,
    default="disabled",
    choices=["disabled", "bucket_renewal"],
)
parser.add_argument("--bucket_renewal_min_steps", type=int, default=1)
parser.add_argument("--bucket_renewal_hysteresis", type=float, default=0.0)
parser.add_argument("--bucket_prior_strength", type=float, default=8.0)
parser.add_argument("--bucket_min_observations", type=int, default=8)
parser.add_argument("--bucket_latency_ema_alpha", type=float, default=0.2)
parser.add_argument("--collect_draft_diagnostics", action="store_true")
parser.add_argument("--collect_bucket_oracle", action="store_true")
parser.add_argument("--causal_oracle", action="store_true")
parser.add_argument("--causal_oracle_future_cost_profile", type=str)
parser.add_argument("--global_oracle_graph", action="store_true")
parser.add_argument("--global_oracle_max_states", type=int, default=0)
parser.add_argument("--global_oracle_log_interval", type=int, default=25)
parser.add_argument("--global_oracle_epsilon_cost_ms", type=float, default=1.0)
parser.add_argument("--truncated_global_horizon", type=int, default=0)
parser.add_argument("--truncated_lcp_validation_candidates", type=int, default=16)
parser.add_argument("--strict_greedy_local_oracle", action="store_true")
parser.add_argument("--strict_greedy_verifier_profile", type=str)
parser.add_argument("--strict_greedy_epsilon_ms", type=float, default=1.0)
parser.add_argument("--strict_greedy_replay_policy", type=str)
parser.add_argument(
    "--strict_greedy_capacity_collector",
    action="store_true",
    help=(
        "Probe STOP/CONTINUE branches at FailFast decision states while "
        "executing the unmodified FailFast CONTINUE support policy."
    ),
)
parser.add_argument("--log_verifier_calls", action="store_true")
parser.add_argument(
    "--audit_greedy_consistency",
    action="store_true",
    help=(
        "Diagnostic only: recompute every committed verifier token from its "
        "exact prefix and log batched-versus-prefix greedy consistency."
    ),
)
parser.add_argument(
    "--audit_greedy_problem_ids",
    type=int,
    nargs="+",
    help="Limit greedy consistency auditing to these dataset problem IDs.",
)
parser.add_argument("--bucket_oracle_force_continue", action="store_true")
parser.add_argument("--adaptive-td", action="store_true")
parser.add_argument(
    "--adaptive-feature-schema",
    choices=sorted(FEATURE_SCHEMAS),
    default="otrc_v1_td",
)
parser.add_argument(
    "--adaptive-controller",
    choices=["avg_td", "dist_time_token"],
    default="avg_td",
)
parser.add_argument("--adaptive-max-refinement-steps", type=int, default=16)
parser.add_argument("--adaptive-fixed-refinement-steps", type=int)
parser.add_argument("--adaptive-learning-rate", type=float, default=0.02)
parser.add_argument(
    "--adaptive-value-parameterization",
    choices=["independent_q", "shared_value_advantage"],
    default="independent_q",
)
parser.add_argument(
    "--adaptive-shared-value-learning-rate",
    type=float,
    default=0.015,
)
parser.add_argument(
    "--adaptive-shared-advantage-learning-rate",
    type=float,
    default=0.02,
)
parser.add_argument("--adaptive-mc-learning-rate", type=float, default=0.01)
parser.add_argument("--adaptive-mc-mix", type=float, default=0.5)
parser.add_argument(
    "--adaptive-update-mode",
    choices=["td", "factual_return", "mixed"],
    default="mixed",
)
parser.add_argument(
    "--adaptive-credit-assignment",
    choices=[
        "per_step_td",
        "verifier_boundary_factual",
        "verifier_boundary_factual_no_bootstrap",
    ],
    default="per_step_td",
)
parser.add_argument("--adaptive-rho-alpha", type=float, default=0.05)
parser.add_argument("--adaptive-rho-warmup-boundaries", type=int, default=0)
parser.add_argument(
    "--adaptive-policy-weight-ema-beta",
    type=float,
    default=0.0,
    help=(
        "EMA beta for policy decision weights. A value of 0 disables policy "
        "weight EMA."
    ),
)
parser.add_argument(
    "--adaptive-policy-weight-ema-mode",
    choices=["global_step", "action_step"],
    default="global_step",
    help=(
        "Policy-weight EMA clock. global_step advances BOTH STOP and CONTINUE "
        "EMA heads after every factual learner update; action_step reproduces "
        "the historical selected-action-only EMA."
    ),
)
parser.add_argument("--adaptive-risk-beta", type=float, default=1.0)
parser.add_argument("--adaptive-stop-probability-threshold", type=float, default=0.75)
parser.add_argument("--adaptive-uncertainty-prior", type=float, default=1.0)
parser.add_argument("--adaptive-epistemic-scale", type=float, default=0.1)
parser.add_argument("--adaptive-q-margin", type=float, default=0.0)
parser.add_argument("--adaptive-explore-epsilon", type=float, default=0.10)
parser.add_argument("--adaptive-explore-min", type=float, default=0.01)
parser.add_argument("--adaptive-explore-decay", type=float, default=0.998)
parser.add_argument("--adaptive-warmup-rounds", type=int, default=20)
parser.add_argument("--adaptive-early-stop-min-observations", type=int, default=32)
parser.add_argument(
    "--adaptive-policy-mode",
    choices=["legacy", "symmetric"],
    default="legacy",
)
parser.add_argument("--adaptive-min-action-probability", type=float, default=0.10)
parser.add_argument("--adaptive-max-importance-weight", type=float, default=5.0)
parser.add_argument("--adaptive-use-step-feature", action="store_true")
parser.add_argument(
    "--adaptive-disable-features",
    nargs="*",
    choices=sorted({
        name
        for names in FEATURE_SCHEMAS.values()
        for name in names
        if name != "bias"
    }),
    default=[],
)
parser.add_argument(
    "--adaptive-use-margin-feature",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--adaptive-use-stability-feature",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--adaptive-force-continue", action="store_true")
parser.add_argument("--adaptive-state-path", type=str)
parser.add_argument("--adaptive-freeze", action="store_true")
parser.add_argument("--adaptive-counterfactual-replay", action="store_true")
parser.add_argument("--adaptive-log-decisions", action="store_true")
parser.add_argument("--adaptive-profile-overhead", action="store_true")
parser.add_argument("--adaptive-factual-ema-alpha", type=float, default=0.2)
parser.add_argument("--adaptive-weight-snapshot-interval", type=int, default=100)
parser.add_argument(
    "--dist-decision-rule",
    choices=["expected_regret", "probability"],
    default="expected_regret",
)
parser.add_argument("--dist-stop-probability-threshold", type=float, default=0.55)
parser.add_argument("--dist-stop-regret-weight", type=float, default=1.0)
parser.add_argument("--dist-continue-regret-weight", type=float, default=1.0)
parser.add_argument("--dist-latency-ema-alpha", type=float, default=0.2)
parser.add_argument("--quiet_generation", action="store_true")
parser.add_argument("--disable_progress", action="store_true")
parser.add_argument("--skip_artifacts", action="store_true")
parser.add_argument("--skip_plots", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--run_ar', action='store_true')
parser.add_argument('--ar_dynamic', action='store_true')
parser.add_argument('--run_dllm_sf', action='store_true')
parser.add_argument('--baseline_sweep', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--reuse_drafts', action='store_true')
parser.add_argument('--disable_reusing_drafter_kvs', action='store_true')
parser.add_argument('--read_pickle', action='store_true')
args, _ = parser.parse_known_args()
if args.target_device < 0 or args.drafter_device < 0:
    raise ValueError("CUDA device indices must be non-negative")
if torch.cuda.is_available():
    required_devices = max(args.target_device, args.drafter_device) + 1
    if torch.cuda.device_count() < required_devices:
        raise ValueError(
            f"requested CUDA device index requires {required_devices} GPUs, "
            f"but only {torch.cuda.device_count()} are visible"
        )

if args.target_model_name is None:
    args.target_model_name = args.verifier_model_name
if args.audit_greedy_consistency and args.decoding_strategy != "greedy":
    raise ValueError("--audit_greedy_consistency requires greedy decoding")
if args.adaptive_td and args.frontier_stop_mode != "disabled":
    raise ValueError("--adaptive-td cannot be combined with --frontier_stop_mode")
if (
    args.adaptive_controller != "avg_td"
    and args.adaptive_feature_schema != "otrc_v1_td"
):
    raise ValueError("OTRC feature schemas require --adaptive-controller avg_td")
if args.adaptive_td and args.bucket_oracle_force_continue:
    raise ValueError(
        "--bucket_oracle_force_continue would invalidate adaptive TD trajectories"
    )
if args.adaptive_force_continue and args.adaptive_fixed_refinement_steps is not None:
    raise ValueError(
        "--adaptive-force-continue cannot be combined with fixed refinement depth"
    )
if (args.adaptive_state_path or args.adaptive_freeze or args.adaptive_counterfactual_replay) and not args.adaptive_td:
    raise ValueError("adaptive replay options require --adaptive-td")
if args.adaptive_counterfactual_replay and not args.collect_bucket_oracle:
    raise ValueError("--adaptive-counterfactual-replay requires --collect_bucket_oracle")
if args.adaptive_counterfactual_replay and not args.adaptive_freeze:
    raise ValueError("--adaptive-counterfactual-replay requires --adaptive-freeze")
if args.causal_oracle and not args.collect_bucket_oracle:
    raise ValueError("--causal_oracle requires --collect_bucket_oracle")
if args.causal_oracle and (args.adaptive_td or args.frontier_stop_mode != "disabled"):
    raise ValueError("--causal_oracle requires the unmodified FailFast drafting policy")
if args.causal_oracle_future_cost_profile and not args.causal_oracle:
    raise ValueError("--causal_oracle_future_cost_profile requires --causal_oracle")
if args.global_oracle_graph:
    if args.causal_oracle or args.adaptive_td or args.frontier_stop_mode != "disabled":
        raise ValueError(
            "--global_oracle_graph requires unmodified FailFast without another controller"
        )
    if args.decoding_strategy != "greedy":
        raise ValueError("--global_oracle_graph requires greedy decoding")
    if args.benchmark_modes != ["dllm_ar"]:
        raise ValueError("--global_oracle_graph requires --benchmark_modes dllm_ar")
    if not args.collect_bucket_oracle:
        raise ValueError("--global_oracle_graph requires --collect_bucket_oracle")
    if args.reuse_drafts:
        raise ValueError("--global_oracle_graph does not support --reuse_drafts")
    if not args.disable_reusing_drafter_kvs:
        raise ValueError(
            "--global_oracle_graph requires --disable_reusing_drafter_kvs "
            "so counterfactual states are replayable and path-independent"
        )
    if args.global_oracle_max_states < 0:
        raise ValueError("--global_oracle_max_states must be non-negative")
    if args.global_oracle_log_interval <= 0:
        raise ValueError("--global_oracle_log_interval must be positive")
    if args.global_oracle_epsilon_cost_ms < 0.0:
        raise ValueError("--global_oracle_epsilon_cost_ms must be non-negative")
    if args.warmup_questions:
        raise ValueError("--global_oracle_graph does not support warmup questions")
    if args.truncated_global_horizon < 0:
        raise ValueError("--truncated_global_horizon must be non-negative")
    if args.truncated_lcp_validation_candidates < 0:
        raise ValueError(
            "--truncated_lcp_validation_candidates must be non-negative"
        )
elif args.truncated_global_horizon:
    raise ValueError("--truncated_global_horizon requires --global_oracle_graph")
if args.strict_greedy_local_oracle:
    if args.causal_oracle or args.global_oracle_graph or args.adaptive_td:
        raise ValueError("strict greedy oracle cannot be combined with another oracle/controller")
    if args.frontier_stop_mode != "disabled":
        raise ValueError("strict greedy oracle requires unmodified FailFast")
    if args.decoding_strategy != "greedy":
        raise ValueError("strict greedy oracle requires greedy decoding")
    if args.benchmark_modes != ["dllm_ar"]:
        raise ValueError("strict greedy oracle requires --benchmark_modes dllm_ar")
    if not args.strict_greedy_verifier_profile:
        raise ValueError("strict greedy oracle requires --strict_greedy_verifier_profile")
    if args.strict_greedy_epsilon_ms < 0.0:
        raise ValueError("--strict_greedy_epsilon_ms must be non-negative")
elif args.strict_greedy_verifier_profile:
    raise ValueError("--strict_greedy_verifier_profile requires strict greedy oracle")
if args.strict_greedy_capacity_collector and not args.strict_greedy_local_oracle:
    raise ValueError(
        "--strict_greedy_capacity_collector requires --strict_greedy_local_oracle"
    )
if args.strict_greedy_capacity_collector and args.strict_greedy_replay_policy:
    raise ValueError("capacity collector cannot replay an oracle policy")
if args.strict_greedy_replay_policy and not args.strict_greedy_local_oracle:
    raise ValueError("--strict_greedy_replay_policy requires strict greedy oracle")
def build_adaptive_controller(args):
    if args.global_oracle_graph or args.strict_greedy_local_oracle:
        return ScriptedOracleRefinementController(
            max_refinement_steps=max(64, args.adaptive_max_refinement_steps),
            feature_schema=(
                "otrc_v2_2_compact_td"
                if args.strict_greedy_capacity_collector
                else "otrc_v1_td"
            ),
            factual_ema_alpha=args.adaptive_factual_ema_alpha,
        )
    if not args.adaptive_td:
        return None
    if args.adaptive_controller == "dist_time_token":
        return DistributionalTimeTokenController(
            DistributionalControllerConfig(
                learning_rate=args.adaptive_learning_rate,
                latency_alpha=args.dist_latency_ema_alpha,
                throughput_alpha=args.adaptive_rho_alpha,
                decision_rule=args.dist_decision_rule,
                stop_probability_threshold=args.dist_stop_probability_threshold,
                stop_regret_weight=args.dist_stop_regret_weight,
                continue_regret_weight=args.dist_continue_regret_weight,
                explore_epsilon=args.adaptive_explore_epsilon,
                explore_min=args.adaptive_explore_min,
                explore_decay=args.adaptive_explore_decay,
                warmup_rounds=args.adaptive_warmup_rounds,
                max_refinement_steps=args.adaptive_max_refinement_steps,
                fixed_refinement_steps=args.adaptive_fixed_refinement_steps,
                force_continue=args.adaptive_force_continue,
                profile_overhead=args.adaptive_profile_overhead,
                seed=args.seed,
            )
        )
    return OnlineTDRefinementController(
        AdaptiveTDConfig(
            feature_dim=len(FEATURE_SCHEMAS[args.adaptive_feature_schema]),
            feature_schema=args.adaptive_feature_schema,
            feature_version={
                "otrc_v1_td": 1,
                "otrc_v2_td": 2,
                "otrc_v2_1_td": 21,
                "otrc_v2_2_td": 22,
                "otrc_v2_2_compact_td": 226,
            }[args.adaptive_feature_schema],
            learning_rate=args.adaptive_learning_rate,
            value_parameterization=args.adaptive_value_parameterization,
            shared_value_learning_rate=(
                args.adaptive_shared_value_learning_rate
            ),
            shared_advantage_learning_rate=(
                args.adaptive_shared_advantage_learning_rate
            ),
            mc_learning_rate=args.adaptive_mc_learning_rate,
            mc_mix=args.adaptive_mc_mix,
            update_mode=args.adaptive_update_mode,
            rho_alpha=args.adaptive_rho_alpha,
            risk_beta=args.adaptive_risk_beta,
            stop_probability_threshold=args.adaptive_stop_probability_threshold,
            uncertainty_prior=args.adaptive_uncertainty_prior,
            epistemic_scale=args.adaptive_epistemic_scale,
            q_margin=args.adaptive_q_margin,
            explore_epsilon=args.adaptive_explore_epsilon,
            explore_min=args.adaptive_explore_min,
            explore_decay=args.adaptive_explore_decay,
            warmup_rounds=args.adaptive_warmup_rounds,
            early_stop_min_observations=args.adaptive_early_stop_min_observations,
            max_refinement_steps=args.adaptive_max_refinement_steps,
            fixed_refinement_steps=args.adaptive_fixed_refinement_steps,
            force_continue=args.adaptive_force_continue,
            profile_overhead=args.adaptive_profile_overhead,
            seed=args.seed,
            policy_mode=args.adaptive_policy_mode,
            min_action_probability=args.adaptive_min_action_probability,
            max_importance_weight=args.adaptive_max_importance_weight,
            full_stream_bootstrap=True,
            credit_assignment=args.adaptive_credit_assignment,
            disabled_features=tuple(args.adaptive_disable_features),
            factual_ema_alpha=args.adaptive_factual_ema_alpha,
            rho_warmup_boundaries=args.adaptive_rho_warmup_boundaries,
            policy_weight_ema_beta=args.adaptive_policy_weight_ema_beta,
            policy_weight_ema_mode=args.adaptive_policy_weight_ema_mode,
            weight_snapshot_interval=args.adaptive_weight_snapshot_interval,
        )
    )


args.adaptive_td_controller = build_adaptive_controller(args)
args.strict_greedy_profile = (
    load_verifier_profile(args.strict_greedy_verifier_profile)
    if args.strict_greedy_local_oracle
    else None
)
args.strict_greedy_decision_rows = []
args.verifier_call_rows = []
args.greedy_consistency_rows = []
args.output_token_rows = []
args.strict_greedy_replay_data = (
    json.loads(Path(args.strict_greedy_replay_policy).read_text(encoding="utf-8"))
    if args.strict_greedy_replay_policy
    else None
)
args.strict_greedy_selected_policy = {}
if args.adaptive_state_path:
    with open(args.adaptive_state_path, "r", encoding="utf-8") as handle:
        args.adaptive_td_controller.load_snapshot(json.load(handle))

def apply_mode_settings(args):
    if args.mode == "verifier_ar":
        args.run_ar = False
        args.run_dllm_sf = False
        args.baseline_sweep = True
    elif args.mode == "ar_ar":
        args.run_ar = True
        args.run_dllm_sf = False
        args.baseline_sweep = True
    elif args.mode == "dllm_ar":
        args.run_ar = False
        args.run_dllm_sf = False
        args.baseline_sweep = False
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

BENCHMARK_MODES = ("verifier_ar", "ar_ar", "dllm_ar")
BENCHMARK_CSV_COLUMNS = [
    "problem_id",
    "mode",
    "actual_total_time",
    "theo_total_time",
    "actual_draft_time",
    "theo_draft_time",
    "actual_verify_time",
    "theo_verify_time",
    "actual_draft_verify_ratio",
    "theo_draft_verify_ratio",
    "acceptance_rate_percent",
    "actual_speedup_vs_AR",
    "actual_e2e_speedup_vs_AR",
    "theo_speedup_vs_AR",
    "actual_e2e_time",
    "actual_e2e_ms_per_output_token",
    "output_tokens_per_ms",
    "actual_post_verify_time",
    "actual_algorithm_time",
    "actual_algorithm_ms_per_output_token",
    "actual_unattributed_core_time",
    "output_tokens",
    "accepted_tokens",
    "drafted_tokens",
    "num_speculation_rounds",
    "total_num_forward_passes",
    "bucket_stop_actions",
    "bucket_decision_steps",
    "bucket_predicted_gain_mean",
    "bucket_expected_output_mean",
    "bucket_stop_ms_per_output_mean",
    "bucket_continue_ms_per_output_mean",
    "bucket_fill_forward_passes",
    "bucket_denoising_forward_passes",
    "adaptive_decisions",
    "adaptive_stop_actions",
    "adaptive_exploration_actions",
    "adaptive_calibration_decisions",
    "adaptive_calibration_stop_actions",
    "adaptive_learned_stop_actions",
    "adaptive_early_stop_observations",
    "adaptive_behavior_stop_probability_mean",
    "adaptive_selected_action_probability_mean",
    "adaptive_importance_weight_mean",
    "adaptive_mean_refinement_step",
    "adaptive_controller_ms",
    "adaptive_rho_tokens_per_ms",
    "adaptive_stop_available_decisions",
    "adaptive_candidate_coverage_decisions",
    "adaptive_outer_verify_eligible_decisions",
    "adaptive_stop_then_extend_actions",
    "adaptive_stop_then_verify_actions",
    "adaptive_outer_action_matches",
    "adaptive_outer_action_mismatches",
    "modeled_ms_per_output_token",
    "modeled_speedup",
    "output_token_hash",
    "predicted_answer",
    "reference_answer",
    "is_correct",
]

FRONTIER_ROUND_DIAGNOSTIC_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "lowconf_threshold",
    "initial_spec_len",
    "max_spec_len",
    "draft_len",
    "accepted_len",
    "emitted_len",
    "full_accept",
    "extension_capacity",
    "full_accept_with_extension_capacity",
    "extension_count",
    "bucket_stop_requested",
    "bucket_decision_steps",
    "stop_reason",
    "predicted_accepted_tokens",
    "actual_accepted_tokens",
    "prediction_error",
]

FRONTIER_EXTENSION_DIAGNOSTIC_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "event_id",
    "lowconf_threshold",
    "trigger",
    "from_len",
    "to_len",
    "extension_size",
    "actual_extension_accepted_tokens",
    "original_prefix_fully_accepted",
]

FRONTIER_GAIN_DIAGNOSTIC_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "from_step",
    "to_step",
    "lowconf_threshold",
    "from_target_len",
    "to_target_len",
    "same_target_len",
    "predicted_next_gain",
    "predicted_gain_source",
    "gain_bucket_count",
    "gain_bucket_weight",
    "actual_next_gain",
    "prediction_error",
    "decision_should_continue",
    "masks_remaining",
    "expected_output",
    "stop_ms_per_output",
    "continue_ms_per_output",
    "calibration_tokens",
]

BUCKET_ORACLE_SNAPSHOT_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "context_len",
    "step",
    "target_len",
    "draft_passes_elapsed",
    "draft_latency_elapsed_ms",
    "masks_remaining",
    "committed_tokens",
    "filled_tokens",
    "draft_proposal",
    "accept_probabilities",
    "predicted_expected_output",
    "predicted_next_gain",
    "predicted_stop_ms_per_output",
    "predicted_continue_ms_per_output",
    "predicted_should_continue",
    "predicted_gain_source",
    "gain_bucket_count",
    "gain_bucket_weight",
    "calibration_tokens",
    "adaptive_policy_action",
    "adaptive_policy_reason",
    "adaptive_stop_probability",
    "adaptive_advantage_mean",
    "adaptive_advantage_risk",
    "adaptive_q_stop_mean",
    "adaptive_q_continue_mean",
    "adaptive_rho_tokens_per_ms",
    "adaptive_stop_available",
    "accepted_len_if_stop",
    "emitted_len_if_stop",
    "actual_verify_latency_ms",
    "actual_accept_check_latency_ms",
    "actual_shared_post_verify_overhead_ms",
    "actual_post_verify_latency_ms",
]

CAUSAL_ORACLE_CANDIDATE_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "context_len",
    "candidate_index",
    "candidate_source",
    "step",
    "target_len",
    "draft_passes_elapsed",
    "draft_latency_elapsed_ms",
    "estimated_draft_overhead_ms",
    "effective_draft_latency_ms",
    "draft_proposal",
    "accepted_len_if_stop",
    "emitted_len_if_stop",
    "counterfactual_verify_latency_ms",
    "counterfactual_accept_check_latency_ms",
    "counterfactual_total_latency_ms",
    "counterfactual_ms_per_output_token",
    "future_reference_emitted_len",
    "future_token_deficit",
    "expected_extra_verifier_rounds",
    "future_draft_penalty_ms",
    "future_verify_penalty_ms",
    "future_post_verify_penalty_ms",
    "future_round_penalty_ms",
    "adjusted_counterfactual_total_latency_ms",
    "oracle_action",
    "selected",
]

CAUSAL_ORACLE_DECISION_COLUMNS = [
    "problem_id",
    "mode",
    "round_id",
    "context_len",
    "num_candidates",
    "snapshot_fallback_used",
    "oracle_snapshot_attempts",
    "oracle_snapshot_skipped_missing_fill",
    "selected_candidate_index",
    "selected_step",
    "selected_target_len",
    "selected_draft_passes",
    "selected_draft_latency_ms",
    "selected_expected_accepted_len",
    "selected_expected_emitted_len",
    "selected_counterfactual_verify_latency_ms",
    "selected_counterfactual_accept_check_latency_ms",
    "selected_counterfactual_ms_per_output_token",
    "oracle_cost_model",
    "profile_tokens_per_round",
    "profile_draft_ms_per_round",
    "profile_verify_ms_per_round",
    "profile_post_verify_ms_per_round",
    "selected_expected_extra_verifier_rounds",
    "selected_future_draft_penalty_ms",
    "selected_future_verify_penalty_ms",
    "selected_future_post_verify_penalty_ms",
    "selected_future_round_penalty_ms",
    "selected_adjusted_counterfactual_total_latency_ms",
    "oracle_action_trace",
    "physical_draft_passes",
    "physical_draft_latency_ms",
    "excluded_extra_draft_latency_ms",
    "counterfactual_probe_wall_time_ms",
    "executed_accepted_len",
    "executed_emitted_len",
    "executed_verify_latency_ms",
    "executed_post_verify_latency_ms",
    "counterfactual_matches_execution",
]

def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def generation_print(args, *values, **kwargs):
    if not args.quiet_generation:
        print(*values, **kwargs)

def token_sequence_hash(token_ids):
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

def normalize_math_answer(value):
    if value is None:
        return None
    return value.strip().replace(",", "").replace("$", "").replace(" ", "")

def extract_predicted_answer(text):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return normalize_math_answer(matches[-1]) if matches else None

def extract_reference_answer(raw_data):
    answer = raw_data.get("answer")
    if not answer:
        return None
    return normalize_math_answer(answer.rsplit("####", 1)[-1])

def summarize_frontier_diagnostics(stats_each_round):
    stop_actions = 0
    decision_steps = 0
    fill_passes = 0
    denoising_passes = 0
    predicted_gains = []
    expected_outputs = []
    stop_costs = []
    continue_costs = []
    adaptive_decisions = []
    for round_stats in stats_each_round:
        frontier_stats = round_stats.get("frontier_stats") or {}
        breakdown = frontier_stats.get("forward_pass_breakdown") or {}
        fill_passes += int(breakdown.get("fill", 0))
        denoising_passes += int(breakdown.get("denoising", 0))
        adaptive_decisions.extend(frontier_stats.get("adaptive_decisions") or [])
        for action in frontier_stats.get("refinement_actions", []):
            stop_actions += int(action.get("action") == "bucket_renewal_lower_cost")
        bucket_steps = [
            step for step in frontier_stats.get("steps", [])
            if step.get("bucket_expected_output") is not None
        ]
        if bucket_steps:
            decision_steps += len(bucket_steps)
            predicted_gains.extend(
                float(step["predicted_next_gain"])
                for step in bucket_steps
                if step.get("predicted_next_gain") is not None
            )
            final_step = bucket_steps[-1]
            expected_outputs.append(float(final_step["bucket_expected_output"]))
            stop_costs.extend(
                float(step["bucket_stop_ms_per_output"])
                for step in bucket_steps
                if step.get("bucket_stop_ms_per_output") is not None
            )
            continue_costs.extend(
                float(step["bucket_continue_ms_per_output"])
                for step in bucket_steps
                if step.get("bucket_continue_ms_per_output") is not None
            )
    return {
        "bucket_stop_actions": stop_actions,
        "bucket_decision_steps": decision_steps,
        "bucket_predicted_gain_mean": safe_div(sum(predicted_gains), len(predicted_gains)),
        "bucket_expected_output_mean": safe_div(sum(expected_outputs), len(expected_outputs)),
        "bucket_stop_ms_per_output_mean": safe_div(sum(stop_costs), len(stop_costs)),
        "bucket_continue_ms_per_output_mean": safe_div(sum(continue_costs), len(continue_costs)),
        "bucket_fill_forward_passes": fill_passes,
        "bucket_denoising_forward_passes": denoising_passes,
        "adaptive_decisions": len(adaptive_decisions),
        "adaptive_stop_actions": sum(
            item.get("action") == "stop" for item in adaptive_decisions
        ),
        "adaptive_exploration_actions": sum(
            bool(item.get("exploration_used")) for item in adaptive_decisions
        ),
        "adaptive_calibration_decisions": sum(
            bool(item.get("early_stop_calibration_active"))
            for item in adaptive_decisions
        ),
        "adaptive_calibration_stop_actions": sum(
            item.get("action") == "stop"
            and item.get("reason") == "early_stop_calibration_exploration"
            for item in adaptive_decisions
        ),
        "adaptive_learned_stop_actions": sum(
            item.get("action") == "stop"
            and item.get("reason") == "stop_probability_threshold"
            for item in adaptive_decisions
        ),
        "adaptive_early_stop_observations": max(
            (
                int(item.get("early_stop_observations_after_round", 0))
                for item in adaptive_decisions
            ),
            default=0,
        ),
        "adaptive_behavior_stop_probability_mean": safe_div(
            sum(
                float(item.get("behavior_stop_probability", 0.0))
                for item in adaptive_decisions
            ),
            len(adaptive_decisions),
        ),
        "adaptive_selected_action_probability_mean": safe_div(
            sum(
                float(item.get("selected_action_probability", 1.0))
                for item in adaptive_decisions
            ),
            len(adaptive_decisions),
        ),
        "adaptive_importance_weight_mean": safe_div(
            sum(
                float(item.get("importance_weight", 1.0))
                for item in adaptive_decisions
            ),
            len(adaptive_decisions),
        ),
        "adaptive_mean_refinement_step": safe_div(
            sum(float(item.get("step", 0)) for item in adaptive_decisions),
            len(adaptive_decisions),
        ),
        "adaptive_controller_ms": sum(
            float(item.get("controller_latency_ms", 0.0))
            for item in adaptive_decisions
        ),
        "adaptive_stop_available_decisions": sum(
            bool(item.get("stop_available")) for item in adaptive_decisions
        ),
        "adaptive_candidate_coverage_decisions": sum(
            bool(item.get("candidate_coverage_available"))
            for item in adaptive_decisions
        ),
        "adaptive_outer_verify_eligible_decisions": sum(
            bool(item.get("outer_failfast_verify_eligible"))
            for item in adaptive_decisions
        ),
        "adaptive_stop_then_extend_actions": sum(
            item.get("action") == "stop"
            and item.get("realized_post_stop_outer_action") == "extend"
            for item in adaptive_decisions
        ),
        "adaptive_stop_then_verify_actions": sum(
            item.get("action") == "stop"
            and item.get("realized_post_stop_outer_action") == "verify"
            for item in adaptive_decisions
        ),
        "adaptive_outer_action_matches": sum(
            item.get("outer_action_matches_plan") is True
            for item in adaptive_decisions
        ),
        "adaptive_outer_action_mismatches": sum(
            item.get("outer_action_matches_plan") is False
            for item in adaptive_decisions
        ),
        "adaptive_rho_tokens_per_ms": (
            args.adaptive_td_controller.rho
            if getattr(args, "adaptive_td", False)
            else 0.0
        ),
    }

def frontier_stop_enabled(args):
    return bool(
        getattr(args, "adaptive_td", False)
        or getattr(args, "strict_greedy_local_oracle", False)
    ) or (
        getattr(args, "frontier_stop_mode", "disabled")
        not in (None, "disabled", "none", "off")
    )

def ensure_frontier_runtime_state(args):
    if not hasattr(args, "bucket_acceptance_calibration"):
        args.bucket_acceptance_calibration = {}
    calibration = args.bucket_acceptance_calibration
    for table_name in (
        "token_step_position_confidence_margin",
        "token_step_confidence_margin",
        "token_step_position_confidence",
        "token_step_confidence",
        "token_position_confidence_margin",
        "token_position_confidence",
        "token_confidence_margin",
        "token_confidence",
    ):
        calibration.setdefault(table_name, {})
    calibration.setdefault("total_checked_tokens", 0)
    if not hasattr(args, "bucket_gain_calibration"):
        args.bucket_gain_calibration = {
            "length_score_masks": {},
            "score_masks": {},
            "length_score": {},
            "score": {},
            "step": {},
            "global": [0.0, 0],
        }
    if not hasattr(args, "bucket_ema_dllm_forward_ms"):
        args.bucket_ema_dllm_forward_ms = None
    if not hasattr(args, "bucket_ema_target_round_ms"):
        args.bucket_ema_target_round_ms = None
    if not hasattr(args, "bucket_ema_post_verify_ms"):
        args.bucket_ema_post_verify_ms = None
    if not hasattr(args, "bucket_verify_latency_bins"):
        args.bucket_verify_latency_bins = {}
    if not hasattr(args, "bucket_draft_latency_bins"):
        args.bucket_draft_latency_bins = {}

def reset_frontier_runtime_state(args, preserve_hardware_latency=False):
    preserved = {}
    if preserve_hardware_latency:
        for name in (
            "bucket_ema_dllm_forward_ms",
            "bucket_ema_target_round_ms",
            "bucket_ema_post_verify_ms",
            "bucket_verify_latency_bins",
            "bucket_draft_latency_bins",
        ):
            if hasattr(args, name):
                preserved[name] = getattr(args, name)
    for name in (
        "bucket_acceptance_calibration",
        "bucket_gain_calibration",
        "bucket_ema_dllm_forward_ms",
        "bucket_ema_target_round_ms",
        "bucket_ema_post_verify_ms",
        "bucket_verify_latency_bins",
        "bucket_draft_latency_bins",
        "last_frontier_stats",
    ):
        if hasattr(args, name):
            delattr(args, name)
    for name, value in preserved.items():
        setattr(args, name, value)

def update_ema(old_value, new_value, alpha):
    if old_value is None:
        return float(new_value)
    return (1.0 - alpha) * float(old_value) + alpha * float(new_value)

def calibration_bin(value):
    return str(max(0, min(9, int(float(value) * 10.0))))

def update_calibration_bucket(table, key, accepted):
    accepted_count, total_count = table.get(key, [0.0, 0.0])
    table[key] = [accepted_count + float(accepted), total_count + 1.0]


def bucket_length_bin(length):
    return str(max(1, math.ceil(int(length) / 8)))


def bucket_context_bin(context_len):
    return str(max(0, int(context_len) // 256))


def update_latency_bucket(table, key, value, alpha):
    current_value, count = table.get(key, [None, 0])
    table[key] = [update_ema(current_value, value, alpha), int(count) + 1]

def update_frontier_latency_cost(
    args,
    forward_pass_latencies,
    verify_time,
    draft_len,
    context_len=None,
):
    ensure_frontier_runtime_state(args)
    alpha = args.bucket_latency_ema_alpha
    if forward_pass_latencies:
        avg_forward_ms = sum(forward_pass_latencies) / len(forward_pass_latencies)
        args.bucket_ema_dllm_forward_ms = update_ema(args.bucket_ema_dllm_forward_ms, avg_forward_ms, alpha)
    if draft_len > 0 and verify_time > 0:
        args.bucket_ema_target_round_ms = update_ema(
            args.bucket_ema_target_round_ms,
            verify_time * 1000.0,
            alpha,
        )
    if context_len is not None:
        context_key = bucket_context_bin(context_len)
        if forward_pass_latencies:
            update_latency_bucket(
                args.bucket_draft_latency_bins,
                context_key,
                sum(forward_pass_latencies) / len(forward_pass_latencies),
                alpha,
            )
        if draft_len > 0 and verify_time > 0:
            verify_key = f"{context_key}:{bucket_length_bin(draft_len)}"
            update_latency_bucket(
                args.bucket_verify_latency_bins,
                verify_key,
                verify_time * 1000.0,
                alpha,
            )

def update_frontier_controller_cost(args, controller_time):
    ensure_frontier_runtime_state(args)
    if controller_time > 0:
        args.bucket_ema_post_verify_ms = update_ema(
            args.bucket_ema_post_verify_ms,
            controller_time * 1000.0,
            args.bucket_latency_ema_alpha,
        )

def update_frontier_acceptance_calibration(args, frontier_stats, accepted_outcomes):
    if not frontier_stats or not accepted_outcomes:
        return
    ensure_frontier_runtime_state(args)
    calibration = args.bucket_acceptance_calibration
    draft_token_stats = frontier_stats.get("draft_token_stats", [])
    for idx, accepted in enumerate(accepted_outcomes):
        if idx >= len(draft_token_stats):
            break
        token_stats = draft_token_stats[idx]
        confidence = float(token_stats.get("confidence", 0.0))
        margin = float(token_stats.get("margin", 0.0))
        confidence_bin = calibration_bin(confidence)
        margin_bin = calibration_bin(margin)
        position_bin = position_bucket(idx)
        commit_step = str(max(1, int(token_stats.get("commit_step", 1))))
        update_calibration_bucket(
            calibration["token_step_position_confidence_margin"],
            f"{commit_step}:{position_bin}:{confidence_bin}:{margin_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_step_confidence_margin"],
            f"{commit_step}:{confidence_bin}:{margin_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_step_position_confidence"],
            f"{commit_step}:{position_bin}:{confidence_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_step_confidence"],
            f"{commit_step}:{confidence_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_position_confidence_margin"],
            f"{position_bin}:{confidence_bin}:{margin_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_position_confidence"],
            f"{position_bin}:{confidence_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_confidence_margin"],
            f"{confidence_bin}:{margin_bin}",
            accepted,
        )
        update_calibration_bucket(
            calibration["token_confidence"],
            confidence_bin,
            accepted,
        )
        calibration["total_checked_tokens"] += 1


def complete_adaptive_td_trajectory(
    args,
    frontier_stats,
    *,
    emitted_tokens,
    verifier_latency_ms,
    post_verify_latency_ms=0.0,
    round_latency_ms=None,
    terminal=False,
):
    if (
        not getattr(args, "adaptive_td", False)
        or getattr(args, "adaptive_freeze", False)
        or not frontier_stats
    ):
        return
    controller = args.adaptive_td_controller
    trajectory = frontier_stats.get("adaptive_trajectory") or []
    controller.complete_trajectory(
        trajectory,
        emitted_tokens=int(emitted_tokens),
        verifier_latency_ms=float(verifier_latency_ms),
        post_verify_latency_ms=float(post_verify_latency_ms),
        round_latency_ms=(
            None if round_latency_ms is None else float(round_latency_ms)
        ),
        terminal=bool(terminal),
    )


def record_adaptive_td_decisions(
    args,
    frontier_stats,
    *,
    problem_id,
    round_id,
    accepted_draft_tokens,
    emitted_tokens,
    verifier_latency_ms,
    round_total_latency_ms,
):
    if (
        not getattr(args, "adaptive_td", False)
        or not args.adaptive_log_decisions
        or not frontier_stats
    ):
        return
    controller = args.adaptive_td_controller
    round_total_latency_ms = max(0.0, float(round_total_latency_ms))
    round_throughput = safe_div(emitted_tokens, round_total_latency_ms)
    logging_started = time.perf_counter()
    if not hasattr(args, "adaptive_decision_rows"):
        args.adaptive_decision_rows = []
    final_draft_length = int(frontier_stats.get("actual_spec_len") or 0)
    extension_events = frontier_stats.get("extension_events") or []
    for decision_id, item in enumerate(frontier_stats.get("adaptive_decisions") or []):
        target_len = int(item.get("target_len", 0))
        matching_extensions = [
            event
            for event in extension_events
            if int(event.get("from_len", -1)) == target_len
            and event.get("trigger") == "high_confidence_extend"
        ]
        high_confidence_extension = sum(
            int(event.get("extension_size", 0))
            for event in matching_extensions
        )
        finalized_fields = {
            "final_draft_length": final_draft_length,
            "draft_length_delta_after_decision": final_draft_length - target_len,
            "high_confidence_extension_realized": bool(
                high_confidence_extension
            ),
            "high_confidence_extension_size": int(high_confidence_extension),
            "realized_post_stop_outer_action": None,
            "outer_action_matches_plan": None,
            "early_stop_observations_after_round": int(
                controller.early_stop_observations
            ),
        }
        if item.get("action") == "stop":
            realized_action = (
                "extend" if high_confidence_extension else "verify"
            )
            finalized_fields["realized_post_stop_outer_action"] = realized_action
            finalized_fields["outer_action_matches_plan"] = (
                realized_action == item.get("post_stop_outer_action")
            )
        item.update(finalized_fields)
        feature_values = {
            name: float(value)
            for name, value in zip(
                getattr(controller, "feature_names", FEATURE_NAMES),
                item.get("features") or [],
            )
        }
        args.adaptive_decision_rows.append({
            "adaptive_controller": getattr(controller, "controller_name", "avg_td"),
            "problem_id": int(problem_id),
            "round_id": int(round_id),
            "decision_id": int(decision_id),
            **item,
            "features": json.dumps(item.get("features") or []),
            "feature_names": json.dumps(list(
                getattr(controller, "feature_names", FEATURE_NAMES)
            )),
            "draft_proposal": json.dumps(item.get("draft_proposal") or []),
            "draft_length": target_len,
            **feature_values,
            "accepted_draft_tokens": int(accepted_draft_tokens),
            "emitted_tokens": int(emitted_tokens),
            "verifier_latency_ms": float(verifier_latency_ms),
            "round_total_latency_ms": round_total_latency_ms,
            "round_throughput_tokens_per_ms": round_throughput,
        })
    controller.record_profile(
        "logging",
        (time.perf_counter() - logging_started) * 1000.0,
    )

def build_benchmark_drafter_configs(args):
    dllm_config = (
        ("dllm", args.drafter_thresholds[0], "sf", None, None, None)
        if args.dllm_variant == "fixed"
        else (
            "dllm",
            args.drafter_thresholds[0],
            "df",
            args.sweep_lowconf_threshold[0],
            args.sweep_max_spec_len[0],
            args.sweep_incr_len[0],
        )
    )
    return {
        "verifier_ar": ("verifier_ar", None, "none", None, None, None),
        "ar_ar": ("ar", None, "sf", None, None, None),
        "dllm_ar": dllm_config,
    }

def build_benchmark_row(
    args,
    problem_id,
    mode,
    draft_time_total,
    verify_time_total,
    total_num_forward_passes,
    num_speculation_rounds,
    accepted_tokens,
    drafted_tokens,
    actual_e2e_time,
    post_verify_time_total,
    output_token_ids,
    predicted_answer,
    reference_answer,
    frontier_diagnostics,
    device_transfer_time_total,
):
    latency = args.latency["vLLM_A6000"]
    actual_draft_time = draft_time_total
    actual_verify_time = verify_time_total
    actual_total_time = actual_draft_time + actual_verify_time
    actual_algorithm_time = actual_total_time + post_verify_time_total
    theo_draft_time = total_num_forward_passes * latency["draft_fwd_pass"]
    theo_verify_time = num_speculation_rounds * latency["target_tpt"][args.target_model_name_clean]
    theo_total_time = theo_draft_time + theo_verify_time
    output_tokens = len(output_token_ids)
    modeled_ms_per_output_token = safe_div(theo_total_time, output_tokens)
    actual_e2e_ms_per_output_token = safe_div(actual_e2e_time * 1000.0, output_tokens)
    actual_unattributed_core_time = max(0.0, actual_e2e_time - actual_algorithm_time)
    e2e_time_excluding_transfer = max(
        0.0,
        actual_e2e_time - device_transfer_time_total,
    )
    return {
        "problem_id": problem_id,
        "mode": mode,
        "actual_total_time": actual_total_time,
        "theo_total_time": theo_total_time,
        "actual_draft_time": actual_draft_time,
        "theo_draft_time": theo_draft_time,
        "actual_verify_time": actual_verify_time,
        "theo_verify_time": theo_verify_time,
        "actual_draft_verify_ratio": safe_div(actual_draft_time, actual_verify_time),
        "theo_draft_verify_ratio": safe_div(theo_draft_time, theo_verify_time),
        "acceptance_rate_percent": safe_div(accepted_tokens, drafted_tokens) * 100.0,
        "actual_speedup_vs_AR": None,
        "actual_e2e_speedup_vs_AR": None,
        "theo_speedup_vs_AR": None,
        "actual_e2e_time": actual_e2e_time,
        "actual_e2e_ms_per_output_token": actual_e2e_ms_per_output_token,
        "output_tokens_per_ms": safe_div(output_tokens, actual_e2e_time * 1000.0),
        "actual_post_verify_time": post_verify_time_total,
        "actual_algorithm_time": actual_algorithm_time,
        "device_transfer_time": device_transfer_time_total,
        "device_transfer_ms_per_output_token": safe_div(
            device_transfer_time_total * 1000.0,
            output_tokens,
        ),
        "actual_e2e_time_excluding_transfer": (
            e2e_time_excluding_transfer
        ),
        "e2e_ms_per_output_token_excluding_transfer": safe_div(
            e2e_time_excluding_transfer * 1000.0,
            output_tokens,
        ),
        "actual_algorithm_ms_per_output_token": safe_div(actual_algorithm_time * 1000.0, output_tokens),
        "actual_unattributed_core_time": actual_unattributed_core_time,
        "output_tokens": output_tokens,
        "accepted_tokens": accepted_tokens,
        "drafted_tokens": drafted_tokens,
        "num_speculation_rounds": num_speculation_rounds,
        "total_num_forward_passes": total_num_forward_passes,
        **frontier_diagnostics,
        "modeled_ms_per_output_token": modeled_ms_per_output_token,
        "modeled_speedup": safe_div(latency["target_tpt"][args.target_model_name_clean], modeled_ms_per_output_token),
        "output_token_hash": token_sequence_hash(output_token_ids),
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        "is_correct": predicted_answer is not None and predicted_answer == reference_answer,
    }

def append_benchmark_rows(args, rows):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def append_csv_rows(path, fieldnames, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def append_frontier_diagnostic_rows(args, problem_id, mode, stats_each_round):
    round_rows = []
    extension_rows = []
    gain_rows = []
    for round_id, round_stats in enumerate(stats_each_round):
        frontier_stats = round_stats.get("frontier_stats") or {}
        if not frontier_stats or frontier_stats.get("mode") in (None, "disabled", "none", "off"):
            continue

        draft_proposal = round_stats.get("~draft_proposal") or []
        draft_len = len(draft_proposal)
        accepted_len = int(round_stats.get("accepted_len", 0))
        extension_events = frontier_stats.get("extension_events") or []
        max_spec_len = int(frontier_stats.get("max_spec_len") or draft_len)
        full_accept = draft_len > 0 and accepted_len == draft_len
        extension_capacity = draft_len < max_spec_len
        actions = frontier_stats.get("refinement_actions") or []
        cost_stop_requested = any(
            action.get("action") == "bucket_renewal_lower_cost"
            for action in actions
        )
        predicted_accepted = frontier_stats.get("final_frontier_score")
        steps = frontier_stats.get("steps") or []
        prediction_error = (
            float(predicted_accepted) - accepted_len
            if predicted_accepted is not None
            else None
        )
        round_rows.append({
            "problem_id": problem_id,
            "mode": mode,
            "round_id": round_id,
            "lowconf_threshold": frontier_stats.get("lowconf_threshold"),
            "initial_spec_len": frontier_stats.get("initial_spec_len"),
            "max_spec_len": max_spec_len,
            "draft_len": draft_len,
            "accepted_len": accepted_len,
            "emitted_len": len(round_stats.get("emitted_tokens") or []),
            "full_accept": int(full_accept),
            "extension_capacity": int(extension_capacity),
            "full_accept_with_extension_capacity": int(full_accept and extension_capacity),
            "extension_count": len(extension_events),
            "bucket_stop_requested": int(cost_stop_requested),
            "bucket_decision_steps": sum(
                step.get("predicted_next_gain") is not None for step in steps
            ),
            "stop_reason": frontier_stats.get("stop_reason"),
            "predicted_accepted_tokens": predicted_accepted,
            "actual_accepted_tokens": accepted_len,
            "prediction_error": prediction_error,
        })

        for event_id, event in enumerate(extension_events):
            from_len = int(event.get("from_len", 0))
            extension_size = int(event.get("extension_size", 0))
            actual_gain = max(0, min(extension_size, accepted_len - from_len))
            extension_rows.append({
                "problem_id": problem_id,
                "mode": mode,
                "round_id": round_id,
                "event_id": event_id,
                "lowconf_threshold": frontier_stats.get("lowconf_threshold"),
                "trigger": event.get("trigger"),
                "from_len": from_len,
                "to_len": event.get("to_len"),
                "extension_size": extension_size,
                "actual_extension_accepted_tokens": actual_gain,
                "original_prefix_fully_accepted": int(accepted_len >= from_len),
            })

        for from_index, (current_step, next_step) in enumerate(zip(steps, steps[1:])):
            predicted_gain = current_step.get("predicted_next_gain")
            actual_gain = next_step.get("frontier_gain")
            if predicted_gain is None or actual_gain is None:
                continue
            from_target_len = int(current_step.get("target_len", 0))
            to_target_len = int(next_step.get("target_len", 0))
            gain_rows.append({
                "problem_id": problem_id,
                "mode": mode,
                "round_id": round_id,
                "from_step": from_index + 1,
                "to_step": from_index + 2,
                "lowconf_threshold": frontier_stats.get("lowconf_threshold"),
                "from_target_len": from_target_len,
                "to_target_len": to_target_len,
                "same_target_len": int(from_target_len == to_target_len),
                "predicted_next_gain": float(predicted_gain),
                "predicted_gain_source": current_step.get("predicted_next_gain_source"),
                "gain_bucket_count": current_step.get("gain_bucket_count"),
                "gain_bucket_weight": current_step.get("gain_bucket_weight"),
                "actual_next_gain": float(actual_gain),
                "prediction_error": float(predicted_gain) - float(actual_gain),
                "decision_should_continue": int(bool(current_step.get("bucket_should_continue"))),
                "masks_remaining": current_step.get("masks_remaining"),
                "expected_output": current_step.get("bucket_expected_output"),
                "stop_ms_per_output": current_step.get("bucket_stop_ms_per_output"),
                "continue_ms_per_output": current_step.get("bucket_continue_ms_per_output"),
                "calibration_tokens": current_step.get("bucket_calibration_tokens"),
            })

    append_csv_rows(
        os.path.join(args.output_dir, "frontier_round_diagnostics.csv"),
        FRONTIER_ROUND_DIAGNOSTIC_COLUMNS,
        round_rows,
    )
    append_csv_rows(
        os.path.join(args.output_dir, "frontier_extension_diagnostics.csv"),
        FRONTIER_EXTENSION_DIAGNOSTIC_COLUMNS,
        extension_rows,
    )
    append_csv_rows(
        os.path.join(args.output_dir, "frontier_gain_diagnostics.csv"),
        FRONTIER_GAIN_DIAGNOSTIC_COLUMNS,
        gain_rows,
    )


def evaluate_oracle_proposal(
    target_model,
    orig_model_inputs,
    current_token_ids,
    draft_proposal,
):
    result = evaluate_oracle_proposal_tokens(
        target_model,
        orig_model_inputs,
        current_token_ids,
        draft_proposal,
        max_append_tokens=len(draft_proposal) + 1,
        eos_token_id=None,
    )
    return (
        result["accepted_len"],
        result["emitted_len"],
        result["verify_latency_ms"],
        result["post_verify_latency_ms"],
    )


def evaluate_oracle_proposal_tokens(
    target_model,
    orig_model_inputs,
    current_token_ids,
    draft_proposal,
    *,
    max_append_tokens,
    eos_token_id,
):
    combined_ids = current_token_ids + draft_proposal
    proposal_tensor = torch.tensor(
        [combined_ids],
        device=target_model.device,
        dtype=torch.long,
    )
    full_input_ids = torch.cat(
        [orig_model_inputs["input_ids"], proposal_tensor],
        dim=1,
    )
    full_attention_mask = torch.cat(
        [orig_model_inputs["attention_mask"], torch.ones_like(proposal_tensor)],
        dim=1,
    )
    if proposal_tensor.is_cuda:
        torch.cuda.synchronize(proposal_tensor.device)
    verify_start = time.perf_counter()
    with torch.inference_mode():
        oracle_outputs = target_model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
            logits_to_keep=len(draft_proposal) + 1,
        )
    if proposal_tensor.is_cuda:
        torch.cuda.synchronize(proposal_tensor.device)
    verify_latency_ms = (time.perf_counter() - verify_start) * 1000.0

    post_verify_start = time.perf_counter()
    verify_logits = oracle_outputs.logits[0, :len(draft_proposal)]
    accepted_len = 0
    final_token = None
    for index, draft_token_id in enumerate(draft_proposal):
        target_token_id = int(torch.argmax(verify_logits[index], dim=-1).item())
        if int(draft_token_id) != target_token_id:
            final_token = target_token_id
            break
        accepted_len += 1
    if final_token is None:
        final_token = int(torch.argmax(oracle_outputs.logits[0, -1, :], dim=-1).item())
    tokens_to_append = [
        int(token_id)
        for token_id in draft_proposal[:accepted_len]
    ] + [final_token]
    if eos_token_id is not None and int(eos_token_id) in tokens_to_append:
        eos_index = tokens_to_append.index(int(eos_token_id))
        tokens_to_append = tokens_to_append[:eos_index + 1]
    tokens_to_append = tokens_to_append[:max(0, int(max_append_tokens))]
    post_verify_latency_ms = (time.perf_counter() - post_verify_start) * 1000.0
    del verify_logits, oracle_outputs
    return {
        "accepted_len": min(accepted_len, len(tokens_to_append)),
        "emitted_len": len(tokens_to_append),
        "tokens_to_append": tokens_to_append,
        "final_token": final_token,
        "verify_latency_ms": verify_latency_ms,
        "post_verify_latency_ms": post_verify_latency_ms,
    }


@torch.inference_mode()
def audit_committed_greedy_tokens(
    target_model,
    orig_model_inputs,
    current_token_ids,
    draft_proposal,
    accepted_len,
    tokens_to_append,
    batched_output_logits,
    *,
    eos_token_id,
):
    """Compare batched verifier choices with one-prefix-at-a-time argmax."""
    rows = []
    replay_prefix = list(current_token_ids)
    proposal_len = len(draft_proposal)
    for emitted_offset, emitted_token in enumerate(tokens_to_append):
        replay_tensor = torch.tensor(
            [replay_prefix],
            device=target_model.device,
            dtype=torch.long,
        )
        replay_input_ids = torch.cat(
            [orig_model_inputs["input_ids"], replay_tensor],
            dim=1,
        )
        replay_attention_mask = torch.cat(
            [orig_model_inputs["attention_mask"], torch.ones_like(replay_tensor)],
            dim=1,
        )
        replay_outputs = target_model(
            input_ids=replay_input_ids,
            attention_mask=replay_attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        prefix_logits = replay_outputs.logits[0, -1].float()

        if emitted_offset < accepted_len:
            batched_logit_index = emitted_offset
            token_role = "accepted_draft"
        elif accepted_len < proposal_len:
            batched_logit_index = accepted_len
            token_role = "correction"
        else:
            batched_logit_index = proposal_len
            token_role = "bonus"
        batched_logits = batched_output_logits[batched_logit_index].float()

        batched_values, batched_ids = torch.topk(batched_logits, k=2)
        prefix_values, prefix_ids = torch.topk(prefix_logits, k=2)
        # topk does not preserve argmax tie-breaking when logits are equal.
        batched_token = int(torch.argmax(batched_logits, dim=-1).item())
        prefix_token = int(torch.argmax(prefix_logits, dim=-1).item())
        rows.append({
            "emitted_offset": int(emitted_offset),
            "absolute_output_position": int(len(current_token_ids) + emitted_offset),
            "token_role": token_role,
            "proposal_length": int(proposal_len),
            "accepted_length": int(accepted_len),
            "emitted_token": int(emitted_token),
            "batched_argmax_token": batched_token,
            "prefix_argmax_token": prefix_token,
            "emitted_matches_batched": int(int(emitted_token) == batched_token),
            "batched_matches_prefix": int(batched_token == prefix_token),
            "batched_margin": float((batched_values[0] - batched_values[1]).item()),
            "prefix_margin": float((prefix_values[0] - prefix_values[1]).item()),
            "batched_top2_token": int(batched_ids[1].item()),
            "prefix_top2_token": int(prefix_ids[1].item()),
            "emitted_is_eos": int(
                eos_token_id is not None and int(emitted_token) == int(eos_token_id)
            ),
        })
        replay_prefix.append(int(emitted_token))
        del replay_outputs, prefix_logits, batched_logits
    return rows


def enumerate_global_oracle_round_edges(
    args,
    problem_id,
    target_model,
    dllm,
    orig_model_inputs,
    current_token_ids,
    num_target_tokens,
    drafter_threshold,
    lowconf_threshold,
    max_spec_len,
    incr_len,
):
    controller = args.adaptive_td_controller
    pending_scripts = [tuple()]
    visited_scripts = set()
    decision_states = {}
    edges = []
    replay_count = 0
    search_started = time.perf_counter()

    while pending_scripts:
        script = pending_scripts.pop()
        if script in visited_scripts:
            continue
        visited_scripts.add(script)
        replay_count += 1
        if (
            args.global_oracle_max_states
            and replay_count > args.global_oracle_max_states
        ):
            raise RuntimeError(
                "Exact global oracle replay limit reached; no approximate result "
                "was written. Increase --global_oracle_max_states or set it to 0."
            )
        controller.set_script(script)
        args.last_frontier_stats = None
        transformers.set_seed(args.seed)
        if orig_model_inputs["input_ids"].is_cuda:
            torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
        draft_started = time.perf_counter()
        try:
            (
                draft_proposal,
                actual_spec_len,
                num_forward_passes,
                forward_pass_latencies,
            ) = get_next_tokens_dllm(
                dllm,
                args,
                orig_model_inputs,
                current_token_ids,
                spec_len=args.spec_len,
                output_seqlen=3 * args.block_size,
                small_block_size=args.small_block_size,
                threshold=drafter_threshold,
                is_drafter=True,
                lowconf_threshold=lowconf_threshold,
                max_spec_len=max_spec_len,
                incr_len=incr_len,
                last_round_rejected=None,
            )
        except OracleBranchRequired as required:
            if required.decision_index != len(script):
                raise RuntimeError("oracle replay consumed an inconsistent script")
            decision_states[script] = {
                "problem_id": int(problem_id),
                "prefix_len": len(current_token_ids),
                "script": list(script),
                **required.state,
            }
            pending_scripts.append(script + (STOP,))
            pending_scripts.append(script + (CONTINUE,))
            continue
        if orig_model_inputs["input_ids"].is_cuda:
            torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
        draft_replay_wall_time_ms = (time.perf_counter() - draft_started) * 1000.0
        if controller.script_position != len(script):
            raise RuntimeError("oracle replay completed without consuming its script")
        if not draft_proposal:
            raise RuntimeError("global oracle drafter returned an empty proposal")

        frontier_stats = getattr(args, "last_frontier_stats", None) or {}
        verification = evaluate_oracle_proposal_tokens(
            target_model,
            orig_model_inputs,
            current_token_ids,
            [int(token_id) for token_id in draft_proposal],
            max_append_tokens=num_target_tokens - len(current_token_ids),
            eos_token_id=args.target_tokenizer.eos_token_id,
        )
        child_tokens = tuple(
            list(current_token_ids) + verification["tokens_to_append"]
        )
        decision_trace = []
        for decision in frontier_stats.get("adaptive_decisions") or []:
            decision_trace.append({
                "step": int(decision.get("step", 0)),
                "target_len": int(decision.get("target_len", 0)),
                "remaining_masks": int(decision.get("remaining_masks", 0)),
                "newly_unmasked": int(decision.get("newly_unmasked", 0)),
                "action": decision.get("action"),
                "stop_available": bool(decision.get("stop_available")),
                "outer_action_after_stop": decision.get("post_stop_outer_action"),
                "elapsed_draft_ms": float(
                    decision.get("oracle_elapsed_draft_ms", 0.0)
                ),
            })
        extension_events = frontier_stats.get("extension_events") or []
        verify_latency_ms = float(verification["verify_latency_ms"])
        post_verify_latency_ms = float(verification["post_verify_latency_ms"])
        draft_latency_ms = float(sum(forward_pass_latencies))
        edge = {
            "candidate_index": len(edges),
            "state": len(current_token_ids),
            "child_state": len(child_tokens),
            "step": len(script),
            "draft_passes": int(num_forward_passes),
            "draft_latency_ms": draft_latency_ms,
            "verify_latency_ms": verify_latency_ms,
            "post_verify_latency_ms": post_verify_latency_ms,
            "edge_latency_ms": (
                draft_latency_ms + verify_latency_ms + post_verify_latency_ms
            ),
            "emitted_len": int(verification["emitted_len"]),
            "accepted_len": int(verification["accepted_len"]),
            "proposal_len": int(actual_spec_len),
            "blocks": 1 + len(extension_events),
            "action_script": list(script),
            "action_script_text": "".join(
                "S" if action == STOP else "C" for action in script
            ),
            "decision_trace": decision_trace,
            "draft_proposal": [int(token_id) for token_id in draft_proposal],
            "tokens_to_append": list(verification["tokens_to_append"]),
            "child_tokens": child_tokens,
            "final_token": int(verification["final_token"]),
            "is_failfast": int(all(action == CONTINUE for action in script)),
            "replay_forward_latency_ms": float(sum(forward_pass_latencies)),
            "draft_replay_wall_time_ms": draft_replay_wall_time_ms,
            "draft_non_forward_replay_ms": max(
                0.0, draft_replay_wall_time_ms - draft_latency_ms
            ),
            "stop_reason": frontier_stats.get("stop_reason"),
            "extension_events": extension_events,
        }
        edges.append(edge)

    failfast_edges = [edge for edge in edges if edge["is_failfast"]]
    if len(failfast_edges) != 1:
        raise RuntimeError(
            f"expected one all-CONTINUE FailFast replay, found {len(failfast_edges)}"
        )
    return {
        "edges": edges,
        "decision_states": decision_states,
        "replays": replay_count,
        "search_wall_time_ms": (time.perf_counter() - search_started) * 1000.0,
    }


def run_oracle_draft_trajectory(
    args,
    dllm,
    orig_model_inputs,
    current_token_ids,
    drafter_threshold,
    lowconf_threshold,
    max_spec_len,
    incr_len,
    action_script=(),
    default_action=CONTINUE,
    prev_prefill_output=None,
    last_round_rejected=None,
):
    controller = args.adaptive_td_controller
    controller.set_script(action_script, default_action=default_action)
    args.last_frontier_stats = None
    transformers.set_seed(args.seed)
    if orig_model_inputs["input_ids"].is_cuda:
        torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
    started = time.perf_counter()
    draft_result = get_next_tokens_dllm(
        dllm,
        args,
        orig_model_inputs,
        current_token_ids,
        spec_len=args.spec_len,
        output_seqlen=3 * args.block_size,
        small_block_size=args.small_block_size,
        threshold=drafter_threshold,
        is_drafter=True,
        lowconf_threshold=lowconf_threshold,
        max_spec_len=max_spec_len,
        incr_len=incr_len,
        last_round_rejected=last_round_rejected,
        prev_prefill_output=prev_prefill_output,
    )
    if args.disable_reusing_drafter_kvs:
        (
            proposal,
            actual_spec_len,
            num_forward_passes,
            forward_pass_latencies,
        ) = draft_result
        prefill_output = None
    else:
        (
            proposal,
            actual_spec_len,
            prefill_output,
            num_forward_passes,
            forward_pass_latencies,
        ) = draft_result
    if orig_model_inputs["input_ids"].is_cuda:
        torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
    return {
        "proposal": [int(token_id) for token_id in proposal],
        "proposal_len": int(actual_spec_len),
        "draft_passes": int(num_forward_passes),
        "draft_latency_ms": float(sum(forward_pass_latencies)),
        "draft_wall_time_ms": (time.perf_counter() - started) * 1000.0,
        "frontier_stats": getattr(args, "last_frontier_stats", None) or {},
        "action_script": tuple(controller.executed_actions),
        "prefill_output": prefill_output,
    }


def run_strict_greedy_local_oracle_problem(
    args,
    problem_id,
    target_model,
    dllm,
    orig_model_inputs,
    num_target_tokens,
    drafter_threshold,
    lowconf_threshold,
    max_spec_len,
    incr_len,
):
    profile = args.strict_greedy_profile
    mean_verify_ms = float(profile["mean_verify_latency_ms"])
    mean_tokens_per_verify = float(profile["mean_tokens_per_verify"])
    capacity_collector = bool(args.strict_greedy_capacity_collector)
    rho_profile = mean_tokens_per_verify / mean_verify_ms
    feature_names = tuple(args.adaptive_td_controller.feature_names)
    eos_token_id = args.target_tokenizer.eos_token_id
    current_token_ids = []
    stats_each_round = []
    draft_latency_ms = 0.0
    verify_latency_ms = 0.0
    post_verify_latency_ms = 0.0
    draft_passes = 0
    round_id = 0
    prev_prefill_output = None
    replay_rounds = None
    if args.strict_greedy_replay_data is not None:
        replay_rounds = args.strict_greedy_replay_data.get("policies", {}).get(
            str(problem_id)
        )
        if replay_rounds is None and getattr(
            args, "strict_greedy_record_diagnostics", True
        ):
            raise ValueError(
                f"strict greedy replay policy has no problem_id={problem_id}"
            )
    selected_round_actions = []

    def snapshot_causal_profile():
        controller = args.adaptive_td_controller
        return (
            controller.factual_draft_latency_ema_ms,
            controller.factual_verifier_latency_ema_ms,
            controller.factual_tokens_per_verifier_ema,
        )

    def restore_causal_profile(snapshot):
        controller = args.adaptive_td_controller
        (
            controller.factual_draft_latency_ema_ms,
            controller.factual_verifier_latency_ema_ms,
            controller.factual_tokens_per_verifier_ema,
        ) = snapshot

    round_causal_profile = snapshot_causal_profile()

    def run_draft(
        script,
        default_action,
        reuse_selected_cache=False,
        preserve_causal_profile=False,
    ):
        if capacity_collector:
            restore_causal_profile(round_causal_profile)
        try:
            result = run_oracle_draft_trajectory(
                args,
                dllm,
                orig_model_inputs,
                current_token_ids,
                drafter_threshold,
                lowconf_threshold,
                max_spec_len,
                incr_len,
                action_script=script,
                default_action=default_action,
                prev_prefill_output=(
                    prev_prefill_output if reuse_selected_cache else None
                ),
            )
        except BaseException:
            if capacity_collector:
                restore_causal_profile(round_causal_profile)
            raise
        if capacity_collector and not preserve_causal_profile:
            restore_causal_profile(round_causal_profile)
        return result

    def evaluate_branch(script, decision_index, current_state):
        draft = run_draft(script, CONTINUE)
        branch_prefill_output = draft.pop("prefill_output", None)
        del branch_prefill_output
        required_prefix = tuple(script[:decision_index + 1])
        executed_prefix = tuple(
            draft["action_script"][:decision_index + 1]
        )
        if executed_prefix != required_prefix:
            raise RuntimeError(
                "strict greedy branch diverged before the requested decision"
            )
        decisions = draft["frontier_stats"].get("adaptive_decisions") or []
        decision_record = next(
            (
                record
                for record in decisions
                if bool(record.get("stop_available"))
                and int(record.get("oracle_script_index", -1)) == decision_index
            ),
            None,
        )
        elapsed_before_ms = float(
            current_state.get("elapsed_draft_ms", 0.0)
            if decision_record is None
            else decision_record.get(
                "oracle_elapsed_draft_ms",
                current_state.get("elapsed_draft_ms", 0.0),
            )
        )
        verification = evaluate_oracle_proposal_tokens(
            target_model,
            orig_model_inputs,
            current_token_ids,
            draft["proposal"],
            max_append_tokens=num_target_tokens - len(current_token_ids),
            eos_token_id=eos_token_id,
        )
        remaining_forward_ms = max(
            0.0, draft["draft_latency_ms"] - elapsed_before_ms
        )
        draft_overhead_ms = max(
            0.0, draft["draft_wall_time_ms"] - draft["draft_latency_ms"]
        )
        remaining_fraction = safe_div(
            remaining_forward_ms, draft["draft_latency_ms"]
        )
        local_draft_ms = (
            remaining_forward_ms + draft_overhead_ms * remaining_fraction
        )
        local_post_verify_ms = float(verification["post_verify_latency_ms"])
        branch_context_len = int(current_state.get(
            "context_len",
            orig_model_inputs["input_ids"].shape[1] + len(current_token_ids),
        ))
        predicted_verify_ms = predict_verifier_latency_ms(
            profile,
            branch_context_len,
            draft["proposal_len"],
        )
        target_len = int(current_state.get("proposal_length", 0))
        extension_events = [
            event
            for event in draft["frontier_stats"].get("extension_events") or []
            if int(event.get("from_len", -1)) >= target_len
        ]
        outer_path = format_outer_path(len(extension_events))
        return {
            "local_draft_ms": local_draft_ms,
            "local_post_verify_ms": local_post_verify_ms,
            "predicted_verify_ms": predicted_verify_ms,
            "measured_verify_ms": float(verification["verify_latency_ms"]),
            "local_cost_ms": (
                local_draft_ms + predicted_verify_ms + local_post_verify_ms
            ),
            "emitted_tokens": int(verification["emitted_len"]),
            "accepted_tokens": int(verification["accepted_len"]),
            "outer_path": outer_path,
            "proposal_len": int(draft["proposal_len"]),
            "extension_count": len(extension_events),
        }

    while len(current_token_ids) < num_target_tokens and eos_token_id not in current_token_ids:
        round_decisions = []
        if replay_rounds is not None:
            if round_id >= len(replay_rounds):
                raise RuntimeError(
                    f"strict greedy replay exhausted policy at round {round_id}"
                )
            action_script = list(replay_rounds[round_id]["actions"])
            final_draft = run_draft(
                tuple(action_script), None, reuse_selected_cache=True
            )
            if tuple(final_draft["action_script"]) != tuple(action_script):
                raise RuntimeError(
                    f"strict greedy replay diverged at problem={problem_id}, "
                    f"round={round_id}"
                )
            if getattr(args, "strict_greedy_record_diagnostics", True):
                replay_records = (
                    final_draft["frontier_stats"].get("adaptive_decisions") or []
                )
                for record in replay_records:
                    if not bool(record.get("stop_available")):
                        continue
                    decision_index = int(record.get("oracle_script_index", -1))
                    if decision_index < 0:
                        continue
                    target_len = int(record.get("target_len", 0))
                    refinement_step = int(record.get("step", 0))
                    context_len = int(record.get(
                        "context_len",
                        orig_model_inputs["input_ids"].shape[1]
                        + len(current_token_ids),
                    ))
                    draft_proposal = record.get("draft_proposal") or []
                    row = {
                        "sample_id": int(problem_id),
                        "round_id": int(round_id),
                        "decision_id": decision_index,
                        "block_id": max(
                            0,
                            (target_len - 1) // max(1, int(incr_len)),
                        ),
                        "refinement_step": refinement_step,
                        "context_len": context_len,
                        "prefix_output_tokens": int(len(current_token_ids)),
                        "draft_proposal": json.dumps(draft_proposal),
                        "state_key": build_oracle_state_key(
                            problem_id,
                            context_len,
                            target_len,
                            refinement_step,
                            draft_proposal,
                        ),
                        "accumulated_proposal_length": target_len,
                        "remaining_masks": int(record.get("remaining_masks", 0)),
                        "newly_committed": int(record.get("newly_unmasked", 0)),
                        "chosen_action": str(record.get("action")),
                        "features": json.dumps(record.get("features") or []),
                        "replay_only": 1,
                    }
                    args.strict_greedy_decision_rows.append(row)
                    round_decisions.append(row)
        else:
            action_script = []
            while True:
                try:
                    final_draft = run_draft(tuple(action_script), None)
                    break
                except OracleBranchRequired as required:
                    if required.decision_index != len(action_script):
                        raise RuntimeError(
                            "strict greedy replay consumed an inconsistent script"
                        )
                    decision_index = len(action_script)
                    stop_script, continue_script = one_action_rollout_scripts(
                        action_script
                    )
                    stop_branch = evaluate_branch(
                        stop_script,
                        decision_index,
                        required.state,
                    )
                    continue_branch = evaluate_branch(
                        continue_script,
                        decision_index,
                        required.state,
                    )
                    decision = choose_strict_greedy_action(
                        GreedyBranch(
                            stop_branch["local_cost_ms"],
                            stop_branch["emitted_tokens"],
                        ),
                        GreedyBranch(
                            continue_branch["local_cost_ms"],
                            continue_branch["emitted_tokens"],
                        ),
                        mean_verify_latency_ms=mean_verify_ms,
                        mean_tokens_per_verify=mean_tokens_per_verify,
                        epsilon_ms=args.strict_greedy_epsilon_ms,
                        baseline_action=CONTINUE,
                    )
                    chosen_branch = (
                        continue_branch
                        if capacity_collector
                        else (
                            stop_branch
                            if decision.action == STOP
                            else continue_branch
                        )
                    )
                    target_len = int(required.state.get("proposal_length", 0))
                    context_len = int(required.state.get(
                        "context_len",
                        orig_model_inputs["input_ids"].shape[1]
                        + len(current_token_ids),
                    ))
                    refinement_step = int(
                        required.state.get("refinement_step", 0)
                    )
                    draft_proposal = required.state.get("draft_proposal") or []
                    features = list(required.state.get("features") or [])
                    if capacity_collector and len(features) != len(feature_names):
                        raise RuntimeError(
                            "capacity collector received an unexpected feature vector: "
                            f"{len(features)} != {len(feature_names)}"
                        )
                    delta_draft_ms = (
                        continue_branch["local_draft_ms"]
                        - stop_branch["local_draft_ms"]
                    )
                    delta_verify_profile_ms = (
                        continue_branch["predicted_verify_ms"]
                        - stop_branch["predicted_verify_ms"]
                    )
                    delta_post_ms = (
                        continue_branch["local_post_verify_ms"]
                        - stop_branch["local_post_verify_ms"]
                    )
                    delta_future_verify_penalty_ms = (
                        decision.continue_penalty_ms - decision.stop_penalty_ms
                    )
                    delta_components_ms = (
                        delta_draft_ms
                        + delta_verify_profile_ms
                        + delta_post_ms
                        + delta_future_verify_penalty_ms
                    )
                    oracle_label = (
                        "tie"
                        if abs(decision.delta_j_ms) <= args.strict_greedy_epsilon_ms
                        else decision.action
                    )
                    row = {
                    "sample_id": int(problem_id),
                    "round_id": int(round_id),
                    "decision_id": int(decision_index),
                    "block_id": max(0, (target_len - 1) // max(1, int(incr_len))),
                    "refinement_step": refinement_step,
                    "context_len": context_len,
                    "prefix_output_tokens": int(len(current_token_ids)),
                    "draft_proposal": json.dumps(draft_proposal),
                    "features": json.dumps(features),
                    "state_key": build_oracle_state_key(
                        problem_id,
                        context_len,
                        target_len,
                        refinement_step,
                        draft_proposal,
                    ),
                    "replay_only": 0,
                    "remaining_masks": int(required.state.get("remaining_masks", 0)),
                    "newly_committed": int(required.state.get("newly_committed", 0)),
                    "accumulated_proposal_length": target_len,
                    "baseline_action": CONTINUE,
                    "stop_local_cost_ms": stop_branch["local_cost_ms"],
                    "stop_local_draft_ms": stop_branch["local_draft_ms"],
                    "stop_local_post_verify_ms": stop_branch[
                        "local_post_verify_ms"
                    ],
                    "stop_predicted_verify_ms": stop_branch["predicted_verify_ms"],
                    "stop_measured_verify_ms": stop_branch["measured_verify_ms"],
                    "stop_Y": stop_branch["emitted_tokens"],
                    "stop_outer_path": stop_branch["outer_path"],
                    "stop_next_verify_proposal_length": stop_branch["proposal_len"],
                    "continue_local_cost_ms": continue_branch["local_cost_ms"],
                    "continue_local_draft_ms": continue_branch[
                        "local_draft_ms"
                    ],
                    "continue_local_post_verify_ms": continue_branch[
                        "local_post_verify_ms"
                    ],
                    "continue_predicted_verify_ms": continue_branch[
                        "predicted_verify_ms"
                    ],
                    "continue_measured_verify_ms": continue_branch[
                        "measured_verify_ms"
                    ],
                    "continue_Y": continue_branch["emitted_tokens"],
                    "continue_outer_path": continue_branch["outer_path"],
                    "continue_next_verify_proposal_length": continue_branch["proposal_len"],
                    "mean_verify_latency_ms": mean_verify_ms,
                    "mean_tokens_per_verify": mean_tokens_per_verify,
                    "predicted_extra_calls_stop": decision.stop_extra_calls,
                    "predicted_extra_calls_continue": decision.continue_extra_calls,
                    "penalty_stop_ms": decision.stop_penalty_ms,
                    "penalty_continue_ms": decision.continue_penalty_ms,
                    "J_stop_ms": decision.stop_score_ms,
                    "J_continue_ms": decision.continue_score_ms,
                    "DeltaJ_ms": decision.delta_j_ms,
                    "DeltaY_tokens": (
                        continue_branch["emitted_tokens"]
                        - stop_branch["emitted_tokens"]
                    ),
                    "Delta_draft_ms": delta_draft_ms,
                    "Delta_verify_profile_ms": delta_verify_profile_ms,
                    "Delta_post_verify_ms": delta_post_ms,
                    "Delta_future_verify_penalty_ms": (
                        delta_future_verify_penalty_ms
                    ),
                    "DeltaJ_component_sum_ms": delta_components_ms,
                    "DeltaJ_identity_error_ms": (
                        decision.delta_j_ms - delta_components_ms
                    ),
                    "rho_profile_tokens_per_ms": rho_profile,
                    "DeltaG_profile_tokens": -rho_profile * decision.delta_j_ms,
                    "immediate_compute_difference_ms": (
                        continue_branch["local_cost_ms"]
                        - stop_branch["local_cost_ms"]
                    ),
                    "verifier_penalty_difference_ms": (
                        decision.continue_penalty_ms - decision.stop_penalty_ms
                    ),
                    "oracle_label": oracle_label,
                    "oracle_action": decision.action,
                    "executed_action": (
                        CONTINUE if capacity_collector else decision.action
                    ),
                    "chosen_action": decision.action,
                    "differs_from_baseline": int(decision.action != CONTINUE),
                    "changed_by_verifier_penalty": int(
                        decision.action != decision.immediate_action
                    ),
                    "tie_fallback_used": int(decision.tie_fallback_used),
                    "verify_to_extend_flip": int(
                        stop_branch["outer_path"] == "VERIFY"
                        and continue_branch["outer_path"].startswith("EXTEND")
                    ),
                    "selected_predicted_extra_calls": (
                        decision.stop_extra_calls
                        if decision.action == STOP
                        else decision.continue_extra_calls
                    ),
                    "selected_outer_path": chosen_branch["outer_path"],
                    }
                    row.update({
                        f"feature_{name}": float(value)
                        for name, value in zip(feature_names, features)
                    })
                    if getattr(args, "strict_greedy_record_diagnostics", True):
                        args.strict_greedy_decision_rows.append(row)
                    round_decisions.append(row)
                    action_script.append(
                        CONTINUE if capacity_collector else decision.action
                    )
                    if len(action_script) > 128:
                        raise RuntimeError(
                            "strict greedy oracle exceeded the refinement safety bound"
                        )
            final_draft = run_draft(
                tuple(action_script),
                None,
                reuse_selected_cache=True,
                preserve_causal_profile=capacity_collector,
            )
            if tuple(final_draft["action_script"]) != tuple(action_script):
                raise RuntimeError(
                    f"selected strict greedy path diverged with real KV reuse at "
                    f"problem={problem_id}, round={round_id}"
                )
        selected_round_actions.append({
            "round_id": int(round_id),
            "actions": list(action_script),
        })

        verification = evaluate_oracle_proposal_tokens(
            target_model,
            orig_model_inputs,
            current_token_ids,
            final_draft["proposal"],
            max_append_tokens=num_target_tokens - len(current_token_ids),
            eos_token_id=eos_token_id,
        )
        tokens_to_append = verification["tokens_to_append"]
        prefix_len = len(current_token_ids)
        current_token_ids.extend(tokens_to_append)
        final_draft_ms = float(final_draft["draft_wall_time_ms"])
        draft_latency_ms += final_draft_ms
        verify_latency_ms += float(verification["verify_latency_ms"])
        post_verify_latency_ms += float(verification["post_verify_latency_ms"])
        draft_passes += int(final_draft["draft_passes"])
        prev_prefill_output = final_draft["prefill_output"]
        if capacity_collector:
            args.adaptive_td_controller.observe_factual_verifier_call(
                len(tokens_to_append),
                float(verification["verify_latency_ms"]),
            )
            for decision_row in round_decisions:
                decision_row["selected_path_measured_verify_ms"] = float(
                    verification["verify_latency_ms"]
                )
                decision_row["continue_repeat_absolute_difference_ms"] = abs(
                    float(decision_row["continue_measured_verify_ms"])
                    - float(verification["verify_latency_ms"])
                )
            round_causal_profile = snapshot_causal_profile()
        stats_each_round.append({
            "target_tokens": tokens_to_append,
            "prefix_len": prefix_len,
            "spec_len": int(final_draft["proposal_len"]),
            "~draft_proposal": final_draft["proposal"],
            "accepted_len": int(verification["accepted_len"]),
            "acceptance_rate": safe_div(
                verification["accepted_len"], final_draft["proposal_len"]
            ),
            "num_forward_passes": int(final_draft["draft_passes"]),
            "draft_time_ms": final_draft_ms,
            "verify_time_ms": float(verification["verify_latency_ms"]),
            "post_verify_time_ms": float(verification["post_verify_latency_ms"]),
            "final_token": int(tokens_to_append[-1]),
            "bonus_token": None,
            "emitted_tokens": tokens_to_append,
            "strict_greedy_decisions": len(round_decisions),
        })
        if getattr(args, "strict_greedy_record_diagnostics", True):
            args.verifier_call_rows.append({
                "problem_id": int(problem_id),
                "mode": (
                    "strict_greedy_capacity_collector"
                    if capacity_collector
                    else "strict_greedy_local_oracle"
                ),
                "round_id": int(round_id),
                "context_length": int(
                    orig_model_inputs["input_ids"].shape[1] + prefix_len
                ),
                "proposal_length": int(final_draft["proposal_len"]),
                "accepted_tokens": int(verification["accepted_len"]),
                "emitted_tokens": len(tokens_to_append),
                "verify_latency_ms": float(verification["verify_latency_ms"]),
            })
        round_id += 1

    if replay_rounds is not None and round_id != len(replay_rounds):
        raise RuntimeError(
            f"strict greedy replay used {round_id}/{len(replay_rounds)} rounds"
        )
    if replay_rounds is None and args.strict_greedy_replay_data is None:
        args.strict_greedy_selected_policy[str(problem_id)] = selected_round_actions

    return {
        "current_token_ids": current_token_ids,
        "stats_each_round": stats_each_round,
        "global": {
            "draft_latency_ms": draft_latency_ms,
            "verify_latency_ms": verify_latency_ms,
            "post_verify_latency_ms": post_verify_latency_ms,
            "total_latency_ms": (
                draft_latency_ms + verify_latency_ms + post_verify_latency_ms
            ),
            "draft_passes": draft_passes,
            "rounds": round_id,
        },
    }


def run_truncated_global_oracle_problem(
    args,
    problem_id,
    target_model,
    dllm,
    orig_model_inputs,
    num_target_tokens,
    drafter_threshold,
    lowconf_threshold,
    max_spec_len,
    incr_len,
):
    search_started = time.perf_counter()
    baseline_prepass_started = time.perf_counter()
    eos_token_id = args.target_tokenizer.eos_token_id
    baseline_edges = []
    latency_profile = VerifierLatencyProfile()
    post_verify_samples = []
    prefix = tuple()

    while len(prefix) < num_target_tokens and eos_token_id not in prefix:
        draft = run_oracle_draft_trajectory(
            args,
            dllm,
            orig_model_inputs,
            list(prefix),
            drafter_threshold,
            lowconf_threshold,
            max_spec_len,
            incr_len,
        )
        verified = evaluate_oracle_proposal_tokens(
            target_model,
            orig_model_inputs,
            list(prefix),
            draft["proposal"],
            max_append_tokens=num_target_tokens - len(prefix),
            eos_token_id=eos_token_id,
        )
        latency_profile.add(
            orig_model_inputs["input_ids"].shape[1] + len(prefix),
            draft["proposal_len"],
            verified["accepted_len"],
            verified["verify_latency_ms"],
        )
        post_verify_samples.append(float(verified["post_verify_latency_ms"]))
        child = tuple(list(prefix) + verified["tokens_to_append"])
        if len(child) <= len(prefix):
            raise RuntimeError("baseline prepass did not advance the target prefix")
        edge = {
            **draft,
            "state": prefix,
            "child_state": child,
            "accepted_len": int(verified["accepted_len"]),
            "emitted_len": int(verified["emitted_len"]),
            "tokens_to_append": list(verified["tokens_to_append"]),
            "verify_latency_ms": float(verified["verify_latency_ms"]),
            "post_verify_latency_ms": float(verified["post_verify_latency_ms"]),
            "edge_latency_ms": (
                draft["draft_latency_ms"]
                + float(verified["verify_latency_ms"])
                + float(verified["post_verify_latency_ms"])
            ),
            "terminal": bool(
                eos_token_id in verified["tokens_to_append"]
                or len(child) >= num_target_tokens
            ),
            "method": "baseline_real_prepass",
            "candidate_source": "baseline_real_prepass",
            "blocks": 1 + len(
                (draft.get("frontier_stats") or {}).get("extension_events") or []
            ),
        }
        baseline_edges.append(edge)
        prefix = child

    baseline_prepass_wall_time_ms = (
        time.perf_counter() - baseline_prepass_started
    ) * 1000.0
    target_tokens = list(prefix)
    baseline_real_latency_ms = sum(
        float(edge["edge_latency_ms"]) for edge in baseline_edges
    )
    post_verify_median_ms = sorted(post_verify_samples)[
        len(post_verify_samples) // 2
    ] if post_verify_samples else 0.0
    baseline_tail_cache = {tuple(target_tokens): 0.0}
    baseline_edge_cache = {}
    baseline_suffix_ms = 0.0
    for edge in reversed(baseline_edges):
        baseline_edge_cache[edge["state"]] = edge
        baseline_suffix_ms += float(edge["edge_latency_ms"])
        baseline_tail_cache[edge["state"]] = baseline_suffix_ms
    baseline_tail_stats = {"calls": 0, "hits": 0, "misses": 0}
    round_cache = {}
    round_search_stats = {
        "trajectory_runs": 0,
        "trajectory_wall_time_ms": 0.0,
        "candidate_edges": 0,
        "lcp_validations": 0,
        "lcp_validation_model_time_ms": 0.0,
    }
    lcp_validation_remaining = int(args.truncated_lcp_validation_candidates)
    curve_states = {}

    def shortcut(prefix_tokens, proposal):
        return greedy_lcp_verification(
            target_tokens,
            len(prefix_tokens),
            proposal,
            max_append_tokens=num_target_tokens - len(prefix_tokens),
            eos_token_id=eos_token_id,
        )

    def make_edge(prefix_tokens, draft, proposal, proposal_len, passes, draft_ms, script, source):
        nonlocal lcp_validation_remaining
        verified = shortcut(prefix_tokens, proposal)
        predicted_verify_ms = latency_profile.estimate(
            orig_model_inputs["input_ids"].shape[1] + len(prefix_tokens),
            proposal_len,
            verified["accepted_len"],
        )
        if lcp_validation_remaining > 0:
            actual = evaluate_oracle_proposal_tokens(
                target_model,
                orig_model_inputs,
                list(prefix_tokens),
                proposal,
                max_append_tokens=num_target_tokens - len(prefix_tokens),
                eos_token_id=eos_token_id,
            )
            semantic_keys = ("accepted_len", "emitted_len", "tokens_to_append", "final_token")
            if any(actual[key] != verified[key] for key in semantic_keys):
                raise RuntimeError("greedy LCP shortcut disagrees with the real verifier")
            lcp_validation_remaining -= 1
            round_search_stats["lcp_validations"] += 1
            round_search_stats["lcp_validation_model_time_ms"] += float(
                actual["verify_latency_ms"] + actual["post_verify_latency_ms"]
            )
        child = tuple(list(prefix_tokens) + verified["tokens_to_append"])
        return {
            "state": tuple(prefix_tokens),
            "child_state": child,
            "proposal": list(proposal),
            "proposal_len": int(proposal_len),
            "draft_passes": int(passes),
            "draft_latency_ms": float(draft_ms),
            "verify_latency_ms": float(predicted_verify_ms),
            "post_verify_latency_ms": float(post_verify_median_ms),
            "edge_latency_ms": (
                float(draft_ms) + float(predicted_verify_ms) + float(post_verify_median_ms)
            ),
            "accepted_len": int(verified["accepted_len"]),
            "emitted_len": int(verified["emitted_len"]),
            "tokens_to_append": list(verified["tokens_to_append"]),
            "action_script": tuple(script),
            "terminal": bool(
                eos_token_id in verified["tokens_to_append"]
                or len(child) >= num_target_tokens
            ),
            "candidate_source": source,
            "blocks": 1 + len(
                (draft.get("frontier_stats") or {}).get("extension_events") or []
            ),
        }

    def expand_round(prefix_tokens):
        prefix_tokens = tuple(prefix_tokens)
        if prefix_tokens in round_cache:
            return round_cache[prefix_tokens]
        trajectory_prefixes = [tuple()]
        visited = set()
        edges = []
        decision_map = {}
        while trajectory_prefixes:
            fixed_script = trajectory_prefixes.pop()
            if fixed_script in visited:
                continue
            visited.add(fixed_script)
            if (
                args.global_oracle_max_states
                and len(visited) > args.global_oracle_max_states
            ):
                raise RuntimeError(
                    "Horizon oracle trajectory limit reached; no truncated result was written"
                )
            draft = run_oracle_draft_trajectory(
                args,
                dllm,
                orig_model_inputs,
                list(prefix_tokens),
                drafter_threshold,
                lowconf_threshold,
                max_spec_len,
                incr_len,
                action_script=fixed_script,
                default_action=CONTINUE,
            )
            round_search_stats["trajectory_runs"] += 1
            round_search_stats["trajectory_wall_time_ms"] += float(
                draft["draft_wall_time_ms"]
            )
            edges.append(make_edge(
                prefix_tokens,
                draft,
                draft["proposal"],
                draft["proposal_len"],
                draft["draft_passes"],
                draft["draft_latency_ms"],
                draft["action_script"],
                "natural_round_terminal",
            ))
            snapshots = draft["frontier_stats"].get("oracle_refinement_snapshots") or []
            for snapshot in snapshots:
                decision_index = int(snapshot.get("oracle_decision_index", -1))
                action_prefix = tuple(snapshot.get("oracle_action_prefix") or [])
                if decision_index < len(fixed_script) or not action_prefix:
                    continue
                stop_script = action_prefix[:-1] + (STOP,)
                decision_map[action_prefix[:-1]] = {
                    "step": int(snapshot["step"]),
                    "target_len": int(snapshot["target_len"]),
                    "masks_remaining": int(snapshot["masks_remaining"]),
                    "newly_committed": int(snapshot.get("newly_committed", 0)),
                    "committed_tokens": int(snapshot["committed_tokens"]),
                    "outer_action_if_stop": snapshot.get("outer_action_if_stop"),
                }
                if snapshot.get("outer_action_if_stop") == "extend":
                    trajectory_prefixes.append(stop_script)
                    continue
                edges.append(make_edge(
                    prefix_tokens,
                    draft,
                    [int(token_id) for token_id in snapshot["draft_proposal"]],
                    int(snapshot["target_len"]),
                    int(snapshot["draft_passes_elapsed"]),
                    float(snapshot["draft_latency_elapsed_ms"]),
                    stop_script,
                    "snapshot_stop_verify",
                ))
        best_by_key = {}
        for edge in edges:
            key = (edge["child_state"], tuple(edge["action_script"]))
            current = best_by_key.get(key)
            if current is None or edge["edge_latency_ms"] < current["edge_latency_ms"]:
                best_by_key[key] = edge
        result = list(best_by_key.values())
        round_search_stats["candidate_edges"] += len(result)
        round_cache[prefix_tokens] = result
        curve_states[prefix_tokens] = decision_map
        return result

    def baseline_value(prefix_tokens):
        prefix_tokens = tuple(prefix_tokens)
        baseline_tail_stats["calls"] += 1
        if prefix_tokens in baseline_tail_cache:
            baseline_tail_stats["hits"] += 1
            return baseline_tail_cache[prefix_tokens]
        baseline_tail_stats["misses"] += 1
        draft = run_oracle_draft_trajectory(
            args,
            dllm,
            orig_model_inputs,
            list(prefix_tokens),
            drafter_threshold,
            lowconf_threshold,
            max_spec_len,
            incr_len,
        )
        round_search_stats["trajectory_runs"] += 1
        round_search_stats["trajectory_wall_time_ms"] += float(
            draft["draft_wall_time_ms"]
        )
        edge = make_edge(
            prefix_tokens,
            draft,
            draft["proposal"],
            draft["proposal_len"],
            draft["draft_passes"],
            draft["draft_latency_ms"],
            draft["action_script"],
            "baseline_tail",
        )
        baseline_edge_cache[prefix_tokens] = edge
        suffix = 0.0 if edge["terminal"] else baseline_value(edge["child_state"])
        baseline_tail_cache[prefix_tokens] = float(edge["edge_latency_ms"]) + suffix
        return baseline_tail_cache[prefix_tokens]

    solved = solve_truncated_horizon(
        tuple(),
        args.truncated_global_horizon,
        expand_round,
        baseline_value,
    )
    selected_edges = list(solved["path"])
    if selected_edges and not selected_edges[-1]["terminal"]:
        tail_prefix = selected_edges[-1]["child_state"]
        baseline_value(tail_prefix)
        while tail_prefix in baseline_edge_cache:
            edge = baseline_edge_cache[tail_prefix]
            selected_edges.append(edge)
            if edge["terminal"]:
                break
            tail_prefix = edge["child_state"]

    predicted_latency_ms = sum(float(edge["edge_latency_ms"]) for edge in selected_edges)
    replay_started = time.perf_counter()
    replay_prefix = tuple()
    replay_edges = []
    for selected in selected_edges:
        draft = run_oracle_draft_trajectory(
            args,
            dllm,
            orig_model_inputs,
            list(replay_prefix),
            drafter_threshold,
            lowconf_threshold,
            max_spec_len,
            incr_len,
            action_script=selected["action_script"],
            default_action=CONTINUE,
        )
        if draft["proposal"] != selected["proposal"]:
            raise RuntimeError("selected-path replay produced a different draft proposal")
        actual = evaluate_oracle_proposal_tokens(
            target_model,
            orig_model_inputs,
            list(replay_prefix),
            draft["proposal"],
            max_append_tokens=num_target_tokens - len(replay_prefix),
            eos_token_id=eos_token_id,
        )
        replay_child = tuple(list(replay_prefix) + actual["tokens_to_append"])
        if replay_child != selected["child_state"]:
            raise RuntimeError("selected-path replay produced a different target prefix")
        replay_edges.append({
            **selected,
            "draft_passes": draft["draft_passes"],
            "draft_latency_ms": draft["draft_latency_ms"],
            "verify_latency_ms": float(actual["verify_latency_ms"]),
            "post_verify_latency_ms": float(actual["post_verify_latency_ms"]),
            "edge_latency_ms": (
                draft["draft_latency_ms"]
                + float(actual["verify_latency_ms"])
                + float(actual["post_verify_latency_ms"])
            ),
        })
        replay_prefix = replay_child
    real_replay_latency_ms = sum(float(edge["edge_latency_ms"]) for edge in replay_edges)
    if list(replay_prefix) != target_tokens:
        raise RuntimeError("Horizon oracle replay did not reproduce the greedy target sequence")
    replay_wall_time_ms = (time.perf_counter() - replay_started) * 1000.0

    curve_rows = []
    for (state, remaining_horizon), _ in solved["policy"].items():
        edges = round_cache[state]
        decisions = curve_states.get(state, {})
        edge_values = {}
        for edge in edges:
            if edge["terminal"]:
                suffix = 0.0
            elif remaining_horizon == 1:
                suffix = baseline_value(edge["child_state"])
            else:
                suffix = solved["memo"][(edge["child_state"], remaining_horizon - 1)]
            edge_values[id(edge)] = float(edge["edge_latency_ms"]) + suffix
        for script, decision in decisions.items():
            values = {}
            selected_by_action = {}
            for action in (STOP, CONTINUE):
                action_prefix = script + (action,)
                candidates = [
                    edge for edge in edges
                    if tuple(edge["action_script"][:len(action_prefix)]) == action_prefix
                ]
                if candidates:
                    best = min(candidates, key=lambda edge: edge_values[id(edge)])
                    values[action] = edge_values[id(best)]
                    selected_by_action[action] = best
            if STOP not in values or CONTINUE not in values:
                continue
            stop_edge = selected_by_action[STOP]
            continue_edge = selected_by_action[CONTINUE]
            global_action = min(values, key=values.get)
            block_start = len(script)
            while block_start > 0:
                previous = decisions.get(script[:block_start - 1])
                if (
                    previous is None
                    or int(previous["target_len"]) != int(decision["target_len"])
                ):
                    break
                block_start -= 1
            block_key = "".join(
                "S" if action == STOP else "C"
                for action in script[:block_start]
            )
            curve_rows.append({
                "problem_id": int(problem_id),
                "sample_id": int(problem_id),
                "prefix_len": len(state),
                "horizon": int(remaining_horizon),
                "block_key": block_key,
                "script": "".join("S" if x == STOP else "C" for x in script),
                "refinement_step": int(decision["step"]),
                "step": int(decision["step"]),
                "target_len": int(decision["target_len"]),
                "accumulated_proposal_length": int(decision["target_len"]),
                "block_id": max(0, (int(decision["target_len"]) - 1) // max(1, int(incr_len))),
                "remaining_masks": int(decision["masks_remaining"]),
                "newly_committed": int(decision["newly_committed"]),
                "cumulative_committed": int(decision["committed_tokens"]),
                "stop_global_cost_ms": values[STOP],
                "continue_global_cost_ms": values[CONTINUE],
                "stop_cost_h2": values[STOP],
                "continue_cost_h2": values[CONTINUE],
                "global_action": global_action,
                "stop_immediate_ms_per_token": float(stop_edge["edge_latency_ms"]) / max(1, stop_edge["emitted_len"]),
                "continue_immediate_ms_per_token": float(continue_edge["edge_latency_ms"]) / max(1, continue_edge["emitted_len"]),
                "immediate_cost_ms": float(stop_edge["edge_latency_ms"]),
                "baseline_tail_cost_ms": max(
                    0.0,
                    values[STOP] - float(stop_edge["edge_latency_ms"]),
                ),
                "outer_action_if_stop": decision.get("outer_action_if_stop"),
                "stop_accepted_tokens": int(stop_edge["accepted_len"]),
                "stop_emitted_tokens": int(stop_edge["emitted_len"]),
                "accepted_tokens": int(stop_edge["accepted_len"]),
                "emitted_tokens": int(stop_edge["emitted_len"]),
                "stop_draft_passes": int(stop_edge["draft_passes"]),
                "stop_future_verifier_calls": int(remaining_horizon),
                "horizon_future_verifier_calls": int(remaining_horizon),
            })
    curve_rows, delayed_rows, patience_rows = analyze_stop_depth_curves(
        curve_rows,
        args.global_oracle_epsilon_cost_ms,
    )
    for rows in (delayed_rows, patience_rows):
        for row in rows:
            row["problem_id"] = int(problem_id)

    search_wall_time_ms = (time.perf_counter() - search_started) * 1000.0 - replay_wall_time_ms
    method_label = f"truncated_global_h{args.truncated_global_horizon}"
    summary = {
        "problem_id": int(problem_id),
        "sample_id": int(problem_id),
        "method": method_label,
        "oracle_search_wall_time_ms": search_wall_time_ms,
        "predicted_selected_path_latency_ms": predicted_latency_ms,
        "real_replay_latency_ms": real_replay_latency_ms,
        "baseline_latency_ms": baseline_real_latency_ms,
        "speedup_vs_baseline": safe_div(baseline_real_latency_ms, real_replay_latency_ms),
        "generated_tokens": len(target_tokens),
        "dllm_forwards": sum(int(edge["draft_passes"]) for edge in replay_edges),
        "verifier_calls": len(replay_edges),
        "blocks": sum(int(edge.get("blocks", 1)) for edge in replay_edges),
        "speculative_rounds": len(replay_edges),
        "tokens_per_verifier_call": safe_div(len(target_tokens), len(replay_edges)),
        "memo_total_calls": solved["memo_calls"],
        "memo_unique_states": len(solved["memo"]),
        "memo_hits": solved["memo_hits"],
        "memo_hit_rate": solved["memo_hit_rate"],
        "baseline_tail_cache_hits": baseline_tail_stats["hits"],
        "baseline_tail_cache_misses": baseline_tail_stats["misses"],
        "number_delayed_benefit_events": len(delayed_rows),
        "patience1_failure_count": sum(row["patience"] == 1 and row["would_fail"] for row in patience_rows),
        "patience2_failure_count": sum(row["patience"] == 2 and row["would_fail"] for row in patience_rows),
        "patience3_failure_count": sum(row["patience"] == 3 and row["would_fail"] for row in patience_rows),
        "latency_prediction_error_percent": safe_div(
            real_replay_latency_ms - predicted_latency_ms,
            real_replay_latency_ms,
        ) * 100.0,
        "lcp_semantic_validations": round_search_stats["lcp_validations"],
        "trajectory_runs": round_search_stats["trajectory_runs"],
        "candidate_edges": round_search_stats["candidate_edges"],
        "verifier_latency_profile_observations": latency_profile.summary()[
            "observations"
        ],
        "output_token_hash": token_sequence_hash(target_tokens),
    }
    search_other_ms = max(
        0.0,
        search_wall_time_ms
        - baseline_prepass_wall_time_ms
        - round_search_stats["trajectory_wall_time_ms"]
        - round_search_stats["lcp_validation_model_time_ms"],
    )
    timing_rows = [
        {"problem_id": problem_id, "component": "search", "time_ms": search_wall_time_ms},
        {"problem_id": problem_id, "component": "selected_path_replay", "time_ms": replay_wall_time_ms},
        {"problem_id": problem_id, "component": "baseline_real_prepass", "time_ms": baseline_prepass_wall_time_ms},
        {"problem_id": problem_id, "component": "counterfactual_drafter", "time_ms": round_search_stats["trajectory_wall_time_ms"]},
        {"problem_id": problem_id, "component": "lcp_semantic_validation", "time_ms": round_search_stats["lcp_validation_model_time_ms"]},
        {"problem_id": problem_id, "component": "search_python_and_other", "time_ms": search_other_ms},
    ]
    cache_rows = [{
        "problem_id": problem_id,
        "memo_calls": solved["memo_calls"],
        "memo_unique_states": len(solved["memo"]),
        "memo_hits": solved["memo_hits"],
        "memo_hit_rate": solved["memo_hit_rate"],
        "baseline_tail_calls": baseline_tail_stats["calls"],
        "baseline_tail_hits": baseline_tail_stats["hits"],
        "baseline_tail_misses": baseline_tail_stats["misses"],
        "baseline_tail_entries": len(baseline_tail_cache),
        "round_cache_entries": len(round_cache),
        "estimated_cache_entries": (
            len(solved["memo"])
            + len(baseline_tail_cache)
            + len(baseline_edge_cache)
            + len(round_cache)
        ),
        "estimated_cache_memory_bytes": estimate_cache_bytes(
            solved["memo"],
            solved["policy"],
            baseline_tail_cache,
            baseline_edge_cache,
            round_cache,
        ),
    }]
    trace_rows = []
    for round_id, edge in enumerate(replay_edges):
        trace_rows.append({
            "problem_id": problem_id,
            "round_id": round_id,
            "prefix_len": len(edge["state"]),
            "action_script": "".join("S" if x == STOP else "C" for x in edge["action_script"]),
            "proposal_len": edge["proposal_len"],
            "accepted_len": edge["accepted_len"],
            "emitted_len": edge["emitted_len"],
            "draft_passes": edge["draft_passes"],
            "draft_latency_ms": edge["draft_latency_ms"],
            "verify_latency_ms": edge["verify_latency_ms"],
            "post_verify_latency_ms": edge["post_verify_latency_ms"],
            "round_latency_ms": edge["edge_latency_ms"],
            "candidate_source": edge["candidate_source"],
        })
    output_prefix = method_label
    outputs = (
        (f"{output_prefix}_sample_summary.csv", [summary]),
        (f"{output_prefix}_block_curves.csv", curve_rows),
        (f"{output_prefix}_delayed_benefit_events.csv", delayed_rows),
        (f"{output_prefix}_patience_analysis.csv", patience_rows),
        (f"{output_prefix}_optimal_trace.csv", trace_rows),
        (f"{output_prefix}_cache_stats.csv", cache_rows),
        (f"{output_prefix}_timing_profile.csv", timing_rows),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for filename, rows in outputs:
        if rows:
            columns = sorted({key for row in rows for key in row})
            append_csv_rows(os.path.join(args.output_dir, filename), columns, rows)
    stats_each_round = [{
        "target_tokens": edge["tokens_to_append"],
        "prefix_len": len(edge["state"]),
        "spec_len": edge["proposal_len"],
        "~draft_proposal": edge["proposal"],
        "accepted_len": edge["accepted_len"],
        "acceptance_rate": safe_div(edge["accepted_len"], edge["proposal_len"]),
        "num_forward_passes": edge["draft_passes"],
        "draft_time_ms": edge["draft_latency_ms"],
        "verify_time_ms": edge["verify_latency_ms"],
        "post_verify_time_ms": edge["post_verify_latency_ms"],
        "final_token": edge["tokens_to_append"][-1],
        "bonus_token": None,
        "emitted_tokens": edge["tokens_to_append"],
        "frontier_stats": None,
    } for edge in replay_edges]
    return {
        "current_token_ids": target_tokens,
        "stats_each_round": stats_each_round,
        "summary": summary,
        "global": {
            "draft_latency_ms": sum(edge["draft_latency_ms"] for edge in replay_edges),
            "verify_latency_ms": sum(edge["verify_latency_ms"] for edge in replay_edges),
            "post_verify_latency_ms": sum(edge["post_verify_latency_ms"] for edge in replay_edges),
            "total_latency_ms": real_replay_latency_ms,
            "draft_passes": summary["dllm_forwards"],
            "rounds": summary["verifier_calls"],
        },
    }


def run_global_oracle_problem(
    args,
    problem_id,
    target_model,
    dllm,
    orig_model_inputs,
    num_target_tokens,
    drafter_threshold,
    lowconf_threshold,
    max_spec_len,
    incr_len,
):
    problem_started = time.perf_counter()
    queue = [tuple()]
    prefix_by_length = {0: tuple()}
    expansions = {}
    edges_by_state = {}
    decision_states = {}
    terminal_length = None
    total_replays = 0
    dp_calls = 0
    memo_hits = 0

    while queue:
        prefix = queue.pop(0)
        state = len(prefix)
        dp_calls += 1
        if state in expansions:
            memo_hits += 1
            continue
        if terminal_length is not None and state >= terminal_length:
            continue
        expansion = enumerate_global_oracle_round_edges(
            args,
            problem_id,
            target_model,
            dllm,
            orig_model_inputs,
            list(prefix),
            num_target_tokens,
            drafter_threshold,
            lowconf_threshold,
            max_spec_len,
            incr_len,
        )
        expansions[state] = expansion
        total_replays += int(expansion["replays"])
        state_edges = []
        for edge in expansion["edges"]:
            child = edge["child_tokens"]
            child_state = len(child)
            existing = prefix_by_length.get(child_state)
            if existing is not None and existing != child:
                raise RuntimeError(
                    "Greedy lossless invariant failed: two oracle branches produced "
                    f"different target prefixes of length {child_state}"
                )
            prefix_by_length[child_state] = child
            state_edges.append(edge)
            reached_terminal = (
                args.target_tokenizer.eos_token_id in edge["tokens_to_append"]
                or child_state >= num_target_tokens
            )
            if reached_terminal:
                if terminal_length is None:
                    terminal_length = child_state
                elif terminal_length != child_state:
                    raise RuntimeError(
                        "oracle branches reached different greedy terminal lengths"
                    )
            else:
                queue.append(child)
        edges_by_state[state] = state_edges
        for script, decision in expansion["decision_states"].items():
            decision_states[(state, script)] = decision
        if len(expansions) % args.global_oracle_log_interval == 0:
            logging.info(
                "[Global oracle problem %s] expanded=%s queued=%s replays=%s",
                problem_id,
                len(expansions),
                len(queue),
                total_replays,
            )

    if terminal_length is None:
        raise RuntimeError("global oracle did not reach EOS or max_new_tokens")
    pruned_edges = {}
    solvable = {terminal_length}
    for state in sorted(edges_by_state, reverse=True):
        valid = [
            edge for edge in edges_by_state[state]
            if int(edge["child_state"]) in solvable
        ]
        if valid:
            pruned_edges[state] = valid
            solvable.add(state)
    if 0 not in solvable:
        raise RuntimeError("global oracle root cannot reach the terminal state")

    solved = solve_canonical_oracle_graph(pruned_edges, terminal_length)
    path_summaries = {
        policy: summarize_policy_path(path)
        for policy, path in solved["paths"].items()
    }
    global_path = solved["paths"]["global"]
    failfast_path = solved["paths"]["failfast"]
    current_token_ids = list(prefix_by_length[terminal_length])
    if [edge["tokens_to_append"] for edge in global_path] != [
        edge["tokens_to_append"] for edge in failfast_path
    ]:
        global_output = [token for edge in global_path for token in edge["tokens_to_append"]]
        failfast_output = [token for edge in failfast_path for token in edge["tokens_to_append"]]
        if global_output != failfast_output:
            raise RuntimeError("global and FailFast paths did not reproduce identical output")

    optimal_states = {int(edge["state"]) for edge in global_path}
    failfast_states = {int(edge["state"]) for edge in failfast_path}
    node_rows = []

    def suffix_path(state):
        path = []
        current = int(state)
        while current != terminal_length:
            edge = solved["policies"]["global"][current]
            path.append(edge)
            current = int(edge["child_state"])
        return path

    def controllable_decision(edge, index):
        decisions = [
            item for item in edge["decision_trace"]
            if item.get("stop_available")
        ]
        return decisions[index] if index < len(decisions) else {}

    for (state, script), decision in sorted(
        decision_states.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        edges = pruned_edges.get(state, [])
        action_values = {}
        action_edges = {}
        for action in (STOP, CONTINUE):
            prefix_script = script + (action,)
            candidates = [
                edge for edge in edges
                if tuple(edge["action_script"][:len(prefix_script)]) == prefix_script
            ]
            if not candidates:
                continue
            best_edge = min(
                candidates,
                key=lambda edge: (
                    float(edge["edge_latency_ms"])
                    + solved["values"]["global"][int(edge["child_state"])],
                    int(edge["draft_passes"]),
                ),
            )
            action_edges[action] = best_edge
            action_values[action] = (
                float(best_edge["edge_latency_ms"])
                + solved["values"]["global"][int(best_edge["child_state"])]
            )
        if STOP not in action_values or CONTINUE not in action_values:
            continue
        global_action = min(action_values, key=action_values.get)
        stop_edge = action_edges[STOP]
        continue_edge = action_edges[CONTINUE]
        stop_immediate = float(stop_edge["edge_latency_ms"]) / max(
            1, int(stop_edge["emitted_len"])
        )
        continue_immediate = float(continue_edge["edge_latency_ms"]) / max(
            1, int(continue_edge["emitted_len"])
        )
        myopic_action = STOP if stop_immediate <= continue_immediate else CONTINUE
        delayed = global_action == CONTINUE and myopic_action == STOP
        target_len = int(decision.get("proposal_length", 0))
        stop_suffix = suffix_path(stop_edge["child_state"])
        continue_suffix = suffix_path(continue_edge["child_state"])
        stop_decision = controllable_decision(stop_edge, len(script))
        continue_outer_action = (
            "extend"
            if any(
                int(event.get("from_len", -1)) == target_len
                for event in continue_edge.get("extension_events", [])
            )
            else "verify"
        )
        block_start = len(script)
        while block_start > 0:
            parent_script = script[:block_start - 1]
            parent_state = decision_states.get((state, parent_script))
            if (
                parent_state is None
                or int(parent_state.get("proposal_length", 0)) != target_len
            ):
                break
            block_start -= 1
        block_key = "".join(
            "S" if action == STOP else "C" for action in script[:block_start]
        )
        row = {
            "problem_id": int(problem_id),
            "prefix_len": int(state),
            "script": "".join("S" if x == STOP else "C" for x in script),
            "decision_index": len(script),
            "refinement_step": int(decision.get("refinement_step", 0)),
            "target_len": target_len,
            "block_id": max(0, (target_len - 1) // max(1, int(incr_len))),
            "block_key": block_key,
            "stop_global_cost_ms": action_values[STOP],
            "continue_global_cost_ms": action_values[CONTINUE],
            "stop_immediate_ms_per_token": stop_immediate,
            "continue_immediate_ms_per_token": continue_immediate,
            "global_action": global_action,
            "myopic_action": myopic_action,
            "global_myopic_disagree": int(global_action != myopic_action),
            "delayed_benefit": int(delayed),
            "global_regret_if_myopic_ms": (
                action_values[myopic_action] - action_values[global_action]
            ),
            "on_global_prefix": int(state in optimal_states),
            "on_failfast_prefix": int(state in failfast_states),
            "outer_action_if_stop": stop_decision.get(
                "outer_action_after_stop"
            ),
            "outer_action_after_natural_continue": continue_outer_action,
            "stop_accepted_tokens": int(stop_edge["accepted_len"]),
            "continue_accepted_tokens": int(continue_edge["accepted_len"]),
            "stop_emitted_tokens": int(stop_edge["emitted_len"]),
            "continue_emitted_tokens": int(continue_edge["emitted_len"]),
            "stop_draft_passes": int(stop_edge["draft_passes"]),
            "continue_draft_passes": int(continue_edge["draft_passes"]),
            "stop_future_verifier_calls": 1 + len(stop_suffix),
            "continue_future_verifier_calls": 1 + len(continue_suffix),
            "natural_termination": 0,
        }
        node_rows.append(row)

    curve_groups = {}
    for row in node_rows:
        curve_groups.setdefault(
            (row["prefix_len"], row["block_key"], row["target_len"]), []
        ).append(row)
    natural_rows = []
    for (_, _, _), rows in curve_groups.items():
        last = max(rows, key=lambda item: item["refinement_step"])
        next_script = last["script"] + "C"
        same_block_continuation = any(
            row["script"] == next_script
            and row["target_len"] == last["target_len"]
            for row in rows
        )
        if same_block_continuation:
            continue
        natural_rows.append({
            **last,
            "script": next_script,
            "decision_index": int(last["decision_index"]) + 1,
            "refinement_step": int(last["refinement_step"]) + 1,
            "stop_global_cost_ms": float(last["continue_global_cost_ms"]),
            "continue_global_cost_ms": math.inf,
            "stop_immediate_ms_per_token": float(
                last["continue_immediate_ms_per_token"]
            ),
            "continue_immediate_ms_per_token": math.inf,
            "global_action": STOP,
            "myopic_action": STOP,
            "global_myopic_disagree": 0,
            "delayed_benefit": 0,
            "global_regret_if_myopic_ms": 0.0,
            "stop_accepted_tokens": int(last["continue_accepted_tokens"]),
            "stop_emitted_tokens": int(last["continue_emitted_tokens"]),
            "stop_draft_passes": int(last["continue_draft_passes"]),
            "stop_future_verifier_calls": int(
                last["continue_future_verifier_calls"]
            ),
            "outer_action_if_stop": last[
                "outer_action_after_natural_continue"
            ],
            "natural_termination": 1,
        })
    node_rows.extend(natural_rows)
    node_rows, delayed_rows, patience_rows = analyze_stop_depth_curves(
        node_rows,
        args.global_oracle_epsilon_cost_ms,
    )
    for row in delayed_rows:
        row["problem_id"] = int(problem_id)
    for row in patience_rows:
        row["problem_id"] = int(problem_id)

    def public_edge(edge, policy=None, round_id=None):
        return {
            key: (
                json.dumps(value)
                if isinstance(value, (list, tuple, dict))
                else value
            )
            for key, value in {
                **edge,
                "policy": policy,
                "round_id": round_id,
            }.items()
            if key not in {"child_tokens"}
        }

    edge_rows = [
        public_edge(edge)
        for state in sorted(pruned_edges)
        for edge in pruned_edges[state]
    ]
    optimal_rounds = [
        public_edge(edge, "global", round_id)
        for round_id, edge in enumerate(global_path)
    ]
    failfast_trace = [
        public_edge(edge, "failfast", round_id)
        for round_id, edge in enumerate(failfast_path)
    ]
    optimal_trace = []
    for round_id, edge in enumerate(global_path):
        previous_elapsed_ms = 0.0
        child_future_ms = solved["values"]["global"][int(edge["child_state"])]
        for decision_id, decision in enumerate(edge["decision_trace"]):
            elapsed_ms = float(decision.get("elapsed_draft_ms", previous_elapsed_ms))
            optimal_trace.append({
                "problem_id": int(problem_id),
                "round_id": round_id,
                "prefix_len": int(edge["state"]),
                "block_id": max(
                    0,
                    (int(decision.get("target_len", 0)) - 1)
                    // max(1, int(incr_len)),
                ),
                "decision_id": decision_id,
                "refinement_step": int(decision.get("step", 0)),
                "remaining_masks": int(decision.get("remaining_masks", 0)),
                "newly_committed_tokens": int(decision.get("newly_unmasked", 0)),
                "action": decision.get("action"),
                "outer_decision_after_stop": decision.get(
                    "outer_action_after_stop"
                ),
                "cumulative_proposal_length": int(
                    decision.get("target_len", 0)
                ),
                "next_state_type": (
                    "outer" if decision.get("action") == STOP else "inner"
                ),
                "branch_immediate_latency_ms": max(
                    0.0, elapsed_ms - previous_elapsed_ms
                ),
                "global_cost_to_go_ms": max(
                    0.0,
                    float(edge["edge_latency_ms"]) - elapsed_ms + child_future_ms,
                ),
                "optimal_future_verifier_calls": 1 + len(
                    suffix_path(edge["child_state"])
                ),
                "optimal_future_dllm_forwards": int(edge["draft_passes"]) + sum(
                    int(item["draft_passes"])
                    for item in suffix_path(edge["child_state"])
                ),
            })
            previous_elapsed_ms = elapsed_ms
    search_wall_time_ms = (time.perf_counter() - problem_started) * 1000.0
    global_summary = path_summaries["global"]
    failfast_summary = path_summaries["failfast"]
    summary_row = {
        "problem_id": int(problem_id),
        "generated_tokens": terminal_length,
        "output_token_hash": token_sequence_hash(current_token_ids),
        "baseline_total_latency_ms": failfast_summary["total_latency_ms"],
        "oracle_optimal_latency_ms": global_summary["total_latency_ms"],
        "oracle_speedup": safe_div(
            failfast_summary["total_latency_ms"],
            global_summary["total_latency_ms"],
        ),
        "baseline_dllm_forwards": failfast_summary["draft_passes"],
        "oracle_dllm_forwards": global_summary["draft_passes"],
        "baseline_verifier_calls": failfast_summary["rounds"],
        "oracle_verifier_calls": global_summary["rounds"],
        "baseline_blocks": sum(int(edge["blocks"]) for edge in failfast_path),
        "oracle_blocks": sum(int(edge["blocks"]) for edge in global_path),
        "baseline_rounds": failfast_summary["rounds"],
        "oracle_rounds": global_summary["rounds"],
        "baseline_tokens_per_second": safe_div(
            terminal_length * 1000.0, failfast_summary["total_latency_ms"]
        ),
        "oracle_tokens_per_second": safe_div(
            terminal_length * 1000.0, global_summary["total_latency_ms"]
        ),
        "oracle_search_wall_time_ms": search_wall_time_ms,
        "unique_dp_states": len(pruned_edges) + 1,
        "dp_calls": dp_calls,
        "memo_hits": memo_hits,
        "memo_hit_rate": safe_div(memo_hits, dp_calls),
        "oracle_replays": total_replays,
        "num_delayed_benefit_blocks": len({
            (row["prefix_len"], row["block_key"]) for row in delayed_rows
        }),
        "num_local_minimum_traps": len(delayed_rows),
        "num_local_minima": sum(
            int(row["is_local_minimum"]) for row in node_rows
        ),
        "patience1_failures": sum(
            row["patience"] == 1 and row["would_fail"] for row in patience_rows
        ),
        "patience2_failures": sum(
            row["patience"] == 2 and row["would_fail"] for row in patience_rows
        ),
        "patience3_failures": sum(
            row["patience"] == 3 and row["would_fail"] for row in patience_rows
        ),
        "global_never_slower_validation": int(
            global_summary["total_latency_ms"]
            <= failfast_summary["total_latency_ms"] + 1e-6
        ),
        "cache_mode": "disabled_path_independent_replay",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for filename, rows in (
        ("global_oracle_edges.csv", edge_rows),
        ("global_oracle_nodes.csv", node_rows),
        ("global_oracle_block_curves.csv", node_rows),
        ("global_oracle_delayed_benefit_events.csv", delayed_rows),
        ("global_oracle_patience_analysis.csv", patience_rows),
        ("global_oracle_optimal_trace.csv", optimal_trace),
        ("global_oracle_optimal_rounds.csv", optimal_rounds),
        ("global_oracle_failfast_trace.csv", failfast_trace),
        ("global_oracle_problem_summary.csv", [summary_row]),
    ):
        if rows:
            columns = sorted({key for row in rows for key in row})
            append_csv_rows(os.path.join(args.output_dir, filename), columns, rows)

    stats_each_round = []
    for edge in global_path:
        stats_each_round.append({
            "target_tokens": edge["tokens_to_append"],
            "prefix_len": int(edge["state"]),
            "spec_len": int(edge["proposal_len"]),
            "~draft_proposal": edge["draft_proposal"],
            "accepted_len": int(edge["accepted_len"]),
            "acceptance_rate": safe_div(
                edge["accepted_len"], edge["proposal_len"]
            ),
            "num_forward_passes": int(edge["draft_passes"]),
            "draft_time_ms": float(edge["draft_latency_ms"]),
            "verify_time_ms": float(edge["verify_latency_ms"]),
            "post_verify_time_ms": float(edge["post_verify_latency_ms"]),
            "final_token": int(edge["final_token"]),
            "bonus_token": None,
            "emitted_tokens": edge["tokens_to_append"],
            "frontier_stats": None,
        })
    return {
        "current_token_ids": current_token_ids,
        "stats_each_round": stats_each_round,
        "summary": summary_row,
        "global": global_summary,
        "failfast": failfast_summary,
    }


def choose_causal_oracle_proposal(
    args,
    problem_id,
    mode,
    round_id,
    target_model,
    orig_model_inputs,
    current_token_ids,
    frontier_stats,
    physical_draft_latency_ms,
    physical_draft_passes,
    physical_forward_latency_ms,
    factual_proposal,
):
    snapshots, fallback_used = prepare_causal_oracle_snapshots(
        frontier_stats,
        factual_proposal,
        physical_draft_passes,
        physical_forward_latency_ms,
    )
    context_len = int(orig_model_inputs["input_ids"].shape[1] + len(current_token_ids))
    rows = []
    total_draft_overhead_ms = max(
        0.0,
        float(physical_draft_latency_ms) - float(physical_forward_latency_ms),
    )
    probe_start = time.perf_counter()
    for candidate_index, snapshot in enumerate(snapshots):
        proposal = [int(token_id) for token_id in snapshot["draft_proposal"]]
        accepted_len, emitted_len, verify_ms, accept_check_ms = evaluate_oracle_proposal(
            target_model,
            orig_model_inputs,
            current_token_ids,
            proposal,
        )
        draft_latency_ms = float(snapshot["draft_latency_elapsed_ms"])
        draft_passes = int(snapshot["draft_passes_elapsed"])
        estimated_draft_overhead_ms = (
            total_draft_overhead_ms
            * draft_passes
            / max(1, int(physical_draft_passes))
        )
        effective_draft_latency_ms = draft_latency_ms + estimated_draft_overhead_ms
        total_latency_ms = effective_draft_latency_ms + verify_ms + accept_check_ms
        rows.append({
            "problem_id": problem_id,
            "mode": mode,
            "round_id": round_id,
            "context_len": context_len,
            "candidate_index": candidate_index,
            "candidate_source": snapshot["candidate_source"],
            "step": int(snapshot["step"]),
            "target_len": int(snapshot["target_len"]),
            "draft_passes_elapsed": draft_passes,
            "draft_latency_elapsed_ms": draft_latency_ms,
            "estimated_draft_overhead_ms": estimated_draft_overhead_ms,
            "effective_draft_latency_ms": effective_draft_latency_ms,
            "draft_proposal": json.dumps(proposal),
            "accepted_len_if_stop": accepted_len,
            "emitted_len_if_stop": emitted_len,
            "counterfactual_verify_latency_ms": verify_ms,
            "counterfactual_accept_check_latency_ms": accept_check_ms,
            "counterfactual_total_latency_ms": total_latency_ms,
            "counterfactual_ms_per_output_token": total_latency_ms / max(1, emitted_len),
            "selected": False,
            "_snapshot": snapshot,
            "_proposal": proposal,
        })
    probe_wall_time_ms = (time.perf_counter() - probe_start) * 1000.0
    if args.causal_oracle_future_cost_profile:
        if not hasattr(args, "causal_oracle_loaded_future_cost_profile"):
            args.causal_oracle_loaded_future_cost_profile = load_future_cost_profile(
                args.causal_oracle_future_cost_profile
            )
        future_stats = stats_for_problem(
            args.causal_oracle_loaded_future_cost_profile,
            problem_id,
        )
        selected, action_trace = select_greedy_future_adjusted_candidate(
            rows,
            future_stats,
        )
        selected["_oracle_cost_model"] = "failfast_future_round_adjusted_greedy"
        selected["_future_stats"] = future_stats
    else:
        selected = min(
            rows,
            key=lambda row: (
                row["counterfactual_ms_per_output_token"],
                row["draft_passes_elapsed"],
                row["target_len"],
            ),
        )
        selected["selected"] = True
        selected_passes = int(selected["draft_passes_elapsed"])
        action_trace = []
        stop_recorded = False
        for row in sorted(
            rows,
            key=lambda item: (
                item["draft_passes_elapsed"],
                item["target_len"],
                item["candidate_index"],
            ),
        ):
            if row is selected:
                action = "stop"
                stop_recorded = True
            elif not stop_recorded and row["draft_passes_elapsed"] < selected_passes:
                action = "continue"
            else:
                action = "not_reached"
            row["oracle_action"] = action
            if action != "not_reached":
                action_trace.append({
                    "candidate_index": int(row["candidate_index"]),
                    "draft_passes": int(row["draft_passes_elapsed"]),
                    "target_len": int(row["target_len"]),
                    "action": action,
                })
        selected["_oracle_cost_model"] = "local_ms_per_output_token"
        selected["_future_stats"] = {
            "tokens_per_round": 0.0,
            "draft_ms_per_round": 0.0,
            "verify_ms_per_round": 0.0,
            "post_verify_ms_per_round": 0.0,
        }
    selected["_action_trace"] = action_trace
    selected["_num_candidates"] = len(rows)
    selected["_snapshot_fallback_used"] = int(fallback_used)
    csv_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    append_csv_rows(
        os.path.join(args.output_dir, "causal_oracle_candidates.csv"),
        CAUSAL_ORACLE_CANDIDATE_COLUMNS,
        csv_rows,
    )
    return selected, probe_wall_time_ms


def append_causal_oracle_decision(args, row):
    append_csv_rows(
        os.path.join(args.output_dir, "causal_oracle_decisions.csv"),
        CAUSAL_ORACLE_DECISION_COLUMNS,
        [row],
    )


def append_bucket_oracle_rows(
    args,
    problem_id,
    mode,
    round_id,
    target_model,
    orig_model_inputs,
    current_token_ids,
    frontier_stats,
    shared_post_verify_overhead_ms,
):
    if not args.collect_bucket_oracle:
        return
    snapshots = (frontier_stats or {}).get("oracle_refinement_snapshots") or []
    rows = []
    for snapshot in snapshots:
        draft_proposal = [int(token_id) for token_id in snapshot["draft_proposal"]]
        accepted_len, emitted_len, verify_ms, accept_check_ms = evaluate_oracle_proposal(
            target_model,
            orig_model_inputs,
            current_token_ids,
            draft_proposal,
        )
        rows.append({
            "problem_id": problem_id,
            "mode": mode,
            "round_id": round_id,
            "context_len": int(
                orig_model_inputs["input_ids"].shape[1] + len(current_token_ids)
            ),
            "step": snapshot.get("step"),
            "target_len": snapshot.get("target_len"),
            "draft_passes_elapsed": snapshot.get("draft_passes_elapsed"),
            "draft_latency_elapsed_ms": snapshot.get("draft_latency_elapsed_ms"),
            "masks_remaining": snapshot.get("masks_remaining"),
            "committed_tokens": snapshot.get("committed_tokens"),
            "filled_tokens": snapshot.get("filled_tokens"),
            "draft_proposal": json.dumps(draft_proposal),
            "accept_probabilities": json.dumps(
                snapshot.get("accept_probabilities") or []
            ),
            "predicted_expected_output": snapshot.get("predicted_expected_output"),
            "predicted_next_gain": snapshot.get("predicted_next_gain"),
            "predicted_stop_ms_per_output": snapshot.get(
                "predicted_stop_ms_per_output"
            ),
            "predicted_continue_ms_per_output": snapshot.get(
                "predicted_continue_ms_per_output"
            ),
            "predicted_should_continue": snapshot.get("predicted_should_continue"),
            "predicted_gain_source": snapshot.get("predicted_gain_source"),
            "gain_bucket_count": snapshot.get("gain_bucket_count"),
            "gain_bucket_weight": snapshot.get("gain_bucket_weight"),
            "calibration_tokens": snapshot.get("calibration_tokens"),
            "adaptive_policy_action": snapshot.get("adaptive_policy_action"),
            "adaptive_policy_reason": snapshot.get("adaptive_policy_reason"),
            "adaptive_stop_probability": snapshot.get(
                "adaptive_stop_probability"
            ),
            "adaptive_advantage_mean": snapshot.get("adaptive_advantage_mean"),
            "adaptive_advantage_risk": snapshot.get("adaptive_advantage_risk"),
            "adaptive_q_stop_mean": snapshot.get("adaptive_q_stop_mean"),
            "adaptive_q_continue_mean": snapshot.get("adaptive_q_continue_mean"),
            "adaptive_rho_tokens_per_ms": snapshot.get(
                "adaptive_rho_tokens_per_ms"
            ),
            "adaptive_stop_available": snapshot.get("adaptive_stop_available"),
            "accepted_len_if_stop": accepted_len,
            "emitted_len_if_stop": emitted_len,
            "actual_verify_latency_ms": verify_ms,
            "actual_accept_check_latency_ms": accept_check_ms,
            "actual_shared_post_verify_overhead_ms": shared_post_verify_overhead_ms,
            "actual_post_verify_latency_ms": (
                accept_check_ms + shared_post_verify_overhead_ms
            ),
        })
    append_csv_rows(
        os.path.join(args.output_dir, "bucket_oracle_snapshots.csv"),
        BUCKET_ORACLE_SNAPSHOT_COLUMNS,
        rows,
    )

apply_mode_settings(args)
args.target_model_name_clean = args.target_model_name.split("/", 1)[1]
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
)

args.benchmark_drafter_configs = build_benchmark_drafter_configs(args)
populate_dataset(args)

args.latency = {
    "vLLM_A6000": {
        "draft_fwd_pass": 6.1,
        "target_tpt": {
            "Qwen2.5-3B-Instruct": 7.2,
            "Qwen2.5-7B-Instruct": 13.5,
            "Qwen2.5-14B-Instruct": 24.7,
            "Qwen2.5-32B-Instruct": 52.6,
        },
    },
}

target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
args.target_tokenizer = target_tokenizer

_original_forward_dict = {}

def create_forward_hook(model_name, times_dict):
    def forward_hook(module, input, output):
        pass
    return forward_hook

times_dict = {"draft": [], "verify": []}

if not args.read_pickle:
    logging.info(f"{Colors.BOLD}=== Loading target model: {args.target_model_name} ==={Colors.RESET}")
    try:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model_name,
            torch_dtype="auto",
            device_map={"": args.target_device},
            attn_implementation="sdpa"
        )
    except Exception as e:
        msg = str(e).lower()
        if isinstance(e, RuntimeError) and ("out of memory" in msg or 'cuda' in msg) or isinstance(e, torch.cuda.OutOfMemoryError):
            logging.error(f"{Colors.RED}CUDA OutOfMemory while loading target model {args.target_model_name}: {e}{Colors.RESET}")
            sys.exit(1)
        raise

    dllm = None
    dllm_tokenizer = target_tokenizer
    if any(config[0] == "dllm" for config in args.benchmark_drafter_configs.values()):
        import transformers.modeling_rope_utils as rope_utils
        import transformers.modeling_utils as modeling_utils
        
        has_patched_rope = False
        original_tied_weights_fn = getattr(modeling_utils.PreTrainedModel, 'get_expanded_tied_weights_keys', None)
        
        if hasattr(rope_utils, 'ROPE_INIT_FUNCTIONS') and 'default' not in rope_utils.ROPE_INIT_FUNCTIONS:
            def custom_rope_init_fn(config, device, **kwargs):
                import torch
                dim = config.hidden_size // config.num_attention_heads
                base = getattr(config, 'rope_theta', 1000000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                return inv_freq, 1.0
            rope_utils.ROPE_INIT_FUNCTIONS['default'] = custom_rope_init_fn
            has_patched_rope = True
            
        if hasattr(modeling_utils.PreTrainedModel, 'get_expanded_tied_weights_keys'):
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}

        try:
            import shutil, os
            hf_cache_dir = "/root/.cache/huggingface/modules"
            if os.path.exists(hf_cache_dir):
                shutil.rmtree(hf_cache_dir)
            
            dllm_path = args.dllm_dir or "/content/failfasttesting/Fast_dLLM_v2_1_5B"
            
            logging.info(f"{Colors.BOLD}=== Loading dLLM model from: {dllm_path} ==={Colors.RESET}")
            dllm = AutoModelForCausalLM.from_pretrained(
                dllm_path,
                torch_dtype="auto",
                device_map={"": args.drafter_device},
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa"
            )
            
            dllm.lm_head.weight = dllm.model.embed_tokens.weight

        except Exception as e:
            msg = str(e).lower()
            if isinstance(e, RuntimeError) and ("out of memory" in msg or 'cuda' in msg) or isinstance(e, torch.cuda.OutOfMemoryError):
                logging.error(f"{Colors.RED}CUDA OutOfMemory while loading dLLM: {e}{Colors.RESET}")
                sys.exit(1)
            raise
        finally:
            if has_patched_rope and 'default' in rope_utils.ROPE_INIT_FUNCTIONS:
                del rope_utils.ROPE_INIT_FUNCTIONS['default']
            if hasattr(modeling_utils.PreTrainedModel, 'get_expanded_tied_weights_keys') and original_tied_weights_fn is not None:
                modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = original_tied_weights_fn

    dllm_tokenizer = target_tokenizer
    if any(config[0] == "ar" for config in args.benchmark_drafter_configs.values()):
        try:
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.drafter_model_name,
                torch_dtype="auto",
                device_map={"": args.drafter_device}
            )
        except Exception as e:
            msg = str(e).lower()
            if isinstance(e, RuntimeError) and ("out of memory" in msg or 'cuda' in msg) or isinstance(e, torch.cuda.OutOfMemoryError):
                logging.error(f"{Colors.RED}CUDA OutOfMemory while loading drafter model {args.drafter_model_name}: {e}{Colors.RESET}")
                sys.exit(1)
            raise
        draft_tokenizer = target_tokenizer

measured_problem_ids = args.problem_ids or list(range(args.num_questions))
if len(measured_problem_ids) != args.num_questions:
    raise ValueError("--problem_ids must contain exactly --num_questions values")
if len(set(measured_problem_ids)) != len(measured_problem_ids):
    raise ValueError("--problem_ids must not contain duplicates")
if any(problem_id < 0 or problem_id >= len(args.dataset) for problem_id in measured_problem_ids):
    raise ValueError("--problem_ids contains an index outside the selected dataset")
warmup_problem_ids = [
    problem_id
    for problem_id in range(len(args.dataset))
    if problem_id not in set(measured_problem_ids)
][:args.warmup_questions]
benchmark_runs = (
    [(problem_id, True) for problem_id in warmup_problem_ids]
    + [(problem_id, False) for problem_id in measured_problem_ids]
)
measured_run_started = False
for problem_id, is_warmup in tqdm(
    benchmark_runs,
    desc="Problems",
    position=0,
    disable=args.disable_progress,
):
    if not is_warmup and not measured_run_started:
        reset_frontier_runtime_state(
            args,
            preserve_hardware_latency=args.frontier_stop_mode == "bucket_renewal",
        )
        measured_run_started = True
    transformers.set_seed(args.seed)
    raw_data = format_problem_and_options(args, problem_id)
    messages = [
        {"role": "user", "content": get_first_user_msg(args, raw_data)},
    ]
    text = args.target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if not args.read_pickle:
        base_orig_model_inputs = target_tokenizer([text], return_tensors="pt").to(target_model.device)
        num_target_tokens = args.max_new_tokens

    problem_benchmark_rows = []
    ar_drafter_speedup = {k: None for k in args.latency.keys()}
    for benchmark_mode in args.benchmark_modes:
        args.mode = benchmark_mode
        apply_mode_settings(args)
        transformers.set_seed(args.seed)
        drafter_config = args.benchmark_drafter_configs[benchmark_mode]
        draft_type, drafter_threshold, freq_scheme, lowconf_threshold, max_spec_len, incr_len = drafter_config
        drafter_name = format_drafter_name(args, drafter_config)
        
        output_dir_pickles, output_dir_figures = get_output_dir(args, str(problem_id), drafter_config)
        
        if args.read_pickle:
            if not os.path.exists(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle")):
                logging.warning(f"{Colors.RED}No cached pickle found for {drafter_name} (token budget {args.max_new_tokens})!{Colors.RESET}")
            
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"), "rb") as f:
                pickled_data = pickle.load(f)
            
            accepted_tokens = pickled_data["accepted_tokens"]
            drafted_tokens = pickled_data["drafted_tokens"]
            rejected_tokens = pickled_data["rejected_tokens"]
            num_speculation_rounds = pickled_data["num_speculation_rounds"]
            total_num_forward_passes = pickled_data["total_num_forward_passes"]
            current_token_ids = pickled_data.get("output_token_ids") or get_output_tokens(pickled_data["stats_each_round"])
            if draft_type == "verifier_ar" and not current_token_ids and pickled_data["stats_each_round"]:
                current_token_ids = pickled_data["stats_each_round"][0].get("target_tokens", [])
            draft_time_total = sum(x.get("draft_time_ms", 0.0) for x in pickled_data["stats_each_round"]) / 1000.0
            verify_time_total = sum(x.get("verify_time_ms", 0.0) for x in pickled_data["stats_each_round"]) / 1000.0
            post_verify_time_total = sum(x.get("post_verify_time_ms", 0.0) for x in pickled_data["stats_each_round"]) / 1000.0
            device_transfer_time_total = sum(
                x.get("device_transfer_time_ms", 0.0)
                for x in pickled_data["stats_each_round"]
            ) / 1000.0
            actual_e2e_time = pickled_data.get("actual_e2e_time", draft_time_total + verify_time_total)
        else:
            orig_model_inputs = {key: value.clone() for key, value in base_orig_model_inputs.items()}
            logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running drafter: {drafter_name} ==={Colors.RESET}")
            accepted_tokens = 0
            rejected_tokens = 0
            num_speculation_rounds = 0
            total_num_forward_passes = 0
            current_token_ids = []
            prev_prefill_output = None
            draft_time_total = 0.0
            verify_time_total = 0.0
            post_verify_time_total = 0.0
            device_transfer_time_total = 0.0
            args.device_transfer_time_total = 0.0
            oracle_diagnostic_time_total = 0.0
            pickled_data = {
                "orig_model_inputs": orig_model_inputs["input_ids"][0].tolist(),
                "raw_data": raw_data,
                "num_target_tokens": num_target_tokens,
                "stats_each_round": [],
            }

            if orig_model_inputs["input_ids"].is_cuda:
                torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
            generation_start = time.perf_counter()

            inner_bar = None
            if is_interactive() and not args.disable_progress:
                inner_bar = tqdm(total=num_target_tokens, miniters=1, desc=f"Verification (Problem {problem_id})",
                                position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

            if args.mode == "verifier_ar":
                logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running verifier-only AR generation ({args.target_model_name}) ==={Colors.RESET}")
                
                from transformers import TextStreamer
                streamer = None if args.quiet_generation else TextStreamer(target_tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                generation_print(args, f"\n⏳ BẮT ĐẦU AR-ONLY GENERATION (Live Stream):", flush=True)

                if orig_model_inputs["input_ids"].is_cuda:
                    torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
                verify_start = time.perf_counter()
                generated_ids = target_model.generate(
                    **orig_model_inputs,
                    max_new_tokens=num_target_tokens,
                    do_sample=False,
                    pad_token_id=target_model.config.eos_token_id,
                    eos_token_id=target_model.config.eos_token_id,
                    streamer=streamer
                )
                if orig_model_inputs["input_ids"].is_cuda:
                    torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
                verify_time_total = time.perf_counter() - verify_start
                current_token_ids = generated_ids[0][orig_model_inputs['input_ids'].shape[1]:].tolist()
                accepted_tokens = len(current_token_ids)
                drafted_tokens = accepted_tokens
                num_speculation_rounds = accepted_tokens
                total_num_forward_passes = 0
                pickled_data["stats_each_round"].append({
                    "mode": "verifier_ar",
                    "target_tokens": current_token_ids,
                    "spec_len": drafted_tokens,
                    "accepted_len": accepted_tokens,
                    "acceptance_rate": 1.0,
                    "num_forward_passes": total_num_forward_passes,
                    "draft_time_ms": 0.0,
                    "verify_time_ms": verify_time_total * 1000.0,
                })
            else:
                if is_interactive() and not args.disable_progress:
                    inner_bar = tqdm(total=num_target_tokens, miniters=1, desc=f"Verification (Problem {problem_id})",
                                    position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

                global_oracle_result = None
                if args.global_oracle_graph or args.strict_greedy_local_oracle:
                    if args.strict_greedy_local_oracle:
                        args.strict_greedy_record_diagnostics = not is_warmup
                        oracle_runner = run_strict_greedy_local_oracle_problem
                    else:
                        oracle_runner = (
                            run_truncated_global_oracle_problem
                            if args.truncated_global_horizon
                            else run_global_oracle_problem
                        )
                    global_oracle_result = oracle_runner(
                        args,
                        problem_id,
                        target_model,
                        dllm,
                        orig_model_inputs,
                        num_target_tokens,
                        drafter_threshold,
                        lowconf_threshold,
                        max_spec_len,
                        incr_len,
                    )
                    current_token_ids = global_oracle_result["current_token_ids"]
                    pickled_data["stats_each_round"] = global_oracle_result[
                        "stats_each_round"
                    ]
                    global_stats = global_oracle_result["global"]
                    draft_time_total = global_stats["draft_latency_ms"] / 1000.0
                    verify_time_total = global_stats["verify_latency_ms"] / 1000.0
                    post_verify_time_total = (
                        global_stats["post_verify_latency_ms"] / 1000.0
                    )
                    total_num_forward_passes = int(global_stats["draft_passes"])
                    num_speculation_rounds = int(global_stats["rounds"])

                while not (
                    args.global_oracle_graph or args.strict_greedy_local_oracle
                ) and len(current_token_ids) < num_target_tokens:
                    logging.debug(f"--- [{drafter_name}_{freq_scheme}] Speculation round {num_speculation_rounds} ---")

                    if orig_model_inputs["input_ids"].is_cuda:
                        torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
                    if (
                        draft_type == "dllm"
                        and args.adaptive_td
                        and args.adaptive_credit_assignment in {
                            "verifier_boundary_factual",
                            "verifier_boundary_factual_no_bootstrap",
                        }
                    ):
                        args.adaptive_td_controller.begin_factual_draft_round()
                    draft_start = time.perf_counter()
                    if draft_type == "ar":
                        if freq_scheme == "sf":
                            draft_proposal, drafter_probs = get_next_n_tokens_ar(
                                draft_model,
                                orig_model_inputs,
                                current_token_ids,
                                n=args.spec_len,
                                temperature=TEMPERATURE,
                                do_sample=args.decoding_strategy == "sampling",
                            )
                            spec_len = args.spec_len
                        else:
                            draft_proposal, confidences, drafter_probs = get_next_tokens_ar(
                                draft_model,
                                orig_model_inputs,
                                current_token_ids,
                                n=args.spec_len,
                                lowconf_threshold=lowconf_threshold,
                                max_spec_len=max_spec_len,
                                incr_len=incr_len,
                                temperature=TEMPERATURE,
                                do_sample=args.decoding_strategy == "sampling",
                            )
                            spec_len = len(draft_proposal)
                        
                        if not args.quiet_generation:
                            draft_text = target_tokenizer.decode(draft_proposal, skip_special_tokens=True)
                            generation_print(args, f"\n[VÒNG {num_speculation_rounds}] 🤖 DRAFTER NHÁP: {draft_text!r}", flush=True)
                        num_forward_passes = spec_len
                        
                    elif draft_type == "dllm":
                        if freq_scheme == "sf":
                            spec_len = args.spec_len
                            if args.disable_reusing_drafter_kvs:
                                draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                        spec_len=spec_len,
                                                                        output_seqlen=3*args.block_size,
                                                                        small_block_size=args.small_block_size,
                                                                        threshold=drafter_threshold,
                                                                        is_drafter=True,)
                            else:
                                draft_proposal, prefill_output, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                        spec_len=spec_len,
                                                                        output_seqlen=3*args.block_size,
                                                                        small_block_size=args.small_block_size,
                                                                        threshold=drafter_threshold,
                                                                        is_drafter=True,
                                                                        prev_prefill_output=prev_prefill_output)
                                prev_prefill_output = prefill_output
                        else:
                            last_round_proposal = pickled_data["stats_each_round"][-1]["~draft_proposal"] if num_speculation_rounds > 0 else []
                            last_round_accepted_len = pickled_data["stats_each_round"][-1]["accepted_len"] if num_speculation_rounds > 0 else 0
                            if last_round_accepted_len < len(last_round_proposal) - 1:
                                if args.reuse_drafts:
                                    last_round_rejected = last_round_proposal[last_round_accepted_len+1:] if num_speculation_rounds > 0 else []
                                else:
                                    last_round_rejected = None
                            else:
                                last_round_rejected = None
                            
                            draft_proposal, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = get_next_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                        spec_len=args.spec_len,
                                                                        output_seqlen=3*args.block_size,
                                                                        small_block_size=args.small_block_size,
                                                                        threshold=drafter_threshold,
                                                                        is_drafter=True,
                                                                        prev_prefill_output=prev_prefill_output,
                                                                        lowconf_threshold=lowconf_threshold,
                                                                        max_spec_len=max_spec_len,
                                                                        incr_len=incr_len,
                                                                        last_round_rejected=last_round_rejected,
                                                                        )
                            prev_prefill_output = prefill_output
                            spec_len = actual_spec_len
                            
                    if draft_type == "dllm":
                        if not args.quiet_generation:
                            draft_text = target_tokenizer.decode(draft_proposal, skip_special_tokens=True)
                            generation_print(args, f"\n[VÒNG {num_speculation_rounds}] ⚡ DLLM NHÁP: {draft_text!r}", flush=True)

                    if orig_model_inputs["input_ids"].is_cuda:
                        torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
                    physical_draft_time = time.perf_counter() - draft_start
                    draft_time = physical_draft_time
                    if (
                        draft_type == "dllm"
                        and args.adaptive_td
                        and args.adaptive_credit_assignment in {
                            "verifier_boundary_factual",
                            "verifier_boundary_factual_no_bootstrap",
                        }
                    ):
                        args.adaptive_td_controller.end_factual_draft_round()
                    frontier_stats_this_round = (
                        getattr(args, "last_frontier_stats", None)
                        if draft_type == "dllm"
                        else None
                    )
                    causal_oracle_decision = None
                    if args.causal_oracle and draft_type == "dllm" and not is_warmup:
                        physical_draft_passes = int(num_forward_passes)
                        selected, probe_wall_time_ms = choose_causal_oracle_proposal(
                            args,
                            problem_id,
                            benchmark_mode,
                            num_speculation_rounds,
                            target_model,
                            orig_model_inputs,
                            current_token_ids,
                            frontier_stats_this_round,
                            physical_draft_time * 1000.0,
                            physical_draft_passes,
                            sum(forward_pass_latencies),
                            draft_proposal,
                        )
                        selected_snapshot = selected["_snapshot"]
                        draft_proposal = selected["_proposal"]
                        spec_len = len(draft_proposal)
                        num_forward_passes = int(selected["draft_passes_elapsed"])
                        forward_pass_latencies = forward_pass_latencies[
                            :num_forward_passes
                        ]
                        draft_time = selected["effective_draft_latency_ms"] / 1000.0
                        excluded_extra_draft_time = max(
                            0.0,
                            physical_draft_time - draft_time,
                        )
                        oracle_diagnostic_time_total += (
                            probe_wall_time_ms / 1000.0 + excluded_extra_draft_time
                        )
                        prev_prefill_output = None
                        causal_oracle_decision = {
                            "problem_id": problem_id,
                            "mode": benchmark_mode,
                            "round_id": num_speculation_rounds,
                            "context_len": int(
                                orig_model_inputs["input_ids"].shape[1]
                                + len(current_token_ids)
                            ),
                            "num_candidates": selected["_num_candidates"],
                            "snapshot_fallback_used": selected[
                                "_snapshot_fallback_used"
                            ],
                            "oracle_snapshot_attempts": int(
                                (frontier_stats_this_round or {}).get(
                                    "oracle_snapshot_attempts", 0
                                )
                            ),
                            "oracle_snapshot_skipped_missing_fill": int(
                                (frontier_stats_this_round or {}).get(
                                    "oracle_snapshot_skipped_missing_fill", 0
                                )
                            ),
                            "selected_candidate_index": selected["candidate_index"],
                            "selected_step": int(selected_snapshot["step"]),
                            "selected_target_len": int(selected_snapshot["target_len"]),
                            "selected_draft_passes": num_forward_passes,
                            "selected_draft_latency_ms": draft_time * 1000.0,
                            "selected_expected_accepted_len": selected[
                                "accepted_len_if_stop"
                            ],
                            "selected_expected_emitted_len": selected[
                                "emitted_len_if_stop"
                            ],
                            "selected_counterfactual_verify_latency_ms": selected[
                                "counterfactual_verify_latency_ms"
                            ],
                            "selected_counterfactual_accept_check_latency_ms": selected[
                                "counterfactual_accept_check_latency_ms"
                            ],
                            "selected_counterfactual_ms_per_output_token": selected[
                                "counterfactual_ms_per_output_token"
                            ],
                            "oracle_cost_model": selected["_oracle_cost_model"],
                            "profile_tokens_per_round": selected["_future_stats"][
                                "tokens_per_round"
                            ],
                            "profile_draft_ms_per_round": selected["_future_stats"][
                                "draft_ms_per_round"
                            ],
                            "profile_verify_ms_per_round": selected["_future_stats"][
                                "verify_ms_per_round"
                            ],
                            "profile_post_verify_ms_per_round": selected[
                                "_future_stats"
                            ]["post_verify_ms_per_round"],
                            "selected_expected_extra_verifier_rounds": selected.get(
                                "expected_extra_verifier_rounds", 0.0
                            ),
                            "selected_future_draft_penalty_ms": selected.get(
                                "future_draft_penalty_ms", 0.0
                            ),
                            "selected_future_verify_penalty_ms": selected.get(
                                "future_verify_penalty_ms", 0.0
                            ),
                            "selected_future_post_verify_penalty_ms": selected.get(
                                "future_post_verify_penalty_ms", 0.0
                            ),
                            "selected_future_round_penalty_ms": selected.get(
                                "future_round_penalty_ms", 0.0
                            ),
                            "selected_adjusted_counterfactual_total_latency_ms": (
                                selected.get(
                                    "adjusted_counterfactual_total_latency_ms",
                                    selected["counterfactual_total_latency_ms"],
                                )
                            ),
                            "oracle_action_trace": json.dumps(
                                selected["_action_trace"]
                            ),
                            "physical_draft_passes": physical_draft_passes,
                            "physical_draft_latency_ms": physical_draft_time * 1000.0,
                            "excluded_extra_draft_latency_ms": (
                                excluded_extra_draft_time * 1000.0
                            ),
                            "counterfactual_probe_wall_time_ms": probe_wall_time_ms,
                        }
                    draft_time_total += draft_time
                    total_num_forward_passes += num_forward_passes
                    
                    if not draft_proposal:
                        logging.info(f"{Colors.RED}[Round {num_speculation_rounds}] Warning: Draft model returned no tokens{Colors.RESET}")
                        break
                    
                    prefix_len = len(current_token_ids)
                    combined_ids = current_token_ids + draft_proposal
                    verify_input_tensor = timed_token_tensor(
                        combined_ids,
                        target_model.device,
                        args,
                    )
                    full_input_ids = torch.cat([orig_model_inputs['input_ids'], verify_input_tensor], dim=1)

                    verify_mask_tensor = torch.ones_like(verify_input_tensor)
                    full_attention_mask = torch.cat([orig_model_inputs['attention_mask'], verify_mask_tensor], dim=1)

                    if verify_input_tensor.is_cuda:
                        torch.cuda.synchronize(verify_input_tensor.device)
                    verify_start = time.perf_counter()
                    with torch.inference_mode():
                        outputs = target_model(
                            input_ids=full_input_ids,
                            attention_mask=full_attention_mask,
                            use_cache=False,
                            logits_to_keep=len(draft_proposal) + 1,
                        )
                    if verify_input_tensor.is_cuda:
                        torch.cuda.synchronize(verify_input_tensor.device)
                    verify_time = time.perf_counter() - verify_start
                    verify_time_total += verify_time
                    
                    verify_logits = outputs.logits[0, :len(draft_proposal)]
                    post_verify_start = time.perf_counter()
                    
                    # ---------------------------------------------------------
                    # 🚀 BƯỚC C. ACCEPT/REJECT (CÓ TÍCH HỢP RESIDUAL SAMPLING)
                    # ---------------------------------------------------------
                    accepted_len = 0
                    bonus_token = None
                    target_tokens = [] # Dành cho logging tương thích cũ
                    checked_outcomes = []

                    accept_check_start = time.perf_counter()
                    generation_print(args, f"🔍 BƯỚC CHẤM BÀI CỦA VERIFIER:", flush=True)
                    for i in range(len(draft_proposal)):
                        if draft_type == "ar" and args.decoding_strategy == "sampling":
                            # === TOÁN HỌC RESIDUAL SAMPLING CHO AR-AR ===
                            p_drafter = drafter_probs[i]
                            # verify_logits là raw logit, cần chia cho TEMPERATURE
                            p_target = torch.softmax(verify_logits[i, :] / TEMPERATURE, dim=-1)
                            
                            # 🚀 FIX LỖI LỆCH TỪ ĐIỂN: Bơm thêm số 0 vào mảng nhỏ hơn để cân bằng size
                            if p_drafter.size(-1) < p_target.size(-1):
                                p_drafter = torch.nn.functional.pad(p_drafter, (0, p_target.size(-1) - p_drafter.size(-1)), value=0.0)
                            elif p_target.size(-1) < p_drafter.size(-1):
                                p_target = torch.nn.functional.pad(p_target, (0, p_drafter.size(-1) - p_target.size(-1)), value=0.0)
                            
                            draft_token_id = draft_proposal[i]
                            p_d = p_drafter[draft_token_id].item()
                            p_t = p_target[draft_token_id].item()
                            
                            # Tính tỉ lệ chấp nhận (Q/P)
                            acceptance_prob = min(1.0, p_t / p_d) if p_d > 0 else 1.0
                            r = torch.rand(1).item()
                            is_match = (r < acceptance_prob)
                            
                            if not args.quiet_generation:
                                draft_word = target_tokenizer.decode([draft_token_id])
                                status = f"✅ NHẬN (Tỉ lệ duyệt: {acceptance_prob*100:.1f}%)" if is_match else f"❌ GẠCH BỎ (Tỉ lệ duyệt: {acceptance_prob*100:.1f}%)"
                                generation_print(args, f"   Vị trí {i}: Drafter đoán [{draft_word!r}] -> {status}", flush=True)
                            
                            if not args.quiet_generation:
                                target_tokens.append(draft_token_id if is_match else torch.argmax(p_target).item())

                            if is_match:
                                accepted_len += 1
                            else:
                                # Nếu bị gạch, bốc thăm chữ mới dựa trên xác suất thặng dư (Q - P)
                                residual_probs = torch.clamp(p_target - p_drafter, min=0.0)
                                residual_sum = residual_probs.sum()
                                if residual_sum > 0:
                                    residual_probs = residual_probs / residual_sum
                                    final_token = torch.multinomial(residual_probs, 1).item()
                                else:
                                    final_token = torch.argmax(p_target).item()
                                    
                                if not args.quiet_generation:
                                    target_word = target_tokenizer.decode([final_token])
                                    generation_print(args, f"   👉 Dừng duyệt! Dùng Residual Sampling bốc được chữ thay thế: [{target_word!r}]", flush=True)
                                break
                                
                        else:
                            # === DLLM VẪN DÙNG EXACT MATCH (Do bản chất Distillation argmax) ===
                            target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
                            is_match = (draft_proposal[i] == target_pred)
                            if frontier_stop_enabled(args):
                                checked_outcomes.append(is_match)

                            if not args.quiet_generation:
                                draft_word = target_tokenizer.decode([draft_proposal[i]])
                                target_word = target_tokenizer.decode([target_pred])
                                status = "✅ NHẬN" if is_match else "❌ GẠCH BỎ"
                                generation_print(args, f"   Vị trí {i}: Đoán [{draft_word!r}] | Sửa thành [{target_word!r}] -> {status}", flush=True)
                            
                            if not args.quiet_generation:
                                target_tokens.append(target_pred)

                            if is_match:
                                accepted_len += 1
                            else:
                                final_token = target_pred
                                if not args.quiet_generation:
                                    generation_print(args, f"   👉 Dừng duyệt tại đây! Chốt sửa lỗi thành: [{target_word!r}]", flush=True)
                                break
                    else:
                        # NẾU TOÀN BỘ NHÁP ĐƯỢC CHẤP NHẬN -> TẶNG KÈM 1 CHỮ BONUS
                        final_token_logits = outputs.logits[0, -1, :]
                        if draft_type == "ar" and args.decoding_strategy == "sampling":
                            final_probs = torch.softmax(final_token_logits / TEMPERATURE, dim=-1)
                            final_token = torch.multinomial(final_probs, 1).item()
                        else:
                            final_token = torch.argmax(final_token_logits, dim=-1).item()
                            
                        bonus_token = final_token
                        if not args.quiet_generation:
                            bonus_word = target_tokenizer.decode([final_token])
                            generation_print(args, f"   👉 Trúng phóc 100%! Verifier tặng kèm 1 token bonus: [{bonus_word!r}]", flush=True)
                    accept_check_time_ms = (
                        time.perf_counter() - accept_check_start
                    ) * 1000.0
                    
                    generation_print(args, f"🎯 TỔNG KẾT VÒNG {num_speculation_rounds}: Chấp nhận {accepted_len}/{len(draft_proposal)} token.\n" + "-"*50, flush=True)
                    # ---------------------------------------------------------
                
                    if not args.quiet_generation:
                        get_proposal_str(args, spec_len, accepted_len, draft_proposal, final_token)
                    
                    tokens_to_append = draft_proposal[:accepted_len] + [final_token]
                    if target_tokenizer.eos_token_id in tokens_to_append:
                        eos_index = tokens_to_append.index(target_tokenizer.eos_token_id)
                        tokens_to_append = tokens_to_append[:eos_index + 1]
                    remaining_tokens = num_target_tokens - len(current_token_ids)
                    tokens_to_append = tokens_to_append[:remaining_tokens]
                    oracle_prefix_token_ids = list(current_token_ids)
                    current_token_ids.extend(tokens_to_append)
                    
                    accepted_tokens += accepted_len
                    rejected_tokens += len(draft_proposal) - accepted_len
                    if (
                        draft_type == "dllm"
                        and args.frontier_stop_mode == "bucket_renewal"
                    ):
                        update_frontier_latency_cost(
                            args,
                            forward_pass_latencies,
                            verify_time,
                            len(draft_proposal),
                            context_len=int(full_input_ids.shape[1] - len(draft_proposal)),
                        )
                        update_frontier_acceptance_calibration(
                            args,
                            frontier_stats_this_round,
                            checked_outcomes,
                        )

                    if (
                        draft_type == "dllm"
                        and args.adaptive_td
                        and args.adaptive_credit_assignment not in {
                            "verifier_boundary_factual",
                            "verifier_boundary_factual_no_bootstrap",
                        }
                    ):
                        complete_adaptive_td_trajectory(
                            args,
                            frontier_stats_this_round,
                            emitted_tokens=len(tokens_to_append),
                            verifier_latency_ms=verify_time * 1000.0,
                            round_latency_ms=(draft_time + verify_time) * 1000.0,
                            terminal=(
                                target_tokenizer.eos_token_id in tokens_to_append
                                or len(current_token_ids) >= num_target_tokens
                            ),
                        )

                    if (
                        draft_type == "dllm"
                        and args.frontier_stop_mode == "bucket_renewal"
                    ):
                        observed_post_verify_time = time.perf_counter() - post_verify_start
                        update_frontier_controller_cost(args, observed_post_verify_time)
                    if verify_input_tensor.is_cuda:
                        torch.cuda.synchronize(verify_input_tensor.device)
                    post_verify_time = time.perf_counter() - post_verify_start
                    post_verify_time_total += post_verify_time
                    if (
                        draft_type == "dllm"
                        and args.adaptive_td
                        and args.adaptive_credit_assignment in {
                            "verifier_boundary_factual",
                            "verifier_boundary_factual_no_bootstrap",
                        }
                    ):
                        complete_adaptive_td_trajectory(
                            args,
                            frontier_stats_this_round,
                            emitted_tokens=len(tokens_to_append),
                            verifier_latency_ms=verify_time * 1000.0,
                            post_verify_latency_ms=post_verify_time * 1000.0,
                            round_latency_ms=(
                                draft_time + verify_time + post_verify_time
                            )
                            * 1000.0,
                            terminal=(
                                target_tokenizer.eos_token_id in tokens_to_append
                                or len(current_token_ids) >= num_target_tokens
                            ),
                        )
                    audit_this_problem = (
                        args.audit_greedy_consistency
                        and (
                            not args.audit_greedy_problem_ids
                            or int(problem_id) in args.audit_greedy_problem_ids
                        )
                    )
                    if audit_this_problem:
                        audit_rows = audit_committed_greedy_tokens(
                            target_model,
                            orig_model_inputs,
                            oracle_prefix_token_ids,
                            draft_proposal,
                            accepted_len,
                            tokens_to_append,
                            outputs.logits[0],
                            eos_token_id=target_tokenizer.eos_token_id,
                        )
                        for audit_row in audit_rows:
                            audit_row.update({
                                "problem_id": int(problem_id),
                                "mode": benchmark_mode,
                                "round_id": int(num_speculation_rounds),
                                "prefix_length": int(prefix_len),
                            })
                        args.greedy_consistency_rows.extend(audit_rows)
                    if causal_oracle_decision is not None:
                        causal_oracle_decision.update({
                            "executed_accepted_len": accepted_len,
                            "executed_emitted_len": len(tokens_to_append),
                            "executed_verify_latency_ms": verify_time * 1000.0,
                            "executed_post_verify_latency_ms": post_verify_time * 1000.0,
                            "counterfactual_matches_execution": int(
                                accepted_len
                                == causal_oracle_decision[
                                    "selected_expected_accepted_len"
                                ]
                                and len(tokens_to_append)
                                == min(
                                    causal_oracle_decision[
                                        "selected_expected_emitted_len"
                                    ],
                                    remaining_tokens,
                                )
                            ),
                        })
                        append_causal_oracle_decision(
                            args,
                            causal_oracle_decision,
                        )
                    if (
                        draft_type == "dllm"
                        and args.adaptive_td
                    ):
                        if not args.adaptive_freeze:
                            if (
                                getattr(
                                    args.adaptive_td_controller.config,
                                    "feature_schema",
                                    "otrc_v1_td",
                                )
                                in (
                                    "otrc_v2_td",
                                    "otrc_v2_1_td",
                                    "otrc_v2_2_td",
                                    "otrc_v2_2_compact_td",
                                )
                            ):
                                args.adaptive_td_controller.observe_factual_verifier_call(
                                    len(tokens_to_append),
                                    verify_time * 1000.0,
                                )
                            args.adaptive_td_controller.observe_round(
                                len(tokens_to_append),
                                (draft_time + verify_time + post_verify_time) * 1000.0,
                            )
                        record_adaptive_td_decisions(
                            args,
                            frontier_stats_this_round,
                            problem_id=problem_id,
                            round_id=num_speculation_rounds,
                            accepted_draft_tokens=accepted_len,
                            emitted_tokens=len(tokens_to_append),
                            verifier_latency_ms=verify_time * 1000.0,
                            round_total_latency_ms=(
                                draft_time + verify_time + post_verify_time
                            )
                            * 1000.0,
                        )

                    if (
                        draft_type == "dllm"
                        and args.collect_bucket_oracle
                        and not args.causal_oracle
                        and not is_warmup
                    ):
                        oracle_diagnostic_start = time.perf_counter()
                        del verify_logits, outputs
                        if bonus_token is not None:
                            del final_token_logits
                        append_bucket_oracle_rows(
                            args,
                            problem_id,
                            benchmark_mode,
                            num_speculation_rounds,
                            target_model,
                            orig_model_inputs,
                            oracle_prefix_token_ids,
                            frontier_stats_this_round,
                            max(0.0, post_verify_time * 1000.0 - accept_check_time_ms),
                        )
                        oracle_diagnostic_time_total += (
                            time.perf_counter() - oracle_diagnostic_start
                        )
                    
                    info_this_round = {
                        "target_tokens": target_tokens,
                        "prefix_len": prefix_len,
                        "spec_len": spec_len,
                        "~draft_proposal": draft_proposal,
                        "accepted_len": accepted_len,
                        "acceptance_rate": accepted_len / spec_len,
                        "num_forward_passes": num_forward_passes,
                        "draft_time_ms": draft_time * 1000.0,
                        "verify_time_ms": verify_time * 1000.0,
                        "post_verify_time_ms": post_verify_time * 1000.0,
                        "device_transfer_time_ms": max(
                            0.0,
                            getattr(args, "device_transfer_time_total", 0.0)
                            - device_transfer_time_total,
                        ) * 1000.0,
                        "final_token": final_token,
                        "bonus_token": bonus_token,
                        "emitted_tokens": tokens_to_append,
                        "frontier_stats": frontier_stats_this_round,
                    }
                    pickled_data["stats_each_round"].append(info_this_round)
                    device_transfer_time_total = getattr(
                        args,
                        "device_transfer_time_total",
                        0.0,
                    )
                    if args.log_verifier_calls and not is_warmup:
                        args.verifier_call_rows.append({
                            "problem_id": int(problem_id),
                            "mode": benchmark_mode,
                            "round_id": int(num_speculation_rounds),
                            "context_length": int(
                                full_input_ids.shape[1] - len(draft_proposal)
                            ),
                            "proposal_length": int(spec_len),
                            "accepted_tokens": int(accepted_len),
                            "emitted_tokens": len(tokens_to_append),
                            "verify_latency_ms": verify_time * 1000.0,
                        })
                    
                    num_speculation_rounds += 1
                    
                    if inner_bar is not None:
                        inner_bar.update(len(tokens_to_append))

                    if target_tokenizer.eos_token_id in tokens_to_append:
                        break

            if orig_model_inputs["input_ids"].is_cuda:
                torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
            if args.global_oracle_graph or args.strict_greedy_local_oracle:
                actual_e2e_time = (
                    global_oracle_result["global"]["total_latency_ms"] / 1000.0
                )
            else:
                actual_e2e_time = (
                    time.perf_counter()
                    - generation_start
                    - oracle_diagnostic_time_total
                )

            if inner_bar is not None:
                inner_bar.close()

        stats_each_round = pickled_data["stats_each_round"]
        drafted_tokens = sum(x["spec_len"] for x in stats_each_round)
        accepted_tokens = sum(x["accepted_len"] for x in stats_each_round)
        rejected_tokens = drafted_tokens - accepted_tokens
        acceptance_rate = safe_div(accepted_tokens, drafted_tokens)
        if draft_type == "verifier_ar":
            total_num_forward_passes = 0
            num_speculation_rounds = drafted_tokens
        avg_spec_len = safe_div(drafted_tokens, num_speculation_rounds)
        avg_acc_len = safe_div(accepted_tokens, num_speculation_rounds)
        max_spec_len = max((x["spec_len"] for x in stats_each_round), default=0)
        max_acc_len = max((x["accepted_len"] for x in stats_each_round), default=0)

        logging.info(f"{Colors.BOLD}--- [Problem {problem_id}, {drafter_name}] Statistics ---{Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Acceptance rate: {acceptance_rate * 100:.1f}% ({accepted_tokens}/{drafted_tokens}){Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Accepted/speculated: avg {avg_acc_len:.2f}/{avg_spec_len:.2f}, max {max_acc_len}/{max_spec_len}{Colors.RESET}")
        
        total_output_tokens = len(current_token_ids)
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Avg fwd passes/round: {safe_div(total_num_forward_passes, num_speculation_rounds):.2f} ({total_num_forward_passes}/{num_speculation_rounds}) (total output tokens: {total_output_tokens}){Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Total draft time: {draft_time_total * 1000.0:.1f}ms, total verify time: {verify_time_total * 1000.0:.1f}ms{Colors.RESET}")
        if device_transfer_time_total > 0.0:
            logging.info(
                f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] "
                f"Two-GPU transfer time: {device_transfer_time_total * 1000.0:.1f}ms; "
                f"E2E excluding transfer: "
                f"{max(0.0, actual_e2e_time - device_transfer_time_total) * 1000.0:.1f}ms"
                f"{Colors.RESET}"
            )
        for hardware in args.latency.keys():
            latency_draft = total_num_forward_passes * args.latency[hardware]["draft_fwd_pass"]
            latency_target = num_speculation_rounds * args.latency[hardware]["target_tpt"][args.target_model_name_clean]
            total_tpt = latency_draft + latency_target
            avg_tpt = safe_div(total_tpt, total_output_tokens)
            speedup = safe_div(args.latency[hardware]["target_tpt"][args.target_model_name_clean], avg_tpt)
            logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Speedup: {speedup:.2f}x (Drafter ratio {safe_div(latency_draft, total_tpt) * 100:.1f}% ({latency_draft:.1f}ms/{total_tpt:.1f}ms); Avg TPT of SD: {avg_tpt:.2f}ms) (num output tokens: {total_output_tokens}){Colors.RESET}")
            
            if draft_type == "ar" and ar_drafter_speedup[hardware] is None:
                ar_drafter_speedup[hardware] = speedup
            if ar_drafter_speedup[hardware] is not None:
                logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Win over AR drafter: {safe_div(speedup, ar_drafter_speedup[hardware]):.3f}x.{Colors.RESET}")

        stats_each_round = pickled_data["stats_each_round"]
        if not args.skip_plots:
            if args.overwrite:
                visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=output_dir_figures, filename=f"{drafter_name}")
            else:
                visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=None, filename=None)

        if not args.quiet_generation:
            print_sd_trajectory(pickled_data, target_tokenizer)

        generated_text = target_tokenizer.decode(current_token_ids, skip_special_tokens=True)
        if (
            args.audit_greedy_consistency
            and not is_warmup
            and (
                not args.audit_greedy_problem_ids
                or int(problem_id) in args.audit_greedy_problem_ids
            )
        ):
            args.output_token_rows.extend({
                "problem_id": int(problem_id),
                "mode": benchmark_mode,
                "output_position": int(position),
                "token_id": int(token_id),
                "is_eos": int(int(token_id) == int(target_tokenizer.eos_token_id)),
            } for position, token_id in enumerate(current_token_ids))
        predicted_answer = extract_predicted_answer(generated_text)
        reference_answer = extract_reference_answer(raw_data)
        
        pickled_data["num_speculation_rounds"] = num_speculation_rounds
        pickled_data["total_num_forward_passes"] = total_num_forward_passes
        pickled_data["accepted_tokens"] = accepted_tokens
        pickled_data["drafted_tokens"] = drafted_tokens
        pickled_data["rejected_tokens"] = rejected_tokens
        pickled_data["acceptance_rate"] = acceptance_rate
        pickled_data["total_output_tokens"] = total_output_tokens
        pickled_data["output_token_ids"] = current_token_ids
        pickled_data["actual_e2e_time"] = actual_e2e_time
        pickled_data["actual_post_verify_time"] = post_verify_time_total
        pickled_data["actual_algorithm_time"] = draft_time_total + verify_time_total + post_verify_time_total
        pickled_data["device_transfer_time"] = device_transfer_time_total
        pickled_data["actual_e2e_time_excluding_transfer"] = max(
            0.0,
            actual_e2e_time - device_transfer_time_total,
        )
        pickled_data["generated_text"] = generated_text
        pickled_data["predicted_answer"] = predicted_answer
        pickled_data["reference_answer"] = reference_answer
        pickled_data["is_correct"] = predicted_answer is not None and predicted_answer == reference_answer
        
        should_write_artifacts = not args.skip_artifacts and (
            (args.overwrite and not args.read_pickle)
            or not os.path.exists(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"))
        )
        if should_write_artifacts:
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"), "wb") as f:
                pickle.dump(pickled_data, f)
                logging.info(f"Saved pickled data to {os.path.join(output_dir_pickles, f'{args.max_new_tokens}.pickle')}")
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.txt"), "w") as f:
                pp = pprint.PrettyPrinter(width=1000, stream=f)
                pp.pprint(pickled_data)
        elif not args.skip_artifacts:
            logging.info(f"Skipping save for pickled data to {os.path.join(output_dir_pickles, f'{args.max_new_tokens}.pickle')}")

        if not is_warmup and draft_type == "dllm" and (
            frontier_stop_enabled(args)
            or args.collect_draft_diagnostics
            or args.collect_bucket_oracle
        ) and not (args.global_oracle_graph or args.strict_greedy_local_oracle):
            append_frontier_diagnostic_rows(
                args,
                problem_id,
                benchmark_mode,
                stats_each_round,
            )

        problem_benchmark_rows.append(build_benchmark_row(
            args,
            problem_id,
            benchmark_mode,
            draft_time_total,
            verify_time_total,
            total_num_forward_passes,
            num_speculation_rounds,
            accepted_tokens,
            drafted_tokens,
            actual_e2e_time,
            post_verify_time_total,
            current_token_ids,
            predicted_answer,
            reference_answer,
            summarize_frontier_diagnostics(stats_each_round),
            device_transfer_time_total,
        ))

    baseline_row = next((row for row in problem_benchmark_rows if row["mode"] == "verifier_ar"), None)
    for row in problem_benchmark_rows:
        if row["mode"] == "verifier_ar":
            row["actual_speedup_vs_AR"] = 1.0
            row["actual_e2e_speedup_vs_AR"] = 1.0
            row["theo_speedup_vs_AR"] = 1.0
        elif baseline_row is not None:
            row["actual_speedup_vs_AR"] = safe_div(baseline_row["actual_algorithm_time"], row["actual_algorithm_time"])
            baseline_e2e_tpt = safe_div(baseline_row["actual_e2e_time"], baseline_row["output_tokens"])
            row_e2e_tpt = safe_div(row["actual_e2e_time"], row["output_tokens"])
            row["actual_e2e_speedup_vs_AR"] = safe_div(baseline_e2e_tpt, row_e2e_tpt)
            row["theo_speedup_vs_AR"] = safe_div(baseline_row["theo_total_time"], row["theo_total_time"])
    if not is_warmup:
        append_benchmark_rows(args, problem_benchmark_rows)

if args.frontier_stop_mode == "bucket_renewal":
    ensure_frontier_runtime_state(args)
    runtime_report = {
        "acceptance_calibration": args.bucket_acceptance_calibration,
        "gain_calibration": args.bucket_gain_calibration,
        "verify_latency_bins": args.bucket_verify_latency_bins,
        "draft_latency_bins": args.bucket_draft_latency_bins,
        "ema_dllm_forward_ms": args.bucket_ema_dllm_forward_ms,
        "ema_target_round_ms": args.bucket_ema_target_round_ms,
        "ema_post_verify_ms": args.bucket_ema_post_verify_ms,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "bucket_renewal_runtime_state.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(runtime_report, handle, indent=2)

if args.adaptive_td:
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "adaptive_td_runtime_state.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(args.adaptive_td_controller.snapshot(), handle, indent=2)
    decision_rows = getattr(args, "adaptive_decision_rows", [])
    if decision_rows:
        decision_columns = sorted(
            {column for row in decision_rows for column in row}
        )
        with open(
            os.path.join(args.output_dir, "adaptive_td_decisions.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=decision_columns)
            writer.writeheader()
            writer.writerows(decision_rows)
    transition_rows = getattr(
        args.adaptive_td_controller,
        "full_stream_transitions",
        [],
    )
    if transition_rows:
        transition_columns = sorted(
            {column for row in transition_rows for column in row}
        )
        with open(
            os.path.join(args.output_dir, "adaptive_full_stream_transitions.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=transition_columns)
            writer.writeheader()
            writer.writerows(transition_rows)

if args.strict_greedy_local_oracle and args.strict_greedy_decision_rows:
    os.makedirs(args.output_dir, exist_ok=True)
    decision_columns = sorted({
        column for row in args.strict_greedy_decision_rows for column in row
    })
    with open(
        os.path.join(args.output_dir, "greedy_local_oracle_decisions.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_columns)
        writer.writeheader()
        writer.writerows(args.strict_greedy_decision_rows)

if (
    args.strict_greedy_local_oracle
    and args.strict_greedy_replay_data is None
    and args.strict_greedy_selected_policy
):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "strict_greedy_policy.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "version": "strict_greedy_local_policy_v2_one_action_rollout",
                "policies": args.strict_greedy_selected_policy,
            },
            handle,
            indent=2,
        )

if args.log_verifier_calls or args.strict_greedy_local_oracle:
    if args.verifier_call_rows:
        os.makedirs(args.output_dir, exist_ok=True)
        call_columns = sorted({
            column for row in args.verifier_call_rows for column in row
        })
        with open(
            os.path.join(args.output_dir, "verifier_calls.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=call_columns)
            writer.writeheader()
            writer.writerows(args.verifier_call_rows)

if args.audit_greedy_consistency:
    os.makedirs(args.output_dir, exist_ok=True)
    for filename, rows in (
        ("greedy_consistency_audit.csv", args.greedy_consistency_rows),
        ("output_token_trace.csv", args.output_token_rows),
    ):
        if not rows:
            continue
        columns = sorted({column for row in rows for column in row})
        with open(
            os.path.join(args.output_dir, filename),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
