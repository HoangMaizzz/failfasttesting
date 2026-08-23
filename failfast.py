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
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController
from bucket_renewal import position_bucket

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
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
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
    generated_ids = generated_ids.tolist()[:spec_len]
    
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
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
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
    generated_ids = generated_ids.tolist()[:actual_spec_len]
    
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
parser.add_argument("--bucket_oracle_force_continue", action="store_true")
parser.add_argument("--adaptive-td", action="store_true")
parser.add_argument("--adaptive-max-refinement-steps", type=int, default=16)
parser.add_argument("--adaptive-fixed-refinement-steps", type=int)
parser.add_argument("--adaptive-learning-rate", type=float, default=0.02)
parser.add_argument("--adaptive-mc-learning-rate", type=float, default=0.01)
parser.add_argument("--adaptive-mc-mix", type=float, default=0.5)
parser.add_argument(
    "--adaptive-update-mode",
    choices=["td", "factual_return", "mixed"],
    default="mixed",
)
parser.add_argument("--adaptive-rho-alpha", type=float, default=0.05)
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
parser.add_argument("--adaptive-use-step-feature", action="store_true")
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

if args.target_model_name is None:
    args.target_model_name = args.verifier_model_name
if args.adaptive_td and args.frontier_stop_mode != "disabled":
    raise ValueError("--adaptive-td cannot be combined with --frontier_stop_mode")
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
args.adaptive_td_controller = (
    OnlineTDRefinementController(
        AdaptiveTDConfig(
            learning_rate=args.adaptive_learning_rate,
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
        )
    )
    if args.adaptive_td
    else None
)
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
    return bool(getattr(args, "adaptive_td", False)) or (
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
        args.adaptive_decision_rows.append({
            "problem_id": int(problem_id),
            "round_id": int(round_id),
            "decision_id": int(decision_id),
            **item,
            "features": json.dumps(item.get("features") or []),
            "draft_length": target_len,
            "remaining_mask_ratio": float((item.get("features") or [0.0, 0.0])[1]),
            "newly_unmasked_ratio": float((item.get("features") or [0.0, 0.0, 0.0])[2]),
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
    for index, draft_token_id in enumerate(draft_proposal):
        target_token_id = int(torch.argmax(verify_logits[index], dim=-1).item())
        if int(draft_token_id) != target_token_id:
            break
        accepted_len += 1
    post_verify_latency_ms = (time.perf_counter() - post_verify_start) * 1000.0
    del verify_logits, oracle_outputs
    return accepted_len, accepted_len + 1, verify_latency_ms, post_verify_latency_ms


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
            device_map={"": 0},
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
                device_map={"": 0},
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
                device_map={"": 0}
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

                while len(current_token_ids) < num_target_tokens:
                    logging.debug(f"--- [{drafter_name}_{freq_scheme}] Speculation round {num_speculation_rounds} ---")

                    if orig_model_inputs["input_ids"].is_cuda:
                        torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
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
                    draft_time = time.perf_counter() - draft_start
                    draft_time_total += draft_time
                    total_num_forward_passes += num_forward_passes
                    
                    if not draft_proposal:
                        logging.info(f"{Colors.RED}[Round {num_speculation_rounds}] Warning: Draft model returned no tokens{Colors.RESET}")
                        break
                    
                    prefix_len = len(current_token_ids)
                    combined_ids = current_token_ids + draft_proposal
                    verify_input_tensor = torch.tensor([combined_ids], device=target_model.device, dtype=torch.long)
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
                    frontier_stats_this_round = getattr(args, "last_frontier_stats", None) if draft_type == "dllm" else None
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

                    if draft_type == "dllm" and args.adaptive_td:
                        complete_adaptive_td_trajectory(
                            args,
                            frontier_stats_this_round,
                            emitted_tokens=len(tokens_to_append),
                            verifier_latency_ms=verify_time * 1000.0,
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
                        and not args.adaptive_freeze
                    ):
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
                        "final_token": final_token,
                        "bonus_token": bonus_token,
                        "emitted_tokens": tokens_to_append,
                        "frontier_stats": frontier_stats_this_round,
                    }
                    pickled_data["stats_each_round"].append(info_this_round)
                    
                    num_speculation_rounds += 1
                    
                    if inner_bar is not None:
                        inner_bar.update(len(tokens_to_append))

                    if target_tokenizer.eos_token_id in tokens_to_append:
                        break

            if orig_model_inputs["input_ids"].is_cuda:
                torch.cuda.synchronize(orig_model_inputs["input_ids"].device)
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
        ):
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
