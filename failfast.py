import os
import sys
import copy
import time
import torch
import pickle
import pprint
import logging
import argparse
import transformers
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

transformers.logging.set_verbosity_error()

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
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs

def get_next_n_tokens_ar(model, orig_model_inputs, token_ids_so_far, n):
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = model.generate(
        **new_model_inputs,
        max_new_tokens=n,
        do_sample=False,
        pad_token_id=model.config.eos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()

def get_next_tokens_ar(
    model,
    orig_model_inputs,
    token_ids_so_far,
    n,
    lowconf_threshold,
    max_spec_len,
    incr_len,
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

    current_tokens = torch.tensor(token_ids_so_far, device=device, dtype=torch.long).unsqueeze(0)
    current_mask = torch.ones_like(current_tokens, dtype=torch.long)
    current_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], current_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], current_mask], dim=1)
    }

    with torch.no_grad():
        while len(drafted) < cap:
            chunk_size = min(incr_len, cap - len(drafted))
            
            generate_output = model.generate(
                **current_inputs,
                max_new_tokens=chunk_size,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id,
            )
            
            generated_ids = generate_output.sequences[0][len(current_inputs["input_ids"][0]):]
            generated_ids = generated_ids.tolist()
            
            scores = generate_output.scores
            found_lowconf = False
            for i, (token_id, score_logits) in enumerate(zip(generated_ids, scores)):
                probs = torch.softmax(score_logits, dim=-1)
                conf = probs[0, token_id].item()
                drafted.append(token_id)
                confidences.append(conf)
                
                if conf < lowconf_threshold:
                    found_lowconf = True
            
            if found_lowconf:
                return drafted, confidences
            
            if len(drafted) < cap:
                new_tokens = torch.tensor(generated_ids, device=device, dtype=torch.long).unsqueeze(0)
                new_mask = torch.ones_like(new_tokens, dtype=torch.long)
                current_inputs = {
                    'input_ids': torch.cat([current_inputs['input_ids'], new_tokens], dim=1),
                    'attention_mask': torch.cat([current_inputs['attention_mask'], new_mask], dim=1)
                }

    return drafted, confidences

def get_next_n_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
    num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    if args.disable_reusing_drafter_kvs:
        generated_ids, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
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
        )
    else:
        generated_ids, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
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
        )
    
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
    num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)

    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    if args.disable_reusing_drafter_kvs:
        generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
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
        )
    else:
        generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
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
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:actual_spec_len]
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
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
parser.add_argument("--target_model_name", type=str, default=None)
parser.add_argument("--verifier_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--drafter_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--dllm_dir", type=str, default=None)
parser.add_argument("--num_questions", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--small_block_size", type=int, default=8)
parser.add_argument("--spec_len", type=int, default=10)
parser.add_argument("--drafter_thresholds", type=float, nargs="+", default=[0.05])
parser.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
parser.add_argument("--sweep_lowconf_threshold", type=float, nargs="+", default=[0.45])
parser.add_argument("--sweep_max_spec_len", type=int, nargs="+", default=[60])
parser.add_argument("--sweep_incr_len", type=int, nargs="+", default=[10])
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

apply_mode_settings(args)
args.target_model_name_clean = args.target_model_name.split("/", 1)[1]
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
)

construct_drafter_configs(args)
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
            attn_implementation="eager"
        )
    except Exception as e:
        msg = str(e).lower()
        if isinstance(e, RuntimeError) and ("out of memory" in msg or 'cuda' in msg) or isinstance(e, torch.cuda.OutOfMemoryError):
            logging.error(f"{Colors.RED}CUDA OutOfMemory while loading target model {args.target_model_name}: {e}{Colors.RESET}")
            sys.exit(1)
        raise
    dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

    dllm = None
    dllm_tokenizer = target_tokenizer
    if args.mode == "dllm_ar" or args.run_dllm_sf:
        import transformers.modeling_rope_utils as rope_utils
        import transformers.modeling_utils as modeling_utils
        
        has_patched_rope = False
        original_tied_weights_fn = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
        
        if hasattr(rope_utils, 'ROPE_INIT_FUNCTIONS') and 'default' not in rope_utils.ROPE_INIT_FUNCTIONS:
            def custom_rope_init_fn(config, device, **kwargs):
                import torch
                dim = config.hidden_size // config.num_attention_heads
                base = getattr(config, 'rope_theta', 1000000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                return inv_freq, 1.0
            rope_utils.ROPE_INIT_FUNCTIONS['default'] = custom_rope_init_fn
            has_patched_rope = True
            
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}

        try:
            import shutil, os
            hf_cache_dir = "/root/.cache/huggingface/modules"
            if os.path.exists(hf_cache_dir):
                shutil.rmtree(hf_cache_dir)
            
            dllm_path = "/content/failfasttesting/Fast_dLLM_v2_1.5B"
            
            logging.info(f"{Colors.BOLD}=== Loading dLLM model from: {dllm_path} ==={Colors.RESET}")
            dllm = AutoModelForCausalLM.from_pretrained(
                dllm_path,
                torch_dtype="auto",
                device_map={"": 0},
                trust_remote_code=True,
                local_files_only=True
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
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = original_tied_weights_fn

    dllm_tokenizer = target_tokenizer
    if args.mode == "ar_ar":
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

for problem_id in tqdm(range(args.num_questions), desc="Problems", position=0):
    transformers.set_seed(42)
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
        orig_model_inputs = target_tokenizer([text], return_tensors="pt").to(target_model.device)
        num_target_tokens = args.max_new_tokens

    ar_drafter_speedup = {k: None for k in args.latency.keys()}
    for drafter_config in args.drafter_configs:
        transformers.set_seed(42)
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
            current_token_ids = get_output_tokens(pickled_data["stats_each_round"])
        else:
            logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running drafter: {drafter_name} ==={Colors.RESET}")
            accepted_tokens = 0
            rejected_tokens = 0
            num_speculation_rounds = 0
            total_num_forward_passes = 0
            current_token_ids = []
            prev_prefill_output = None
            draft_time_total = 0.0
            verify_time_total = 0.0
            pickled_data = {
                "orig_model_inputs": orig_model_inputs["input_ids"][0].tolist(),
                "raw_data": raw_data,
                "num_target_tokens": num_target_tokens,
                "stats_each_round": [],
            }

            if is_interactive():
                inner_bar = tqdm(total=num_target_tokens, miniters=1, desc=f"Verification (Problem {problem_id})",
                                position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

            if args.mode == "verifier_ar":
                draft_model = None
                logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running verifier-only AR generation ({args.target_model_name}) ==={Colors.RESET}")
                verify_start = time.perf_counter()
                generated_ids = target_model.generate(
                    **orig_model_inputs,
                    max_new_tokens=num_target_tokens,
                    do_sample=False,
                    pad_token_id=target_model.config.eos_token_id,
                    eos_token_id=target_model.config.eos_token_id,
                )
                verify_time_total = time.perf_counter() - verify_start
                current_token_ids = generated_ids[0][orig_model_inputs['input_ids'].shape[1]:].tolist()
                num_speculation_rounds = 1
                total_num_forward_passes = len(current_token_ids)
                pickled_data["stats_each_round"].append({
                    "mode": "verifier_ar",
                    "spec_len": num_target_tokens,
                    "accepted_len": num_target_tokens,
                    "acceptance_rate": 1.0,
                    "num_forward_passes": total_num_forward_passes,
                    "draft_time_ms": 0.0,
                    "verify_time_ms": verify_time_total * 1000.0,
                })
            else:
                if is_interactive():
                    inner_bar = tqdm(total=num_target_tokens, miniters=1, desc=f"Verification (Problem {problem_id})",
                                    position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

                while len(current_token_ids) < num_target_tokens:
                    logging.debug(f"--- [{drafter_name}_{freq_scheme}] Speculation round {num_speculation_rounds} ---")

                    draft_start = time.perf_counter()
                    if draft_type == "ar":
                        if freq_scheme == "sf":
                            draft_proposal = get_next_n_tokens_ar(draft_model, orig_model_inputs, current_token_ids, n=args.spec_len)
                            spec_len = args.spec_len
                        else:
                            draft_proposal, confidences = get_next_tokens_ar(
                                draft_model,
                                orig_model_inputs,
                                current_token_ids,
                                n=args.spec_len,
                                lowconf_threshold=lowconf_threshold,
                                max_spec_len=max_spec_len,
                                incr_len=incr_len,
                            )
                            spec_len = len(draft_proposal)
                        
                        draft_text = target_tokenizer.decode(draft_proposal, skip_special_tokens=True)
                        print(f"\n[VÒNG {num_speculation_rounds}] 🤖 DRAFTER NHÁP: {draft_text!r}", flush=True)
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
                        draft_text = target_tokenizer.decode(draft_proposal, skip_special_tokens=True)
                        print(f"\n[VÒNG {num_speculation_rounds}] ⚡ DLLM NHÁP: {draft_text!r}", flush=True)
                        
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

                    verify_start = time.perf_counter()
                    with torch.no_grad():
                        outputs = target_model(input_ids=full_input_ids, attention_mask=full_attention_mask)
                    verify_time = time.perf_counter() - verify_start
                    verify_time_total += verify_time
                    
                    start_index = orig_model_inputs['input_ids'].shape[1] + prefix_len - 1
                    end_index = start_index + len(draft_proposal)
                    verify_logits = outputs.logits[0, start_index:end_index]
                    target_tokens = torch.argmax(verify_logits, dim=-1).tolist()
                    
                    # ---------------------------------------------------------
                    # C. ACCEPT/REJECT: IN CHI TIẾT TỪNG TOKEN ĐỂ THEO DÕI
                    # ---------------------------------------------------------
                    accepted_len = 0
                    bonus_token = None
                    
                    print(f"🔍 BƯỚC CHẤM BÀI CỦA VERIFIER:", flush=True)
                    for i in range(len(draft_proposal)):
                        target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
                        is_match = (draft_proposal[i] == target_pred)
                        
                        draft_word = target_tokenizer.decode([draft_proposal[i]])
                        target_word = target_tokenizer.decode([target_pred])
                        status = "✅ NHẬN" if is_match else "❌ GẠCH BỎ"
                        print(f"   Vị trí {i}: Đoán [{draft_word!r}] | Sửa thành [{target_word!r}] -> {status}", flush=True)

                        if is_match:
                            accepted_len += 1
                        else:
                            final_token = target_pred
                            print(f"   👉 Dừng duyệt tại đây! Chốt sửa lỗi thành: [{target_word!r}]", flush=True)
                            break
                    else:
                        final_token_logits = outputs.logits[0, -1, :]
                        final_token = torch.argmax(final_token_logits, dim=-1).item()
                        bonus_token = final_token
                        bonus_word = target_tokenizer.decode([final_token])
                        print(f"   👉 Trúng phóc 100%! Verifier tặng kèm 1 token bonus: [{bonus_word!r}]", flush=True)
                    
                    print(f"🎯 TỔNG KẾT VÒNG {num_speculation_rounds}: Chấp nhận {accepted_len}/{len(draft_proposal)} token.\n" + "-"*50, flush=True)
                    # ---------------------------------------------------------
                
                    proposal_str = get_proposal_str(args, spec_len, accepted_len, draft_proposal, final_token)
                    
                    tokens_to_append = draft_proposal[:accepted_len] + [final_token]
                    current_token_ids.extend(tokens_to_append)
                    
                    accepted_tokens += accepted_len
                    rejected_tokens += len(draft_proposal) - accepted_len
                    
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
                        "final_token": final_token,
                        "bonus_token": bonus_token,
                    }
                    pickled_data["stats_each_round"].append(info_this_round)
                    
                    num_speculation_rounds += 1
                    
                    if is_interactive():
                        inner_bar.update(len(tokens_to_append))

                    if target_tokenizer.eos_token_id in tokens_to_append:
                        break

            if is_interactive():
                inner_bar.close()

        drafted_tokens = sum([x["spec_len"] for x in pickled_data["stats_each_round"]])
        acceptance_rate = accepted_tokens / drafted_tokens
        avg_spec_len = sum([x["spec_len"] for x in pickled_data["stats_each_round"]]) / num_speculation_rounds
        avg_acc_len = sum([x["accepted_len"] for x in pickled_data["stats_each_round"]]) / num_speculation_rounds
        max_spec_len = max([x["spec_len"] for x in pickled_data["stats_each_round"]])
        max_acc_len = max([x["accepted_len"] for x in pickled_data["stats_each_round"]])

        logging.info(f"{Colors.BOLD}--- [Problem {problem_id}, {drafter_name}] Statistics ---{Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Acceptance rate: {acceptance_rate * 100:.1f}% ({accepted_tokens}/{drafted_tokens}){Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Accepted/speculated: avg {avg_acc_len:.2f}/{avg_spec_len:.2f}, max {max_acc_len}/{max_spec_len}{Colors.RESET}")
        
        total_output_tokens = len(current_token_ids)
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Avg fwd passes/round: {total_num_forward_passes / num_speculation_rounds:.2f} ({total_num_forward_passes}/{num_speculation_rounds}) (total output tokens: {total_output_tokens}){Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Total draft time: {draft_time_total * 1000.0:.1f}ms, total verify time: {verify_time_total * 1000.0:.1f}ms{Colors.RESET}")
        for hardware in args.latency.keys():
            latency_draft = total_num_forward_passes * args.latency[hardware]["draft_fwd_pass"]
            latency_target = num_speculation_rounds * args.latency[hardware]["target_tpt"][args.target_model_name_clean]
            total_tpt = latency_draft + latency_target
            avg_tpt = total_tpt / total_output_tokens
            speedup = args.latency[hardware]["target_tpt"][args.target_model_name_clean] / avg_tpt
            logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Speedup: {speedup:.2f}x (Drafter ratio {latency_draft / total_tpt * 100:.1f}% ({latency_draft:.1f}ms/{total_tpt:.1f}ms); Avg TPT of SD: {avg_tpt:.2f}ms) (num output tokens: {total_output_tokens}){Colors.RESET}")
            
            if draft_type == "ar" and ar_drafter_speedup[hardware] is None:
                ar_drafter_speedup[hardware] = speedup
            if ar_drafter_speedup[hardware] is not None:
                logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Win over AR drafter: {speedup / ar_drafter_speedup[hardware]:.3f}x.{Colors.RESET}")

        stats_each_round = pickled_data["stats_each_round"]
        if args.overwrite:
            visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=output_dir_figures, filename=f"{drafter_name}")
        else:
            visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=None, filename=None)
        
        print_sd_trajectory(pickled_data, target_tokenizer)
        
        pickled_data["num_speculation_rounds"] = num_speculation_rounds
        pickled_data["total_num_forward_passes"] = total_num_forward_passes
        pickled_data["accepted_tokens"] = accepted_tokens
        pickled_data["drafted_tokens"] = drafted_tokens
        pickled_data["rejected_tokens"] = rejected_tokens
        pickled_data["acceptance_rate"] = acceptance_rate
        pickled_data["total_output_tokens"] = total_output_tokens
        
        if (args.overwrite and not args.read_pickle) or (not os.path.exists(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"))):
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"), "wb") as f:
                pickle.dump(pickled_data, f)
                logging.info(f"Saved pickled data to {os.path.join(output_dir_pickles, f'{args.max_new_tokens}.pickle')}")
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.txt"), "w") as f:
                pp = pprint.PrettyPrinter(width=1000, stream=f)
                pp.pprint(pickled_data)
        else:
            logging.info(f"Skipping save for pickled data to {os.path.join(output_dir_pickles, f'{args.max_new_tokens}.pickle')}")