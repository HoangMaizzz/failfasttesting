import argparse
import ast
import csv
import logging
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class Colors:
    RED = "\033[91m"
    RESET = "\033[0m"


def load_failfast_get_next_tokens_dllm():
    failfast_path = Path(__file__).with_name("failfast.py")
    tree = ast.parse(failfast_path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "get_next_tokens_dllm"
    )
    namespace = {"torch": torch, "logging": logging, "Colors": Colors}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(failfast_path), "exec"), namespace)
    return namespace["get_next_tokens_dllm"]


get_next_tokens_dllm = load_failfast_get_next_tokens_dllm()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--spec_len", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--incr_len", type=int, default=10)
    parser.add_argument("--kl_threshold", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--drafter_model_path", type=str, default="/content/failfasttesting/Fast_dLLM_v2_1_5B")
    parser.add_argument("--verifier_model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--theoretical_tflops", type=float, default=150.0)
    parser.add_argument("--disable_reusing_drafter_kvs", action="store_true")
    parser.add_argument("--allow_remote_drafter", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model):
    return next(model.parameters()).device


def model_parameter_count(model):
    return sum(param.numel() for param in model.parameters())


def format_gsm8k_prompt(question):
    return (
        "Solve the following math problem efficiently and clearly. "
        "Please reason step by step, separate logical reasoning steps with two newline characters, "
        "and put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )


def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def tokenize_prompt(tokenizer, prompt, device):
    text = apply_chat_template(tokenizer, prompt)
    return tokenizer([text], return_tensors="pt").to(device)


def load_drafter(args):
    import transformers.modeling_rope_utils as rope_utils
    import transformers.modeling_utils as modeling_utils

    has_patched_rope = False
    original_tied_weights_fn = getattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys", None)

    if hasattr(rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def custom_rope_init_fn(config, device, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 1000000.0)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            return inv_freq, 1.0
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = custom_rope_init_fn
        has_patched_rope = True

    if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys"):
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = lambda self, all_submodels=False: {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.drafter_model_path,
            trust_remote_code=True,
            local_files_only=not args.allow_remote_drafter,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.drafter_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=not args.allow_remote_drafter,
            attn_implementation="sdpa",
        )
        if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()
        return model, tokenizer
    finally:
        if has_patched_rope and "default" in rope_utils.ROPE_INIT_FUNCTIONS:
            del rope_utils.ROPE_INIT_FUNCTIONS["default"]
        if hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys") and original_tied_weights_fn is not None:
            modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = original_tied_weights_fn


def load_verifier(args):
    tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.verifier_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_draft_to_verify_projection(drafter_model, drafter_tokenizer, verifier_model, verifier_tokenizer, cache_path):
    draft_vocab_size = getattr(drafter_model.config, "vocab_size", len(drafter_tokenizer))
    verify_vocab_size = getattr(verifier_model.config, "vocab_size", len(verifier_tokenizer))
    if draft_vocab_size == verify_vocab_size and drafter_tokenizer.__class__ == verifier_tokenizer.__class__:
        return None
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    projection = torch.full((draft_vocab_size,), -1, dtype=torch.long)
    for draft_token_id in tqdm(range(draft_vocab_size), desc="Vocab projection"):
        text = drafter_tokenizer.decode([draft_token_id], skip_special_tokens=False)
        verify_ids = verifier_tokenizer.encode(text, add_special_tokens=False)
        if len(verify_ids) == 1 and 0 <= verify_ids[0] < verify_vocab_size:
            projection[draft_token_id] = verify_ids[0]

    torch.save(projection, cache_path)
    return projection


def map_draft_tokens_to_verify_tokens(draft_tokens, drafter_tokenizer, verifier_tokenizer):
    verify_tokens = []
    aligned = []
    for token_id in draft_tokens:
        text = drafter_tokenizer.decode([token_id], skip_special_tokens=False)
        ids = verifier_tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            verify_tokens.append(ids[0])
            aligned.append(True)
        else:
            encoded = verifier_tokenizer.encode(text, add_special_tokens=False)
            verify_tokens.append(encoded[0] if encoded else verifier_tokenizer.eos_token_id)
            aligned.append(False)
    return verify_tokens, aligned


def build_attention_mask(input_ids):
    return torch.ones_like(input_ids, dtype=torch.long)


def forward_logits(model, input_ids):
    attention_mask = build_attention_mask(input_ids)
    with torch.inference_mode():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            try:
                outputs = model(input_ids=input_ids)
            except TypeError:
                outputs = model(input_ids)
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise RuntimeError("Verifier or drafter forward pass did not return logits.")


def project_verify_probs_to_draft_vocab(verify_probs, draft_vocab_size, projection):
    if projection is None:
        return verify_probs
    projection = projection.to(verify_probs.device)
    valid = projection >= 0
    projected = torch.zeros(
        verify_probs.shape[:-1] + (draft_vocab_size,),
        dtype=verify_probs.dtype,
        device=verify_probs.device,
    )
    projected[..., valid] = verify_probs[..., projection[valid]]
    return projected / projected.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def ddsd_kl_verify(draft_logits, verify_logits, draft_tokens, aligned, threshold, temperature, projection):
    draft_logits = draft_logits.float() / temperature
    verify_logits = verify_logits.float() / temperature
    log_p_draft = F.log_softmax(draft_logits, dim=-1)
    p_verify_native = F.softmax(verify_logits, dim=-1)
    p_verify = project_verify_probs_to_draft_vocab(p_verify_native, draft_logits.shape[-1], projection)
    kl_scores = F.kl_div(log_p_draft, p_verify, reduction="none").sum(dim=-1)

    accepted_len = 0
    for is_aligned, score in zip(aligned, kl_scores):
        if not is_aligned or score.item() >= threshold:
            break
        accepted_len += 1

    replacement_token = None
    if accepted_len < len(draft_tokens):
        replacement_probs = p_verify[accepted_len]
        replacement_token = torch.multinomial(replacement_probs, 1).item()

    return accepted_len, replacement_token, kl_scores.detach().cpu().tolist()


def theoretical_seconds(param_count, sequence_length, forward_passes, tflops):
    return (2.0 * param_count * sequence_length * forward_passes) / (tflops * 1e12)


def append_csv_row(csv_path, row):
    columns = [
        "problem_id",
        "actual_draft_time",
        "actual_verify_time",
        "actual_total_time",
        "theo_draft_time",
        "theo_verify_time",
        "draft_time_percentage",
        "acceptance_rate",
    ]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_problem(args, problem_id, question, drafter_model, drafter_tokenizer, verifier_model, verifier_tokenizer, projection, drafter_params, verifier_params):
    set_seed(args.seed)
    prompt = format_gsm8k_prompt(question)
    drafter_device = get_model_device(drafter_model)
    verifier_device = get_model_device(verifier_model)
    orig_drafter_inputs = tokenize_prompt(drafter_tokenizer, prompt, drafter_device)
    orig_verifier_inputs = tokenize_prompt(verifier_tokenizer, prompt, verifier_device)
    drafter_prompt_len = orig_drafter_inputs["input_ids"].shape[1]
    verifier_prompt_len = orig_verifier_inputs["input_ids"].shape[1]
    helper_args = SimpleNamespace(
        block_size=args.block_size,
        disable_reusing_drafter_kvs=args.disable_reusing_drafter_kvs,
    )

    current_draft_tokens = []
    current_verify_tokens = []
    prev_prefill_output = None
    actual_draft_time = 0.0
    actual_verify_time = 0.0
    theo_draft_time = 0.0
    theo_verify_time = 0.0
    accepted_tokens = 0
    total_drafted_tokens = 0

    print(f"\nProblem {problem_id}")
    print(prompt)

    round_id = 0
    while len(current_draft_tokens) < args.max_new_tokens:
        draft_start = time.time()
        draft_output = get_next_tokens_dllm(
            drafter_model,
            helper_args,
            orig_drafter_inputs,
            current_draft_tokens,
            spec_len=args.spec_len,
            output_seqlen=3 * args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.drafter_threshold,
            is_drafter=True,
            prev_prefill_output=prev_prefill_output,
            lowconf_threshold=args.lowconf_threshold,
            max_spec_len=args.max_spec_len,
            incr_len=args.incr_len,
            last_round_rejected=None,
        )
        draft_time = time.time() - draft_start
        actual_draft_time += draft_time

        if args.disable_reusing_drafter_kvs:
            draft_tokens, _, num_forward_passes, _ = draft_output
        else:
            draft_tokens, _, prev_prefill_output, num_forward_passes, _ = draft_output

        draft_tokens = draft_tokens[:args.max_new_tokens - len(current_draft_tokens)]
        if not draft_tokens:
            break

        verify_tokens, aligned = map_draft_tokens_to_verify_tokens(draft_tokens, drafter_tokenizer, verifier_tokenizer)
        drafter_full_input = torch.cat(
            [
                orig_drafter_inputs["input_ids"],
                torch.tensor([current_draft_tokens + draft_tokens], dtype=torch.long, device=drafter_device),
            ],
            dim=1,
        )
        verifier_full_input = torch.cat(
            [
                orig_verifier_inputs["input_ids"],
                torch.tensor([current_verify_tokens + verify_tokens], dtype=torch.long, device=verifier_device),
            ],
            dim=1,
        )

        verify_start = time.time()
        draft_all_logits = forward_logits(drafter_model, drafter_full_input)
        verify_all_logits = forward_logits(verifier_model, verifier_full_input)
        verify_time = time.time() - verify_start
        actual_verify_time += verify_time

        draft_start_idx = drafter_prompt_len + len(current_draft_tokens)
        draft_end_idx = draft_start_idx + len(draft_tokens)
        verify_start_idx = verifier_prompt_len + len(current_verify_tokens)
        verify_end_idx = verify_start_idx + len(verify_tokens)
        draft_logits = draft_all_logits[0, draft_start_idx:draft_end_idx, :]
        verify_logits = verify_all_logits[0, verify_start_idx:verify_end_idx, :].to(draft_logits.device)

        accepted_len, replacement_token, kl_scores = ddsd_kl_verify(
            draft_logits,
            verify_logits,
            draft_tokens,
            aligned,
            args.kl_threshold,
            args.temperature,
            projection,
        )

        accepted_draft = draft_tokens[:accepted_len]
        accepted_verify = verify_tokens[:accepted_len]
        tokens_to_append = list(accepted_draft)
        verify_tokens_to_append = list(accepted_verify)
        if replacement_token is not None and len(current_draft_tokens) + len(tokens_to_append) < args.max_new_tokens:
            replacement_text = drafter_tokenizer.decode([replacement_token], skip_special_tokens=False)
            replacement_verify_ids = verifier_tokenizer.encode(replacement_text, add_special_tokens=False)
            tokens_to_append.append(replacement_token)
            verify_tokens_to_append.append(replacement_verify_ids[0] if replacement_verify_ids else verifier_tokenizer.eos_token_id)

        current_draft_tokens.extend(tokens_to_append)
        current_verify_tokens.extend(verify_tokens_to_append)
        accepted_tokens += accepted_len
        total_drafted_tokens += len(draft_tokens)

        sequence_length = max(drafter_full_input.shape[1], verifier_full_input.shape[1])
        theo_draft_time += theoretical_seconds(drafter_params, sequence_length, max(num_forward_passes, 1), args.theoretical_tflops)
        theo_verify_time += theoretical_seconds(verifier_params, sequence_length, 1, args.theoretical_tflops)

        draft_text = drafter_tokenizer.decode(draft_tokens, skip_special_tokens=True)
        accepted_text = drafter_tokenizer.decode(accepted_draft, skip_special_tokens=True)
        print(f"Round {round_id}: draft={draft_text!r}")
        print(f"Round {round_id}: accepted={accepted_text!r}")
        print(f"Round {round_id}: kl={[round(score, 4) for score in kl_scores]}")

        round_id += 1
        if not tokens_to_append:
            break
        if drafter_tokenizer.eos_token_id in tokens_to_append:
            break

    actual_total_time = actual_draft_time + actual_verify_time
    draft_time_percentage = actual_draft_time / actual_total_time * 100.0 if actual_total_time else 0.0
    acceptance_rate = accepted_tokens / total_drafted_tokens * 100.0 if total_drafted_tokens else 0.0
    output_text = drafter_tokenizer.decode(current_draft_tokens, skip_special_tokens=True)
    print(f"Output: {output_text!r}")

    return {
        "problem_id": problem_id,
        "actual_draft_time": actual_draft_time,
        "actual_verify_time": actual_verify_time,
        "actual_total_time": actual_total_time,
        "theo_draft_time": theo_draft_time,
        "theo_verify_time": theo_verify_time,
        "draft_time_percentage": draft_time_percentage,
        "acceptance_rate": acceptance_rate,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dual_diffusion_results.csv"

    dataset = load_dataset("openai/gsm8k", "main")["test"]
    drafter_model, drafter_tokenizer = load_drafter(args)
    verifier_model, verifier_tokenizer = load_verifier(args)
    projection = build_draft_to_verify_projection(
        drafter_model,
        drafter_tokenizer,
        verifier_model,
        verifier_tokenizer,
        output_dir / "draft_to_verify_projection.pt",
    )
    drafter_params = model_parameter_count(drafter_model)
    verifier_params = model_parameter_count(verifier_model)

    end_index = min(args.start_index + args.num_questions, len(dataset))
    for problem_id in tqdm(range(args.start_index, end_index), desc="DDSD GSM8K"):
        row = run_problem(
            args,
            problem_id,
            dataset["question"][problem_id],
            drafter_model,
            drafter_tokenizer,
            verifier_model,
            verifier_tokenizer,
            projection,
            drafter_params,
            verifier_params,
        )
        append_csv_row(csv_path, row)


if __name__ == "__main__":
    main()
