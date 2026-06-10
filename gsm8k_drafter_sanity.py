import argparse
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="/content/failfasttesting/Fast_dLLM_v2_1_5B")
    parser.add_argument("--allow_remote", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(question):
    return (
        "Solve the following math problem efficiently and clearly. "
        "Please reason step by step, separate logical reasoning steps with two newline characters, "
        "and put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )


def load_model(args):
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
            args.model_path,
            trust_remote_code=True,
            local_files_only=not args.allow_remote,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=not args.allow_remote,
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


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    question = dataset["question"][args.problem_id]
    answer = dataset["answer"][args.problem_id]
    prompt = format_prompt(question)
    model, tokenizer = load_model(args)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"Problem {args.problem_id}")
    print(question)
    print("\nExpected answer:")
    print(answer)
    print("\nModel output:")

    with torch.inference_mode():
        output = model.generate(
            inputs["input_ids"],
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            steps=args.steps,
        )

    output_ids = output[0] if isinstance(output, tuple) else output
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
