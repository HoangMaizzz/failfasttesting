"""Separate, untimed cache-vs-full-prefix audit before the benchmark."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16_two_gpu import qwen2_device_map
from verifier_kv_cache import VerifierKVCache


@torch.inference_mode()
def audit_prompt(model, prompt, reference_length=32):
    context = prompt.clone()
    reference = []
    for _ in range(reference_length):
        token = model(input_ids=context, attention_mask=torch.ones_like(context),
                      use_cache=False, logits_to_keep=1).logits[0, -1].argmax().item()
        reference.append(token)
        context = torch.cat([context, context.new_tensor([[token]])], dim=1)
    cache = VerifierKVCache()
    emitted = []
    rows = []
    # Rejection first/middle, all-accepted bonus, then terminal inside an accepted prefix.
    for scenario, size, reject, cap in [
        ("reject_first", 8, 0, None), ("reject_middle", 8, 3, None),
        ("all_accepted", 8, None, None), ("short_proposal", 4, None, None),
        ("terminal_truncation", 8, None, 2),
    ]:
        proposal = reference[len(emitted):len(emitted) + size]
        if reject is not None:
            proposal[reject] = (proposal[reject] + 1) % model.config.vocab_size
        full = torch.cat([prompt, prompt.new_tensor([emitted + proposal])], dim=1)
        expected = model(input_ids=full, attention_mask=torch.ones_like(full),
                         use_cache=False, logits_to_keep=size + 1).logits[0].argmax(-1).tolist()
        actual = cache.verify(model, full, torch.ones_like(full), size).logits[0].argmax(-1).tolist()
        if actual != expected:
            raise RuntimeError(f"KV cache/full-prefix argmax mismatch in {scenario}")
        accepted = 0
        while accepted < size and proposal[accepted] == actual[accepted]:
            accepted += 1
        addition = proposal[:accepted] + [actual[accepted]]
        if cap is not None:
            addition = addition[:cap]
        emitted.extend(addition)
        if emitted != reference[:len(emitted)]:
            raise RuntimeError(f"Speculative output differs from greedy reference in {scenario}")
        cache.commit(prompt.shape[1] + len(emitted))
        rows.append({"scenario": scenario, "accepted": accepted, "emitted": len(addition),
                     "argmax_match": True, "greedy_output_match": True, **cache.stats()})
    if cache.input_tokens >= cache.full_prefix_tokens:
        raise RuntimeError("Cache did not reduce processed verifier tokens.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--two_gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": False, "scope": "3 synthetic prompts, 5 rollback cases each; not a full lossless proof", "prompts": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.two_gpu and torch.cuda.device_count() < 2:
            raise RuntimeError("Two CUDA devices required.")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map=qwen2_device_map(args.model) if args.two_gpu else {"": 0},
        ).eval()
        for i, text in enumerate([
            "Compute 17 times 23. Show the calculation.",
            "A shop sold 12 apples then 7 more. How many apples were sold? " * 16,
            "Write a Python function to reverse a list without changing the input. " * 64,
        ]):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            rows = audit_prompt(model, prompt)
            report["prompts"].append({"prompt_tokens": prompt.shape[1], "cases": rows})
            print(f"KV audit prompt {i + 1}/3 passed ({prompt.shape[1]} input tokens)", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
