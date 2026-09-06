"""Incremental speculative verification; the final committed token stays uncached."""
import torch
from transformers import DynamicCache


class VerifierKVCache:
    def __init__(self):
        self.cache = DynamicCache()
        self.pending = False
        self.context_length = None
        self.calls = 0
        self.input_tokens = 0
        self.full_prefix_tokens = 0
        self.last_full_length = 0

    def verify(self, model, full_input_ids, attention_mask, proposal_length):
        if self.pending:
            raise RuntimeError("Commit or discard the previous verifier result first.")
        if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
            raise ValueError("Verifier cache supports a single sequence only.")
        length = full_input_ids.shape[1]
        context = length - proposal_length
        if proposal_length <= 0 or context <= 0 or attention_mask.shape != full_input_ids.shape:
            raise ValueError("Invalid proposal/context/attention mask shape.")
        cached = self.cache.get_seq_length()
        if self.context_length is not None and (
            context != self.context_length or cached != context - 1
        ):
            raise RuntimeError("Verifier KV cache is not aligned with the committed prefix.")
        positions = torch.arange(cached, length, device=full_input_ids.device)
        # The full mask covers past + new tokens; position IDs preserve full-prefix semantics.
        with torch.inference_mode():
            outputs = model(
                input_ids=full_input_ids[:, cached:], attention_mask=attention_mask,
                past_key_values=self.cache, use_cache=True,
                cache_position=positions, position_ids=positions.unsqueeze(0),
                logits_to_keep=proposal_length + 1,
            )
        self.cache = outputs.past_key_values
        if self.cache is None or self.cache.get_seq_length() != length:
            raise RuntimeError("Verifier did not return a complete KV cache.")
        self.calls += 1
        self.input_tokens += length - cached
        self.full_prefix_tokens += length
        self.last_full_length = length
        self.context_length = context
        self.pending = True
        return outputs

    def commit(self, committed_length):
        if not self.pending:
            raise RuntimeError("No verifier result to commit.")
        if not self.context_length < committed_length <= self.last_full_length + 1:
            raise ValueError("Invalid committed length.")
        # Keep only accepted-prefix KVs. Correction/bonus is processed next round.
        # Also valid when EOS or the output limit truncates inside accepted tokens.
        self.cache.crop(committed_length - 1)
        if self.cache.get_seq_length() != committed_length - 1:
            raise RuntimeError("Verifier KV rollback failed.")
        self.context_length = committed_length
        self.pending = False

    def stats(self):
        return {"verify_calls": self.calls, "verifier_input_tokens": self.input_tokens,
                "full_prefix_input_tokens_without_cache": self.full_prefix_tokens,
                "retained_cache_tokens": self.cache.get_seq_length()}
