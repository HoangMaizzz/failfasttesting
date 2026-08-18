from typing import Callable, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from .configuration import Fast_dLLM_QwenConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange, repeat

logging.set_verbosity_error()
# logging.set_verbosity_warning()
# logging.set_verbosity_debug()

logger = logging.get_logger(__name__)


class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    


def get_rejected_overlap_info(last_round_rejected, curr_round_proposal):
    """
    Finds the longest suffix of curr_round_proposal that exists in last_round_rejected.
    
    Args:
        last_round_rejected (list): The list containing rejected tokens.
        curr_round_proposal (list): The list containing the current proposal tokens.

    Returns:
        tuple: (length_to_end, start_index_rejected, start_index_proposal)
        
        - length_to_end: Length from the match start in rejected to the end of the rejected list.
        - start_index_rejected: Index in last_round_rejected where the match begins.
        - start_index_proposal: Index in curr_round_proposal where the matching suffix begins.
        
        Returns (0, -1, -1) if no match is found.
    
    NOTE(ruipan): maybe we should add a constraint that the suffix should be at least length 2?

    Example:
        >>> last_rejected = [
        ...     1077, 594, 1477, 400, 69, 4080, 16, 15087, 1447, 41306, 
        ...     4080, 16, 8, 284, 1124, 37018, 90, 18, 4080, 16, 7287, 
        ...     17, 15170, 12, 16, 12, 17, 92, 284, 1124, 37018, 19999, 
        ...     18, 12, 17, 15170, 12, 18, 92, 284, 1124, 37018, 19999, 
        ...     20, 15170, 12, 18, 92, 284, 1124, 37018, 90
        ... ]
        >>> curr_proposal = [11, 1077, 594, 1477, 400, 69, 4080, 16, 15087, 1447]
        >>> get_rejected_overlap_info(last_rejected, curr_proposal)
        (53, 0, 1)

        # Explanation:
        # The longest matching suffix is [1077, 594, ..., 1447].
        # It starts at index 1 in curr_proposal.
        # It is found at index 0 in last_rejected.
        # Length from index 0 to the end of last_rejected (len 53) is 53.
    """
    len_proposal = len(curr_round_proposal)
    len_rejected = len(last_round_rejected)
    
    # Iterate through curr_round_proposal to create suffixes.
    # i represents the start index in the proposal list.
    for i in range(len_proposal):
        # Create the suffix x
        suffix = curr_round_proposal[i:]
        len_suffix = len(suffix)
        
        # Slide through last_round_rejected to find this suffix
        for j in range(len_rejected - len_suffix + 1):
            
            # Check if the slice matches the suffix
            if last_round_rejected[j : j + len_suffix] == suffix:
                
                # Found the suffix.
                # j is the index in rejected.
                # i is the index in proposal.
                length_to_end = len_rejected - j
                return length_to_end, j, i
                
    # Return defaults if no suffix matches
    return 0, -1, -1





@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None

@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """
    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal

def eval_block_diff_mask(q_idx, kv_idx, block_size=None):
    # Compute block indices
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size

    return block_q >= block_kv

class Fast_dLLM_QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Fast_dLLM_QwenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.training:
            #split q into two parts
            q_1 = query_states[:,:,:query_states.shape[2]//2]
            q_2 = query_states[:,:,query_states.shape[2]//2:]
            #split k into two parts
            k_1 = key_states[:,:,:key_states.shape[2]//2]
            k_2 = key_states[:,:,key_states.shape[2]//2:]
            q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)
            query_states = torch.cat((q_1, q_2), dim=-2)
            key_states = torch.cat((k_1, k_2), dim=-2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if block_past_key_values is not None:
            if len(block_past_key_values) <= self.layer_idx:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = block_past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]
                
                block_cache_key_states[:, :, replace_position:replace_position+key_states.shape[2]] = key_states
                block_cache_value_states[:, :, replace_position:replace_position+value_states.shape[2]] = value_states
                key_states = block_cache_key_states
                value_states = block_cache_value_states

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if update_past_key_values:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            elif len(past_key_value) > self.layer_idx:
                key_states = torch.cat((past_key_value[self.layer_idx][0], key_states), dim=-2)
                value_states = torch.cat((past_key_value[self.layer_idx][1], value_states), dim=-2)

        if self.training:
            attn_output = fused_flex_attention(query_states, key_states, value_states, mask=attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                is_causal=False,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # main diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

@use_kernel_forward_from_hub("RMSNorm")
class Fast_dLLM_QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Fast_dLLM_QwenRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Fast_dLLM_QwenDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Fast_dLLM_QwenAttention(config=config, layer_idx=layer_idx)

        self.mlp = Fast_dLLM_QwenMLP(config)
        self.input_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class Fast_dLLM_QwenPreTrainedModel(PreTrainedModel):
    config_class = Fast_dLLM_QwenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_QwenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Fast_dLLM_QwenDecoderLayer,
        "attentions": Fast_dLLM_QwenAttention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Fast_dLLM_QwenRMSNorm):
            module.weight.data.fill_(1.0)


class Fast_dLLM_QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: Fast_dLLM_QwenConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class Fast_dLLM_QwenModel(Fast_dLLM_QwenPreTrainedModel):
    def __init__(self, config: Fast_dLLM_QwenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_QwenRotaryEmbedding(config=config)
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def eval_mask(self, seqlen, block_size, cache_seq_len):
        q_indices = torch.arange(seqlen) + cache_seq_len
        k_indices = torch.arange(seqlen + cache_seq_len)
        mask = eval_block_diff_mask(
            q_idx=q_indices[:, None], 
            kv_idx=k_indices[None, :], 
            block_size=block_size
        )
        return mask

    def gen_mask(self, seqlen, block_size, B, H):
        mask = create_block_mask(
            partial(block_diff_mask, block_size=block_size, n=seqlen),
            B=B, H=H, Q_LEN=seqlen*2, KV_LEN=seqlen*2)

        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if use_block_cache and block_past_key_values is None:
            block_past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if self.training:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]//2, device=inputs_embeds.device
                )
            else:
                if use_block_cache:
                    block_start_position = past_seen_tokens+replace_position if replace_position is not None else past_seen_tokens
                    cache_position = torch.arange(
                        block_start_position, block_start_position + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                else:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1] if not self.training else inputs_embeds.shape[1]//2, device=inputs_embeds.device
                    )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        if self.training:
            attention_mask = self.gen_mask(labels.shape[1], self.bd_size, labels.shape[0], self.config.num_attention_heads).to(device=inputs_embeds.device)
        else:
            if use_block_cache and block_past_key_values.get_seq_length() != 0:
                attention_mask = None
            else:
                attention_mask = self.eval_mask(input_ids.shape[1], block_size, past_key_values.get_seq_length() if past_key_values is not None else 0).to(device=inputs_embeds.device)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                update_past_key_values=update_past_key_values,
                use_block_cache=use_block_cache,
                block_past_key_values=block_past_key_values,
                replace_position=replace_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )


class Fast_dLLM_QwenForCausalLM(Fast_dLLM_QwenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Fast_dLLM_QwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        mask_id: Optional[int] = 151665,
        **kwargs
    ) -> CausalLMOutputWithPastAndBlockCache:

        if self.training:
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()

            noisy_input_ids = input_ids.clone()

            input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            b, l = input_ids.shape
            t = torch.rand((b,), device=input_ids.device)
            eps=1e-3
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)

            mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
            x_t = torch.where(mask_indices, mask_id, input_ids).reshape(labels.shape)
            noisy_input_ids[labels != -100] = x_t[labels != -100]
            mask = (noisy_input_ids != mask_id)
            labels[mask] = -100
            input_ids = torch.cat([noisy_input_ids, input_ids.reshape(labels.shape)], dim=1)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()

            complementary_input_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)

            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(complementary_mask_indices, mask_id, complementary_input_ids).reshape(labels.shape)
            complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
            complementary_mask = (complementary_noisy_input_ids != mask_id)
            complementary_labels[complementary_mask] = -100
            complementary_input_ids = torch.cat([complementary_noisy_input_ids, complementary_input_ids.reshape(complementary_labels.shape)], dim=1)

            input_ids = torch.cat([input_ids, complementary_input_ids], dim=0)
            labels = torch.cat([labels, complementary_labels], dim=0)

        outputs: BaseModelOutputWithPastAndBlockCache = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_size=block_size,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.training:
            hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            block_past_key_values=outputs.block_past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens, 
        mask_id=151665,
        threshold=1,
        small_block_size=8,
        block_size=32,
        stop_token=151645,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0,
        use_block_cache=False,
        **kwargs
    ):
        """Vanilla generate of Fast-dLLM"""
        num_blocks = max_new_tokens // block_size
        original_input_length = input_ids.shape[1]

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            block_past_key_values = None
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]


                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        # Select tokens with probability greater than threshold from p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                        top2_probs = torch.topk(p_1t, k=2, dim=-1).values
                        x1_margin = (top2_probs[..., 0] - top2_probs[..., 1]).float()
                        x1_margin = torch.where(mask_idx[:, start:end], x1_margin, torch.zeros_like(x1_margin))

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

            input_ids = x_t
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        return input_ids


    @torch.no_grad()
    def generate_draft_tokens(
        self,
        input_ids,
        max_new_tokens, 
        mask_id=151665,
        threshold=1,
        small_block_size=8,
        block_size=32,
        stop_token=151645,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0,
        use_block_cache=False,
        is_drafter=False,  # breaks after finishing the first small block
        spec_len=None,  # verification frequency
        return_prefill_kvs=True,  # returns the KVs of tokens up to the largest multiple of block_size less than input seqlen
        prev_prefill_output=None,  # takes in the previous output to reuse KVs
        args=None,
        **kwargs
    ):
        """Generate n draft tokens"""
        num_blocks = max_new_tokens // block_size
        # if num_blocks == 1:
        #     logger.debug(f"{Colors.RED}Warning: <{max_new_tokens} tokens might be generated if only 1 block is needed. {Colors.RESET}")
        original_input_length = input_ids.shape[1]
        if is_drafter:
            draft_token_start_idx = original_input_length  # inclusive
            draft_token_end_idx = draft_token_start_idx + spec_len  # exclusive
            # logger.debug(f"{Colors.MAGENTA}draft tokens from {draft_token_start_idx} to {draft_token_end_idx} (in-block idx: {draft_token_start_idx % block_size} to {draft_token_end_idx % block_size}){Colors.RESET}")
        
        num_forward_passes = 0
        forward_pass_latencies = []  # ms
        draft_tokens_unmasked = False
        prefill_output = None
        
        if input_ids.shape[1] > block_size:  # NOTE(ruipan): this fwd pass populates the KV cache
            if prev_prefill_output is not None and prev_prefill_output.logits.shape[1] == (input_ids.shape[1] // block_size * block_size):
                logits, past_key_values = prev_prefill_output.logits, prev_prefill_output.past_key_values
                # logger.debug(f"{Colors.CYAN}Reusing previous drafter prefill output ({prev_prefill_output.logits.shape[1]} tokens) to populate KV cache, avoided a fwd pass!{Colors.RESET}")
                if return_prefill_kvs:
                    prefill_output = prev_prefill_output
            else:
                ##################Start of timer##################
                start_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                """
                NOTE(ruipan): this fwd pass only processes up to the 
                largest multiple of block_size less than input seqlen.
                E.g., if seqlen is 40, blk size is 32, only process first 32 tokens.
                This makes sure the KVs are attended to correctly (on complete blocks).
                """
                output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                forward_pass_latencies.append(start_time.elapsed_time(end_time))
                ##################End of timer##################
                logger.debug(f"{Colors.CYAN}Initial fwd pass to populate KV cache because input seqlen {input_ids.shape[1]} > block_size {block_size}{Colors.RESET}")
                # logger.debug(f"{Colors.CYAN}The prefill pass operated on tokens up until index {input_ids.shape[1] // block_size * block_size}{Colors.RESET}")
                num_forward_passes += 1
                logits, past_key_values = output.logits, output.past_key_values
                if input_ids.shape[1] % block_size == 0:
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    # NOTE(ruipan): because input len is a multiple of block_size, this prefill generated a bonus token
                if return_prefill_kvs:
                    prefill_output = output
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                logger.debug(f"{Colors.GREEN}Stopping generation as stop_token 1 {stop_token} found.{Colors.RESET}")
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            """
            NOTE(ruipan): pad the input with mask tokens up until the total length becomes a multiple of the block size
            as a result, x_init len is always a multiple of block_size.
            """
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            # logger.debug(f"{Colors.MAGENTA}Padded {block_size-prompt_length%block_size} tokens, seqlen is now {x_init.shape[1]}{Colors.RESET}")
            
            x_t = x_init.clone()
            block_past_key_values = None
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        logger.debug(f"{Colors.RED}Warning: stop token found in prompt!{Colors.RESET}")
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:  # NOTE(ruipan): no mask tokens left in the current block
                    ##################Start of timer##################
                    start_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    end_time = torch.cuda.Event(enable_timing=True)
                    end_time.record()
                    torch.cuda.synchronize()
                    forward_pass_latencies.append(start_time.elapsed_time(end_time))
                    ##################End of timer##################
                    num_forward_passes += 1
                    logger.debug(f"{Colors.CYAN}Doing 1 more fwd pass because mask_idx.sum() == 0{Colors.RESET}")
                    
                    logits, past_key_values = output.logits, output.past_key_values
                    if return_prefill_kvs:
                        prefill_output = output
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    # NOTE(ruipan): probably updating the KVs because this concluded a big block
                    logger.debug(f"{Colors.CYAN}Decoded token {next_token.tolist()[0]} ({args.target_tokenizer.decode(next_token.tolist()[0])}) while updating KVs{Colors.RESET}")
                    break
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    logger.debug(f"{Colors.MAGENTA}small_block_start_idx {small_block_start_idx}, small_block_end_idx {small_block_end_idx}, start {start}, end {end}{Colors.RESET}")
                    while True:
                        if is_drafter and draft_tokens_unmasked:  # NOTE(ruipan): all draft tokens have been unmasked in the last iter
                            break
                        mask_idx = (x_t[:, -block_size:] == mask_id)  # NOTE(ruipan): only look at the last big block
                        # logger.debug(f"{Colors.MAGENTA}Block {block_idx}, small block {small_block_idx}, current is_mask: {mask_idx}{Colors.RESET}")  # XXX
                        if mask_idx[:, start:end].sum() == 0:
                            # logger.debug(f"{Colors.MAGENTA}No mask tokens in the current small block, moving on to the next block{Colors.RESET}")
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                logger.debug(f"{Colors.MAGENTA}breaking after x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0 {Colors.RESET}")
                                break
                        
                        ##################Start of timer##################
                        start_time = torch.cuda.Event(enable_timing=True)
                        start_time.record()
                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            # logger.debug(f"{Colors.GREEN}input: x_t[:, -block_size:] {x_t[:, -block_size:]}{Colors.RESET}")
                            # logger.debug(f"{Colors.GREEN}past_key_values len {past_key_values.key_cache[0].shape}{Colors.RESET}")
                            # logger.debug(f"{Colors.CYAN}past_key_values.to_legacy_cache() {past_key_values.to_legacy_cache()}{Colors.RESET}")
                            logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            # NOTE(ruipan): a one-token right-shift of logits
                            # [t0, t1, t2, t3] -> [t0, t0, t1, t2]
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]  # NOTE(ruipan): only looking at logits in the current small block region
                        end_time = torch.cuda.Event(enable_timing=True)
                        end_time.record()
                        torch.cuda.synchronize()
                        forward_pass_latencies.append(start_time.elapsed_time(end_time))
                        ##################End of timer##################
                        num_forward_passes += 1
                        logger.debug(f"{Colors.CYAN}block {block_idx}, small_block_idx {small_block_idx}, fwd pass. start {start}, end {end}.{Colors.RESET}")
                        
                        # logger.debug(f"{Colors.MAGENTA}logits {logits}{Colors.RESET}")

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)  # XXX only mini_block (8) tokens
                        # Select tokens with probability greater than threshold from p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]  # only allowed to update MASK tokens
                        # NOTE(ruipan): seems that forward() itself does not guarantee not modifying the non-mask tokens

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        if is_drafter and draft_token_end_idx <= x_t.shape[1] \
                            and x_t[:, draft_token_start_idx:draft_token_end_idx].ne(mask_id).all():
                            logger.debug(f"{Colors.GREEN}All draft tokens unmasked. draft_token_start_idx {draft_token_start_idx}, draft_token_end_idx {draft_token_end_idx}, is not mask: {x_t[:, draft_token_start_idx:draft_token_end_idx].ne(mask_id)}{Colors.RESET}")
                            # none of the first n tokens after the original input tokens are mask tokens
                            draft_tokens_unmasked = True
                            # NOTE(ruipan): potential index error here if draft_token_end_idx is beyond the current block
                            # want to also make sure draft_token_end_idx is not beyond the current length of x_t

                    if is_drafter and draft_tokens_unmasked:
                        logger.debug(f"{Colors.YELLOW}Exiting after {num_forward_passes} fwd passes{Colors.RESET}")
                        input_ids = x_t
                        break
                if is_drafter and draft_tokens_unmasked:
                    break
                
            input_ids = x_t
            
            if is_drafter and draft_tokens_unmasked:
                break
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        logger.debug(f"{Colors.YELLOW}forward_pass_latencies: {[f'{latency:.2f}ms' for latency in forward_pass_latencies]}{Colors.RESET}")
        
        if return_prefill_kvs:
            return input_ids, prefill_output, num_forward_passes, forward_pass_latencies
        return input_ids, num_forward_passes, forward_pass_latencies


    @torch.no_grad()
    def generate_draft_tokens_arbitrary_length(
        self,
        input_ids,
        max_new_tokens, 
        mask_id=151665,
        threshold=1,
        small_block_size=8,
        block_size=32,
        stop_token=151645,
        stopping_criteria=None,
        top_p=0.95,
        temperature=0,
        use_block_cache=False,
        is_drafter=False,  # breaks after finishing the first small block
        spec_len=None,  # verification frequency
        return_prefill_kvs=True,  # returns the KVs of tokens up to the largest multiple of block_size less than input seqlen
        prev_prefill_output=None,  # takes in the previous output to reuse KVs
        args=None,
        lowconf_threshold=None,
        max_spec_len=None,
        incr_len=None,
        last_round_rejected=None,  # rejected tokens from the last round (that might be reusable)
        return_frontier_stats=False,
        **kwargs
    ):
        """Generate n draft tokens, where n is dynamically determined"""
        num_blocks = max_new_tokens // block_size
        if num_blocks == 1:
            logger.debug(f"{Colors.RED}Warning: <{max_new_tokens} tokens might be generated if only 1 block is needed. {Colors.RESET}")
        original_input_length = input_ids.shape[1]
        if is_drafter:
            draft_token_start_idx = original_input_length  # inclusive
            draft_token_end_idx = draft_token_start_idx + spec_len  # exclusive
        
        num_forward_passes = 0
        forward_pass_latencies = []  # ms
        draft_tokens_unmasked = False
        prefill_output = None
        conf_of_unmasked_tokens = []
        frontier_mode = getattr(args, "frontier_stop_mode", "disabled") if args is not None else "disabled"
        frontier_stats = {
            "enabled": bool(return_frontier_stats),
            "mode": frontier_mode,
            "score_type": "expected_accepted_prefix",
            "initial_spec_len": int(spec_len),
            "max_spec_len": int(max_spec_len),
            "incr_len": int(incr_len),
            "lowconf_threshold": None if lowconf_threshold is None else float(lowconf_threshold),
            "steps": [],
            "stop_reason": None,
            "final_frontier_score": None,
            "actual_spec_len": None,
            "draft_token_stats": [],
            "refinement_actions": [],
            "extension_events": [],
            "forward_pass_breakdown": {
                "prefill": 0,
                "cache_update": 0,
                "denoising": 0,
                "fill": 0,
                "total": 0,
            },
        }
        committed_confidences = {}
        committed_margins = {}
        committed_forced = {}
        filled_on_stop_positions = set()
        frontier_scores = []
        frontier_recent_unmasked = []
        frontier_force_stop = False
        frontier_force_extend = False
        accept_probabilities = None
        probability_variants = None

        def frontier_bin(value):
            return str(max(0, min(9, int(float(value) * 10.0))))

        def acceptance_probability_variants(confidence, margin, forced):
            confidence = max(0.0, min(1.0, float(confidence)))
            margin = max(0.0, min(1.0, float(margin)))
            calibration = getattr(args, "frontier_acceptance_calibration", None) if args is not None else None
            raw_confidence = max(0.02, min(0.98, confidence))
            if not calibration:
                return {
                    "full_calibration": raw_confidence,
                    "raw_confidence": raw_confidence,
                    "without_confidence_bucket": raw_confidence,
                    "without_margin_bucket": raw_confidence,
                    "without_forced_bucket": raw_confidence,
                }

            prior = float(getattr(args, "frontier_calibration_prior", 0.5)) if args is not None else 0.5
            prior_count = float(getattr(args, "frontier_calibration_prior_count", 2.0)) if args is not None else 2.0
            bucket_values = []

            for table_name, key in (
                ("confidence_bins", frontier_bin(confidence)),
                ("margin_bins", frontier_bin(margin)),
                ("forced", "1" if forced else "0"),
            ):
                accepted, total = calibration.get(table_name, {}).get(key, (0.0, 0.0))
                total = float(total)
                if total > 0:
                    estimate = (float(accepted) + prior * prior_count) / (total + prior_count)
                    bucket_values.append((
                        table_name,
                        max(0.02, min(0.98, estimate)),
                        min(4.0, total),
                    ))

            def weighted_probability(excluded_bucket=None):
                weighted_sum = raw_confidence
                total_weight = 1.0
                for table_name, estimate, weight in bucket_values:
                    if table_name == excluded_bucket:
                        continue
                    weighted_sum += estimate * weight
                    total_weight += weight
                return weighted_sum / total_weight

            return {
                "full_calibration": weighted_probability(),
                "raw_confidence": raw_confidence,
                "without_confidence_bucket": weighted_probability("confidence_bins"),
                "without_margin_bucket": weighted_probability("margin_bins"),
                "without_forced_bucket": weighted_probability("forced"),
            }

        def calibrated_acceptance_probability(confidence, margin, forced):
            return acceptance_probability_variants(confidence, margin, forced)["full_calibration"]

        def expected_prefix_score(confidences, margins, forced_flags):
            score = 0.0
            survival = 1.0
            probabilities = []
            probabilities_by_variant = {
                "full_calibration": [],
                "raw_confidence": [],
                "without_confidence_bucket": [],
                "without_margin_bucket": [],
                "without_forced_bucket": [],
            }
            for confidence, margin, forced in zip(confidences, margins, forced_flags):
                variants = acceptance_probability_variants(confidence, margin, forced)
                accept_prob = variants["full_calibration"]
                probabilities.append(accept_prob)
                for name, probability in variants.items():
                    probabilities_by_variant[name].append(probability)
                survival *= accept_prob
                score += survival
            return score, probabilities, probabilities_by_variant

        def calibration_prior_probability():
            calibration = getattr(args, "frontier_acceptance_calibration", None) if args is not None else None
            prior = float(getattr(args, "frontier_calibration_prior", 0.5)) if args is not None else 0.5
            prior_count = float(getattr(args, "frontier_calibration_prior_count", 2.0)) if args is not None else 2.0
            if not calibration:
                return prior
            accepted, total = calibration.get("prior", (0.0, 0.0))
            return max(0.02, min(0.98, (float(accepted) + prior * prior_count) / (float(total) + prior_count)))

        def record_extension_event(
            trigger,
            extension,
            accept_probabilities=None,
            probability_variants=None,
            extension_details=None,
        ):
            extension = int(extension)
            if extension <= 0:
                return
            from_len = int(spec_len)
            event = {
                "trigger": trigger,
                "from_len": from_len,
                "to_len": from_len + extension,
                "extension_size": extension,
                "predicted_extension_gain": None,
                "prefix_survival_probability": None,
                "extension_prior_probability": None,
                "accept_probabilities": None,
                "prefix_survival_by_variant": None,
                "predicted_extension_gain_by_variant": None,
            }
            if extension_details:
                event.update(extension_details)
            if accept_probabilities is not None:
                extension_probability = calibration_prior_probability()
                event.update({
                    "extension_prior_probability": float(extension_probability),
                    "accept_probabilities": list(accept_probabilities),
                })
                if probability_variants is not None:
                    prefix_survival_by_variant = {}
                    predicted_gain_by_variant = {}
                    extension_multiplier = sum(
                        extension_probability ** offset
                        for offset in range(1, extension + 1)
                    )
                    for name, probabilities in probability_variants.items():
                        survival = 1.0
                        for probability in probabilities:
                            survival *= probability
                        prefix_survival_by_variant[name] = float(survival)
                        predicted_gain_by_variant[name] = float(survival * extension_multiplier)
                    event["prefix_survival_by_variant"] = prefix_survival_by_variant
                    event["predicted_extension_gain_by_variant"] = predicted_gain_by_variant
            frontier_stats["extension_events"].append(event)

        def v2_position_bin(position):
            if position < 2:
                return "0-1"
            if position < 4:
                return "2-3"
            if position < 8:
                return "4-7"
            return "8+"

        def v2_length_bin(length):
            return str(max(1, (int(length) + 7) // 8))

        def v2_context_bin(context_len):
            return str(max(0, int(context_len) // 256))

        def v2_posterior(table, key, prior, prior_strength, min_observations):
            accepted, total = table.get(key, (0.0, 0.0))
            total = float(total)
            if total < min_observations:
                return None
            probability = (float(accepted) + prior_strength * prior) / (total + prior_strength)
            return max(0.02, min(0.98, probability))

        def v2_token_hazard(confidence, margin, position):
            raw_probability = max(0.02, min(0.98, float(confidence)))
            calibration = getattr(args, "frontier_v2_hazard_calibration", {}) if args is not None else {}
            min_observations = int(getattr(args, "frontier_v2_min_hazard_observations", 8))
            prior_strength = float(getattr(args, "frontier_v2_hazard_prior_strength", 8.0))
            confidence_key = frontier_bin(confidence)
            margin_key = frontier_bin(margin)
            position_key = v2_position_bin(position)
            candidates = (
                (
                    "token_position_confidence_margin",
                    f"{position_key}:{confidence_key}:{margin_key}",
                ),
                ("token_confidence_margin", f"{confidence_key}:{margin_key}"),
                ("token_position_confidence", f"{position_key}:{confidence_key}"),
                ("token_confidence", confidence_key),
            )
            for table_name, key in candidates:
                probability = v2_posterior(
                    calibration.get(table_name, {}),
                    key,
                    raw_probability,
                    prior_strength,
                    min_observations,
                )
                if probability is not None:
                    return probability
            return raw_probability

        def v2_expected_prefix_score(confidences, margins):
            probabilities = [
                v2_token_hazard(confidence, margin, position)
                for position, (confidence, margin) in enumerate(zip(confidences, margins))
            ]
            score = 0.0
            survival = 1.0
            for probability in probabilities:
                survival *= probability
                score += survival
            return score, probabilities

        def v2_calibration_ready():
            calibration = getattr(args, "frontier_v2_hazard_calibration", {}) if args is not None else {}
            total = int(calibration.get("total_checked_tokens", 0))
            required = int(getattr(args, "frontier_v2_min_calibration_tokens", 64))
            return total >= required, total

        def v2_extension_hazards(from_len, extension):
            calibration = getattr(args, "frontier_v2_hazard_calibration", {}) if args is not None else {}
            min_observations = int(getattr(args, "frontier_v2_min_hazard_observations", 8))
            prior_strength = float(getattr(args, "frontier_v2_extension_prior_strength", 2.0))
            block_key = v2_length_bin(from_len)
            hazards = []
            for offset in range(1, extension + 1):
                probability = v2_posterior(
                    calibration.get("extension_block_offset", {}),
                    f"{block_key}:{offset}",
                    0.5,
                    prior_strength,
                    min_observations,
                )
                if probability is None:
                    probability = v2_posterior(
                        calibration.get("extension_offset", {}),
                        str(offset),
                        0.5,
                        prior_strength,
                        min_observations,
                    )
                if probability is None:
                    return None
                hazards.append(probability)
            return hazards

        def v2_calibrated_prefix_survival(raw_probability, from_len):
            calibration = getattr(args, "frontier_v2_hazard_calibration", {}) if args is not None else {}
            min_observations = int(
                getattr(args, "frontier_v2_min_prefix_calibration_observations", 8)
            )
            prior_strength = float(
                getattr(args, "frontier_v2_prefix_calibration_prior_strength", 8.0)
            )
            probability_key = frontier_bin(raw_probability)
            candidates = (
                (
                    "block",
                    calibration.get("extension_prefix_block_probability", {}).get(
                        f"{v2_length_bin(from_len)}:{probability_key}"
                    ),
                ),
                (
                    "global",
                    calibration.get("extension_prefix_probability", {}).get(probability_key),
                ),
            )
            for source, stats in candidates:
                if not stats:
                    continue
                accepted, total = map(float, stats)
                if total < min_observations:
                    continue
                probability = (accepted + prior_strength * raw_probability) / (
                    total + prior_strength
                )
                return max(0.02, min(0.98, probability)), source, int(total)
            return raw_probability, "uncalibrated", 0

        def v2_conditional_gain_correction(from_len):
            calibration = getattr(args, "frontier_v2_hazard_calibration", {}) if args is not None else {}
            min_observations = int(
                getattr(args, "frontier_v2_min_gain_calibration_observations", 8)
            )
            prior_strength = float(
                getattr(args, "frontier_v2_gain_calibration_prior_strength", 8.0)
            )
            candidates = (
                (
                    "block",
                    calibration.get("extension_conditional_gain_by_block", {}).get(
                        v2_length_bin(from_len)
                    ),
                ),
                ("global", calibration.get("extension_conditional_gain_global")),
            )
            for source, stats in candidates:
                if not stats:
                    continue
                predicted_sum, actual_sum, count = map(float, stats)
                if count < min_observations or predicted_sum <= 0.0:
                    continue
                observed_ratio = actual_sum / predicted_sum
                weight = count / (count + prior_strength)
                correction = 1.0 + weight * (observed_ratio - 1.0)
                return max(0.5, min(2.0, correction)), source, int(count)
            return 1.0, "uncalibrated", 0

        def v2_nearest_latency(table, context_len, proposal_len=None):
            if not table:
                return None
            context_index = int(v2_context_bin(context_len))
            proposal_index = int(v2_length_bin(proposal_len)) if proposal_len is not None else None
            best = None
            for key, value in table.items():
                parts = str(key).split(":")
                key_context = int(parts[0])
                key_proposal = int(parts[1]) if len(parts) > 1 else None
                distance = abs(key_context - context_index) * 4
                if proposal_index is not None and key_proposal is not None:
                    distance += abs(key_proposal - proposal_index)
                latency = float(value[0])
                count = int(value[1])
                candidate = (distance, -count, latency)
                if best is None or candidate < best:
                    best = candidate
            return None if best is None else best[2]

        def v2_latency_estimates(proposal_len):
            recent_forward_ms = forward_pass_latencies[-1] if forward_pass_latencies else 6.1
            context_len = int(getattr(args, "frontier_current_context_len", input_ids.shape[1]))
            draft_forward_ms = v2_nearest_latency(
                getattr(args, "frontier_v2_draft_latency_bins", {}),
                context_len,
            )
            if draft_forward_ms is None:
                draft_forward_ms = float(
                    getattr(args, "frontier_ema_dllm_forward_ms", None) or recent_forward_ms
                )
            verify_round_ms = v2_nearest_latency(
                getattr(args, "frontier_v2_verify_latency_bins", {}),
                context_len,
                proposal_len,
            )
            if verify_round_ms is None:
                verify_round_ms = getattr(args, "frontier_ema_target_round_ms", None)
            if verify_round_ms is None:
                verify_round_ms = max(draft_forward_ms, 13.5)
            controller_ms = float(getattr(args, "frontier_ema_controller_ms", None) or 0.0)
            return (
                max(float(draft_forward_ms), 1e-6),
                max(float(verify_round_ms), 1e-6),
                max(controller_ms, 0.0),
            )
        
        if input_ids.shape[1] > block_size:
            if prev_prefill_output is not None and prev_prefill_output.logits.shape[1] == (input_ids.shape[1] // block_size * block_size):
                logits, past_key_values = prev_prefill_output.logits, prev_prefill_output.past_key_values
                if return_prefill_kvs:
                    prefill_output = prev_prefill_output
            else:
                ##################Start of timer##################
                start_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                forward_pass_latencies.append(start_time.elapsed_time(end_time))
                ##################End of timer##################
                # logger.debug(f"{Colors.CYAN}Initial fwd pass to populate KV cache because input seqlen {input_ids.shape[1]} > block_size {block_size}{Colors.RESET}")
                # logger.debug(f"{Colors.CYAN}The prefill pass operated on tokens up until index {input_ids.shape[1] // block_size * block_size}{Colors.RESET}")
                num_forward_passes += 1
                frontier_stats["forward_pass_breakdown"]["prefill"] += 1
                logits, past_key_values = output.logits, output.past_key_values
                if input_ids.shape[1] % block_size == 0:
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                if return_prefill_kvs:
                    prefill_output = output
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                logger.debug(f"{Colors.GREEN}Stopping generation as stop_token 1 {stop_token} found.{Colors.RESET}")
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            
            x_t = x_init.clone()
            block_past_key_values = None
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        logger.debug(f"{Colors.RED}Warning: stop token found in prompt.{Colors.RESET}")
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:
                    ##################Start of timer##################
                    start_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    end_time = torch.cuda.Event(enable_timing=True)
                    end_time.record()
                    torch.cuda.synchronize()
                    forward_pass_latencies.append(start_time.elapsed_time(end_time))
                    ##################End of timer##################
                    num_forward_passes += 1
                    frontier_stats["forward_pass_breakdown"]["cache_update"] += 1
                    logger.debug(f"{Colors.CYAN}Doing 1 more fwd pass because mask_idx.sum() == 0{Colors.RESET}")
                    
                    logits, past_key_values = output.logits, output.past_key_values
                    if return_prefill_kvs:
                        prefill_output = output
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    logger.debug(f"{Colors.CYAN}Decoded token {next_token.tolist()[0]} ({args.target_tokenizer.decode(next_token.tolist()[0])}) while updating KVs{Colors.RESET}")
                    break
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    # logger.debug(f"{Colors.MAGENTA}small_block_start_idx {small_block_start_idx}, small_block_end_idx {small_block_end_idx}, start {start}, end {end}{Colors.RESET}")
                    while True:
                        if is_drafter and draft_tokens_unmasked:
                            break
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            # logger.debug(f"{Colors.MAGENTA}No mask tokens in the current small block, moving on to the next block{Colors.RESET}")
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                # logger.debug(f"{Colors.MAGENTA}breaking after x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0 {Colors.RESET}")
                                break
                        
                        ##################Start of timer##################
                        start_time = torch.cuda.Event(enable_timing=True)
                        start_time.record()
                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]  # NOTE(ruipan): only looking at logits in the current small block region
                        end_time = torch.cuda.Event(enable_timing=True)
                        end_time.record()
                        torch.cuda.synchronize()
                        forward_pass_latencies.append(start_time.elapsed_time(end_time))
                        ##################End of timer##################
                        num_forward_passes += 1
                        frontier_stats["forward_pass_breakdown"]["denoising"] += 1
                        logger.debug(f"{Colors.CYAN}block {block_idx}, small_block_idx {small_block_idx}, fwd pass. start {start}, end {end}.{Colors.RESET}")
                        
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        # Select tokens with probability greater than threshold from p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                        top2_probs = torch.topk(p_1t, k=2, dim=-1).values
                        x1_margin = (top2_probs[..., 0] - top2_probs[..., 1]).float()
                        x1_margin = torch.where(mask_idx[:, start:end], x1_margin, torch.zeros_like(x1_margin))
                        
                        # record the confidence of already-unmasked tokens
                        conf_of_unmasked_tokens.extend(x1_p[x1_p != -torch.inf].float().cpu().numpy().tolist())  # FIXME TODO(ruipan): n=5, small blk size 8 -- might include non-drafted tokens?!

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]  # only allowed to update MASK tokens

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        if is_drafter and return_frontier_stats and frontier_stats["mode"] not in ("disabled", "none", "off") and lowconf_threshold is not None and draft_token_end_idx <= x_t.shape[1]:
                            frontier_mode = getattr(args, "frontier_stop_mode", "disabled") if args is not None else "disabled"
                            tau_f = float(lowconf_threshold)
                            target_len = int(spec_len)
                            draft_end_idx = draft_token_start_idx + target_len
                            block_abs_start = x_t.shape[1] - block_size
                            current_step_confidences = {}
                            current_step_margins = {}
                            unmasked_this_step = int(unmask_idx.sum().item())

                            for batch_idx, local_idx in unmask_idx.nonzero(as_tuple=False).tolist():
                                if batch_idx != 0:
                                    continue
                                absolute_pos = block_abs_start + small_block_start_idx + local_idx
                                if draft_token_start_idx <= absolute_pos < draft_end_idx:
                                    committed_confidences[int(absolute_pos)] = float(x1_p[batch_idx, local_idx].float().item())
                                    committed_margins[int(absolute_pos)] = float(x1_margin[batch_idx, local_idx].float().item())
                                    committed_forced[int(absolute_pos)] = bool(x1_p[batch_idx, local_idx].float().item() <= threshold)

                            for local_idx in range(x1_p.shape[1]):
                                absolute_pos = block_abs_start + small_block_start_idx + local_idx
                                if draft_token_start_idx <= absolute_pos < draft_end_idx:
                                    current_step_confidences[int(absolute_pos)] = float(max(x1_p[0, local_idx].float().item(), 0.0))
                                    current_step_margins[int(absolute_pos)] = float(max(x1_margin[0, local_idx].float().item(), 0.0))

                            confidences = []
                            margins = []
                            forced_flags = []
                            recoverable = []
                            for absolute_pos in range(draft_token_start_idx, draft_end_idx):
                                if absolute_pos in committed_confidences:
                                    confidences.append(committed_confidences[absolute_pos])
                                    margins.append(committed_margins.get(absolute_pos, 0.0))
                                    forced_flags.append(committed_forced.get(absolute_pos, False))
                                    recoverable.append(False)
                                elif x_t[0, absolute_pos].item() != mask_id:
                                    confidences.append(1.0)
                                    margins.append(1.0)
                                    forced_flags.append(False)
                                    recoverable.append(False)
                                elif absolute_pos in current_step_confidences:
                                    confidences.append(current_step_confidences[absolute_pos])
                                    margins.append(current_step_margins.get(absolute_pos, 0.0))
                                    forced_flags.append(False)
                                    recoverable.append(True)
                                else:
                                    confidences.append(0.0)
                                    margins.append(0.0)
                                    forced_flags.append(False)
                                    recoverable.append(True)

                            frontier_k = 0
                            for confidence in confidences:
                                if confidence >= tau_f:
                                    frontier_k += 1
                                else:
                                    break

                            frontier_score, accept_probabilities, probability_variants = expected_prefix_score(
                                confidences,
                                margins,
                                forced_flags,
                            )
                            v2_ready, v2_calibration_tokens = v2_calibration_ready()
                            if frontier_mode == "cost_aware_v2":
                                frontier_score, accept_probabilities = v2_expected_prefix_score(
                                    confidences,
                                    margins,
                                )
                            if frontier_k >= target_len:
                                frontier_confidence = None
                                frontier_recoverable = False
                            else:
                                frontier_confidence = confidences[frontier_k]
                                frontier_recoverable = recoverable[frontier_k]

                            previous_score = frontier_scores[-1] if frontier_scores else None
                            frontier_gain = None if previous_score is None else frontier_score - previous_score
                            frontier_scores.append(frontier_score)
                            frontier_recent_unmasked.append(unmasked_this_step)
                            if args is not None:
                                frontier_recent_unmasked = frontier_recent_unmasked[-max(1, int(getattr(args, "frontier_patience", 2))):]

                            masks_remaining = int((x_t[:, draft_token_start_idx:draft_end_idx] == mask_id).sum().item())
                            step_record = {
                                "step": len(frontier_scores),
                                "target_len": target_len,
                                "frontier_k": int(frontier_k),
                                "frontier_score": float(frontier_score),
                                "frontier_gain": None if frontier_gain is None else float(frontier_gain),
                                "frontier_confidence": frontier_confidence,
                                "frontier_recoverable": bool(frontier_recoverable),
                                "expected_accept_prob_frontier": None if frontier_k >= target_len else float(accept_probabilities[frontier_k]),
                                "unmasked_this_step": unmasked_this_step,
                                "masks_remaining": masks_remaining,
                                "v2_hazard_ready": bool(v2_ready),
                                "v2_calibration_tokens": int(v2_calibration_tokens),
                            }
                            frontier_stats["steps"].append(step_record)
                            frontier_stats["final_frontier_score"] = float(frontier_score)

                            min_steps = int(getattr(args, "frontier_min_steps", 2)) if args is not None else 2
                            patience = int(getattr(args, "frontier_patience", 2)) if args is not None else 2
                            gain_epsilon = float(getattr(args, "frontier_gain_epsilon", 0.0)) if args is not None else 0.0
                            max_unmask = 1
                            cost_token_equiv = float(getattr(args, "frontier_dynamic_cost_token_equiv", getattr(args, "frontier_cost_token_equiv", 0.2))) if args is not None else 0.2
                            aggressive_irrecoverable = bool(getattr(args, "frontier_aggressive_irrecoverable", False)) if args is not None else False
                            force_stop_modes = ("mask_efficiency", "frontier", "cost_aware_no_extend")

                            stop_reason = None
                            frontier_force_extend = False
                            predicted_gain = None
                            if frontier_gain is not None:
                                if len(frontier_scores) >= 3:
                                    prev_gain = frontier_scores[-2] - frontier_scores[-3]
                                    ratio = max(0.0, min(1.0, frontier_gain / max(prev_gain, 1e-12)))
                                    predicted_gain = max(0.0, frontier_gain * ratio)
                                else:
                                    predicted_gain = max(0.0, frontier_gain)
                            step_record["predicted_next_gain"] = (
                                None if predicted_gain is None else float(predicted_gain)
                            )
                            if frontier_k >= target_len and frontier_mode in force_stop_modes:
                                stop_reason = "frontier_all_pass"
                            elif aggressive_irrecoverable and frontier_k < target_len and not frontier_recoverable and frontier_confidence is not None and frontier_confidence < tau_f:
                                stop_reason = "frontier_irrecoverable_low_conf"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "mask_efficiency":
                                if len(frontier_recent_unmasked) >= patience and all(x <= max_unmask for x in frontier_recent_unmasked[-patience:]):
                                    stop_reason = "mask_efficiency_stall"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "frontier":
                                if frontier_gain is not None and frontier_gain <= gain_epsilon and unmasked_this_step <= max_unmask:
                                    stop_reason = "frontier_stall"
                            elif len(frontier_scores) >= min_steps and frontier_mode in ("cost_aware", "cost_aware_no_extend"):
                                if predicted_gain is not None:
                                    if predicted_gain <= cost_token_equiv:
                                        stop_reason = "cost_aware_low_expected_gain"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "cost_aware_v2":
                                draft_forward_ms, verify_round_ms, controller_ms = v2_latency_estimates(target_len)
                                elapsed_draft_ms = float(sum(forward_pass_latencies))
                                expected_output = 1.0 + float(frontier_score)
                                hysteresis = float(getattr(args, "frontier_v2_hysteresis", 0.03))
                                extension_cost_margin = float(
                                    getattr(args, "frontier_v2_extension_cost_margin", -0.03)
                                )
                                stop_fill_ms = draft_forward_ms if masks_remaining > 0 else 0.0
                                stop_ms_per_output = (
                                    elapsed_draft_ms + stop_fill_ms + verify_round_ms + controller_ms
                                ) / max(expected_output, 1e-6)
                                step_record.update({
                                    "v2_expected_output": expected_output,
                                    "v2_draft_forward_ms": draft_forward_ms,
                                    "v2_verify_round_ms": verify_round_ms,
                                    "v2_controller_ms": controller_ms,
                                    "v2_stop_ms_per_output": stop_ms_per_output,
                                    "v2_fallback": not v2_ready,
                                })

                                if v2_ready and masks_remaining > 0 and predicted_gain is not None:
                                    expected_unmask = max(
                                        1.0,
                                        sum(frontier_recent_unmasked[-patience:])
                                        / max(1, len(frontier_recent_unmasked[-patience:])),
                                    )
                                    next_fill_ms = draft_forward_ms if masks_remaining > expected_unmask else 0.0
                                    next_expected_output = expected_output + float(predicted_gain)
                                    continue_ms_per_output = (
                                        elapsed_draft_ms
                                        + draft_forward_ms
                                        + next_fill_ms
                                        + verify_round_ms
                                        + controller_ms
                                    ) / max(next_expected_output, 1e-6)
                                    step_record["v2_continue_ms_per_output"] = continue_ms_per_output
                                    if continue_ms_per_output >= stop_ms_per_output * (1.0 - hysteresis):
                                        stop_reason = "cost_aware_v2_verify_lower_cost"
                                elif v2_ready and masks_remaining == 0 and frontier_k >= target_len:
                                    extension = min(incr_len, max_spec_len - spec_len)
                                    extension_hazards = v2_extension_hazards(target_len, extension)
                                    step_record["v2_extension_history_ready"] = extension_hazards is not None
                                    if extension <= 0:
                                        stop_reason = "cost_aware_v2_max_spec_len"
                                    elif extension_hazards is not None:
                                        raw_prefix_survival = 1.0
                                        for probability in accept_probabilities:
                                            raw_prefix_survival *= probability
                                        calibrated_prefix_survival, prefix_calibration_source, prefix_calibration_count = (
                                            v2_calibrated_prefix_survival(
                                                raw_prefix_survival,
                                                target_len,
                                            )
                                        )
                                        raw_conditional_gain = 0.0
                                        extension_survival = 1.0
                                        for probability in extension_hazards:
                                            extension_survival *= probability
                                            raw_conditional_gain += extension_survival
                                        conditional_gain_correction, conditional_gain_source, conditional_gain_count = (
                                            v2_conditional_gain_correction(target_len)
                                        )
                                        calibrated_conditional_gain = min(
                                            float(extension),
                                            raw_conditional_gain * conditional_gain_correction,
                                        )
                                        raw_extension_gain = raw_prefix_survival * raw_conditional_gain
                                        extension_gain = (
                                            calibrated_prefix_survival * calibrated_conditional_gain
                                        )
                                        blocks_generated = max(
                                            1,
                                            (target_len + small_block_size - 1) // small_block_size,
                                        )
                                        extension_blocks = max(
                                            1,
                                            (extension + small_block_size - 1) // small_block_size,
                                        )
                                        denoising_passes = frontier_stats["forward_pass_breakdown"]["denoising"]
                                        estimated_extension_passes = max(
                                            1.0,
                                            denoising_passes * extension_blocks / blocks_generated,
                                        )
                                        _, extended_verify_ms, _ = v2_latency_estimates(target_len + extension)
                                        extend_ms_per_output = (
                                            elapsed_draft_ms
                                            + estimated_extension_passes * draft_forward_ms
                                            + extended_verify_ms
                                            + controller_ms
                                        ) / max(expected_output + extension_gain, 1e-6)
                                        step_record.update({
                                            "v2_extension_size": int(extension),
                                            "v2_extension_hazards": list(extension_hazards),
                                            "v2_prefix_survival": float(calibrated_prefix_survival),
                                            "v2_raw_prefix_survival": float(raw_prefix_survival),
                                            "v2_calibrated_prefix_survival": float(calibrated_prefix_survival),
                                            "v2_prefix_calibration_source": prefix_calibration_source,
                                            "v2_prefix_calibration_count": prefix_calibration_count,
                                            "v2_raw_conditional_gain": float(raw_conditional_gain),
                                            "v2_calibrated_conditional_gain": float(calibrated_conditional_gain),
                                            "v2_raw_extension_gain": raw_extension_gain,
                                            "v2_extension_gain": float(extension_gain),
                                            "v2_extension_gain_correction": conditional_gain_correction,
                                            "v2_gain_calibration_source": conditional_gain_source,
                                            "v2_gain_calibration_count": conditional_gain_count,
                                            "v2_extend_ms_per_output": float(extend_ms_per_output),
                                            "v2_extended_verify_round_ms": float(extended_verify_ms),
                                            "v2_extension_cost_margin": float(extension_cost_margin),
                                        })
                                        if extend_ms_per_output <= stop_ms_per_output * (1.0 + extension_cost_margin):
                                            frontier_force_extend = True
                                            frontier_stats["refinement_actions"].append({
                                                "step": len(frontier_scores),
                                                "action": "cost_aware_v2_extend",
                                                "phase": "extension",
                                                "target_len": target_len,
                                                "masks_remaining": masks_remaining,
                                                "extension": int(extension),
                                                "raw_predicted_extension_gain": raw_extension_gain,
                                                "predicted_extension_gain": float(extension_gain),
                                                "extension_gain_correction": conditional_gain_correction,
                                                "gain_calibration_source": conditional_gain_source,
                                                "gain_calibration_count": conditional_gain_count,
                                                "stop_ms_per_output": float(stop_ms_per_output),
                                                "extend_ms_per_output": float(extend_ms_per_output),
                                                "extension_cost_margin": float(extension_cost_margin),
                                            })
                                        else:
                                            stop_reason = "cost_aware_v2_verify_lower_cost"

                            if stop_reason is not None and frontier_mode not in ("disabled", "none", "off"):
                                frontier_force_stop = frontier_mode in force_stop_modes or frontier_mode == "cost_aware_v2"
                                frontier_stats["refinement_actions"].append({
                                    "step": len(frontier_scores),
                                    "action": stop_reason,
                                    "phase": "extension" if masks_remaining == 0 else "refinement",
                                    "target_len": target_len,
                                    "masks_remaining": masks_remaining,
                                })
                                if masks_remaining > 0:
                                    fill_start_time = torch.cuda.Event(enable_timing=True)
                                    fill_start_time.record()
                                    fill_logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                                    fill_logits = torch.cat([fill_logits[:, :1, :], fill_logits[:, :-1, :]], dim=1)
                                    fill_end_time = torch.cuda.Event(enable_timing=True)
                                    fill_end_time.record()
                                    torch.cuda.synchronize()
                                    forward_pass_latencies.append(fill_start_time.elapsed_time(fill_end_time))
                                    num_forward_passes += 1
                                    frontier_stats["forward_pass_breakdown"]["fill"] += 1

                                    fill_probs = torch.softmax(fill_logits, dim=-1)
                                    fill_top2_probs = torch.topk(fill_probs, k=2, dim=-1).values
                                    fill_margins = (fill_top2_probs[..., 0] - fill_top2_probs[..., 1]).float()
                                    draft_mask = (x_t[:, draft_token_start_idx:draft_end_idx] == mask_id)
                                    for rel_pos in draft_mask[0].nonzero(as_tuple=False).flatten().tolist():
                                        absolute_pos = draft_token_start_idx + rel_pos
                                        local_pos = absolute_pos - block_abs_start
                                        if 0 <= local_pos < fill_logits.shape[1]:
                                            token_id = fill_logits[:, local_pos, :].argmax(dim=-1)
                                            token_conf = torch.gather(fill_probs[:, local_pos, :], dim=-1, index=token_id.unsqueeze(-1)).squeeze(-1)
                                            x_t[:, absolute_pos] = token_id
                                            committed_confidences[int(absolute_pos)] = float(token_conf[0].float().item())
                                            committed_margins[int(absolute_pos)] = float(fill_margins[0, local_pos].float().item())
                                            committed_forced[int(absolute_pos)] = True
                                            filled_on_stop_positions.add(int(absolute_pos))
                                            conf_of_unmasked_tokens.append(float(token_conf[0].float().item()))
                        
                        # logger.debug(f"{Colors.CYAN}x1_p {x1_p.tolist()[0]}{Colors.RESET}")
                        # logger.debug(f"{Colors.CYAN}current conf_of_unmasked_tokens {conf_of_unmasked_tokens}{Colors.RESET}")
                        small_block_tokens = args.target_tokenizer.decode(x_1[0], skip_special_tokens=False)
                        # logger.debug(f"{Colors.CYAN}Small_block_tokens: {small_block_tokens}{Colors.RESET}")

                        if is_drafter and draft_token_end_idx <= x_t.shape[1] \
                            and x_t[:, draft_token_start_idx:draft_token_end_idx].ne(mask_id).all():
                            
                            if frontier_force_stop:
                                frontier_stats["stop_reason"] = frontier_stats["stop_reason"] or frontier_stats["refinement_actions"][-1]["action"]
                                logger.debug(f"{Colors.GREEN}Frontier controller stopped refinement. reason={frontier_stats.get('stop_reason')} spec_len={spec_len}{Colors.RESET}")
                                draft_tokens_unmasked = True
                            elif frontier_force_extend:
                                extension = min(incr_len, max_spec_len - spec_len)
                                if extension > 0:
                                    record_extension_event(
                                        "cost_aware_v2_extend",
                                        extension,
                                        accept_probabilities,
                                        probability_variants,
                                        {
                                            "raw_predicted_extension_gain": frontier_stats["steps"][-1].get(
                                                "v2_raw_extension_gain"
                                            ),
                                            "predicted_extension_gain": frontier_stats["steps"][-1].get(
                                                "v2_extension_gain"
                                            ),
                                            "extension_gain_correction": frontier_stats["steps"][-1].get(
                                                "v2_extension_gain_correction"
                                            ),
                                            "gain_calibration_source": frontier_stats["steps"][-1].get(
                                                "v2_gain_calibration_source"
                                            ),
                                            "gain_calibration_count": frontier_stats["steps"][-1].get(
                                                "v2_gain_calibration_count"
                                            ),
                                            "prefix_survival_probability": frontier_stats["steps"][-1].get(
                                                "v2_prefix_survival"
                                            ),
                                            "raw_prefix_survival_probability": frontier_stats["steps"][-1].get(
                                                "v2_raw_prefix_survival"
                                            ),
                                            "calibrated_prefix_survival_probability": frontier_stats["steps"][-1].get(
                                                "v2_calibrated_prefix_survival"
                                            ),
                                            "prefix_calibration_source": frontier_stats["steps"][-1].get(
                                                "v2_prefix_calibration_source"
                                            ),
                                            "prefix_calibration_count": frontier_stats["steps"][-1].get(
                                                "v2_prefix_calibration_count"
                                            ),
                                            "raw_conditional_extension_gain": frontier_stats["steps"][-1].get(
                                                "v2_raw_conditional_gain"
                                            ),
                                            "calibrated_conditional_extension_gain": frontier_stats["steps"][-1].get(
                                                "v2_calibrated_conditional_gain"
                                            ),
                                        },
                                    )
                                    draft_token_end_idx += extension
                                    spec_len += extension
                                    draft_tokens_unmasked = False
                                    frontier_force_extend = False
                                    logger.debug(f"{Colors.GREEN}Cost-aware v2 extended proposal by {extension} tokens. spec_len={spec_len}{Colors.RESET}")
                                else:
                                    frontier_stats["stop_reason"] = frontier_stats["stop_reason"] or "cost_aware_v2_max_spec_len"
                                    draft_tokens_unmasked = True
                            elif any([x < lowconf_threshold for x in conf_of_unmasked_tokens]):
                                frontier_stats["stop_reason"] = frontier_stats["stop_reason"] or "failfast_low_confidence"
                                logger.debug(f"{Colors.GREEN}All draft tokens ({draft_token_start_idx}:{draft_token_end_idx}) unmasked. Some are low-confidence, stop speculating. spec_len is {spec_len}{Colors.RESET}")
                                # none of the first n tokens after the original input tokens are mask tokens
                                draft_tokens_unmasked = True
                                
                                if last_round_rejected is not None:
                                    ###start of logic of reusing rejected drafts from the last round###
                                    curr_proposal = x_t[:, draft_token_start_idx:draft_token_end_idx].tolist()[0]
                                    logger.debug(f"{Colors.GREEN}Current proposal: {curr_proposal}{Colors.RESET}")
                                    num_salvagable_tokens, start_index_rejected, start_index_proposal = \
                                        get_rejected_overlap_info(last_round_rejected, curr_proposal)
                                    if num_salvagable_tokens != 0 and start_index_proposal != len(curr_proposal) - 1:
                                        logger.debug(f"{Colors.GREEN}last_round_rejected: {last_round_rejected}{Colors.RESET}, start_index_rejected {start_index_rejected}, start_index_proposal {start_index_proposal}")
                                        veri_len = start_index_proposal + num_salvagable_tokens
                                        
                                        if veri_len > spec_len:
                                            record_extension_event(
                                                "rejected_overlap_reuse",
                                                veri_len - spec_len,
                                            )
                                            spec_len = veri_len
                                            draft_token_end_idx = draft_token_start_idx + spec_len
                                            draft_proposal = curr_proposal[:start_index_proposal] + last_round_rejected[start_index_rejected:]
                                            new_draft_proposal = torch.tensor(draft_proposal, device=self.device).unsqueeze(0)
                                            prefix = x_t[:, :draft_token_start_idx]
                                            x_t = torch.cat([prefix, new_draft_proposal], dim=1)
                                            logger.info(f"{Colors.YELLOW}Reusing: {len(curr_proposal)}->{spec_len}. Current num_forward_passes {num_forward_passes}{Colors.RESET}")
                                    ###end of logic of reusing rejected drafts from the last round###
                                
                            elif (
                                frontier_mode in ("disabled", "none", "off")
                                and len(conf_of_unmasked_tokens) >= max_spec_len
                            ) or (
                                frontier_mode not in ("disabled", "none", "off")
                                and spec_len >= max_spec_len
                            ):
                                frontier_stats["stop_reason"] = frontier_stats["stop_reason"] or "failfast_max_spec_len"
                                logger.debug(f"{Colors.GREEN}Already drafted {len(conf_of_unmasked_tokens)}>{max_spec_len} high-confidence tokens, stopping just in case.{Colors.RESET}")
                                draft_tokens_unmasked = True
                            else:
                                logger.debug(f"{Colors.GREEN}All draft tokens ({draft_token_start_idx}:{draft_token_end_idx}) unmasked. All are high-confidence, continue speculating.{Colors.RESET}")
                                extension = incr_len if frontier_mode in ("disabled", "none", "off") else min(incr_len, max_spec_len - spec_len)
                                record_extension_event(
                                    "high_confidence_extend",
                                    extension,
                                    accept_probabilities,
                                    probability_variants,
                                )
                                draft_token_end_idx += extension
                                spec_len += extension
                                draft_tokens_unmasked = False

                    if is_drafter and draft_tokens_unmasked:
                        logger.debug(f"{Colors.YELLOW}Exiting after {num_forward_passes} fwd passes{Colors.RESET}")
                        input_ids = x_t
                        break
                if is_drafter and draft_tokens_unmasked:
                    break
                
            input_ids = x_t
            
            if is_drafter and draft_tokens_unmasked:
                break
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        logger.debug(f"{Colors.YELLOW}forward_pass_latencies: {[f'{latency:.2f}ms' for latency in forward_pass_latencies]}{Colors.RESET}")
        
        frontier_stats["actual_spec_len"] = int(spec_len)
        frontier_stats["forward_pass_breakdown"]["total"] = int(num_forward_passes)
        if return_frontier_stats and is_drafter and frontier_stats["mode"] not in ("disabled", "none", "off"):
            draft_token_stats = []
            draft_end_idx = min(draft_token_start_idx + int(spec_len), input_ids.shape[1])
            for absolute_pos in range(draft_token_start_idx, draft_end_idx):
                draft_token_stats.append({
                    "relative_pos": int(absolute_pos - draft_token_start_idx),
                    "token_id": int(input_ids[0, absolute_pos].item()),
                    "confidence": float(committed_confidences.get(absolute_pos, 0.0)),
                    "margin": float(committed_margins.get(absolute_pos, 0.0)),
                    "forced": bool(committed_forced.get(absolute_pos, False)),
                    "filled_on_stop": bool(absolute_pos in filled_on_stop_positions),
                })
            frontier_stats["draft_token_stats"] = draft_token_stats
        if return_prefill_kvs:
            if return_frontier_stats:
                return input_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies, frontier_stats
            return input_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies
        if return_frontier_stats:
            return input_ids, spec_len, num_forward_passes, forward_pass_latencies, frontier_stats
        return input_ids, spec_len, num_forward_passes, forward_pass_latencies


    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        # Calculate probabilities
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim=-1)
            x_1 = p_1t.argmax(dim=-1)
            return x_1, p_1t
                            
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        probs[indices_to_remove] = 0

        # Renormalize so that the probabilities of remaining tokens sum to 1
        # Add a small epsilon value to prevent division by zero
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples=1).unsqueeze(0).squeeze(-1)

        return x_1, p_1t
