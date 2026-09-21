# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local transformer used by MossTTSRealtime for RVQ codebook decoding."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.utils import TransformersKwargs, logging
from .configuration_mossttsrealtime import MossTTSRealtimeLocalTransformerConfig

logger = logging.get_logger(__name__)


class MossTTSRealtimeLocalTransformerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MossTTSRealtimeLocalTransformerMLP(nn.Module):
    def __init__(self, config: MossTTSRealtimeLocalTransformerConfig):
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
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MossTTSRealtimeLocalTransformerAttention(nn.Module):
    def __init__(self, config: MossTTSRealtimeLocalTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = MossTTSRealtimeLocalTransformerRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MossTTSRealtimeLocalTransformerRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MossTTSRealtimeLocalTransformerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MossTTSRealtimeLocalTransformerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MossTTSRealtimeLocalTransformerAttention(config=config, layer_idx=layer_idx)
        self.mlp = MossTTSRealtimeLocalTransformerMLP(config)
        self.input_layernorm = MossTTSRealtimeLocalTransformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossTTSRealtimeLocalTransformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MossTTSRealtimeLocalTransformerPreTrainedModel(PreTrainedModel):

    config_class = MossTTSRealtimeLocalTransformerConfig
    config: MossTTSRealtimeLocalTransformerConfig

    base_model_prefix = "local_transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MossTTSRealtimeLocalTransformerDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": MossTTSRealtimeLocalTransformerDecoderLayer,
        "attentions": MossTTSRealtimeLocalTransformerAttention,
    }


class MossTTSRealtimeLocalTransformerRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MossTTSRealtimeLocalTransformerConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = getattr(config, "rope_type", "linear")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MossTTSRealtimeLocalTransformer(MossTTSRealtimeLocalTransformerPreTrainedModel):
    def __init__(self, config: MossTTSRealtimeLocalTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size, config.hidden_size, config.audio_pad_token) for _ in range(config.rvq - 1)]
        )
        self.layers = nn.ModuleList(
            [MossTTSRealtimeLocalTransformerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MossTTSRealtimeLocalTransformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MossTTSRealtimeLocalTransformerRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        codebook_idx: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if position_ids is not None and not torch.compiler.is_compiling():
            position_ids = None

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if use_cache and past_key_values is None:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            past_key_values = StaticCache(config=self.config, max_cache_len=self.config.rvq, device=device)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)

        if inputs_embeds is None:
            if codebook_idx is not None:
                if codebook_idx <= 0:
                    raise ValueError(f"`codebook_idx` must be in [1, {len(self.embed_tokens)}], got {codebook_idx}.")
                if codebook_idx > len(self.embed_tokens):
                    raise ValueError(f"`codebook_idx` must be in [1, {len(self.embed_tokens)}], got {codebook_idx}.")
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(1)
                token_emb = self.embed_tokens[codebook_idx - 1](input_ids[:, 0]).unsqueeze(1)  # [B,1,H]
                inputs_embeds = token_emb
            else:
                if input_ids.shape[1] != cache_position.shape[0]:
                    raise ValueError(
                        "`input_ids` and `cache_position` must align in sequence length: "
                        f"got {input_ids.shape[1]} and {cache_position.shape[0]}."
                    )
                codebook_idxs = torch.clamp(cache_position - 1, min=0, max=len(self.embed_tokens) - 1)
                inputs_embeds = torch.stack(
                    [
                        self.embed_tokens[codebook_idx](input_ids[:, seq_idx])
                        for seq_idx, codebook_idx in enumerate(codebook_idxs.tolist())
                    ],
                    dim=1,
                )

                input_ids_are_first_codebook = bool(cache_position[0] == 0)
                if backbone_last_hidden_state is not None:
                    inputs_embeds[:, 0, :] = backbone_last_hidden_state[:, 0, :]
                else:
                    if not torch.compiler.is_compiling() and input_ids_are_first_codebook:
                        logger.warning(
                            "When the first codebook token is provided, `backbone_last_hidden_state` should also be provided for correct inference."
                        )

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MossTTSRealtimeLocalTransformerForCausalLM(MossTTSRealtimeLocalTransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, config):
        super().__init__(config)
        self.model = MossTTSRealtimeLocalTransformer(config)
        self.audio_vocab_size = self.config.audio_vocab_size

        self.local_lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False) for _ in range(config.rvq)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        codebook_idx: Optional[int] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=backbone_last_hidden_state,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            codebook_idx=codebook_idx,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                slice_indices = slice(0, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep
        hs = hidden_states[:, slice_indices, :]

        if cache_position is not None:
            if codebook_idx is None:
                raise ValueError("`codebook_idx` must be provided when `cache_position` is provided.")
            logits = self.local_lm_heads[codebook_idx](hs[:, 0, :]).unsqueeze(1)
        else:
            if hs.shape[1] > len(self.local_lm_heads):
                raise ValueError(
                    f"Cannot project {hs.shape[1]} codebooks with only {len(self.local_lm_heads)} LM heads."
                )
            logits_list = []
            for i in range(hs.shape[1]):
                logits_list.append(self.local_lm_heads[i](hs[:, i, :]))
            logits = torch.stack(logits_list, dim=1)

        logits = logits.contiguous()
        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits, None, self.audio_vocab_size, shift_labels=labels.contiguous())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "MossTTSRealtimeLocalTransformer",
    "MossTTSRealtimeLocalTransformerAttention",
    "MossTTSRealtimeLocalTransformerConfig",
    "MossTTSRealtimeLocalTransformerDecoderLayer",
    "MossTTSRealtimeLocalTransformerForCausalLM",
    "MossTTSRealtimeLocalTransformerPreTrainedModel",
    "MossTTSRealtimeLocalTransformerRMSNorm",
    "MossTTSRealtimeLocalTransformerRotaryEmbedding",
]

