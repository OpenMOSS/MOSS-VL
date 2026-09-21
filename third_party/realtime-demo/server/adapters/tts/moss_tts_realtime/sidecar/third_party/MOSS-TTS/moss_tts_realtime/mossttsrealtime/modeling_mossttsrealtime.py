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
"""MossTTSRealtime model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3 import Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer
from .configuration_mossttsrealtime import MossTTSRealtimeConfig
from .modeling_mossttsrealtime_local import MossTTSRealtimeLocalTransformerForCausalLM


class MossTTSRealtimePretrainedModel(PreTrainedModel):
    config_class = MossTTSRealtimeConfig
    config: MossTTSRealtimeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])


@dataclass
class MossTTSRealtimeOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    local_loss: Optional[torch.FloatTensor] = None
    local_logits: Optional[torch.FloatTensor] = None
    local_past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    local_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    local_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    backbone_loss: Optional[torch.FloatTensor] = None


class MossTTSRealtime(MossTTSRealtimePretrainedModel):
    def __init__(self, config: MossTTSRealtimeConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.ModuleList([])
        self.embed_tokens.append(
            nn.Embedding(
                config.language_config.vocab_size,
                config.language_config.hidden_size,
                config.language_config.pad_token_id,
            )
        )
        self.audio_vocab_size = self.config.audio_vocab_size
        for _ in range(self.config.rvq):
            self.embed_tokens.append(
                nn.Embedding(self.audio_vocab_size, config.language_config.hidden_size, self.config.audio_pad_token)
            )
        self.language_model = Qwen3Model._from_config(config.language_config)
        self.local_transformer = MossTTSRealtimeLocalTransformerForCausalLM._from_config(config.local_config)
        self.post_init()

    def get_input_embeddings(self, input_ids):
        if input_ids.device != self.embed_tokens[0].weight.device:
            input_ids = input_ids.to(self.embed_tokens[0].weight.device)
        inputs_embeds = self.embed_tokens[0](input_ids[..., 0])
        for i, embed in enumerate(self.embed_tokens):
            if i == 0:
                continue
            inputs_embeds = inputs_embeds + embed(input_ids[..., i])
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hidden_out_layers: Optional[list] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        outputs = self.language_model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        loss = None
        local_outputs = None
        if labels is not None:
            audio_labels = labels[:, :, 1:]
            train_mask = ~(audio_labels == -100).all(dim=-1)
            local_input_ids = audio_labels[train_mask][..., : self.config.rvq - 1]
            local_input_ids[local_input_ids == -100] = self.config.audio_pad_token
            local_input_ids = F.pad(local_input_ids, (1, 0), value=0)

            train_idx = train_mask.nonzero(as_tuple=True)
            hidden_positions = torch.clamp(train_idx[1] - 1, min=0)
            local_hidden_states = outputs.last_hidden_state[train_idx[0], hidden_positions, :].reshape(
                -1, 1, self.config.local_config.hidden_size
            )
            local_labels = audio_labels[train_mask]

            local_outputs = self.local_transformer(
                input_ids=local_input_ids,
                backbone_last_hidden_state=local_hidden_states,
                use_cache=use_cache,
                return_dict=True,
                labels=local_labels,
                **kwargs,
            )
            loss = local_outputs.loss

        output = MossTTSRealtimeOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            local_logits=local_outputs.logits if local_outputs is not None else None,
            local_past_key_values=local_outputs.past_key_values if local_outputs is not None else None,
            local_hidden_states=local_outputs.hidden_states if local_outputs is not None else None,
            local_attentions=local_outputs.attentions if local_outputs is not None else None,
        )
        if not return_dict:
            return output.to_tuple()
        return output


__all__ = ["MossTTSRealtime", "MossTTSRealtimeConfig", "MossTTSRealtimeOutputWithPast", "MossTTSRealtimePretrainedModel"]
