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

"""MossTTSRealtimeModel configuration."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


def _ensure_config(cfg: Any, cls: type[PretrainedConfig]) -> PretrainedConfig:
    if isinstance(cfg, cls):
        return cfg
    if cfg is None:
        return cls()
    if isinstance(cfg, dict):
        return cls(**cfg)
    raise TypeError(f"Unsupported config type for {cls.__name__}: {type(cfg)}")


class MossTTSRealtimeLocalTransformerConfig(PretrainedConfig):
    model_type = "moss_tts_realtime_local_transformer"

    def __init__(
        self,
        head_dim: int = 128,
        use_cache: bool = True,
        hidden_size: int = 2048,
        rms_norm_eps: float = 1e-6,
        num_hidden_layers: int = 4,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 33,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        rope_theta: int = 1000000,
        rope_type: str = "linear",
        pad_token_id: int = 1024,
        rope_parameters: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        if rope_parameters is None:
            rope_parameters = {"rope_type": rope_type, "rope_theta": rope_theta, "factor": 1.0}
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id

        self.audio_pad_token = 1024
        self.audio_vocab_size = 1027
        self.rvq = 16


class MossTTSRealtimeConfig(PretrainedConfig):
    model_type = "moss_tts_realtime"

    def __init__(
        self,
        language_config: Qwen3Config | dict | None = None,
        local_config: MossTTSRealtimeLocalTransformerConfig | dict | None = None,
        rvq: int = 16,
        audio_pad_token: int = 1024,
        audio_vocab_size: int = 1027,
        reference_audio_pad: int = 151654,
        text_pad: int = 151655,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rvq = rvq
        self.initializer_range = initializer_range
        self.audio_pad_token = audio_pad_token
        self.audio_vocab_size = audio_vocab_size
        self.reference_audio_pad = reference_audio_pad
        self.text_pad = text_pad
        self.language_config = _ensure_config(language_config, Qwen3Config)
        self.local_config = _ensure_config(local_config, MossTTSRealtimeLocalTransformerConfig)

        attn_impl = self._attn_implementation
        self.language_config._attn_implementation = attn_impl
        self.local_config._attn_implementation = attn_impl


__all__ = ["MossTTSRealtimeConfig", "MossTTSRealtimeLocalTransformerConfig"]
