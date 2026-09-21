from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS_MODEL_DIR = REPO_ROOT / "models" / "MOSS-TTS-Nano"
DEFAULT_TTS_ONNX_OUTPUT_DIR = REPO_ROOT / "models" / "MOSS-TTS-Nano-100M-ONNX"

LOCAL_MIXED_PREFIX_SAMPLE_CHANNELS = 3
FIXED_SAMPLED_TEXT_TEMPERATURE = 1.0
FIXED_SAMPLED_TEXT_TOP_P = 1.0
FIXED_SAMPLED_TEXT_TOP_K = 50
FIXED_SAMPLED_AUDIO_TEMPERATURE = 0.8
FIXED_SAMPLED_AUDIO_TOP_P = 0.95
FIXED_SAMPLED_AUDIO_TOP_K = 25
FIXED_SAMPLED_AUDIO_REPETITION_PENALTY = 1.2


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MOSS-TTS-Nano TTS ONNX artifacts.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_TTS_MODEL_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TTS_ONNX_OUTPUT_DIR),
        help="Directory that will receive the ONNX files and metadata.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-seq-len", type=int, default=24)
    parser.add_argument("--sample-past-len", type=int, default=24)
    parser.add_argument("--disable-eager-attn", action="store_true")
    return parser.parse_args()


def _metadata_model_source(raw_value: str | Path) -> str:
    text = str(raw_value).strip()
    if not text:
        return ""
    path_value = Path(text).expanduser()
    if path_value.is_absolute():
        return path_value.name
    return text.replace("\\", "/")


def _flatten_past_key_values(past_key_values: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    flat: list[torch.Tensor] = []
    for key, value in past_key_values:
        flat.extend([key.to(torch.float32), value.to(torch.float32)])
    return tuple(flat)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    # The legacy ONNX tracer can mis-lower torch.stack(..., dim=-1) here and
    # emit a Concat on the wrong axis, which breaks RoPE in the exported graph.
    neg_odd = (-hidden_states[..., 1::2]).unsqueeze(4)
    even = hidden_states[..., ::2].unsqueeze(4)
    rotated = torch.cat((neg_odd, even), dim=4)
    return rotated.reshape_as(hidden_states)


def apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class ExportableGlobalTransformerCore(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.transformer = model.transformer
        self.hidden_size = int(model.config.gpt2_config.n_embd)
        self.num_heads = int(model.config.gpt2_config.n_head)
        self.head_dim = self.hidden_size // self.num_heads

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.hidden_size)

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        return position_ids.masked_fill(~attention_mask, 0)

    def _run_attention(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        *,
        full_attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        qkv = attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query_position_ids = full_position_ids[:, -hidden_states.shape[1]:]
        if getattr(attn, "rotary_emb", None) is not None:
            cos, sin = attn.rotary_emb(
                query_position_ids.to(device=query.device),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)

        present = (key, value) if use_cache else None

        query_states = query.permute(0, 2, 1, 3)
        key_states = key.permute(0, 2, 3, 1)
        value_states = value.permute(0, 2, 1, 3)

        scale = 1.0
        if getattr(attn, "scale_attn_weights", True):
            scale /= self.head_dim ** 0.5
        if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
            scale /= float(attn.layer_idx + 1)

        scores = torch.matmul(query_states, key_states) * scale

        query_length = hidden_states.shape[1]
        key_length = full_attention_mask.shape[1]
        query_positions = torch.arange(query_length, device=query.device, dtype=torch.long)
        query_positions = query_positions + (key_length - query_length)
        key_positions = torch.arange(key_length, device=query.device, dtype=torch.long)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        key_mask = full_attention_mask[:, None, None, :]
        attention_mask = causal_mask & key_mask
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, value_states).permute(0, 2, 1, 3).contiguous()
        attn_output = self._merge_heads(attn_output)
        attn_output = attn.c_proj(attn_output)
        attn_output = attn.resid_dropout(attn_output)
        return attn_output, present

    def _run_decode_attention(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        *,
        past_valid_lengths: torch.Tensor,
        query_position_ids: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        qkv = attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if getattr(attn, "rotary_emb", None) is not None:
            cos, sin = attn.rotary_emb(
                query_position_ids.to(device=query.device, dtype=torch.long),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)

        present = (key, value) if use_cache else None

        query_states = query.permute(0, 2, 1, 3)
        key_states = key.permute(0, 2, 3, 1)
        value_states = value.permute(0, 2, 1, 3)

        scale = 1.0
        if getattr(attn, "scale_attn_weights", True):
            scale /= self.head_dim ** 0.5
        if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
            scale /= float(attn.layer_idx + 1)

        scores = torch.matmul(query_states, key_states) * scale
        batch_size = hidden_states.shape[0]
        key_length = key.shape[1]
        key_positions = torch.arange(key_length, device=query.device, dtype=torch.long).view(1, 1, 1, key_length)
        valid_key_lengths = past_valid_lengths.to(dtype=torch.long).view(batch_size, 1, 1, 1) + 1
        key_mask = key_positions < valid_key_lengths
        query_positions = query_position_ids.to(dtype=torch.long).view(batch_size, 1, 1, 1)
        causal_mask = key_positions <= query_positions
        attention_mask = key_mask & causal_mask
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, value_states).permute(0, 2, 1, 3).contiguous()
        attn_output = self._merge_heads(attn_output)
        attn_output = attn.c_proj(attn_output)
        attn_output = attn.resid_dropout(attn_output)
        return attn_output, present

    def _run_transformer(
        self,
        *,
        inputs_embeds: torch.Tensor,
        full_attention_mask: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        query_attention_mask = full_attention_mask[:, -inputs_embeds.shape[1]:]
        full_position_ids = self._build_position_ids(full_attention_mask)
        query_position_ids = full_position_ids[:, -inputs_embeds.shape[1]:]

        hidden_states = inputs_embeds
        if getattr(self.transformer, "position_embedding_type", "absolute") == "absolute":
            hidden_states = hidden_states + self.transformer.wpe(query_position_ids)
        hidden_states = self.transformer.drop(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states_ln = block.ln_1(hidden_states)
            attn_output, present = self._run_attention(
                block.attn,
                hidden_states_ln,
                full_attention_mask=full_attention_mask,
                full_position_ids=full_position_ids,
                layer_past=None if past_key_values is None else past_key_values[layer_index],
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return hidden_states, tuple(presents) if presents is not None else None

    def _run_decode_step(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_valid_lengths: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        query_position_ids = past_valid_lengths.to(dtype=torch.long).unsqueeze(1)
        hidden_states = inputs_embeds
        if getattr(self.transformer, "position_embedding_type", "absolute") == "absolute":
            hidden_states = hidden_states + self.transformer.wpe(query_position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states_ln = block.ln_1(hidden_states)
            attn_output, present = self._run_decode_attention(
                block.attn,
                hidden_states_ln,
                past_valid_lengths=past_valid_lengths,
                query_position_ids=query_position_ids,
                layer_past=None if past_key_values is None else past_key_values[layer_index],
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            if presents is not None:
                presents.append(present)

        hidden_states = self.transformer.ln_f(hidden_states)
        return hidden_states, tuple(presents) if presents is not None else None

    def _build_inputs_embeds(self, input_ids_i32: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids_i32.to(torch.long)
        text_ids = input_ids[..., 0]
        inputs_embeds = self.model.transformer.wte(text_ids)
        audio_pad_token_id = int(self.model.config.audio_pad_token_id)

        for channel_index, embedding in enumerate(self.model.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid_mask = channel_ids.ne(audio_pad_token_id)
            safe_ids = torch.where(valid_mask, channel_ids, torch.zeros_like(channel_ids))
            audio_embeds = embedding(safe_ids) * valid_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + audio_embeds

        return inputs_embeds.to(torch.float32)


class ExportableLocalTransformerCore(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.hidden_size = int(transformer.config.hidden_size)
        self.num_heads = int(transformer.config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.hidden_size)

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        return position_ids.masked_fill(~attention_mask, 0)

    def _run_attention(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        *,
        full_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        qkv = attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        position_ids = self._build_position_ids(full_attention_mask)
        if getattr(attn, "rotary_emb", None) is not None:
            cos, sin = attn.rotary_emb(
                position_ids.to(device=query.device),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        query_states = query.permute(0, 2, 1, 3)
        key_states = key.permute(0, 2, 3, 1)
        value_states = value.permute(0, 2, 1, 3)

        scale = 1.0
        if getattr(attn, "scale_attn_weights", True):
            scale /= self.head_dim ** 0.5
        if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
            scale /= float(attn.layer_idx + 1)

        scores = torch.matmul(query_states, key_states) * scale
        query_length = hidden_states.shape[1]
        key_length = full_attention_mask.shape[1]
        query_positions = torch.arange(query_length, device=query.device, dtype=torch.long)
        query_positions = query_positions + (key_length - query_length)
        key_positions = torch.arange(key_length, device=query.device, dtype=torch.long)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        key_mask = full_attention_mask[:, None, None, :]
        attention_mask = causal_mask & key_mask
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, value_states).permute(0, 2, 1, 3).contiguous()
        attn_output = self._merge_heads(attn_output)
        attn_output = attn.c_proj(attn_output)
        attn_output = attn.resid_dropout(attn_output)
        return attn_output

    def _run_decode_attention(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        *,
        past_valid_lengths: torch.Tensor,
        query_position_ids: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        qkv = attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if getattr(attn, "rotary_emb", None) is not None:
            cos, sin = attn.rotary_emb(
                query_position_ids.to(device=query.device, dtype=torch.long),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)

        present = (key, value) if use_cache else None

        query_states = query.permute(0, 2, 1, 3)
        key_states = key.permute(0, 2, 3, 1)
        value_states = value.permute(0, 2, 1, 3)

        scale = 1.0
        if getattr(attn, "scale_attn_weights", True):
            scale /= self.head_dim ** 0.5
        if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
            scale /= float(attn.layer_idx + 1)

        scores = torch.matmul(query_states, key_states) * scale
        batch_size = hidden_states.shape[0]
        key_length = key.shape[1]
        key_positions = torch.arange(key_length, device=query.device, dtype=torch.long).view(1, 1, 1, key_length)
        valid_key_lengths = past_valid_lengths.to(dtype=torch.long).view(batch_size, 1, 1, 1) + 1
        key_mask = key_positions < valid_key_lengths
        query_positions = query_position_ids.to(dtype=torch.long).view(batch_size, 1, 1, 1)
        causal_mask = key_positions <= query_positions
        attention_mask = key_mask & causal_mask
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, value_states).permute(0, 2, 1, 3).contiguous()
        attn_output = self._merge_heads(attn_output)
        attn_output = attn.c_proj(attn_output)
        attn_output = attn.resid_dropout(attn_output)
        return attn_output, present

    def _run_transformer(
        self,
        *,
        inputs_embeds: torch.Tensor,
        full_attention_mask: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        position_ids = self._build_position_ids(full_attention_mask)
        hidden_states = inputs_embeds
        if getattr(self.transformer, "position_embedding_type", "absolute") == "absolute":
            hidden_states = hidden_states + self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(hidden_states)
        hidden_states = hidden_states * full_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states_ln = block.ln_1(hidden_states)
            attn_output = self._run_attention(
                block.attn,
                hidden_states_ln,
                full_attention_mask=full_attention_mask,
            )
            hidden_states = residual + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            hidden_states = hidden_states * full_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                qkv = block.attn.c_attn(hidden_states_ln)
                _, key, value = qkv.split(self.hidden_size, dim=-1)
                key = self._split_heads(key)
                value = self._split_heads(value)
                if getattr(block.attn, "rotary_emb", None) is not None:
                    cos, sin = block.attn.rotary_emb(
                        position_ids.to(device=key.device),
                        device=key.device,
                        dtype=key.dtype,
                    )
                    key = apply_rotary_pos_emb(key, cos, sin)
                if past_key_values is not None:
                    past_key, past_value = past_key_values[layer_index]
                    key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
                    value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)
                presents.append((key, value))

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states * full_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return hidden_states, tuple(presents) if presents is not None else None

    def _run_decode_step(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_valid_lengths: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        query_position_ids = past_valid_lengths.to(dtype=torch.long).unsqueeze(1)
        hidden_states = inputs_embeds
        if getattr(self.transformer, "position_embedding_type", "absolute") == "absolute":
            hidden_states = hidden_states + self.transformer.wpe(query_position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states_ln = block.ln_1(hidden_states)
            attn_output, present = self._run_decode_attention(
                block.attn,
                hidden_states_ln,
                past_valid_lengths=past_valid_lengths,
                query_position_ids=query_position_ids,
                layer_past=None if past_key_values is None else past_key_values[layer_index],
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            if presents is not None:
                presents.append(present)

        hidden_states = self.transformer.ln_f(hidden_states)
        return hidden_states, tuple(presents) if presents is not None else None


class TtsPrefillWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.core = ExportableGlobalTransformerCore(model)

    def forward(self, input_ids_i32: torch.Tensor, attention_mask_i32: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_states, present = self.core._run_transformer(
            inputs_embeds=self.core._build_inputs_embeds(input_ids_i32),
            full_attention_mask=attention_mask_i32.to(torch.bool),
            past_key_values=None,
            use_cache=True,
        )
        return (hidden_states.to(torch.float32), *_flatten_past_key_values(present or ()))


class TtsDecodeWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.core = ExportableGlobalTransformerCore(model)
        self.model = model
        self.num_layers = int(model.config.gpt2_config.n_layer)

    def forward(self, input_ids_i32: torch.Tensor, past_valid_lengths_i32: torch.Tensor, *past: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(past) != self.num_layers * 2:
            raise ValueError(f"Expected {self.num_layers * 2} flattened KV tensors, got {len(past)}.")
        rebuilt_past = tuple(
            (past[layer_index * 2].to(torch.float32), past[layer_index * 2 + 1].to(torch.float32))
            for layer_index in range(self.num_layers)
        )
        hidden_states, present = self.core._run_decode_step(
            inputs_embeds=self.core._build_inputs_embeds(input_ids_i32),
            past_valid_lengths=past_valid_lengths_i32.to(torch.int32).reshape(-1),
            past_key_values=rebuilt_past,
            use_cache=True,
        )
        return (hidden_states.to(torch.float32), *_flatten_past_key_values(present or ()))


class LocalDecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.core = ExportableLocalTransformerCore(model.local_transformer)
        self.max_audio_prefix_length = int(model.config.n_vq - 1)
        self.local_dtype = model.local_transformer.ln_f.weight.dtype

    def forward(
        self,
        global_hidden: torch.Tensor,
        text_token_id_i32: torch.Tensor,
        audio_prefix_token_ids_i32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = global_hidden.shape[0]

        text_token_ids = text_token_id_i32.to(torch.long).reshape(batch_size)
        pieces: list[torch.Tensor] = [
            global_hidden.to(dtype=self.local_dtype).unsqueeze(1),
            self.model.transformer.wte(text_token_ids).to(dtype=self.local_dtype).unsqueeze(1),
        ]

        prefix_ids = audio_prefix_token_ids_i32.to(torch.long)
        audio_pad_token_id = int(self.model.config.audio_pad_token_id)
        for prefix_index in range(self.max_audio_prefix_length):
            current_ids = prefix_ids[:, prefix_index]
            valid_mask = current_ids.ne(audio_pad_token_id)
            safe_ids = torch.where(valid_mask, current_ids, torch.zeros_like(current_ids))
            prefix_embed = self.model.audio_embeddings[prefix_index](safe_ids).to(dtype=self.local_dtype)
            prefix_embed = prefix_embed * valid_mask.unsqueeze(-1).to(dtype=self.local_dtype)
            pieces.append(prefix_embed.unsqueeze(1))

        local_inputs = torch.cat(pieces, dim=1)
        local_hidden_states, _ = self.core._run_transformer(
            inputs_embeds=local_inputs,
            full_attention_mask=torch.ones(
                local_inputs.shape[:2],
                dtype=torch.bool,
                device=local_inputs.device,
            ),
            past_key_values=None,
            use_cache=False,
        )
        text_logits = self.model.text_lm_head(local_hidden_states[:, 0, :]).to(torch.float32)
        audio_logits = torch.stack(
            [
                audio_head(local_hidden_states[:, channel_index + 1, :]).to(torch.float32)
                for channel_index, audio_head in enumerate(self.model.audio_lm_heads)
            ],
            dim=1,
        )
        return text_logits, audio_logits


class LocalCachedStepWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.core = ExportableLocalTransformerCore(model.local_transformer)
        self.local_dtype = model.local_transformer.ln_f.weight.dtype
        self.num_layers = int(model.config.local_transformer_layers)
        self.hidden_dim = int(model.local_transformer.config.hidden_size)
        self.num_channels = int(model.config.n_vq)

    def _select_audio_embedding(
        self,
        audio_token_ids: torch.Tensor,
        channel_indices: torch.Tensor,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for channel_index, embedding in enumerate(self.model.audio_embeddings):
            active_mask = channel_indices.eq(channel_index)
            safe_ids = torch.where(active_mask, audio_token_ids, torch.zeros_like(audio_token_ids))
            embed = embedding(safe_ids).to(dtype=self.local_dtype)
            embed = embed * active_mask.unsqueeze(-1).to(dtype=self.local_dtype)
            pieces.append(embed)
        return torch.stack(pieces, dim=0).sum(dim=0)

    def forward(
        self,
        global_hidden: torch.Tensor,
        text_token_id_i32: torch.Tensor,
        audio_token_id_i32: torch.Tensor,
        channel_index_i32: torch.Tensor,
        step_type_i32: torch.Tensor,
        past_valid_lengths_i32: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(past) != self.num_layers * 2:
            raise ValueError(f"Expected {self.num_layers * 2} flattened KV tensors, got {len(past)}.")
        batch_size = global_hidden.shape[0]
        rebuilt_past = tuple(
            (past[layer_index * 2].to(torch.float32), past[layer_index * 2 + 1].to(torch.float32))
            for layer_index in range(self.num_layers)
        )

        step_type = step_type_i32.to(torch.long).reshape(batch_size)
        text_token_ids = text_token_id_i32.to(torch.long).reshape(batch_size)
        audio_token_ids = audio_token_id_i32.to(torch.long).reshape(batch_size)
        channel_indices = channel_index_i32.to(torch.long).reshape(batch_size)

        global_embed = global_hidden.to(dtype=self.local_dtype)
        text_embed = self.model.transformer.wte(text_token_ids).to(dtype=self.local_dtype)
        audio_embed = self._select_audio_embedding(audio_token_ids, channel_indices)

        input_embed = (
            global_embed * step_type.eq(0).unsqueeze(-1).to(dtype=self.local_dtype)
            + text_embed * step_type.eq(1).unsqueeze(-1).to(dtype=self.local_dtype)
            + audio_embed * step_type.eq(2).unsqueeze(-1).to(dtype=self.local_dtype)
        )

        local_hidden_states, present = self.core._run_decode_step(
            inputs_embeds=input_embed.unsqueeze(1),
            past_valid_lengths=past_valid_lengths_i32.to(torch.int32).reshape(-1),
            past_key_values=rebuilt_past,
            use_cache=True,
        )
        last_hidden = local_hidden_states[:, 0, :]
        text_logits = self.model.text_lm_head(last_hidden).to(torch.float32)
        audio_logits = torch.stack(
            [audio_head(last_hidden).to(torch.float32) for audio_head in self.model.audio_lm_heads],
            dim=1,
        )
        return (text_logits, audio_logits, *_flatten_past_key_values(present or ()))


def apply_repetition_penalty_from_seen_mask(
    logits: torch.Tensor,
    seen_mask_i32: torch.Tensor,
    repetition_penalty_f32: torch.Tensor,
) -> torch.Tensor:
    seen_mask = seen_mask_i32.to(torch.bool)
    penalty = repetition_penalty_f32.to(dtype=logits.dtype).reshape(-1, 1)
    penalized_logits = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen_mask, penalized_logits, logits)


def sample_from_topk_topp_with_random_u(
    logits: torch.Tensor,
    random_u_f32: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    scores = logits.to(torch.float32)
    if temperature != 1.0:
        scores = scores / float(temperature)
    topk_scores, topk_indices = torch.topk(scores, k=int(top_k), dim=1, largest=True, sorted=True)
    if 0.0 < top_p < 1.0:
        topk_probs = torch.softmax(topk_scores, dim=1)
        topk_cumsum = torch.cumsum(topk_probs, dim=1)
        topk_prev_cumsum = topk_cumsum - topk_probs
        topk_keep = topk_prev_cumsum < float(top_p)
        topk_scores = topk_scores.masked_fill(~topk_keep, float("-inf"))
    topk_probs = torch.softmax(topk_scores, dim=1)
    cdf = torch.cumsum(topk_probs, dim=1)
    random_u = torch.clamp(random_u_f32.to(dtype=cdf.dtype).reshape(-1, 1), min=0.0, max=0.99999994)
    selected_positions = torch.sum((cdf < random_u).to(torch.int64), dim=1)
    selected_positions = torch.clamp(selected_positions, max=topk_indices.shape[1] - 1)
    return topk_indices.gather(1, selected_positions.unsqueeze(1)).squeeze(1).to(torch.long)


class LocalGreedyFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.core = ExportableLocalTransformerCore(model.local_transformer)
        self.local_dtype = model.local_transformer.ln_f.weight.dtype
        self.num_layers = int(model.config.local_transformer_layers)
        self.num_heads = int(model.local_transformer.config.num_attention_heads)
        self.hidden_dim = int(model.local_transformer.config.hidden_size)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_channels = int(model.config.n_vq)
        self.audio_codebook_sizes = [int(item) for item in model.config.audio_codebook_sizes]
        if len(set(self.audio_codebook_sizes)) != 1:
            raise ValueError("LocalGreedyFrameWrapper requires equal audio codebook sizes across all channels.")
        self.audio_codebook_size = int(self.audio_codebook_sizes[0])
        self.audio_assistant_slot_token_id = int(model.config.audio_assistant_slot_token_id)
        self.audio_end_token_id = int(model.config.audio_end_token_id)

    def _create_empty_local_past(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        empty_kv = tuple(
            (
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
            )
            for _ in range(self.num_layers)
        )
        return empty_kv

    @staticmethod
    def _argmax_token(logits: torch.Tensor) -> torch.Tensor:
        _values, indices = torch.max(logits, dim=1, keepdim=False)
        return indices.to(torch.long)

    def forward(
        self,
        global_hidden: torch.Tensor,
        repetition_seen_mask_i32: torch.Tensor,
        repetition_penalty_f32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = global_hidden.shape[0]
        past_valid_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=global_hidden.device)
        present = self._create_empty_local_past(batch_size=batch_size, device=global_hidden.device)

        local_hidden_states, present = self.core._run_decode_step(
            inputs_embeds=global_hidden.to(dtype=self.local_dtype).unsqueeze(1),
            past_valid_lengths=past_valid_lengths,
            past_key_values=present,
            use_cache=True,
        )
        last_hidden = local_hidden_states[:, 0, :]
        text_logits = self.model.text_lm_head(last_hidden).to(torch.float32)
        candidate_scores = torch.stack(
            [
                text_logits[:, self.audio_assistant_slot_token_id],
                text_logits[:, self.audio_end_token_id],
            ],
            dim=1,
        )
        assistant_selected = candidate_scores[:, 0] >= candidate_scores[:, 1]
        next_text_token = torch.where(
            assistant_selected,
            torch.full((batch_size,), self.audio_assistant_slot_token_id, dtype=torch.long, device=global_hidden.device),
            torch.full((batch_size,), self.audio_end_token_id, dtype=torch.long, device=global_hidden.device),
        )

        generated_tokens: list[torch.Tensor] = []
        current_input_embed = self.model.transformer.wte(next_text_token).to(dtype=self.local_dtype).unsqueeze(1)
        current_past_valid_lengths = past_valid_lengths + 1

        for channel_index in range(self.num_channels):
            local_hidden_states, present = self.core._run_decode_step(
                inputs_embeds=current_input_embed,
                past_valid_lengths=current_past_valid_lengths,
                past_key_values=present,
                use_cache=True,
            )
            last_hidden = local_hidden_states[:, 0, :]
            audio_logits = self.model.audio_lm_heads[channel_index](last_hidden).to(torch.float32)
            penalized_logits = apply_repetition_penalty_from_seen_mask(
                audio_logits,
                repetition_seen_mask_i32[:, channel_index, : self.audio_codebook_size],
                repetition_penalty_f32,
            )
            sampled_token = self._argmax_token(penalized_logits)
            generated_tokens.append(sampled_token.to(torch.int32))
            current_past_valid_lengths = current_past_valid_lengths + 1
            if channel_index + 1 < self.num_channels:
                current_input_embed = self.model.audio_embeddings[channel_index](sampled_token).to(dtype=self.local_dtype).unsqueeze(1)

        frame_token_ids = torch.stack(generated_tokens, dim=1)
        return assistant_selected.to(torch.int32).reshape(batch_size, 1), frame_token_ids


class LocalMixedSampledPrefixGreedyFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module, prefix_sample_channels: int = LOCAL_MIXED_PREFIX_SAMPLE_CHANNELS) -> None:
        super().__init__()
        self.model = model
        self.core = ExportableLocalTransformerCore(model.local_transformer)
        self.local_dtype = model.local_transformer.ln_f.weight.dtype
        self.num_layers = int(model.config.local_transformer_layers)
        self.num_heads = int(model.local_transformer.config.num_attention_heads)
        self.hidden_dim = int(model.local_transformer.config.hidden_size)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_channels = int(model.config.n_vq)
        self.prefix_sample_channels = max(1, min(int(prefix_sample_channels), self.num_channels))
        self.audio_codebook_sizes = [int(item) for item in model.config.audio_codebook_sizes]
        if len(set(self.audio_codebook_sizes)) != 1:
            raise ValueError("LocalMixedSampledPrefixGreedyFrameWrapper requires equal audio codebook sizes.")
        self.audio_codebook_size = int(self.audio_codebook_sizes[0])
        self.audio_assistant_slot_token_id = int(model.config.audio_assistant_slot_token_id)

    def _create_empty_local_past(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
            )
            for _ in range(self.num_layers)
        )

    @staticmethod
    def _argmax_token(logits: torch.Tensor) -> torch.Tensor:
        _values, indices = torch.max(logits, dim=1, keepdim=False)
        return indices.to(torch.long)

    def _decode_step(
        self,
        *,
        input_embed: torch.Tensor,
        present: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        past_valid_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        local_hidden_states, next_present = self.core._run_decode_step(
            inputs_embeds=input_embed,
            past_valid_lengths=past_valid_lengths,
            past_key_values=present,
            use_cache=True,
        )
        return local_hidden_states[:, 0, :], next_present

    def forward(
        self,
        global_hidden: torch.Tensor,
        sampled_prefix_token_ids_i32: torch.Tensor,
        repetition_seen_mask_i32: torch.Tensor,
        repetition_penalty_f32: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = global_hidden.shape[0]
        device = global_hidden.device
        present = self._create_empty_local_past(batch_size=batch_size, device=device)
        past_valid_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)

        _, present = self._decode_step(
            input_embed=global_hidden.to(dtype=self.local_dtype).unsqueeze(1),
            present=present,
            past_valid_lengths=past_valid_lengths,
        )
        past_valid_lengths = past_valid_lengths + 1

        generated_tokens: list[torch.Tensor] = []
        current_input_embed = self.model.transformer.wte(
            torch.full((batch_size,), self.audio_assistant_slot_token_id, dtype=torch.long, device=device)
        ).to(dtype=self.local_dtype).unsqueeze(1)
        sampled_prefix_token_ids = sampled_prefix_token_ids_i32.to(torch.long)

        for channel_index in range(self.num_channels):
            last_hidden, present = self._decode_step(
                input_embed=current_input_embed,
                present=present,
                past_valid_lengths=past_valid_lengths,
            )
            if channel_index < self.prefix_sample_channels:
                sampled_token = sampled_prefix_token_ids[:, channel_index]
            else:
                audio_logits = self.model.audio_lm_heads[channel_index](last_hidden).to(torch.float32)
                penalized_logits = apply_repetition_penalty_from_seen_mask(
                    audio_logits,
                    repetition_seen_mask_i32[:, channel_index, : self.audio_codebook_size],
                    repetition_penalty_f32,
                )
                sampled_token = self._argmax_token(penalized_logits)
            generated_tokens.append(sampled_token.to(torch.int32))
            past_valid_lengths = past_valid_lengths + 1
            if channel_index + 1 < self.num_channels:
                current_input_embed = self.model.audio_embeddings[channel_index](sampled_token).to(
                    dtype=self.local_dtype
                ).unsqueeze(1)

        return torch.stack(generated_tokens, dim=1)


class LocalFixedSampledFrameWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.core = ExportableLocalTransformerCore(model.local_transformer)
        self.local_dtype = model.local_transformer.ln_f.weight.dtype
        self.num_layers = int(model.config.local_transformer_layers)
        self.num_heads = int(model.local_transformer.config.num_attention_heads)
        self.hidden_dim = int(model.local_transformer.config.hidden_size)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_channels = int(model.config.n_vq)
        self.audio_codebook_sizes = [int(item) for item in model.config.audio_codebook_sizes]
        if len(set(self.audio_codebook_sizes)) != 1:
            raise ValueError("LocalFixedSampledFrameWrapper requires equal audio codebook sizes across all channels.")
        self.audio_codebook_size = int(self.audio_codebook_sizes[0])
        self.audio_assistant_slot_token_id = int(model.config.audio_assistant_slot_token_id)
        self.audio_end_token_id = int(model.config.audio_end_token_id)

    def _create_empty_local_past(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        global_hidden: torch.Tensor,
        repetition_seen_mask_i32: torch.Tensor,
        assistant_random_u_f32: torch.Tensor,
        audio_random_u_f32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = global_hidden.shape[0]
        device = global_hidden.device
        past_valid_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        present = self._create_empty_local_past(batch_size=batch_size, device=device)
        repetition_penalty = torch.full(
            (batch_size,),
            FIXED_SAMPLED_AUDIO_REPETITION_PENALTY,
            dtype=torch.float32,
            device=device,
        )

        local_hidden_states, present = self.core._run_decode_step(
            inputs_embeds=global_hidden.to(dtype=self.local_dtype).unsqueeze(1),
            past_valid_lengths=past_valid_lengths,
            past_key_values=present,
            use_cache=True,
        )
        last_hidden = local_hidden_states[:, 0, :]
        text_logits = self.model.text_lm_head(last_hidden).to(torch.float32)
        candidate_scores = torch.stack(
            [
                text_logits[:, self.audio_assistant_slot_token_id],
                text_logits[:, self.audio_end_token_id],
            ],
            dim=1,
        )
        if FIXED_SAMPLED_TEXT_TEMPERATURE != 1.0:
            candidate_scores = candidate_scores / float(FIXED_SAMPLED_TEXT_TEMPERATURE)
        assistant_probs = torch.softmax(candidate_scores, dim=1)[:, 0]
        assistant_random_u = torch.clamp(
            assistant_random_u_f32.to(dtype=assistant_probs.dtype).reshape(-1),
            min=0.0,
            max=0.99999994,
        )
        assistant_selected = assistant_random_u <= assistant_probs
        next_text_token = torch.where(
            assistant_selected,
            torch.full((batch_size,), self.audio_assistant_slot_token_id, dtype=torch.long, device=device),
            torch.full((batch_size,), self.audio_end_token_id, dtype=torch.long, device=device),
        )

        generated_tokens: list[torch.Tensor] = []
        current_input_embed = self.model.transformer.wte(next_text_token).to(dtype=self.local_dtype).unsqueeze(1)
        current_past_valid_lengths = past_valid_lengths + 1
        audio_random_u = audio_random_u_f32.to(torch.float32)

        for channel_index in range(self.num_channels):
            local_hidden_states, present = self.core._run_decode_step(
                inputs_embeds=current_input_embed,
                past_valid_lengths=current_past_valid_lengths,
                past_key_values=present,
                use_cache=True,
            )
            last_hidden = local_hidden_states[:, 0, :]
            audio_logits = self.model.audio_lm_heads[channel_index](last_hidden).to(torch.float32)
            penalized_logits = apply_repetition_penalty_from_seen_mask(
                audio_logits,
                repetition_seen_mask_i32[:, channel_index, : self.audio_codebook_size],
                repetition_penalty,
            )
            sampled_token = sample_from_topk_topp_with_random_u(
                penalized_logits,
                audio_random_u[:, channel_index],
                temperature=FIXED_SAMPLED_AUDIO_TEMPERATURE,
                top_k=FIXED_SAMPLED_AUDIO_TOP_K,
                top_p=FIXED_SAMPLED_AUDIO_TOP_P,
            )
            generated_tokens.append(sampled_token.to(torch.int32))
            current_past_valid_lengths = current_past_valid_lengths + 1
            if channel_index + 1 < self.num_channels:
                current_input_embed = self.model.audio_embeddings[channel_index](sampled_token).to(
                    dtype=self.local_dtype
                ).unsqueeze(1)

        frame_token_ids = torch.stack(generated_tokens, dim=1)
        return assistant_selected.to(torch.int32).reshape(batch_size, 1), frame_token_ids


def export_onnx(
    *,
    module: nn.Module,
    output_path: Path,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def patch_global_attention_graph(output_path: Path) -> None:
    model = onnx.load(str(output_path))
    changed = 0
    for node in model.graph.node:
        if node.name.startswith("/model/transformer/h.") and node.name.endswith("/attn/Transpose_3"):
            for attr in node.attribute:
                if attr.name != "perm":
                    continue
                current_perm = list(attr.ints)
                if current_perm == [0, 2, 1, 3]:
                    attr.ints[:] = [0, 2, 3, 1]
                    changed += 1
        if node.op_type == "Softmax":
            for attr in node.attribute:
                if attr.name != "axis":
                    continue
                if attr.i == 2:
                    attr.i = 3
                    changed += 1
    if changed > 0:
        onnx.save(model, str(output_path))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if not args.disable_eager_attn and hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("eager", local_attn_implementation="sdpa")
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    config = model.config
    n_layer = int(config.gpt2_config.n_layer)
    n_head = int(config.gpt2_config.n_head)
    hidden_size = int(config.gpt2_config.n_embd)
    head_dim = hidden_size // n_head
    row_width = int(config.n_vq + 1)
    hidden_dim = int(config.hidden_size)
    local_layers = int(config.local_transformer_layers)
    local_heads = int(model.local_transformer.config.num_attention_heads)
    local_hidden_size = int(model.local_transformer.config.hidden_size)
    local_head_dim = local_hidden_size // local_heads

    prefill_output_names = ["global_hidden"] + [
        name
        for layer_index in range(n_layer)
        for name in (f"present_key_{layer_index}", f"present_value_{layer_index}")
    ]
    decode_input_names = ["input_ids", "past_valid_lengths"] + [
        name
        for layer_index in range(n_layer)
        for name in (f"past_key_{layer_index}", f"past_value_{layer_index}")
    ]
    decode_output_names = prefill_output_names
    local_cached_output_names = ["text_logits", "audio_logits"] + [
        name
        for layer_index in range(local_layers)
        for name in (f"local_present_key_{layer_index}", f"local_present_value_{layer_index}")
    ]
    local_cached_input_names = [
        "global_hidden",
        "text_token_id",
        "audio_token_id",
        "channel_index",
        "step_type",
        "past_valid_lengths",
    ] + [
        name
        for layer_index in range(local_layers)
        for name in (f"local_past_key_{layer_index}", f"local_past_value_{layer_index}")
    ]

    prefill_dynamic_axes = {
        "input_ids": {0: "batch", 1: "prefill_seq"},
        "attention_mask": {0: "batch", 1: "prefill_seq"},
        "global_hidden": {0: "batch", 1: "prefill_seq"},
    }
    decode_dynamic_axes = {
        "input_ids": {0: "batch", 1: "step_seq"},
        "past_valid_lengths": {0: "batch"},
        "global_hidden": {0: "batch", 1: "step_seq"},
    }
    for layer_index in range(n_layer):
        decode_dynamic_axes[f"past_key_{layer_index}"] = {0: "batch", 1: "past_seq"}
        decode_dynamic_axes[f"past_value_{layer_index}"] = {0: "batch", 1: "past_seq"}
        prefill_dynamic_axes[f"present_key_{layer_index}"] = {0: "batch", 1: "prefill_seq"}
        prefill_dynamic_axes[f"present_value_{layer_index}"] = {0: "batch", 1: "prefill_seq"}
        decode_dynamic_axes[f"present_key_{layer_index}"] = {0: "batch", 1: "total_seq"}
        decode_dynamic_axes[f"present_value_{layer_index}"] = {0: "batch", 1: "total_seq"}
    local_cached_dynamic_axes = {
        "global_hidden": {0: "batch"},
        "text_token_id": {0: "batch"},
        "audio_token_id": {0: "batch"},
        "channel_index": {0: "batch"},
        "step_type": {0: "batch"},
        "past_valid_lengths": {0: "batch"},
        "text_logits": {0: "batch"},
        "audio_logits": {0: "batch"},
    }
    for layer_index in range(local_layers):
        local_cached_dynamic_axes[f"local_past_key_{layer_index}"] = {0: "batch", 1: "local_past_seq"}
        local_cached_dynamic_axes[f"local_past_value_{layer_index}"] = {0: "batch", 1: "local_past_seq"}
        local_cached_dynamic_axes[f"local_present_key_{layer_index}"] = {0: "batch", 1: "local_total_seq"}
        local_cached_dynamic_axes[f"local_present_value_{layer_index}"] = {0: "batch", 1: "local_total_seq"}
    local_greedy_frame_input_names = ["global_hidden", "repetition_seen_mask", "repetition_penalty"]
    local_greedy_frame_output_names = ["should_continue", "frame_token_ids"]
    local_greedy_frame_dynamic_axes = {
        "global_hidden": {0: "batch"},
        "repetition_seen_mask": {0: "batch"},
        "repetition_penalty": {0: "batch"},
        "should_continue": {0: "batch"},
        "frame_token_ids": {0: "batch"},
    }
    local_mixed_frame_input_names = ["global_hidden", "sampled_prefix_token_ids", "repetition_seen_mask", "repetition_penalty"]
    local_mixed_frame_output_names = ["frame_token_ids"]
    local_mixed_frame_dynamic_axes = {
        "global_hidden": {0: "batch"},
        "sampled_prefix_token_ids": {0: "batch"},
        "repetition_seen_mask": {0: "batch"},
        "repetition_penalty": {0: "batch"},
        "frame_token_ids": {0: "batch"},
    }
    local_fixed_sampled_frame_input_names = [
        "global_hidden",
        "repetition_seen_mask",
        "assistant_random_u",
        "audio_random_u",
    ]
    local_fixed_sampled_frame_output_names = ["should_continue", "frame_token_ids"]
    local_fixed_sampled_frame_dynamic_axes = {
        "global_hidden": {0: "batch"},
        "repetition_seen_mask": {0: "batch"},
        "assistant_random_u": {0: "batch"},
        "audio_random_u": {0: "batch"},
        "should_continue": {0: "batch"},
        "frame_token_ids": {0: "batch"},
    }

    prefill_input_ids = torch.full(
        (1, int(args.sample_seq_len), row_width),
        int(config.audio_pad_token_id),
        dtype=torch.int32,
    )
    prefill_input_ids[:, :, 0] = int(config.pad_token_id)
    prefill_attention_mask = torch.ones((1, int(args.sample_seq_len)), dtype=torch.int32)

    past_tensors = tuple(
        tensor
        for _ in range(n_layer)
        for tensor in (
            torch.zeros((1, int(args.sample_past_len), n_head, head_dim), dtype=torch.float32),
            torch.zeros((1, int(args.sample_past_len), n_head, head_dim), dtype=torch.float32),
        )
    )
    local_past_tensors = tuple(
        tensor
        for _ in range(local_layers)
        for tensor in (
            torch.zeros((1, int(args.sample_past_len), local_heads, local_head_dim), dtype=torch.float32),
            torch.zeros((1, int(args.sample_past_len), local_heads, local_head_dim), dtype=torch.float32),
        )
    )
    decode_input_ids = torch.full((1, 1, row_width), int(config.audio_pad_token_id), dtype=torch.int32)
    decode_input_ids[:, :, 0] = int(config.pad_token_id)
    decode_past_valid_lengths = torch.full((1,), int(args.sample_past_len), dtype=torch.int32)

    prefill_output_path = output_dir / "moss_tts_prefill.onnx"
    export_onnx(
        module=TtsPrefillWrapper(model),
        output_path=prefill_output_path,
        args=(prefill_input_ids, prefill_attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=prefill_output_names,
        dynamic_axes=prefill_dynamic_axes,
        opset=int(args.opset),
    )
    patch_global_attention_graph(prefill_output_path)

    decode_output_path = output_dir / "moss_tts_decode_step.onnx"
    export_onnx(
        module=TtsDecodeWrapper(model),
        output_path=decode_output_path,
        args=(decode_input_ids, decode_past_valid_lengths, *past_tensors),
        input_names=decode_input_names,
        output_names=decode_output_names,
        dynamic_axes=decode_dynamic_axes,
        opset=int(args.opset),
    )
    patch_global_attention_graph(decode_output_path)

    export_onnx(
        module=LocalDecoderWrapper(model),
        output_path=output_dir / "moss_tts_local_decoder.onnx",
        args=(
            torch.zeros((1, hidden_dim), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.int32),
            torch.full((1, int(config.n_vq - 1)), int(config.audio_pad_token_id), dtype=torch.int32),
        ),
        input_names=["global_hidden", "text_token_id", "audio_prefix_token_ids"],
        output_names=["text_logits", "audio_logits"],
        dynamic_axes={},
        opset=int(args.opset),
    )
    patch_global_attention_graph(output_dir / "moss_tts_local_decoder.onnx")

    export_onnx(
        module=LocalCachedStepWrapper(model),
        output_path=output_dir / "moss_tts_local_cached_step.onnx",
        args=(
            torch.zeros((1, hidden_dim), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            torch.full((1,), int(args.sample_past_len), dtype=torch.int32),
            *local_past_tensors,
        ),
        input_names=local_cached_input_names,
        output_names=local_cached_output_names,
        dynamic_axes=local_cached_dynamic_axes,
        opset=int(args.opset),
    )
    patch_global_attention_graph(output_dir / "moss_tts_local_cached_step.onnx")

    export_onnx(
        module=LocalFixedSampledFrameWrapper(model),
        output_path=output_dir / "moss_tts_local_fixed_sampled_frame.onnx",
        args=(
            torch.zeros((1, hidden_dim), dtype=torch.float32),
            torch.zeros((1, int(config.n_vq), int(config.audio_codebook_sizes[0])), dtype=torch.int32),
            torch.full((1,), 0.5, dtype=torch.float32),
            torch.full((1, int(config.n_vq)), 0.5, dtype=torch.float32),
        ),
        input_names=local_fixed_sampled_frame_input_names,
        output_names=local_fixed_sampled_frame_output_names,
        dynamic_axes=local_fixed_sampled_frame_dynamic_axes,
        opset=int(args.opset),
    )
    patch_global_attention_graph(output_dir / "moss_tts_local_fixed_sampled_frame.onnx")

    metadata = {
        "format_version": 1,
        "checkpoint_path": _metadata_model_source(args.checkpoint_path),
        "files": {
            "prefill": "moss_tts_prefill.onnx",
            "decode_step": "moss_tts_decode_step.onnx",
            "local_decoder": "moss_tts_local_decoder.onnx",
            "local_cached_step": "moss_tts_local_cached_step.onnx",
            "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx",
        },
        "model_config": {
            "n_vq": int(config.n_vq),
            "row_width": row_width,
            "hidden_size": hidden_dim,
            "global_layers": n_layer,
            "global_heads": n_head,
            "head_dim": head_dim,
            "local_layers": local_layers,
            "local_heads": local_heads,
            "local_head_dim": local_head_dim,
            "vocab_size": int(config.gpt2_config.vocab_size),
            "audio_codebook_sizes": [int(item) for item in config.audio_codebook_sizes],
            "audio_pad_token_id": int(config.audio_pad_token_id),
            "pad_token_id": int(config.pad_token_id),
            "im_start_token_id": int(config.im_start_token_id),
            "im_end_token_id": int(config.im_end_token_id),
            "audio_start_token_id": int(config.audio_start_token_id),
            "audio_end_token_id": int(config.audio_end_token_id),
            "audio_user_slot_token_id": int(config.audio_user_slot_token_id),
            "audio_assistant_slot_token_id": int(config.audio_assistant_slot_token_id),
        },
        "onnx": {
            "opset": int(args.opset),
            "prefill_output_names": prefill_output_names,
            "decode_input_names": decode_input_names,
            "decode_output_names": decode_output_names,
            "local_cached_input_names": local_cached_input_names,
            "local_cached_output_names": local_cached_output_names,
            "local_fixed_sampled_frame_input_names": local_fixed_sampled_frame_input_names,
            "local_fixed_sampled_frame_output_names": local_fixed_sampled_frame_output_names,
            "fixed_sampled_frame_constants": {
                "text_temperature": FIXED_SAMPLED_TEXT_TEMPERATURE,
                "text_top_p": FIXED_SAMPLED_TEXT_TOP_P,
                "text_top_k": FIXED_SAMPLED_TEXT_TOP_K,
                "audio_temperature": FIXED_SAMPLED_AUDIO_TEMPERATURE,
                "audio_top_p": FIXED_SAMPLED_AUDIO_TOP_P,
                "audio_top_k": FIXED_SAMPLED_AUDIO_TOP_K,
                "audio_repetition_penalty": FIXED_SAMPLED_AUDIO_REPETITION_PENALTY,
            },
        },
    }
    write_json(output_dir / "tts_browser_onnx_meta.json", metadata)


if __name__ == "__main__":
    main()
