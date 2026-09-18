"""Externally driven single-request steps for MOSS-VL realtime generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_vl_realtime.payload_types import build_realtime_append_text


@dataclass(slots=True)
class MossVLRealtimeRequestState:
    """All mutable model state owned by one realtime request."""

    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    position_ids: torch.LongTensor
    past_key_values: Any
    next_token_logits: torch.Tensor
    text_cache_position: int
    vision_cache_position: int
    next_mrope_position: int
    visible_vision_length: int = 0
    full_vision_token_info: list[dict[str, Any]] | None = None
    cross_attention_mask: torch.Tensor | None = None

    @property
    def pending_text_length(self) -> int:
        """Number of appended text tokens not written to the KV cache yet."""
        return self.input_ids.shape[1] - self.text_cache_position


def compute_realtime_mrope_for_segment(
    *,
    new_input_ids: torch.LongTensor,
    new_grid_thw: torch.LongTensor | None,
    start_position: int,
    image_token_id: int,
    merge_size: int,
) -> tuple[torch.LongTensor, torch.LongTensor | None, int]:
    """Compute text and vision MRoPE positions for one appended segment."""
    if new_input_ids.ndim != 2 or new_input_ids.shape[0] != 1:
        raise ValueError("new_input_ids must have shape (1, sequence_length)")
    if start_position < 0:
        raise ValueError("start_position must be non-negative")
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")

    device = new_input_ids.device
    sequence_length = new_input_ids.shape[1]
    text_positions = torch.empty(
        (3, 1, sequence_length), dtype=torch.long, device=device
    )
    vision_position_chunks: list[torch.Tensor] = []
    current_position = start_position
    frame_index = 0
    frame_count = 0 if new_grid_thw is None else new_grid_thw.shape[0]

    for token_index in range(sequence_length):
        token_id = int(new_input_ids[0, token_index].item())
        if token_id != image_token_id:
            text_positions[:, 0, token_index] = current_position
            current_position += 1
            continue

        if new_grid_thw is None or frame_index >= frame_count:
            raise ValueError("image token has no aligned frame grid")

        grid = new_grid_thw[frame_index]
        temporal = int(grid[0].item())
        grid_height = int(grid[1].item())
        grid_width = int(grid[2].item())
        if temporal != 1:
            raise ValueError("realtime frame events must contain exactly one frame")
        if grid_height % merge_size or grid_width % merge_size:
            raise ValueError("frame grid must be divisible by merge_size")

        merged_height = grid_height // merge_size
        merged_width = grid_width // merge_size
        max_spatial_extent = max(merged_height, merged_width)

        y = torch.arange(merged_height, dtype=torch.long, device=device)
        x = torch.arange(merged_width, dtype=torch.long, device=device)
        y = y.view(merged_height, 1).expand(-1, merged_width).reshape(-1)
        x = x.view(1, merged_width).expand(merged_height, -1).reshape(-1)
        time_positions = torch.full_like(y, current_position)
        grid_positions = torch.stack(
            [time_positions, time_positions + y, time_positions + x], dim=0
        )

        separator_position = current_position + max_spatial_extent
        separator_positions = torch.full(
            (3, 1), separator_position, dtype=torch.long, device=device
        )
        vision_position_chunks.append(
            torch.cat([grid_positions, separator_positions], dim=1)
        )
        text_positions[:, 0, token_index] = separator_position
        current_position = separator_position + 1
        frame_index += 1

    if frame_index != frame_count:
        raise ValueError(
            f"received {frame_count} frame grids but found {frame_index} image tokens"
        )

    vision_positions = None
    if vision_position_chunks:
        vision_positions = torch.cat(vision_position_chunks, dim=1).unsqueeze(1)
    return text_positions, vision_positions, current_position


class MossVLRealtimeStepper:
    """Run bounded MOSS-VL realtime steps without queues or a model-owned loop."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self.image_token_id = int(model.config.image_token_id)
        self.merge_size = int(model.model.visual.spatial_merge_size)
        self.vision_seq_pad_multiple = int(model.config.vision_seq_pad_multiple)

    @torch.no_grad()
    def initial_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MossVLRealtimeRequestState:
        """Cache a pure-text initial context and return request-owned state."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MOSS-VL realtime stepping only supports batch size 1")
        if bool((input_ids == self.image_token_id).any()):
            raise ValueError("initial_prefill only accepts pure-text input")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        sequence_length = input_ids.shape[1]
        position_ids = (
            torch.arange(sequence_length, dtype=torch.long, device=device)
            .view(1, 1, sequence_length)
            .expand(3, 1, sequence_length)
            .contiguous()
        )
        cache_position = torch.arange(sequence_length, dtype=torch.long, device=device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        return MossVLRealtimeRequestState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=outputs.past_key_values,
            next_token_logits=outputs.logits[:, -1, :].float(),
            text_cache_position=sequence_length,
            vision_cache_position=0,
            next_mrope_position=sequence_length,
        )

    @torch.no_grad()
    def decode_one_step(
        self,
        state: MossVLRealtimeRequestState,
        *,
        forced_token_id: int | None = None,
    ) -> torch.LongTensor:
        """Choose and cache one text token when no event is being coalesced."""
        next_token = self.sample_next_token(state, forced_token_id=forced_token_id)
        self.commit_pending_tokens(state)
        return next_token

    @torch.no_grad()
    def sample_next_token(
        self,
        state: MossVLRealtimeRequestState,
        *,
        forced_token_id: int | None = None,
    ) -> torch.LongTensor:
        """Append one sampled token without writing it to the KV cache yet.

        The released realtime loop samples a token, drains external events, and
        then forwards the sampled token and drained event in one packed step.
        Keeping the token pending preserves that ordering for an external
        scheduler. Call ``apply_event_and_extend`` to coalesce an event or
        ``commit_pending_tokens`` when no event arrived.
        """
        if state.pending_text_length:
            raise RuntimeError("commit the pending token before sampling again")
        scores = state.next_token_logits.clone()
        ellipsis_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|...|>")
        if isinstance(ellipsis_token_id, int) and ellipsis_token_id >= 0:
            scores[:, ellipsis_token_id] = float("-inf")

        if forced_token_id is None:
            next_token = scores.argmax(dim=-1)
        else:
            if forced_token_id < 0 or forced_token_id >= scores.shape[-1]:
                raise ValueError("forced_token_id is outside the vocabulary")
            next_token = torch.tensor(
                [forced_token_id], dtype=torch.long, device=state.input_ids.device
            )

        state.input_ids = torch.cat([state.input_ids, next_token[:, None]], dim=1)
        state.attention_mask = torch.cat(
            [
                state.attention_mask,
                state.attention_mask.new_ones((1, 1)),
            ],
            dim=1,
        )
        token_position = torch.full(
            (3, 1, 1),
            state.next_mrope_position,
            dtype=torch.long,
            device=state.input_ids.device,
        )
        state.position_ids = torch.cat([state.position_ids, token_position], dim=-1)
        state.next_mrope_position += 1
        state.cross_attention_mask = self._build_cross_attention_mask(state)
        return next_token

    @torch.no_grad()
    def commit_pending_tokens(self, state: MossVLRealtimeRequestState) -> None:
        """Write pending text tokens to the cache without appending an event."""
        pending_text_length = state.pending_text_length
        if pending_text_length <= 0:
            raise RuntimeError("request state has no pending text tokens")
        cache_position = torch.arange(
            state.text_cache_position,
            state.text_cache_position + pending_text_length,
            dtype=torch.long,
            device=state.input_ids.device,
        )
        state.cross_attention_mask = self._build_cross_attention_mask(state)
        self._forward_new_tokens(state, cache_position=cache_position)
        state.text_cache_position += pending_text_length

    @torch.no_grad()
    def apply_event_and_extend(
        self,
        state: MossVLRealtimeRequestState,
        *,
        prompt: str | None = None,
        prompts: Sequence[str] = (),
        frames: Sequence[tuple[Any, float]] = (),
    ) -> None:
        """Append one drain cycle of prompts/frames and cache the new segment."""
        all_prompts: list[str] = [prompt] if prompt is not None else []
        all_prompts.extend(prompts)
        ordered_frames = sorted(frames, key=lambda item: item[1])
        frame_images = [image for image, _ in ordered_frames]
        frame_timestamps = [float(timestamp) for _, timestamp in ordered_frames]
        append_text = build_realtime_append_text(
            prompts=all_prompts, frame_timestamps=frame_timestamps
        )
        if not append_text:
            raise ValueError("event must contain a prompt or at least one frame")

        device = state.input_ids.device
        pixel_values = None
        grid_thw = None
        media_nums_per_sample = None
        if frame_images:
            inputs = self.processor(
                text=append_text,
                images=frame_images,
                add_special_tokens=False,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            grid_thw = inputs["grid_thw"].to(device)
            media_nums_per_sample = inputs.get("media_nums_per_sample")
            new_input_ids = inputs["input_ids"].to(device)
        else:
            inputs = self.processor.tokenizer(
                append_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            new_input_ids = inputs["input_ids"].to(device)

        new_text_positions, new_vision_positions, next_position = (
            compute_realtime_mrope_for_segment(
                new_input_ids=new_input_ids,
                new_grid_thw=grid_thw,
                start_position=state.next_mrope_position,
                image_token_id=self.image_token_id,
                merge_size=self.merge_size,
            )
        )
        new_token_count = new_input_ids.shape[1]
        pending_text_length = state.pending_text_length
        total_forward_token_count = pending_text_length + new_token_count
        cache_position = torch.arange(
            state.text_cache_position,
            state.text_cache_position + total_forward_token_count,
            dtype=torch.long,
            device=device,
        )

        state.input_ids = torch.cat([state.input_ids, new_input_ids], dim=1)
        state.attention_mask = torch.cat(
            [
                state.attention_mask,
                state.attention_mask.new_ones((1, new_token_count)),
            ],
            dim=1,
        )
        state.position_ids = torch.cat([state.position_ids, new_text_positions], dim=-1)
        state.next_mrope_position = next_position

        vision_cache_positions = None
        if grid_thw is not None:
            vision_cache_positions, new_vision_positions = self._append_vision_state(
                state, grid_thw, new_vision_positions
            )
        state.cross_attention_mask = self._build_cross_attention_mask(state)

        self._forward_new_tokens(
            state,
            cache_position=cache_position,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            media_nums_per_sample=media_nums_per_sample,
            vision_position_ids=new_vision_positions,
            vision_cache_position=vision_cache_positions,
        )
        state.text_cache_position += total_forward_token_count

    def _append_vision_state(
        self,
        state: MossVLRealtimeRequestState,
        grid_thw: torch.LongTensor,
        vision_position_ids: torch.LongTensor | None,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        tokens_per_media = grid_thw.prod(dim=1) // (self.merge_size**2)
        actual_new_length = int((tokens_per_media + grid_thw[:, 0]).sum().item())
        padded_new_length = actual_new_length
        if (
            self.vision_seq_pad_multiple > 1
            and actual_new_length % self.vision_seq_pad_multiple
        ):
            padded_new_length = (
                (actual_new_length + self.vision_seq_pad_multiple - 1)
                // self.vision_seq_pad_multiple
                * self.vision_seq_pad_multiple
            )

        previous_visible_length = state.visible_vision_length
        medias: list[dict[str, Any]] = []
        if state.full_vision_token_info is not None:
            medias.extend(state.full_vision_token_info[0].get("medias", []))

        local_offset = 0
        for index in range(grid_thw.shape[0]):
            temporal, grid_height, grid_width = (
                int(value.item()) for value in grid_thw[index]
            )
            token_count = grid_height * grid_width // (self.merge_size**2) * temporal
            media_length = token_count + temporal
            medias.append(
                {
                    "start": previous_visible_length + local_offset,
                    "end": previous_visible_length + local_offset + media_length,
                    "length": media_length,
                    "num_frames": temporal,
                    "grid_h": grid_height,
                    "grid_w": grid_width,
                    "vision_tokens_per_frame": token_count // temporal,
                    "has_separator": True,
                }
            )
            local_offset += media_length

        state.visible_vision_length = previous_visible_length + actual_new_length
        state.vision_cache_position = previous_visible_length + padded_new_length
        state.full_vision_token_info = [
            {
                "medias": medias,
                "total_length": state.visible_vision_length,
                "pad_start": state.visible_vision_length,
                "pad_end": state.vision_cache_position,
            }
        ]

        cache_positions = torch.arange(
            previous_visible_length,
            state.vision_cache_position,
            dtype=torch.long,
            device=grid_thw.device,
        )
        if vision_position_ids is None:
            vision_position_ids = torch.zeros(
                (3, 1, 0), dtype=torch.long, device=grid_thw.device
            )
        padding_length = padded_new_length - vision_position_ids.shape[-1]
        if padding_length > 0:
            padding = torch.zeros(
                (3, 1, padding_length), dtype=torch.long, device=grid_thw.device
            )
            vision_position_ids = torch.cat([vision_position_ids, padding], dim=-1)
        return cache_positions, vision_position_ids

    def _build_cross_attention_mask(
        self, state: MossVLRealtimeRequestState
    ) -> torch.Tensor | None:
        if state.full_vision_token_info is None:
            return None
        medias = state.full_vision_token_info[0].get("medias", [])
        if not medias:
            return None
        frame_count = sum(int(media["num_frames"]) for media in medias)
        cumulative_images = (state.input_ids == self.image_token_id).cumsum(dim=1)
        frame_indices = torch.arange(
            frame_count, dtype=torch.long, device=state.input_ids.device
        )
        visible = cumulative_images.unsqueeze(-1) > frame_indices
        return (~visible).unsqueeze(1)

    def _forward_new_tokens(
        self,
        state: MossVLRealtimeRequestState,
        *,
        cache_position: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.LongTensor | None = None,
        media_nums_per_sample: list[int] | None = None,
        vision_position_ids: torch.LongTensor | None = None,
        vision_cache_position: torch.LongTensor | None = None,
    ) -> None:
        model_inputs = self.model.prepare_inputs_for_real_time_generation(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            cache_position=cache_position,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            media_nums_per_sample=media_nums_per_sample,
            vision_position_ids=vision_position_ids,
            vision_cache_position=vision_cache_position,
            cross_attention_mask=state.cross_attention_mask,
            full_vision_token_info=state.full_vision_token_info,
            logits_to_keep=1,
        )
        outputs = self.model(**model_inputs, return_dict=True)
        state.past_key_values = outputs.past_key_values
        state.next_token_logits = outputs.logits[:, -1, :].float()
