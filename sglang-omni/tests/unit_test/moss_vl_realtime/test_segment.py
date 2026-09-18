from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_vl_realtime import (
    REALTIME_FULL_GRID_THW_KEY,
    FramePromptEvent,
    MossVLRealtimeSegmentBuilder,
)
from sglang_omni.models.moss_vl_realtime.sglang_model import use_realtime_full_grid_thw


class _Processor:
    def __init__(self, image_token_id: int) -> None:
        self.image_token_id = image_token_id
        self.tokenizer = _Tokenizer()

    def __call__(self, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[101, self.image_token_id, 102]]),
            "pixel_values": torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4),
            "grid_thw": torch.tensor([[1, 4, 4]]),
        }


class _Tokenizer:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return {"input_ids": torch.tensor([[201, 202]])}


def _event(seq_no: int, timestamp: float) -> FramePromptEvent:
    return FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=seq_no,
        timestamp=timestamp,
        frame_ref=f"relay://frame-{seq_no}",
        fingerprint=f"frame-{seq_no}",
    )


def test_segment_encodes_only_delta_frame_but_retains_full_metadata() -> None:
    image_token_id = 999
    builder = MossVLRealtimeSegmentBuilder(
        _Processor(image_token_id),
        image_token_id=image_token_id,
        merge_size=2,
    )
    first = builder.build([_event(0, 0.0)], [object()])
    second = builder.build(
        [_event(1, 0.5)],
        [object()],
        previous_mrope_positions=first.multimodal_inputs.mrope_positions,
        previous_visible_counts=first.multimodal_inputs.visible_frame_counts,
        previous_grid_thw=first.full_grid_thw,
        next_mrope_position=first.next_mrope_position,
        committed_encoder_length=first.encoder_delta_length,
        pending_text_tokens=1,
    )

    assert first.encoder_delta_length == 5
    assert second.encoder_delta_length == 5
    assert second.multimodal_inputs.num_image_tokens == 10
    item = second.multimodal_inputs.mm_items[0]
    assert item.feature.shape[0] == 1
    assert item.grid_thw.tolist() == [[1, 4, 4]]
    assert item.model_specific_data[REALTIME_FULL_GRID_THW_KEY].tolist() == [
        [1, 4, 4],
        [1, 4, 4],
    ]
    # The coalesced sampled token sees one old frame; the image placeholder
    # then makes the new frame visible to subsequent text.
    previous_length = first.multimodal_inputs.visible_frame_counts.numel()
    appended = second.multimodal_inputs.visible_frame_counts[previous_length:]
    assert appended.tolist() == [1, 1, 2, 2]
    assert len(second.encoder_pad_ids) == 5
    assert second.raw_append_ids[-3:] == (101, image_token_id, 102)


def test_full_grid_swap_is_scoped_to_mask_construction() -> None:
    data = {
        "grid_thw": torch.tensor([[1, 4, 4]]),
        REALTIME_FULL_GRID_THW_KEY: torch.tensor([[1, 4, 4], [1, 8, 8]]),
    }
    mm_inputs = [
        SimpleNamespace(
            mm_items=[SimpleNamespace(model_specific_data=data)],
        )
    ]

    with use_realtime_full_grid_thw(mm_inputs):
        assert data["grid_thw"].shape == (2, 3)
    assert data["grid_thw"].tolist() == [[1, 4, 4]]


def test_segment_fills_decode_positions_since_previous_event() -> None:
    image_token_id = 999
    builder = MossVLRealtimeSegmentBuilder(
        _Processor(image_token_id),
        image_token_id=image_token_id,
        merge_size=2,
    )
    first = builder.build([_event(0, 0.0)], [object()])
    second = builder.build(
        [_event(1, 1.0)],
        [object()],
        previous_mrope_positions=first.multimodal_inputs.mrope_positions,
        previous_visible_counts=first.multimodal_inputs.visible_frame_counts,
        previous_grid_thw=first.full_grid_thw,
        next_mrope_position=first.next_mrope_position + 2,
        committed_encoder_length=first.encoder_delta_length,
        committed_decoder_length=(first.multimodal_inputs.mrope_positions.shape[1] + 2),
        pending_text_tokens=1,
    )

    old_len = first.multimodal_inputs.mrope_positions.shape[1]
    full_positions = second.multimodal_inputs.mrope_positions
    assert full_positions.shape[1] == old_len + 2 + 1 + 3
    assert (
        full_positions[:, old_len : old_len + 3].tolist()
        == [
            [
                first.next_mrope_position,
                first.next_mrope_position + 1,
                first.next_mrope_position + 2,
            ],
        ]
        * 3
    )


def test_prompt_segment_extends_only_decoder_and_keeps_frame_visibility() -> None:
    image_token_id = 999
    builder = MossVLRealtimeSegmentBuilder(
        _Processor(image_token_id),
        image_token_id=image_token_id,
        merge_size=2,
    )
    first = builder.build([_event(0, 0.0)], [object()])
    prompt_event = FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=1,
        timestamp=0.0,
        frame_ref=None,
        prompt="How many?",
        final=True,
    )

    prompt = builder.build(
        [prompt_event],
        previous_mrope_positions=first.multimodal_inputs.mrope_positions,
        previous_visible_counts=first.multimodal_inputs.visible_frame_counts,
        previous_grid_thw=first.full_grid_thw,
        next_mrope_position=first.next_mrope_position,
        committed_encoder_length=first.encoder_delta_length,
        pending_text_tokens=1,
    )

    assert prompt.encoder_delta_length == 0
    assert prompt.encoder_pad_ids == ()
    assert prompt.raw_append_ids == (201, 202)
    assert prompt.multimodal_inputs.mm_items == []
    assert prompt.multimodal_inputs.num_image_tokens == first.encoder_delta_length
    assert prompt.multimodal_inputs.visible_frame_counts[-3:].tolist() == [1, 1, 1]


class _MultiFrameProcessor:
    """Processor mock expanding one image placeholder per frame."""

    def __init__(self, image_token_id: int) -> None:
        self.image_token_id = image_token_id
        self.tokenizer = _Tokenizer()
        self.last_text: str | None = None
        self.last_images: list = []

    def __call__(self, **kwargs):
        self.last_text = kwargs["text"]
        self.last_images = list(kwargs["images"])
        frame_count = len(self.last_images)
        assert self.last_text.count("<|image|>") == frame_count
        input_ids = [101]
        for _ in range(frame_count):
            input_ids += [self.image_token_id, 102]
        return {
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": torch.arange(48 * frame_count, dtype=torch.float32).reshape(
                frame_count, 3, 4, 4
            ),
            "grid_thw": torch.tensor([[1, 4, 4]] * frame_count),
        }


def test_batch_segment_orders_prompts_before_frames_sorted_by_timestamp() -> None:
    image_token_id = 999
    processor = _MultiFrameProcessor(image_token_id)
    builder = MossVLRealtimeSegmentBuilder(
        processor,
        image_token_id=image_token_id,
        merge_size=2,
    )
    prompt_first = FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=0,
        timestamp=0.0,
        frame_ref=None,
        prompt="first question",
    )
    frame_late = _event(1, 2.0)
    frame_early = _event(2, 1.0)
    prompt_second = FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=3,
        timestamp=3.0,
        frame_ref=None,
        prompt="second question",
    )
    image_late, image_early = object(), object()

    segment = builder.build(
        [prompt_first, frame_late, frame_early, prompt_second],
        [image_late, image_early],
        next_mrope_position=4,
        committed_decoder_length=4,
    )

    # Prompts keep arrival order and each is followed by <|silence|>; frames
    # follow sorted by timestamp, with the images repacked in the same order.
    assert processor.last_text == (
        "<|im_end|>\n<|im_start|>user\nfirst question"
        "<|im_end|>\n<|im_start|>assistant\n<|silence|>"
        "<|im_end|>\n<|im_start|>user\nsecond question"
        "<|im_end|>\n<|im_start|>assistant\n<|silence|>"
        "<|vision_start|><|time_start|>1.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
        "<|vision_start|><|time_start|>2.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
    )
    assert processor.last_images == [image_early, image_late]
    assert segment.events == (prompt_first, frame_late, frame_early, prompt_second)
    assert segment.encoder_delta_length == 10
    assert segment.full_grid_thw.tolist() == [[1, 4, 4], [1, 4, 4]]
    item = segment.multimodal_inputs.mm_items[0]
    assert item.model_specific_data["realtime_added_frames"] == 2
    assert item.feature.shape[0] == 2
    assert segment.multimodal_inputs.num_image_tokens == 10
