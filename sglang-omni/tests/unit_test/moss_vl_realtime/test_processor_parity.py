from __future__ import annotations

import os
from pathlib import Path

import pytest

from sglang_omni.models.moss_vl_realtime.payload_types import (
    FramePromptEvent,
    build_realtime_append_text,
    build_realtime_frame_text,
)


def test_frame_prompt_event_validation() -> None:
    event = FramePromptEvent(
        request_id="request-1",
        session_id="session-1",
        seq_no=0,
        timestamp=5.0,
        frame_ref="relay://frame-1",
        prompt="What changed?",
    )
    assert event.timestamp == 5.0

    with pytest.raises(ValueError, match="timestamp"):
        FramePromptEvent(
            request_id="request-1",
            session_id="session-1",
            seq_no=1,
            timestamp=-1.0,
            frame_ref="relay://frame-2",
        )


def test_realtime_append_text_matches_released_model_ordering() -> None:
    assert build_realtime_frame_text(5.04) == (
        "<|vision_start|><|time_start|>5.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
    )
    assert build_realtime_append_text(
        prompts=["What changed?"], frame_timestamps=[5.0]
    ) == (
        "<|im_end|>\n<|im_start|>user\nWhat changed?"
        "<|im_end|>\n<|im_start|>assistant\n<|silence|>"
        "<|vision_start|><|time_start|>5.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
    )


def test_realtime_append_text_matches_reference_drain_batch_semantics() -> None:
    # Mirrors the Transformers drain: prompts first in arrival order, one
    # <|silence|> after each prompt, including prompt-only cycles, then
    # the frame wrappers in (caller-sorted) timestamp order.
    assert build_realtime_append_text(
        prompts=["first question", "second question"],
        frame_timestamps=[1.0, 2.0],
    ) == (
        "<|im_end|>\n<|im_start|>user\nfirst question"
        "<|im_end|>\n<|im_start|>assistant\n<|silence|>"
        "<|im_end|>\n<|im_start|>user\nsecond question"
        "<|im_end|>\n<|im_start|>assistant\n<|silence|>"
        "<|vision_start|><|time_start|>1.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
        "<|vision_start|><|time_start|>2.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
    )
    # Prompt-only turns also consume the trained opener before generation.
    assert build_realtime_append_text(prompts=["hello"]) == (
        "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<|silence|>"
    )
    assert build_realtime_append_text(frame_timestamps=[1.0]) == (
        "<|vision_start|><|time_start|>1.0 seconds"
        "<|time_end|><|image|><|vision_end|>"
    )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run the MOSS-VL processor golden test")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def test_single_frame_segment_matches_realtime_image_processor() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torchvision_functional = pytest.importorskip("torchvision.transforms.functional")

    model_path = _required_path("MOSS_VL_MODEL_PATH")
    video_path = _required_path("MOSS_VL_TEST_VIDEO")
    timestamp = float(os.environ.get("MOSS_VL_TEST_TIMESTAMP", "5.0"))

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    frame_tensor, _ = processor.video_processor._fetch_video_segment(
        str(video_path), [timestamp]
    )
    frame = torchvision_functional.to_pil_image(frame_tensor[0])

    offline = processor(
        text=["<|video|>"],
        videos=[{"video_path": str(video_path), "segments": [[timestamp]]}],
        add_special_tokens=False,
        return_tensors="pt",
    )
    realtime = processor(
        text=[build_realtime_frame_text(timestamp)],
        images=[frame],
        add_special_tokens=False,
        return_tensors="pt",
    )

    assert processor.tokenizer.decode(offline["input_ids"][0]) == (
        f"<|vision_start|><|time_start|>{timestamp:.1f} seconds"
        "<|time_end|><|image_pad|><|vision_end|>"
    )
    for key in ("input_ids", "attention_mask", "grid_thw"):
        assert torch.equal(offline[key], realtime[key]), key
    pixel_diff = (offline["pixel_values"] - realtime["pixel_values"]).abs()
    assert pixel_diff.max().item() <= 4 / 127.5 + 1e-6
    assert pixel_diff.mean().item() <= 5e-4

    if "media_nums_per_sample" in offline or "media_nums_per_sample" in realtime:
        assert (
            "media_nums_per_sample" in offline and "media_nums_per_sample" in realtime
        )
        assert offline["media_nums_per_sample"] == realtime["media_nums_per_sample"]

    if "cross_attention_mask" in offline or "cross_attention_mask" in realtime:
        assert "cross_attention_mask" in offline and "cross_attention_mask" in realtime
        assert torch.equal(
            offline["cross_attention_mask"], realtime["cross_attention_mask"]
        )
