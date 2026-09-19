from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    append_visible_frame_counts,
    compute_visible_frame_counts,
)

IMAGE_TOKEN_ID = 99


def test_image_placeholder_is_the_visibility_boundary() -> None:
    # prompt/time tokens, first image, separator, second image, trailing text
    input_ids = torch.tensor([[10, 11, IMAGE_TOKEN_ID, 12, IMAGE_TOKEN_ID, 13]])

    counts = compute_visible_frame_counts(
        input_ids,
        image_token_id=IMAGE_TOKEN_ID,
    )

    assert counts.dtype == torch.int32
    assert counts.tolist() == [0, 0, 1, 1, 2, 2]


def test_incremental_segment_starts_with_previously_visible_frames() -> None:
    previous = torch.tensor([0, 1, 1], dtype=torch.int32)
    # A provisional sampled token and timestamp occur before the new image.
    new_ids = torch.tensor([20, 21, IMAGE_TOKEN_ID, 22])

    combined = append_visible_frame_counts(
        previous,
        new_ids,
        image_token_id=IMAGE_TOKEN_ID,
    )

    assert combined.tolist() == [0, 1, 1, 1, 1, 2, 2]


def test_prompt_only_segment_keeps_current_visibility() -> None:
    combined = append_visible_frame_counts(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([30, 31, 32]),
        image_token_id=IMAGE_TOKEN_ID,
    )
    assert combined.tolist() == [0, 1, 1, 1, 1]


def test_visibility_rejects_batch_size_greater_than_one() -> None:
    with pytest.raises(ValueError, match="batch size 1"):
        compute_visible_frame_counts(
            torch.zeros((2, 3), dtype=torch.long),
            image_token_id=IMAGE_TOKEN_ID,
        )
