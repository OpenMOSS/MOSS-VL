"""Incremental frame-visibility metadata for MOSS-VL cross-attention."""

from __future__ import annotations

import torch


def compute_visible_frame_counts(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    initial_visible_frames: int = 0,
) -> torch.Tensor:
    """Return the cumulative visible-frame count for each new text token.

    The image placeholder token is the boundary where its frame becomes
    visible. Tokens before that placeholder only attend to previously committed
    frames, matching the offline processor's prefix-causal mask.
    """
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("realtime visibility only supports batch size 1")
        input_ids = input_ids[0]
    elif input_ids.ndim != 1:
        raise ValueError("input_ids must be 1-D or have shape (1, sequence_length)")
    if isinstance(image_token_id, bool) or not isinstance(image_token_id, int):
        raise TypeError("image_token_id must be an integer")
    if (
        isinstance(initial_visible_frames, bool)
        or not isinstance(initial_visible_frames, int)
        or initial_visible_frames < 0
    ):
        raise ValueError("initial_visible_frames must be a non-negative integer")

    return (input_ids == image_token_id).to(dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    ) + initial_visible_frames


def append_visible_frame_counts(
    previous_counts: torch.Tensor | None,
    new_input_ids: torch.Tensor,
    *,
    image_token_id: int,
) -> torch.Tensor:
    """Append one segment while retaining globally indexed visibility rows."""
    if previous_counts is None:
        initial_visible_frames = 0
    else:
        if not isinstance(previous_counts, torch.Tensor):
            raise TypeError("previous_counts must be a torch.Tensor")
        if previous_counts.ndim != 1:
            raise ValueError("previous_counts must be one-dimensional")
        if previous_counts.numel() == 0:
            initial_visible_frames = 0
        else:
            initial_visible_frames = int(previous_counts[-1].item())

    new_counts = compute_visible_frame_counts(
        new_input_ids,
        image_token_id=image_token_id,
        initial_visible_frames=initial_visible_frames,
    )
    if previous_counts is None or previous_counts.numel() == 0:
        return new_counts
    return torch.cat(
        [previous_counts.to(device=new_counts.device, dtype=torch.int32), new_counts]
    )
