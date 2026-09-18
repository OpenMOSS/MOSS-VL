"""SGLang MOSS-VL wrapper for delta-frame encoding."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sglang.srt.models.moss_vl import MossVLForConditionalGeneration

from sglang_omni.models.moss_vl_realtime.segment import REALTIME_FULL_GRID_THW_KEY


@contextmanager
def use_realtime_full_grid_thw(mm_inputs: list[Any] | None) -> Iterator[None]:
    """Expose cumulative grids only while building frame-visibility masks."""
    swapped: list[tuple[dict[str, Any], Any]] = []
    for mm_input in mm_inputs or ():
        if mm_input is None or not getattr(mm_input, "mm_items", None):
            continue
        data = mm_input.mm_items[0].model_specific_data
        full_grid = data.get(REALTIME_FULL_GRID_THW_KEY)
        if full_grid is None:
            continue
        swapped.append((data, data.get("grid_thw")))
        data["grid_thw"] = full_grid
    try:
        yield
    finally:
        for data, delta_grid in swapped:
            data["grid_thw"] = delta_grid


class MossVLRealtimeForConditionalGeneration(MossVLForConditionalGeneration):
    """Encode only delta pixels while masking against all committed frames."""

    def _build_cross_attention_custom_mask(self, forward_batch):
        with use_realtime_full_grid_thw(forward_batch.mm_inputs):
            return super()._build_cross_attention_custom_mask(forward_batch)
