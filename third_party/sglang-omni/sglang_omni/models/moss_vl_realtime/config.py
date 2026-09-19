"""Pipeline configuration for MOSS-VL realtime."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.moss_vl_realtime.platform_compat import device_spec

_PKG = "sglang_omni.models.moss_vl_realtime"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="moss_vl_realtime",
            process="moss_vl_realtime",
            factory=f"{_PKG}.stages.create_sglang_moss_vl_realtime_executor",
            factory_args={
                "device": device_spec(0),
                "max_running_requests": 1,
                "max_new_tokens": 4096,
                "context_length": 262144,
                "mem_fraction_static": 0.40,
                "disable_cuda_graph": False,
                "page_size": 1,
                "enable_async_decode": False,
            },
            gpu=0,
            terminal=True,
        )
    ]


class MossVLRealtimePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MossVLRealtimeForConditionalGeneration"
    supports_video_realtime: ClassVar[bool] = True

    model_path: str
    entry_stage: str = "moss_vl_realtime"
    stages: list[StageConfig] = Field(default_factory=_stages)

    # Vision-KV two-level sliding window (all off by default; the windowed
    # server path is unchanged when disabled). Environment variables override
    # these fields at the serving process: REALTIME_FRAME_WINDOW_ENABLED,
    # REALTIME_FRAME_WINDOW_RAW_S, REALTIME_FRAME_POOL_WINDOW_S,
    # REALTIME_FRAME_POOL_RATIO.
    realtime_frame_window_enabled: bool = False
    realtime_frame_window_raw_s: float | None = None
    realtime_frame_pool_window_s: float | None = None
    realtime_frame_pool_ratio: int | None = None
    realtime_frame_pooling_enabled: bool | None = None

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        window_args: dict[str, Any] = {}
        if self.realtime_frame_window_enabled:
            window_args["realtime_frame_window_enabled"] = True
        if self.realtime_frame_window_raw_s is not None:
            window_args["realtime_frame_window_raw_s"] = (
                self.realtime_frame_window_raw_s
            )
        if self.realtime_frame_pool_window_s is not None:
            window_args["realtime_frame_pool_window_s"] = (
                self.realtime_frame_pool_window_s
            )
        if self.realtime_frame_pool_ratio is not None:
            window_args["realtime_frame_pool_ratio"] = self.realtime_frame_pool_ratio
        if self.realtime_frame_pooling_enabled is not None:
            window_args["realtime_frame_pooling_enabled"] = (
                self.realtime_frame_pooling_enabled
            )
        if window_args:
            for stage in self.stages:
                if stage.name != "moss_vl_realtime":
                    continue
                stage.factory_args = {**stage.factory_args, **window_args}


EntryClass = MossVLRealtimePipelineConfig
