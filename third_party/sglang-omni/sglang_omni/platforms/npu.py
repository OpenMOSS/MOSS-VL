from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.interface import OmniPlatform

logger = logging.getLogger(__name__)

if False:  # TYPE_CHECKING
    from sglang_omni.pipeline.stage_workers import StageLaunchConfig


class NPUOmniPlatform(OmniPlatform):
    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    @property
    def visible_devices_env_key(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_stage_process_env(
        self,
        spec: "StageLaunchConfig",
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Pin each TP rank process to one physical NPU via ASCEND_RT_VISIBLE_DEVICES."""
        if spec.tp_size <= 1:
            return {}

        source_env = env if env is not None else os.environ
        key = self.visible_devices_env_key
        original_visible = source_env.get(key)
        if spec.gpu_id is None:
            raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
        if original_visible:
            visible_devices = [item.strip() for item in original_visible.split(",")]
            if spec.gpu_id >= len(visible_devices):
                raise ValueError(
                    f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                    f"but {key} only exposes {visible_devices}"
                )
            mapped_gpu = visible_devices[spec.gpu_id]
        else:
            mapped_gpu = str(spec.gpu_id)

        return {
            key: mapped_gpu,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        }

    def enable_code2wav_graph(self):
        return False
