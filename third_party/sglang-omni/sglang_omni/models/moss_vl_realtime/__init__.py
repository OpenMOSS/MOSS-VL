"""MOSS-VL realtime serving primitives."""

from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    KV_TRANSACTION_ATTR,
    RUNTIME_STATE_ATTR,
    MossVLRealtimeScheduleBatch,
    commit_moss_vl_realtime_batch,
    is_moss_vl_realtime_batch,
    prepare_moss_vl_realtime_encoder_info_extend,
    rollback_moss_vl_realtime_batch,
)
from sglang_omni.models.moss_vl_realtime.frame_window import (
    FrameWindowEvent,
    FrameWindowPlan,
    RealtimeFrameRecord,
    RealtimeFrameWindowConfig,
    apply_frame_window_plan,
    plan_frame_window,
    stage_segment_frame_records,
)
from sglang_omni.models.moss_vl_realtime.kv_layout import (
    MossVLRealtimeKVLayout,
    insert_encoder_slots,
    read_req_to_token_layout,
    write_req_to_token_layout,
)
from sglang_omni.models.moss_vl_realtime.model_runner import MossVLRealtimeModelRunner
from sglang_omni.models.moss_vl_realtime.model_step import (
    MossVLRealtimeRequestState,
    MossVLRealtimeStepper,
    compute_realtime_mrope_for_segment,
)
from sglang_omni.models.moss_vl_realtime.payload_types import (
    FramePromptEvent,
    build_realtime_append_text,
    build_realtime_frame_text,
)
from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimeKVAppendTransaction,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.segment import (
    REALTIME_FULL_GRID_THW_KEY,
    MossVLRealtimeSegment,
    MossVLRealtimeSegmentBuilder,
)
from sglang_omni.models.moss_vl_realtime.session_state import (
    MossVLRealtimeSession,
    MossVLRealtimeSessionController,
)
from sglang_omni.models.moss_vl_realtime.visibility import (
    append_visible_frame_counts,
    compute_visible_frame_counts,
)

__all__ = [
    "KV_TRANSACTION_ATTR",
    "REALTIME_FULL_GRID_THW_KEY",
    "RUNTIME_STATE_ATTR",
    "FramePromptEvent",
    "FrameWindowEvent",
    "FrameWindowPlan",
    "MossVLRealtimeKVAppendTransaction",
    "MossVLRealtimeKVLayout",
    "MossVLRealtimeModelRunner",
    "MossVLRealtimePhase",
    "MossVLRealtimeRequestState",
    "MossVLRealtimeRuntimeState",
    "MossVLRealtimeScheduleBatch",
    "MossVLRealtimeSegment",
    "MossVLRealtimeSegmentBuilder",
    "MossVLRealtimeSession",
    "MossVLRealtimeSessionController",
    "MossVLRealtimeStepper",
    "RealtimeFrameRecord",
    "RealtimeFrameWindowConfig",
    "append_visible_frame_counts",
    "apply_frame_window_plan",
    "build_realtime_append_text",
    "build_realtime_frame_text",
    "commit_moss_vl_realtime_batch",
    "compute_realtime_mrope_for_segment",
    "compute_visible_frame_counts",
    "insert_encoder_slots",
    "is_moss_vl_realtime_batch",
    "plan_frame_window",
    "prepare_moss_vl_realtime_encoder_info_extend",
    "read_req_to_token_layout",
    "rollback_moss_vl_realtime_batch",
    "stage_segment_frame_records",
    "write_req_to_token_layout",
]
