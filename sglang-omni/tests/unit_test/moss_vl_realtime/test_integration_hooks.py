from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_vl_realtime import (
    RUNTIME_STATE_ATTR,
    MossVLRealtimeModelRunner,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
    MossVLRealtimeScheduleBatch,
)


def test_realtime_schedule_batch_owns_incremental_encoder_hooks() -> None:
    assert "prepare_encoder_info_extend" in MossVLRealtimeScheduleBatch.__dict__
    assert "prepare_for_extend" in MossVLRealtimeScheduleBatch.__dict__


def test_model_runner_advances_committed_decode_state() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        encoder_length=5,
        decoder_length=8,
        next_mrope_position=11,
        phase=MossVLRealtimePhase.DECODING,
    )
    req = SimpleNamespace()
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_decode(None, None, batch, [])

    assert state.decoder_length == 9
    assert state.next_mrope_position == 12
    assert state.phase is MossVLRealtimePhase.DECODING


def test_model_runner_commits_cumulative_realtime_metadata() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-meta",
        session_id="session-meta",
        req_pool_index=0,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_staged_mrope_positions="mrope",
        _moss_vl_realtime_staged_visible_frame_counts="visible",
        _moss_vl_realtime_staged_full_grid_thw="grid",
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.mrope_positions == "mrope"
    assert state.visible_frame_counts == "visible"
    assert state.full_grid_thw == "grid"
    assert not hasattr(req, "_moss_vl_realtime_staged_full_grid_thw")


def test_model_runner_initializes_committed_text_prefill_state() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-initial",
        session_id="session-initial",
    )
    req = SimpleNamespace(req_pool_idx=3)
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([7]))
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.req_pool_index == 3
    assert state.encoder_length == 0
    assert state.decoder_length == 7
    assert state.next_mrope_position == 7
    assert state.phase is MossVLRealtimePhase.DECODING


def test_model_runner_commits_prompt_turn_transition() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-turn",
        session_id="session-turn",
        req_pool_index=0,
        turn_id=2,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_staged_events=[
            {
                "seq_no": 7,
                "timestamp": 4.0,
                "prompt": "What changed?",
                "final": False,
            },
        ],
        _moss_vl_realtime_staged_turn_transition={
            "interrupted_turn_id": 2,
            "turn_id": 3,
            "prompt_seq_nos": [7],
        },
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([9]))
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.turn_id == 3
    (processed,) = req._moss_vl_realtime_processed_events
    assert processed["interrupted_turn_id"] == 2
    assert processed["turn_id"] == 3
    assert not hasattr(req, "_moss_vl_realtime_staged_turn_transition")


def test_model_runner_assigns_turn_per_prompt_in_batch() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-batch",
        session_id="session-batch",
        req_pool_index=0,
        turn_id=0,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_staged_events=[
            {"seq_no": 0, "timestamp": 0.0, "prompt": "p0", "final": False},
            {"seq_no": 1, "timestamp": 1.0, "frame_ref": "relay://f1", "final": False},
            {"seq_no": 2, "timestamp": 2.0, "prompt": "p2", "final": False},
        ],
        _moss_vl_realtime_staged_turn_transition={
            "interrupted_turn_id": 0,
            "turn_id": 2,
            "prompt_seq_nos": [0, 2],
        },
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([3]))
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.turn_id == 2
    processed = req._moss_vl_realtime_processed_events
    assert [
        (
            event["seq_no"],
            event.get("interrupted_turn_id"),
            event.get("turn_id"),
        )
        for event in processed
    ] == [(0, 0, 1), (1, None, None), (2, 1, 2)]
    assert not hasattr(req, "_moss_vl_realtime_staged_events")
