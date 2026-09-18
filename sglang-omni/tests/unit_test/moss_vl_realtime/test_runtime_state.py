from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)


def _state() -> MossVLRealtimeRuntimeState:
    return MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        req_pool_index=1,
        encoder_length=2,
        decoder_length=2,
        surviving_frame_count=1,
        next_mrope_position=9,
    )


def _page_table() -> torch.Tensor:
    table = torch.zeros((2, 12), dtype=torch.int64)
    table[1, :4] = torch.tensor([11, 12, 21, 22])
    return table


def test_kv_append_commits_layout_lengths_and_positions_atomically() -> None:
    state = _state()
    table = _page_table()

    transaction = state.begin_kv_append(
        table,
        new_encoder_slots=[13, 14],
        new_decoder_slots=[23, 24, 25],
        added_frames=1,
        next_mrope_position=17,
    )

    assert table[1, :9].tolist() == [11, 12, 13, 14, 21, 22, 23, 24, 25]
    assert state.phase is MossVLRealtimePhase.EXTENDING
    assert state.encoder_length == 2
    assert state.decoder_length == 2

    transaction.commit()

    assert state.phase is MossVLRealtimePhase.DECODING
    assert state.encoder_length == 4
    assert state.decoder_length == 5
    assert state.surviving_frame_count == 2
    assert state.next_mrope_position == 17


def test_kv_append_rolls_back_page_table_and_releases_only_new_slots() -> None:
    state = _state()
    table = _page_table()
    released: list[tuple[int, ...]] = []

    with (
        pytest.raises(RuntimeError, match="forward failed"),
        state.begin_kv_append(
            table,
            new_encoder_slots=[13],
            new_decoder_slots=[23],
            added_frames=1,
            next_mrope_position=15,
            release_slots=released.append,
        ),
    ):
        raise RuntimeError("forward failed")

    assert table[1, :4].tolist() == [11, 12, 21, 22]
    assert table[1, 4:].count_nonzero().item() == 0
    assert released == [(13, 23)]
    assert state.phase is MossVLRealtimePhase.WAITING_FOR_EVENT
    assert state.encoder_length == 2
    assert state.decoder_length == 2
    assert state.surviving_frame_count == 1
    assert state.next_mrope_position == 9


def test_runtime_state_rejects_nested_append_and_pool_rebinding() -> None:
    state = _state()
    transaction = state.begin_kv_append(_page_table(), new_decoder_slots=[23])

    with pytest.raises(RuntimeError, match="already in flight"):
        state.begin_kv_append(_page_table(), new_decoder_slots=[24])
    with pytest.raises(RuntimeError, match="ownership cannot change"):
        state.bind_req_pool_index(0)

    transaction.rollback()


def test_kv_append_accepts_allocator_tensor_slots() -> None:
    state = _state()
    table = _page_table()

    transaction = state.begin_kv_append(
        table,
        new_encoder_slots=torch.tensor([13, 14], dtype=torch.int64),
        new_decoder_slots=torch.tensor([23], dtype=torch.int64),
    )
    transaction.commit()

    assert table[1, :7].tolist() == [11, 12, 13, 14, 21, 22, 23]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_runtime_state_rejects_invalid_token_rate(value: float) -> None:
    with pytest.raises(ValueError, match="max_tokens_per_turn"):
        MossVLRealtimeRuntimeState(
            request_id="req-rate",
            session_id="session-rate",
            max_tokens_per_turn=value,
        )
