"""Request-to-token layout helpers for incremental MOSS-VL vision KV."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


def _slot_tuple(slots: Iterable[int], name: str) -> tuple[int, ...]:
    if isinstance(slots, torch.Tensor):
        if slots.ndim != 1:
            raise ValueError(f"{name} tensor must be one-dimensional")
        values = tuple(slots.tolist())
    else:
        values = tuple(slots)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integers")
        if value <= 0:
            raise ValueError(f"{name} must contain positive KV slot IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate KV slot IDs")
    return values


@dataclass(frozen=True, slots=True)
class MossVLRealtimeKVLayout:
    """Logical encoder-prefix and decoder-text slots for one live request.

    SGLang cross-attention reads the first encoder_length cells from the
    request-to-token row. New vision slots therefore have to be inserted before
    the decoder mapping. The decoder KV tensors stay in their original physical
    slots; only this small page-table row changes.
    """

    encoder_slots: tuple[int, ...] = ()
    decoder_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        encoder_slots = _slot_tuple(self.encoder_slots, "encoder_slots")
        decoder_slots = _slot_tuple(self.decoder_slots, "decoder_slots")
        if set(encoder_slots).intersection(decoder_slots):
            raise ValueError("encoder and decoder slots must not overlap")
        object.__setattr__(self, "encoder_slots", encoder_slots)
        object.__setattr__(self, "decoder_slots", decoder_slots)

    @property
    def encoder_length(self) -> int:
        return len(self.encoder_slots)

    @property
    def decoder_length(self) -> int:
        return len(self.decoder_slots)

    @property
    def total_length(self) -> int:
        return self.encoder_length + self.decoder_length

    def append_encoder(self, slots: Iterable[int]) -> MossVLRealtimeKVLayout:
        new_slots = _slot_tuple(slots, "new encoder slots")
        return MossVLRealtimeKVLayout(
            encoder_slots=self.encoder_slots + new_slots,
            decoder_slots=self.decoder_slots,
        )

    def append_decoder(self, slots: Iterable[int]) -> MossVLRealtimeKVLayout:
        new_slots = _slot_tuple(slots, "new decoder slots")
        return MossVLRealtimeKVLayout(
            encoder_slots=self.encoder_slots,
            decoder_slots=self.decoder_slots + new_slots,
        )

    def as_tuple(self) -> tuple[int, ...]:
        return self.encoder_slots + self.decoder_slots


def read_req_to_token_layout(
    req_to_token: torch.Tensor,
    *,
    req_pool_index: int,
    encoder_length: int,
    decoder_length: int,
) -> MossVLRealtimeKVLayout:
    """Read one request row using explicit committed lengths."""
    _validate_row_arguments(
        req_to_token,
        req_pool_index=req_pool_index,
        total_length=encoder_length + decoder_length,
    )
    if encoder_length < 0 or decoder_length < 0:
        raise ValueError("committed KV lengths must be non-negative")
    row = req_to_token[req_pool_index]
    encoder_slots = tuple(int(value) for value in row[:encoder_length].tolist())
    decoder_start = encoder_length
    decoder_end = decoder_start + decoder_length
    decoder_slots = tuple(
        int(value) for value in row[decoder_start:decoder_end].tolist()
    )
    return MossVLRealtimeKVLayout(
        encoder_slots=encoder_slots,
        decoder_slots=decoder_slots,
    )


def write_req_to_token_layout(
    req_to_token: torch.Tensor,
    *,
    req_pool_index: int,
    layout: MossVLRealtimeKVLayout,
    clear_tail: bool = False,
) -> None:
    """Install a logical layout without copying any physical KV tensors."""
    _validate_row_arguments(
        req_to_token,
        req_pool_index=req_pool_index,
        total_length=layout.total_length,
    )
    row = req_to_token[req_pool_index]
    values = torch.tensor(
        layout.as_tuple(),
        dtype=req_to_token.dtype,
        device=req_to_token.device,
    )
    row[: layout.total_length].copy_(values)
    if clear_tail:
        row[layout.total_length :].zero_()


def insert_encoder_slots(
    req_to_token: torch.Tensor,
    *,
    req_pool_index: int,
    encoder_length: int,
    decoder_length: int,
    new_slots: Iterable[int],
) -> MossVLRealtimeKVLayout:
    """Insert new vision KV slots before the existing decoder mapping."""
    layout = read_req_to_token_layout(
        req_to_token,
        req_pool_index=req_pool_index,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
    ).append_encoder(new_slots)
    write_req_to_token_layout(
        req_to_token,
        req_pool_index=req_pool_index,
        layout=layout,
    )
    return layout


def _validate_row_arguments(
    req_to_token: torch.Tensor,
    *,
    req_pool_index: int,
    total_length: int,
) -> None:
    if not isinstance(req_to_token, torch.Tensor) or req_to_token.ndim != 2:
        raise TypeError("req_to_token must be a 2-D torch.Tensor")
    if isinstance(req_pool_index, bool) or not isinstance(req_pool_index, int):
        raise TypeError("req_pool_index must be an integer")
    if req_pool_index < 0 or req_pool_index >= req_to_token.shape[0]:
        raise IndexError("req_pool_index is outside req_to_token")
    if total_length < 0 or total_length > req_to_token.shape[1]:
        raise ValueError("KV layout does not fit in req_to_token row")
