"""StagePayload adapters and token streaming for MOSS-VL realtime."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
from sglang_omni.models.moss_vl_realtime.runtime_state import MossVLRealtimeRuntimeState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

DEFAULT_SYSTEM_PROMPT = (
    # Kept byte-identical to the Transformers reference default
    # (mossvl_streaming_tf_5.12.1 modeling_moss_vl.py), including the em dash.
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant "
    "has happened, and respond when the visual content warrants it."
)


@dataclass
class MossVLRealtimeRequestData(SGLangARRequestData):
    runtime_state: MossVLRealtimeRuntimeState | None = None
    initial_input_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    turn_generated_token_ids: list[int] = field(default_factory=list)
    turn_emitted_text: str = ""
    current_input_seq_no: int | None = None
    current_input_timestamp: float | None = None
    silence_output_seq: int = 0
    ready_emitted: bool = False


def make_moss_vl_realtime_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    vocab_size: int | None = None,
) -> tuple[
    Callable[[StagePayload], MossVLRealtimeRequestData],
    Callable[[MossVLRealtimeRequestData], StagePayload],
]:
    eos_token_id = int(tokenizer.eos_token_id)
    # The NaN/boundary check compares sampled ids against this limit; use the
    # model vocab when available (the LM head may be wider than the tokenizer
    # vocab, e.g. due to padding) rather than rejecting legal tokens.
    vocab_size = max(
        int(vocab_size or 0),
        int(getattr(tokenizer, "vocab_size", 0)),
        len(tokenizer),
    )

    def request_builder(payload: StagePayload) -> MossVLRealtimeRequestData:
        params = payload.request.params or {}
        source = _request_source(payload)
        session_id = str(source.get("session_id") or payload.request_id)
        system_prompt = str(source.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        initial_prompt = str(source.get("initial_prompt") or source.get("prompt") or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]
        prefill = source.get("prefill_messages")
        if prefill is not None:
            if (
                not isinstance(prefill, list)
                or not 1 <= len(prefill) <= 64
                or any(
                    not isinstance(m, dict)
                    or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str)
                    for m in prefill
                )
            ):
                raise ValueError("invalid prefill_messages")
            messages = [{"role": m["role"], "content": m["content"]} for m in prefill]
            # UI transcripts omit the trained opener consumed by the runtime.
            for message in messages:
                if message["role"] == "assistant" and not message["content"].startswith(
                    "<|silence|>"
                ):
                    message["content"] = "<|silence|>" + message["content"]
            if sum(len(m["content"]) for m in messages) > 131072:
                raise ValueError("prefill_messages text exceeds 131072 characters")
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            if messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": ""})
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise ValueError("chat template output is missing input_ids")
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).flatten().tolist()
        request_max_new_tokens = int(params.get("max_new_tokens", max_new_tokens))
        if request_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        max_tokens_per_turn = float(params.get("max_tokens_per_turn", 86400))
        if not math.isfinite(max_tokens_per_turn) or max_tokens_per_turn <= 0:
            raise ValueError("max_tokens_per_turn must be finite and positive")
        temperature = float(params.get("temperature", 0.0) or 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=float(params.get("top_p", 1.0)),
            top_k=int(params.get("top_k", -1)),
            stop_token_ids=[eos_token_id],
            ignore_eos=True,
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )
        state = MossVLRealtimeRuntimeState(
            request_id=payload.request_id,
            session_id=session_id,
            max_tokens_per_turn=max_tokens_per_turn,
            decode_allowance=request_max_new_tokens,
        )
        if params.get("realtime_accounting_v2"):
            from sglang_omni.models.moss_vl_realtime.accounting import (
                RealtimeAccounting,
            )

            state.accounting = RealtimeAccounting(session_id)
        setattr(req, RUNTIME_STATE_ATTR, state)
        # Realtime sessions must never share radix-cached KV: a matched prefix
        # lands at the head of the *decoder* region of the realtime page row
        # (encoder slots occupy the row head), while upstream's release path
        # protects ``cache_protected_len`` slots at the *row* head. Release
        # would then free tree-owned slots, double-free them on the tree's
        # later eviction, and corrupt every subsequent session ("encoder and
        # decoder slots must not overlap"). Skipping inserts from birth keeps
        # the tree empty, so bootstrap matches always miss and every slot in
        # the row is owned by the request itself. The cost is recomputing the
        # ~100-token system prompt per session, which is negligible.
        req.skip_radix_cache_insert = True
        if params.get("benchmark_ignore_eos"):
            # Benchmark mode: keep ignore_eos=True even after the final extend so
            # the request decodes exactly max_new_tokens without an EOS stop.
            req._moss_vl_realtime_keep_ignore_eos = True
        return MossVLRealtimeRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            runtime_state=state,
            initial_input_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            stage_payload=payload,
        )

    def result_adapter(data: MossVLRealtimeRequestData) -> StagePayload:
        payload = data.stage_payload
        text = _decode(tokenizer, data.generated_token_ids)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "session_id": data.runtime_state.session_id,
                "completion_tokens": len(data.generated_token_ids),
                "finish_reason": data.finish_reason,
                "modality": "text",
                "turn_id": data.runtime_state.turn_id,
            },
        )

    return request_builder, result_adapter


def make_moss_vl_realtime_stream_output_builder(
    *,
    tokenizer: Any,
    silence_token_ids: tuple[int, ...],
    context_length: int = 0,
) -> Callable[[str, MossVLRealtimeRequestData, Any], list[OutgoingMessage]]:
    eos_token_id = int(tokenizer.eos_token_id)
    silence_token_ids = tuple(int(token_id) for token_id in silence_token_ids)
    if not silence_token_ids:
        raise ValueError("silence_token_ids must not be empty")

    def build(
        request_id: str,
        data: MossVLRealtimeRequestData,
        req_output: Any,
    ) -> list[OutgoingMessage]:
        messages: list[OutgoingMessage] = []
        accounting = data.runtime_state.accounting
        if accounting is not None:
            if req_output.data is not None:
                accounting.sampled()
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": "realtime.accounting",
                        "modality": "control",
                        "usage": accounting.snapshot(),
                        "watermark": accounting.step,
                        "context_limit": int(
                            data.runtime_state.context_limit or context_length
                        ),
                    },
                    metadata={"modality": "control"},
                )
            )
        if (data.stage_payload.request.params or {}).get("include_usage"):
            state = data.runtime_state
            # Include the sampled pending token; it still needs its next forward.
            decoder_tokens = int(state.decoder_length) + int(
                req_output.data is not None
            )
            token_space_used = (
                int(state.effective_appended_encoder_length) + decoder_tokens
            )
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": "session.usage",
                        "modality": "control",
                        "decoder_tokens": decoder_tokens,
                        "encoder_tokens": int(state.effective_appended_encoder_length),
                        "encoder_kv_tokens": int(state.encoder_length),
                        "token_space_used": token_space_used,
                        "context_limit": int(context_length),
                        "context_remaining": max(
                            0, int(context_length) - token_space_used
                        ),
                    },
                    metadata={"modality": "control"},
                )
            )
        processed_events = getattr(
            data.req,
            "_moss_vl_realtime_processed_events",
            None,
        )
        if processed_events is not None:
            for processed_event in processed_events:
                data.current_input_seq_no = int(processed_event["seq_no"])
                data.current_input_timestamp = float(processed_event["timestamp"])
                if processed_event.get("prompt") is not None:
                    data.turn_generated_token_ids.clear()
                    data.turn_emitted_text = ""
                    messages.append(
                        OutgoingMessage(
                            request_id=request_id,
                            type="stream",
                            data={
                                "event": "response.turn.interrupted",
                                "turn_id": processed_event["interrupted_turn_id"],
                                "next_turn_id": processed_event["turn_id"],
                                "seq_no": processed_event["seq_no"],
                                "modality": "control",
                            },
                            metadata={"modality": "control"},
                        )
                    )
                processed_type = (
                    "input.frame.processed"
                    if processed_event.get("frame_ref") is not None
                    else "input.prompt.processed"
                )
                processed_data = {
                    "event": processed_type,
                    "seq_no": processed_event["seq_no"],
                    "timestamp": processed_event["timestamp"],
                    "final": processed_event["final"],
                    "modality": "control",
                }
                if processed_event.get("prompt") is not None:
                    processed_data.update(
                        {
                            "interrupted_turn_id": processed_event[
                                "interrupted_turn_id"
                            ],
                            "turn_id": processed_event["turn_id"],
                        }
                    )
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=processed_data,
                        metadata={"modality": "control"},
                    )
                )
            del data.req._moss_vl_realtime_processed_events
        token_data = req_output.data
        if token_data is None:
            return messages
        token_id = int(token_data)
        if not data.ready_emitted:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": "session.ready",
                        "session_id": data.runtime_state.session_id,
                        "turn_id": data.runtime_state.turn_id,
                        "modality": "control",
                    },
                    metadata={"modality": "control"},
                )
            )
            data.ready_emitted = True
        if token_id != eos_token_id:
            data.generated_token_ids.append(token_id)
            data.turn_generated_token_ids.append(token_id)
        is_silence = _ends_with_token_ids(
            data.turn_generated_token_ids, silence_token_ids
        )
        messages.extend(
            emit_text(
                request_id,
                data,
                token_id=token_id,
                final=token_id == eos_token_id or is_silence,
            )
        )
        if is_silence:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": "response.turn.silence",
                        "turn_id": data.runtime_state.turn_id,
                        "seq_no": data.current_input_seq_no,
                        "timestamp": data.current_input_timestamp,
                        "silence_seq": data.silence_output_seq,
                        "modality": "control",
                    },
                    metadata={
                        "modality": "control",
                        "token_id": token_id,
                        "turn_id": data.runtime_state.turn_id,
                    },
                )
            )
            data.silence_output_seq += 1
        return messages

    def emit_text(
        request_id: str,
        data: MossVLRealtimeRequestData,
        *,
        token_id: int | None = None,
        final: bool = False,
    ) -> list[OutgoingMessage]:
        full_text = _decode(tokenizer, data.generated_token_ids)
        data.emitted_text = full_text
        turn_text = _decode(tokenizer, data.turn_generated_token_ids)
        # Byte-level tokens may end mid-codepoint. Keep the emitted cursor
        # unchanged until the bytes complete, or flush at a terminal boundary.
        if not final and turn_text.endswith("\ufffd"):
            return []
        delta = _text_delta(data.turn_emitted_text, turn_text)
        data.turn_emitted_text = turn_text
        payload = data.stage_payload
        if not delta or not (payload.request.params or {}).get("stream", False):
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={
                    "text": delta,
                    "modality": "text",
                    "turn_id": data.runtime_state.turn_id,
                },
                metadata={
                    "modality": "text",
                    "token_id": token_id,
                    "turn_id": data.runtime_state.turn_id,
                },
            )
        ]

    def flush(
        request_id: str, data: MossVLRealtimeRequestData
    ) -> list[OutgoingMessage]:
        return emit_text(request_id, data, final=True)

    build.flush = flush
    return build


def _request_source(payload: StagePayload) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if isinstance(payload.request.inputs, dict):
        source.update(payload.request.inputs)
    elif isinstance(payload.request.inputs, str):
        source["initial_prompt"] = payload.request.inputs
    if isinstance(payload.data, dict):
        source.update(payload.data)
    return source


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _text_delta(previous: str, current: str) -> str:
    if current.startswith(previous):
        return current[len(previous) :]
    common = 0
    limit = min(len(previous), len(current))
    while common < limit and previous[common] == current[common]:
        common += 1
    return current[common:]


def _ends_with_token_ids(values: list[int], suffix: tuple[int, ...]) -> bool:
    if len(values) < len(suffix):
        return False
    return tuple(values[-len(suffix) :]) == suffix
