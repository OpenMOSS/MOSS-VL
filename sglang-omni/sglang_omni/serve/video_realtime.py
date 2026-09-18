"""Stateful binary-frame WebSocket API for MOSS-VL realtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from fastapi import FastAPI, WebSocket
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.models.moss_vl_realtime.frame_store import (
    MAX_FRAME_BYTES,
    SharedMemoryFrameStore,
)
from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent
from sglang_omni.serve._cleanup import await_cleanup

logger = logging.getLogger(__name__)
CONFIGURE_TIMEOUT_S = 180.0
MAX_INPUT_QUEUE_CAPACITY = 256


class VideoRealtimeSubmissionError(RuntimeError):
    """An input could not be submitted to the active model request."""


async def warmup_video_realtime(
    client: Client,
    *,
    model_name: str,
    timeout_s: float = 180.0,
    dp_rank: int | None = None,
) -> None:
    """Exercise the dynamic vision path before accepting user traffic.

    ``dp_rank`` pins the warmup to one data-parallel replica (per-replica
    warmup); ``None`` lets the coordinator choose, as before.
    """
    if timeout_s <= 0:
        raise ValueError("video realtime warmup timeout must be positive")

    request_id = f"video_warmup_req_{uuid.uuid4().hex}"
    session_id = f"video_warmup_sess_{uuid.uuid4().hex}"
    frame_store = SharedMemoryFrameStore()
    frame_ref: str | None = None
    processed = False

    request = GenerateRequest(
        model=model_name,
        prompt={
            "initial_prompt": "Warm up the realtime vision path.",
            "session_id": session_id,
        },
        sampling=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=8,
        ),
        stream=True,
        max_tokens=8,
        output_modalities=["text"],
        extra_params={"realtime_warmup": True},
    )

    async def _run() -> None:
        nonlocal frame_ref, processed
        submitted = False
        generate_kwargs: dict[str, Any] = {}
        if dp_rank is not None:
            generate_kwargs["dp_rank"] = dp_rank
        async for chunk in client.generate(
            request, request_id=request_id, **generate_kwargs
        ):
            if chunk.control_event == "session.ready" and not submitted:
                output = BytesIO()
                Image.new("RGB", (640, 352), color=(127, 127, 127)).save(
                    output,
                    format="PNG",
                )
                frame_ref = frame_store.put(request_id, output.getvalue())
                event = FramePromptEvent(
                    request_id=request_id,
                    session_id=session_id,
                    seq_no=0,
                    timestamp=0.0,
                    frame_ref=frame_ref,
                    final=True,
                )
                await client.update_request(request_id, event.to_dict())
                submitted = True
                continue
            if chunk.control_event == "input.frame.processed":
                processed = True
                if frame_ref is not None:
                    frame_store.forget(request_id, frame_ref)

        if not submitted:
            raise RuntimeError("video realtime warmup never reached session.ready")
        if not processed:
            raise RuntimeError("video realtime warmup frame was not processed")

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    finally:
        try:
            await client.abort(request_id)
        except Exception:
            logger.debug(
                "Video realtime warmup request was already closed", exc_info=True
            )
        frame_store.cleanup(request_id)


class VideoPrefillMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=131072)


class VideoSessionConfigure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session.configure"]
    # Mirrors the Transformers realtime reference: the default initial user
    # prompt is empty; production callers pass the SFT prompt explicitly.
    prompt: str = ""
    system_prompt: str | None = None
    prefill_messages: list[VideoPrefillMessage] | None = Field(
        default=None, min_length=1, max_length=64
    )

    @field_validator("prefill_messages")
    @classmethod
    def bound_prefill_text(cls, messages):
        if messages is not None and sum(len(m.content) for m in messages) > 131072:
            raise ValueError("prefill_messages text exceeds 131072 characters")
        return messages

    max_new_tokens: int = Field(default=4096, gt=0)
    max_tokens_per_turn: float = Field(
        default=86400.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Maximum generation rate in tokens per second.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    input_queue_capacity: int = Field(default=4, ge=1, le=MAX_INPUT_QUEUE_CAPACITY)
    benchmark_ignore_eos: bool = False
    include_usage: bool = False


class VideoFrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input.frame"]
    seq_no: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    prompt: str | None = None
    final: bool = False
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]

    @field_validator("prompt")
    @classmethod
    def normalize_optional_prompt(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class VideoPromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input.prompt"]
    seq_no: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    final: bool = False

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


@dataclass
class _PendingFrame:
    metadata: VideoFrameMetadata


class VideoRealtimeSession:
    """Own one persistent generation request and its ordered binary frames."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        frame_store: SharedMemoryFrameStore,
        allow_benchmark_mode: bool = False,
        parked_request_timeout_s: float = 300.0,
        configure_timeout_s: float = CONFIGURE_TIMEOUT_S,
        dp_rank: int | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.frame_store = frame_store
        self.allow_benchmark_mode = bool(allow_benchmark_mode)
        self.parked_request_timeout_s = float(parked_request_timeout_s)
        self.configure_timeout_s = float(configure_timeout_s)
        # Data-parallel replica pinned by the session manager at open time.
        # ``None`` for single-replica serving (session routing unchanged).
        self.dp_rank = dp_rank
        if not math.isfinite(self.configure_timeout_s) or self.configure_timeout_s <= 0:
            raise ValueError("configure_timeout_s must be finite and positive")
        self._configure_deadline = time.monotonic() + self.configure_timeout_s
        self.session_id = f"video_sess_{uuid.uuid4().hex}"
        self.request_id = f"video_req_{uuid.uuid4().hex}"
        self.pending_frame: _PendingFrame | None = None
        self.input_queue_capacity = 4
        self.input_capacity_changed = asyncio.Condition()
        self.outstanding_seq_nos: set[int] = set()
        self.accepted_by_seq: dict[int, asyncio.Event] = {}
        self.frame_refs_by_seq: dict[int, str] = {}
        self.last_timestamp = 0.0
        self.next_seq_no = 0
        self.response_task: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()
        self.configured = False
        self.ready = False
        self.current_turn_id = 0
        self.final_received = False
        self.request_finished = False
        self.abort_sent = False
        self.closed = False
        self._teardown_task: asyncio.Task | None = None
        self.cleanup_timeout_s = 5.0

    async def run(self) -> None:
        inputs: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(
            maxsize=2 * MAX_INPUT_QUEUE_CAPACITY + 2
        )
        tasks: list[asyncio.Task] = []
        try:
            await self.send(
                {
                    "type": "session.created",
                    "session_id": self.session_id,
                    "request_id": self.request_id,
                    "model": self.model_name,
                    "turn_id": self.current_turn_id,
                    "capabilities": ["session.usage"],
                    "configure_timeout_s": self.configure_timeout_s,
                }
            )
            tasks = [
                asyncio.create_task(self._receive_inputs(inputs)),
                asyncio.create_task(self._process_inputs(inputs)),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self.closed = True
            await await_cleanup(asyncio.create_task(self._finish_run(tasks)))

    async def _finish_run(self, tasks) -> None:
        try:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    self.cleanup_timeout_s,
                )
        finally:
            await self.teardown()

    async def _receive_inputs(self, inputs: asyncio.Queue) -> None:
        while not self.closed:
            timeout = (
                None
                if self.configured
                else max(0.0, self._configure_deadline - time.monotonic())
            )
            try:
                message = await asyncio.wait_for(
                    self.websocket.receive(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Configuration may have completed while receive was pending.
                if self.configured:
                    continue
                self.closed = True
                await self.send_error_safely(
                    "session.configure was not received before the deadline",
                    code="configuration_timeout",
                )
                return
            if message["type"] == "websocket.disconnect":
                self.closed = True
                return
            if message["type"] != "websocket.receive":
                continue
            try:
                if message.get("bytes") is not None:
                    kind, payload = "bytes", message["bytes"]
                    if len(payload) > MAX_FRAME_BYTES:
                        self.closed = True
                        await self.send_error_safely(
                            "frame payload exceeds max_frame_bytes",
                            code="frame_too_large",
                        )
                        return
                else:
                    raw = message.get("text")
                    if raw is None:
                        raise ValueError("empty WebSocket message")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise TypeError("top-level message must be a JSON object")
                    if payload.get("type") == "session.abort":
                        # Teardown cancels a blocked input worker before aborting
                        # the model request; control never waits for input credit.
                        self.closed = True
                        return
                    kind = "json"
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                await self.send_error_safely(str(exc), code="invalid_request")
                continue
            # One metadata/binary pair per configured credit, plus one pair
            # waiting for credit. Never block the control receiver on queue.put.
            if inputs.qsize() >= 2 * self.input_queue_capacity + 2:
                self.closed = True
                await self.send_error_safely(
                    "too many queued input messages; wait for input acknowledgements",
                    code="input_queue_full",
                )
                return
            inputs.put_nowait((kind, payload))

    async def _process_inputs(self, inputs: asyncio.Queue) -> None:
        while not self.closed:
            kind, payload = await inputs.get()
            try:
                if self.closed:
                    return
                if kind == "bytes":
                    await self.handle_frame_bytes(payload)
                else:
                    await self.handle_json(payload)
            except VideoRealtimeSubmissionError as exc:
                logger.exception(
                    "Realtime input submission failed for request %s", self.request_id
                )
                await self.send_error_safely(str(exc), code="input_submission_failed")
                self.closed = True
            except (TypeError, ValidationError, ValueError, RuntimeError) as exc:
                await self.send_error_safely(str(exc), code="invalid_request")
            finally:
                inputs.task_done()

    async def handle_json(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "session.configure":
            await self.configure(VideoSessionConfigure.model_validate(payload))
        elif event_type == "input.frame":
            await self.prepare_frame(VideoFrameMetadata.model_validate(payload))
        elif event_type == "input.prompt":
            await self.handle_prompt(VideoPromptInput.model_validate(payload))
        elif event_type == "session.abort":
            self.closed = True
            await self.abort_request()
        else:
            raise ValueError(f"unsupported event type: {event_type!r}")

    async def configure(self, config: VideoSessionConfigure) -> None:
        if self.configured:
            raise ValueError("session is already configured")
        if config.benchmark_ignore_eos and not self.allow_benchmark_mode:
            raise ValueError("benchmark_ignore_eos requires server benchmark mode")
        self.input_queue_capacity = config.input_queue_capacity
        self.configured = True
        await self.send(
            {
                "type": "session.configured",
                "session_id": self.session_id,
                "request_id": self.request_id,
                "input_queue_capacity": self.input_queue_capacity,
                "max_tokens_per_turn": config.max_tokens_per_turn,
                "max_frame_bytes": MAX_FRAME_BYTES,
                "parked_request_timeout_s": self.parked_request_timeout_s,
            }
        )
        extra_params: dict[str, Any] = {
            "max_tokens_per_turn": config.max_tokens_per_turn,
        }
        if config.include_usage:
            extra_params["include_usage"] = True
        if config.benchmark_ignore_eos:
            extra_params["benchmark_ignore_eos"] = True
        request = GenerateRequest(
            model=self.model_name,
            prompt={
                "initial_prompt": config.prompt,
                "system_prompt": config.system_prompt,
                "session_id": self.session_id,
                **(
                    {
                        "prefill_messages": [
                            m.model_dump() for m in config.prefill_messages
                        ]
                    }
                    if config.prefill_messages is not None
                    else {}
                ),
            },
            sampling=SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
            ),
            stream=True,
            max_tokens=config.max_new_tokens,
            output_modalities=["text"],
            extra_params=extra_params,
        )
        self.response_task = asyncio.create_task(self.stream_response(request))

    async def prepare_frame(self, metadata: VideoFrameMetadata) -> None:
        if not self.configured:
            raise ValueError("configure the session before sending frames")
        if not self.ready:
            raise ValueError("wait for session.ready before sending frames")
        if self.final_received:
            raise ValueError("session already received its final frame")
        if self.pending_frame is not None:
            raise ValueError("previous frame metadata is still awaiting binary data")
        self._validate_event_order(metadata.seq_no, metadata.timestamp)
        await self._reserve_input(metadata.seq_no)
        self.pending_frame = _PendingFrame(metadata=metadata)
        await self.send(
            {
                "type": "input.frame.ready",
                "seq_no": metadata.seq_no,
            }
        )

    async def handle_prompt(self, metadata: VideoPromptInput) -> None:
        if not self.configured:
            raise ValueError("configure the session before sending prompts")
        if not self.ready:
            raise ValueError("wait for session.ready before sending prompts")
        if self.final_received:
            raise ValueError("session already received its final event")
        if self.pending_frame is not None:
            raise ValueError("previous frame metadata is still awaiting binary data")
        self._validate_event_order(metadata.seq_no, self.last_timestamp)
        await self._reserve_input(metadata.seq_no)
        try:
            event = FramePromptEvent(
                request_id=self.request_id,
                session_id=self.session_id,
                seq_no=metadata.seq_no,
                timestamp=self.last_timestamp,
                frame_ref=None,
                prompt=metadata.prompt,
                final=metadata.final,
            )
        except Exception:
            await self._release_input(metadata.seq_no)
            raise
        try:
            await self.client.update_request(self.request_id, event.to_dict())
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            raise VideoRealtimeSubmissionError(
                f"failed to submit prompt event {metadata.seq_no}: {exc}"
            ) from exc
        self.next_seq_no += 1
        self.final_received = metadata.final
        await self._send_accepted(
            metadata.seq_no,
            {
                "type": "input.prompt.accepted",
                "seq_no": metadata.seq_no,
                "final": metadata.final,
                "pending_events": len(self.outstanding_seq_nos),
                "interrupts_current_turn": True,
            },
        )

    async def handle_frame_bytes(self, payload: bytes) -> None:
        pending = self.pending_frame
        if pending is None:
            raise ValueError("binary frame must follow input.frame metadata")
        self.pending_frame = None
        metadata = pending.metadata
        frame_ref: str | None = None
        try:
            frame_ref = self.frame_store.put(self.request_id, payload)
        except ValueError:
            await self._release_input(metadata.seq_no)
            raise
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            raise VideoRealtimeSubmissionError(
                f"failed to store frame event {metadata.seq_no}: {exc}"
            ) from exc
        try:
            event = FramePromptEvent(
                request_id=self.request_id,
                session_id=self.session_id,
                seq_no=metadata.seq_no,
                timestamp=metadata.timestamp,
                frame_ref=frame_ref,
                prompt=metadata.prompt,
                final=metadata.final,
                fingerprint=hashlib.sha256(payload).hexdigest(),
            )
        except Exception:
            await self._release_input(metadata.seq_no)
            self.frame_store.discard(self.request_id, frame_ref)
            raise
        assert frame_ref is not None
        self.frame_refs_by_seq[metadata.seq_no] = frame_ref
        try:
            await self.client.update_request(self.request_id, event.to_dict())
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            self.frame_refs_by_seq.pop(metadata.seq_no, None)
            self.frame_store.discard(self.request_id, frame_ref)
            raise VideoRealtimeSubmissionError(
                f"failed to submit frame event {metadata.seq_no}: {exc}"
            ) from exc
        self.next_seq_no += 1
        self.final_received = metadata.final
        self.last_timestamp = float(metadata.timestamp)
        await self._send_accepted(
            metadata.seq_no,
            {
                "type": "input.frame.accepted",
                "seq_no": metadata.seq_no,
                "timestamp": metadata.timestamp,
                "final": metadata.final,
                "pending_events": len(self.outstanding_seq_nos),
                "interrupts_current_turn": metadata.prompt is not None,
            },
        )

    async def stream_response(self, request: GenerateRequest) -> None:
        streamed_text = ""
        try:
            generate_kwargs: dict[str, Any] = {}
            if self.dp_rank is not None:
                generate_kwargs["dp_rank"] = self.dp_rank
            async for chunk in self.client.generate(
                request, request_id=self.request_id, **generate_kwargs
            ):
                if chunk.control_event == "session.usage":
                    await self.send(
                        {"type": "session.usage", **dict(chunk.control_data or {})}
                    )
                    continue
                if chunk.control_event == "session.ready":
                    self.ready = True
                    await self.send(
                        {
                            "type": "session.ready",
                            "session_id": self.session_id,
                            "request_id": self.request_id,
                            "turn_id": int(
                                (chunk.control_data or {}).get("turn_id", 0)
                            ),
                        }
                    )
                    continue
                if chunk.control_event == "response.turn.interrupted":
                    control_data = dict(chunk.control_data or {})
                    seq_no = int(control_data["seq_no"])
                    accepted = self.accepted_by_seq.get(seq_no)
                    if accepted is not None:
                        await accepted.wait()
                    await self.send(
                        {
                            "type": "response.turn.interrupted",
                            "turn_id": int(control_data["turn_id"]),
                            "next_turn_id": int(control_data["next_turn_id"]),
                            "seq_no": seq_no,
                        }
                    )
                    continue
                if chunk.control_event == "response.turn.silence":
                    control_data = dict(chunk.control_data or {})
                    await self.send(
                        {
                            "type": "response.turn.silence",
                            "turn_id": int(control_data["turn_id"]),
                            "seq_no": control_data.get("seq_no"),
                            "timestamp": control_data.get("timestamp"),
                            "silence_seq": int(control_data["silence_seq"]),
                        }
                    )
                    continue
                if chunk.control_event in (
                    "input.frame.processed",
                    "input.prompt.processed",
                ):
                    control_data = dict(chunk.control_data or {})
                    seq_no = int(control_data["seq_no"])
                    accepted = self.accepted_by_seq.get(seq_no)
                    if accepted is not None:
                        await accepted.wait()
                    frame_ref = self.frame_refs_by_seq.pop(seq_no, None)
                    if frame_ref is not None:
                        self.frame_store.forget(self.request_id, frame_ref)
                    await self._release_input(seq_no)
                    processed_payload = {
                        "type": chunk.control_event,
                        "seq_no": seq_no,
                        "timestamp": control_data["timestamp"],
                        "final": control_data["final"],
                        "pending_events": len(self.outstanding_seq_nos),
                    }
                    if control_data.get("turn_id") is not None:
                        self.current_turn_id = int(control_data["turn_id"])
                        processed_payload.update(
                            {
                                "interrupted_turn_id": int(
                                    control_data["interrupted_turn_id"]
                                ),
                                "turn_id": self.current_turn_id,
                            }
                        )
                    await self.send(processed_payload)
                    continue
                # Non-terminal chunks are per-turn deltas and pass through
                # verbatim. The terminal chunk carries the full session text
                # aggregated by the result adapter, so strip the streamed
                # session prefix off it.
                text = chunk.text
                if text and chunk.finish_reason is None:
                    streamed_text += text
                elif text and streamed_text and text.startswith(streamed_text):
                    text = text[len(streamed_text) :]
                if text:
                    await self.send(
                        {
                            "type": "response.text.delta",
                            "delta": text,
                            "turn_id": (
                                self.current_turn_id
                                if chunk.turn_id is None
                                else int(chunk.turn_id)
                            ),
                        }
                    )
                if chunk.finish_reason is not None:
                    await self.send(
                        {
                            "type": "response.done",
                            "finish_reason": chunk.finish_reason,
                            "turn_id": self.current_turn_id,
                        }
                    )
            self.request_finished = True
            self.frame_store.cleanup(self.request_id)
            await self.send({"type": "session.done"})
            self.closed = True
            await self._close_input_queue()
            await self.close_websocket()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.abort_sent:
                # Teardown owns the terminal event for an explicit abort.
                return
            logger.exception("Realtime response failed for request %s", self.request_id)
            await self.send_error_safely(
                str(exc),
                code="response_failed",
            )
            self.closed = True
            await self._close_input_queue()
            await self.close_websocket()

    async def teardown(self) -> None:
        if self._teardown_task is None:
            self._teardown_task = asyncio.create_task(self._teardown())
        await await_cleanup(self._teardown_task)

    async def _teardown(self) -> None:
        self.closed = True
        try:
            if not self.request_finished and self.configured:
                try:
                    await asyncio.wait_for(self.abort_request(), self.cleanup_timeout_s)
                except Exception:
                    logger.exception(
                        "Failed to abort realtime request %s during teardown",
                        self.request_id,
                    )
        finally:
            await self._release_local_resources()
        if not self.request_finished:
            # Give clients a terminal event instead of a bare connection close.
            try:
                await asyncio.wait_for(
                    self.send(
                        {
                            "type": "session.done",
                            "session_id": self.session_id,
                            "aborted": True,
                        }
                    ),
                    self.cleanup_timeout_s,
                )
            except (OSError, RuntimeError, WebSocketDisconnect, TimeoutError):
                logger.debug(
                    "Realtime session %s teardown: WebSocket already closed",
                    self.session_id,
                    exc_info=True,
                )
        try:
            await asyncio.wait_for(self.close_websocket(), self.cleanup_timeout_s)
        except TimeoutError:
            logger.warning("Realtime WebSocket close timed out for %s", self.request_id)

    async def _release_local_resources(self) -> None:
        task = self.response_task
        try:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True), self.cleanup_timeout_s
                )
        except TimeoutError:
            logger.warning(
                "Realtime response cleanup timed out for %s", self.request_id
            )
        finally:
            try:
                self.frame_store.cleanup(self.request_id)
            finally:
                self.frame_refs_by_seq.clear()
                self.pending_frame = None
                await self._close_input_queue()

    async def abort_request(self) -> None:
        if self.abort_sent:
            return
        self.abort_sent = True
        await self.client.abort(self.request_id)

    async def close_websocket(self) -> None:
        try:
            if (
                self.websocket.application_state is WebSocketState.CONNECTED
                and self.websocket.client_state is WebSocketState.CONNECTED
            ):
                await self.websocket.close()
        except (OSError, RuntimeError, WebSocketDisconnect):
            logger.debug(
                "Realtime WebSocket was already closed for request %s",
                self.request_id,
                exc_info=True,
            )

    async def send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def _reserve_input(self, seq_no: int) -> None:
        async with self.input_capacity_changed:
            await self.input_capacity_changed.wait_for(
                lambda: self.closed
                or len(self.outstanding_seq_nos) < self.input_queue_capacity
            )
            if self.closed:
                raise RuntimeError("realtime session is closed")
            if seq_no in self.outstanding_seq_nos:
                raise ValueError(f"event {seq_no} is already pending")
            self.outstanding_seq_nos.add(seq_no)
            self.accepted_by_seq[seq_no] = asyncio.Event()

    def _validate_event_order(self, seq_no: int, timestamp: float) -> None:
        """Reject out-of-order events at the protocol edge.

        These violations used to reach the scheduler, which aborted the whole
        request; the WebSocket edge now rejects them with a per-event error and
        keeps the session alive.
        """
        if seq_no != self.next_seq_no:
            raise ValueError(f"expected seq_no {self.next_seq_no}, received {seq_no}")
        if timestamp < self.last_timestamp:
            raise ValueError(
                f"timestamp moved backwards: {timestamp} < {self.last_timestamp}"
            )

    async def _release_input(self, seq_no: int) -> None:
        async with self.input_capacity_changed:
            self.outstanding_seq_nos.discard(seq_no)
            accepted = self.accepted_by_seq.pop(seq_no, None)
            if accepted is not None:
                accepted.set()
            self.input_capacity_changed.notify_all()

    async def _close_input_queue(self) -> None:
        async with self.input_capacity_changed:
            for accepted in self.accepted_by_seq.values():
                accepted.set()
            self.accepted_by_seq.clear()
            self.outstanding_seq_nos.clear()
            self.input_capacity_changed.notify_all()

    async def _send_accepted(self, seq_no: int, payload: dict[str, Any]) -> None:
        await self.send(payload)
        accepted = self.accepted_by_seq.get(seq_no)
        if accepted is not None:
            accepted.set()

    async def send_error(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"type": "error", "message": message}
        if code is not None:
            payload["code"] = code
        if retryable:
            payload["retryable"] = True
        await self.send(payload)

    async def send_error_safely(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
    ) -> bool:
        """Best-effort error delivery that never replaces the original failure."""
        try:
            await self.send_error(
                message,
                code=code,
                retryable=retryable,
            )
        except (OSError, RuntimeError, WebSocketDisconnect):
            logger.debug(
                "Could not deliver realtime error %s for request %s",
                code,
                self.request_id,
                exc_info=True,
            )
            return False
        return True


class VideoRealtimeSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        allow_benchmark_mode: bool = False,
        parked_request_timeout_s: float = 300.0,
        max_sessions: int = 1,
        replica_count: int = 1,
        configure_timeout_s: float = CONFIGURE_TIMEOUT_S,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.allow_benchmark_mode = bool(allow_benchmark_mode)
        self.parked_request_timeout_s = float(parked_request_timeout_s)
        self.max_sessions = int(max_sessions)
        self.configure_timeout_s = float(configure_timeout_s)
        if not math.isfinite(self.configure_timeout_s) or self.configure_timeout_s <= 0:
            raise ValueError("configure_timeout_s must be finite and positive")
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        # Native DP: the entry stage is replicated ``replica_count`` times and
        # replicas are homogeneous, so the aggregate cap splits evenly. Session
        # accounting lives here (open/close), not in the scheduler, because a
        # parked session still holds its slot on the owning replica.
        self.replica_count = int(replica_count)
        if self.replica_count < 1:
            raise ValueError("replica_count must be at least 1")
        if self.max_sessions % self.replica_count != 0:
            raise ValueError(
                "max_sessions must be a multiple of replica_count "
                f"({self.max_sessions} sessions over {self.replica_count} replicas)"
            )
        self.frame_store = SharedMemoryFrameStore()
        self.sessions: dict[str, VideoRealtimeSession] = {}

    def _assign_replica(self) -> int | None:
        """Least-loaded replica with a free slot; ``None`` when dp == 1.

        Session counts (including not-yet-configured opens) drive the choice;
        the session then pins its persistent request to that replica so the
        two counts can never diverge.
        """
        if self.replica_count == 1:
            return None
        loads = [0] * self.replica_count
        for session in self.sessions.values():
            if session.dp_rank is not None:
                loads[session.dp_rank] += 1
        per_replica_capacity = self.max_sessions // self.replica_count
        free = [
            rank
            for rank in range(self.replica_count)
            if loads[rank] < per_replica_capacity
        ]
        if not free:
            raise RuntimeError(
                "video realtime service has no free session slot "
                f"(capacity {self.max_sessions})"
            )
        return min(free, key=lambda rank: loads[rank])

    def open(
        self, websocket: WebSocket, *, session_factory=None
    ) -> VideoRealtimeSession:
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(
                "video realtime service has no free session slot "
                f"(capacity {self.max_sessions})"
            )
        dp_rank = self._assign_replica()
        session = (session_factory or VideoRealtimeSession)(
            websocket,
            client=self.client,
            model_name=self.model_name,
            frame_store=self.frame_store,
            allow_benchmark_mode=self.allow_benchmark_mode,
            parked_request_timeout_s=self.parked_request_timeout_s,
            configure_timeout_s=self.configure_timeout_s,
            dp_rank=dp_rank,
        )
        self.sessions[session.session_id] = session
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            try:
                await session.teardown()
            finally:
                self.sessions.pop(session_id, None)


def register_video_realtime(
    app: FastAPI,
    *,
    allow_benchmark_mode: bool = False,
    parked_request_timeout_s: float = 300.0,
    max_sessions: int = 1,
    replica_count: int = 1,
    configure_timeout_s: float = CONFIGURE_TIMEOUT_S,
) -> None:
    manager = VideoRealtimeSessionManager(
        client=app.state.client,
        model_name=app.state.model_name,
        allow_benchmark_mode=allow_benchmark_mode,
        parked_request_timeout_s=parked_request_timeout_s,
        max_sessions=max_sessions,
        replica_count=replica_count,
        configure_timeout_s=configure_timeout_s,
    )
    app.state.video_realtime_manager = manager

    @app.get("/v1/video/realtime/capabilities")
    async def video_realtime_capabilities():
        return {"prefill_messages": True}

    @app.websocket("/v1/video/realtime")
    async def video_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            session = manager.open(websocket)
        except RuntimeError as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "session_capacity_exceeded",
                    "message": str(exc),
                }
            )
            await websocket.close(code=1013)
            return
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)
