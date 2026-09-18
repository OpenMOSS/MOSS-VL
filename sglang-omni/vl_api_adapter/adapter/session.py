"""Per-response settlement over the native persistent-request transport."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from contextlib import suppress
from io import BytesIO

from PIL import Image
from pydantic import ValidationError

from sglang_omni.models.moss_vl_realtime.accounting import (
    ACCOUNTING_EVENT,
    ACCOUNTING_PARAM,
    FINALIZE_ACTION,
)
from sglang_omni.serve._cleanup import await_cleanup
from sglang_omni.serve.video_realtime import MAX_FRAME_BYTES, VideoRealtimeSession

from .schemas import CAPABILITIES, Configure, Frame, Prompt
from .usage import UsageLedger, empty_usage

logger = logging.getLogger(__name__)


class AccountingClient:
    """Consume private snapshots before the native session sees a wire event."""

    def __init__(self, client, session):
        self.client, self.session = client, session

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def generate(self, request, **kwargs):
        request.extra_params[ACCOUNTING_PARAM] = True
        async for chunk in self.client.generate(request, **kwargs):
            if chunk.control_event == ACCOUNTING_EVENT:
                self.session.record_accounting(dict(chunk.control_data or {}))
                continue
            yield chunk


class V2Session(VideoRealtimeSession):
    def __init__(
        self,
        *args,
        context_limit,
        accounting_stage,
        binary_timeout_s=10.0,
        finalize_timeout_s=20.0,
        model_version=None,
        **kwargs,
    ):
        if type(context_limit) is not int or context_limit <= 0:
            raise ValueError("context_limit must be a positive integer")
        for value in (binary_timeout_s, finalize_timeout_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("v2 deadlines must be finite and positive")
        super().__init__(*args, **kwargs)
        self.raw_client = self.client
        self.client = AccountingClient(self.raw_client, self)
        self.context_limit = int(context_limit)
        self.accounting_stage = accounting_stage
        self.binary_timeout_s = binary_timeout_s
        self.finalize_timeout_s = finalize_timeout_s
        self.model_version = model_version or None
        self.ledger = UsageLedger()
        self.wants_usage = False
        self._last_usage_at = None
        self._response_id = None
        self._response_seq = 0
        self._response_turn = None
        self._finish_reason = "stop"
        self._fatal_error = None
        self._request_started = False
        self._terminal_sent = False
        self._finalizing = False
        self._finalized = False
        self._finish_lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._binary_timer = None
        self._timer_tasks = set()
        self._finish_task = None
        self._finish_waiters = set()

    def record_accounting(self, data):
        if self._finalized:
            return
        if int(data.get("context_limit", 0)) != self.context_limit:
            raise ValueError("backend accounting context limit changed")
        self.ledger.record(data.get("usage"), data.get("watermark"))

    async def configure(self, config):
        if self.configured:
            raise ValueError("session is already configured")
        self.wants_usage = config.include_usage
        await super().configure(config.model_copy(update={"include_usage": True}))

    async def stream_response(self, request):
        self._request_started = True
        await super().stream_response(request)

    async def _wire(self, payload):
        if payload.get("type") == "error" and payload.get("seq_no") is None:
            payload = {key: value for key, value in payload.items() if key != "seq_no"}
        async with asyncio.timeout(5):
            await VideoRealtimeSession.send(self, payload)

    async def _settle_response(self, reason):
        if self._response_id is None:
            return
        payload = {
            "type": "response.done",
            "response_id": self._response_id,
            "response_seq": self._response_seq,
            "turn_id": self._response_turn,
            "finish_reason": "stop" if reason == "silence" else reason,
            "boundary": "silence" if reason == "silence" else "request_end",
            "usage": self.ledger.settle(),
        }
        self._response_id = None
        self._response_turn = None
        await self._wire(payload)

    async def send(self, payload):
        kind = payload.get("type")
        if kind == "session.done":
            await self._finish("completed" if self.request_finished else "aborted")
            return
        async with self._event_lock:
            if self._finalizing or self._finalized:
                return
            payload = dict(payload)
            if kind == "session.created":
                payload.update(
                    capabilities=CAPABILITIES,
                    protocol_version="vl-api-v2",
                    model_version=self.model_version,
                    model_version_source=(
                        "deployment" if self.model_version else "unknown"
                    ),
                )
            elif kind == "session.configured":
                payload["context_limit"] = self.context_limit
            elif kind == "session.usage":
                now = time.monotonic()
                if not self.wants_usage or (
                    self._last_usage_at is not None and now - self._last_usage_at < 1
                ):
                    return
                self._last_usage_at = now
                payload["context_limit"] = self.context_limit
                payload["context_remaining"] = max(
                    0, self.context_limit - int(payload["token_space_used"])
                )
            elif kind == "response.text.delta":
                text = payload.get("delta", "")
                if not text:
                    return
                if text.strip() and self.ledger.watermark < 0:
                    raise RuntimeError(
                        "visible output arrived without authoritative accounting"
                    )
                if (
                    self._response_id is not None
                    and payload["turn_id"] != self._response_turn
                ):
                    raise RuntimeError("turn changed without an interruption boundary")
                if self._response_id is None and text.strip():
                    self._response_seq += 1
                    self._response_id = "response_" + uuid.uuid4().hex
                    self._response_turn = payload["turn_id"]
                payload.update(
                    response_id=self._response_id,
                    response_seq=self._response_seq if self._response_id else None,
                )
            elif kind == "response.turn.silence":
                await self._wire(payload)
                await self._settle_response("silence")
                return
            elif kind == "response.turn.interrupted":
                # Interrupted segments are not settled; the next watermark
                # or terminal total includes their consumption.
                payload["response_id"] = self._response_id
                self._response_id = None
                self._response_turn = None
            elif kind == "response.done":
                self._finish_reason = payload.get("finish_reason", "stop")
                return  # Native done is request-terminal, never a second segment.
            elif kind == "input.frame.ready":
                self._start_binary_timer(payload["seq_no"])
            elif kind == "error":
                if payload.get("code") != "invalid_request":
                    self._fatal_error = self._fatal_error or payload
                    return  # Classify from backend finalization before delivery.
            await self._wire(payload)

    def _start_binary_timer(self, seq_no):
        self._cancel_binary_timer()

        async def expire():
            await asyncio.sleep(self.binary_timeout_s)
            self._fatal_error = {
                "type": "error",
                "code": "session_timeout",
                "seq_no": seq_no,
                "message": "frame binary receive deadline exceeded",
            }
            self.closed = True
            await self._finish("error")

        self._binary_timer = asyncio.create_task(expire())
        self._timer_tasks.add(self._binary_timer)
        self._binary_timer.add_done_callback(self._timer_finished)

    def _timer_finished(self, task):
        self._timer_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("v2 binary deadline cleanup failed: %s", task.exception())

    def _cancel_binary_timer(self):
        task, self._binary_timer = self._binary_timer, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    @staticmethod
    def _ref(kind, payload, pending):
        if kind == "bytes" and pending is not None:
            return pending.metadata.seq_no, "input.frame"
        if isinstance(payload, dict):
            value = payload.get("seq_no")
            return (value if type(value) is int else None), payload.get("type")
        return None, None

    async def _reject(self, message, seq_no=None, event_type=None):
        error = {"type": "error", "code": "invalid_request", "message": message}
        if seq_no is not None:
            error["seq_no"] = seq_no
        await self.send(error)
        if event_type == "input.frame":
            await self.send(
                {"type": "input.frame.rejected", "seq_no": seq_no, "reason": message}
            )

    async def _receive_inputs(self, inputs):
        while not self.closed:
            remaining = (
                None
                if self.configured
                else max(0, self._configure_deadline - time.monotonic())
            )
            try:
                message = await asyncio.wait_for(self.websocket.receive(), remaining)
            except TimeoutError:
                if self.configured:
                    continue
                self._fatal_error = {
                    "type": "error",
                    "code": "session_timeout",
                    "message": "session.configure deadline exceeded",
                }
                self.closed = True
                return
            if self.closed:
                return
            if message["type"] == "websocket.disconnect":
                self.closed = True
                return
            if message["type"] != "websocket.receive":
                continue
            kind, payload = None, None
            try:
                if message.get("bytes") is not None:
                    kind, payload = "bytes", message["bytes"]
                    if self.pending_frame is None or self._binary_timer is None:
                        seq, _ = self._ref(kind, payload, self.pending_frame)
                        if self.pending_frame is not None:
                            self._fatal_error = {
                                "type": "error",
                                "code": "response_failed",
                                "seq_no": seq,
                                "message": "duplicate binary before frame acceptance",
                            }
                            self.closed = True
                            return
                        await self._reject(
                            "binary frame requires an outstanding input.frame.ready",
                            seq,
                        )
                        continue
                    self._cancel_binary_timer()
                else:
                    kind = "json"
                    payload = json.loads(message.get("text") or "")
                    if not isinstance(payload, dict):
                        raise ValueError("control message must be a JSON object")
                    if payload.get("type") == "session.abort":
                        if set(payload) != {"type"}:
                            raise ValueError("session.abort accepts only type")
                        self.closed = True
                        return
            except (ValueError, TypeError) as exc:
                seq, event_type = self._ref(kind, payload, self.pending_frame)
                await self._reject(str(exc), seq, event_type)
                continue
            if inputs.qsize() >= 2 * self.input_queue_capacity + 2:
                seq, _ = self._ref(kind, payload, self.pending_frame)
                self._fatal_error = {
                    "type": "error",
                    "code": "response_failed",
                    "seq_no": seq,
                    "message": "too many queued inputs; follow ready/accepted backpressure",
                }
                self.closed = True
                return
            inputs.put_nowait((kind, payload))

    async def _process_inputs(self, inputs):
        while not self.closed:
            kind, payload = await inputs.get()
            seq, event_type = self._ref(kind, payload, self.pending_frame)
            try:
                if self.closed:
                    return
                if kind == "bytes":
                    await self.handle_frame_bytes(payload)
                else:
                    await self.handle_json(payload)
            except (TypeError, ValidationError, ValueError) as exc:
                await self._reject(str(exc), seq, event_type)
            except Exception as exc:
                self._fatal_error = {
                    "type": "error",
                    "code": (
                        "session_timeout"
                        if isinstance(exc, TimeoutError)
                        else "response_failed"
                    ),
                    "message": str(exc),
                    "seq_no": seq,
                }
                self.closed = True
                return
            finally:
                inputs.task_done()

    async def handle_json(self, payload):
        kind = payload.get("type")
        if kind == "session.configure":
            await self.configure(Configure.model_validate(payload))
        elif kind == "input.frame":
            await self.prepare_frame(Frame.model_validate(payload))
        elif kind == "input.prompt":
            await self.handle_prompt(Prompt.model_validate(payload))
        else:
            raise ValueError(f"unsupported event type: {kind!r}")

    async def handle_frame_bytes(self, payload):
        self._cancel_binary_timer()
        pending = self.pending_frame
        if pending is None:
            raise ValueError("binary frame must follow input.frame metadata")
        if not payload or len(payload) > MAX_FRAME_BYTES:
            self.pending_frame = None
            await self._release_input(pending.metadata.seq_no)
            raise ValueError("frame is empty or exceeds the advertised byte limit")

        def validate_image():
            try:
                with Image.open(BytesIO(payload)) as image:
                    if Image.MIME.get(image.format) != pending.metadata.mime_type:
                        raise ValueError("image encoding does not match mime_type")
                    image.verify()
            except Exception as exc:
                raise ValueError(f"invalid image: {exc}") from exc

        try:
            await asyncio.wait_for(
                asyncio.to_thread(validate_image), self.binary_timeout_s
            )
        except ValueError:
            self.pending_frame = None
            await self._release_input(pending.metadata.seq_no)
            raise
        if self.closed:
            return
        await super().handle_frame_bytes(payload)

    async def _final_snapshot(self):
        if not self._request_started:
            return {"usage": empty_usage(), "watermark": 0, "failure_code": None}
        response = await self.raw_client.admin(
            FINALIZE_ACTION,
            {"request_id": self.request_id, "session_id": self.session_id},
            stages=[self.accounting_stage],
            timeout_s=self.finalize_timeout_s,
        )
        results = response.get("results", [])
        if not response.get("success") or len(results) != 1:
            raise RuntimeError("backend accounting finalization failed")
        result = results[0].get("data", {})
        if (
            result.get("request_id") != self.request_id
            or result.get("session_id") != self.session_id
            or result.get("final") is not True
        ):
            raise RuntimeError("missing authoritative final accounting snapshot")
        return result

    async def _finish(self, reason):
        caller = asyncio.current_task()
        self._finish_waiters.add(caller)
        if self._finish_task is None:
            self._finalizing = True
            self.closed = True
            self._cancel_binary_timer()
            self._finish_task = asyncio.create_task(self._finish_impl(reason))
        try:
            await await_cleanup(self._finish_task)
        finally:
            self._finish_waiters.discard(caller)

    async def _finish_impl(self, reason):
        async with self._finish_lock:
            if self._finalized:
                return
            self._finalizing = True
            self.closed = True
            self._cancel_binary_timer()
            try:
                final = await asyncio.wait_for(
                    self._final_snapshot(), self.finalize_timeout_s
                )
                async with self._event_lock:
                    self.ledger.record(final["usage"], final["watermark"])
                    error = self._fatal_error
                    code = final.get("failure_code")
                    if code:
                        error = {
                            **(error or {"message": "backend request failed"}),
                            "type": "error",
                            "code": code,
                        }
                        if final.get("seq_no") is not None:
                            error["seq_no"] = final["seq_no"]
                    if error:
                        code = error.get("code", "response_failed")
                        if code in ("configuration_timeout", "session_timeout"):
                            code = "session_timeout"
                        elif code != "context_exhausted":
                            code = "response_failed"
                        reason = (
                            "context_exhausted"
                            if code == "context_exhausted"
                            else "error"
                        )
                        await self._wire({**error, "code": code})
                    elif reason == "completed":
                        await self._settle_response(self._finish_reason)
                    self._terminal_sent = True
                    await self._wire(
                        {
                            "type": "session.done",
                            "session_id": self.session_id,
                            "request_id": self.request_id,
                            "reason": reason,
                            "aborted": reason == "aborted",
                            "usage": dict(self.ledger.latest),
                        }
                    )
            except Exception as exc:
                # No invented zero/partial terminal total when the model worker
                # cannot provide a trustworthy finalization acknowledgement.
                with suppress(Exception):
                    await self._wire(
                        {
                            "type": "error",
                            "code": "response_failed",
                            "message": f"authoritative accounting unavailable: {type(exc).__name__}: {exc}",
                        }
                    )
                    await asyncio.wait_for(self.websocket.close(code=1011), 5)
            finally:
                self._finalized = True
                timers = [
                    task
                    for task in self._timer_tasks
                    if task is not asyncio.current_task()
                    and task not in self._finish_waiters
                ]
                for timer in timers:
                    timer.cancel()
                await asyncio.gather(*timers, return_exceptions=True)
                self._timer_tasks.clear()
                task = self.response_task
                if (
                    task is not None
                    and task is not asyncio.current_task()
                    and task not in self._finish_waiters
                ):
                    task.cancel()
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            asyncio.gather(task, return_exceptions=True), 5
                        )
                with suppress(Exception):
                    await asyncio.wait_for(self.raw_client.abort(self.request_id), 5)
                try:
                    self.frame_store.cleanup(self.request_id)
                finally:
                    self.frame_refs_by_seq.clear()
                    self.pending_frame = None
                    await self._close_input_queue()
                    with suppress(Exception):
                        await asyncio.wait_for(
                            VideoRealtimeSession.close_websocket(self), 5
                        )

    async def teardown(self):
        await self._finish("completed" if self.request_finished else "aborted")

    async def close_websocket(self):
        await self._finish("completed" if self.request_finished else "aborted")
