"""Per-round bridge state machine between a legacy client and omni.

One downstream WebSocket connection equals one analysis round and owns one
fresh upstream omni session (legacy contract: rounds are independent and
isolated; sessions are never reused).

Flow:
    downstream start        -> connect upstream, map params, configure,
                               wait configured+ready, send legacy ``ready``
    frame meta + binary     -> buffered in arrival order (cap
                               FRAME_BUFFER_CAP), then drained one by one
                               through the omni three-step handshake; each
                               accepted frame produces one legacy
                               ``frame_ack``
    response.text.delta     -> renamed to legacy ``output`` verbatim
    session.done            -> forge ``output`` with text ``<|im_end|>``;
                               if no visible text was produced this round the
                               legacy contract judges a bare marker invalid,
                               so an ``error`` is delivered instead
    stop / disconnect       -> session.abort upstream, close, release slot

Round finalization: the legacy contract has no explicit "start generating"
signal. When the frame queue is drained and no new frame arrives within
FINALIZE_QUIET_S, the last frame is sent upstream with ``final=true`` so the
omni session winds down. Omni then keeps emitting <|silence|> tokens until
its token budget is exhausted instead of stopping, so the round is closed
after SILENCE_FINALIZE_S without text deltas (the same 1 s silence rule the
legacy caller applies), or on session.done, whichever comes first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from websockets.exceptions import ConnectionClosed

from . import config
from .params import map_start_to_configure
from .upstream import OmniUpstream, UpstreamBusy, UpstreamError

logger = logging.getLogger(__name__)

END_MARKER = "<|im_end|>"


class BridgeSession:
    def __init__(self, downstream: Any, on_client_done: Any = None) -> None:
        self.downstream = downstream
        # Invoked once at the start of teardown, before upstream abort/close,
        # so the adapter's inflight slot frees the moment the client is done
        # and an immediate next round does not see a stale busy.
        self._on_client_done = on_client_done
        self.upstream: OmniUpstream | None = None
        self.frames: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()
        self.started = False
        self.start_prompt = ""
        self.ready_sent = False
        self.final_sent = False
        self.round_finished = False
        self.visible_text = False
        self.last_timestamp: float | None = None
        self.pending_meta: float | None = None
        self._pending_meta_deadline: float | None = None
        self._frame_changed = asyncio.Event()
        self._input_closed = False
        self._send_lock = asyncio.Lock()
        self._start_task: asyncio.Task[None] | None = None
        self._pump: asyncio.Task[None] | None = None
        self._finalizer: asyncio.Task[None] | None = None
        self._watch: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._abort_initiated = False
        self._last_delta_at: float | None = None
        self._silence_since: float | None = None
        self._done = asyncio.Event()

    async def run(self) -> None:
        start_deadline = time.monotonic() + config.START_TIMEOUT_S
        try:
            while True:
                try:
                    if (
                        self._pending_meta_deadline is not None
                        and not self.round_finished
                    ):
                        remaining = self._pending_meta_deadline - time.monotonic()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        raw = await asyncio.wait_for(self.downstream.recv(), remaining)
                    elif self.started or self.round_finished:
                        raw = await self.downstream.recv()
                    else:
                        # Idle pre-start connections must not hold the
                        # MAX_INFLIGHT slot forever.
                        remaining = start_deadline - time.monotonic()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        raw = await asyncio.wait_for(self.downstream.recv(), remaining)
                except asyncio.TimeoutError:
                    if self._pending_meta_deadline is not None:
                        await self._fail_round(
                            "timed out waiting for frame binary data"
                        )
                        return
                    logger.info(
                        "closing idle connection: no start within %.1fs",
                        config.START_TIMEOUT_S,
                    )
                    await self._close_downstream(
                        code=1008, reason="no start received in time"
                    )
                    return
                except ConnectionClosed:
                    self._stop_requested = True
                    break
                if self.round_finished:
                    # After the end marker the caller sends stop and closes;
                    # ignore anything else rather than erroring.
                    if isinstance(raw, str) and '"stop"' in raw:
                        break
                    continue
                if isinstance(raw, bytes):
                    await self._handle_binary(raw)
                else:
                    keep_going = await self._handle_text(raw)
                    if not keep_going:
                        break
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------
    # downstream message handling

    async def _handle_text(self, raw: str) -> bool:
        """Returns False when the connection should be torn down."""
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("message is not valid JSON")
            return True
        if not isinstance(message, dict):
            await self._send_error("message must be a JSON object")
            return True
        msg_type = message.get("type")
        if msg_type == "start":
            if self.started:
                await self._send_error("session already started")
                return True
            self.started = True
            prompt = message.get("prompt")
            self.start_prompt = prompt if isinstance(prompt, str) else ""
            self._start_task = asyncio.create_task(self._start(message))
        elif msg_type == "frame":
            await self._handle_frame_meta(message)
        elif msg_type == "stop":
            self._stop_requested = True
            return False
        else:
            await self._send_error(f"unsupported message type: {msg_type!r}")
        return True

    async def _handle_frame_meta(self, message: dict[str, Any]) -> None:
        if not self.ready_sent:
            await self._send_error("wait for ready before sending frames")
            return
        if self._input_closed or self.round_finished:
            await self._send_error("round already finalized; extra frame rejected")
            return
        if self.pending_meta is not None:
            await self._send_error("frame metadata must be followed by binary data")
            return
        timestamp = message.get("timestamp")
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            await self._send_error("frame timestamp must be a non-negative number")
            return
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            await self._send_error(
                f"timestamp moved backwards: {timestamp} < {self.last_timestamp}"
            )
            return
        self.pending_meta = timestamp
        self._pending_meta_deadline = time.monotonic() + config.FRAME_RECEIVE_TIMEOUT_S
        self._frame_changed.set()

    async def _handle_binary(self, payload: bytes) -> None:
        if self._input_closed or self.round_finished:
            await self._send_error("round already finalized; extra frame rejected")
            return
        if self.pending_meta is None:
            await self._send_error("binary frame must follow frame metadata")
            return
        if len(payload) > config.MAX_FRAME_BYTES:
            await self._send_error("frame payload exceeds 10 MiB limit")
            self._clear_pending_meta()
            return
        if self.frames.qsize() >= config.FRAME_BUFFER_CAP:
            await self._send_error(
                f"frame buffer full ({config.FRAME_BUFFER_CAP} frames)"
            )
            self._clear_pending_meta()
            return
        self.frames.put_nowait((self.pending_meta, payload))
        self.last_timestamp = self.pending_meta
        self._clear_pending_meta()

    def _clear_pending_meta(self) -> None:
        self.pending_meta = None
        self._pending_meta_deadline = None
        self._frame_changed.set()

    # ------------------------------------------------------------------
    # upstream lifecycle

    async def _start(self, start: dict[str, Any]) -> None:
        try:
            async with asyncio.timeout(config.SETUP_TIMEOUT_S):
                await self._initialize(start)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            await self._fail_round("upstream session setup timed out")
        except Exception:
            logger.exception("unexpected upstream session setup failure")
            await self._fail_round("upstream session setup failed: internal error")

    async def _initialize(self, start: dict[str, Any]) -> None:
        try:
            configure = map_start_to_configure(start)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # Unconvertible parameter types must surface as a consumable
            # error, not hang the client.
            await self._fail_round(f"invalid start parameters: {exc}")
            return
        try:
            self.upstream = OmniUpstream(config.OMNI_WS_URL, self._on_upstream_event)
            await self.upstream.connect()
            if self._stop_requested or self.round_finished:
                return  # client stopped/disconnected while connecting
            await self.upstream.configure(configure)
        except UpstreamBusy:
            await self._fail_round(config.BUSY_MESSAGE)
            return
        except UpstreamError as exc:
            await self._fail_round(f"upstream session setup failed: {exc}")
            return
        if self._stop_requested or self.round_finished:
            return  # client stopped/disconnected while configuring
        self.ready_sent = True
        await self._send({"type": "ready"})
        self._pump = asyncio.create_task(self._pump_frames())
        self._finalizer = asyncio.create_task(self._finalize_on_silence())
        self._watch = asyncio.create_task(self._watch_upstream())

    async def _watch_upstream(self) -> None:
        """Surface an unexpected upstream drop as a consumable error."""
        assert self.upstream is not None
        await self.upstream.wait_closed()
        if self.round_finished:
            return
        await self._send_error(
            self.upstream.last_error or "upstream session ended unexpectedly"
        )
        await self._finish_round()
        await self._close_downstream()

    async def _pump_frames(self) -> None:
        seq_no = 0
        try:
            item = await self.frames.get()
            while True:
                final = False
                if self.frames.empty():
                    nxt = await self._next_frame_or_none()
                    if nxt is None:
                        final = True
                    else:
                        await self._send_one_frame(seq_no, item, final=False)
                        seq_no += 1
                        item = nxt
                        continue
                await self._send_one_frame(seq_no, item, final=final)
                seq_no += 1
                if final:
                    self.final_sent = True
                    return
                item = await self.frames.get()
        except asyncio.CancelledError:
            raise
        except (UpstreamError, asyncio.TimeoutError) as exc:
            logger.warning("frame pump failed: %s", exc)
            await self._send_error(f"frame submission failed: {exc}")
            await self._finish_round()
            await self._close_downstream()

    async def _next_frame_or_none(self) -> tuple[float, bytes] | None:
        """Wait for complete input without finalizing an in-progress binary."""
        deadline = time.monotonic() + config.FINALIZE_QUIET_S
        while True:
            if self.round_finished or self._stop_requested:
                raise asyncio.CancelledError
            self._frame_changed.clear()
            try:
                return self.frames.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self.pending_meta is not None:
                # The receiver owns the binary deadline, including the first
                # frame, before this pump has any complete frame to consume.
                await self._frame_changed.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # No await between the final input check and closing admission.
                self._input_closed = True
                return None
            try:
                await asyncio.wait_for(self._frame_changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass  # Recheck queue and pending metadata before committing.

    async def _finalize_on_silence(self) -> None:
        """Close the round when the model goes silent after the final frame.

        Omni does not stop on its own: after the visible answer it streams
        <|silence|> tokens until the token budget is exhausted, so waiting
        for session.done would blow the legacy 10 s round budget. The legacy
        caller wraps up after ~1 s of output silence; mirror that here.
        """
        try:
            while not self.round_finished:
                await asyncio.sleep(0.1)
                if not self.final_sent:
                    continue
                now = time.monotonic()
                if self.visible_text:
                    quiet = (
                        self._last_delta_at is not None
                        and now - self._last_delta_at >= config.SILENCE_FINALIZE_S
                    )
                else:
                    quiet = (
                        self._silence_since is not None
                        and now - self._silence_since >= config.SILENCE_FINALIZE_S
                    )
                if quiet:
                    logger.info(
                        "finalizing round after %.1fs of upstream silence",
                        config.SILENCE_FINALIZE_S,
                    )
                    await self._complete_round()
                    if self.upstream is not None:
                        self._abort_initiated = True
                        await self.upstream.abort()
                    return
        except asyncio.CancelledError:
            raise

    async def _send_one_frame(
        self, seq_no: int, item: tuple[float, bytes], *, final: bool
    ) -> None:
        timestamp, payload = item
        assert self.upstream is not None
        # The legacy round prompt rides on the final frame: upstream treats a
        # frame-attached prompt as the question that opens the answer turn.
        prompt = (self.start_prompt or None) if final else None
        await self.upstream.send_frame(
            seq_no, timestamp, payload, final=final, prompt=prompt
        )
        await self._send({"type": "frame_ack"})

    async def _on_upstream_event(self, message: dict[str, Any]) -> None:
        event_type = message.get("type")
        if event_type == "response.text.delta":
            delta = message.get("delta", "")
            if delta:
                if delta.strip():
                    self.visible_text = True
                self._last_delta_at = time.monotonic()
                self._silence_since = None
                await self._send({"type": "output", "text": delta})
        elif event_type == "response.turn.silence":
            # The model is emitting <|silence|> tokens (skipped from text).
            # After the final frame, a sustained run of silence means the
            # round's answer — if any — is complete.
            if self.final_sent and self._silence_since is None:
                self._silence_since = time.monotonic()
        elif event_type == "response.done":
            # Intermediate turns may finish before the session does; the
            # terminal signal for the round is session.done.
            pass
        elif event_type == "session.done":
            if (
                message.get("aborted")
                and not self._abort_initiated
                and not self.round_finished
            ):
                # An abort we did not request (upstream fault mid-round) is
                # an error for the legacy caller, not a clean round end —
                # forging the end marker here would fake a valid answer.
                await self._send_error("upstream session aborted unexpectedly")
                await self._finish_round()
                await self._close_downstream()
                return
            await self._complete_round()
        elif event_type == "error":
            await self._send_error(message.get("message", "upstream error"))
        # input.frame.processed / session.usage / response.turn.*: omni
        # internals with no legacy counterpart; intentionally dropped.

    async def _complete_round(self) -> None:
        if self.round_finished:
            return
        if self.visible_text:
            await self._send({"type": "output", "text": END_MARKER})
        else:
            await self._send_error("model produced no visible output this round")
        await self._finish_round()

    # ------------------------------------------------------------------
    # teardown

    async def _fail_round(self, message: str) -> None:
        self._input_closed = True
        self._clear_pending_meta()
        await self._finish_round()
        try:
            await self._send_error(message)
        finally:
            await self._close_downstream()

    async def _finish_round(self) -> None:
        self.round_finished = True
        self._done.set()

    async def _shutdown(self) -> None:
        if self._on_client_done is not None:
            await self._on_client_done()
        for task in (self._start_task, self._pump, self._finalizer, self._watch):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self.upstream is not None:
            if not self.upstream.terminal_seen:
                self._abort_initiated = True
                await self.upstream.abort()
            await self.upstream.close()
        try:
            await self.downstream.close()
        except (OSError, ConnectionClosed):
            pass

    async def _close_downstream(self, code: int = 1000, reason: str = "") -> None:
        try:
            await self.downstream.close(code, reason)
        except (OSError, ConnectionClosed):
            pass

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.downstream.send(json.dumps(payload, ensure_ascii=False))
            except (OSError, ConnectionClosed):
                pass

    async def _send_error(self, message: str) -> None:
        await self._send({"type": "error", "message": message})
