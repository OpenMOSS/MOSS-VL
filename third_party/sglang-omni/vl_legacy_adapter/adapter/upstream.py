"""Thin async client for the omni video realtime protocol.

Implements, against ``WS /v1/video/realtime``:

* connect and receive the server-initiated ``session.created``;
* ``session.configure`` then wait for ``session.configured`` + ``session.ready``;
* per-frame three-step handshake: ``input.frame`` metadata ->
  wait ``input.frame.ready`` -> send binary -> wait ``input.frame.accepted``;
* dispatch of stream events (``response.text.delta``, ``response.done``,
  ``session.done``, ``error``, ...) to a callback;
* ``session.abort`` and close.

Capacity exhaustion at accept time surfaces as ``error
[session_capacity_exceeded]`` followed by close code 1013; both are mapped
to :class:`UpstreamBusy`.

A single listener task owns all reads from the upstream socket; the frame
handshake only writes, so there is exactly one reader per connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from . import config

logger = logging.getLogger(__name__)

_CLOSE_TRY_AGAIN = 1013


class UpstreamError(RuntimeError):
    """Any non-busy upstream failure."""


class UpstreamBusy(UpstreamError):
    """Omni reported no free session slot (error + close 1013)."""


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class OmniUpstream:
    def __init__(self, url: str, on_event: EventCallback) -> None:
        self.url = url
        self.on_event = on_event
        self.ws: Any = None
        self.session_id: str | None = None
        self._listener: asyncio.Task[None] | None = None
        self._pending: dict[tuple[str, int], asyncio.Future[dict[str, Any]]] = {}
        self._configured: asyncio.Future[dict[str, Any]] | None = None
        self._ready: asyncio.Future[dict[str, Any]] | None = None
        self._closed_event = asyncio.Event()
        self.terminal_seen = False
        self.last_error: str | None = None

    async def connect(self) -> None:
        try:
            self.ws = await connect(
                self.url,
                max_size=config.WS_MAX_SIZE,
                open_timeout=config.READY_TIMEOUT_S,
            )
            raw = await asyncio.wait_for(self.ws.recv(), config.READY_TIMEOUT_S)
        except ConnectionClosed as exc:
            if exc.rcvd is not None and exc.rcvd.code == _CLOSE_TRY_AGAIN:
                raise UpstreamBusy("omni closed with 1013 (try again)") from exc
            raise UpstreamError(f"upstream connection failed: {exc}") from exc
        except (OSError, asyncio.TimeoutError, WebSocketException) as exc:
            raise UpstreamError(f"upstream connection failed: {exc}") from exc

        message = self._decode(raw)
        if message.get("type") == "error":
            if message.get("code") == "session_capacity_exceeded":
                raise UpstreamBusy(message.get("message", "capacity exceeded"))
            raise UpstreamError(message.get("message", "upstream error"))
        if message.get("type") != "session.created":
            raise UpstreamError(f"unexpected first upstream event: {message!r}")
        self.session_id = message.get("session_id")
        self._listener = asyncio.create_task(self._listen())

    async def configure(self, payload: dict[str, Any]) -> None:
        """Send session.configure and wait for configured + ready."""
        loop = asyncio.get_running_loop()
        self._configured = loop.create_future()
        self._ready = loop.create_future()
        waiting = asyncio.gather(self._configured, self._ready)
        try:
            await self.ws.send(json.dumps(payload))
            await asyncio.wait_for(waiting, config.READY_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError, WebSocketException) as exc:
            raise UpstreamError(f"upstream configuration failed: {exc}") from exc
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)

    async def send_frame(
        self,
        seq_no: int,
        timestamp: float,
        data: bytes,
        *,
        final: bool,
        prompt: str | None = None,
    ) -> None:
        """Run the full three-step handshake for one frame."""
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[dict[str, Any]] = loop.create_future()
        accepted: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[("ready", seq_no)] = ready
        self._pending[("accepted", seq_no)] = accepted
        try:
            metadata: dict[str, Any] = {
                "type": "input.frame",
                "seq_no": seq_no,
                "timestamp": float(timestamp),
                "mime_type": "image/jpeg",
                "final": bool(final),
            }
            if prompt:
                metadata["prompt"] = prompt
            await self.ws.send(json.dumps(metadata))
            await asyncio.wait_for(ready, config.ACK_TIMEOUT_S)
            await self.ws.send(data)
            await asyncio.wait_for(accepted, config.ACK_TIMEOUT_S)
        finally:
            self._pending.pop(("ready", seq_no), None)
            self._pending.pop(("accepted", seq_no), None)

    async def abort(self) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps({"type": "session.abort"}))
        except (OSError, ConnectionClosed):
            pass

    async def close(self) -> None:
        if self._listener is not None:
            self._listener.cancel()
            await asyncio.gather(self._listener, return_exceptions=True)
        if self.ws is not None:
            try:
                await self.ws.close()
            except (OSError, ConnectionClosed):
                pass

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raise UpstreamError("unexpected binary message from upstream")
        try:
            message = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise UpstreamError("upstream message is not valid JSON") from exc
        if not isinstance(message, dict):
            raise UpstreamError("upstream message is not a JSON object")
        return message

    def _fail_all_pending(self, exc: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        for future in (self._configured, self._ready):
            if future is not None and not future.done():
                future.set_exception(exc)

    async def _listen(self) -> None:
        try:
            async for raw in self.ws:
                message = self._decode(raw)
                event_type = message.get("type")
                if event_type == "error":
                    self.last_error = message.get("message", "upstream error")
                    exc = UpstreamError(self.last_error)
                    # Per-event errors (e.g. invalid_request) must not kill
                    # the session; resolve nothing and just report.
                    await self.on_event(message)
                    continue
                if event_type == "session.configured":
                    if self._configured is not None and not self._configured.done():
                        self._configured.set_result(message)
                    continue
                if event_type == "session.ready":
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(message)
                    continue
                if event_type == "input.frame.ready":
                    future = self._pending.get(("ready", int(message["seq_no"])))
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue
                if event_type in ("input.frame.accepted", "input.prompt.accepted"):
                    future = self._pending.get(("accepted", int(message["seq_no"])))
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue
                if event_type == "session.done":
                    self.terminal_seen = True
                # input.frame.processed, response.text.delta, response.done,
                # session.done, session.usage, response.turn.* -> callback
                await self.on_event(message)
        except ConnectionClosed as exc:
            if self.last_error is None and not self.terminal_seen:
                if exc.rcvd is not None and exc.rcvd.code == _CLOSE_TRY_AGAIN:
                    self.last_error = config.BUSY_MESSAGE
                elif not self._closed_event.is_set():
                    self.last_error = f"upstream connection closed: {exc}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report, then close
            logger.exception("upstream listener failed")
            self.last_error = f"upstream listener error: {exc}"
        finally:
            self._closed_event.set()
            if self.last_error is not None:
                self._fail_all_pending(UpstreamError(self.last_error))
            else:
                self._fail_all_pending(UpstreamError("upstream connection closed"))
