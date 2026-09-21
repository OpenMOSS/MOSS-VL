"""Unit tests: the sglang-omni adapter (client/session/pool) against a fake server.

Run:  <repo>/.venv/bin/python -m server.tests.test_sglang_omni_adapter

`FakeSglangOmniServer` is an in-process WS server (python `websockets`) that
speaks the sglang-omni /v1/video/realtime protocol: session.created →
session.configure → session.configured → session.ready, two-phase frame
upload (input.frame → input.frame.ready → raw binary → accepted/processed),
scriptable response events.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from server.adapters.vlm.moss_vl_hf.online_pool import BUSY, DOWN, READY, NoFreeReplica
from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool
from server.config import Settings

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"\xff\xd9"  # minimal jpeg-marker blob

try:
    import websockets
except ImportError:  # pragma: no cover — env still provisioning
    websockets = None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    transformers = None
# ^ warm the heavy import at collection time: the session's text-token mirror
# does a LAZY `from transformers import AutoTokenizer` on the receiver thread,
# and a cold first import (~3s) eats poll_until's 3s budget → flaky tests when
# this file runs standalone (in full-suite runs some earlier module has
# already paid the import).


class FakeSglangOmniServer:
    """Speaks the sglang-omni realtime WS protocol; scriptable from the test.

    max_sessions mirrors omni's --max-running-requests: once that many
    connections are live, further connects get error[session_capacity_exceeded]
    + close 1013. None (default) = unlimited; reject_capacity=True = always
    reject. ready_delay_s (default 0) delays input.frame.ready to simulate a
    full input queue applying backpressure."""

    def __init__(self, *, send_processed: bool = True, reject_capacity: bool = False,
                 max_sessions: Optional[int] = None, ready_delay_s: float = 0.0):
        self.send_processed = send_processed
        self.reject_capacity = reject_capacity
        self.max_sessions = max_sessions
        self.ready_delay_s = ready_delay_s
        self.received: List[Dict[str, Any]] = []   # every client JSON message
        self.binaries: List[bytes] = []            # raw frame payloads
        self.configure_payload: Optional[Dict[str, Any]] = None
        self.aborts = 0
        self.active_sessions = 0                   # live WS connections
        self.rejected_connects = 0
        # stop() sends session.abort and waits for the server to close; ending
        # the session on abort keeps teardown fast and mirrors response.done →
        # session.done → server-initiated close
        self.end_on_abort = True
        self._reject_next_input: Optional[str] = None
        self._ws: Any = None
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._handler_done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ---- lifecycle ----

    def start(self) -> "FakeSglangOmniServer":
        self._thread.start()
        assert self._started.wait(5.0), "fake server did not start"
        return self

    def close(self) -> None:
        self._handler_done.wait(2.0)  # let the WS handler finish first

        async def shutdown() -> None:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:  # noqa: BLE001
                    pass
            self._server.close()
            await self._server.wait_closed()
            # let in-flight handler/keepalive tasks settle before loop stop
            await asyncio.sleep(0.1)
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=3)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def main() -> None:
            # close_timeout: don't linger on the close handshake — the test
            # client (websocket-client) replies from its recv thread while the
            # main thread may already be tearing the socket down
            self._server = await websockets.serve(
                self._handler, "127.0.0.1", 0, process_request=self._process_request,
                close_timeout=0.2)
            self.port = self._server.sockets[0].getsockname()[1]
            self._started.set()

        self._loop.run_until_complete(main())
        self._loop.run_forever()

    async def _process_request(self, connection: Any, request: Any) -> Any:
        if request.path == "/health":
            return connection.respond(200, json.dumps({"status": "ok", "loaded": True}))
        return None

    # ---- protocol handler ----

    async def _handler(self, ws: Any) -> None:
        self._ws = ws
        try:
            if self.reject_capacity or (
                    self.max_sessions is not None
                    and self.active_sessions >= self.max_sessions):
                self.rejected_connects += 1
                await ws.send(json.dumps({
                    "type": "error", "code": "session_capacity_exceeded",
                    "message": "server already hosts a session"}))
                await ws.close(code=1013)
                return
            self.active_sessions += 1
            pending_frame_seq: Optional[int] = None  # per-connection
            try:
                await ws.send(json.dumps({
                    "type": "session.created", "session_id": "fake-omni-session",
                    "request_id": "req-1", "model": "fake-moss-vl", "turn_id": 0}))
                try:
                    async for raw in ws:
                        if isinstance(raw, (bytes, bytearray)):
                            self.binaries.append(bytes(raw))
                            await ws.send(json.dumps({
                                "type": "input.frame.accepted", "seq_no": pending_frame_seq,
                                "interrupts_current_turn": False}))
                            if self.send_processed:
                                await ws.send(json.dumps({
                                    "type": "input.frame.processed",
                                    "seq_no": pending_frame_seq}))
                            continue
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            # mirrors omni: malformed JSON → invalid_request,
                            # the connection stays up
                            await ws.send(json.dumps({
                                "type": "error", "code": "invalid_request",
                                "message": "malformed JSON"}))
                            continue
                        self.received.append(message)
                        mtype = message.get("type")
                        if mtype == "session.configure":
                            self.configure_payload = message
                            await ws.send(json.dumps({
                                "type": "session.configured",
                                "max_tokens_per_turn": 86400}))
                            await ws.send(json.dumps({"type": "session.ready", "turn_id": 0}))
                        elif mtype == "input.frame":
                            if await self._maybe_reject(ws, message):
                                continue
                            if self.ready_delay_s:
                                # full input queue: ready is delayed, not dropped
                                await asyncio.sleep(self.ready_delay_s)
                            await ws.send(json.dumps({
                                "type": "input.frame.ready", "seq_no": message["seq_no"]}))
                            pending_frame_seq = message["seq_no"]
                        elif mtype == "input.prompt":
                            if await self._maybe_reject(ws, message):
                                continue
                            await ws.send(json.dumps({
                                "type": "input.prompt.accepted", "seq_no": message["seq_no"],
                                "interrupts_current_turn": True}))
                            if self.send_processed:
                                await ws.send(json.dumps({
                                    "type": "input.prompt.processed",
                                    "seq_no": message["seq_no"]}))
                        elif mtype == "session.abort":
                            self.aborts += 1
                            if self.end_on_abort:
                                await ws.send(json.dumps(
                                    {"type": "session.done", "aborted": True}))
                                await ws.close()
                except Exception:  # noqa: BLE001 — client vanished mid-conversation
                    pass
            finally:
                self.active_sessions -= 1
        finally:
            self._handler_done.set()

    async def _maybe_reject(self, ws: Any, message: Dict[str, Any]) -> bool:
        if self._reject_next_input is None:
            return False
        code = self._reject_next_input
        self._reject_next_input = None
        await ws.send(json.dumps({
            "type": "error", "code": code, "message": f"fake rejection ({code})"}))
        return True

    # ---- scripting helpers (test thread → server loop) ----

    def send_event(self, payload: Dict[str, Any]) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(payload)), self._loop)
        future.result(timeout=5.0)

    def send_delta(self, text: str, turn_id: int = 0) -> None:
        self.send_event({"type": "response.text.delta", "delta": text, "turn_id": turn_id})

    def send_ping(self) -> None:
        """Transport-level ping (what uvicorn sends every 20s server-side)."""
        future = asyncio.run_coroutine_threadsafe(self._ws.ping(), self._loop)
        future.result(timeout=5.0)

    def kill_connection(self) -> None:
        """Drop the live session connection (simulates the omni side dying)."""
        async def _kill() -> None:
            if self._ws is not None:
                await self._ws.close()
        asyncio.run_coroutine_threadsafe(_kill(), self._loop).result(timeout=5.0)

    def send_silence(self, turn_id: int = 0) -> None:
        self.send_event({"type": "response.turn.silence", "turn_id": turn_id, "seq_no": 0})

    def send_interrupted(self, turn_id: int = 0, next_turn_id: int = 1) -> None:
        self.send_event({"type": "response.turn.interrupted", "turn_id": turn_id,
                         "next_turn_id": next_turn_id, "seq_no": 0})

    def reject_next_input(self, code: str = "invalid_request") -> None:
        self._reject_next_input = code


def make_pool(server: FakeSglangOmniServer, **overrides: Any) -> SglangOmniPool:
    kwargs = dict(sglang_omni_urls=server.url, sglang_omni_connect_timeout_s=5.0,
                  sglang_omni_health_interval_s=600.0)  # prober effectively off in tests
    kwargs.update(overrides)
    pool = SglangOmniPool(Settings(**kwargs))
    pool.capacity_retry_delay_s = 0.05
    return pool


def poll_until(session: Any, want: str, timeout: float = 3.0) -> List[str]:
    """Collect chunks until `want` shows up (or timeout)."""
    deadline = time.monotonic() + timeout
    chunks: List[str] = []
    while time.monotonic() < deadline:
        batch = session.poll_output(timeout_seconds=0.2, max_items=32)
        chunks.extend(batch.chunks)
        if want in chunks:
            return chunks
        if not batch.active:
            return chunks
    raise AssertionError(f"timed out waiting for {want!r}; got {chunks!r}")


# ---------------------------------------------------------------- session tests


def test_handshake_and_frame_two_phase() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="描述画面", temperature=0.7)
        assert session.active
        # handshake happened, configure carried the mapped fields only
        cfg = server.configure_payload
        assert cfg is not None and cfg["type"] == "session.configure"
        assert cfg["prompt"] == "描述画面"
        assert cfg["input_queue_capacity"] == 4
        assert set(cfg) <= {"type", "prompt", "system_prompt", "max_new_tokens",
                            "max_tokens_per_turn", "temperature", "top_p",
                            "input_queue_capacity"}
        # not supplied by the caller → omni default (86400 = unthrottled)
        assert cfg["max_tokens_per_turn"] == 86400.0

        # two-phase upload: metadata → frame.ready → binary → accepted/processed
        st = session.put_frame(JPEG, timestamp=1.0)
        assert st["frames_received"] == 1 and st["frames_dropped"] == 0
        frame_msgs = [m for m in server.received if m.get("type") == "input.frame"]
        assert len(frame_msgs) == 1
        assert frame_msgs[0]["seq_no"] == 0 and frame_msgs[0]["mime_type"] == "image/jpeg"
        assert server.binaries == [JPEG]
        for _ in range(20):  # processed arrives async
            if session.status()["frames_consumed"] == 1:
                break
            time.sleep(0.05)
        assert session.status()["frames_consumed"] == 1
        session.stop(timeout_seconds=2.0)
        print("handshake + two-phase frame: OK")
    finally:
        server.close()


def test_credit_drop_and_prompt_priority() -> None:
    # capacity 2, server never processes → the 3rd pure frame waits then drops;
    # a prompt frame and a pure prompt must still go through
    server = FakeSglangOmniServer(send_processed=False).start()
    try:
        pool = make_pool(server, sglang_omni_input_queue_capacity=2,
                         sglang_omni_input_drop_wait_seconds=0.3)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="")

        t0 = time.monotonic()
        st1 = session.put_frame(JPEG, timestamp=1.0)
        st2 = session.put_frame(JPEG, timestamp=2.0)
        st3 = session.put_frame(JPEG, timestamp=3.0)   # credit full → dropped
        waited = time.monotonic() - t0
        assert st1["frames_received"] == 1 and st2["frames_received"] == 2
        assert st3.get("frame_dropped") is True and st3["frames_dropped"] == 1
        assert 0.25 <= waited <= 2.0, f"drop wait should be ~0.3s, took {waited:.2f}s"

        # prompt-carrying inputs bypass the credit gate, never drop
        st4 = session.put_prompt_frame("现在呢？", JPEG, timestamp=4.0)
        assert st4["frames_received"] == 3 and st4["frames_dropped"] == 1
        assert st4["prompts_received"] == 1
        st5 = session.put_prompt("纯文本提问")
        assert st5["prompts_received"] == 2

        seqs = [m["seq_no"] for m in server.received
                if m.get("type") in ("input.frame", "input.prompt")]
        assert seqs == [0, 1, 2, 3], f"seq_no must stay dense after the drop: {seqs}"
        prompts = [m for m in server.received if m.get("type") == "input.prompt"]
        assert prompts[0]["prompt"] == "纯文本提问"
        session.stop(timeout_seconds=2.0)
        print("credit drop + prompt priority: OK")
    finally:
        server.close()


def test_event_to_control_token_mapping() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="")

        # first delta of a turn is preceded by <|round_start|>
        server.send_delta("你好", turn_id=0)
        chunks = poll_until(session, "你好")
        assert chunks == ["<|round_start|>", "你好"], chunks

        server.send_silence(turn_id=0)
        chunks = poll_until(session, "<|silence|>")
        assert chunks == ["<|silence|>"], chunks

        # barge-in ack → <|eot_id|>; stragglers from the old turn are dropped
        server.send_interrupted(turn_id=0, next_turn_id=1)
        chunks = poll_until(session, "<|eot_id|>")
        assert chunks == ["<|eot_id|>"], chunks
        server.send_delta("残留", turn_id=0)   # stale turn — must be filtered
        server.send_delta("新一轮", turn_id=1)
        chunks = poll_until(session, "新一轮")
        assert chunks == ["<|round_start|>", "新一轮"], chunks

        status = session.status()
        assert status["outputs_emitted"] >= 5 and status["text_tokens"] > 0
        session.stop(timeout_seconds=2.0)
        print("event → control-token mapping: OK")
    finally:
        server.close()


def test_response_done_deactivates() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="")
        server.send_event({"type": "response.done", "finish_reason": "stop", "turn_id": 0})
        batch = session.poll_output(timeout_seconds=2.0)
        # active=False may need one extra poll (the event lands between polls)
        deadline = time.monotonic() + 3.0
        while batch.active and time.monotonic() < deadline:
            batch = session.poll_output(timeout_seconds=0.2)
        assert not batch.active and not session.active
        session.stop(timeout_seconds=2.0)
        print("response.done → active=False: OK")
    finally:
        server.close()


def test_invalid_request_rolls_back_seq() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="")

        server.reject_next_input("invalid_request")
        try:
            session.put_prompt("这条会被拒")
        except (RuntimeError, ValueError) as exc:
            assert "invalid_request" in str(exc)
        else:
            raise AssertionError("rejected input must raise")
        assert session.active, "invalid_request rejects one event, not the session"

        session.put_prompt("下一条复用同一个 seq_no")
        prompts = [m for m in server.received if m.get("type") == "input.prompt"]
        assert [m["seq_no"] for m in prompts] == [0, 0], prompts
        session.stop(timeout_seconds=2.0)
        print("invalid_request seq rollback: OK")
    finally:
        server.close()


# ---------------------------------------------------------------- pool tests


def test_pool_acquire_capacity_and_release() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        assert not pool.is_loaded()
        pool.load("", -1, "online_streaming")
        assert pool.is_loaded() and pool.capacity == 1 and pool.busy == 0
        assert not hasattr(pool, "set_replica_health"), \
            "app.py keys the local worker supervisor off this method — must not exist"

        session = pool.start_realtime_session(prompt="")
        assert pool.busy == 1 and pool.replicas[0].state == BUSY
        try:
            pool.start_realtime_session(prompt="")
        except NoFreeReplica as exc:
            assert "1/1" in str(exc)
        else:
            raise AssertionError("busy pool must raise NoFreeReplica")

        session.stop(timeout_seconds=2.0)
        assert pool.busy == 0 and pool.replicas[0].state == READY
        print("pool acquire/capacity/release: OK")
    finally:
        server.close()


def test_pool_capacity_exceeded_marks_full() -> None:
    server = FakeSglangOmniServer(reject_capacity=True).start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        try:
            pool.start_realtime_session(prompt="")
        except NoFreeReplica:
            pass
        else:
            raise AssertionError("capacity-exceeded replica must raise NoFreeReplica")
        # multi-slot semantics: full ≠ wedged — the replica is marked BUSY
        # (full), not quarantined DOWN; only the all-replicas-rejected path
        # gets the one teardown-grace retry before NoFreeReplica
        assert pool.replicas[0].state == BUSY, pool.replicas[0].state
        assert server.rejected_connects == 2  # initial + the one grace retry
        print("capacity_exceeded → BUSY(full), grace retry once: OK")
    finally:
        server.close()


def test_prefill_messages_configure_mapping() -> None:
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        prefill = json.dumps([
            {"role": "system", "content": "你是导览员。"},
            {"role": "user", "content": "之前聊过什么？"},
            {"role": "assistant", "content": "你看到了一只猫。"},
        ], ensure_ascii=False)
        payload = pool._configure_payload(dict(
            prompt="继续描述画面", system_prompt=None, prefill_messages=prefill,
            temperature=0.7, top_k=20, top_p=0.8, do_sample=True,
            repetition_penalty=1.05, max_new_tokens=512, max_tokens_per_turn=20,
            min_pixels=None, video_fps=2.0))
        assert payload["system_prompt"] == "你是导览员。"
        assert "继续描述画面" in payload["prompt"]
        assert "user: 之前聊过什么？" in payload["prompt"]
        assert "assistant: 你看到了一只猫。" in payload["prompt"]
        assert payload["max_new_tokens"] == 512
        assert payload["temperature"] == 0.7 and payload["top_p"] == 0.8
        # max_tokens_per_turn IS supported (tokens/second rate cap) and maps
        assert payload["max_tokens_per_turn"] == 20.0
        # unsupported knobs never reach the wire (extra=forbid → 422)
        for banned in ("top_k", "do_sample", "repetition_penalty",
                       "video_fps", "prefill_messages"):
            assert banned not in payload, banned
        # do_sample=False forces greedy
        greedy = pool._configure_payload(dict(prompt="", do_sample=False, temperature=0.7))
        assert greedy["temperature"] == 0.0
        print("prefill_messages → configure mapping: OK")
    finally:
        server.close()


class _KeepaliveFakeWS:
    """Bare-minimum stand-in for the websocket-client socket: only what
    SglangOmniClient._start_keepalive touches. `pong=True` simulates a peer
    that answers pings (pong arrival bumps client._last_seen)."""

    def __init__(self, client, pong: bool):
        self._client = client
        self._pong = pong
        self.pings = 0
        self.aborted = False

    def ping(self) -> None:
        self.pings += 1
        if self._pong:
            # the receiver thread would observe the pong frame; simulate it
            self._client._last_seen = time.monotonic()

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        pass


def test_server_ping_does_not_kill_session() -> None:
    """Regression: uvicorn's 20s server-side ping must be answered and skipped,
    not mistaken for an 'unexpected binary event' (2026-09-06 live bug: every
    session died exactly 20s in)."""
    server = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server)
        pool.load("", -1, "online_streaming")
        session = pool.start_realtime_session(prompt="")
        server.send_ping()
        server.send_ping()
        # events still flow after transport pings
        server.send_delta("活着", turn_id=0)
        chunks = poll_until(session, "活着")
        assert "活着" in chunks
        assert session.active
        session.stop(timeout_seconds=2.0)
        print("server ping survival: OK")
    finally:
        server.close()


def test_multi_replica_failover() -> None:
    """Two replicas, one slot each: when s1's transport dies, releasing it
    quarantines replica 0, and the relay's fresh session must land on replica
    1 and keep working. With BOTH replicas quarantined, NoFreeReplica — the
    exact terminal state observed in single-replica production 2026-09-06."""
    server_a = FakeSglangOmniServer().start()
    server_b = FakeSglangOmniServer().start()
    try:
        pool = make_pool(server_a, sglang_omni_urls=f"{server_a.url},{server_b.url}")
        pool.load("", -1, "online_streaming")

        # s1 lands on replica 0 (least-loaded, ties → lowest index)
        s1 = pool.start_realtime_session(prompt="")
        assert server_a.configure_payload is not None
        assert server_b.configure_payload is None

        # transport dies → receiver marks the session ws_closed
        server_a.kill_connection()
        deadline = time.monotonic() + 5.0
        while s1.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not s1.active, "session did not notice the dead transport"
        s1.stop(timeout_seconds=2.0)  # release replica 0 as transport dead → DOWN
        assert pool._replicas[0].state == DOWN, pool._replicas[0].state

        # the relay's replacement session must land on replica 1 and work
        s2 = pool.start_realtime_session(prompt="")
        assert server_b.configure_payload is not None, "relay did not fail over to replica 1"
        server_b.send_delta("接力成功", turn_id=0)
        assert "接力成功" in poll_until(s2, "接力成功")
        s2.stop(timeout_seconds=2.0)  # clean release → replica 1 free again

        # now kill replica 1's session too → both DOWN → NoFreeReplica
        s3 = pool.start_realtime_session(prompt="")
        server_b.kill_connection()
        deadline = time.monotonic() + 5.0
        while s3.active and time.monotonic() < deadline:
            time.sleep(0.05)
        s3.stop(timeout_seconds=2.0)
        assert all(r.state == DOWN for r in pool._replicas)
        try:
            pool.start_realtime_session(prompt="")
            raise AssertionError("expected NoFreeReplica with all replicas DOWN")
        except NoFreeReplica:
            pass
        print("multi-replica failover: OK")
    finally:
        server_a.close()
        server_b.close()


def test_keepalive_aborts_on_missed_pong() -> None:
    from server.adapters.vlm.moss_vl_sglang_omni.client import SglangOmniClient
    client = SglangOmniClient("http://127.0.0.1:1", ping_interval_s=0.05,
                              ping_timeout_s=0.05)
    fake = _KeepaliveFakeWS(client, pong=False)
    client._ws = fake
    client._last_seen = time.monotonic()
    client._start_keepalive()
    try:
        deadline = time.monotonic() + 2.0
        while not fake.aborted and time.monotonic() < deadline:
            time.sleep(0.02)
        assert fake.aborted, "keepalive never aborted a peer that misses pongs"
        assert fake.pings >= 1
    finally:
        client._closed.set()


def test_keepalive_tolerates_answering_peer() -> None:
    from server.adapters.vlm.moss_vl_sglang_omni.client import SglangOmniClient
    client = SglangOmniClient("http://127.0.0.1:1", ping_interval_s=0.05,
                              ping_timeout_s=0.05)
    fake = _KeepaliveFakeWS(client, pong=True)
    client._ws = fake
    client._last_seen = time.monotonic()
    client._start_keepalive()
    try:
        time.sleep(0.4)  # several ping cycles
        assert not fake.aborted, "keepalive aborted a peer that answers pings"
        assert fake.pings >= 2
    finally:
        client._closed.set()


def main() -> int:
    if websockets is None:
        print("SKIP: the `websockets` package is not installed in this env")
        return 0
    test_handshake_and_frame_two_phase()
    test_credit_drop_and_prompt_priority()
    test_event_to_control_token_mapping()
    test_response_done_deactivates()
    test_invalid_request_rolls_back_seq()
    test_pool_acquire_capacity_and_release()
    test_pool_capacity_exceeded_marks_full()
    test_prefill_messages_configure_mapping()
    test_server_ping_does_not_kill_session()
    test_multi_replica_failover()
    test_keepalive_aborts_on_missed_pong()
    test_keepalive_tolerates_answering_peer()
    print("\nSGLANG-OMNI ADAPTER TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
