"""Tests for tools/stress_realtime.py against an in-process fake omni server.

`FakeOmniServer` follows the server/tests/test_sglang_omni_adapter.py
FakeSglangOmniServer pattern (websockets in a daemon thread) but stands alone:
it verifies the wire protocol strictly (configure field set, dense seq_no,
monotonic 0.1s timestamps, two-phase frame ordering) and scripts responses
(text deltas + silence per prompt). It can also impersonate the gateway plane
(POST /v1/realtime/sessions via process_request, ws_token check on upgrade).

Run:  <repo>/.venv/bin/python -m pytest tools/ -q
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

websockets = pytest.importorskip("websockets")

import stress_realtime as sr

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"  # minimal jpeg-marker blob
WS_TOKEN = "tok-123"


class ConnRecord:
    """Everything one connection received, in arrival order."""

    def __init__(self) -> None:
        self.configure: Optional[Dict[str, Any]] = None
        self.seqs: List[int] = []            # seq_no of every input.* metadata
        self.frame_timestamps: List[float] = []
        self.prompts: List[str] = []
        self.binaries: List[bytes] = []
        self.order: List[str] = []           # metaN / readyN / binN / promptN
        self.aborts = 0
        self.rejected_already = False
        self.spontaneous_deltas = 0          # unprompted deltas the server pushed


class FakeOmniServer:
    def __init__(self, *, reject_capacity: bool = False,
                 reject_first_input: bool = False, gateway: bool = False,
                 reply: tuple = ("你好，", "画面正常。"), delta_delay: float = 0.02,
                 spontaneous: bool = False, spontaneous_interval: float = 0.1):
        self.reject_capacity = reject_capacity
        self.reject_first_input = reject_first_input
        self.gateway = gateway
        self.reply = reply
        self.delta_delay = delta_delay
        # MOSS-VL-Realtime speaks unprompted: continuous narration deltas on
        # the current turn_id, ended by a silence — this is what broke naive
        # turn attribution in the smoke run
        self.spontaneous = spontaneous
        self.spontaneous_interval = spontaneous_interval
        self.connections: List[ConnRecord] = []
        self.gateway_posts = 0
        self.port = 0
        self.http_port = 0
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        # gateway mode: base URL is the REST plane; direct mode: the WS server
        port = self.http_port if self.gateway else self.port
        return f"http://127.0.0.1:{port}"

    def start(self) -> "FakeOmniServer":
        self._thread.start()
        assert self._started.wait(5.0), "fake server did not start"
        if self.gateway:
            outer = self

            class RestHandler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:
                    if self.path != "/v1/realtime/sessions":
                        self.send_error(404)
                        return
                    outer.gateway_posts += 1
                    body = json.dumps({
                        "session_id": f"gw-{outer.gateway_posts}",
                        "ws_token": WS_TOKEN,
                        # websockets' server rejects plain POSTs before
                        # process_request, so REST and WS live on two ports;
                        # an absolute ws_url keeps them on one base
                        "ws_url": f"ws://127.0.0.1:{outer.port}/v1/realtime",
                        "expires_in": 60}).encode()
                    self.send_response(201)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args: Any) -> None:
                    pass

            self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), RestHandler)
            self.http_port = int(self._httpd.server_address[1])
            threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        async def shutdown() -> None:
            self._server.close()
            await self._server.wait_closed()
            await asyncio.sleep(0.05)
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
            self._server = await websockets.serve(
                self._handler, "127.0.0.1", 0, close_timeout=0.2)
            self.port = self._server.sockets[0].getsockname()[1]
            self._started.set()

        self._loop.run_until_complete(main())
        self._loop.run_forever()

    async def _respond(self, ws: Any, turn_id: int) -> None:
        """The answer to a prompt: deltas + silence on the NEW turn_id."""
        try:
            await asyncio.sleep(self.delta_delay)
            for piece in self.reply:
                await ws.send(json.dumps(
                    {"type": "response.text.delta", "delta": piece, "turn_id": turn_id}))
                await asyncio.sleep(self.delta_delay)
            await ws.send(json.dumps(
                {"type": "response.turn.silence", "turn_id": turn_id}))
        except Exception:  # noqa: BLE001 — client vanished
            pass

    async def _spontaneous(self, ws: Any, state: Dict[str, int], rec: ConnRecord) -> None:
        """Unprompted narration rounds on whatever turn is current."""
        try:
            while True:
                await asyncio.sleep(self.spontaneous_interval)
                tid = state["turn_id"]
                await ws.send(json.dumps(
                    {"type": "response.text.delta", "delta": "自发",
                     "turn_id": tid}))
                rec.spontaneous_deltas += 1
                await asyncio.sleep(self.delta_delay)
                # skip this round's silence if a prompt advanced the turn
                # mid-round (the interrupt sequence owns that transition)
                if tid == state["turn_id"]:
                    await ws.send(json.dumps(
                        {"type": "response.turn.silence", "turn_id": tid}))
        except Exception:  # noqa: BLE001 — client vanished
            pass

    def _maybe_reject(self, rec: ConnRecord) -> bool:
        if self.reject_first_input and not rec.rejected_already:
            rec.rejected_already = True
            return True
        return False

    async def _handler(self, ws: Any) -> None:
        rec = ConnRecord()
        self.connections.append(rec)
        try:
            if self.gateway:
                path = str(getattr(getattr(ws, "request", None), "path", "") or "")
                if f"ws_token={WS_TOKEN}" not in path:
                    await ws.send(json.dumps(
                        {"type": "error", "code": "ws_token_invalid",
                         "message": "missing or unknown ws_token"}))
                    await ws.close(code=1008)
                    return
            if self.reject_capacity:
                await ws.send(json.dumps({
                    "type": "error", "code": "session_capacity_exceeded",
                    "message": "server already hosts a session"}))
                await ws.close(code=1013)
                return
            await ws.send(json.dumps({
                "type": "session.created", "session_id": f"fake-{len(self.connections)}",
                "request_id": "req-1", "model": "fake-moss-vl", "turn_id": 0}))
            state = {"turn_id": 0}
            spontaneous_task = (asyncio.create_task(self._spontaneous(ws, state, rec))
                                if self.spontaneous else None)
            pending_seq: Optional[int] = None
            try:
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        assert pending_seq is not None, "binary frame without input.frame metadata"
                        rec.order.append(f"bin{pending_seq}")
                        rec.binaries.append(bytes(raw))
                        await ws.send(json.dumps({
                            "type": "input.frame.accepted", "seq_no": pending_seq,
                            "interrupts_current_turn": False, "pending_events": 0}))
                        await ws.send(json.dumps({
                            "type": "input.frame.processed", "seq_no": pending_seq}))
                        pending_seq = None
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "session.configure":
                        rec.configure = msg
                        await ws.send(json.dumps({
                            "type": "session.configured", "max_tokens_per_turn": 86400}))
                        await ws.send(json.dumps({"type": "session.ready", "turn_id": 0}))
                    elif mtype == "input.frame":
                        seq = msg["seq_no"]
                        rec.seqs.append(seq)
                        rec.frame_timestamps.append(msg["timestamp"])
                        rec.order.append(f"meta{seq}")
                        if self._maybe_reject(rec):
                            await ws.send(json.dumps({
                                "type": "error", "code": "invalid_request",
                                "message": "fake rejection", "seq_no": seq}))
                            continue
                        rec.order.append(f"ready{seq}")
                        await ws.send(json.dumps(
                            {"type": "input.frame.ready", "seq_no": seq}))
                        pending_seq = seq
                    elif mtype == "input.prompt":
                        seq = msg["seq_no"]
                        rec.seqs.append(seq)
                        rec.order.append(f"prompt{seq}")
                        rec.prompts.append(msg.get("prompt"))
                        if self._maybe_reject(rec):
                            await ws.send(json.dumps({
                                "type": "error", "code": "invalid_request",
                                "message": "fake rejection", "seq_no": seq}))
                            continue
                        # barge-in sequence per sglang-omni video_realtime.py:
                        # accepted → processed(interrupted_turn_id=old, turn_id=NEW)
                        # → turn.interrupted(old, next=new) → straggler old-turn
                        # delta → answer streams on the NEW turn
                        old_turn = state["turn_id"]
                        new_turn = old_turn + 1
                        await ws.send(json.dumps({
                            "type": "input.prompt.accepted", "seq_no": seq,
                            "interrupts_current_turn": True}))
                        await ws.send(json.dumps({
                            "type": "input.prompt.processed", "seq_no": seq,
                            "interrupted_turn_id": old_turn, "turn_id": new_turn}))
                        state["turn_id"] = new_turn
                        await ws.send(json.dumps({
                            "type": "response.turn.interrupted",
                            "turn_id": old_turn, "next_turn_id": new_turn}))
                        await ws.send(json.dumps({
                            "type": "response.text.delta", "delta": "（残留）",
                            "turn_id": old_turn}))
                        asyncio.create_task(self._respond(ws, new_turn))
                    elif mtype == "session.abort":
                        rec.aborts += 1
                        await ws.send(json.dumps({"type": "session.done", "aborted": True}))
                        await ws.close()
            finally:
                if spontaneous_task is not None:
                    spontaneous_task.cancel()
        except Exception:  # noqa: BLE001 — client vanished mid-conversation
            pass


@pytest.fixture()
def jpeg_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(3):
        (d / f"f{i}.jpg").write_bytes(JPEG + bytes([i]))
    return d


def make_cfg(server: FakeOmniServer, jpeg_dir: Path, tmp_path: Path,
             *, mode: str = "direct", sessions: int = 2, duration: float = 1.5,
             fps: float = 10.0, prompt_interval: float = 0.5,
             ramp: str = "", ramp_rest: float = 0.05,
             out_name: str = "report") -> sr.StressConfig:
    return sr.parse_args([
        "--mode", mode, "--url", server.url,
        "--sessions", str(sessions), "--duration", str(duration),
        "--fps", str(fps), "--video", str(jpeg_dir),
        "--prompt-interval", str(prompt_interval),
        *(["--ramp", ramp] if ramp else []),
        "--ramp-rest", str(ramp_rest),
        "--out", str(tmp_path / out_name),
    ])


def read_report(tmp_path: Path, name: str = "report") -> tuple:
    payload = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
    md = (tmp_path / f"{name}.md").read_text(encoding="utf-8")
    return payload, md


MD_SECTIONS = ["## 1. 硬件", "## 2. 视频输入", "## 3. 会话负载", "## 4. 并发",
               "## 5. 时延", "## 6. 稳定性", "## 7. 资源", "## 8. 容量策略"]


# ------------------------------------------------------------------ tests


def test_direct_protocol_and_metrics(jpeg_dir: Path, tmp_path: Path) -> None:
    server = FakeOmniServer().start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path)
        assert sr.run(cfg) == 0
    finally:
        server.close()

    assert len(server.connections) == 2
    for rec in server.connections:
        # handshake: configure carries exactly the allowed fields (extra=forbid)
        assert rec.configure is not None
        assert set(rec.configure) == {"type", "prompt", "system_prompt",
                                      "max_new_tokens", "temperature", "top_p",
                                      "input_queue_capacity"}
        assert rec.configure["type"] == "session.configure"
        # seq_no: dense, starting at 0, no gaps (frames and prompts share it)
        assert rec.seqs == list(range(len(rec.seqs))), rec.seqs
        assert len(rec.seqs) > 5
        # timestamps: monotonic non-decreasing, 0.1s granularity
        ts = rec.frame_timestamps
        assert all(b >= a for a, b in zip(ts, ts[1:]))
        assert all(abs(t * 10 - round(t * 10)) < 1e-6 for t in ts)
        # two-phase: every binary preceded by its ready, order meta → ready → bin
        for i, seq in enumerate(rec.seqs):
            pass  # seq order checked above; per-frame order below
        for j, marker in enumerate(rec.order):
            if marker.startswith("bin"):
                n = marker[3:]
                assert rec.order[j - 1] == f"ready{n}", rec.order
        assert rec.prompts, "prompt-interval schedule should fire within duration"
        assert rec.aborts == 1

    payload, md = read_report(tmp_path)
    level = payload["levels"][0]
    assert level["success_rate"] == 1.0
    assert level["frames_accepted"] == sum(r["frames_sent"] for r in level["sessions"])
    assert level["prompts_sent"] >= 2
    assert level["ttft_first_s"]["p50"] is not None
    assert level["turn_latency_s"]["p50"] is not None
    assert level["frame_throughput_fps"] > 0
    for r in level["sessions"]:
        assert r["ok"] and r["ttft_first_s"] is not None and r["turns_completed"] >= 1
        assert r["deltas"] >= 2 and r["silence_events"] >= 1
    assert payload["stable_concurrency"] == 2
    for section in MD_SECTIONS:
        assert section in md
    assert "稳定并发" in md


def test_gateway_mode(jpeg_dir: Path, tmp_path: Path) -> None:
    server = FakeOmniServer(gateway=True).start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, mode="gateway", sessions=2,
                       duration=1.0, out_name="gw")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    assert server.gateway_posts == 2
    assert len(server.connections) == 2  # both ws upgrades carried a valid token
    payload, _ = read_report(tmp_path, "gw")
    level = payload["levels"][0]
    assert level["success_rate"] == 1.0
    assert all(r["session_id"].startswith("gw-") for r in level["sessions"])


def test_capacity_rejection(jpeg_dir: Path, tmp_path: Path) -> None:
    server = FakeOmniServer(reject_capacity=True).start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=2, duration=1.0,
                       out_name="cap")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    payload, md = read_report(tmp_path, "cap")
    level = payload["levels"][0]
    assert level["capacity_rejected"] == 2
    assert level["success_rate"] == 0.0
    assert payload["stable_concurrency"] is None
    assert "session_capacity_exceeded" in md
    assert all(r["capacity_rejected"] for r in level["sessions"])


def test_invalid_request_seq_rollback(jpeg_dir: Path, tmp_path: Path) -> None:
    server = FakeOmniServer(reject_first_input=True).start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=1, duration=1.5,
                       out_name="rej")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    (rec,) = server.connections
    # seq 0 was rejected once, rolled back, and reused by the next input
    assert rec.seqs[0] == 0 and rec.seqs[1] == 0
    assert rec.seqs[1:] == list(range(len(rec.seqs) - 1)), rec.seqs
    payload, _ = read_report(tmp_path, "rej")
    level = payload["levels"][0]
    assert level["invalid_requests"] == 1
    assert level["error_events"] == 1
    assert level["success_rate"] == 1.0  # invalid_request rejects one event, not the session


def test_ramp_stable_concurrency(jpeg_dir: Path, tmp_path: Path) -> None:
    server = FakeOmniServer().start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=1, duration=1.0,
                       ramp="1,2", ramp_rest=0.05, out_name="ramp")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    payload, md = read_report(tmp_path, "ramp")
    assert [l["level"] for l in payload["levels"]] == [1, 2]
    assert all(l["success_rate"] == 1.0 for l in payload["levels"])
    assert payload["stable_concurrency"] == 2
    assert "| 并发 | 成功率 |" in md
    assert len(server.connections) == 3  # 1 + 2 across the two levels


def test_turn_attribution_with_spontaneous_speech(jpeg_dir: Path, tmp_path: Path) -> None:
    """Regression for the gateway smoke bug: the model talks unprompted on the
    current turn_id; only the prompt's processed event (new turn_id) may arm
    latency measurement, and only that turn's deltas/silence close it."""
    server = FakeOmniServer(spontaneous=True, spontaneous_interval=0.1).start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=1, duration=2.0,
                       fps=20.0, prompt_interval=0.6, out_name="attr")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    rec = server.connections[0]
    assert rec.spontaneous_deltas > 0, "fake must actually narrate unprompted"

    payload, _ = read_report(tmp_path, "attr")
    r = payload["levels"][0]["sessions"][0]
    assert r["prompts_sent"] >= 2
    # the pre-fix bug left these null/0 despite deltas flowing
    assert r["ttft_first_s"] is not None
    assert len(r["turn_ttfts_s"]) == r["turns_completed"]
    assert r["turns_completed"] == r["prompts_sent"]
    assert r["turns_unanswered"] == 0
    # spontaneous deltas are counted globally but never attributed to a turn:
    # global delta count exceeds the answer deltas alone
    assert r["deltas"] > r["turns_completed"] * len(server.reply)
    # TTFT = first NEW-turn delta minus prompt send; bounded by the fake's
    # answer delay, never ~0 from pre-prompt spontaneous deltas
    for ttft in r["turn_ttfts_s"]:
        assert 0.0 < ttft <= 1.0
    # each prompt barged in on the current turn → one interrupted event each
    assert r["interrupted_events"] == r["prompts_sent"]


def test_turn_arm_timeout_fallback(jpeg_dir: Path, tmp_path: Path) -> None:
    """No processed event at all (abnormal server): the arm timeout must keep
    the turn measured instead of leaving ttft null forever."""
    server = FakeOmniServer().start()
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=1, duration=1.0,
                       out_name="armfb")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    # simulate the abnormal path directly: an open turn whose processed event
    # never came arms itself via the PROMPT_ARM_TIMEOUT_S fallback
    sess = sr.StressSession(cfg, 0, _NullCursor(), ["问"], None)
    turn = sr._OpenTurn(t_send=time.monotonic(), prompt_seq=0)
    sess._open_turn = turn
    assert not sess._turn_matches(turn, {"type": "response.text.delta",
                                         "delta": "x", "turn_id": 7})
    turn.arm_any = True  # what the sender loop sets after the timeout
    assert sess._turn_matches(turn, {"type": "response.text.delta",
                                     "delta": "x", "turn_id": 7})


class _NullCursor(sr.FrameCursor):
    async def next(self) -> bytes:
        return JPEG


def test_gpu_sampler_start_join() -> None:
    """Regression: the sampler's stop Event shadowed threading.Thread._stop()
    and join() crashed with "'Event' object is not callable" — which is how the
    real ramp run lost its report during teardown. Works with or without
    nvidia-smi on the host."""
    sampler = sr.GpuSampler(interval_s=1.0)
    sampler.start()
    time.sleep(0.2)
    sampler.stop()
    sampler.join(timeout=5.0)   # must not raise
    assert not sampler.is_alive()


def test_ramp_checkpoints_report_after_each_level(jpeg_dir: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Every finished ramp level must flush a cumulative report, so a crash in
    a later level or in teardown never loses earlier results."""
    server = FakeOmniServer().start()
    writes: List[Path] = []
    real_write = sr.write_reports

    def spy_write(cfg: sr.StressConfig, *args: Any, **kwargs: Any) -> None:
        real_write(cfg, *args, **kwargs)
        writes.append(Path(cfg.out))

    monkeypatch.setattr(sr, "write_reports", spy_write)
    try:
        cfg = make_cfg(server, jpeg_dir, tmp_path, sessions=1, duration=0.8,
                       ramp="1,2", ramp_rest=0.05, out_name="ckpt")
        assert sr.run(cfg) == 0
    finally:
        server.close()

    # one flush per level + the final flush
    assert len(writes) == 3, writes
    payload, _ = read_report(tmp_path, "ckpt")
    assert [l["level"] for l in payload["levels"]] == [1, 2]
