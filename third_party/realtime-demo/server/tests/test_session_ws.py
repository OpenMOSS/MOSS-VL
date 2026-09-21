"""B4/B5/B7 integration: REST lifecycle + session WS over a live uvicorn server.

Run:  <repo>/.venv/bin/python -m server.tests.test_session_ws

Boots the real FastAPI app in-process (fake ASR/TTS/VLM engines injected via
build_runtime), then drives it with a real `websockets` client: create → attach →
PTT voice turn with mic PCM + JPEG frame → captions + binary TTS audio →
reconnect with ?last_seq replay → supersede → session.update/ping → DELETE →
grace GC after an abrupt disconnect.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import threading
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import websockets
from PIL import Image

from server import protocol as p
from server.config import Settings
from server.deps import Runtime
from server.session.manager import SessionManager
from server.tests.fakes import (
    DEFAULT_REPLY,
    FakeAsrAdapter,
    FakeTtsAdapter,
    FakeVlmAdapter,
    LOUD_CHUNK,
)

GRACE_S = 2.0


def make_fake_runtime() -> Runtime:
    settings = dataclasses.replace(
        Settings(),
        session_grace_seconds=GRACE_S,
        status_interval_s=0.5,
        max_sessions=1,  # keep the 409-capacity flow testable with fake engines
    )
    rt = Runtime.__new__(Runtime)
    rt.settings = settings
    rt.plan = None
    rt.vlm_supervisor = None
    rt.tts_pool = None
    rt.asr = FakeAsrAdapter()
    rt.tts = FakeTtsAdapter()
    rt.vlm = FakeVlmAdapter(token_interval=0.02)
    rt.index = None       # persistence off for this test (open_persistence no-ops)
    rt.media = None
    rt.history = None
    rt.session_manager = SessionManager(settings)
    rt._tts_sessions = {}
    rt._lock = threading.Lock()
    return rt


def http(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 48), (30, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


class Collector:
    """Reassembles the wire: JSON events, with binary frames attached to their descriptor."""

    def __init__(self, ws):
        self.ws = ws
        self.events: List[Dict[str, Any]] = []

    async def next_event(self, timeout: float = 8.0) -> Dict[str, Any]:
        while True:
            message = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            if isinstance(message, (bytes, bytearray)):
                assert self.events, "binary frame before any descriptor"
                tag, _, payload = p.parse_binary(message)
                assert tag == p.TAG_TTS_PCM, f"unexpected binary tag 0x{tag:02x}"
                self.events[-1]["_pcm"] = payload
                continue
            ev = json.loads(message)
            self.events.append(ev)
            return ev

    async def until(self, type_: str, timeout: float = 10.0) -> Dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            assert remaining > 0, f"timeout waiting {type_}; got {[e['type'] for e in self.events]}"
            ev = await self.next_event(timeout=remaining)
            if ev["type"] == type_:
                return ev

    def of(self, type_: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["type"] == type_]

    @property
    def max_seq(self) -> int:
        return max((e.get("seq", 0) for e in self.events), default=0)


async def start_server() -> Tuple[uvicorn.Server, asyncio.Task, str, Runtime]:
    from server import app as app_module

    fake_rt = make_fake_runtime()
    app_module.build_runtime = lambda plan=None: fake_rt  # lifespan resolves this at call time
    app_module.get_settings = lambda: dataclasses.replace(fake_rt.settings, tts_spawn=False)

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=0,
                            log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
        assert not task.done(), "uvicorn failed to start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"127.0.0.1:{port}", fake_rt


async def run_ptt_turn(col: Collector) -> None:
    await col.ws.send(json.dumps({"type": "input.audio.start"}))
    await col.ws.send(p.video_binary(jpeg_bytes(), timestamp_ms=2500))
    for _ in range(4):
        await col.ws.send(p.mic_binary(LOUD_CHUNK))
    await col.ws.send(json.dumps({"type": "input.audio.commit"}))
    await col.until(p.RESPONSE_DONE)


async def main_async() -> None:
    server, server_task, host, rt = await start_server()
    base = f"http://{host}"
    try:
        # ---- B5: REST create + capacity + snapshot ----
        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions",
                                             {"config": {"capture_mode": "ptt"}})
        assert code == 201, (code, body)
        sid, ws_url = body["session_id"], body["ws_url"]
        assert ws_url == f"/api/session/{sid}/ws" and body["expires_at"] > 0

        code, second = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 409, (code, second)  # MAX_SESSIONS=1 backstop in this fixture

        code, snap = await asyncio.to_thread(http, "GET", f"{base}/api/sessions/{sid}")
        assert code == 200 and snap["phase"] == "created", snap
        print("REST create/capacity/snapshot: OK")

        # ---- B4: attach + full PTT voice turn over the wire ----
        async with websockets.connect(f"ws://{host}{ws_url}") as ws:
            col = Collector(ws)
            created = await col.next_event()
            assert created["type"] == p.SESSION_CREATED and created["session_id"] == sid
            assert created["audio_in"]["sample_rate"] == 16000

            await run_ptt_turn(col)

            td = col.of(p.TRANSCRIPTION_DONE)[0]
            assert td["text"] == rt.asr.text
            deltas = "".join(e["delta"] for e in col.of(p.RESPONSE_TEXT_DELTA))
            assert deltas == DEFAULT_REPLY, deltas
            audio = [e for e in col.of(p.RESPONSE_AUDIO_DELTA)]
            assert audio and all("_pcm" in e and len(e["_pcm"]) == e["pcm_bytes"] for e in audio)
            assert col.of(p.RESPONSE_DONE)[0]["stop_reason"] == p.STOP_END_TURN
            assert rt.vlm.sessions[0].prompt_frames, "JPEG frame must reach the VLM with the prompt"

            # ping/pong + live config update
            await ws.send(json.dumps({"type": "ping", "seq": 7}))
            pong = await col.until(p.PONG)
            assert pong["ping_seq"] == 7
            await ws.send(json.dumps({"type": "session.update",
                                      "config": {"tts_voice": "Alice", "vad_sensitivity": 0.8}}))
            updated = await col.until(p.SESSION_UPDATED)
            assert updated["config"]["tts_voice"] == "Alice"

            resume_from = td["seq"]  # pretend we lost everything after the transcription
            expected_replay_deltas = deltas
            max_seen = col.max_seq
        print("WS attach + PTT turn + ping + update: OK "
              f"(audio chunks: {len(audio)}, events: {len(col.events)})")

        # ---- B7: reconnect with ?last_seq → ring replay, session survives ----
        async with websockets.connect(f"ws://{host}{ws_url}?last_seq={resume_from}") as ws:
            col = Collector(ws)
            created = await col.next_event()
            assert created["type"] == p.SESSION_CREATED and created["replayed"] > 0
            await col.until(p.RESPONSE_DONE)  # the replayed turn tail
            replay_deltas = "".join(e["delta"] for e in col.of(p.RESPONSE_TEXT_DELTA))
            assert replay_deltas == expected_replay_deltas, replay_deltas
            replay_audio = col.of(p.RESPONSE_AUDIO_DELTA)
            assert replay_audio and all("_pcm" in e for e in replay_audio), "audio must replay too"
            assert all(e.get("seq", 10**9) > resume_from for e in col.events)
            assert not col.of(p.STATUS), "transient events must not replay"

            # a fresh typed turn works on the resumed socket
            await ws.send(json.dumps({"type": "text.input", "text": "继续"}))
            done = await col.until(p.RESPONSE_DONE)
            assert done["stop_reason"] == p.STOP_END_TURN
            max_seen = col.max_seq

            # ---- supersede: a second socket takes over; the first is released ----
            async with websockets.connect(f"ws://{host}{ws_url}?last_seq={max_seen}") as ws2:
                col2 = Collector(ws2)
                created2 = await col2.next_event()
                assert created2["type"] == p.SESSION_CREATED
                await ws2.send(json.dumps({"type": "ping", "seq": 1}))
                await col2.until(p.PONG)
                # old socket: server stops serving it (close or silence, no crash)
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        assert not isinstance(msg, (bytes, bytearray))
                except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                    pass
        print("reconnect replay + resumed turn + supersede: OK")

        # ---- B5: DELETE tears down; VLM realtime loop stops ----
        code, _ = await asyncio.to_thread(http, "DELETE", f"{base}/api/sessions/{sid}")
        assert code == 200
        code, _ = await asyncio.to_thread(http, "GET", f"{base}/api/sessions/{sid}")
        assert code == 404
        assert rt.vlm.sessions[0].active is False, "DELETE must stop the VLM session"
        code, _ = await asyncio.to_thread(http, "DELETE", f"{base}/api/sessions/{sid}")
        assert code == 404
        print("DELETE teardown: OK")

        # ---- B7: abrupt disconnect → grace GC ----
        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 201, (code, body)
        sid2 = body["session_id"]
        ws = await websockets.connect(f"ws://{host}/api/session/{sid2}/ws")
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created
        await ws.close()
        await asyncio.sleep(GRACE_S + 1.0)
        code, _ = await asyncio.to_thread(http, "GET", f"{base}/api/sessions/{sid2}")
        assert code == 404, "session must be GC'd after grace"
        assert rt.vlm.sessions[1].active is False
        print("grace GC after disconnect: OK")

        # ---- unknown session id → 4404 ----
        ws = await websockets.connect(f"ws://{host}/api/session/sess_nope/ws")
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == p.ERROR and msg["code"] == "not_found"
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except websockets.exceptions.ConnectionClosed as exc:
            assert exc.rcvd is not None and exc.rcvd.code == 4404, exc
        print("unknown session close(4404): OK")

        # ---- speech endpoints: one-shot ASR (dictation) + TTS (read-aloud) ----
        def post_raw(url: str, data: bytes, content_type: str) -> Tuple[int, bytes, str]:
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": content_type})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.status, resp.read(), resp.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read(), exc.headers.get("Content-Type", "")

        code, raw, _ = await asyncio.to_thread(
            post_raw, f"{base}/api/asr", LOUD_CHUNK * 8, "application/octet-stream")
        assert code == 200, (code, raw)
        assert json.loads(raw)["text"] == rt.asr.text
        code, raw, _ = await asyncio.to_thread(post_raw, f"{base}/api/asr", b"", "application/octet-stream")
        assert code == 400, (code, raw)

        code, raw, ctype = await asyncio.to_thread(
            post_raw, f"{base}/api/tts",
            json.dumps({"text": "你好，这是一段朗读测试。", "voice": "Alice"}).encode(),
            "application/json")
        assert code == 200 and ctype.startswith("audio/wav"), (code, ctype)
        assert raw[:4] == b"RIFF" and len(raw) > 44, "must be a non-empty WAV"
        assert rt.tts.engine.segments, "the segmenter must have fed the engine"
        print("speech endpoints (/api/asr, /api/tts): OK")
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=10)


def main() -> int:
    asyncio.run(main_async())
    print("\nSESSION WS INTEGRATION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
