"""TTS provider registry / sizing / client-protocol tests (no GPU, no models).

Run:  <repo>/.venv/bin/python -m server.tests.test_tts_providers

Covers the multi-provider seam added with cosyvoice3 + moss_tts_realtime:
- every provider string (and alias) builds its adapter offline
- placement sizes vLLM engines by TTS_SESSIONS_PER_ENGINE and sidecars by
  TTS_SESSIONS_PER_SIDECAR
- Cosyvoice3SpeechEngine resolves ref_text transcripts from the vendored
  demo.jsonl and sends ref_audio+ref_text on the wire
- MossRtNativeEngine drives the upstream fast_api session protocol
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("TTS_ENABLED", "0")
os.environ.setdefault("ASR_ENABLED", "0")
os.environ.setdefault("AUTOLOAD_VLM", "0")

from server.config import Settings  # noqa: E402
from server.adapters.registry import build_tts  # noqa: E402
from server.adapters.tts.providers import (  # noqa: E402
    canonical_tts_provider, is_vllm_engine_provider)


def test_registry_builds_all_providers() -> None:
    base = Settings()
    expected = {
        "moss_tts_nano": "MossTtsNanoEngine",
        "nano": "MossTtsNanoEngine",
        "vllm_omni": "VllmOmniSpeechEngine",
        "cosyvoice3": "Cosyvoice3SpeechEngine",
        "cosy": "Cosyvoice3SpeechEngine",
        "moss_tts_realtime": "MossRtSpeechEngine",
        "mossrt": "MossRtSpeechEngine",
        "cosyvoice3_native": "Cosyvoice3NativeEngine",
        "moss_tts_realtime_native": "MossRtNativeEngine",
    }
    for provider, engine_name in expected.items():
        adapter = build_tts(dataclasses.replace(base, tts_provider=provider))
        assert type(adapter.engine).__name__ == engine_name, (provider, type(adapter.engine))
        assert adapter.caps.channels in (1, 2)
    try:
        build_tts(dataclasses.replace(base, tts_provider="kokoro"))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unknown provider must raise")
    print("registry: OK")


def test_provider_classification() -> None:
    assert canonical_tts_provider(" Fun-CosyVoice3 ") == "cosyvoice3"
    assert is_vllm_engine_provider("vllm")
    assert is_vllm_engine_provider("mossrt")
    assert is_vllm_engine_provider("cosy")
    assert not is_vllm_engine_provider("nano")
    assert not is_vllm_engine_provider("cosyvoice3_native")
    assert not is_vllm_engine_provider("moss_tts_realtime_native")
    print("classification: OK")


def test_placement_sizing() -> None:
    from server.gpu.placement import plan_placement  # local import: heavier module
    from server.gpu.topology import parse_smi_csv
    smi_8x = "\n".join(
        f"{i}, GPU-{i:08x}, NVIDIA H200, 143771, 4, 9.0" for i in range(8))
    topology = parse_smi_csv(smi_8x)
    for provider, per_engine in [("moss_tts_nano", False), ("vllm_omni", True),
                                 ("cosyvoice3", True), ("moss_tts_realtime", True),
                                 ("cosyvoice3_native", False),
                                 ("moss_tts_realtime_native", False)]:
        s = dataclasses.replace(Settings(), tts_provider=provider,
                                offline_provider="none", vlm_worker_gpus="",
                                tts_sessions_per_sidecar=2, tts_sessions_per_engine=4)
        plan = plan_placement(topology, s)
        workers = len(plan.workers)
        expected = -(-workers // (4 if per_engine else 2))  # ceil
        assert len(plan.tts) == expected, (provider, len(plan.tts), expected, workers)
    print("placement sizing: OK")


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeTts/1"
    captured: list = []

    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        type(self).captured.append((self.path, body))
        if self.path == "/v1/audio/speech":
            self.send_response(200)
            self.send_header("X-Audio-Sample-Rate", "24000")
            self.send_header("X-Audio-Channels", "1")
            self.end_headers()
            self.wfile.write(b"\x00\x01" * 512)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_GET(self):  # noqa: N802
        type(self).captured.append((self.path, None))
        if self.path.startswith("/tts/session/") and self.path.endswith("/audio"):
            self.send_response(200)
            self.send_header("X-Audio-Sample-Rate", "24000")
            self.send_header("X-Audio-Channels", "1")
            self.end_headers()
            self.wfile.write(b"\x00\x02" * 512)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):  # noqa: A002
        pass


def _serve() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_cosyvoice3_speech_payload() -> None:
    srv, url = _serve()
    try:
        s = dataclasses.replace(Settings(), tts_provider="cosyvoice3")
        adapter = build_tts(s)
        engine = type(adapter.engine)(s, url)
        engine.ready = True  # skip health for the wire test
        _Handler.captured.clear()
        pcm = b"".join(engine.synthesize_pcm("你好。", "Yuewen"))
        assert pcm, "no PCM streamed"
        path, body = _Handler.captured[-1]
        assert path == "/v1/audio/speech"
        assert body["model"] == s.tts_cosy3_served_name
        assert body["ref_audio"].startswith("data:audio/")
        assert body["ref_text"], "cosyvoice3 must send a reference transcript"
        assert body["response_format"] == "pcm" and body["stream"] is True
        assert engine.sample_rate == 24000 and engine.channels == 1
    finally:
        srv.shutdown()
    print("cosyvoice3 payload: OK")


def test_mossrt_native_session_protocol() -> None:
    srv, url = _serve()
    try:
        s = dataclasses.replace(Settings(), tts_provider="moss_tts_realtime_native")
        adapter = build_tts(s)
        engine = type(adapter.engine)(s, url)
        engine.ready = True
        _Handler.captured.clear()
        pcm = b"".join(engine.synthesize_pcm("你好。", "Yuewen"))
        assert pcm, "no PCM streamed"
        paths = [p for p, _ in _Handler.captured]
        assert paths[0] == "/tts/session/start", paths
        assert paths[1] == "/tts/session/push", paths
        assert paths[2].endswith("/audio"), paths
        assert paths[3] == "/tts/session/close", paths
        start_body = _Handler.captured[0][1]
        assert start_body["new_turn"] is True
        assert os.path.isfile(start_body["prompt_audio"]), start_body
        push_body = _Handler.captured[1][1]
        assert push_body["is_final"] is True and push_body["text"] == "你好。"
    finally:
        srv.shutdown()
    print("mossrt native session protocol: OK")


def test_mossrt_streaming_one_session_per_turn() -> None:
    """Item 4: a whole turn rides ONE MOSS-TTS-Realtime session — segments are
    pushed as they arrive and audio streams continuously (no per-segment cold
    start). Drives TtsSession streaming against a fake /tts/session/* server."""
    import queue as _queue
    import time
    from http.server import ThreadingHTTPServer
    from server.voice.tts_session import TtsSession
    from server.adapters.tts.moss_tts_realtime.adapter import MossRtNativeEngine

    sessions: dict = {}
    calls = {"start": 0, "push": 0, "final": 0}

    class Handler(BaseHTTPRequestHandler):
        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):  # noqa: N802
            b = self._body()
            if self.path == "/tts/session/start":
                sessions[b["session_id"]] = _queue.Queue(); calls["start"] += 1
            elif self.path == "/tts/session/push":
                q = sessions.get(b["session_id"])
                if q is not None:
                    if (b.get("text") or "").strip():
                        calls["push"] += 1; q.put(b"\x11\x22" * 2000)
                    if b.get("is_final"):
                        calls["final"] += 1; q.put(None)
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

        def do_GET(self):  # noqa: N802
            q = sessions.get(self.path.split("/")[-2])
            self.send_response(200)
            self.send_header("X-Audio-Sample-Rate", "24000")
            self.send_header("X-Audio-Channels", "1")
            self.end_headers()
            while True:
                item = q.get()
                if item is None:
                    break
                try:
                    self.wfile.write(item); self.wfile.flush()
                except Exception:
                    break

        def log_message(self, *a):  # noqa: A002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng = MossRtNativeEngine(Settings(), f"http://127.0.0.1:{srv.server_port}")
        eng.ready = True
        assert eng.supports_streaming
        events: list = []
        sess = TtsSession(eng, "t", emit=lambda e: events.append(e))
        sess.set_voice("Yuewen")
        sess.start_turn("t1")
        for seg in ("第一段。", "第二段。", "第三段。"):
            sess.feed_segment(seg, "t1"); time.sleep(0.03)
        sess.end_turn()
        deadline = time.time() + 5
        while time.time() < deadline and not any(e["type"] == "tts_turn_end" for e in events):
            time.sleep(0.03)
        types = [e["type"] for e in events]
        total = sum(len(e["pcm"]) for e in events if e["type"] == "tts_audio_chunk")
        assert calls["start"] == 1, f"one session per turn, got {calls['start']}"
        assert calls["push"] == 3 and calls["final"] == 1, calls
        assert total == 3 * 4000, total
        assert types[0] == "tts_turn_start" and types[-1] == "tts_turn_end", types
        # barge-in mid-stream: abort, no spurious turn_end
        events.clear()
        sess.start_turn("t2"); sess.feed_segment("被打断……", "t2")
        time.sleep(0.08); sess.cancel_turn(); time.sleep(0.15)
        t2 = [e["type"] for e in events]
        assert "tts_turn_abort" in t2 and "tts_turn_end" not in t2, t2
        sess.close()
    finally:
        srv.shutdown()
    print("mossrt streaming (one session/turn + barge-in): OK")


def test_offline_tts_segmentation() -> None:
    """P2: /api/tts streams clause-by-clause (first-audio ~constant, not linear
    in length). Verify the splitter yields multiple small units with a short
    leading unit and bounded unit length."""
    from server.routers.speech import _segments_for_tts

    text = ("你好，我现在看到画面里有一个人正在桌子前面认真地工作着，"
            "他的桌上摆放着电脑键盘鼠标以及一些文件资料。背景是一间明亮的办公室。")
    units = _segments_for_tts(text, first_clause_chars=16, max_chars=40)
    assert len(units) >= 3, units                       # not one giant blob
    assert "".join(units) == text, ("units must reconstruct the text", units)
    assert len(units[0]) <= 32, ("fast first cut", units[0])
    assert all(len(u) <= 40 for u in units), ("units bounded by max_chars", units)
    # a short single clause stays one unit (no spurious splitting)
    assert _segments_for_tts("你好。", 16, 40) == ["你好。"]
    # empty / whitespace → nothing to synthesize
    assert _segments_for_tts("   ", 16, 40) == []
    print("offline tts segmentation: OK")


def main() -> int:
    test_registry_builds_all_providers()
    test_provider_classification()
    test_placement_sizing()
    test_cosyvoice3_speech_payload()
    test_mossrt_native_session_protocol()
    test_mossrt_streaming_one_session_per_turn()
    test_offline_tts_segmentation()
    print("TTS PROVIDERS TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
