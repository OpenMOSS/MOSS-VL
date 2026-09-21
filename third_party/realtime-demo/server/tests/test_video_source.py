"""`input.video.source` tests: segment record, media_ts derivation, the
model-facing timeline note (VLM text only — transcript stays clean), the
journaled `input.video.source.changed` bubble row, and validation errors.

Run:  <repo>/.venv/bin/python -m server.tests.test_video_source

No GPU, no network — fake engines, temp DATA_DIR.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import tempfile
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image

from server import protocol as p
from server.config import Settings
from server.persistence import HistoryRecorder, IndexStore
from server.schemas import SessionConfig
from server.session.manager import SessionManager
from server.session.orchestrator import EngineSet
from server.tests.fakes import FakeAsrAdapter, FakeTtsEngine, FakeVlmSession
from server.voice.tts_session import TtsSession


def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), (30, 200, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def drain_until(state, want_type: str, timeout: float = 8.0) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise AssertionError(
                f"timed out waiting for {want_type}; got: {[e['type'] for e in events]}")
        try:
            item = await asyncio.wait_for(state.out_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            continue
        ev = json.loads(item.text)
        events.append(ev)
        if ev["type"] == want_type:
            return events


def first(events: List[Dict[str, Any]], type_: str) -> Dict[str, Any]:
    for e in events:
        if e["type"] == type_:
            return e
    raise AssertionError(f"no {type_} in {[e['type'] for e in events]}")


def test_protocol_registration() -> None:
    assert p.CLIENT_VIDEO_SOURCE == "input.video.source"
    assert p.CLIENT_VIDEO_SOURCE in p.CLIENT_EVENT_TYPES
    assert p.VIDEO_SOURCE_CHANGED == "input.video.source.changed"
    print("protocol registration: OK")


async def test_source_change_flow(data_dir: str) -> None:
    settings = dataclasses.replace(
        Settings(), data_dir=data_dir, history_db_path="", session_grace_seconds=60.0)
    index = IndexStore(settings)
    index.open()
    recorder = HistoryRecorder(settings, index)
    recorder.open()
    manager = SessionManager(settings, history=recorder)
    vlm = FakeVlmSession()
    asr = FakeAsrAdapter()
    tts_engine = FakeTtsEngine()

    async def factory() -> EngineSet:
        return EngineSet(vlm=vlm, asr=asr,
                         tts=TtsSession(tts_engine, "vsrc", lambda payload: None))

    state = await manager.create(SessionConfig(video_source="camera"), factory)
    cid = state.session_id
    orch = state.orchestrator

    # ---- file segment announced at session time 132s ----
    await orch.handle_event(p.CLIENT_VIDEO_SOURCE, {
        "kind": "file", "name": "clip.mp4", "duration_s": 200.0,
        "session_ts_start": 132.0,
    })
    events = await drain_until(state, p.VIDEO_SOURCE_CHANGED)
    changed = first(events, p.VIDEO_SOURCE_CHANGED)
    assert changed["kind"] == "file" and changed["session_ts_start"] == 132.0
    assert changed["name"] == "clip.mp4" and changed["duration_s"] == 200.0

    # frame stamped with the monotone SESSION clock (132s + 3.5s into the file)
    await orch.push_frame(jpeg_bytes(), 135.5)

    # ---- turn 1: note rides the VLM text; transcript stays clean ----
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "what happens here"})
    events = await drain_until(state, p.RESPONSE_DONE)
    td = first(events, p.TEXT_DONE)
    assert td["text"] == "what happens here", td
    assert td["media_ts"] == 3.5, f"file position must derive from the segment: {td}"

    assert vlm.prompts, "the fake VLM saw no prompt"
    sent = vlm.prompts[-1]
    assert sent.startswith("[Video source changed: video file 'clip.mp4'"), sent
    assert "session time 132.0s" in sent and sent.endswith("what happens here"), sent

    # ---- turn 2: the note was drained — no repeat ----
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "and now?"})
    await drain_until(state, p.RESPONSE_DONE)
    assert vlm.prompts[-1] == "and now?", vlm.prompts[-1]

    # ---- swap back to camera: turns stop carrying seek anchors ----
    await orch.handle_event(p.CLIENT_VIDEO_SOURCE, {
        "kind": "camera", "session_ts_start": 150.0,
    })
    await drain_until(state, p.VIDEO_SOURCE_CHANGED)
    await orch.push_frame(jpeg_bytes(), 152.0)
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "look at me"})
    events = await drain_until(state, p.RESPONSE_DONE)
    td = first(events, p.TEXT_DONE)
    assert "media_ts" not in td, f"camera-segment turns must carry no seek anchor: {td}"
    assert vlm.prompts[-1].startswith("[Video source changed: a live camera feed"), vlm.prompts[-1]

    # ---- validation errors ----
    await orch.handle_event(p.CLIENT_VIDEO_SOURCE, {"kind": "hologram", "session_ts_start": 0})
    err = first(await drain_until(state, p.ERROR), p.ERROR)
    assert err["code"] == "bad_request", err
    await orch.handle_event(p.CLIENT_VIDEO_SOURCE, {"kind": "file"})
    err = first(await drain_until(state, p.ERROR), p.ERROR)
    assert err["code"] == "bad_request", err

    await manager.close(cid, reason="client")
    recorder.close()

    # ---- history: dedicated source-change rows, clean turn rows ----
    turns = index.get_transcript(cid)
    sources = [t["source"] for t in turns]
    assert sources.count("source_change") == 2, sources
    sc = next(t for t in turns if t["source"] == "source_change")
    m = json.loads(sc["metrics_json"])
    assert m["kind"] == "file" and m["session_ts_start"] == 132.0 and m["name"] == "clip.mp4", m
    typed = next(t for t in turns if t["source"] == "typed")
    assert typed["text"] == "what happens here", "journal must keep the CLEAN text"
    assert json.loads(typed["metrics_json"])["media_ts"] == 3.5
    index.close()
    print("source change flow (segment, note, media_ts, history rows): OK")


def main() -> int:
    test_protocol_registration()
    with tempfile.TemporaryDirectory(prefix="vsrc_") as d:
        asyncio.run(test_source_change_flow(d))
    print("\nVIDEO SOURCE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
