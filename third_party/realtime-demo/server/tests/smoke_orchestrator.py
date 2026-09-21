"""B3 smoke: scripted no-WS orchestrator turns against fake engines.

Run:  <repo>/.venv/bin/python -m server.tests.smoke_orchestrator

Covers: PTT voice turn (transcription.done → response.created → text.delta(s) →
audio.delta(s) → response.done, with audio binary), frame attach, text.input turn,
client cancel, and PTT barge-in of an in-flight response.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from server import protocol as p
from server.config import Settings
from server.schemas import SessionConfig
from server.session.manager import SessionManager
from server.session.orchestrator import EngineSet
from server.tests.fakes import (
    DEFAULT_REPLY,
    FakeAsrAdapter,
    FakeTtsEngine,
    FakeVlmSession,
    loud_pcm_chunks,
)
from server.voice.tts_session import TtsSession


def make_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(), **overrides)


def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), (200, 30, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def drain_until(state, want_type: str, timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Drain the session out-queue until `want_type` is seen (or timeout)."""
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
        ev["_binary"] = item.binary
        events.append(ev)
        if ev["type"] == want_type:
            return events


def types_of(events: List[Dict[str, Any]], *skip: str) -> List[str]:
    return [e["type"] for e in events if e["type"] not in skip]


def first_index(events: List[Dict[str, Any]], type_: str) -> int:
    for i, e in enumerate(events):
        if e["type"] == type_:
            return i
    return -1


async def build(reply: str = DEFAULT_REPLY, token_interval: float = 0.0,
                **settings_overrides) -> Tuple[SessionManager, Any, FakeVlmSession, FakeTtsEngine, FakeAsrAdapter]:
    settings = make_settings(session_grace_seconds=60.0, **settings_overrides)
    manager = SessionManager(settings)
    vlm = FakeVlmSession(reply=reply, token_interval=token_interval)
    asr = FakeAsrAdapter()
    tts_engine = FakeTtsEngine()

    async def factory() -> EngineSet:
        tts = TtsSession(tts_engine, "smoke", lambda payload: None)
        return EngineSet(vlm=vlm, asr=asr, tts=tts)

    state = await manager.create(SessionConfig(), factory)
    return manager, state, vlm, tts_engine, asr


async def test_voice_turn() -> None:
    manager, state, vlm, tts_engine, asr = await build()
    orch = state.orchestrator

    # frame arrives before the turn → attached to the prompt
    await orch.push_frame(jpeg_bytes(), 1.25)

    await orch.handle_event(p.CLIENT_AUDIO_START, {})
    for chunk in loud_pcm_chunks(4):
        orch.push_pcm(chunk)
    await orch.handle_event(p.CLIENT_AUDIO_COMMIT, {})

    events = await drain_until(state, p.RESPONSE_DONE)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), f"seq must be monotonic: {seqs}"

    core = types_of(events, p.STATUS, p.SPEECH_STARTED)
    # ordering of the §B3 verify sequence
    i_td = first_index(events, p.TRANSCRIPTION_DONE)
    i_rc = first_index(events, p.RESPONSE_CREATED)
    i_delta = first_index(events, p.RESPONSE_TEXT_DELTA)
    i_audio = first_index(events, p.RESPONSE_AUDIO_DELTA)
    i_done = first_index(events, p.RESPONSE_DONE)
    assert 0 <= i_td < i_rc < i_delta < i_audio < i_done, core

    td = events[i_td]
    assert td["text"] == asr.text and td["item_id"].startswith("item_")

    full_text = "".join(e["delta"] for e in events if e["type"] == p.RESPONSE_TEXT_DELTA)
    assert full_text == DEFAULT_REPLY, full_text
    text_done = next(e for e in events if e["type"] == p.RESPONSE_TEXT_DONE)
    assert text_done["text"] == DEFAULT_REPLY

    audio_events = [e for e in events if e["type"] == p.RESPONSE_AUDIO_DELTA]
    assert audio_events and all(e["_binary"] is not None for e in audio_events)
    tag, ts, payload = p.parse_binary(audio_events[0]["_binary"])
    assert tag == p.TAG_TTS_PCM and len(payload) == audio_events[0]["pcm_bytes"]
    assert first_index(events, p.RESPONSE_AUDIO_DONE) < i_done

    done = events[i_done]
    assert done["stop_reason"] == p.STOP_END_TURN

    # control tokens must never leak into captions
    assert "<|" not in full_text and "<|" not in text_done["text"]

    # the prompt reached the VLM with the fresh frame attached
    assert vlm.prompt_frames and vlm.prompt_frames[0][0] == asr.text
    assert vlm.prompt_frames[0][2] == 1.25

    # no stray extra responses
    assert sum(1 for e in events if e["type"] == p.RESPONSE_CREATED) == 1

    m = orch.metrics
    assert m["turns"] == 1 and m["responses"] == 1
    assert m["asr_ms"] is not None and m["vlm_ttft_ms"] is not None and m["tts_ttfa_ms"] is not None

    await manager.aclose()
    print("voice turn: OK  (tts units:", len(tts_engine.segments), "audio chunks:", len(audio_events), ")")


async def test_empty_ptt_press() -> None:
    """A too-short PTT press yields an empty transcription and no model turn."""
    manager, state, vlm, _, _ = await build()
    orch = state.orchestrator
    await orch.handle_event(p.CLIENT_AUDIO_START, {})
    orch.push_pcm(b"\x00\x00" * 160)  # 20 ms of silence
    await orch.handle_event(p.CLIENT_AUDIO_COMMIT, {})
    events = await drain_until(state, p.TRANSCRIPTION_DONE)
    assert events[-1]["text"] == ""
    await asyncio.sleep(0.3)
    assert not vlm.prompts, "empty press must not reach the VLM"
    await manager.aclose()
    print("empty PTT press: OK")


async def test_text_input_turn() -> None:
    manager, state, vlm, _, _ = await build()
    orch = state.orchestrator
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "介绍一下你自己"})
    events = await drain_until(state, p.RESPONSE_DONE)
    assert first_index(events, p.TRANSCRIPTION_DONE) == -1, "typed turns emit no transcription"
    assert vlm.prompts == ["介绍一下你自己"]
    full = "".join(e["delta"] for e in events if e["type"] == p.RESPONSE_TEXT_DELTA)
    assert full == DEFAULT_REPLY
    await manager.aclose()
    print("text.input turn: OK")


async def test_cancel_and_barge_in() -> None:
    # slow token stream so the response is in flight when we interrupt
    manager, state, vlm, _, _ = await build(token_interval=0.05)
    orch = state.orchestrator

    # 1) client-driven cancel
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "第一个问题"})
    await drain_until(state, p.RESPONSE_TEXT_DELTA)
    await orch.handle_event(p.CLIENT_RESPONSE_CANCEL, {})
    events = await drain_until(state, p.RESPONSE_DONE)
    done = events[-1]
    assert done["stop_reason"] == p.STOP_CANCELLED
    assert vlm.interrupts >= 1, "cancel must soft-interrupt the VLM"

    # swallow the injected <|eot_id|> tail (no new response may appear from it)
    await asyncio.sleep(0.5)

    # 2) PTT press barge-in over a fresh in-flight response
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "第二个问题"})
    await drain_until(state, p.RESPONSE_TEXT_DELTA)
    await orch.handle_event(p.CLIENT_AUDIO_START, {})  # user starts speaking
    events = await drain_until(state, p.RESPONSE_DONE, timeout=4.0)
    assert first_index(events, p.SPEECH_STARTED) >= 0
    assert events[-1]["stop_reason"] == p.STOP_INTERRUPTED
    assert vlm.interrupts >= 2

    # complete the pressed turn → a fresh full response still works after barge-in
    for chunk in loud_pcm_chunks(4):
        orch.push_pcm(chunk)
    await orch.handle_event(p.CLIENT_AUDIO_COMMIT, {})
    events = await drain_until(state, p.RESPONSE_DONE, timeout=10.0)
    assert events[-1]["stop_reason"] == p.STOP_END_TURN
    full = "".join(e["delta"] for e in events if e["type"] == p.RESPONSE_TEXT_DELTA)
    assert full == DEFAULT_REPLY, full

    await manager.aclose()
    print("cancel + barge-in: OK")


async def test_narration_rounds_and_streaming_frames() -> None:
    """Board-parity: <|silence|> closes each spontaneous round (bubble split,
    token never surfaced) and frames keep streaming while a round is open."""
    manager, state, vlm, _, _ = await build()
    orch = state.orchestrator

    # two unprompted narration rounds → two distinct response created/done pairs
    vlm.narrate("镜头前出现了一只猫。")
    first = await drain_until(state, p.RESPONSE_DONE)
    vlm.narrate("猫跳上了桌子。")
    second = await drain_until(state, p.RESPONSE_DONE)

    events = first + second
    created = [e for e in events if e["type"] == p.RESPONSE_CREATED]
    done = [e for e in events if e["type"] == p.RESPONSE_DONE]
    assert len(created) == 2 and len(done) == 2, [e["type"] for e in events]
    assert created[0]["response_id"] != created[1]["response_id"]
    assert all(e["stop_reason"] == p.STOP_END_TURN for e in done)
    texts = [e for e in events if e["type"] in (p.RESPONSE_TEXT_DELTA, p.RESPONSE_TEXT_DONE)]
    joined = "".join(str(e.get("delta") or e.get("text") or "") for e in texts)
    assert "<|silence|>" not in joined and "<|" not in joined, joined
    round1 = next(e["text"] for e in first if e["type"] == p.RESPONSE_TEXT_DONE)
    round2 = next(e["text"] for e in second if e["type"] == p.RESPONSE_TEXT_DONE)
    assert round1 == "镜头前出现了一只猫。" and round2 == "猫跳上了桌子。"

    # frames are NOT held while a round is generating: open a slow round, then
    # push a frame mid-generation — it must reach the model immediately
    vlm.token_interval = 0.05
    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "描述一下"})
    await drain_until(state, p.RESPONSE_CREATED)
    assert not vlm.frames, "no pure frame should have reached the model yet"
    await orch.push_frame(jpeg_bytes(), 3.0)
    assert len(vlm.frames) == 1 and vlm.frames[0][1] == 3.0, vlm.frames
    assert orch.metrics["frames_forwarded"] == 1
    assert "frames_skipped_busy" not in orch.metrics
    await drain_until(state, p.RESPONSE_DONE)

    await manager.aclose()
    print("narration rounds split on silence + frames stream while open: OK")


async def test_video_attach_and_media_ts() -> None:
    """File-streaming extras: video.attach round-trip + media_ts turn stamps."""
    manager, state, _vlm, _, _ = await build()
    orch = state.orchestrator

    # frames carry the video position; the newest one stamps subsequent turns
    await orch.push_frame(jpeg_bytes(), 7.5)

    await orch.handle_event(p.CLIENT_VIDEO_ATTACH,
                            {"media": "sha256:" + "ab" * 32, "name": "clip.mp4",
                             "duration_s": 12.5})
    events = await drain_until(state, p.VIDEO_ATTACHED)
    attached = events[-1]
    assert attached["media"] == "sha256:" + "ab" * 32
    assert attached["name"] == "clip.mp4" and attached["duration_s"] == 12.5

    await orch.handle_event(p.CLIENT_TEXT_INPUT, {"text": "这一段发生了什么"})
    events = await drain_until(state, p.RESPONSE_DONE)
    text_done = next(e for e in events if e["type"] == p.TEXT_DONE)
    assert text_done["media_ts"] == 7.5, text_done
    assert events[-1]["media_ts"] == 7.5, events[-1]

    # malformed handles are rejected, never journaled
    await orch.handle_event(p.CLIENT_VIDEO_ATTACH, {"media": "data:video/mp4;base64,xxx"})
    events = await drain_until(state, p.ERROR)
    assert events[-1]["code"] == "bad_request"

    await manager.aclose()
    print("video.attach + media_ts: OK")


async def test_backpressure_hooks() -> None:
    manager, state, *_ = await build(audio_buffer_high_s=0.5, audio_buffer_low_s=0.2,
                                     tts_max_pending_units=2, tts_unit_max_age_s=30.0)
    orch = state.orchestrator

    # client-reported playback backlog trips the gate, drains, then re-opens
    orch._client_buffer = (5.0, asyncio.get_event_loop().time())
    import time as _t
    orch._client_buffer = (5.0, _t.monotonic())
    assert orch.audio_queue_seconds() > 0.5
    assert orch.should_emit_next_unit() is False, "gate must pause above high-water"
    orch._client_buffer = (0.0, _t.monotonic())
    assert orch.should_emit_next_unit() is True, "gate must resume below low-water"

    # backlog beyond max_pending drops the oldest units
    from server.session.orchestrator import ResponseCtx, TtsUnit
    r = ResponseCtx(response_id="resp_x", t0=_t.monotonic())
    orch._response = r
    for i in range(5):
        orch._units.append(TtsUnit("segment", "resp_x", f"unit-{i}"))
    orch.drop_stale()
    kept = [u.text for u in orch._units]
    assert kept == ["unit-3", "unit-4"], kept
    assert orch.metrics["units_dropped"] == 3
    orch._response = None
    orch._units.clear()

    await manager.aclose()
    print("back-pressure hooks: OK")


def main() -> int:
    asyncio.run(test_voice_turn())
    asyncio.run(test_empty_ptt_press())
    asyncio.run(test_text_input_turn())
    asyncio.run(test_cancel_and_barge_in())
    asyncio.run(test_narration_rounds_and_streaming_frames())
    asyncio.run(test_video_attach_and_media_ts())
    asyncio.run(test_backpressure_hooks())
    print("\nORCHESTRATOR SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
