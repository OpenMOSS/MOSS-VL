#!/usr/bin/env python
"""E2E probe for the session plane against a LIVE backend (backend_overhaul.md §6).

Creates a session over REST, attaches the WS, optionally streams camera frames,
runs one turn (typed text or a WAV pushed as PTT mic audio), prints the caption
stream + client-measured latencies, and saves the received TTS PCM to a WAV.

Examples (server on :8000 with the realtime model loaded):
  .venv/bin/python scripts/e2e_session.py --text "画面里发生了什么？" --image tests/red.jpg
  .venv/bin/python scripts/e2e_session.py --wav /path/to/question_16k.wav --frames 5 --fps 1
  .venv/bin/python scripts/e2e_session.py --base http://127.0.0.1:8000 --text 你好 --out reply.wav

The WAV is converted to PCM16 mono 16 kHz automatically (stdlib audioop).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")  # repo root when run as scripts/e2e_session.py

import websockets  # noqa: E402

from server import protocol as p  # noqa: E402

try:
    import audioop  # py3.12 stdlib
except ImportError:  # pragma: no cover
    audioop = None

CHUNK_MS = 160
ASR_SR = 16000


def http(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def load_wav_pcm16_mono_16k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if audioop is None:
        assert (sr, ch, width) == (ASR_SR, 1, 2), "need 16k mono PCM16 WAV (audioop unavailable)"
        return raw
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
    if ch == 2:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    elif ch != 1:
        raise SystemExit(f"unsupported channel count: {ch}")
    if sr != ASR_SR:
        raw, _ = audioop.ratecv(raw, 2, 1, sr, ASR_SR, None)
    return raw


def make_jpeg(path: Optional[str]) -> bytes:
    if path:
        with open(path, "rb") as f:
            return f.read()
    # photo-like synthetic frame: gradient + noise + shapes. Flat solid-color
    # frames at some resolutions hit an under-tested vision-padding branch in
    # the streaming model (see VISION_SEQ_PAD_MULTIPLE in server/config.py) —
    # real camera frames are fine, so the probe frame should look like one.
    import random

    from PIL import Image, ImageDraw

    random.seed(7)
    img = Image.new("RGB", (448, 336))
    px = img.load()
    for y in range(336):
        for x in range(448):
            px[x, y] = (
                min(255, 40 + x // 3 + random.randint(0, 24)),
                min(255, 30 + y // 3 + random.randint(0, 24)),
                min(255, 90 + random.randint(0, 24)),
            )
    d = ImageDraw.Draw(img)
    d.ellipse([150, 90, 300, 240], fill=(210, 60, 50), outline=(255, 255, 255), width=4)
    d.rectangle([40, 250, 170, 310], fill=(60, 160, 90))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class Wire:
    def __init__(self, ws, verbose: bool):
        self.ws = ws
        self.verbose = verbose
        self.events: List[Dict[str, Any]] = []
        self.pcm = bytearray()
        self.sample_rate = 48000
        self.channels = 1
        self.first_delta_t: Optional[float] = None
        self.first_audio_t: Optional[float] = None
        self.turn_t0: Optional[float] = None
        # Server-seq anchor: responses created (seq) after it are OUR turn's.
        # The model opens spontaneous narration rounds (session start, frame
        # changes) — the probe must not print or exit on those. Set by sending
        # a ping right after the turn input and taking the pong's seq.
        self.anchor_seq: Optional[int] = None
        self.own_responses: set = set()

    _pending_audio_rid: Optional[str] = None

    async def next_event(self, timeout: float) -> Dict[str, Any]:
        while True:
            message = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            if isinstance(message, (bytes, bytearray)):
                tag, _, payload = p.parse_binary(message)
                if tag == p.TAG_TTS_PCM and self._pending_audio_rid in self.own_responses:
                    self.pcm.extend(payload)
                    if self.first_audio_t is None:
                        self.first_audio_t = time.monotonic()
                continue
            ev = json.loads(message)
            self.events.append(ev)
            t = ev["type"]
            if (t == p.RESPONSE_CREATED and self.anchor_seq is not None
                    and int(ev.get("seq") or 0) > self.anchor_seq and not self.own_responses):
                # the model is sequential: the FIRST round after our input is the
                # answer; later rounds are spontaneous narration of newer frames
                self.own_responses.add(ev.get("response_id"))
            if t == p.RESPONSE_AUDIO_DELTA:
                self._pending_audio_rid = ev.get("response_id")
                self.sample_rate = int(ev.get("sample_rate") or self.sample_rate)
                self.channels = int(ev.get("channels") or self.channels)
            if t == p.RESPONSE_TEXT_DELTA:
                if ev.get("response_id") not in self.own_responses:
                    if self.verbose:
                        print(f"\n[spontaneous:{ev.get('response_id')}] {ev.get('delta')!r}")
                    continue
                if self.first_delta_t is None:
                    self.first_delta_t = time.monotonic()
                    print("\n[captions] ", end="", flush=True)
                print(ev["delta"], end="", flush=True)
            elif t == p.STATUS:
                if self.verbose:
                    print(f"\n[status] {json.dumps(ev.get('queues'), ensure_ascii=False)}")
            elif self.verbose or t in (p.TRANSCRIPTION_DONE, p.RESPONSE_CREATED,
                                       p.RESPONSE_DONE, p.ERROR):
                fields = {k: v for k, v in ev.items() if k not in ("type", "v")}
                print(f"\n[{t}] {json.dumps(fields, ensure_ascii=False)[:240]}")
            return ev

    def re_anchor(self, seq: int) -> None:
        """Move the own-response anchor (voice turns: the transcription.done seq)."""
        self.anchor_seq = int(seq)
        self.own_responses.clear()
        self.first_delta_t = None
        self.first_audio_t = None
        self.pcm = bytearray()

    async def until(self, type_: str, timeout: float, own_only: bool = False) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"waiting for {type_}; got {[e['type'] for e in self.events[-8:]]}")
            ev = await self.next_event(timeout=remaining)
            if ev["type"] == type_ and (not own_only or ev.get("response_id") in self.own_responses):
                return ev


async def frame_pump(ws, jpeg: bytes, count: int, fps: float, t_start: float) -> None:
    for i in range(count):
        ts_ms = int((time.monotonic() - t_start) * 1000)
        await ws.send(p.video_binary(jpeg, timestamp_ms=ts_ms))
        await asyncio.sleep(1.0 / max(0.1, fps))


async def run(args: argparse.Namespace) -> int:
    base = args.base.rstrip("/")
    code, body = http("GET", f"{base}/api/status")
    if code != 200:
        print(f"backend not reachable at {base} ({code}): {body}")
        return 2
    print(f"[status] vlm loaded={body['vlm'].get('loaded')} asr={body['voice']['asr'].get('message')} "
          f"tts={body['voice']['tts'].get('message')}")

    config: Dict[str, Any] = {"capture_mode": "ptt"}
    if args.config:
        config.update(json.loads(args.config))
    code, body = http("POST", f"{base}/api/sessions", {"config": config})
    if code != 201:
        print(f"create session failed ({code}): {body}")
        return 2
    sid, ws_url = body["session_id"], body["ws_url"]
    print(f"[session] {sid}")

    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
    try:
        async with websockets.connect(f"{ws_base}{ws_url}", max_size=8 * 2**20) as ws:
            wire = Wire(ws, args.verbose)
            created = await wire.until(p.SESSION_CREATED, timeout=10)
            print(f"[created] audio_out={created.get('audio_out')} audio_in={created.get('audio_in')}")

            t_start = time.monotonic()
            pump = None
            if args.frames > 0:
                jpeg = make_jpeg(args.image)
                pump = asyncio.create_task(frame_pump(ws, jpeg, args.frames, args.fps, t_start))
                # let a frame or two land before the turn
                await asyncio.sleep(min(2.0, 1.5 / max(0.1, args.fps)))
            elif args.image:
                await ws.send(p.video_binary(make_jpeg(args.image), timestamp_ms=0))
                await asyncio.sleep(0.2)

            if args.wav:
                pcm = load_wav_pcm16_mono_16k(args.wav)
                chunk_bytes = ASR_SR * 2 * CHUNK_MS // 1000
                print(f"[mic] streaming {len(pcm) / (ASR_SR * 2):.2f}s of PCM as PTT turn")
                await ws.send(json.dumps({"type": "input.audio.start"}))
                for i in range(0, len(pcm), chunk_bytes):
                    await ws.send(p.mic_binary(pcm[i:i + chunk_bytes]))
                    if not args.fast:
                        await asyncio.sleep(CHUNK_MS / 1000)
                wire.turn_t0 = time.monotonic()
                await ws.send(json.dumps({"type": "input.audio.commit"}))
                await ws.send(json.dumps({"type": "ping", "seq": 424242}))
                pong = await wire.until(p.PONG, timeout=10)
                wire.anchor_seq = int(pong.get("seq") or 0)
                td = await wire.until(p.TRANSCRIPTION_DONE, timeout=30)
                asr_ms = (time.monotonic() - wire.turn_t0) * 1000
                print(f"\n[asr] {asr_ms:.0f}ms → {td.get('text')!r}")
                if not td.get("text"):
                    print("[warn] empty transcription — nothing to answer")
                # narration rounds may have slipped in while ASR ran — the answer
                # is the first round created after the transcription landed
                wire.re_anchor(int(td.get("seq") or 0))
            else:
                wire.turn_t0 = time.monotonic()
                await ws.send(json.dumps({"type": "text.input", "text": args.text}))
                await ws.send(json.dumps({"type": "ping", "seq": 424242}))
                pong = await wire.until(p.PONG, timeout=10)
                wire.anchor_seq = int(pong.get("seq") or 0)

            done = await wire.until(p.RESPONSE_DONE, timeout=args.timeout, own_only=True)
            print()

            t0 = wire.turn_t0 or t_start
            ttft = (wire.first_delta_t - t0) * 1000 if wire.first_delta_t else None
            ttfa = (wire.first_audio_t - t0) * 1000 if wire.first_audio_t else None
            audio_s = len(wire.pcm) / max(1, wire.sample_rate * wire.channels * 2)
            print(f"[turn] stop_reason={done.get('stop_reason')}  "
                  f"ttft={ttft and f'{ttft:.0f}ms'}  ttfa={ttfa and f'{ttfa:.0f}ms'}  "
                  f"audio={audio_s:.2f}s @{wire.sample_rate}Hz×{wire.channels}")

            if pump is not None:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)

            if wire.pcm and args.out:
                with wave.open(args.out, "wb") as wf:
                    wf.setnchannels(wire.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(wire.sample_rate)
                    wf.writeframes(bytes(wire.pcm))
                print(f"[audio] saved → {args.out}")
    finally:
        if not args.keep:
            code, _ = http("DELETE", f"{base}/api/sessions/{sid}")
            print(f"[teardown] DELETE → {code}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--text", default="请描述画面中的内容。", help="typed turn (used when --wav absent)")
    ap.add_argument("--wav", default=None, help="WAV file pushed as a PTT mic turn")
    ap.add_argument("--image", default=None, help="JPEG for frames (default: synthetic)")
    ap.add_argument("--frames", type=int, default=3, help="stream N frames before/through the turn")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", default="e2e_reply.wav", help="save received TTS PCM here")
    ap.add_argument("--config", default=None, help="extra session config JSON")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--fast", action="store_true", help="don't pace mic chunks in real time")
    ap.add_argument("--keep", action="store_true", help="don't DELETE the session at the end")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
