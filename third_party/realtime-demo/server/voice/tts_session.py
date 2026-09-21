"""Per-session TTS worker — synthesizes queued text units into PCM chunks.

Adapted from board voice_runtime.py TtsSession, reduced to the session
orchestrator's contract: the orchestrator owns segmentation and the
back-pressure/drop-stale policy (asr-tts_research.md §1C) and hands over
ready-cut units via `feed_segment`; this class owns the worker thread, the
engine streaming, and turn_id-guarded cancellation so a barge-in abandons
in-flight audio immediately.

Emits dict events to `emit` (rebound by the orchestrator, called from the
worker thread): tts_turn_start · tts_audio_chunk{pcm: bytes} · tts_turn_end ·
tts_turn_abort · tts_error. Audio is raw PCM bytes — the session WS forwards
it as binary frames (protocol tag 0x11).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from ..logging_conf import get_logger

log = get_logger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]

# Voice values that mean "no speech" — the session runs captions-only (the UI
# offers this as the "(none)" / "（无）" voice option to disable TTS).
MUTE_VOICES = {"none", "(none)", "（无）", "无", "off", "mute", "silent"}


def is_muted_voice(voice: Optional[str]) -> bool:
    v = (voice or "").strip()
    return v.lower() in MUTE_VOICES or v in MUTE_VOICES


class TtsSession:
    def __init__(self, engine: Any, client_id: str, emit: EmitFn,
                 on_close: Optional[Callable[[], None]] = None):
        self.engine = engine
        self.client_id = client_id
        self.emit = emit
        # fired once from close() — the TTS sidecar pool uses it to release
        # this session's engine lease (least-loaded routing)
        self._on_close = on_close
        self.turn_id: Optional[str] = None
        self.voice: Optional[str] = None
        self.muted: bool = False  # voice "(none)"/"（无）" → captions only, no synth
        # streaming mode (item 4): if the engine can consume incremental text
        # (MOSS-TTS-Realtime native session API), a whole turn rides ONE warm
        # session — segments are pushed as they arrive and audio flows
        # continuously, with no per-segment cold start or prosody reset.
        self._streaming = bool(getattr(engine, "supports_streaming", False))
        self._stream: Any = None
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._cancel = threading.Event()
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name=f"tts-{client_id}", daemon=True)
        self._worker.start()

    def set_voice(self, voice: str) -> None:
        self.muted = is_muted_voice(voice)
        self.voice = None if self.muted else (voice or None)

    def start_turn(self, turn_id: str) -> None:
        self.cancel_turn()
        self._cancel.clear()
        self.turn_id = turn_id
        self.emit({
            "type": "tts_turn_start",
            "turn_id": turn_id,
            "sample_rate": self.engine.sample_rate,
            "channels": self.engine.channels,
        })
        if self._streaming:
            # open the turn's continuous session up front so the first pushed
            # segment starts producing audio immediately
            self._queue.put({"type": "stream_open", "turn_id": turn_id, "voice": self.voice})

    def feed_segment(self, text: str, turn_id: Optional[str] = None) -> None:
        """Enqueue one ready-cut text unit for synthesis."""
        if self.muted:
            return  # "(none)" voice: no synthesis (captions still stream)
        if turn_id and turn_id != self.turn_id:
            self.start_turn(turn_id)
        if not self.turn_id:
            self.start_turn(turn_id or f"tts-{int(time.time() * 1000)}")
        text = (text or "").strip()
        if not text:
            return
        # streaming: push the delta into the open turn session; else one job per segment
        kind = "push" if self._streaming else "segment"
        self._queue.put({"type": kind, "turn_id": self.turn_id, "text": text, "voice": self.voice})

    def end_turn(self) -> None:
        """Mark the turn complete: tts_turn_end is emitted after the last unit plays out."""
        if self.turn_id:
            self._queue.put({"type": "stream_final" if self._streaming else "end",
                             "turn_id": self.turn_id})

    def cancel_turn(self) -> None:
        if self.turn_id:
            self._cancel.set()
            self.emit({"type": "tts_turn_abort", "turn_id": self.turn_id})
        self.turn_id = None
        # barge-in: tear down any in-flight streaming session immediately
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.cancel_turn()
        self._closed.set()
        self._queue.put(None)
        self._worker.join(timeout=1.0)
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            try:
                on_close()
            except Exception:  # noqa: BLE001
                pass

    def _worker_loop(self) -> None:
        while not self._closed.is_set():
            job = self._queue.get()
            if job is None:
                return
            turn_id = job.get("turn_id")
            jtype = job.get("type")
            if self._cancel.is_set() or turn_id != self.turn_id:
                continue
            try:
                if jtype == "stream_open":
                    self._open_stream(turn_id, job.get("voice"))
                elif jtype == "push":
                    if self._stream is not None:
                        self._stream.push_text(str(job.get("text") or ""))
                elif jtype == "stream_final":
                    if self._stream is not None:
                        self._stream.end_input()  # emit loop drains to EOF → tts_turn_end
                elif jtype == "end":
                    self.emit({"type": "tts_turn_end", "turn_id": turn_id})
                elif jtype == "segment":
                    self._synthesize_segment(turn_id, str(job.get("text") or "").strip(), job.get("voice"))
            except Exception as exc:  # noqa: BLE001
                log.warning("TTS %s failed for %s: %s", jtype, turn_id, exc)
                self.emit({"type": "tts_error", "turn_id": turn_id, "message": str(exc)})

    def _synthesize_segment(self, turn_id: str, text: str, voice: Optional[str]) -> None:
        if not text:
            return
        for pcm in self.engine.synthesize_pcm(text, voice=voice):
            if self._cancel.is_set() or turn_id != self.turn_id:
                break
            self.emit({
                "type": "tts_audio_chunk",
                "turn_id": turn_id,
                "sample_rate": self.engine.sample_rate,
                "channels": self.engine.channels,
                "pcm": pcm,
            })

    # ---- streaming mode (item 4) --------------------------------------------

    def _open_stream(self, turn_id: str, voice: Optional[str]) -> None:
        stream = self.engine.open_stream(voice)
        self._stream = stream
        threading.Thread(target=self._stream_emit_loop, args=(stream, turn_id),
                         name=f"tts-emit-{self.client_id}", daemon=True).start()

    def _stream_emit_loop(self, stream: Any, turn_id: str) -> None:
        """Drain the turn's continuous audio → tts_audio_chunk events; when the
        stream reaches EOF (after end_input) emit tts_turn_end. A barge-in
        closes the stream, so this exits without a spurious turn-end."""
        try:
            for pcm in stream.audio_chunks():
                if self._cancel.is_set() or turn_id != self.turn_id:
                    break
                self.emit({
                    "type": "tts_audio_chunk",
                    "turn_id": turn_id,
                    "sample_rate": stream.sample_rate,
                    "channels": stream.channels,
                    "pcm": pcm,
                })
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS stream emit failed for %s: %s", turn_id, exc)
        finally:
            if not self._cancel.is_set() and turn_id == self.turn_id:
                self.emit({"type": "tts_turn_end", "turn_id": turn_id})
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
