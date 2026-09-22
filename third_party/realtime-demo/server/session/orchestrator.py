"""Per-session orchestrator — VAD → ASR → VLM → TTS (backend_overhaul.md §B3).

The server-owned state machine the browser used to stitch across three sockets:

- consumes mic PCM (PTT delimits the turn; `auto` runs the RMS endpointer) and
  finalizes turns through the ASR adapter;
- on a user turn, barge-ins the in-flight response (soft VLM interrupt that keeps
  the KV cache + TTS flush), attaches the freshest camera frame, and feeds the
  prompt to the VLM realtime session;
- drains the VLM output queue, maps control tokens (§2.4) server-side, streams
  clean text as `response.text.delta` captions, and cuts TTS units;
- feeds TTS units through the §1C back-pressure/drop-stale hooks
  (`audio_queue_seconds` / `should_emit_next_unit` / `drop_stale`) and relays PCM as
  `response.audio.delta` + binary frames.

Threading: everything here runs on the event loop; blocking model calls go through
`asyncio.to_thread`; the TTS worker thread re-enters via `call_soon_threadsafe`.
Audio-affecting control events (PTT start/commit, capture-mode switches) travel
through the same queue as mic PCM so they can never overtake buffered audio.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple

from .. import protocol as p
from ..config import Settings
from ..logging_conf import get_logger
from ..device_compat import is_available, current_device, memory_allocated, memory_reserved
from ..memory import inject as mem_inject
from ..memory import session as mem_session
from ..persistence.media import normalize_hash
from ..realtime.session import SILENCE_TOKENS
from ..schemas import SessionConfig
from ..voice.segmenter import Segmenter
from ..voice.vad import ActivityMeter, measure_pcm
from .state import SessionState

log = get_logger(__name__)

ROUND_START = "<|round_start|>"
END_TOKENS = frozenset({"<|round_end|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"})
HIDDEN_TOKENS = frozenset({"<|response|>", "<|assistant|>"})
ERROR_PREFIX = "[ERROR]"
CONTROL_TOKEN_RE = re.compile(r"(<\|[^<>|]{1,32}\|>)")
# rollover cooldown: post-reseat the prefix sits far below the idle threshold,
# so this is only a belt-and-braces guard against a stale text_tokens reading
ROLLOVER_COOLDOWN_S = 30.0
# Qwen-vocab special tokens the realtime SFT reuses as round scaffolding — they
# decode atomically as literal text and must never reach captions or TTS
# (observed live: rounds opening with "</tool_call>").
JUNK_TOKEN_RE = re.compile(r"</?tool_call>|<tool_response>|</tool_response>")

# first-clause boundary chars for the fast first TTS cut (asr-tts_research §1A T4)
CLAUSE_CHARS = "，,、；;：:。！？!?\n"

# fields a live `session.update` may patch (prompts are KV-prefilled at creation)
_UPDATABLE_CONFIG = frozenset(
    {"asr_language", "capture_mode", "vad_sensitivity", "vad_rms_threshold",
     "vad_silence_ms", "tts_voice", "speaking_rate"}
)


def looks_like_jpeg(raw: bytes) -> bool:
    """Cheap SOI-marker check — full decode now happens inside the VLM session
    (worker-side under VLM_DEPLOY=workers), never on the gateway."""
    return len(raw) > 2 and raw[0] == 0xFF and raw[1] == 0xD8


@dataclass
class EngineSet:
    """The engines a session orchestrates. `tts` may be None (voice-out disabled)."""

    vlm: Any                 # VlmRealtimeSession (already started)
    asr: Any                 # AsrAdapter (open_stream) or None
    tts: Any                 # voice.tts_session.TtsSession or None (captions-only)

    def close(self) -> None:
        """Best-effort synchronous teardown (used when create fails mid-way)."""
        if self.tts is not None:
            try:
                self.tts.close()
            except Exception:  # noqa: BLE001
                pass
        if self.vlm is not None:
            try:
                self.vlm.stop(timeout_seconds=5.0)
            except Exception:  # noqa: BLE001
                pass


@dataclass
class ResponseCtx:
    response_id: str
    t0: float                                  # turn start (monotonic) for TTFT/TTFA
    media_ts: Optional[float] = None           # video position (s) when the turn opened
    text: str = ""
    units_emitted: int = 0
    first_text_wall: Optional[float] = None    # epoch s of the first/last model text
    last_text_wall: Optional[float] = None     # (chunk emitted_at — generation time, not delivery)
    finalized: bool = False                    # text side closed
    done: bool = False                         # response.done emitted
    stop_reason: str = p.STOP_END_TURN
    first_delta_at: Optional[float] = None
    audio_started: bool = False
    audio_seconds: float = 0.0
    audio_first_emit: Optional[float] = None


@dataclass
class TtsUnit:
    kind: str                                  # "segment" | "flush"
    response_id: str
    text: str = ""
    created: float = field(default_factory=time.monotonic)


class Orchestrator:
    def __init__(self, state: SessionState, engines: EngineSet, settings: Settings,
                 memory: Any = None, rollover: Any = None,
                 reseat_factory: Optional[Callable[..., Any]] = None):
        self.state = state
        self.engines = engines
        self.settings = settings
        # L2 memory (server/memory/); None = disabled. Every call into it is
        # best-effort: memory must never be able to break or stall a turn.
        self.memory = memory
        # rollover (design §6): the per-session RolloverManager (None = trigger
        # never armed) plus the factory that starts a replacement VLM realtime
        # session with the rebuilt prefix. A missing factory leaves rollover
        # fully inert (tests, samp sessions, partial wiring).
        self._rollover = rollover
        self._reseat_factory = reseat_factory
        self._reseat_in_progress = False
        self._reseat_lock = asyncio.Lock()
        self._reseat_task: Optional[asyncio.Task] = None
        self._context_command: Optional[str] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._context_epoch = 0
        # failure relay (GATEWAY_PLAN P2): a dead VLM transport gets ONE memory
        # -prefix re-seat attempt before the terminal state; reset on every
        # successful reseat so a later death can relay again
        self._failure_relay_attempted = False
        self._last_rollover_at = 0.0
        # worker's exact text-KV token count, cached off the 1 Hz vlm status
        # (None until the worker's token-counter patch reports one)
        self._last_text_tokens: Optional[float] = None
        self._vlm_drain_task: Optional[asyncio.Task] = None
        self._loop = asyncio.get_running_loop()
        self._closed = False
        self._tasks: list[asyncio.Task] = []

        # ---- audio-in lane (PCM chunks + ordered control markers) ----
        self._audio_in: Deque[Any] = deque()
        self._audio_event = asyncio.Event()
        self._audio_in_max = 256  # ~40 s of 160 ms chunks
        self._asr_stream: Any = None
        self._asr_committed = False
        self._capturing = False
        self._meter = self._build_meter()

        # ---- video ----
        # newest client frame as (jpeg bytes, capture ts, arrival monotonic) —
        # kept encoded: decode is the VLM session's job (worker-side)
        self._latest_frame: Optional[Tuple[bytes, Optional[float], float]] = None
        self._vlm_dead = False
        # current video-source segment from `input.video.source` (None until the
        # client announces one — legacy clients never do). Frame timestamps ride
        # one monotone session clock (live_control_overhaul.md §12); file
        # positions derive from the segment's session_ts_start/media_offset.
        self._segment: Optional[Dict[str, Any]] = None
        # model-facing timeline notes buffered per source change; drained into
        # the NEXT user turn's VLM text (never the transcript, never alone — an
        # unaccompanied put_prompt would trigger an unprompted reply)
        self._pending_source_notes: list[str] = []
        # latest user turn awaiting background fact extraction; consumed when
        # the assistant reply that answers it finalizes (memory, design §3)
        self._pending_fact_user_text: Optional[str] = None
        self._memory_tasks: set[asyncio.Task] = set()
        self._close_task: Optional[asyncio.Task] = None

        # ---- response / TTS lanes ----
        # live responses (typically ≤2: one finalized-and-draining-audio + the
        # open one the model is writing). A finished answer must keep speaking
        # while the next narration round streams text — only user barge-in or
        # the drop-stale policy may cut audio short.
        self._responses: Dict[str, ResponseCtx] = {}
        self._response: Optional[ResponseCtx] = None  # the OPEN (latest) one
        # after a mid-round barge-in, drop model output until the injected
        # <|eot_id|> ack (or a fresh <|round_start|>) — the residual tail must
        # not auto-open a ghost response
        self._drop_model_tail = False
        self._pending_turn_t0: Optional[float] = None
        self._serial = itertools.count(1)
        self._segmenter = Segmenter(
            min_chars=settings.tts_seg_min_chars,
            soft_chars=settings.tts_seg_soft_chars,
            max_chars=settings.tts_seg_max_chars,
        )
        self._units: Deque[TtsUnit] = deque()
        self._units_event = asyncio.Event()
        self._tts_turn_id: Optional[str] = None
        self._tts_fed_current = False
        self._bp_paused = False
        self._tts_events: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        # client-reported unplayed audio: (buffered_seconds, monotonic_at)
        self._client_buffer: Tuple[float, float] = (0.0, 0.0)

        self.metrics: Dict[str, Any] = {
            "turns": 0, "responses": 0, "frames_forwarded": 0, "silence_outputs": 0,
            "units_dropped": 0, "asr_ms": None, "vlm_ttft_ms": None, "tts_ttfa_ms": None,
        }

        if engines.tts is not None:
            engines.tts.emit = self._tts_emit_threadsafe

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        for coro, name in (
            (self._audio_loop(), "audio"),
            (self._vlm_drain_loop(), "vlm"),
            (self._tts_pump_loop(), "tts-pump"),
            (self._tts_feeder_loop(), "tts-feed"),
            (self._status_loop(), "status"),
        ):
            task = asyncio.get_running_loop().create_task(
                coro, name=f"orch-{name}-{self.state.session_id[:12]}")
            self._tasks.append(task)
            if name == "audio":
                self._audio_task = task
            if name == "vlm":
                # tracked by name: a rollover re-seat cancels and recreates it
                self._vlm_drain_task = task

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            lifetime = getattr(self.memory, "lifetime", None)
            if lifetime is not None:
                lifetime.seal()
            if self._rollover is not None and hasattr(self._rollover, "seal"):
                self._rollover.seal()
            self._close_task = asyncio.create_task(self._close_impl())
        try:
            await asyncio.shield(self._close_task)
        except Exception:
            self._close_task = None  # cleanup errors remain visible and retryable
            raise

    async def _close_impl(self) -> None:
        tasks = [*self._tasks, *self._memory_tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._memory_tasks.clear()

        stream, self._asr_stream = self._asr_stream, None
        if stream is not None:
            try:
                await asyncio.to_thread(stream.close)
            except Exception:  # noqa: BLE001
                pass
        if self.engines.tts is not None:
            try:
                await asyncio.to_thread(self.engines.tts.close)
            except Exception:  # noqa: BLE001
                pass
        try:
            await asyncio.to_thread(self.engines.vlm.stop, 5.0)
        except Exception:  # noqa: BLE001
            pass
        if self._rollover is not None and hasattr(self._rollover, "close"):
            await asyncio.to_thread(self._rollover.close)
        if self.memory is not None:
            try:
                await asyncio.to_thread(self.memory.close)
            except Exception:
                log.exception("memory cleanup failed for %s", self.state.session_id)
                raise
        self._latest_frame = None
        self._pending_fact_user_text = None
        log.info("orchestrator closed for %s (metrics=%s)", self.state.session_id, self.metrics)

    # ------------------------------------------------------------------ ingress (from the WS router)

    def push_pcm(self, pcm: bytes) -> None:
        if self._closed or self._context_command or not pcm:
            return
        if len(self._audio_in) >= self._audio_in_max:
            # drop the oldest PCM chunk (never a control marker) to bound latency
            for i, item in enumerate(self._audio_in):
                if isinstance(item, (bytes, bytearray)):
                    del self._audio_in[i]
                    break
        self._audio_in.append(pcm)
        self._audio_event.set()

    def push_audio_marker(self, op: str, **fields: Any) -> None:
        """Order-preserving control marker on the audio lane (start/commit/mode)."""
        if self._closed:
            return
        self._audio_in.append({"op": op, **fields})
        self._audio_event.set()

    async def push_frame(self, jpeg: bytes, timestamp: Optional[float]) -> None:
        if self._closed or self._context_command:
            return
        if not looks_like_jpeg(jpeg):
            self._emit_error("bad_frame", "frame payload is not a JPEG")
            return
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                self._emit_error("bad_frame", "timestamp must be finite and non-negative")
                return
            if not math.isfinite(timestamp) or timestamp < 0:
                self._emit_error("bad_frame", "timestamp must be finite and non-negative")
                return
        self._latest_frame = (jpeg, timestamp, time.monotonic())
        if self.memory is not None:
            try:
                # throttled + enqueue-only; real dedup runs on the writer thread
                self.memory.note_frame(jpeg, timestamp,
                                       media_ts=self._media_ts_fields().get("media_ts"))
            except Exception as exc:  # noqa: BLE001
                log.debug("memory note_frame failed: %s", exc)
        if self._vlm_dead or self._reseat_in_progress:
            return
        # Board parity: frames stream to the model continuously — even
        # mid-generation, where the loop splices their vision tokens into the
        # ongoing answer at the next token step. Backpressure is the frame
        # queue's drop-oldest-pure-frame policy; user turns still clear stale
        # queued frames via put_prompt_frame(drop_pending=True).
        try:
            status = await asyncio.to_thread(self.engines.vlm.put_frame, jpeg, timestamp, len(jpeg))
            if not (status or {}).get("frame_dropped"):
                self.metrics["frames_forwarded"] += 1
            if (status or {}).get("context", {}).get("rollover_required"):
                self._maybe_rollover(idle=False)
        except ValueError as exc:
            self._emit_error("bad_frame", str(exc))
        except Exception as exc:  # noqa: BLE001
            self._mark_vlm_dead(f"put_frame failed: {exc}")

    def _media_ts_fields(self) -> Dict[str, Any]:
        """`{"media_ts": <s>}` — the turn's video position, the replay seek anchor.

        Legacy sessions (no `input.video.source` ever received): the newest
        frame's capture ts verbatim — file-streaming clients stamped frames
        with the video position directly. Segment-aware sessions: frames carry
        the monotone session clock, so a FILE position derives as
        `frame_ts - session_ts_start + media_offset`; non-file segments have no
        seekable position (empty). Empty when no frame has arrived.
        """
        latest = self._latest_frame
        if latest is None or latest[1] is None:
            return {}
        frame_ts = float(latest[1])
        seg = self._segment
        if seg is None:
            return {"media_ts": round(frame_ts, 3)}
        if seg["kind"] != "file" or frame_ts < seg["session_ts_start"]:
            # non-file segment, or a stale frame from before this segment
            return {}
        pos = frame_ts - seg["session_ts_start"] + seg.get("media_offset", 0.0)
        return {"media_ts": round(max(0.0, pos), 3)}

    def _attach_video(self, payload: Dict[str, Any]) -> None:
        """`video.attach` — journal the session's source video (an uploaded CAS
        blob) so the history replay can play it back alongside the transcript."""
        media = normalize_hash(str(payload.get("media") or ""))
        if media is None:
            self._emit_error("bad_request", "video.attach requires media='sha256:<hex>'",
                             event_id=payload.get("event_id"))
            return
        fields: Dict[str, Any] = {"media": f"sha256:{media}",
                                  "item_id": f"item_{next(self._serial)}"}
        name = str(payload.get("name") or "").strip()
        if name:
            fields["name"] = name[:120]
        try:
            duration = float(payload.get("duration_s"))
            if duration > 0:
                fields["duration_s"] = round(duration, 3)
        except (TypeError, ValueError):
            pass
        self.state.emit(p.VIDEO_ATTACHED, **fields)

    _SOURCE_KINDS = frozenset({"camera", "screen", "file", "image", "none"})

    def _set_video_source(self, payload: Dict[str, Any]) -> None:
        """`input.video.source` — a (mid-session) source change. Records the
        segment (file positions derive from it in _media_ts_fields), buffers a
        timeline note for the model's next turn, and emits the journaled
        `input.video.source.changed` the transcript renders as its own bubble."""
        kind = str(payload.get("kind") or "").strip().lower()
        if kind not in self._SOURCE_KINDS:
            self._emit_error("bad_request",
                             "input.video.source requires kind=camera|screen|file|image|none",
                             event_id=payload.get("event_id"))
            return
        try:
            ts_start = max(0.0, float(payload.get("session_ts_start")))
        except (TypeError, ValueError):
            self._emit_error("bad_request",
                             "input.video.source requires numeric session_ts_start (seconds)",
                             event_id=payload.get("event_id"))
            return
        seg: Dict[str, Any] = {"kind": kind, "session_ts_start": round(ts_start, 3)}
        name = str(payload.get("name") or "").strip()
        if name:
            seg["name"] = name[:120]
        media = normalize_hash(str(payload.get("media") or ""))
        if media is not None:
            seg["media"] = f"sha256:{media}"
        for key in ("duration_s", "media_offset"):
            try:
                val = float(payload.get(key))
                if val > 0:
                    seg[key] = round(val, 3)
            except (TypeError, ValueError):
                pass
        self._segment = seg
        self._latest_frame = None  # wait for a frame from the newly selected source
        self._pending_source_notes.append(self._source_note(seg))
        del self._pending_source_notes[:-3]  # a swap-spam burst keeps the last 3
        self.state.emit(p.VIDEO_SOURCE_CHANGED, item_id=f"item_{next(self._serial)}", **seg)

    @staticmethod
    def _source_note(seg: Dict[str, Any]) -> str:
        """Model-facing timeline note: frames ride one monotone session clock,
        so tell the model how this segment maps onto it (file position math)."""
        kind, t = seg["kind"], seg["session_ts_start"]
        name = seg.get("name", "")
        if kind == "file":
            dur = seg.get("duration_s")
            length = f", {int(dur // 60)}:{int(dur % 60):02d} long" if dur else ""
            off = seg.get("media_offset", 0.0)
            pos = f"{int(off // 60)}:{int(off % 60):02d}"
            return (f"[Video source changed: video file '{name}'{length} starts playing "
                    f"at session time {t:.1f}s — its position {pos} corresponds to "
                    f"session time {t:.1f}s.]")
        if kind == "image":
            return f"[Video source changed: a still image '{name}' is shown at session time {t:.1f}s.]"
        if kind == "none":
            if name:  # a video that played to its end (vs a feed switched off)
                return (f"[Video source changed: the video '{name}' finished playing "
                        f"at session time {t:.1f}s.]")
            return f"[Video source changed: the video feed stopped at session time {t:.1f}s.]"
        feed = "camera feed" if kind == "camera" else "screen share"
        return f"[Video source changed: a live {feed} starts at session time {t:.1f}s.]"

    async def handle_event(self, type_: str, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        if self._context_command and type_ not in (p.CLIENT_PING, p.CLIENT_PLAYBACK_STATUS):
            self._emit_error("context_busy", "Context command is still running")
            return
        if type_ == p.CLIENT_PING:
            self.state.emit(p.PONG, transient=True, ping_seq=payload.get("seq"))
        elif type_ == p.CLIENT_SESSION_UPDATE:
            self._apply_session_update(payload.get("config") or {})
        elif type_ == p.CLIENT_AUDIO_START:
            await self._on_user_speech_onset()
            self.push_audio_marker("start")
        elif type_ == p.CLIENT_AUDIO_COMMIT:
            self.push_audio_marker("commit")
        elif type_ == p.CLIENT_RESPONSE_CANCEL:
            rid = payload.get("response_id")
            if self._any_response_live() and (not rid or rid in self._responses):
                await self._cancel_response(p.STOP_CANCELLED)
        elif type_ == p.CLIENT_TEXT_INPUT:
            text = str(payload.get("text") or "").strip()
            if text in ("/compact", "/clear"):
                if self._reseat_factory is None or (text == "/compact" and self._rollover is None):
                    self._emit_error("context_unavailable", "Context command is unavailable for this session")
                    return
                self._context_command = text
                self._context_epoch += 1
                self.state.emit(p.CONTEXT_COMMAND, command=text, status="running")
                task = asyncio.create_task(self._run_context_command(text), name="orch-context-command")
                self._tasks.append(task)
                return
            if text:
                # echo the accepted typed turn (replay + history parity with ASR)
                self.state.emit(p.TEXT_DONE, text=text, item_id=f"item_{next(self._serial)}",
                                **self._media_ts_fields())
                await self._user_turn(text)
            else:
                self._emit_error("bad_request", "text.input requires non-empty text",
                                 event_id=payload.get("event_id"))
        elif type_ == p.CLIENT_VIDEO_ATTACH:
            self._attach_video(payload)
        elif type_ == p.CLIENT_VIDEO_SOURCE:
            self._set_video_source(payload)
        elif type_ == p.CLIENT_PLAYBACK_STATUS:
            try:
                self._client_buffer = (max(0.0, float(payload.get("buffered_s", 0.0))), time.monotonic())
            except (TypeError, ValueError):
                pass
        else:
            self._emit_error("unsupported_type", f"unsupported client event: {type_}",
                             event_id=payload.get("event_id"))

    # ------------------------------------------------------------------ config

    def _build_meter(self) -> ActivityMeter:
        cfg, s = self.state.config, self.settings
        rms = cfg.vad_rms_threshold if cfg.vad_rms_threshold is not None else s.vad_rms_threshold
        silence = cfg.vad_silence_ms
        if silence is None and cfg.vad_sensitivity is not None:
            # sensitivity 0..1 → 900..250 ms trailing-silence tail (higher = snappier)
            silence = 900.0 - float(cfg.vad_sensitivity) * (900.0 - 250.0)
        if silence is None:
            silence = s.vad_silence_ms
        return ActivityMeter(int(rms), s.vad_min_speech_ms, float(silence))

    def _apply_session_update(self, patch: Dict[str, Any]) -> None:
        clean = {k: v for k, v in patch.items() if k in _UPDATABLE_CONFIG}
        ignored = sorted(set(patch) - set(clean))
        try:
            self.state.config = self.state.config.model_copy(update=clean)
            # re-validate the merged model so bad live patches are rejected whole
            self.state.config = SessionConfig(**self.state.config.model_dump())
        except Exception as exc:  # noqa: BLE001
            self._emit_error("bad_config", f"session.update rejected: {exc}")
            return
        self._meter = self._build_meter()
        if "capture_mode" in clean:
            self.push_audio_marker("mode", mode=self.state.config.capture_mode)
        if clean.get("tts_voice") and self.engines.tts is not None:
            self.engines.tts.set_voice(str(clean["tts_voice"]))
        self.state.emit(p.SESSION_UPDATED,
                        config=self.state.config.model_dump(exclude_none=True),
                        ignored=ignored or None)

    # ------------------------------------------------------------------ audio lane

    async def _audio_loop(self) -> None:
        while True:
            if not self._audio_in:
                self._audio_event.clear()
                await self._audio_event.wait()
                continue
            item = self._audio_in.popleft()
            try:
                if isinstance(item, dict):
                    await self._handle_audio_marker(item)
                else:
                    await self._handle_pcm_chunk(bytes(item))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("audio loop error for %s: %s", self.state.session_id, exc)
                self._emit_error("asr_error", str(exc))

    async def _handle_audio_marker(self, marker: Dict[str, Any]) -> None:
        op = marker.get("op")
        if op == "start":
            await self._open_asr_stream()
            self._capturing = True
        elif op == "commit":
            await self._finalize_turn(auto=False)
        elif op == "mode":
            # abandon any half-captured turn when the capture mode flips
            await self._abort_capture()

    async def _handle_pcm_chunk(self, pcm: bytes) -> None:
        mode = (self.state.config.capture_mode or self.settings.capture_mode).lower()
        rms, duration_ms = measure_pcm(pcm, self.settings.asr_sample_rate)

        if mode == "auto":
            if self._asr_stream is None:
                if rms < self._meter.rms_threshold:
                    return  # gate stream-open on speech onset
                await self._open_asr_stream()
                self._capturing = True
            was_speech = self._meter.seen_speech
            fired = self._meter.update(rms, duration_ms)
            if not was_speech and self._meter.seen_speech:
                await self._on_user_speech_onset()
            await self._send_pcm_to_asr(pcm)
            if fired:
                await self._finalize_turn(auto=True)
            return

        # PTT: only capture between start and commit
        if not self._capturing or self._asr_stream is None:
            return
        self._meter.update(rms, duration_ms)
        await self._send_pcm_to_asr(pcm)

    async def _send_pcm_to_asr(self, pcm: bytes) -> None:
        stream = self._asr_stream
        if stream is None:
            return
        await asyncio.to_thread(stream.send_pcm, pcm)
        if not self._asr_committed:
            await asyncio.to_thread(stream.commit, False)
            self._asr_committed = True

    async def _open_asr_stream(self) -> None:
        if self._asr_stream is not None:
            return
        if self.engines.asr is None:
            raise RuntimeError("ASR is not available for this session")
        epoch = self._context_epoch
        def partial(text):
            if epoch == self._context_epoch:
                self._on_asr_partial(text)
        opening = asyncio.create_task(asyncio.to_thread(self.engines.asr.open_stream, partial))
        try:
            self._asr_stream = await asyncio.shield(opening)
        except asyncio.CancelledError:
            late_stream = await opening
            await asyncio.to_thread(late_stream.close)
            raise
        self._asr_committed = False
        self._meter.reset()

    def _on_asr_partial(self, text: str) -> None:
        """Engine-thread callback (only for engines with streaming partials).

        Partials are transient: they describe the moment, never replay after a
        reconnect, and are not journaled (the final transcription.done is)."""
        if self._context_command or self._closed:
            return
        def emit() -> None:
            if self._context_command or self._closed:
                return
            self.state.emit(p.TRANSCRIPTION_DELTA, transient=True, text=text)
        try:
            self._loop.call_soon_threadsafe(emit)
        except RuntimeError:
            pass
        # speculative recall while the user is still talking: gold evidence is
        # usually retrievable well before end-of-utterance, so the turn itself
        # pays nothing. Results land in a cache — never in the KV cache.
        if self.memory is not None:
            try:
                self.memory.prefetch(text)  # already on a worker thread
            except Exception:  # noqa: BLE001
                pass

    async def _abort_capture(self) -> None:
        stream, self._asr_stream = self._asr_stream, None
        self._capturing = False
        self._asr_committed = False
        self._meter.reset()
        if stream is not None:
            try:
                await asyncio.to_thread(stream.close)
            except Exception:  # noqa: BLE001
                pass

    async def _finalize_turn(self, auto: bool) -> None:
        stream, self._asr_stream = self._asr_stream, None
        self._capturing = False
        self._asr_committed = False
        if stream is None:
            return
        item_id = f"item_{next(self._serial)}"
        mode = (self.state.config.capture_mode or self.settings.capture_mode).lower()
        if mode == "ptt" and not self._meter.has_enough_speech:
            self._meter.reset()
            await asyncio.to_thread(stream.close)
            self.state.emit(p.TRANSCRIPTION_DONE, text="", item_id=item_id, auto=auto)
            return
        self._meter.reset()
        t0 = time.monotonic()
        try:
            text = await asyncio.to_thread(stream.finalize)
        except Exception as exc:  # noqa: BLE001
            self._emit_error("asr_error", str(exc))
            return
        finally:
            try:
                await asyncio.to_thread(stream.close)
            except Exception:  # noqa: BLE001
                pass
        asr_ms = round((time.monotonic() - t0) * 1000.0, 1)
        self.metrics["asr_ms"] = asr_ms
        text = (text or "").strip()
        log.info("ASR %sfinal for %s in %.0fms: %r",
                 "auto " if auto else "", self.state.session_id, asr_ms, text)
        self.state.emit(p.TRANSCRIPTION_DONE, text=text, item_id=item_id, auto=auto, asr_ms=asr_ms,
                        **self._media_ts_fields())
        if text:
            await self._user_turn(text)

    # ------------------------------------------------------------------ turns & barge-in

    def _any_response_live(self) -> bool:
        return any(not r.done for r in self._responses.values())

    async def _on_user_speech_onset(self) -> None:
        """User started speaking (PTT press or server VAD onset) → duck + barge-in."""
        self.state.emit(p.SPEECH_STARTED, transient=True)
        if self._any_response_live():
            await self._cancel_response(p.STOP_INTERRUPTED)

    async def _user_turn(self, text: str) -> None:
        # A user turn always preempts: even with no LIVE response server-side,
        # the client may still be PLAYING buffered audio of a finished reply —
        # speech_started makes it stop that backlog immediately. (Without this,
        # an uncapped model banks minutes of narration PCM ahead of playback and
        # the user hears the old answer talk over their new question.)
        self.state.emit(p.SPEECH_STARTED, transient=True)
        if self._any_response_live():
            await self._cancel_response(p.STOP_INTERRUPTED)
        try:
            await self._ensure_context_room()
        except Exception as exc:
            self._mark_vlm_dead(f"context rollover failed: {exc}")
            return
        self._segmenter.reset()
        self.metrics["turns"] += 1
        self._pending_turn_t0 = time.monotonic()
        if self._vlm_dead:
            self._emit_error("vlm_unavailable", "the realtime model loop is not running")
            return

        if self.memory is not None and not (self.state.config.initial_prompt or self.settings.initial_prompt):
            from ..memory.retrieval import is_history_question, refers_to_dialogue
            from ..memory.rollover import _default_system_prompt
            default_system = not self.state.config.system_prompt or self.state.config.system_prompt.strip() == _default_system_prompt().strip()
            history_check = self.memory.has_dialogue_history if refers_to_dialogue(text) else self.memory.has_history
            if default_system and is_history_question(text) and not await asyncio.to_thread(history_check):
                if self._closed:
                    return
                message = ('当前会话还没有可供核对的早前记录，无法确认。请先提供相关信息。'
                           if self.memory.language.startswith('zh') else
                           'This session has no history to consult. Please provide the relevant information first.')
                self._emit_memory_notice(text, 'no_history', message)
                return

        # buffered source-change timeline notes ride the MODEL text only — the
        # clean `text` was already echoed to transcript/journal by the caller
        notes, self._pending_source_notes = self._pending_source_notes, []
        recall = await self._recall_for_turn(text)
        if self._closed:
            return
        if getattr(recall, 'reason', None):
            # A bounded candidate search cannot prove absence from native KV
            # or the complete history. Do not turn its miss into a system veto.
            log.info('memory recall outcome session=%s reason=%s',
                     self.state.session_id, recall.reason)
        # user text is sanitized because split_special_tokens=False: a literal
        # `<think>` or `<|im_end|>` in ASR/typed input would otherwise tokenize
        # as a real control token and corrupt the chat scaffold
        parts = [*( [recall.block] if recall else [] ), *notes, mem_inject.sanitize_model_text(text)]
        vlm_text = "\n".join(p_ for p_ in parts if p_)
        # index the turn AFTER recalling, so a turn can never recall itself
        self._note_memory_turn("user", text)
        self._pending_fact_user_text = text

        frame = None
        latest = self._latest_frame
        source = (self._segment or {}).get('kind', self.state.config.video_source)
        if latest is not None and (source == 'image' or
                                   (time.monotonic() - latest[2]) <= self.settings.frame_max_age_s):
            frame = latest
        for attempt in range(2):
            injection_position = self._last_text_tokens
            try:
                if frame is not None:
                    await asyncio.to_thread(
                        self.engines.vlm.put_prompt_frame, vlm_text, frame[0], frame[1], len(frame[0]), True)
                else:
                    await asyncio.to_thread(self.engines.vlm.put_prompt, vlm_text)
                if self.memory is not None:
                    self.memory.note_context_turn('user', text)
                    if recall and not self._closed:
                        self.memory.mark_injected(recall.ids, text_tokens=injection_position, block=recall.block)
                        self.state.emit(p.MEMORY_RECALLED, items=recall.items)
                        log.info('memory: recalled %d item(s) for %s', len(recall.ids), self.state.session_id)
                break
            except Exception as exc:
                if type(exc).__name__ == "ContextRolloverRequired" and attempt == 0:
                    try:
                        await self._ensure_context_room()
                    except Exception as rollover_exc:
                        self._mark_vlm_dead(f"context rollover failed: {rollover_exc}")
                        return
                    continue
                if isinstance(exc, ValueError):
                    self._emit_error("invalid_input", str(exc))
                else:
                    self._mark_vlm_dead(f"prompt failed: {exc}")

    def _emit_memory_notice(self, text: str, code: str, message: str) -> None:
        response_id = f'resp_{next(self._serial)}'
        self.state.emit(p.MEMORY_NOTICE, response_id=response_id, code=code, message=message)
        self.state.emit(p.RESPONSE_CREATED, response_id=response_id, source='system')
        self.state.emit(p.RESPONSE_TEXT_DONE, response_id=response_id, source='system', text='')
        self.state.emit(p.RESPONSE_DONE, response_id=response_id, source='system', stop_reason=p.STOP_END_TURN)
        self._note_memory_turn('user', text)
        self._pending_turn_t0 = None
        self._pending_fact_user_text = None
        self.metrics['memory_notices'] = self.metrics.get('memory_notices', 0) + 1

    async def _recall_for_turn(self, text: str):
        """Retrieve → gate → format, off the event loop. Never raises."""
        if self.memory is None:
            return mem_session.EMPTY_RECALL
        try:
            # the worker's exact text-token count anchors the re-injection
            # distance gate; None until the first status carrying it arrives
            recall = await asyncio.to_thread(
                self.memory.recall_for_turn, text, now_tokens=self._last_text_tokens)
        except Exception as exc:  # noqa: BLE001
            log.debug("memory recall failed: %s", exc)
            return mem_session.EMPTY_RECALL
        if self._closed:
            return mem_session.EMPTY_RECALL
        if recall:
            # channel V: a recalled frame re-enters through the normal frame
            # queue at the CURRENT session time — its original time rides the
            # text line instead. Rewinding the literal timestamp lane would put
            # the stream out of the monotone order the model was trained on.
            for jpeg in recall.frames:
                try:
                    await asyncio.to_thread(self.engines.vlm.put_frame, jpeg, None, len(jpeg))
                except Exception as exc:  # noqa: BLE001
                    log.debug("memory frame re-inject failed: %s", exc)
            # re-injection is distance-gated inside the session (design §5);
            # the client gets an itemized event so recall is visible/auditable,
            # never ambient
            if self._closed:
                return mem_session.EMPTY_RECALL
            # The caller commits bookkeeping/events only after prompt submission.
        return recall

    def _note_memory_turn(self, role: str, text: str, *, commit: bool = True) -> None:
        if self.memory is None or not text:
            return
        try:
            media_ts = self._media_ts_fields().get("media_ts")
            if role == "user":
                self.memory.note_user_turn(text, media_ts=media_ts)
            else:
                # commit=False (barge-in / <|eot_id|> interrupted): the text stays
                # as compact-journal context but never enters long-term memory
                from ..memory.retrieval import is_history_question
                from ..memory.rewrite import temporal_order
                self.memory.note_assistant_turn(text, media_ts=media_ts, commit=commit,
                    historical_answer=(is_history_question(self._pending_fact_user_text or '') or
                                       temporal_order(self._pending_fact_user_text or '') is not None))
                if commit:
                    self.memory.note_context_turn('assistant', text)
        except Exception as exc:  # noqa: BLE001
            log.debug("memory note_turn failed: %s", exc)

    def _schedule_memory_facts(self) -> None:
        """A finalized assistant reply closes a QA pair — the segment boundary
        for background fact extraction (memory, design §3). Fire-and-forget:
        never awaited, never allowed to raise on the hot path."""
        if self.memory is None or self._closed:
            return
        user_text, self._pending_fact_user_text = self._pending_fact_user_text, None
        if not user_text:
            return
        if len(self._memory_tasks) >= max(1, self.settings.memory_background_tasks):
            self.metrics['memory_fact_tasks_skipped'] = self.metrics.get('memory_fact_tasks_skipped', 0) + 1
            return
        try:
            task = asyncio.get_running_loop().create_task(
                self.memory.maybe_extract_facts(None, user_text, None))
            self._memory_tasks.add(task)
            task.add_done_callback(self._memory_tasks.discard)
        except Exception as exc:  # noqa: BLE001
            log.debug("memory facts schedule failed: %s", exc)

    async def _cancel_response(self, stop_reason: str) -> None:
        """User barge-in: kill EVERY live response — draining audio included."""
        live = [r for r in self._responses.values() if not r.done]
        if not live:
            return
        self._units.clear()
        if self.engines.tts is not None:
            try:
                self.engines.tts.cancel_turn()
            except Exception:  # noqa: BLE001
                pass
        self._tts_turn_id = None
        self._tts_fed_current = False
        was_generating = False
        for r in live:
            if not r.finalized:
                was_generating = True
                r.finalized = True
                r.stop_reason = stop_reason
                self.state.emit(p.RESPONSE_TEXT_DONE, response_id=r.response_id, text=r.text)
                # the truncated half-answer stays as compact context, never stored
                self._note_memory_turn("assistant", r.text, commit=False)
            else:
                r.stop_reason = stop_reason
            self._complete_response(r)
        if was_generating and not self._vlm_dead:
            self._drop_model_tail = True
            try:
                if getattr(self.engines.vlm, "turn_interrupt_is_local", False):
                    # Atomic with the UI cancellation: invalidate already-polled tails.
                    self.engines.vlm.request_turn_end()
                else:
                    await asyncio.to_thread(self.engines.vlm.request_turn_end)
            except Exception as exc:  # noqa: BLE001
                log.warning("request_turn_end failed for %s: %s", self.state.session_id, exc)

    # ------------------------------------------------------------------ VLM output → captions + TTS units

    async def _vlm_drain_loop(self) -> None:
        errors = 0
        while not self._vlm_dead:
            try:
                engine = self.engines.vlm
                batch = await asyncio.to_thread(engine.poll_output, 0.25, 128)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors <= 3:
                    self._emit_error("vlm_error", str(exc))
                await asyncio.sleep(1.0)
                continue
            errors = 0
            if engine is not self.engines.vlm:
                continue
            for ev in batch.chunk_events:
                is_current = getattr(engine, "output_event_is_current", None)
                if callable(is_current) and not is_current(ev):
                    continue
                emitted_at = ev.get("emitted_at")
                self._route_model_text(
                    str(ev.get("text") or ""),
                    float(emitted_at) if isinstance(emitted_at, (int, float)) else None,
                )
            if not batch.active:
                if self._reseat_in_progress:
                    return
                self._mark_vlm_dead("the realtime model loop exited")
                return

    def _route_model_text(self, text: str, emitted_at: Optional[float] = None) -> None:
        for part in CONTROL_TOKEN_RE.split(text):
            if not part:
                continue
            stripped = part.strip()
            if CONTROL_TOKEN_RE.fullmatch(stripped or part):
                self._handle_control_token(stripped)
            elif stripped.startswith(ERROR_PREFIX):
                self._emit_error("vlm_error", stripped)
                self._finalize_response(p.STOP_ERROR)
            elif part:
                self._handle_content(part, emitted_at)

    def _handle_control_token(self, token: str) -> None:
        if token == ROUND_START:
            if self._drop_model_tail and getattr(self.engines.vlm, "turn_interrupt_is_local", False):
                return
            self._drop_model_tail = False  # an explicit round always speaks
            self._open_response()
        elif token in SILENCE_TOKENS:
            self.metrics["silence_outputs"] += 1
            # the realtime model closes a spoken round by going idle —
            # <|silence|> IS the end-of-turn signal (it never emits
            # round_end/im_end on the output stream). Finalizing here splits
            # the transcript into one bubble per round; the token itself is
            # never surfaced. No-op when no response is open (idle stream).
            self._finalize_response(p.STOP_END_TURN)
            # a silent model with nothing generating IS the idle moment the
            # rollover idle trigger waits for (design §6)
            self._maybe_rollover(idle=True)
        elif token in END_TOKENS:
            if self._drop_model_tail:
                self._drop_model_tail = False  # the barge-in ack landed
                return
            self._finalize_response(p.STOP_END_TURN)
        elif token in HIDDEN_TOKENS:
            pass
        else:
            log.debug("unmapped control token from model: %r", token)

    def _handle_content(self, chunk: str, emitted_at: Optional[float] = None) -> None:
        if self._drop_model_tail:
            self.metrics["discarded_tail_chars"] = (
                self.metrics.get("discarded_tail_chars", 0) + len(chunk))
            return
        chunk = JUNK_TOKEN_RE.sub("", chunk)
        if not chunk:
            return
        r = self._response
        if not chunk.strip() and (r is None or r.finalized or r.done):
            return  # whitespace between control tokens must not open a response
        r = self._open_response()
        if r.finalized:  # stray text after an end token — treat as a fresh round
            return
        if r.first_delta_at is None:
            r.first_delta_at = time.monotonic()
            self.metrics["vlm_ttft_ms"] = round((r.first_delta_at - r.t0) * 1000.0, 1)
        wall = emitted_at if emitted_at is not None else time.time()
        if r.first_text_wall is None:
            r.first_text_wall = wall
        r.last_text_wall = wall
        r.text += chunk
        self.state.emit(p.RESPONSE_TEXT_DELTA, response_id=r.response_id, delta=chunk,
                        emitted_at=round(wall, 3))
        if self.engines.tts is None:
            return
        for segment in self._segmenter.feed(chunk):
            self._push_unit(r, segment)
        if r.units_emitted == 0:
            first = self._first_clause_cut()
            if first:
                self._push_unit(r, first)

    def _first_clause_cut(self) -> Optional[str]:
        """Cut the response's first TTS unit at an early clause boundary (T4)."""
        min_chars = self.settings.tts_first_clause_chars
        if min_chars <= 0:
            return None
        buf = self._segmenter.buffer
        for i, ch in enumerate(buf):
            if ch in CLAUSE_CHARS and i + 1 >= min_chars:
                self._segmenter.buffer = buf[i + 1:].lstrip()
                return buf[: i + 1].strip()
        return None

    def _open_response(self) -> ResponseCtx:
        r = self._response
        if r is not None and not r.finalized and not r.done:
            return r
        # a previous response may still be draining TTS audio — leave it in
        # self._responses; the feeder holds the new round's units until its
        # tts_turn_end (and drop_stale expires them if the wait grows stale).
        # Under narration churn (a round per frame), retire stale UNSPOKEN
        # responses but never cut the one that is actively speaking (§1C:
        # drop the backlog, not the mouth).
        if len(self._responses) >= 2:
            speaking = self._tts_turn_id
            for old in list(self._responses.values()):
                if old.response_id != speaking and not old.done:
                    old.stop_reason = p.STOP_INTERRUPTED
                    self._complete_response(old)
        rid = f"resp_{next(self._serial)}"
        t0 = self._pending_turn_t0 or time.monotonic()
        self._pending_turn_t0 = None
        new = ResponseCtx(response_id=rid, t0=t0,
                          media_ts=self._media_ts_fields().get("media_ts"))
        self._responses[rid] = new
        self._response = new
        self._segmenter.reset()
        self.metrics["responses"] += 1
        self.state.emit(p.RESPONSE_CREATED, response_id=rid)
        return new

    def _finalize_response(self, stop_reason: str) -> None:
        r = self._response
        if r is None or r.finalized or r.done:
            return
        if self.engines.tts is not None:
            for segment in self._segmenter.flush():
                self._push_unit(r, segment)
        r.finalized = True
        r.stop_reason = stop_reason
        self.state.emit(p.RESPONSE_TEXT_DONE, response_id=r.response_id, text=r.text)
        self._note_memory_turn("assistant", r.text, commit=(stop_reason != p.STOP_INTERRUPTED))
        self._schedule_memory_facts()
        self.metrics["last_response_chars"] = len(r.text)
        if self.engines.tts is not None and r.units_emitted > 0:
            self._units.append(TtsUnit("flush", r.response_id))
            self._units_event.set()
        else:
            self._complete_response(r)

    def _complete_response(self, r: ResponseCtx) -> None:
        if r.done:
            return
        r.done = True
        # per-response latencies ride response.done (status is transient — the
        # history recorder keeps these as the turn's metrics)
        extra: Dict[str, Any] = {}
        if r.first_delta_at is not None:
            extra["ttft_ms"] = round((r.first_delta_at - r.t0) * 1000.0, 1)
        if r.audio_first_emit is not None:
            extra["ttfa_ms"] = round((r.audio_first_emit - r.t0) * 1000.0, 1)
        if r.media_ts is not None:
            extra["media_ts"] = r.media_ts
        if r.first_text_wall is not None:
            extra["gen_started_at"] = round(r.first_text_wall, 3)
        if r.last_text_wall is not None:
            extra["gen_ended_at"] = round(r.last_text_wall, 3)
        self.state.emit(p.RESPONSE_DONE, response_id=r.response_id,
                        stop_reason=r.stop_reason, **extra)
        self._responses.pop(r.response_id, None)
        if self._response is r:
            self._response = None

    # ------------------------------------------------------------------ §1C back-pressure / drop-stale hooks

    def audio_queue_seconds(self) -> float:
        """Estimated buffered-but-unplayed client audio (server estimate ⊔ client report)."""
        now = time.monotonic()
        server_est = 0.0
        for r in self._responses.values():
            if r.audio_first_emit is not None and not r.done:
                server_est += max(0.0, r.audio_seconds - (now - r.audio_first_emit))
        reported, at = self._client_buffer
        client_est = reported - (now - at) if at > 0 else 0.0
        return max(0.0, server_est, client_est)

    def should_emit_next_unit(self) -> bool:
        """Hysteresis gate on the unplayed-audio depth (§1C back-pressure)."""
        buffered = self.audio_queue_seconds()
        if self._bp_paused:
            if buffered <= self.settings.audio_buffer_low_s:
                self._bp_paused = False
        elif buffered >= self.settings.audio_buffer_high_s:
            self._bp_paused = True
        return not self._bp_paused

    def drop_stale(self) -> None:
        """Drop oldest/expired un-synthesized units so speech never narrates the past."""
        if not self.settings.tts_drop_stale:
            return  # policy disabled (TTS_DROP_STALE=0): keep every unit
        max_pending = self.settings.tts_max_pending_units
        max_age = self.settings.tts_unit_max_age_s
        now = time.monotonic()

        def segments() -> list:
            return [u for u in self._units if u.kind == "segment"]

        if max_pending > 0:
            while len(segments()) > max_pending:
                for i, unit in enumerate(self._units):
                    if unit.kind == "segment":
                        del self._units[i]
                        self.metrics["units_dropped"] += 1
                        log.info("drop-stale: backlog unit dropped (%r…)", unit.text[:24])
                        break
        if max_age > 0:
            segs = segments()
            for unit in segs[:-1]:  # never expire the sole remaining unit
                if now - unit.created > max_age:
                    try:
                        self._units.remove(unit)
                        self.metrics["units_dropped"] += 1
                        log.info("drop-stale: expired unit dropped (%r…)", unit.text[:24])
                    except ValueError:
                        pass

    def _push_unit(self, r: ResponseCtx, segment: str) -> None:
        segment = (segment or "").strip()
        if not segment:
            return
        self._units.append(TtsUnit("segment", r.response_id, segment))
        r.units_emitted += 1
        self._units_event.set()

    async def _tts_feeder_loop(self) -> None:
        while True:
            if not self._units:
                self._units_event.clear()
                await self._units_event.wait()
                continue
            self.drop_stale()
            if not self._units:
                continue
            head = self._units[0]
            if head.kind == "segment" and not self.should_emit_next_unit():
                await asyncio.sleep(0.1)
                continue
            if self._tts_turn_id and head.response_id != self._tts_turn_id:
                previous = self._responses.get(self._tts_turn_id)
                if previous is not None and not previous.done:
                    # the previous answer is still speaking — never cut it for
                    # follow-up narration; held units age out via drop_stale
                    await asyncio.sleep(0.05)
                    continue
            unit = self._units.popleft()
            r = self._responses.get(unit.response_id)
            if r is None or r.done:
                continue  # unit belongs to a retired response
            tts = self.engines.tts
            if tts is None:
                continue
            if unit.kind == "flush":
                if self._tts_fed_current and self._tts_turn_id == unit.response_id:
                    tts.end_turn()
                else:
                    # nothing was actually synthesized for this response
                    self._complete_response(r)
                continue
            if self._tts_turn_id != unit.response_id:
                tts.start_turn(unit.response_id)
                self._tts_turn_id = unit.response_id
                self._tts_fed_current = False
            # backlog coalescing (tts_serving_plan.md Stage 0): when synthesis
            # runs behind the model, consecutive queued units of THIS response
            # merge into one engine job — the engine's internal chunk batching
            # turns N serial decodes into one batched decode (~3x measured on
            # the pytorch sidecar), which is what lets the queue drain again.
            text = unit.text
            cap = self.settings.tts_coalesce_max_chars
            while (
                cap > 0
                and self._units
                and self._units[0].kind == "segment"
                and self._units[0].response_id == unit.response_id
                and len(text) + 1 + len(self._units[0].text) <= cap
            ):
                text = f"{text} {self._units.popleft().text}"
            tts.feed_segment(text, unit.response_id)
            self._tts_fed_current = True

    # ------------------------------------------------------------------ TTS worker events → audio frames

    def _tts_emit_threadsafe(self, payload: Dict[str, Any]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._tts_events.put_nowait, payload)
        except RuntimeError:
            pass

    async def _tts_pump_loop(self) -> None:
        while True:
            payload = await self._tts_events.get()
            kind = payload.get("type")
            rid = str(payload.get("turn_id") or "")
            r = self._responses.get(rid)
            if kind == "tts_audio_chunk":
                if r is None or r.done:
                    continue  # stale audio after cancel/turnover — never send old speech
                pcm = payload.get("pcm")
                if not pcm:
                    continue
                sample_rate = int(payload.get("sample_rate") or 16000)
                channels = int(payload.get("channels") or 1)
                now = time.monotonic()
                if r.audio_first_emit is None:
                    r.audio_first_emit = now
                    self.metrics["tts_ttfa_ms"] = round((now - r.t0) * 1000.0, 1)
                r.audio_started = True
                r.audio_seconds += len(pcm) / max(1, sample_rate * channels * 2)
                self.state.emit(
                    p.RESPONSE_AUDIO_DELTA,
                    binary=p.audio_binary(bytes(pcm)),
                    response_id=r.response_id,
                    sample_rate=sample_rate,
                    channels=channels,
                    pcm_bytes=len(pcm),
                )
            elif kind == "tts_turn_end":
                if r is not None and r.finalized and not r.done:
                    self.state.emit(p.RESPONSE_AUDIO_DONE, response_id=r.response_id)
                    self._complete_response(r)
            elif kind == "tts_error":
                self._emit_error("tts_error", str(payload.get("message") or "synthesis failed"))
            # tts_turn_start / tts_turn_abort are internal bookkeeping — no client event

    # ------------------------------------------------------------------ status & errors

    async def _status_loop(self) -> None:
        interval = max(0.2, self.settings.status_interval_s)
        while True:
            await asyncio.sleep(interval)
            if self.state.phase != "active":
                continue
            self.state.emit(p.STATUS, transient=True, **self.status_payload())
            # the hard rollover trigger lives here: it fires regardless of
            # idleness, so the 1 Hz tick guarantees eventual firing (design §6)
            self._maybe_rollover(idle=False)

    def _memory_status(self) -> Optional[Dict[str, Any]]:
        if self.memory is None:
            return None
        try:
            return self.memory.status()
        except Exception:  # noqa: BLE001
            return None

    def status_payload(self) -> Dict[str, Any]:
        vlm_status: Dict[str, Any] = {}
        try:
            vlm_status = self.engines.vlm.status()
        except Exception:  # noqa: BLE001
            pass
        # cache the worker's exact text-token count for the rollover trigger
        # and the re-injection distance gate (token-counter patch → 1 Hz status)
        text_tokens = vlm_status.get("text_tokens")
        if isinstance(text_tokens, (int, float)) and not isinstance(text_tokens, bool):
            self._last_text_tokens = float(text_tokens)
        return {
            "response_active": any(not r.done for r in self._responses.values()),
            "capture_mode": self.state.config.capture_mode,
            "capturing": self._capturing or self._asr_stream is not None,
            "vlm_alive": not self._vlm_dead,
            "queues": {
                "audio_in": len(self._audio_in),
                "tts_pending_units": len(self._units),
                "audio_buffer_s": round(self.audio_queue_seconds(), 2),
                "frame_queue": vlm_status.get("frame_queue_size"),
                "output_queue": vlm_status.get("output_queue_size"),
            },
            "memory": self._memory_status(),
            "metrics": {**self.metrics, "events_dropped": self.state.events_dropped},
            # workers mode: the session's worker pushes its GPU snapshot over the
            # proxy status frames (loop-safe dict read); inproc falls back to the
            # local probe (no-op unless torch is already imported)
            "gpu": vlm_status.get("gpu") or _gpu_snapshot(),
            "kv": vlm_status.get("kv"),
            "context": vlm_status.get("context"),
            "vlm": {k: vlm_status.get(k) for k in
                    ("frames_received", "frames_consumed", "frames_dropped", "outputs_emitted",
                     "text_tokens")},
        }

    def _emit_error(self, code: str, message: str, event_id: Optional[str] = None) -> None:
        log.warning("session %s error [%s]: %s", self.state.session_id, code, message)
        self.state.emit(p.ERROR, code=code, message=message, event_id=event_id)

    def _mark_vlm_dead(self, message: str) -> None:
        if self._vlm_dead:
            return
        self._vlm_dead = True
        if (not self._closed
                and not self._reseat_in_progress
                and not self._failure_relay_attempted
                and self._rollover is not None
                and self._reseat_factory is not None
                and self.memory is not None):
            # GATEWAY_PLAN P2 failure relay: the transport died but the memory
            # prefix can rebuild the conversation on a fresh engine — try that
            # once before going terminal. _vlm_dead stays True during the relay,
            # so push_frame keeps its silent-drop behaviour (the latest frame is
            # still tracked for the re-push) and _user_turn keeps answering
            # vlm_unavailable instead of feeding a dead engine.
            self._failure_relay_attempted = True
            log.warning("session %s: VLM died (%s); relaying onto a fresh engine "
                        "with the memory prefix", self.state.session_id, message)
            try:
                task = asyncio.get_running_loop().create_task(
                    self._reseat_vlm(trigger="failure"),
                    name=f"orch-relay-{self.state.session_id[:12]}")
            except Exception as exc:  # noqa: BLE001 — no running loop etc.
                log.warning("failure relay could not be scheduled: %s", exc)
            else:
                self._reseat_task = task
                self._tasks.append(task)
                return
        self._vlm_dead_final(message)

    def _vlm_dead_final(self, message: str) -> None:
        """Terminal dead state: relay disabled, already attempted, or failed."""
        self._vlm_dead = True
        self._emit_error("vlm_stopped", message)
        self._finalize_response(p.STOP_ERROR)

    # ------------------------------------------------------------------ rollover (design §6)

    async def _run_context_command(self, command: str) -> None:
        try:
            # Quiesce ASR (including any recall it is awaiting) and output before
            # touching the old memory lifetime. Threaded work is drained on clear.
            if self._audio_task is not None:
                self._audio_task.cancel()
                await asyncio.gather(self._audio_task, return_exceptions=True)
            self._audio_in.clear()
            await self._abort_capture()
            async with self._reseat_lock:
                if self._closed:
                    return
                self._reseat_in_progress = True
                if self._vlm_drain_task is not None:
                    self._vlm_drain_task.cancel()
                    await asyncio.gather(self._vlm_drain_task, return_exceptions=True)
                await self._cancel_response(p.STOP_INTERRUPTED)
                self._pending_turn_t0 = None
                self._pending_fact_user_text = None
                await self._perform_reseat_vlm(trigger="clear" if command == "/clear" else "manual")
            if not self._closed:
                self.state.emit(p.CONTEXT_COMMAND, command=command, status="completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("context command %s failed for %s", command, self.state.session_id)
            if not self._closed:
                self.state.emit(p.CONTEXT_COMMAND, command=command, status="failed", message=str(exc))
        finally:
            self._reseat_in_progress = False
            self._context_command = None
            if not self._closed:
                self._audio_task = asyncio.create_task(self._audio_loop(), name="orch-audio")
                self._tasks.append(self._audio_task)
                if not self._vlm_dead and (self._vlm_drain_task is None or self._vlm_drain_task.done()):
                    self._vlm_drain_task = asyncio.create_task(self._vlm_drain_loop(), name="orch-vlm")
                    self._tasks.append(self._vlm_drain_task)
                # Do not retain completed command/replaced lane tasks forever.
                current = asyncio.current_task()
                self._tasks[:] = [t for t in self._tasks if t is not current and not t.done()]

    def _context_requires_rollover(self) -> bool:
        try:
            return bool(self.engines.vlm.status().get("context", {}).get("rollover_required"))
        except Exception:
            return False

    async def _ensure_context_room(self) -> None:
        task = self._reseat_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            await asyncio.shield(task)
        if self._context_requires_rollover():
            if self._rollover is None or self._reseat_factory is None or self.memory is None:
                raise RuntimeError("context exhausted; memory rollover is not configured")
            await self._reseat_vlm(trigger="context")
            if self._vlm_dead or self._context_requires_rollover():
                raise RuntimeError("replacement context is unavailable or its prefix is too large")

    def _vlm_generating(self) -> bool:
        """A response the model is still WRITING (TTS draining a finalized one
        does not count — the model is idle while the mouth speaks)."""
        return any(not r.finalized and not r.done for r in self._responses.values())

    def _maybe_rollover(self, *, idle: bool) -> None:
        """Threshold gate → schedule `reseat_vlm`. Never raises, never fires
        mid-streaming-assistant-output: a live generation defers to the next
        hook (the 1 Hz status tick re-checks, so the hard trigger still
        guarantees eventual firing — it exists to interrupt silence, not speech).
        """
        if self._closed or self._context_command or self._reseat_in_progress or self._vlm_dead:
            return
        if self._reseat_task is not None and not self._reseat_task.done():
            return
        context_limit = self._context_requires_rollover()
        if self._rollover is None or self._reseat_factory is None or self.memory is None:
            if context_limit:
                self._mark_vlm_dead("context exhausted; memory rollover is not configured")
                self._tasks.append(asyncio.create_task(asyncio.to_thread(self.engines.vlm.stop, 5.0)))
            return  # rollover inert: no manager, no factory, or no memory
        if not context_limit and time.monotonic() - self._last_rollover_at < ROLLOVER_COOLDOWN_S:
            return
        if not context_limit and self._vlm_generating():
            return
        tokens = self._last_text_tokens
        if tokens is None and not context_limit:
            return
        try:
            # compact prefetch rides ahead of both thresholds (pi provider):
            # the seconds-long /compact is normally done by the time a rollover
            # fires. Runs here = the 1 Hz status tick AND the <|silence|> idle
            # point, the same hooks that evaluate should_rollover.
            self._rollover.maybe_prefetch_compact(tokens)
        except Exception as exc:  # noqa: BLE001
            log.debug("rollover prefetch check failed: %s", exc)
        try:
            fire = context_limit or self._rollover.should_rollover(tokens, idle=idle)
        except Exception as exc:  # noqa: BLE001
            log.debug("rollover check failed: %s", exc)
            return
        if not fire:
            return
        task = asyncio.get_running_loop().create_task(
            self._reseat_vlm(trigger="context" if context_limit else "idle" if idle else "hard"),
            name=f"orch-reseat-{self.state.session_id[:12]}")
        self._tasks.append(task)
        self._reseat_task = task

    async def _call_reseat_factory(self, prefill_json: str, *, clear: bool = False) -> Any:
        """The factory may be sync or async (tests inject plain callables)."""
        result = self._reseat_factory(
            prompt="" if clear else self.state.config.initial_prompt
            if self.state.config.initial_prompt is not None else self.settings.initial_prompt,
            system_prompt=self.state.config.system_prompt,
            prefill_messages=prefill_json)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            opening = asyncio.ensure_future(result)
            try:
                return await asyncio.shield(opening)
            except asyncio.CancelledError:
                # to_thread handshakes cannot be cancelled; reclaim a late lease.
                try:
                    late_session = await opening
                except Exception:
                    pass
                else:
                    await asyncio.to_thread(late_session.stop, 5.0)
                raise
        return result

    async def _reseat_vlm(self, *, trigger: str) -> None:
        expected = self.engines.vlm
        async with self._reseat_lock:
            if self.engines.vlm is not expected or self._closed:
                return
            await self._perform_reseat_vlm(trigger=trigger)

    async def _perform_reseat_vlm(self, *, trigger: str) -> None:
        """Rebuild the prefix and swap the VLM realtime session underneath the
        running conversation. Ordering invariants:

        - the NEW engine must be confirmed started before `engines.vlm` is
          swapped, so a factory failure leaves the old session untouched;
        - capacity conflict (single replica / in-proc infer lock): the old
          engine is stopped FIRST and the factory retried once — the only
          ordering that can ever succeed there;
        - a total failure after the old engine was stopped degrades to the
          same state as a VLM crash (vlm_stopped; the worker's KV watchdog
          remains the backstop it always was).
        """
        self._reseat_in_progress = True
        old = self.engines.vlm
        old_stopped = False
        new = None
        try:
            if trigger in ("context", "clear"):
                # Hard budget: stop growth while rebuilding the memory prefix.
                await self._cancel_response(p.STOP_INTERRUPTED)
                old_stopped = True
                await asyncio.to_thread(old.stop, 5.0)
            if trigger == "clear":
                self._latest_frame = None
                self._pending_source_notes.clear()
                if self._rollover is not None:
                    self._rollover.seal()
                if self.memory is not None:
                    self.memory.lifetime.seal()
                tasks = list(self._memory_tasks)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self._rollover is not None:
                    await asyncio.to_thread(self._rollover.close)
                if self.memory is not None:
                    # Shield the replacement: cleanup cannot race a late threaded
                    # reset creating a fresh lifetime after conversation shutdown.
                    reset = asyncio.create_task(asyncio.to_thread(self.memory.cleared_copy))
                    try:
                        self.memory = await asyncio.shield(reset)
                    except asyncio.CancelledError:
                        fresh = await reset
                        await asyncio.to_thread(fresh.close)
                        raise
                    if self._rollover is not None:
                        self._rollover = self._rollover.cleared_copy(self.memory)
                from ..memory.rollover import _default_system_prompt
                system = self.state.config.system_prompt
                if system is None:
                    system = _default_system_prompt()
                prefill = [{"role": "system", "content": system}]
                kept_ids, est_tokens = [], mem_inject.estimate_tokens(system)
            elif trigger == "manual":
                await asyncio.to_thread(self.memory.writer.drain)
                prefill, kept_ids, est_tokens = await self._rollover.build_prefix(require_summary=True)
                if not prefill:
                    raise RuntimeError("Context compaction was cancelled")
            else:
                prefill, kept_ids, est_tokens = await self._rollover.build_prefix()
            if self._closed:
                return
            prefill_json = json.dumps(prefill, ensure_ascii=False)
            log.info("rollover (%s) for %s: prefix ~%d est tokens, %d kept items, %d messages",
                     trigger, self.state.session_id, est_tokens, len(kept_ids), len(prefill))
            try:
                new = await self._call_reseat_factory(prefill_json, clear=trigger == "clear")
            except Exception as exc:  # noqa: BLE001 — classified below
                capacity_conflict = type(exc).__name__ == "NoFreeReplica" or "already running" in str(exc)
                if not capacity_conflict or old_stopped:
                    raise  # old session still running — degrade to pre-rollover behaviour
                log.info("rollover reseat: no free replica; stopping the old engine first")
                old_stopped = True
                await asyncio.to_thread(old.stop, 5.0)
                new = await self._call_reseat_factory(prefill_json, clear=trigger == "clear")
            if self._closed:
                await asyncio.to_thread(new.stop, 5.0)
                return
            if not old_stopped:
                try:
                    await asyncio.to_thread(old.stop, 5.0)
                except Exception:  # noqa: BLE001 — the replacement is live; a zombie old one only leaks
                    log.warning("rollover: old VLM session stop failed", exc_info=True)
            self.engines.vlm = new
            self._vlm_dead = False
            self._drop_model_tail = False
            # the drain loop exits on _vlm_dead or a dead batch; a reseat must
            # hand the NEW engine a fresh task either way
            task = self._vlm_drain_task
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._vlm_drain_task = asyncio.get_running_loop().create_task(
                self._vlm_drain_loop(), name=f"orch-vlm-{self.state.session_id[:12]}")
            self._tasks.append(self._vlm_drain_task)
            # restore vision context: the new session's frame window is empty
            latest = self._latest_frame
            if latest is not None:
                await asyncio.to_thread(new.put_frame, latest[0], latest[1], len(latest[0]))
            # the injected set is recomputed from EXACTLY what went into the new
            # prefix (design §5); distances reset to the new prefix's size
            if self.memory is not None:
                self.memory.note_rollover(kept_ids, text_tokens=float(est_tokens))
            self._last_text_tokens = float(est_tokens)
            self._last_rollover_at = time.monotonic()
            # a successful reseat re-arms the failure relay: the NEXT transport
            # death gets its own one-shot relay attempt
            self._failure_relay_attempted = False
            self.metrics["rollovers"] = self.metrics.get("rollovers", 0) + 1
            log.info("rollover (%s) complete for %s", trigger, self.state.session_id)
        except asyncio.CancelledError:
            if new is not None and self.engines.vlm is not new:
                await asyncio.to_thread(new.stop, 5.0)
            raise
        except Exception as exc:  # noqa: BLE001 — rollover must never kill a session
            log.exception("rollover reseat failed for %s: %s", self.state.session_id, exc)
            if old_stopped or self._vlm_dead:
                # no old session to fall back to — the old engine was stopped
                # for the capacity retry, or this reseat IS the failure relay
                # (entered with _vlm_dead already set). Surface the same
                # terminal state a VLM crash would (today's behaviour without
                # rollover); _failure_relay_attempted guards against recursion.
                self._vlm_dead_final(f"rollover reseat failed: {exc}")
            if trigger in ("manual", "clear"):
                raise
        finally:
            self._reseat_in_progress = False


_gpu_warned = False


def _gpu_snapshot() -> Optional[Dict[str, Any]]:
    """Cheap GPU memory snapshot for the status pill (None when no CUDA).

    Only consults torch if something else (the VLM adapter) already imported it —
    a cold torch import takes tens of seconds and must never run on the event loop.
    """
    global _gpu_warned
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        if not is_available():
            return None
        idx = current_device()
        return {
            "device": idx,
            "mem_allocated_gb": round(memory_allocated(idx) / 2**30, 2),
            "mem_reserved_gb": round(memory_reserved(idx) / 2**30, 2),
        }
    except Exception as exc:  # noqa: BLE001
        if not _gpu_warned:
            _gpu_warned = True
            log.debug("gpu snapshot unavailable: %s", exc)
        return None
