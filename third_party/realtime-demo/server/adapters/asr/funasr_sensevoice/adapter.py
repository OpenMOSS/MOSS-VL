"""FunASR SenseVoiceSmall ASR adapter (GPU fp16, turn-level).

Ported from board voice_runtime.py:FunasrSenseVoiceEngine/Stream with the key
upgrade: load on GPU with fp16 and do one warmup decode at startup (the first
CUDA call is 10-50x slower). Turn-level: buffer 16 kHz PCM, transcribe on
finalize().

SenseVoice has no native streaming, so realtime partials are emulated: when a
stream is opened with an `on_partial` callback and ASR_PARTIAL_INTERVAL_MS > 0,
a worker thread re-decodes the buffered audio every interval and reports the
hypothesis. Callers only see the standard adapter contract
(`open_stream(on_partial)` + `AsrCaps.streaming_partials`), so swapping in a
genuinely streaming engine later is a pure adapter change.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional

from ....config import Settings
from ....logging_conf import get_logger
from ...base import AsrCaps

log = get_logger(__name__)


class _SenseVoiceStream:
    def __init__(self, engine: "FunasrSenseVoiceAdapter", on_partial: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.on_partial = on_partial
        self._chunks: List[bytes] = []
        self._total = 0     # retained bytes across _chunks (front-dropped past the ceiling)
        self._received = 0  # cumulative bytes EVER received (never dropped) — growth gate
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # partial-caption worker: re-decode the RECENT window every interval and
        # report the hypothesis (started lazily on the first PCM so silent opens
        # are free). Re-decoding only a bounded tail keeps partial cost O(1) on a
        # long turn instead of O(n) per tick / O(n²) per turn.
        self._partial_interval_s = max(0.0, engine.partial_interval_ms / 1000.0)
        self._partial_window_bytes = int(getattr(engine, "partial_window_bytes", 0) or 0)
        self._buffer_max_bytes = int(getattr(engine, "buffer_max_bytes", 0) or 0)
        self._partial_thread: Optional[threading.Thread] = None
        self._decoded_bytes = 0  # cumulative-received size at the last partial decode
        self._last_decode_s = 0.0  # wall time of the last partial decode (adaptive throttle)
        self._last_partial = ""

    def send_pcm(self, pcm16: bytes) -> None:
        if not pcm16 or self._closed.is_set():
            return
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        with self._lock:
            self._chunks.append(bytes(pcm16))
            self._total += len(pcm16)
            self._received += len(pcm16)
            # memory safety ceiling (NOT a turn cap): drop only the OLDEST audio
            # beyond the ceiling so an arbitrarily long turn can't grow the buffer
            # without bound. The retained tail still feeds the final decode.
            if self._buffer_max_bytes > 0:
                while self._total > self._buffer_max_bytes and len(self._chunks) > 1:
                    self._total -= len(self._chunks.pop(0))
        if (self.on_partial is not None and self._partial_interval_s > 0
                and self._partial_thread is None):
            self._partial_thread = threading.Thread(
                target=self._partial_loop, name="asr-partials", daemon=True)
            self._partial_thread.start()

    def commit(self, final: bool) -> None:  # noqa: D401 - turn-level engine ignores commits
        return

    def pcm_len(self) -> int:
        with self._lock:
            return self._total

    def _tail(self, window_bytes: int) -> bytes:
        """Last `window_bytes` of retained audio (whole buffer if window<=0).

        Walks chunks from the newest so cost is O(window), not O(buffer)."""
        with self._lock:
            if window_bytes <= 0 or self._total <= window_bytes:
                return b"".join(self._chunks)
            picked: List[bytes] = []
            got = 0
            for chunk in reversed(self._chunks):
                picked.append(chunk)
                got += len(chunk)
                if got >= window_bytes:
                    break
        return b"".join(reversed(picked))[-window_bytes:]

    def _current_interval_s(self) -> float:
        """Engine-load-aware interval (stub engines without the hook keep the base)."""
        hook = getattr(self.engine, "effective_partial_interval_ms", None)
        if callable(hook):
            return max(self._partial_interval_s, hook() / 1000.0)
        return self._partial_interval_s

    def _partial_loop(self) -> None:
        # never re-decode faster than the last decode took: under GPU contention
        # a slow decode must not pile up behind itself (and block finalize's lock).
        while not self._closed.wait(max(self._current_interval_s(), self._last_decode_s)):
            received = self._received  # cumulative (never dropped) → keeps firing
            # decode only when new audio arrived and there's enough to be useful.
            # gate on cumulative-received, not retained-total: at the memory
            # ceiling `total` plateaus (old dropped = new added) but the sliding
            # window still changes, so partials must keep updating.
            if received <= self._decoded_bytes or received < self.engine.min_pcm_bytes:
                continue
            self._decoded_bytes = received
            # bounded RECENT window, not the whole (growing) turn → O(1) per tick
            pcm = self._tail(self._partial_window_bytes)
            if len(pcm) < self.engine.min_pcm_bytes:
                continue
            t0 = time.monotonic()
            try:
                text = self.engine.transcribe_pcm(pcm)
            except Exception as exc:  # noqa: BLE001 — partials must never kill the turn
                self._last_decode_s = time.monotonic() - t0
                log.debug("partial decode failed (non-fatal): %s", exc)
                continue
            self._last_decode_s = time.monotonic() - t0
            if self._closed.is_set():
                return  # finalize/close raced the decode — the final result wins
            if text and text != self._last_partial and self.on_partial is not None:
                self._last_partial = text
                try:
                    self.on_partial(text)
                except Exception:  # noqa: BLE001
                    pass

    def finalize(self) -> str:
        self._closed.set()  # stop the partial worker before the final decode
        with self._lock:
            pcm = b"".join(self._chunks)
        if len(pcm) < self.engine.min_pcm_bytes:
            return ""
        return self.engine.transcribe_pcm(pcm)

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self._chunks.clear()
            self._total = 0


class FunasrSenseVoiceAdapter:
    def __init__(self, settings: Settings, device: Optional[str] = None):
        self.s = settings
        self.partial_interval_ms = max(0, settings.asr_partial_interval_ms)
        self.caps = AsrCaps(streaming_partials=self.partial_interval_ms > 0,
                            needs_server_vad=False, sample_rate=settings.asr_sample_rate)
        self.model_path = settings.sensevoice_model
        self.vad_model_path = settings.sensevoice_vad_model
        # explicit device = the placement plan's resolved ASR_DEVICE=auto pick
        self.device = device or settings.asr_device
        if self.device.strip().lower() == "auto":
            self.device = "cuda:0"  # no plan available (registry called bare)
        self.language = settings.asr_language
        self.sample_rate = settings.asr_sample_rate
        self.use_itn = settings.asr_use_itn
        self.fp16 = settings.asr_fp16
        self.min_pcm_bytes = settings.asr_min_pcm_bytes
        # partial re-decode window / retained-buffer ceiling, in PCM16 bytes
        # (2 bytes/sample). Bound the emulated-partial cost + memory without
        # ever ending the turn (config.py asr_partial_window_s / asr_buffer_max_s).
        self.partial_window_bytes = int(
            max(0.0, settings.asr_partial_window_s) * settings.asr_sample_rate * 2)
        self.buffer_max_bytes = int(
            max(0.0, settings.asr_buffer_max_s) * settings.asr_sample_rate * 2)
        self.work_dir = os.getenv("ASR_WORK_DIR", os.path.join(tempfile.gettempdir(), "moss_asr"))
        self.ready = False
        self.status_message = "ASR not started"
        self._model = None
        self._postprocess: Optional[Callable[[str], str]] = None
        self._lock = threading.Lock()
        # multi-session load management: Runtime wires the live-session count in;
        # >4 sessions → partial re-decodes back off to 1200 ms, and the decode-
        # lock wait metric surfaces contention in /api/status
        self.session_count_provider: Optional[Callable[[], int]] = None
        self.last_lock_wait_ms: float = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        if not self.s.asr_enabled:
            self.status_message = "ASR disabled"
            return
        for path, label in ((self.model_path, "SenseVoice"), (self.vad_model_path, "FSMN-VAD")):
            if not os.path.exists(path):
                self.status_message = f"{label} model not found: {path}"
                log.warning(self.status_message)
                return
        try:
            from funasr import AutoModel  # type: ignore
            from funasr.utils.postprocess_utils import rich_transcription_postprocess  # type: ignore

            os.makedirs(self.work_dir, exist_ok=True)
            self._model = AutoModel(
                model=self.model_path,
                vad_model=self.vad_model_path,
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                disable_update=True,
                disable_pbar=True,
            )
            self._postprocess = rich_transcription_postprocess
            self.ready = True
            self.status_message = "ready"
            log.info("FunASR SenseVoice loaded on %s (fp16=%s)", self.device, self.fp16)
            if self.s.asr_warmup:
                self._warmup()
        except Exception as exc:  # noqa: BLE001
            self.ready = False
            self.status_message = f"FunASR SenseVoice unavailable: {exc}"
            log.warning(self.status_message)

    def _warmup(self) -> None:
        try:
            silence = b"\x00\x00" * (self.sample_rate // 2)  # 0.5 s
            t0 = time.monotonic()
            self.transcribe_pcm(silence, allow_short=True)
            log.info("ASR warmup decode done in %.3fs", time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            log.warning("ASR warmup failed (non-fatal): %s", exc)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.s.asr_enabled,
            "ready": self.ready,
            "provider": "funasr_sensevoice",
            "model": self.model_path,
            "device": self.device,
            "fp16": self.fp16,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "streaming_partials": self.caps.streaming_partials,
            "partial_interval_ms": self.effective_partial_interval_ms(),
            "partial_window_s": round(self.partial_window_bytes / (self.sample_rate * 2), 1),
            "asr_queue_wait_ms": self.last_lock_wait_ms,
            "message": self.status_message,
        }

    def effective_partial_interval_ms(self) -> int:
        """Base interval, backed off to ≥1200 ms when >4 sessions are live —
        partial re-decodes are the adapter's only superlinear GPU cost."""
        interval = self.partial_interval_ms
        if interval <= 0 or self.session_count_provider is None:
            return interval
        try:
            if int(self.session_count_provider() or 0) > 4:
                return max(interval, 1200)
        except Exception:  # noqa: BLE001
            pass
        return interval

    def open_stream(self, on_partial: Optional[Callable[[str], None]] = None) -> _SenseVoiceStream:
        if not self.ready:
            raise RuntimeError(self.status_message or "ASR is not ready")
        return _SenseVoiceStream(self, on_partial=on_partial)

    # ---- decode ----

    def transcribe_pcm(self, pcm: bytes, allow_short: bool = False) -> str:
        if not self.ready or self._model is None:
            raise RuntimeError(self.status_message or "ASR is not ready")
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not allow_short and len(pcm) < self.min_pcm_bytes:
            return ""
        fd, wav_path = tempfile.mkstemp(prefix="sensevoice_", suffix=".wav", dir=self.work_dir)
        os.close(fd)
        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm)
            lock_wait_t0 = time.monotonic()
            with self._lock:
                self.last_lock_wait_ms = round((time.monotonic() - lock_wait_t0) * 1000.0, 1)
                result = self._model.generate(
                    input=wav_path,
                    cache={},
                    language=self.language,
                    use_itn=self.use_itn,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                    fp16=self.fp16,
                )
            return self._extract_text(result)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _extract_text(self, result: Any) -> str:
        text = ""
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                text = str(first.get("text") or first.get("sentence") or "")
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("sentence") or "")
        else:
            text = str(result or "")
        text = text.strip()
        if self._postprocess and text:
            try:
                text = self._postprocess(text)
            except Exception:  # noqa: BLE001
                pass
        return text.strip()
