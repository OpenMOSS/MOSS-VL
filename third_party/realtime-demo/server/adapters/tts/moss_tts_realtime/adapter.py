"""MOSS-TTS-Realtime (OpenMOSS 1.7B) TTS adapters.

Primary (`moss_tts_realtime`): vLLM-Omni engine — the backend spawns
`vllm serve <MOSS-TTS-Realtime> --omni` (server/sidecars.py; MossTTSRealtime
arch support landed in vLLM-Omni 0.22, CUDA-graph + cross-request fixes by
0.24) and this adapter is the OpenAI /v1/audio/speech streaming-PCM client.
Output is 24 kHz mono PCM16 (MOSS-Audio-Tokenizer); model-card perf: TTFB
~180 ms warm, RTF 0.51 single-stream on an L20.

Voice cloning: `ref_audio` (reference_audio_path semantics upstream). If the
engine build rejects ref_audio, set MOSSRT_SEND_REF_AUDIO=0 to fall back to
the default speaker — the request then carries no clone prompt.

Deferred (documented in sidecar/README.md): the model's context-aware
multi-turn mode (turn-0 KV reset, turn-1+ reuse, 32K ctx) needs a stateful
session protocol that /v1/audio/speech doesn't express; the native sidecar is
the future home of that (and of TtsCaps.token_streaming_input=True).

Fallback (`moss_tts_realtime_native`): client to the UPSTREAM fast_api.py
session server run straight from the vendored repo
(sidecar/third_party/MOSS-TTS/moss_tts_realtime, own .venv-mossrt,
transformers 5.0 + SDPA). Its /tts/session/* protocol is stateful per turn —
one session per synthesize_pcm call today, and the natural surface for the
deferred multi-turn/push_text streaming. Same acquire/release pool surface.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, Iterable, Iterator, Optional

from ....config import Settings
from ....logging_conf import get_logger
from ..common import read_chunk_bytes, stream_read
from ..common.openai_speech import OpenAiSpeechEngine, VOICE_PROMPT_FILES
from ..common.pool import SidecarPoolAdapter

log = get_logger(__name__)


class MossRtStream:
    """One continuous MOSS-TTS-Realtime session for a whole turn (item 4 —
    true incremental streaming). Open once, push text deltas as segments
    arrive, and audio streams from /audio the moment the model has enough —
    NO per-segment cold start, one warm prosody-continuous session.

    Lifecycle: open() → push_text(delta)… → end_input() → audio_chunks()
    drains to EOF → close(). close() also aborts an in-flight turn (barge-in).
    """

    def __init__(self, engine: "MossRtNativeEngine", voice: Optional[str]):
        self.engine = engine
        self.sample_rate = engine.sample_rate
        self.channels = engine.channels
        self.session_id = uuid.uuid4().hex
        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = threading.Event()
        self._reader: Optional[threading.Thread] = None
        engine._post_json("/tts/session/start", {
            "session_id": self.session_id,
            "new_turn": True,
            "prompt_audio": engine._resolve_prompt_path(voice or engine.voice),
        }, timeout=engine.start_timeout)
        self._opened_at = time.monotonic()
        self._reader = threading.Thread(target=self._read_audio, name=f"mossrt-stream-{self.session_id[:8]}", daemon=True)
        self._reader.start()

    def _read_audio(self) -> None:
        # first-chunk latency + throughput telemetry: the ONE number that tells
        # a healthy stream (~3s to first audio on the reference box) from a
        # starving one — session metrics only show ttfa when audio ARRIVES.
        first_at: Optional[float] = None
        total = 0
        try:
            with urllib.request.urlopen(
                f"{self.engine.base_url}/tts/session/{self.session_id}/audio",
                timeout=self.engine.stream_timeout) as response:
                sr = response.headers.get("X-Audio-Sample-Rate")
                ch = response.headers.get("X-Audio-Channels")
                if sr:
                    self.sample_rate = int(sr)
                if ch:
                    self.channels = int(ch)
                while not self._closed.is_set():
                    chunk = stream_read(response, self.engine.read_bytes)
                    if not chunk:
                        break
                    if first_at is None:
                        first_at = time.monotonic()
                        log.info("MOSS-TTS-Realtime stream %s: first audio %.2fs after open",
                                 self.session_id[:8], first_at - self._opened_at)
                    total += len(chunk)
                    self._audio_q.put(chunk)
        except Exception as exc:  # noqa: BLE001 — reader errors end the stream
            if not self._closed.is_set():
                log.warning("MOSS-TTS-Realtime stream reader failed (%s): %s", self.session_id[:8], exc)
        finally:
            dur = time.monotonic() - self._opened_at
            audio_s = total / max(1, 2 * self.channels * self.sample_rate)
            log.info("MOSS-TTS-Realtime stream %s: ended after %.2fs — %d bytes (%.2fs audio%s)",
                     self.session_id[:8], dur, total, audio_s,
                     "" if first_at is not None else ", NO audio ever arrived")
            self._audio_q.put(None)  # EOF sentinel

    def push_text(self, text: str) -> None:
        if self._closed.is_set() or not (text or "").strip():
            return
        self.engine._post_json("/tts/session/push", {
            "session_id": self.session_id, "text": text, "is_final": False,
        }, timeout=self.engine.start_timeout)

    def end_input(self) -> None:
        if self._closed.is_set():
            return
        self.engine._post_json("/tts/session/push", {
            "session_id": self.session_id, "text": "", "is_final": True,
        }, timeout=self.engine.start_timeout)

    def audio_chunks(self) -> Iterator[bytes]:
        while True:
            item = self._audio_q.get()
            if item is None:
                break
            yield item

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.engine._post_json("/tts/session/close", {"session_id": self.session_id}, timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._audio_q.put_nowait(None)  # unblock a waiting audio_chunks()
        except queue.Full:
            pass


class MossRtSpeechEngine(OpenAiSpeechEngine):
    provider_name = "moss_tts_realtime_speech"
    sample_rate_env = "MOSSRT_SAMPLE_RATE"
    channels_env = "MOSSRT_CHANNELS"
    default_sample_rate = 24000
    default_channels = 1
    warmup_env = "MOSSRT_WARMUP"
    # official MOSS-TTS-Realtime model-card params. vllm-omni's
    # MossTTSRealtimeTalker already samples with these + a 50-frame repetition
    # window internally, so this is a belt-and-suspenders alignment of the call.
    sampling = {"temperature": 0.8, "top_p": 0.6, "top_k": 30, "repetition_penalty": 1.1}
    sampling_env = "MOSSRT_SAMPLING"

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        super().__init__(settings, base_url)
        self.send_ref_audio = os.getenv("MOSSRT_SEND_REF_AUDIO", "1").lower() not in {"0", "false", "no"}

    def _served_model_name(self, settings: Settings) -> str:
        return settings.tts_mossrt_served_name


class MossRtVllmAdapter(SidecarPoolAdapter):
    engine_cls = MossRtSpeechEngine


class MossRtNativeEngine:
    """Client for the vendored upstream fast_api.py session server.

    Protocol per synthesis: POST /tts/session/start (new_turn, prompt_audio =
    the voice's builtin prompt WAV path) → POST /tts/session/push (full text,
    is_final) → GET /tts/session/{sid}/audio (PCM16LE stream, X-Audio-*
    headers) → POST /tts/session/close.
    """

    provider_name = "moss_tts_realtime_native_http"
    supports_streaming = True  # true incremental push_text streaming (item 4)

    def open_stream(self, voice: Optional[str] = None) -> MossRtStream:
        """Open a per-turn streaming session (TtsSession uses this when the
        engine advertises supports_streaming)."""
        if not self.ready:
            raise RuntimeError(self.status_message or f"{self.provider_name} is not ready")
        return MossRtStream(self, voice)

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.moss_tts_nano_base_url).rstrip("/")
        self.voice = settings.tts_voice
        self.prompt_dir = settings.tts_voice_prompt_dir
        self.sample_rate = int(os.getenv("MOSSRT_SAMPLE_RATE", "24000"))
        self.channels = int(os.getenv("MOSSRT_CHANNELS", "1"))
        self.enabled = settings.tts_enabled
        self.ready = False
        self.status_message = "TTS not started"
        self.start_timeout = float(os.getenv("MOSS_TTS_NANO_START_TIMEOUT", "15"))
        self.stream_timeout = float(os.getenv("MOSS_TTS_NANO_STREAM_TIMEOUT", "300"))
        self.read_bytes = read_chunk_bytes()

    def start(self) -> None:
        if not self.enabled:
            self.status_message = "TTS disabled"
            return
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                self.ready = 200 <= int(response.status) < 300
            self.status_message = "ready" if self.ready else "unhealthy"
            if self.ready and os.getenv("MOSSRT_NATIVE_WARMUP", "1").lower() not in {"0", "false", "no"}:
                try:
                    t0 = time.monotonic()
                    n = sum(len(c) for c in self.synthesize_pcm("你好。"))
                    log.info("%s warmup at %s: %d PCM bytes in %.1fs",
                             self.provider_name, self.base_url, n, time.monotonic() - t0)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s warmup failed at %s: %s", self.provider_name, self.base_url, exc)
        except Exception as exc:  # noqa: BLE001
            self.ready = os.getenv("VOICE_TTS_SKIP_HEALTHCHECK", "").lower() in {"1", "true", "yes"}
            self.status_message = "ready without healthcheck" if self.ready else \
                f"{self.provider_name} sidecar unreachable: {exc}"
            if not self.ready:
                log.warning(self.status_message)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "provider": self.provider_name,
            "base_url": self.base_url,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "message": self.status_message,
        }

    def _resolve_prompt_path(self, voice: str) -> str:
        """Requested voice → default voice → any prompt file (same degrade
        order as the OpenAI-speech engines)."""
        for name in (voice, self.voice):
            filename = VOICE_PROMPT_FILES.get(name)
            if filename:
                path = os.path.join(self.prompt_dir, filename)
                if os.path.isfile(path):
                    if name != voice:
                        log.warning("voice '%s' has no prompt file; using '%s'", voice, name)
                    return path
        for entry in sorted(os.listdir(self.prompt_dir) if os.path.isdir(self.prompt_dir) else []):
            if entry.lower().endswith((".wav", ".mp3", ".flac")):
                return os.path.join(self.prompt_dir, entry)
        raise RuntimeError(f"no voice prompt audio available under {self.prompt_dir}")

    def _post_json(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        if not self.ready:
            raise RuntimeError(self.status_message or f"{self.provider_name} is not ready")
        session_id = uuid.uuid4().hex
        started = False
        try:
            self._post_json("/tts/session/start", {
                "session_id": session_id,
                "new_turn": True,
                "prompt_audio": self._resolve_prompt_path(voice or self.voice),
            }, timeout=self.start_timeout)
            started = True
            self._post_json("/tts/session/push", {
                "session_id": session_id,
                "text": text,
                "is_final": True,
            }, timeout=self.start_timeout)
            with urllib.request.urlopen(f"{self.base_url}/tts/session/{session_id}/audio",
                                        timeout=self.stream_timeout) as response:
                sr = response.headers.get("X-Audio-Sample-Rate")
                ch = response.headers.get("X-Audio-Channels")
                if sr:
                    self.sample_rate = int(sr)
                if ch:
                    self.channels = int(ch)
                while True:
                    chunk = stream_read(response, self.read_bytes)
                    if not chunk:
                        break
                    yield chunk
        finally:
            if started:
                try:
                    self._post_json("/tts/session/close", {"session_id": session_id}, timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass


class MossRtNativeAdapter(SidecarPoolAdapter):
    engine_cls = MossRtNativeEngine
    token_streaming_input = True  # streams a whole turn through one session
