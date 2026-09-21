"""HTTP client for the in-repo TTS sidecar streaming protocol.

The protocol (originating in the MOSS-TTS-Nano board sidecar, now the shared
contract for every uvicorn TTS sidecar in this repo — moss_tts_nano,
cosyvoice3_native, moss_tts_realtime_native):

    GET  /health                                → {ready, sample_rate, channels, provider?}
    POST /api/generate-stream/start (form)      → {stream_id, audio_url, sample_rate?, channels?}
    GET  /api/generate-stream/{id}/audio        → raw PCM16LE chunks
                                                  (X-Audio-Sample-Rate / X-Audio-Channels headers)
    POST /api/generate-stream/{id}/close

Because the model is out-of-process, the sidecar's torch/deps never collide
with the backend venv. Subclasses override `provider_name` and `form_fields()`.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, Optional

from ....config import Settings
from ....logging_conf import get_logger
from . import read_chunk_bytes, stream_read

log = get_logger(__name__)


class NanoProtocolEngine:
    provider_name = "nano_protocol_http"

    def __init__(self, settings: Settings, base_url: Optional[str] = None,
                 sample_rate: Optional[int] = None, channels: Optional[int] = None):
        # explicit base_url = one engine per pooled sidecar (gpu/supervisor.py)
        self.base_url = (base_url or settings.moss_tts_nano_base_url).rstrip("/")
        self.voice = settings.tts_voice
        self.sample_rate = sample_rate if sample_rate is not None else settings.tts_sample_rate
        self.channels = channels if channels is not None else settings.tts_channels
        self.enabled = settings.tts_enabled
        self.ready = False
        self.status_message = "TTS not started"
        self.start_timeout = float(os.getenv("MOSS_TTS_NANO_START_TIMEOUT", "15"))
        self.stream_timeout = float(os.getenv("MOSS_TTS_NANO_STREAM_TIMEOUT", "300"))
        self.read_bytes = read_chunk_bytes()

    # ---- per-provider hook ------------------------------------------------

    def form_fields(self, text: str, voice: Optional[str]) -> Dict[str, Any]:
        """Form fields for /api/generate-stream/start; subclasses extend."""
        return {"text": text, "voice": voice or self.voice}

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            self.status_message = "TTS disabled"
            return
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
                ok = 200 <= int(response.status) < 300 and bool(payload.get("ready", True))
            self.ready = ok
            self.status_message = "ready" if ok else str(payload)
            self.sample_rate = int(payload.get("sample_rate") or self.sample_rate)
            self.channels = int(payload.get("channels") or self.channels)
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

    # ---- synthesis ---------------------------------------------------------

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        if not self.ready:
            raise RuntimeError(self.status_message or f"{self.provider_name} is not ready")
        stream_id = ""
        try:
            start_payload = self._post_form(
                "/api/generate-stream/start", self.form_fields(text, voice),
                timeout=self.start_timeout)
            if start_payload.get("error"):
                raise RuntimeError(str(start_payload.get("error")))
            stream_id = str(start_payload.get("stream_id") or "")
            audio_url = str(start_payload.get("audio_url") or f"/api/generate-stream/{stream_id}/audio")
            self.sample_rate = int(start_payload.get("sample_rate") or self.sample_rate)
            self.channels = int(start_payload.get("channels") or self.channels)
            url = audio_url if audio_url.startswith("http") else f"{self.base_url}{audio_url}"
            with urllib.request.urlopen(url, timeout=self.stream_timeout) as response:
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
            if stream_id:
                self._close_stream(stream_id)

    def _post_form(self, path: str, fields: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        data = urllib.parse.urlencode(fields).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body or "{}")

    def _close_stream(self, stream_id: str) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate-stream/{urllib.parse.quote(stream_id)}/close",
            data=b"",
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=1.0).close()
        except Exception:  # noqa: BLE001
            pass
