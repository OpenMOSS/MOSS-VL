"""ElevenLabs streaming TTS engine (external API — no GPU, no sidecar).

Wire shape: one HTTP request per ready-cut segment against
`POST /v1/text-to-speech/{voice_id}/stream?output_format=pcm_*` — the response
IS the raw little-endian PCM16 mono chunk stream, which is exactly the
`synthesize_pcm(text, voice) -> Iterable[bytes]` contract every local engine
already serves (see common/openai_speech.py for the in-cluster twin).

Transport: stdlib http.client over an optional SOCKS5 CONNECT tunnel
(ELEVENLABS_PROXY=socks5://host:port) — no third-party dependency; boxes behind
a firewall reach the API through the ssh -D relay. Empty proxy = direct TLS.

Design choices:
- PCM output, not mp3: zero client-side decode; the session plane relays this
  engine's `sample_rate`/`channels` to the player in tts_turn_start, so the
  free-tier-capped pcm_24000 needs no resampling anywhere.
- `eleven_flash_v2_5` default: lowest TTFB of the lineup and Chinese-capable;
  `eleven_multilingual_v2` is the quality fallback (config: ELEVENLABS_MODEL_ID).
- No lease counting: concurrency is ElevenLabs' problem, not ours — acquire()
  hands every session the same stateless engine.
- The API key never crosses into status(), logs, or the frontend.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
import threading
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ....config import Settings
from ....logging_conf import get_logger
from ..common import read_chunk_bytes

log = get_logger(__name__)


def _make_ssl_context() -> ssl.SSLContext:
    """Explicitly-pinned CA bundle. The demo process prepends bundled
    ffmpeg/torch dirs to LD_LIBRARY_PATH, so _ssl links a CONDA OpenSSL whose
    compiled-in CA path (<conda-prefix>/ssl/cert.pem) does not exist — the
    default context then verifies nothing (CERTIFICATE_VERIFY_FAILED)."""
    for cafile in ("/usr/lib/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
        if os.path.isfile(cafile):
            return ssl.create_default_context(cafile=cafile)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — last resort: the process default
        return ssl.create_default_context()


_SSL_CONTEXT = _make_ssl_context()


def _parse_socks5_proxy(url: str) -> Optional[Tuple[str, int]]:
    """socks5://host:port → (host, port); anything else → None (direct)."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url if "://" in url else f"socks5://{url}")
    if parsed.scheme not in ("socks5", "socks5h"):
        raise ValueError(f"ELEVENLABS_PROXY must be socks5://host:port (got {url!r})")
    return (parsed.hostname or "127.0.0.1", parsed.port or 1080)


class _HTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that can reach the origin through a SOCKS5 CONNECT
    tunnel (RFC 1928, no-auth) before the TLS handshake."""

    socks5: Optional[Tuple[str, int]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["context"] = _SSL_CONTEXT  # pinned CA bundle, see _make_ssl_context
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        if not self.socks5:
            return super().connect()
        sock = socket.create_connection(self.socks5, timeout=self.timeout or 30.0)
        host = self.host.encode("idna")
        # greeting: VER=5, one method (no-auth); server must pick it (0x00)
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            sock.close()
            raise OSError("SOCKS5 proxy refused no-auth negotiation")
        port = self.port or 443
        req = (b"\x05\x01\x00\x03" + bytes([len(host)]) + host
               + port.to_bytes(2, "big"))
        sock.sendall(req)
        reply = sock.recv(10)  # VER REP RSV ATYP(=1 ipv4) BND.ADDR BND.PORT
        if len(reply) < 2 or reply[1] != 0x00:
            sock.close()
            raise OSError(f"SOCKS5 CONNECT failed (rep={reply[1] if len(reply) > 1 else '?'})")
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class ElevenLabsEngine:
    provider_name = "elevenlabs"
    channels = 1  # every pcm_* output_format is mono

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.elevenlabs_api_key.strip()
        self.base_url = settings.elevenlabs_base_url.rstrip("/")
        self.model_id = settings.elevenlabs_model_id
        self.output_format = settings.elevenlabs_output_format
        # pcm_24000 → 24000; anything non-pcm (mp3_*, ulaw_*) would need a
        # decoder we deliberately do not ship — config validation catches it
        self.sample_rate = self._parse_sample_rate(self.output_format)
        self.voice = settings.elevenlabs_voice_id  # empty → first account voice
        self.warmup = settings.elevenlabs_warmup
        self.stability = settings.elevenlabs_stability
        # per-SESSION previous_text for cross-segment prosody continuity: each
        # TtsSession owns one worker thread, so thread-local state lines up
        # exactly with a session's segment stream (the engine itself is shared)
        self._tls = threading.local()
        self.socks5 = _parse_socks5_proxy(settings.elevenlabs_proxy)
        self.enabled = settings.tts_enabled and bool(self.api_key)
        self.ready = False
        self.status_message = "TTS not started"
        self.voices: List[Dict[str, str]] = []  # account voices, for the UI
        self.start_timeout = 15.0
        self.stream_timeout = 120.0
        self.read_bytes = read_chunk_bytes()
        parsed = urllib.parse.urlparse(self.base_url)
        self._host = parsed.hostname or "api.elevenlabs.io"
        self._port = parsed.port or 443

    @staticmethod
    def _parse_sample_rate(output_format: str) -> int:
        # pcm_24000 | pcm_22050 | pcm_16000 | pcm_44100
        parts = (output_format or "").split("_")
        if len(parts) == 2 and parts[0] == "pcm" and parts[1].isdigit():
            return int(parts[1])
        raise ValueError(
            f"ELEVENLABS_OUTPUT_FORMAT must be pcm_<rate> (got {output_format!r}); "
            "compressed formats would need a decoder the session plane does not have")

    def _connect(self, timeout: float) -> http.client.HTTPSConnection:
        conn = _HTTPSConnection(self._host, self._port, timeout=timeout)
        conn.socks5 = self.socks5
        return conn

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            self.status_message = ("TTS disabled" if not self.api_key
                                   else "elevenlabs: no ELEVENLABS_API_KEY")
            return
        last_exc: Optional[Exception] = None
        # the egress tunnel can be momentarily flaky at boot — retry instead of
        # writing the lane off for the process's whole lifetime
        for attempt, backoff in enumerate((0.0, 2.0, 5.0)):
            if backoff:
                time.sleep(backoff)
            try:
                self._fetch_voices()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 — external API must never block boot
                last_exc = exc
                log.debug("elevenlabs voices probe attempt %d failed: %s", attempt + 1, exc)
        if last_exc is not None:
            self.ready = False
            self.status_message = f"elevenlabs unreachable: {last_exc}"
            log.warning("elevenlabs: %s", self.status_message)
            return
        if self.warmup:
            # one tiny synth so no real request pays TLS+model cold start;
            # OFF by default — it spends credits on every boot
            try:
                t0 = time.monotonic()
                n = sum(len(c) for c in self.synthesize_pcm("Hello."))
                log.info("elevenlabs warmup: %d PCM bytes in %.1fs", n, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                log.warning("elevenlabs warmup failed: %s", exc)

    def _fetch_voices(self) -> None:
        """GET /v1/voices: health gate + the account's voice list for the UI."""
        conn = self._connect(self.start_timeout)
        try:
            t0 = time.monotonic()
            conn.request("GET", "/v1/voices", headers={"xi-api-key": self.api_key})
            response = conn.getresponse()
            payload = json.loads(response.read() or b"{}")
            if response.status != 200:
                raise RuntimeError(f"GET /v1/voices → HTTP {response.status}: {payload}")
            self.voices = [
                {"voice_id": str(v.get("voice_id") or ""), "name": str(v.get("name") or "")}
                for v in payload.get("voices") or [] if v.get("voice_id")
            ]
            if not self.voices:
                raise RuntimeError("account has no voices")
            if not self.voice:
                self.voice = self.voices[0]["voice_id"]
            self.ready = True
            self.status_message = "ready"
            log.info("elevenlabs ready: %d account voices, default %s, model %s (%dHz%s) "
                     "in %.1fs", len(self.voices), self.voice, self.model_id,
                     self.sample_rate, ", via socks5" if self.socks5 else "",
                     time.monotonic() - t0)
        finally:
            conn.close()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "provider": self.provider_name,
            "model": self.model_id,
            "voice": self.voice,
            "voices": self.voices,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "message": self.status_message,
        }

    # ---- synthesis -------------------------------------------------------------

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        if not self.ready:
            raise RuntimeError(self.status_message or "elevenlabs engine is not ready")
        voice_id = (voice or self.voice).strip()
        path = (f"/v1/text-to-speech/{voice_id}/stream"
                f"?output_format={urllib.parse.quote(self.output_format)}")
        payload: Dict[str, Any] = {
            "text": text,
            "model_id": self.model_id,
            # pinned per request: unset floats with the voice's stored defaults
            # and reads as timbre drift across segments
            "voice_settings": {"stability": self.stability},
        }
        prev = getattr(self._tls, "prev_text", None)
        if prev:
            # cross-segment prosody continuity (docs: previous_text is the
            # sanctioned way to concatenate multiple generations)
            payload["previous_text"] = prev
        conn = self._connect(self.stream_timeout)
        try:
            conn.request(
                "POST", path,
                body=json.dumps(payload),
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read()[:200].decode("utf-8", "replace")
                raise RuntimeError(f"elevenlabs HTTP {response.status}: {detail}")
            # http.client decodes chunked transfer framing for us; read1 returns
            # the moment ANY PCM is buffered, so first-chunk latency is the API's
            while True:
                chunk = response.read1(self.read_bytes)
                if not chunk:
                    break
                yield chunk
            # commit continuity only after a full clean read — a failed segment
            # must not poison the next one's context
            self._tls.prev_text = text[-512:]
        finally:
            conn.close()


class ElevenLabsAdapter:
    """Pool-shaped wrapper so routers/sessions.py can treat this exactly like
    the local SidecarPoolAdapter: acquire()/release() are no-ops (the engine is
    stateless per request; concurrency lives on ElevenLabs' side)."""

    def __init__(self, settings: Settings) -> None:
        self.engine = ElevenLabsEngine(settings)
        self._leases = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.engine.start()

    def status(self) -> Dict[str, Any]:
        return self.engine.status()

    def acquire(self) -> ElevenLabsEngine:
        with self._lock:
            self._leases += 1
        return self.engine

    def release(self, engine: Any) -> None:
        with self._lock:
            if engine is self.engine and self._leases > 0:
                self._leases -= 1
