"""MiniMax T2A v2 streaming TTS engine (external API — no GPU, no sidecar).

Wire shape: one HTTP POST per ready-cut segment against
`POST /v1/t2a_v2[?GroupId=…]` with `"stream": true` — the response is SSE
(`data: {...}` events) whose `data.audio` fields carry HEX-encoded PCM16 mono
chunks. We decode event-by-event and yield raw PCM, matching the
`synthesize_pcm(text, voice) -> Iterable[bytes]` contract every engine serves.

Why MiniMax alongside ElevenLabs: the speech-02 family is the strongest
Chinese-language cloud TTS (Artificial Analysis / HF TTS Arena), and the
mainland endpoint (api.minimaxi.chat) is directly reachable from this box —
no tunnel, no LD_LIBRARY_PATH/CA quirks, ~1s class first-chunk latency.

Transport reuses the ElevenLabs adapter's pinned-CA HTTPSConnection (with the
optional SOCKS5 knob for the global api.minimax.io endpoint).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional

from ....config import Settings
from ....logging_conf import get_logger
from ..common import read_chunk_bytes
from ..elevenlabs.adapter import _HTTPSConnection, _parse_socks5_proxy

log = get_logger(__name__)

# preset voice registry (voice_id → label). MiniMax ships many more; this is
# the curated Chinese-first shortlist the UI offers. start() replaces it with
# the account's system-voice list when /v1/get_voice answers.
PRESET_VOICES: List[Dict[str, str]] = [
    {"voice_id": "male-qn-qingse", "name": "青涩青年（男）"},
    {"voice_id": "male-qn-jingying", "name": "精英青年（男）"},
    {"voice_id": "male-qn-badao", "name": "霸道青年（男）"},
    {"voice_id": "male-qn-daxuesheng", "name": "大学生（男）"},
    {"voice_id": "female-shaonv", "name": "少女（女）"},
    {"voice_id": "female-yujie", "name": "御姐（女）"},
    {"voice_id": "female-chengshu", "name": "成熟（女）"},
    {"voice_id": "female-tianmei", "name": "甜美（女）"},
    {"voice_id": "presenter_male", "name": "主持人（男）"},
    {"voice_id": "presenter_female", "name": "主持人（女）"},
    {"voice_id": "audiobook_male_1", "name": "有声书（男）"},
    {"voice_id": "audiobook_female_1", "name": "有声书（女）"},
]


class MiniMaxEngine:
    provider_name = "minimax"
    channels = 1  # audio_setting.channel=1

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.minimax_api_key.strip()
        self.base_url = settings.minimax_base_url.rstrip("/")
        self.group_id = settings.minimax_group_id.strip()
        self.model = settings.minimax_model
        self.sample_rate = settings.minimax_sample_rate
        self.voice = settings.minimax_voice_id
        self.warmup = settings.minimax_warmup
        self.socks5 = _parse_socks5_proxy(settings.minimax_proxy)
        self.enabled = settings.tts_enabled and bool(self.api_key)
        self.ready = False
        self.status_message = "TTS not started"
        self.voices: List[Dict[str, str]] = list(PRESET_VOICES)
        self.start_timeout = 15.0
        self.stream_timeout = 120.0
        self.read_bytes = read_chunk_bytes()
        parsed = urllib.parse.urlparse(self.base_url)
        self._host = parsed.hostname or "api.minimaxi.chat"
        self._port = parsed.port or 443

    def _connect(self, timeout: float) -> _HTTPSConnection:
        conn = _HTTPSConnection(self._host, self._port, timeout=timeout)
        conn.socks5 = self.socks5
        return conn

    def _path(self, suffix: str) -> str:
        if self.group_id:
            return f"{suffix}?GroupId={urllib.parse.quote(self.group_id)}"
        return suffix

    def _post(self, conn: _HTTPSConnection, path: str, payload: Dict[str, Any]):
        conn.request(
            "POST", path,
            body=json.dumps(payload),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        return conn.getresponse()

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            self.status_message = ("TTS disabled" if not self.api_key
                                   else "minimax: no MINIMAX_API_KEY")
            return
        last_exc: Optional[Exception] = None
        for attempt, backoff in enumerate((0.0, 2.0, 5.0)):
            if backoff:
                time.sleep(backoff)
            try:
                self._probe()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 — external API must never block boot
                last_exc = exc
                log.debug("minimax probe attempt %d failed: %s", attempt + 1, exc)
        if last_exc is not None:
            self.ready = False
            self.status_message = f"minimax unreachable: {last_exc}"
            log.warning("minimax: %s", self.status_message)
            return

    def _probe(self) -> None:
        """Health gate = one 1-char synth (auth + model + egress all proven;
        ~1 char of credits per boot). get_voice is tried for the account's
        voice list but is non-fatal — the mainland endpoint just hangs on it,
        so the curated PRESET_VOICES are the usual list."""
        try:
            self._fetch_voice_list()
        except Exception as exc:  # noqa: BLE001
            log.debug("minimax get_voice skipped (%s) — keeping preset voices", exc)
        t0 = time.monotonic()
        n = sum(len(c) for c in self._synthesize("。", probe=True))
        self.ready = True
        self.status_message = "ready"
        log.info("minimax ready: %d voices, default %s, model %s (%dHz%s); "
                 "probe %d PCM bytes in %.1fs", len(self.voices), self.voice,
                 self.model, self.sample_rate, ", via socks5" if self.socks5 else "",
                 n, time.monotonic() - t0)
        if self.warmup:
            try:
                n = sum(len(c) for c in self._synthesize("你好。"))
                log.info("minimax warmup: %d PCM bytes", n)
            except Exception as exc:  # noqa: BLE001
                log.warning("minimax warmup failed: %s", exc)

    def _fetch_voice_list(self) -> None:
        conn = self._connect(8.0)  # hangs on the mainland endpoint — short leash
        try:
            response = self._post(conn, self._path("/v1/get_voice"), {"voice_type": "all"})
            payload = json.loads(response.read() or b"{}")
            base = payload.get("base_resp") or {}
            if response.status != 200 or base.get("status_code", 0) != 0:
                raise RuntimeError(f"HTTP {response.status} {base.get('status_msg') or payload}")
            system = [
                {"voice_id": str(v.get("voice_id") or ""), "name": str(v.get("voice_name") or v.get("voice_id") or "")}
                for v in payload.get("system_voice") or [] if v.get("voice_id")
            ]
            if system:
                self.voices = system
        finally:
            conn.close()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "provider": self.provider_name,
            "model": self.model,
            "voice": self.voice,
            "voices": self.voices,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "message": self.status_message,
        }

    # ---- synthesis -------------------------------------------------------------

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        if not self.ready:
            raise RuntimeError(self.status_message or "minimax engine is not ready")
        return self._synthesize(text, voice)

    def _synthesize(self, text: str, voice: Optional[str] = None,
                    probe: bool = False) -> Iterable[bytes]:
        payload = {
            "model": self.model,
            "text": text,
            "stream": True,
            "voice_setting": {"voice_id": (voice or self.voice).strip()},
            "audio_setting": {"sample_rate": self.sample_rate, "format": "pcm", "channel": 1},
        }
        # the boot probe gets a short leash: an auth-rejected stream is a plain
        # JSON body the server may not close promptly, and the full stream
        # timeout (120s x retries) would stall the whole backend lifespan
        conn = self._connect(self.start_timeout if probe else self.stream_timeout)
        try:
            response = self._post(conn, self._path("/v1/t2a_v2"), payload)
            content_type = response.headers.get("Content-Type") or ""
            if response.status != 200 or "event-stream" not in content_type:
                detail = response.read()[:200].decode("utf-8", "replace")
                raise RuntimeError(f"minimax HTTP {response.status}: {detail}")
            yield from self._stream_sse_pcm(response)
        finally:
            conn.close()

    def _stream_sse_pcm(self, response) -> Iterable[bytes]:
        """Incrementally parse the SSE body: `data: {json}` events, hex PCM in
        data.audio; data.status==2 marks the tail; base_resp!=0 is an API error.

        The final (status==2) event REPEATS the full audio concatenated —
        yield it only when no status==1 chunks ever arrived (else every
        segment plays twice)."""
        buf = b""
        streamed = False
        while True:
            piece = response.read1(self.read_bytes)
            if not piece:
                break
            buf += piece
            while b"\n\n" in buf or b"\r\n\r\n" in buf:
                # events are separated by a blank line (\n\n or \r\n\r\n)
                idx_n = buf.find(b"\n\n")
                idx_r = buf.find(b"\r\n\r\n")
                if idx_r != -1 and (idx_n == -1 or idx_r <= idx_n):
                    raw, buf = buf[:idx_r], buf[idx_r + 4:]
                else:
                    raw, buf = buf[:idx_n], buf[idx_n + 2:]
                for chunk, is_final in self._handle_event(raw):
                    if is_final and streamed:
                        continue  # full-audio repeat of chunks already yielded
                    streamed = True
                    yield chunk
        if buf.strip():
            for chunk, is_final in self._handle_event(buf):
                if is_final and streamed:
                    continue
                yield chunk

    @staticmethod
    def _handle_event(raw: bytes) -> Iterable[tuple[bytes, bool]]:
        """(pcm, is_final) per audio field. is_final = the event is the
        status==2 tail, whose audio duplicates everything already streamed."""
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:].strip() or b"{}")
            except ValueError:
                continue
            base = event.get("base_resp") or {}
            if base.get("status_code", 0) != 0:
                raise RuntimeError(f"minimax stream error {base.get('status_code')}: "
                                   f"{base.get('status_msg')}")
            data = event.get("data") or {}
            hex_audio = data.get("audio")
            if hex_audio:
                yield bytes.fromhex(hex_audio), data.get("status") == 2
            if data.get("status") == 2:
                return


class MiniMaxAdapter:
    """Pool-shaped wrapper (same acquire/release surface as the local pool);
    the engine is stateless per request — concurrency lives on MiniMax' side."""

    def __init__(self, settings: Settings) -> None:
        self.engine = MiniMaxEngine(settings)
        self._leases = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.engine.start()

    def status(self) -> Dict[str, Any]:
        return self.engine.status()

    def acquire(self) -> MiniMaxEngine:
        with self._lock:
            self._leases += 1
        return self.engine

    def release(self, engine: Any) -> None:
        with self._lock:
            if engine is self.engine and self._leases > 0:
                self._leases -= 1
