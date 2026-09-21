"""OpenAI-compatible /v1/audio/speech streaming-PCM client (vLLM-Omni engines).

The backend spawns `vllm serve <ckpt> --omni` per placement TtsSpec
(server/sidecars.py) and engines here are the OpenAI audio/speech clients.
Continuous batching replaces the pytorch sidecars' serialize-per-job behavior,
so bursts (8 sessions speaking + read-alouds) ride one engine's batch dimension.

Shared by every vLLM-Omni-served TTS model (MOSS-TTS-Nano, Fun-CosyVoice3,
MOSS-TTS-Realtime). Voice cloning travels as a base64 data-URL `ref_audio`
built from the Nano package's builtin prompt WAVs; models that also require a
reference transcript (CosyVoice3) add `ref_text` via `extra_payload()`.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
import urllib.request
from typing import Any, Dict, Iterable, Optional

from ....config import Settings
from ....logging_conf import get_logger
from . import read_chunk_bytes, stream_read

log = get_logger(__name__)

# name → prompt file, mirrored from the Nano runtime's _DEFAULT_VOICE_FILES
# (moss_tts_nano_runtime.py) — the engines have no voice registry of their own.
VOICE_PROMPT_FILES: Dict[str, str] = {
    "Junhao": "zh_1.wav",
    "Zhiming": "zh_2.wav",
    "Weiguo": "zh_5.wav",
    "Xiaoyu": "zh_3.wav",
    "Yuewen": "zh_4.wav",
    "Lingyu": "zh_6.wav",
    "Trump": "en_1.wav",
    "Ava": "en_2.wav",
    "Bella": "en_3.wav",
    "Adam": "en_4.wav",
    "Nathan": "en_5.wav",
    "Sakura": "jp_1.mp3",
    "Yui": "jp_2.wav",
    "Aoi": "jp_3.wav",
    "Hina": "jp_4.wav",
    "Mei": "jp_5.wav",
}


class OpenAiSpeechEngine:
    provider_name = "openai_speech"
    # env knobs for sample-rate/channel defaults; subclasses override the names
    sample_rate_env = "TTS_VLLM_SAMPLE_RATE"
    channels_env = "TTS_VLLM_CHANNELS"
    default_sample_rate = 48000
    default_channels = 1
    warmup_env = "TTS_VLLM_WARMUP"
    send_ref_audio = True
    # Official generation params, sent via the OpenAICreateSpeechRequest
    # `extra_params` field (→ the model's SamplingParams.extra_args). Subclasses
    # set the per-model values; a JSON env override (sampling_env) replaces them.
    # NOTE: vllm-omni's MOSS talkers HARD-CODE the audio-channel sampler and do
    # not read extra_args, so on those engines this aligns the backbone/calling
    # but the audio sampling stays fixed — clean audio needs a provider whose
    # sampler matches its card (moss_tts_realtime) or the pytorch nano sidecar.
    sampling: Dict[str, Any] = {}
    sampling_env = ""

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.moss_tts_nano_base_url).rstrip("/")
        self.model = self._served_model_name(settings)
        self.voice = settings.tts_voice
        self.prompt_dir = settings.tts_voice_prompt_dir
        self.sample_rate = int(os.getenv(self.sample_rate_env, str(self.default_sample_rate)))
        self.channels = int(os.getenv(self.channels_env, str(self.default_channels)))
        self.enabled = settings.tts_enabled
        self.ready = False
        self.status_message = "TTS not started"
        self.start_timeout = float(os.getenv("MOSS_TTS_NANO_START_TIMEOUT", "15"))
        self.stream_timeout = float(os.getenv("MOSS_TTS_NANO_STREAM_TIMEOUT", "300"))
        self.read_bytes = read_chunk_bytes()
        self._ref_audio_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._sampling = self._resolve_sampling()

    def _resolve_sampling(self) -> Dict[str, Any]:
        """Official sampling params, with an optional JSON env override."""
        params = dict(self.sampling)
        raw = os.getenv(self.sampling_env, "") if self.sampling_env else ""
        if raw:
            try:
                override = json.loads(raw)
                if isinstance(override, dict):
                    params.update(override)
            except ValueError:
                log.warning("%s: ignoring non-JSON %s=%r", self.provider_name, self.sampling_env, raw)
        return params

    # ---- per-provider hooks ------------------------------------------------

    def _served_model_name(self, settings: Settings) -> str:
        return settings.tts_vllm_served_name

    def extra_payload(self, text: str, voice: str) -> Dict[str, Any]:
        """Additional /v1/audio/speech fields (e.g. CosyVoice3's ref_text)."""
        return {}

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            self.status_message = "TTS disabled"
            return
        try:
            # vLLM /health is an empty 200 — no JSON payload to read settings from
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                self.ready = 200 <= int(response.status) < 300
            self.status_message = "ready" if self.ready else "unhealthy"
            if self.ready and os.getenv(self.warmup_env, "1").lower() not in {"0", "false", "no"}:
                # one tiny synth so no real request pays cold CUDA-graph capture
                # (asr-tts_research.md T2); failures are non-fatal — log and go
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
                f"{self.provider_name} engine unreachable: {exc}"
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

    # ---- voice prompts -------------------------------------------------------

    def _resolve_prompt_path(self, voice: str) -> str:
        """Requested voice → default voice → any prompt file in the dir.

        The assets checkout ships a subset of the mapped files, so a missing
        prompt degrades to the default voice (like the sidecar's registry of
        existing-files-only) rather than failing the synthesis."""
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
                log.warning("voice '%s' unresolvable; using first prompt in %s: %s",
                            voice, self.prompt_dir, entry)
                return os.path.join(self.prompt_dir, entry)
        raise RuntimeError(f"no voice prompt audio available under {self.prompt_dir}")

    def _ref_audio_data_url(self, voice: str) -> str:
        """Voice name → base64 data-URL of its builtin prompt (cached)."""
        with self._cache_lock:
            cached = self._ref_audio_cache.get(voice)
        if cached:
            return cached
        path = self._resolve_prompt_path(voice)
        with open(path, "rb") as f:
            raw = f.read()
        # sniff the container — the assets ship FLAC bytes behind .wav names
        if raw[:4] == b"fLaC":
            mime = "audio/flac"
        elif raw[:4] == b"RIFF":
            mime = "audio/wav"
        elif raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3"):
            mime = "audio/mpeg"
        else:
            mime = mimetypes.guess_type(path)[0] or "audio/wav"
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        with self._cache_lock:
            self._ref_audio_cache[voice] = data_url
        return data_url

    # ---- synthesis -----------------------------------------------------------

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        if not self.ready:
            raise RuntimeError(self.status_message or f"{self.provider_name} engine is not ready")
        resolved_voice = voice or self.voice
        payload = {
            "model": self.model,
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",  # raw PCM chunks, not SSE
        }
        if self.send_ref_audio:
            # no engine-side presets: the clone prompt IS the voice
            payload["ref_audio"] = self._ref_audio_data_url(resolved_voice)
        if self._sampling:
            # official channel for model-specific sampling (permissive schema →
            # SamplingParams.extra_args); merged so extra_payload can extend it
            payload["extra_params"] = dict(self._sampling)
        payload.update(self.extra_payload(text, resolved_voice))
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.stream_timeout) as response:
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
