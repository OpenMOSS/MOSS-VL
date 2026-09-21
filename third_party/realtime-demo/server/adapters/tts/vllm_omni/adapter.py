"""vLLM-Omni TTS adapter for the Nano weights.

Same Nano weights, faster engine: the backend spawns `vllm serve <ckpt> --omni`
per placement TtsSpec (server/sidecars.py) and this adapter is the OpenAI
/v1/audio/speech client (../common/openai_speech.py, shared with the
cosyvoice3 / moss_tts_realtime providers).

Nano-specific contract (verified against vLLM-Omni ≥ 0.24 docs, 2026-07-11):
- voice cloning REQUIRES `ref_audio` (no builtin presets in the engine) — the
  16 demo voices map to the Nano package's builtin prompt WAVs and travel as
  base64 data-URLs; `ref_text`/`voice` are accepted but ignored upstream.
- streamed output is 48 kHz MONO PCM16 (the pytorch sidecar emits stereo); the
  session contract already carries sample_rate/channels per chunk.
"""
from __future__ import annotations

from ..common.openai_speech import VOICE_PROMPT_FILES, OpenAiSpeechEngine  # noqa: F401 (map re-exported for callers)
from ..common.pool import SidecarPoolAdapter


class VllmOmniSpeechEngine(OpenAiSpeechEngine):
    provider_name = "vllm_omni_speech"
    sample_rate_env = "TTS_VLLM_SAMPLE_RATE"
    channels_env = "TTS_VLLM_CHANNELS"
    default_sample_rate = 48000
    default_channels = 1
    # official MOSS-TTS-Nano quality preset (OpenMOSS / MOSI.AI): temp 0.8,
    # top_p 0.95, top_k 25, repetition_penalty 1.2 — the pytorch nano sidecar
    # uses these. See the extra_args caveat in OpenAiSpeechEngine.
    sampling = {"temperature": 0.8, "top_p": 0.95, "top_k": 25, "repetition_penalty": 1.2}
    sampling_env = "TTS_VLLM_SAMPLING"


class VllmOmniAdapter(SidecarPoolAdapter):
    """Sidecar-pool semantics (acquire/release, least-loaded) over vLLM engines."""

    engine_cls = VllmOmniSpeechEngine
