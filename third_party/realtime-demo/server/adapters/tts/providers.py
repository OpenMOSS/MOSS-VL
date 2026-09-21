"""TTS provider taxonomy — the ONE place that classifies provider strings.

Used by registry.build_tts (adapter choice), server/sidecars.py (spawn recipe)
and server/gpu/placement.py (pool sizing). Keep alias sets in sync with the
registry error message.

Two serving shapes:
- vLLM-Omni engines (`vllm serve <ckpt> --omni`, OpenAI /v1/audio/speech,
  continuous batching → sized by TTS_SESSIONS_PER_ENGINE)
- uvicorn nano-protocol sidecars (one process per pool slot, serialize-per-job
  → sized by TTS_SESSIONS_PER_SIDECAR)
"""
from __future__ import annotations

_ALIASES = {
    "moss_tts_nano": {"moss_tts_nano", "moss-nano", "moss_nano", "nano"},
    "vllm_omni": {"vllm_omni", "vllm-omni", "vllm"},
    "cosyvoice3": {"cosyvoice3", "fun_cosyvoice3", "fun-cosyvoice3", "cosyvoice", "cosy"},
    "moss_tts_realtime": {"moss_tts_realtime", "moss-tts-realtime", "mossrt"},
    "cosyvoice3_native": {"cosyvoice3_native", "cosy_native"},
    "moss_tts_realtime_native": {"moss_tts_realtime_native", "mossrt_native"},
    "elevenlabs": {"elevenlabs", "eleven", "11labs", "el"},
    "minimax": {"minimax", "mini_max", "mm"},
}

VLLM_ENGINE_PROVIDERS = {"vllm_omni", "cosyvoice3", "moss_tts_realtime"}
SIDECAR_PROVIDERS = {"moss_tts_nano", "cosyvoice3_native", "moss_tts_realtime_native"}
# external cloud APIs: no GPU in placement, no process to spawn in sidecars
EXTERNAL_PROVIDERS = {"elevenlabs", "minimax"}


def canonical_tts_provider(name: str) -> str:
    """Alias → canonical provider id; unknown strings pass through unchanged
    (registry.build_tts raises the user-facing NotImplementedError)."""
    cleaned = (name or "").strip().lower()
    for canonical, aliases in _ALIASES.items():
        if cleaned in aliases:
            return canonical
    return cleaned


def is_vllm_engine_provider(name: str) -> bool:
    return canonical_tts_provider(name) in VLLM_ENGINE_PROVIDERS


def is_external_provider(name: str) -> bool:
    return canonical_tts_provider(name) in EXTERNAL_PROVIDERS
