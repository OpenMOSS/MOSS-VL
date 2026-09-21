"""MOSS-TTS-Nano HTTP adapter (pytorch sidecar provider `moss_tts_nano`).

Ported from board voice_runtime.py:MossTtsNanoHttpEngine. The TTS model runs in
the vendored sidecar process (sidecar/backend/moss_tts_nano_sidecar.py, board
runs it on :18100); this adapter is the nano-protocol HTTP client. The protocol
client itself lives in ../common/nano_protocol.py (shared with the native
cosyvoice3/moss-tts-realtime sidecars) — only the nano sampling knobs and the
optional clone-prompt path are provider-specific.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ....config import Settings
from ..common.nano_protocol import NanoProtocolEngine
from ..common.pool import SidecarPoolAdapter


class MossTtsNanoEngine(NanoProtocolEngine):
    provider_name = "moss_tts_nano_http"

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        super().__init__(settings, base_url)
        self.prompt_audio_path = os.getenv("MOSS_TTS_NANO_PROMPT_AUDIO", "")

    def form_fields(self, text: str, voice: Optional[str]) -> Dict[str, Any]:
        fields = super().form_fields(text, voice)
        fields.update({
            "max_new_frames": os.getenv("MOSS_TTS_NANO_MAX_NEW_FRAMES", "375"),
            "voice_clone_max_text_tokens": os.getenv("MOSS_TTS_NANO_VOICE_CLONE_MAX_TEXT_TOKENS", "75"),
            "do_sample": os.getenv("MOSS_TTS_NANO_DO_SAMPLE", "1"),
            "audio_temperature": os.getenv("MOSS_TTS_NANO_AUDIO_TEMPERATURE", "0.8"),
            "audio_top_p": os.getenv("MOSS_TTS_NANO_AUDIO_TOP_P", "0.95"),
            "audio_top_k": os.getenv("MOSS_TTS_NANO_AUDIO_TOP_K", "25"),
            "audio_repetition_penalty": os.getenv("MOSS_TTS_NANO_AUDIO_REPETITION_PENALTY", "1.2"),
            "seed": os.getenv("MOSS_TTS_NANO_SEED", "0"),
        })
        if self.prompt_audio_path:
            fields["prompt_audio_path"] = self.prompt_audio_path
        return fields


class MossTtsNanoAdapter(SidecarPoolAdapter):
    engine_cls = MossTtsNanoEngine
