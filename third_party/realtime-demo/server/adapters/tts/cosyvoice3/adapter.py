"""Fun-CosyVoice3-0.5B TTS adapters.

Primary (`cosyvoice3`): vLLM-Omni engine — the backend spawns
`vllm serve <Fun-CosyVoice3-0.5B-2512> --omni` (server/sidecars.py) and this
adapter is the OpenAI /v1/audio/speech streaming-PCM client. vLLM-Omni ≥ 0.24.0
ships a stabilized CosyVoice3 path (TensorRT-optimized flow, ref-text
templating); output is 24 kHz mono PCM16.

CosyVoice3-specific contract (vLLM-Omni speech API docs, 2026-07):
- cloning-only, NO engine-side presets: every request must carry `ref_audio`
  AND `ref_text` (the transcript of the reference audio). Voices map to the
  vendored Nano prompt WAVs; transcripts come from the vendored
  MOSS-TTS-Nano/assets/demo.jsonl (every shipped prompt WAV has an entry).
- streaming requires response_format=pcm + speed=1.0 (both defaults here).

Fallback (`cosyvoice3_native`): nano-protocol client to the vendored CosyVoice
repo sidecar (sidecar/backend/cosyvoice3_sidecar.py, own .venv-cosy, fp16 +
load_jit + load_trt acceleration) — same acquire/release pool surface.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from ....config import Settings
from ....logging_conf import get_logger
from ..common.nano_protocol import NanoProtocolEngine
from ..common.openai_speech import VOICE_PROMPT_FILES, OpenAiSpeechEngine
from ..common.pool import SidecarPoolAdapter

log = get_logger(__name__)


def load_ref_texts(prompt_dir: str) -> Dict[str, str]:
    """prompt filename → transcript, from the assets checkout's demo.jsonl.

    The jsonl lives one level above the audio dir (assets/demo.jsonl with
    role paths like "assets/audio/zh_1.wav"). Missing/unreadable file → {}.
    """
    path = os.path.join(os.path.dirname(prompt_dir.rstrip(os.sep)), "demo.jsonl")
    texts: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                filename = str(entry.get("role") or "").rsplit("/", 1)[-1]
                text = str(entry.get("text") or "").strip()
                if filename and text:
                    texts.setdefault(filename, text)
    except OSError as exc:
        log.warning("voice transcript map unreadable (%s): %s", path, exc)
    return texts


class Cosyvoice3SpeechEngine(OpenAiSpeechEngine):
    provider_name = "cosyvoice3_speech"
    sample_rate_env = "COSY3_SAMPLE_RATE"
    channels_env = "COSY3_CHANNELS"
    default_sample_rate = 24000
    default_channels = 1
    warmup_env = "COSY3_WARMUP"
    # CosyVoice3 is flow-matching (robust); leave sampling to its deploy yaml by
    # default, but keep the env hook (COSY3_SAMPLING JSON) for tuning.
    sampling: dict = {}
    sampling_env = "COSY3_SAMPLING"

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        super().__init__(settings, base_url)
        self._ref_texts = load_ref_texts(self.prompt_dir)
        self._ref_text_override = os.getenv("COSY3_REF_TEXT", "")
        self._task_type = os.getenv("COSY3_TASK_TYPE", "")
        self._resolved_lock = threading.Lock()
        self._resolved_paths: Dict[str, str] = {}

    def _served_model_name(self, settings: Settings) -> str:
        return settings.tts_cosy3_served_name

    def _ref_text_for(self, voice: str) -> str:
        if self._ref_text_override:
            return self._ref_text_override
        # transcript is keyed by the RESOLVED prompt file (missing prompts
        # degrade to the default voice inside _resolve_prompt_path)
        with self._resolved_lock:
            path = self._resolved_paths.get(voice)
        if path is None:
            path = self._resolve_prompt_path(voice)
            with self._resolved_lock:
                self._resolved_paths[voice] = path
        text = self._ref_texts.get(os.path.basename(path), "")
        if not text:
            raise RuntimeError(
                f"CosyVoice3 needs a reference transcript for prompt {path!r} — add it to "
                "assets/demo.jsonl next to the prompt dir, or set COSY3_REF_TEXT")
        return text

    def extra_payload(self, text: str, voice: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ref_text": self._ref_text_for(voice)}
        if self._task_type:
            payload["task_type"] = self._task_type
        return payload


class Cosyvoice3VllmAdapter(SidecarPoolAdapter):
    engine_cls = Cosyvoice3SpeechEngine


class Cosyvoice3NativeEngine(NanoProtocolEngine):
    provider_name = "cosyvoice3_native_http"

    def __init__(self, settings: Settings, base_url: Optional[str] = None):
        super().__init__(settings, base_url,
                         sample_rate=int(os.getenv("COSY3_SAMPLE_RATE", "24000")),
                         channels=int(os.getenv("COSY3_CHANNELS", "1")))

    def form_fields(self, text: str, voice: Optional[str]) -> Dict[str, Any]:
        fields = super().form_fields(text, voice)
        speed = os.getenv("COSY3_SPEED", "")
        if speed:
            fields["speed"] = speed
        instruct = os.getenv("COSY3_INSTRUCT", "")
        if instruct:  # inference_instruct2 dialect/emotion/speed control
            fields["instruct"] = instruct
        return fields


class Cosyvoice3NativeAdapter(SidecarPoolAdapter):
    engine_cls = Cosyvoice3NativeEngine


# re-export: the sidecar and gen_sentences share the voice map
__all__ = [
    "Cosyvoice3NativeAdapter",
    "Cosyvoice3NativeEngine",
    "Cosyvoice3SpeechEngine",
    "Cosyvoice3VllmAdapter",
    "VOICE_PROMPT_FILES",
    "load_ref_texts",
]
