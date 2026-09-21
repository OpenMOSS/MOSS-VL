"""Build concrete adapters from Settings (+ the GPU placement plan).

This is the pluggable seam: swapping an engine is choosing a provider string in
config, not editing routers. New engines (Qwen3-ASR/Fun-ASR realtime, vLLM-Omni
TTS, Kokoro) slot in here.

`VLM_DEPLOY` picks the VLM shape: `workers` (default) = one worker process per
placement-plan GPU behind a `VlmReplicaPool`; `inproc` = the pre-multi-GPU
in-process `HfMossVlAdapter` (rollback lever).
"""
from __future__ import annotations

from typing import Optional

from ..config import Settings
from ..logging_conf import get_logger
from .asr.funasr_sensevoice.adapter import FunasrSenseVoiceAdapter
from .tts.moss_tts_nano.adapter import MossTtsNanoAdapter
from .vlm.moss_vl_hf.adapter import HfMossVlAdapter

log = get_logger(__name__)


def build_asr(settings: Settings, plan: Optional[object] = None):
    provider = settings.asr_provider.strip().lower()
    if provider in {"funasr_sensevoice", "sensevoice", "funasr"}:
        device = getattr(plan, "asr_device", None)
        return FunasrSenseVoiceAdapter(settings, device=device)
    raise NotImplementedError(
        f"ASR provider '{provider}' not implemented yet. "
        "Available: funasr_sensevoice. (openai_ws / qwen3_asr are planned.)"
    )


def build_tts(settings: Settings, plan: Optional[object] = None):
    from .tts.providers import canonical_tts_provider

    provider = canonical_tts_provider(settings.tts_provider)
    tts_specs = getattr(plan, "tts", None)
    base_urls = [spec.base_url for spec in tts_specs] if tts_specs else None
    if provider == "moss_tts_nano":
        return MossTtsNanoAdapter(settings, base_urls=base_urls)
    if provider == "vllm_omni":
        from .tts.vllm_omni.adapter import VllmOmniAdapter

        return VllmOmniAdapter(settings, base_urls=base_urls)
    if provider == "cosyvoice3":
        from .tts.cosyvoice3.adapter import Cosyvoice3VllmAdapter

        return Cosyvoice3VllmAdapter(settings, base_urls=base_urls)
    if provider == "cosyvoice3_native":
        from .tts.cosyvoice3.adapter import Cosyvoice3NativeAdapter

        return Cosyvoice3NativeAdapter(settings, base_urls=base_urls)
    if provider == "moss_tts_realtime":
        from .tts.moss_tts_realtime.adapter import MossRtVllmAdapter

        return MossRtVllmAdapter(settings, base_urls=base_urls)
    if provider == "moss_tts_realtime_native":
        from .tts.moss_tts_realtime.adapter import MossRtNativeAdapter

        return MossRtNativeAdapter(settings, base_urls=base_urls)
    if provider == "elevenlabs":
        from .tts.elevenlabs import ElevenLabsAdapter

        return ElevenLabsAdapter(settings)
    if provider == "minimax":
        from .tts.minimax import MiniMaxAdapter

        return MiniMaxAdapter(settings)
    raise NotImplementedError(
        f"TTS provider '{provider}' not implemented yet. Available: moss_tts_nano, "
        "vllm_omni, cosyvoice3, cosyvoice3_native, moss_tts_realtime, "
        "moss_tts_realtime_native, elevenlabs, minimax."
    )


def build_vlm(settings: Settings, plan: Optional[object] = None):
    deploy = settings.vlm_deploy.strip().lower()
    if deploy == "sglang_omni":
        # remote sglang-omni realtime servers; no placement plan, no local workers
        from .vlm.moss_vl_sglang_omni.pool import SglangOmniPool

        return SglangOmniPool(settings)
    if deploy == "workers" and plan is not None and getattr(plan, "workers", None):
        from .vlm.moss_vl_hf.online_pool import VlmReplicaPool

        return VlmReplicaPool(settings, plan)
    if deploy == "workers":
        log.warning("VLM_DEPLOY=workers but no placement plan — falling back to inproc")
    return HfMossVlAdapter(settings)


def build_vlm_offline(settings: Settings, plan: Optional[object] = None):
    """The dedicated offline-chat plane (None = fall back to `rt.vlm`).

    Requires both an enabled provider AND offline GPUs in the placement plan —
    1-GPU boxes and OFFLINE_PROVIDER=none get None, and routers/chat.py keeps
    serving offline chat through the online pool exactly as before.
    """
    provider = settings.offline_provider.strip().lower()
    if provider in {"", "none", "off", "0"}:
        return None
    if plan is None or not getattr(plan, "offline", None):
        return None
    if provider == "sglang":
        from .vlm.moss_vl_sglang.adapter import SglangOfflinePool

        return SglangOfflinePool(settings, plan)
    raise NotImplementedError(
        f"offline provider '{provider}' not implemented. Available: sglang, none.")
