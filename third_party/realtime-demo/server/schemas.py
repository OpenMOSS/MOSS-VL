"""Pydantic request/response models for the HTTP endpoints.

Session-WS messages follow server/protocol.py (backend_overhaul.md §2); only the
HTTP surface is typed here.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    model_path: str
    gpu_id: int = 0
    hf_mode: str = "online_streaming"  # online_streaming | offline


class GenerationParams(BaseModel):
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    # Board parity (realtime): SAMPLE, not greedy. Greedy on the silence-trained
    # streaming ckpt makes <|silence|> the deterministic argmax for a static frame,
    # so it never answers; sampling (temp/top_k/top_p above) lets real answers through
    # while idle frames still resolve to silence. This SessionConfig default is what
    # the realtime path actually uses (routers/sessions.py _build_engines), not GEN_*.
    do_sample: bool = True
    # None → server default (Settings.repetition_penalty ← GEN_REPETITION_PENALTY)
    # via the HF adapter's fallback (moss_vl_hf/adapter.py start_realtime_session).
    # The frontend never sends this; an explicit request value still wins.
    repetition_penalty: Optional[float] = None
    max_new_tokens: int = 4096
    # tokens-per-SECOND generation rate cap; None → server default
    # (Settings.max_tokens_per_turn ← GEN_MAX_TOKENS_PER_TURN, default 86400
    # = uncapped; the frontend paces display only)
    max_tokens_per_turn: Optional[int] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    video_fps: Optional[float] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    multi_image_max_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    # data-URL/base64 strings OR CAS handles ("sha256:<hex>" from POST /api/media)
    images: Optional[List[str]] = None
    videos: Optional[List[Any]] = None
    use_template: bool = True
    params: GenerationParams = Field(default_factory=GenerationParams)
    # client-minted thread id; when set, the turn pair is recorded to history
    conversation_id: Optional[str] = None


# --------------------- session control plane (backend_overhaul §2/§B5) ---------------------


class SessionConfig(BaseModel):
    """Per-session config — the 7 session-panel settings + creation-time prompts.

    `system_prompt`/`initial_prompt` are prefilled into the VLM KV cache and only
    apply at session creation; the rest can be patched live via `session.update`.
    """

    asr_language: Optional[str] = None          # engine-global for now (documented in orchestrator)
    capture_mode: str = "ptt"                   # ptt | auto
    vad_sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # 0..1 → silence tail
    vad_rms_threshold: Optional[int] = None     # explicit override beats vad_sensitivity
    vad_silence_ms: Optional[float] = None
    tts_voice: Optional[str] = None
    # creation-time engine pick: "local" (the boot provider's pool) |
    # "elevenlabs" (cloud; requires ELEVENLABS_API_KEY at boot). Not
    # live-updatable — the engine is leased when the session is built.
    tts_engine: Optional[str] = None
    speaking_rate: Optional[float] = None       # stored; MOSS-TTS-Nano has no rate control yet
    video_source: Optional[str] = None          # initial source: 'camera' | 'screen' | 'file' | 'image' | 'none' — recorded with history; mid-session switches ride input.video.source
    system_prompt: Optional[str] = None
    initial_prompt: Optional[str] = None
    gpu_id: Optional[int] = None
    params: GenerationParams = Field(default_factory=GenerationParams)


class CreateSessionRequest(BaseModel):
    config: SessionConfig = Field(default_factory=SessionConfig)


class CreateSessionResponse(BaseModel):
    session_id: str
    ws_url: str
    expires_at: float
    config: SessionConfig
