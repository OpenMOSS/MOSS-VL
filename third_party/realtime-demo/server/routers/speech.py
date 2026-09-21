"""One-shot speech HTTP endpoints for the OFFLINE chat page.

POST /api/asr  raw PCM16 mono @16 kHz (application/octet-stream) → {text, ms}
               The dictation mic posts its rolling buffer every ~1 s for a live
               hypothesis, then once more on stop for the final text.
POST /api/tts  {text, voice?} → audio/wav (STREAMED) — read-aloud for chat
               bubbles. The whole text goes to the engine as ONE job (the
               engine splits + batches internally — tts_serving_plan.md
               Stage 0) and WAV bytes stream out as PCM arrives, so playback
               can start ~1s in instead of after the full synth.

The realtime session plane never touches these — its ASR/TTS ride the session
socket (server/session/orchestrator.py).
"""
from __future__ import annotations

import asyncio
import re
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["speech"])

# ~5 min of 16 kHz PCM16 — dictation buffers are seconds, this is a hard stop
_ASR_MAX_BYTES = 10 * 1024 * 1024
_TTS_MAX_CHARS = 2000

# sentence terminators (keep the mark with its sentence); "." only when not a decimal
_SENT_END_RE = re.compile(r"[。！？!?；;\n]|(?<!\d)\.(?!\d)")
_CLAUSE_BOUNDARY = "，,、：:；; "


def _iter_sentences(text: str) -> List[str]:
    out: List[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        piece = text[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _cut_long(seg: str, max_chars: int) -> List[str]:
    """Split an over-long clause at the nearest boundary/space under max_chars."""
    out: List[str] = []
    while len(seg) > max_chars:
        cut = -1
        for ch in _CLAUSE_BOUNDARY:
            cut = max(cut, seg.rfind(ch, 0, max_chars + 1))
        cut = cut + 1 if cut > max_chars // 2 else max_chars
        out.append(seg[:cut].strip())
        seg = seg[cut:].strip()
    if seg:
        out.append(seg)
    return out


def _segments_for_tts(text: str, first_clause_chars: int, max_chars: int) -> List[str]:
    """Clause-sized units for streaming synthesis (P2 fix).

    Streaming the offline read-aloud clause-by-clause makes first-audio ~constant
    (first clause) instead of linear in total length: pushing the whole text as
    one job forces the sidecar to synthesize everything before emitting a byte.
    A short first unit clears the model's text→audio delay quickly for a fast
    first cut; later units stay sentence-sized for prosody.
    """
    units: List[str] = []
    for sentence in _iter_sentences(text):
        units.extend(_cut_long(sentence, max_chars))
    # keep the FIRST unit short (fast first cut) if the leading sentence is long
    if units and first_clause_chars > 0 and len(units[0]) > first_clause_chars:
        head = units[0]
        cut = -1
        for ch in _CLAUSE_BOUNDARY:
            pos = head.find(ch, first_clause_chars)
            if pos != -1 and (cut == -1 or pos < cut):
                cut = pos
        if cut != -1 and cut + 1 < len(head):
            units[0:1] = [head[:cut + 1].strip(), head[cut + 1:].strip()]
    # coalesce punctuation-only fragments (e.g. a trailing "。" orphaned by a
    # hard length cut) back into the preceding unit so nothing synthesizes silence
    merged: List[str] = []
    for u in units:
        u = u.strip()
        if not u:
            continue
        if merged and not any(c.isalnum() for c in u):
            merged[-1] = merged[-1] + u
        else:
            merged.append(u)
    return merged


@router.post("/asr")
async def transcribe(request: Request, rt: Runtime = Depends(get_runtime)):
    asr = rt.asr
    if asr is None or not getattr(asr, "ready", False):
        raise HTTPException(status_code=503, detail=getattr(asr, "status_message", "ASR is not available"))

    pcm = bytearray()
    async for chunk in request.stream():
        pcm.extend(chunk)
        if len(pcm) > _ASR_MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"PCM exceeds {_ASR_MAX_BYTES} bytes")
    if len(pcm) < 2:
        raise HTTPException(status_code=400, detail="empty PCM body")

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        # allow_short: rolling dictation posts sub-second buffers early on
        text = await asyncio.to_thread(asr.transcribe_pcm, bytes(pcm), True)
    except Exception as exc:  # noqa: BLE001
        log.exception("ASR transcribe failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"text": text, "ms": round((loop.time() - t0) * 1000.0, 1)}


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=_TTS_MAX_CHARS)
    voice: Optional[str] = None


def _wav_stream_header(sample_rate: int, channels: int) -> bytes:
    """44-byte PCM16 WAV header with the streaming-size convention.

    Total length is unknown while PCM is still being synthesized, so RIFF and
    data sizes are 0xFFFFFFFF — decoders read to EOF (same convention ffmpeg
    and browsers use for live WAV)."""
    unknown = (0xFFFFFFFF).to_bytes(4, "little")
    byte_rate = sample_rate * channels * 2
    return b"".join([
        b"RIFF", unknown, b"WAVE",
        b"fmt ", (16).to_bytes(4, "little"), (1).to_bytes(2, "little"),
        channels.to_bytes(2, "little"), sample_rate.to_bytes(4, "little"),
        byte_rate.to_bytes(4, "little"), (channels * 2).to_bytes(2, "little"),
        (16).to_bytes(2, "little"),
        b"data", unknown,
    ])


@router.post("/tts")
async def synthesize(req: TtsRequest, rt: Runtime = Depends(get_runtime)):
    from ..voice.tts_session import is_muted_voice

    tts = rt.tts
    # the "(none)" / "（无）" voice disables TTS — return a valid empty WAV so the
    # chat read-aloud is silent instead of falling back to the default voice
    if is_muted_voice(req.voice):
        empty = _wav_stream_header(48000, 1)
        return StreamingResponse(iter([empty]), media_type="audio/wav", headers={
            "X-Audio-Sample-Rate": "48000", "X-Audio-Channels": "1"})
    status = tts.status() if tts is not None else {}
    if not status.get("ready"):
        raise HTTPException(status_code=503, detail=status.get("message", "TTS is not available"))
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    # every engine in the pool shares the primary's PCM format (settings +
    # health payload), so the response headers can come from tts.engine while
    # the actual lease happens inside the generator — a generator that never
    # starts (instant client disconnect) never runs finally, so acquiring out
    # here could leak the lease
    sample_rate, channels = tts.engine.sample_rate, tts.engine.channels
    voice = req.voice
    settings = get_settings()

    def wav_iter() -> Iterator[bytes]:
        # sync generator: Starlette drives it in a worker thread, so the
        # blocking engine HTTP stream never touches the event loop.
        engine = tts.acquire() if hasattr(tts, "acquire") else tts.engine
        stream = None
        try:
            if getattr(engine, "supports_streaming", False):
                # P2 fix: push the text clause-by-clause through ONE warm session
                # so audio streams from the first clause — first-audio is ~constant
                # instead of linear in total length. Later clauses synthesize while
                # earlier audio is already playing (RTF < 1 keeps it ahead).
                stream = engine.open_stream(voice)
                yield _wav_stream_header(stream.sample_rate, stream.channels)
                for segment in _segments_for_tts(
                        text, settings.tts_first_clause_chars, settings.tts_seg_max_chars):
                    stream.push_text(segment)
                stream.end_input()
                for pcm in stream.audio_chunks():
                    yield pcm
            else:
                # non-streaming engine: whole text as ONE job (the engine cuts +
                # batches chunks internally). read1() still trims first-byte lag.
                yield _wav_stream_header(engine.sample_rate, engine.channels)
                for chunk in engine.synthesize_pcm(text, voice=voice):
                    yield chunk
        except Exception as exc:  # noqa: BLE001 — headers are gone; log and cut the stream
            log.exception("TTS synth failed mid-stream: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(tts, "release"):
                tts.release(engine)

    return StreamingResponse(wav_iter(), media_type="audio/wav", headers={
        "Cache-Control": "no-store",
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Channels": str(channels),
    })
