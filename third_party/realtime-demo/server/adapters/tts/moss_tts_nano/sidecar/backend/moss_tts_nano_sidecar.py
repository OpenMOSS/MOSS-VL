"""Lightweight MOSS-TTS-Nano sidecar for the board voice runtime.

This service intentionally avoids the official demo app's UI and text
normalization stack. The board backend only needs a small PCM streaming API.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, StreamingResponse


logging.basicConfig(
    level=getattr(logging, os.getenv("MOSS_TTS_NANO_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("moss_tts_nano_sidecar")

BOARD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NANO_REPO = BOARD_ROOT / "third_party" / "MOSS-TTS-Nano"
NANO_REPO = Path(os.getenv("MOSS_TTS_NANO_REPO", str(DEFAULT_NANO_REPO))).expanduser().resolve()
if str(NANO_REPO) not in sys.path:
    sys.path.insert(0, str(NANO_REPO))

from moss_tts_nano_runtime import NanoTTSService  # noqa: E402
from app_onnx import OnnxNanoTTSServiceAdapter  # noqa: E402


DEFAULT_MODEL_ROOT = BOARD_ROOT / "third_party" / "models"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MODEL_ROOT / "MOSS-TTS-Nano-100M"
DEFAULT_TOKENIZER_DIR = DEFAULT_MODEL_ROOT / "MOSS-Audio-Tokenizer-Nano"
DEFAULT_CHECKPOINT_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
DEFAULT_TOKENIZER_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"

app = FastAPI(title="Board MOSS-TTS-Nano Sidecar")

_service_lock = threading.Lock()
_service: Optional[Any] = None
_jobs_lock = threading.Lock()
_jobs: Dict[str, "StreamingJob"] = {}


def _source(local_dir: Path, fallback: str) -> str:
    return str(local_dir) if local_dir.exists() else fallback


def _backend_name() -> str:
    normalized = os.getenv("MOSS_TTS_NANO_BACKEND", os.getenv("TTS_NANO_BACKEND", "pytorch")).strip().lower()
    if normalized in {"onnx", "onnx_cpu", "onnx_cuda", "ort"}:
        return "onnx"
    return "pytorch"


def _onnx_model_dir_arg() -> Optional[str]:
    raw = os.getenv("MOSS_TTS_NANO_ONNX_MODEL_DIR", "").strip()
    if raw:
        return str(Path(raw).expanduser())
    # Passing None follows the official app_onnx.py behavior: use the repo's
    # default model dir and auto-download missing browser_onnx assets.
    return None


def _get_service() -> Any:
    global _service
    with _service_lock:
        if _service is None:
            output_dir = Path(
                os.getenv(
                    "MOSS_TTS_NANO_OUTPUT_DIR",
                    str(BOARD_ROOT / "backend" / "uploads" / "nano_tts"),
                )
            ).expanduser()
            if _backend_name() == "onnx":
                cpu_threads = int(os.getenv("MOSS_TTS_NANO_ONNX_CPU_THREADS", str(max(1, int(os.cpu_count() or 1)))))
                execution_provider = os.getenv("MOSS_TTS_NANO_ONNX_EXECUTION_PROVIDER", "cpu")
                _service = OnnxNanoTTSServiceAdapter(
                    model_dir=_onnx_model_dir_arg(),
                    output_dir=output_dir,
                    cpu_threads=max(1, cpu_threads),
                    execution_provider=execution_provider,
                    max_new_frames=int(os.getenv("MOSS_TTS_NANO_MAX_NEW_FRAMES", "375")),
                    text_normalizer_manager=None,
                )
            else:
                _service = NanoTTSService(
                    checkpoint_path=os.getenv(
                        "MOSS_TTS_NANO_CHECKPOINT",
                        _source(DEFAULT_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_ID),
                    ),
                    audio_tokenizer_path=os.getenv(
                        "MOSS_TTS_NANO_AUDIO_TOKENIZER",
                        _source(DEFAULT_TOKENIZER_DIR, DEFAULT_TOKENIZER_ID),
                    ),
                    device=os.getenv("MOSS_TTS_NANO_DEVICE", "cpu"),
                    dtype=os.getenv("MOSS_TTS_NANO_DTYPE", "auto"),
                    attn_implementation=os.getenv("MOSS_TTS_NANO_ATTN", "auto"),
                    output_dir=output_dir,
                )
            _warmup_service(_service)
        return _service


def _bool(value: str, default: bool = True) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _seed(value: str) -> Optional[int]:
    normalized = str(value or "").strip()
    if not normalized or normalized == "0":
        return None
    return int(normalized)


def _resolve_attn(service: Any, requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if _backend_name() == "onnx" or hasattr(service, "execution_provider"):
        if normalized in {"", "auto", "default", "model_default", "eager", "sdpa", "flash_attention_2"}:
            return os.getenv("MOSS_TTS_NANO_ONNX_SAMPLE_MODE", "fixed")
        return requested
    if getattr(getattr(service, "device", None), "type", "") == "cpu" and normalized in {
        "",
        "auto",
        "default",
        "model_default",
        "flash_attention_2",
    }:
        return "eager"
    return requested or "auto"


def _warmup_service(service: Any) -> None:
    if not _bool(os.getenv("MOSS_TTS_NANO_WARMUP_ON_LOAD", "1"), True):
        return
    text = os.getenv("MOSS_TTS_NANO_WARMUP_TEXT", "你好，欢迎使用 Nano-TTS。").strip() or "你好，欢迎使用 Nano-TTS。"
    # comma-separated voices: on the torchair/NPU path each voice preset has a
    # distinct prompt length → first synthesis per voice recompiles (10-15s) or
    # fails once (frozen-parameter graph stale after a discard). Warming every
    # listed voice inside the health gate makes the FIRST user request of each
    # voice hit the ready graph. Default: the two voices the UI offers first.
    voices = [
        v.strip() for v in os.getenv("MOSS_TTS_NANO_WARMUP_VOICES", "Junhao,Yuewen").split(",") if v.strip()
    ] or [os.getenv("MOSS_TTS_NANO_VOICE", "Junhao")]
    started = time.monotonic()
    for voice in voices:
        try:
            service.get_model()
            if _backend_name() == "onnx" or hasattr(service, "execution_provider"):
                result = service.warmup()
            else:
                # warm through the STREAMING path: the torchair graph shapes
                # exercised by synthesize() differ subtly from synthesize_stream's
                # (a first streamed request on the synthesize()-warmed graph hits
                # a GE execute error and discards the model). Consuming the
                # stream here compiles exactly the production shapes.
                result = {}
                for event in service.synthesize_stream(text=text, voice=voice):
                    if isinstance(event, dict) and event.get("type") == "result":
                        result = event
                        break
            audio_path = result.get("audio_path") if isinstance(result, dict) else None
            if audio_path:
                try:
                    Path(str(audio_path)).unlink(missing_ok=True)
                except Exception:
                    pass
            log.info(
                "MOSS-TTS-Nano warmup complete: backend=%s voice=%s text_chars=%d elapsed=%.3fs",
                _backend_name(),
                voice,
                len(text),
                time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("MOSS-TTS-Nano warmup failed: backend=%s voice=%s error=%s", _backend_name(), voice, exc)

def _normalize_audio(audio_array: Any) -> np.ndarray:
    array = np.asarray(audio_array, dtype=np.float32)
    if array.size == 0:
        return array
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {array.shape}")
    if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
        array = array.T
    return array


def _channels(audio_array: np.ndarray) -> int:
    if audio_array.ndim == 1:
        return 1
    return int(audio_array.shape[1])


def _pcm16le(audio_array: Any) -> bytes:
    array = _normalize_audio(audio_array)
    if array.size == 0:
        return b""
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes(order="C")


class StreamingJob:
    def __init__(self) -> None:
        self.stream_id = uuid.uuid4().hex
        self.audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.state = "pending"
        self.error = ""
        self.sample_rate = 48000
        self.channels = 2
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.first_audio_at: Optional[float] = None
        self.emitted_audio_seconds = 0.0
        self.lead_seconds = 0.0
        self.chunk_count = 0
        self.is_closed = False
        self.final_result: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            elapsed = None if self.started_at is None else now - self.started_at
            first_audio_latency = None
            if self.started_at is not None and self.first_audio_at is not None:
                first_audio_latency = self.first_audio_at - self.started_at
            return {
                "stream_id": self.stream_id,
                "state": self.state,
                "ready": self.state == "done",
                "failed": self.state == "failed",
                "closed": self.is_closed,
                "error": self.error,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "elapsed_seconds": elapsed,
                "first_audio_latency_seconds": first_audio_latency,
                "emitted_audio_seconds": self.emitted_audio_seconds,
                "lead_seconds": self.lead_seconds,
                "chunk_count": self.chunk_count,
                "result": self.final_result,
            }


def _create_job() -> StreamingJob:
    job = StreamingJob()
    with _jobs_lock:
        _jobs[job.stream_id] = job
    return job


def _get_job(stream_id: str) -> Optional[StreamingJob]:
    with _jobs_lock:
        return _jobs.get(stream_id)


def _put_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
    while True:
        with job.lock:
            if job.is_closed:
                return
        try:
            job.audio_queue.put(pcm_bytes, timeout=0.1)
            return
        except queue.Full:
            continue


def _run_streaming_job(
    job: StreamingJob,
    *,
    text: str,
    voice: str,
    prompt_audio_path: str,
    max_new_frames: int,
    voice_clone_max_text_tokens: int,
    tts_max_batch_size: int,
    codec_max_batch_size: int,
    attn_implementation: str,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: Optional[int],
) -> None:
    try:
        service = _get_service()
        selected_voice = voice or os.getenv("MOSS_TTS_NANO_VOICE", "Junhao")
        prompt_audio = prompt_audio_path or os.getenv("MOSS_TTS_NANO_PROMPT_AUDIO", "")
        prompt_audio = prompt_audio.strip() or None
        with job.lock:
            job.started_at = time.monotonic()
            job.state = "running"
        log.info(
            "MOSS-TTS-Nano job start: stream_id=%s text_chars=%d voice=%s device=%s",
            job.stream_id,
            len(text or ""),
            selected_voice,
            getattr(service, "device", os.getenv("MOSS_TTS_NANO_DEVICE", "unknown")),
        )
        first_audio_logged = False

        for event in service.synthesize_stream(
            text=text,
            voice=selected_voice,
            mode="voice_clone",
            prompt_audio_path=prompt_audio,
            max_new_frames=int(max_new_frames),
            voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
            tts_max_batch_size=int(tts_max_batch_size),
            codec_max_batch_size=int(codec_max_batch_size),
            attn_implementation=_resolve_attn(service, attn_implementation),
            do_sample=bool(do_sample),
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
            seed=seed,
        ):
            with job.lock:
                if job.is_closed:
                    break
            event_type = str(event.get("type") or "")
            if event_type == "audio":
                audio = _normalize_audio(event["waveform_numpy"])
                pcm_bytes = _pcm16le(audio)
                if not pcm_bytes:
                    continue
                sample_rate = int(event.get("sample_rate") or 48000)
                channels = _channels(audio)
                with job.lock:
                    job.sample_rate = sample_rate
                    job.channels = channels
                    job.chunk_count += 1
                    job.emitted_audio_seconds = float(event.get("emitted_audio_seconds") or 0.0)
                    job.lead_seconds = float(event.get("lead_seconds") or 0.0)
                    if job.first_audio_at is None and not bool(event.get("is_pause")):
                        job.first_audio_at = time.monotonic()
                        if not first_audio_logged:
                            first_audio_logged = True
                            log.info(
                                "MOSS-TTS-Nano first audio: stream_id=%s latency=%.3fs chunk_bytes=%d emitted_audio=%.3fs lead=%.3fs",
                                job.stream_id,
                                job.first_audio_at - (job.started_at or job.first_audio_at),
                                len(pcm_bytes),
                                job.emitted_audio_seconds,
                                job.lead_seconds,
                            )
                _put_audio(job, pcm_bytes)
                continue
            if event_type == "result":
                with job.lock:
                    job.final_result = {
                        "audio_path": event.get("audio_path"),
                        "sample_rate": int(event.get("sample_rate") or job.sample_rate),
                        "voice": event.get("voice") or selected_voice,
                        "mode": event.get("mode"),
                        "prompt_audio_path": event.get("prompt_audio_path"),
                        "elapsed_seconds": event.get("elapsed_seconds"),
                    }
                    job.state = "done"
                    job.completed_at = time.monotonic()
        with job.lock:
            if job.state == "running":
                job.state = "done"
                job.completed_at = time.monotonic()
            elapsed = None if job.started_at is None or job.completed_at is None else job.completed_at - job.started_at
            first_audio_latency = None if job.started_at is None or job.first_audio_at is None else job.first_audio_at - job.started_at
            log.info(
                "MOSS-TTS-Nano job done: stream_id=%s state=%s elapsed=%s first_audio=%s chunks=%d emitted_audio=%.3fs error=%s",
                job.stream_id,
                job.state,
                f"{elapsed:.3f}s" if elapsed is not None else "n/a",
                f"{first_audio_latency:.3f}s" if first_audio_latency is not None else "n/a",
                job.chunk_count,
                job.emitted_audio_seconds,
                job.error,
            )
    except Exception as exc:
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.completed_at = time.monotonic()
        log.exception("MOSS-TTS-Nano job failed: stream_id=%s text_chars=%d", job.stream_id, len(text or ""))
    finally:
        try:
            job.audio_queue.put_nowait(None)
        except queue.Full:
            pass



def _service_voice_names(service: Any) -> list[str]:
    if hasattr(service, "list_voice_names"):
        return list(service.list_voice_names())
    runtime = getattr(service, "runtime", None)
    if runtime is not None and hasattr(runtime, "list_builtin_voices"):
        return [str(item.get("voice")) for item in runtime.list_builtin_voices() if item.get("voice")]
    return [os.getenv("MOSS_TTS_NANO_VOICE", "Junhao")]


def _service_default_voice(service: Any) -> str:
    default_voice = getattr(service, "default_voice", None)
    if default_voice:
        return str(default_voice)
    voices = _service_voice_names(service)
    return voices[0] if voices else os.getenv("MOSS_TTS_NANO_VOICE", "Junhao")


def _service_path(service: Any, attr: str) -> str:
    value = getattr(service, attr, None)
    if value is not None:
        return str(value)
    model_dir = getattr(service, "model_dir", None)
    return str(model_dir or "")

@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        service = _get_service()
        return {
            "status": "ok",
            "ready": True,
            "provider": "moss_tts_nano_sidecar",
            "backend": _backend_name(),
            "device": str(service.device),
            "dtype": str(service.dtype),
            "execution_provider": str(getattr(service, "execution_provider", "")),
            "checkpoint_path": _service_path(service, "checkpoint_path"),
            "audio_tokenizer_path": _service_path(service, "audio_tokenizer_path"),
            "voices": _service_voice_names(service),
            "default_voice": _service_default_voice(service),
            "sample_rate": 48000,
            "channels": 2,
        }
    except Exception as exc:
        return {
            "status": "error",
            "ready": False,
            "error": str(exc),
            "provider": "moss_tts_nano_sidecar",
        }


@app.get("/api/voices")
async def voices() -> Dict[str, Any]:
    service = _get_service()
    return {
        "voices": _service_voice_names(service),
        "default_voice": _service_default_voice(service),
    }


@app.post("/api/generate-stream/start")
async def generate_stream_start(
    text: str = Form(...),
    voice: str = Form(""),
    prompt_audio_path: str = Form(""),
    max_new_frames: int = Form(375),
    voice_clone_max_text_tokens: int = Form(75),
    tts_max_batch_size: int = Form(0),
    codec_max_batch_size: int = Form(0),
    attn_implementation: str = Form("auto"),
    do_sample: str = Form("1"),
    text_temperature: float = Form(1.0),
    text_top_p: float = Form(1.0),
    text_top_k: int = Form(50),
    audio_temperature: float = Form(0.8),
    audio_top_p: float = Form(0.95),
    audio_top_k: int = Form(25),
    audio_repetition_penalty: float = Form(1.2),
    seed: str = Form("0"),
) -> Any:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    job = _create_job()
    thread = threading.Thread(
        target=_run_streaming_job,
        kwargs={
            "job": job,
            "text": normalized_text,
            "voice": voice,
            "prompt_audio_path": prompt_audio_path,
            "max_new_frames": max_new_frames,
            "voice_clone_max_text_tokens": voice_clone_max_text_tokens,
            "tts_max_batch_size": tts_max_batch_size,
            "codec_max_batch_size": codec_max_batch_size,
            "attn_implementation": attn_implementation,
            "do_sample": _bool(do_sample, True),
            "text_temperature": text_temperature,
            "text_top_p": text_top_p,
            "text_top_k": text_top_k,
            "audio_temperature": audio_temperature,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
            "seed": _seed(seed),
        },
        name=f"moss-tts-nano-{job.stream_id}",
        daemon=True,
    )
    thread.start()
    return {
        "stream_id": job.stream_id,
        "audio_url": f"/api/generate-stream/{job.stream_id}/audio",
        "status_url": f"/api/generate-stream/{job.stream_id}/status",
        "result_url": f"/api/generate-stream/{job.stream_id}/result",
        "sample_rate": job.sample_rate,
        "channels": job.channels,
    }


@app.get("/api/generate-stream/{stream_id}/status")
async def generate_stream_status(stream_id: str) -> Any:
    job = _get_job(stream_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "stream not found"})
    return job.snapshot()


@app.get("/api/generate-stream/{stream_id}/audio")
async def generate_stream_audio(stream_id: str) -> Any:
    job = _get_job(stream_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "stream not found"})

    def iter_audio():
        while True:
            item = job.audio_queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        iter_audio(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Codec": "pcm_s16le",
            "X-Audio-Sample-Rate": str(job.sample_rate),
            "X-Audio-Channels": str(job.channels),
            "X-Stream-Id": stream_id,
        },
    )


@app.get("/api/generate-stream/{stream_id}/result")
async def generate_stream_result(stream_id: str) -> Any:
    job = _get_job(stream_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "stream not found"})
    snapshot = job.snapshot()
    if snapshot["failed"]:
        return JSONResponse(status_code=500, content=snapshot)
    if not snapshot["ready"]:
        return JSONResponse(status_code=202, content=snapshot)
    return snapshot


@app.post("/api/generate-stream/{stream_id}/close")
async def generate_stream_close(stream_id: str) -> Any:
    job = _get_job(stream_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "stream not found"})
    with job.lock:
        job.is_closed = True
        if job.state in {"pending", "running"}:
            job.state = "closed"
            job.completed_at = time.monotonic()
    try:
        job.audio_queue.put_nowait(None)
    except queue.Full:
        pass
    return job.snapshot()
