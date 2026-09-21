"""Fun-CosyVoice3 native sidecar — nano-protocol PCM streaming API.

Runs the vendored FunAudioLLM/CosyVoice stack (../third_party/CosyVoice) in
its own venv (.venv-cosy, scripts/build_venv_cosyvoice.sh). Same endpoints,
form fields and headers as moss_tts_nano_sidecar.py so the backend's shared
NanoProtocolEngine client and spawn/health-gate machinery work unchanged.

Model: the ModelScope-layout checkpoint (llm.pt/flow.pt/hift.pt +
cosyvoice3.yaml, e.g. zoo dir Fun-CosyVoice3-0.5B) via COSY3_MODEL_DIR.
Acceleration: fp16 + TensorRT flow-matching estimator (COSY3_FP16/COSY3_TRT,
default on; first boot builds the .plan into the model dir — minutes, covered
by the spawn gate's 600 s slow-first-boot budget), optional in-process vLLM
for the token LM (COSY3_VLLM, needs a vllm-enabled venv and a {model}/vllm
export — off by default).

Voices: CosyVoice3 has no presets — every request is a zero-shot clone from
the Nano voice-prompt WAVs (TTS_VOICE_PROMPT_DIR) with the transcript from
assets/demo.jsonl as prompt text ('You are a helpful assistant.<|endofprompt|>'
prefix per upstream v3 examples). An `instruct` form field switches to
inference_instruct2 (dialect/emotion/speed control).
"""

from __future__ import annotations

import json
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
    level=getattr(logging, os.getenv("COSY3_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("cosyvoice3_sidecar")

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
COSY_REPO = Path(os.getenv("COSY3_REPO", str(SIDECAR_ROOT / "third_party" / "CosyVoice"))).expanduser().resolve()
for entry in (str(COSY_REPO), str(COSY_REPO / "third_party" / "Matcha-TTS")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

PROMPT_PREFIX = os.getenv("COSY3_PROMPT_PREFIX", "You are a helpful assistant.<|endofprompt|>")

# name → prompt file, mirrored from server/adapters/tts/common/openai_speech.py
# (duplicated: this sidecar runs in .venv-cosy without the server package)
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

_DEFAULT_PROMPT_DIR = SIDECAR_ROOT.parents[1] / "moss_tts_nano" / "sidecar" / "third_party" / "MOSS-TTS-Nano" / "assets" / "audio"
PROMPT_DIR = Path(os.getenv("TTS_VOICE_PROMPT_DIR", str(_DEFAULT_PROMPT_DIR))).expanduser()
DEFAULT_VOICE = os.getenv("TTS_VOICE", "Yuewen")

app = FastAPI(title="Fun-CosyVoice3 Sidecar")

_service_lock = threading.Lock()
_service: Optional[Any] = None
_service_error = ""
_jobs_lock = threading.Lock()
_jobs: Dict[str, "StreamingJob"] = {}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_ref_texts() -> Dict[str, str]:
    """prompt filename → transcript from assets/demo.jsonl (sibling of audio/)."""
    path = PROMPT_DIR.parent / "demo.jsonl"
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


_REF_TEXTS = _load_ref_texts()


def _resolve_voice(voice: str) -> tuple[str, str]:
    """voice name → (prompt wav path, transcript). Missing prompt/transcript
    degrades to the default voice, then to any transcripted prompt on disk."""
    candidates = [voice, DEFAULT_VOICE] if voice else [DEFAULT_VOICE]
    for name in candidates:
        filename = VOICE_PROMPT_FILES.get(name, "")
        if not filename:
            continue
        path = PROMPT_DIR / filename
        if path.is_file() and filename in _REF_TEXTS:
            if name != voice and voice:
                log.warning("voice '%s' unavailable for cosyvoice3; using '%s'", voice, name)
            return str(path), _REF_TEXTS[filename]
    for filename, text in sorted(_REF_TEXTS.items()):
        path = PROMPT_DIR / filename
        if path.is_file():
            log.warning("voice '%s' unresolvable; using first transcripted prompt %s", voice, filename)
            return str(path), text
    raise RuntimeError(f"no transcripted voice prompt available under {PROMPT_DIR}")


def _get_service() -> Any:
    global _service, _service_error
    with _service_lock:
        if _service is None:
            model_dir = os.getenv("COSY3_MODEL_DIR", "")
            if not model_dir or not os.path.isdir(model_dir):
                raise RuntimeError(f"COSY3_MODEL_DIR not a directory: {model_dir!r}")
            t0 = time.monotonic()
            try:
                _service = AutoModel(
                    model_dir=model_dir,
                    load_trt=_bool_env("COSY3_TRT", True),
                    load_vllm=_bool_env("COSY3_VLLM", False),
                    fp16=_bool_env("COSY3_FP16", True),
                    trt_concurrent=int(os.getenv("COSY3_TRT_CONCURRENT", "1")),
                )
            except Exception as exc:
                _service_error = str(exc)
                raise
            log.info("CosyVoice3 loaded from %s in %.1fs (sample_rate=%s)",
                     model_dir, time.monotonic() - t0, _service.sample_rate)
            _warmup_service(_service)
        return _service


def _warmup_service(service: Any) -> None:
    if not _bool_env("COSY3_WARMUP_ON_LOAD", True):
        return
    text = os.getenv("COSY3_WARMUP_TEXT", "你好，欢迎使用语音合成。")
    started = time.monotonic()
    try:
        prompt_path, prompt_text = _resolve_voice(DEFAULT_VOICE)
        n = 0
        for out in service.inference_zero_shot(text, PROMPT_PREFIX + prompt_text, prompt_path, stream=True):
            n += int(out["tts_speech"].numel())
        log.info("CosyVoice3 warmup complete: %.3fs, %d samples", time.monotonic() - started, n)
    except Exception as exc:  # noqa: BLE001
        log.warning("CosyVoice3 warmup failed: %s", exc)


def _pcm16le(waveform: Any) -> bytes:
    array = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return b""
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2", copy=False).tobytes(order="C")


class StreamingJob:
    def __init__(self, sample_rate: int) -> None:
        self.stream_id = uuid.uuid4().hex
        self.audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.state = "pending"
        self.error = ""
        self.sample_rate = sample_rate
        self.channels = 1
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.first_audio_at: Optional[float] = None
        self.chunk_count = 0
        self.is_closed = False

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
                "chunk_count": self.chunk_count,
            }


def _create_job(sample_rate: int) -> StreamingJob:
    job = StreamingJob(sample_rate)
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


def _run_streaming_job(job: StreamingJob, *, text: str, voice: str,
                       instruct: str, speed: float) -> None:
    try:
        service = _get_service()
        prompt_path, prompt_text = _resolve_voice(voice)
        with job.lock:
            job.started_at = time.monotonic()
            job.state = "running"
            job.sample_rate = int(service.sample_rate)
        log.info("CosyVoice3 job start: stream_id=%s text_chars=%d voice=%s instruct=%r",
                 job.stream_id, len(text or ""), voice or DEFAULT_VOICE, instruct or None)
        if instruct:
            chunks = service.inference_instruct2(
                text, f"{PROMPT_PREFIX.split('<|endofprompt|>')[0]} {instruct}<|endofprompt|>",
                prompt_path, stream=True, speed=speed)
        else:
            chunks = service.inference_zero_shot(
                text, PROMPT_PREFIX + prompt_text, prompt_path, stream=True, speed=speed)
        for out in chunks:
            with job.lock:
                if job.is_closed:
                    break
            pcm_bytes = _pcm16le(out["tts_speech"].numpy())
            if not pcm_bytes:
                continue
            with job.lock:
                job.chunk_count += 1
                if job.first_audio_at is None:
                    job.first_audio_at = time.monotonic()
                    log.info("CosyVoice3 first audio: stream_id=%s latency=%.3fs chunk_bytes=%d",
                             job.stream_id, job.first_audio_at - (job.started_at or job.first_audio_at),
                             len(pcm_bytes))
            _put_audio(job, pcm_bytes)
        with job.lock:
            if job.state == "running":
                job.state = "done"
                job.completed_at = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.completed_at = time.monotonic()
        log.exception("CosyVoice3 job failed: stream_id=%s", job.stream_id)
    finally:
        try:
            job.audio_queue.put_nowait(None)
        except queue.Full:
            pass


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        service = _get_service()
        return {
            "status": "ok",
            "ready": True,
            "provider": "cosyvoice3_native_sidecar",
            "model_dir": os.getenv("COSY3_MODEL_DIR", ""),
            "voices": sorted(v for v, f in VOICE_PROMPT_FILES.items()
                             if (PROMPT_DIR / f).is_file() and f in _REF_TEXTS),
            "default_voice": DEFAULT_VOICE,
            "sample_rate": int(service.sample_rate),
            "channels": 1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "ready": False,
            "error": str(exc),
            "provider": "cosyvoice3_native_sidecar",
        }


@app.post("/api/generate-stream/start")
async def generate_stream_start(
    text: str = Form(...),
    voice: str = Form(""),
    instruct: str = Form(""),
    speed: float = Form(1.0),
) -> Any:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    try:
        service = _get_service()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"error": str(exc)})
    job = _create_job(int(service.sample_rate))
    thread = threading.Thread(
        target=_run_streaming_job,
        kwargs={"job": job, "text": normalized_text, "voice": voice,
                "instruct": str(instruct or "").strip(), "speed": float(speed or 1.0)},
        name=f"cosyvoice3-{job.stream_id}",
        daemon=True,
    )
    thread.start()
    return {
        "stream_id": job.stream_id,
        "audio_url": f"/api/generate-stream/{job.stream_id}/audio",
        "status_url": f"/api/generate-stream/{job.stream_id}/status",
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
