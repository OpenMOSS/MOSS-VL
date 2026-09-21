"""TTS sidecar supervisor — the backend owns the sidecar's lifecycle.

MOSS-TTS-Nano runs in its own PROCESS (separate GIL + CUDA context: a
synthesis burst must never stall the token-paced generation loop) but in the
SAME .venv — the split is process isolation, not env isolation. The backend
spawns it during FastAPI lifespan startup, health-gates it before the voice
runtime probes it, and terminates it at shutdown, so one entry point
(run_backend.sh) brings the whole voice stack up and down together.

Adopt-don't-own: if something already answers /health on the TTS port (a
hand-started sidecar, or an orphan from a hard-killed backend), it is used
as-is and its lifecycle is left alone — spawning is idempotent.

TTS_PYTHON selects the child interpreter (default: this process's
sys.executable, i.e. the shared venv); point it at another venv's python if
the TTS deps ever have to fork away from the backend's.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

from .config import REPO_ROOT, Settings
from .logging_conf import (
    LOG_DATEFMT,
    LOG_FILE_BACKUPS,
    LOG_FILE_MAX_BYTES,
    LOG_FORMAT,
    get_logger,
)
from .device_compat import set_visible_device

log = get_logger(__name__)

_ROTATING_TEE = os.path.join(REPO_ROOT, "scripts", "deploy", "rotating_tee.py")


def _target_compute_cap(gpu_index: Optional[int]) -> Optional[Tuple[int, int]]:
    """Compute capability of the GPU/NPU this sidecar will run on, via the smi
    topology probe (a subprocess — never initializes CUDA/NPU in the gateway). None
    when undetectable. gpu_index None = the sidecar inherits the gateway's device
    visibility, so device 0 is the representative card (a box is single-SKU)."""
    try:
        from .gpu.topology import probe_topology
        gpus = probe_topology(timeout_s=5.0)
    except Exception:  # topology probe is best-effort — never block a spawn on it
        return None
    if not gpus:
        return None
    want = gpu_index if gpu_index is not None else 0
    for g in gpus:
        if g.index == want:
            return tuple(g.compute_cap)
    return tuple(gpus[0].compute_cap)


def _write_vllm_logging_config(log_path: str) -> str:
    """Generate the dictConfig JSON that makes a vLLM engine rotate its own log.

    VLLM_LOGGING_CONFIG_PATH *replaces* vLLM's default logging config, and
    vLLM also forwards the same file as uvicorn's log_config — so this one
    JSON routes vllm.* AND uvicorn.access/error into the RotatingFileHandler.
    The filename MUST end in .json (uvicorn dispatches on the extension; any
    other suffix falls into fileConfig and crashes the engine at boot).

    Known limitation, accepted: vllm serve is ≥2 processes (API server +
    EngineCore) that each open this handler on the SAME file — a rollover
    race can misplace a few lines around a 32 MiB boundary. TTS volume makes
    that rare; the fallback is piping the engine through rotating_tee.py
    like the pytorch sidecar.
    """
    log_dir = os.path.dirname(os.path.abspath(log_path))
    stem = os.path.basename(log_path)
    if stem.endswith(".log"):
        stem = stem[: -len(".log")]
    cfg_path = os.path.join(log_dir, f".{stem}.logging.json")
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"plain": {"format": LOG_FORMAT, "datefmt": LOG_DATEFMT}},
        "handlers": {"file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.abspath(log_path),
            "maxBytes": LOG_FILE_MAX_BYTES,
            "backupCount": LOG_FILE_BACKUPS,
            "encoding": "utf-8",
            "formatter": "plain",
        }},
        "root": {"handlers": ["file"], "level": level},
        "loggers": {
            # replacing the default config means the vllm logger must be named
            # here; handler-less + propagate so root's file handler renders it
            "vllm": {"level": level, "handlers": [], "propagate": True},
            "uvicorn": {"handlers": [], "propagate": True},
            "uvicorn.error": {"handlers": [], "propagate": True},
            "uvicorn.access": {"handlers": [], "propagate": True},
        },
    }
    os.makedirs(log_dir, exist_ok=True)
    # engines spawn as THREADS of one process — unique tmp per call + atomic
    # replace (same idiom as _ensure_vllm_model_dir)
    fd, tmp = tempfile.mkstemp(prefix=f".{stem}.logging.", dir=log_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp, cfg_path)
    return cfg_path


def _is_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001 — any failure means "not healthy yet"
        return False


def _ensure_vllm_model_dir(src: str, codec_dir: str, shadow_name: str,
                           codec_key: str = "audio_tokenizer_pretrained_name_or_path") -> str:
    """Serve-dir for a vLLM TTS engine, with a LOCAL audio-tokenizer path.

    vLLM-Omni's MOSS TTS models resolve their codec from a served-config
    field, falling back to an HF hub id — a hub fetch that dies on this
    air-gapped box (HF_HUB_OFFLINE → LocalEntryNotFoundError, engine core
    exits rc=1). The field NAME differs per architecture (verified against
    vllm-omni 0.24.0): MossTTSNano reads `audio_tokenizer_pretrained_name_or_path`,
    MossTTSRealtime reads `codec_model_name_or_path`
    (vllm_omni/model_executor/models/moss_tts/configuration_moss_tts.py).
    When the source checkpoint lacks the field, build a shadow dir (symlinks +
    patched config.json) rather than editing the SHARED board checkpoint.
    Safe under the parallel engine spawn (idempotent, atomic config write).

    codec_dir="" means the model needs no codec patch (CosyVoice3 checkpoints
    are self-contained) — the source dir is served as-is.
    """
    if not codec_dir:
        return src
    cfg_path = os.path.join(src, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        log.warning("vllm TTS model config unreadable (%s) — serving %s as-is", exc, src)
        return src
    codec = str(cfg.get(codec_key) or "")
    if codec and os.path.isdir(codec):
        return src  # already points at a local codec
    if not os.path.isdir(codec_dir):
        log.warning("local audio tokenizer missing at %s — serving %s as-is (hub fetch will fail offline)",
                    codec_dir, src)
        return src
    shadow = os.path.join(REPO_ROOT, "models", shadow_name)
    os.makedirs(shadow, exist_ok=True)
    for entry in os.listdir(src):
        if entry == "config.json":
            continue
        link = os.path.join(shadow, entry)
        if not os.path.lexists(link):
            try:
                os.symlink(os.path.join(src, entry), link)
            except FileExistsError:
                pass  # parallel engine spawn raced us — same target either way
    cfg[codec_key] = codec_dir
    # engines spawn as THREADS of one process (parallel pool), so the tmp name
    # must be unique per CALL, not per pid — mkstemp, then atomic replace
    fd, tmp = tempfile.mkstemp(prefix=".config.json.", dir=shadow)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(shadow, "config.json"))
    log.info("vllm TTS shadow model dir ready: %s (codec -> %s)", shadow, codec_dir)
    return shadow


def _adopt_identity_ok(base_url: str, settings: Settings) -> bool:
    """Guard adopt-don't-own against a WRONG-provider orphan on the port.

    A stale process from a previous TTS_PROVIDER answering /health would be
    silently adopted and speak with the wrong model. vLLM engines are checked
    via /v1/models served names; uvicorn sidecars via the /health `provider`
    field (absent on the legacy nano sidecar → accepted). Verification
    ERRORS keep the old adopt behavior — only a positive mismatch refuses.
    """
    from .adapters.tts.providers import VLLM_ENGINE_PROVIDERS, canonical_tts_provider

    provider = canonical_tts_provider(settings.tts_provider)
    base = base_url.rstrip("/")
    try:
        if provider in VLLM_ENGINE_PROVIDERS:
            expected = {
                "vllm_omni": settings.tts_vllm_served_name,
                "cosyvoice3": settings.tts_cosy3_served_name,
                "moss_tts_realtime": settings.tts_mossrt_served_name,
            }[provider]
            with urllib.request.urlopen(f"{base}/v1/models", timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8") or "{}")
            served = {str(m.get("id") or "") for m in payload.get("data") or []}
            if expected not in served:
                log.error("healthy process at %s serves %s, not %r — refusing to adopt "
                          "(sweep 18100+ orphans via demo.sh down)", base_url, sorted(served), expected)
                return False
            return True
        with urllib.request.urlopen(f"{base}/health", timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
        advertised = str(payload.get("provider") or "")
        if not advertised:
            return True  # legacy nano sidecar predates the field
        expected_prefix = {
            "moss_tts_nano": "moss_tts_nano",
            "cosyvoice3_native": "cosyvoice3",
            "moss_tts_realtime_native": "moss_tts_realtime",
        }.get(provider, provider)
        if not advertised.startswith(expected_prefix):
            log.error("healthy sidecar at %s advertises provider %r, expected %r* — refusing to adopt",
                      base_url, advertised, expected_prefix)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — can't verify → adopt as before
        log.warning("adopt identity probe failed at %s (%s) — adopting anyway", base_url, exc)
        return True


def spawn_tts_sidecar(settings: Settings, base_url: Optional[str] = None,
                      gpu_index: Optional[int] = None,
                      log_path: Optional[str] = None) -> Optional[subprocess.Popen]:
    """Ensure a healthy TTS sidecar; return the Popen ONLY if we own it.

    With no overrides this is exactly the single-sidecar behavior; the TTS
    sidecar pool (server/gpu/supervisor.py) calls it once per placement-plan
    TtsSpec with an explicit port/GPU/log.
    """
    from .adapters.tts.providers import (
        canonical_tts_provider, is_external_provider, is_vllm_engine_provider)

    if not (settings.tts_enabled and settings.tts_spawn):
        log.info("TTS sidecar spawn disabled (enabled=%s spawn=%s)", settings.tts_enabled, settings.tts_spawn)
        return None
    if is_external_provider(settings.tts_provider):
        log.info("TTS provider %s is an external API — no sidecar to spawn",
                 settings.tts_provider)
        return None
    base_url = base_url or settings.moss_tts_nano_base_url
    log_path = log_path or settings.tts_sidecar_log
    if _is_healthy(base_url):
        if _adopt_identity_ok(base_url, settings):
            log.info("Adopting already-healthy TTS sidecar at %s", base_url)
            return None
        # refused adopt: fall through and spawn — the bind will fail loudly
        # against the squatter instead of silently speaking the wrong model

    provider = canonical_tts_provider(settings.tts_provider)
    is_vllm = is_vllm_engine_provider(provider)
    port = urllib.parse.urlparse(base_url).port or 18100
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cwd = settings.tts_sidecar_dir
    slow_first_boot = is_vllm  # providers whose FIRST boot exceeds the 240s default
    if is_vllm:
        # vLLM-Omni engine: its own venv (deps
        # conflict with this one), local checkpoint path (air-gapped box), and
        # the same /health endpoint the gate below polls.
        model_dir, served_name, mem_util, extra_args = {
            "vllm_omni": (
                _ensure_vllm_model_dir(settings.tts_vllm_model, settings.tts_vllm_codec,
                                       "moss-tts-nano-vllm"),
                settings.tts_vllm_served_name, settings.tts_vllm_gpu_mem_util,
                settings.tts_vllm_extra_args),
            "cosyvoice3": (  # self-contained checkpoint — no codec patch
                _ensure_vllm_model_dir(settings.tts_cosy3_model, "", "fun-cosyvoice3-vllm"),
                settings.tts_cosy3_served_name, settings.tts_cosy3_gpu_mem_util,
                settings.tts_cosy3_extra_args),
            "moss_tts_realtime": (
                _ensure_vllm_model_dir(settings.tts_mossrt_model, settings.tts_mossrt_codec,
                                       "moss-tts-realtime-vllm",
                                       codec_key="codec_model_name_or_path"),
                settings.tts_mossrt_served_name, settings.tts_mossrt_gpu_mem_util,
                settings.tts_mossrt_extra_args),
        }[provider]
        cmd = [settings.tts_vllm_bin, "serve", model_dir,
               "--omni", "--host", "127.0.0.1", "--port", str(port),
               "--served-model-name", served_name,
               "--gpu-memory-utilization", str(mem_util),
               "--trust-remote-code"]
        if extra_args:
            cmd += extra_args.split()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        # the engine rotates its own log (logs/handler/sidecar/) — raw stdout
        # is discarded below
        env["VLLM_LOGGING_CONFIG_PATH"] = _write_vllm_logging_config(log_path)
        # FlashInfer stays the accelerator wherever it has prebuilt cubins
        # (H200 sm_90, 4090 sm_89). On Blackwell (sm_12x) it has none and must
        # JIT-compile its top-k/top-p sampling kernels — which flashinfer 0.6.12
        # refuses below a CUDA 12.9 toolkit (this venv's usable nvcc is 12.8;
        # the pip 12.9 nvcc ships no binary and the cu13 nvcc is header-
        # incompatible), so the engine core dies at load with "FlashInfer
        # requires GPUs with sm75 or higher". Fall back to vLLM's native sampler
        # ONLY there (correct everywhere, negligibly slower at TTS batch sizes;
        # attention independently auto-selects FLASH_ATTN). Explicit export wins.
        cc = _target_compute_cap(gpu_index)
        if cc and cc[0] >= 12:
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            log.info("TTS engine on sm_%d%d: flashinfer sampler disabled "
                     "(no cubins + CUDA<12.9 JIT); using vLLM native sampler",
                     cc[0], cc[1])
    elif provider == "cosyvoice3_native":
        # vendored CosyVoice repo sidecar in .venv-cosy; first boot may build
        # TensorRT flow-estimator engines (minutes) — slow_first_boot
        cmd = [settings.tts_cosy3_python, "-m", "uvicorn", "cosyvoice3_sidecar:app",
               "--host", "127.0.0.1", "--port", str(port)]
        cwd = settings.tts_cosy3_sidecar_dir
        env.setdefault("COSY3_MODEL_DIR", settings.tts_cosy3_native_model)
        env.setdefault("TTS_VOICE_PROMPT_DIR", settings.tts_voice_prompt_dir)
        env["HF_HUB_OFFLINE"] = "1"
        slow_first_boot = True
    elif provider == "moss_tts_realtime_native":
        # UPSTREAM fast_api.py session server from the vendored MOSS-TTS repo,
        # transformers-5.0 stack in .venv-mossrt; model loads in its lifespan
        # (health answers only once loaded) — slow_first_boot
        cmd = [settings.tts_mossrt_python, "-m", "uvicorn", "fast_api:app",
               "--host", "127.0.0.1", "--port", str(port)]
        cwd = settings.tts_mossrt_sidecar_dir
        env.setdefault("MOSS_TTS_MODEL_PATH", settings.tts_mossrt_model)
        env.setdefault("MOSS_TTS_TOKENIZER_PATH", settings.tts_mossrt_model)
        env.setdefault("MOSS_TTS_CODEC_MODEL_PATH", settings.tts_mossrt_codec)
        env.setdefault("MOSS_TTS_DEVICE", "cuda:0")  # CUDA_VISIBLE_DEVICES picks the card
        env.setdefault("MOSS_TTS_ATTN_IMPL", os.getenv("MOSSRT_ATTN_IMPL", "sdpa"))
        env["HF_HUB_OFFLINE"] = "1"
        # native fast_api loads the ref-audio prompt via torchcodec, which needs
        # the vendored FFmpeg .so closure on LD_LIBRARY_PATH. The sidecar env
        # does NOT inherit it (unlike the VLM/sglang workers, whose supervisors
        # inject it), so `import torchcodec` in .venv-mossrt fails and every
        # start_turn errors "TorchCodec is required" → silent empty audio.
        ff = settings.ffmpeg_libs
        if ff and os.path.isdir(ff):
            _cur = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{ff}:{_cur}" if _cur else ff
        slow_first_boot = True
    else:
        python = settings.tts_python or sys.executable
        cmd = [python, "-m", "uvicorn", "moss_tts_nano_sidecar:app", "--host", "127.0.0.1", "--port", str(port)]
        env["MOSS_TTS_NANO_DEVICE"] = settings.tts_sidecar_device
        # the VENDORED sidecar's relative layout can't find the checkpoints
        # (third_party/models isn't vendored — see its README): hand it the
        # MODELS_DIR paths (same ckpts serve both providers). setdefault so a
        # caller-exported MOSS_TTS_NANO_* still wins.
        env.setdefault("MOSS_TTS_NANO_CHECKPOINT", settings.tts_vllm_model)
        env.setdefault("MOSS_TTS_NANO_AUDIO_TOKENIZER", settings.tts_vllm_codec)
    if gpu_index is not None:
        set_visible_device(env, gpu_index)
    elif settings.tts_sidecar_gpu:
        set_visible_device(env, int(settings.tts_sidecar_gpu))
    log.info("Spawning TTS sidecar: %s (cwd=%s, log=%s)", " ".join(cmd), cwd, log_path)
    if is_vllm:
        # everything python-level goes through the engine's own rotating log
        # (VLLM_LOGGING_CONFIG_PATH above); native stderr is accepted as lost
        spawn_cmd = cmd
        stdout_target = subprocess.DEVNULL
    else:
        # the board-owned pytorch sidecar logs to stdout only (basicConfig) —
        # pipe it through the rotating tee. bash is the process-group leader,
        # so stop_tts_sidecar's killpg reaches sidecar AND tee; the tee itself
        # ignores TERM and drains to EOF (pipefail keeps the sidecar's rc).
        # The pipeline lives in its own session: a gateway restart leaves it
        # running and the next gateway adopts it via /health, tee intact.
        tee = [sys.executable, _ROTATING_TEE, "--quiet", log_path]
        spawn_cmd = ["bash", "-c",
                     'set -o pipefail; exec "$0" "$@" 2>&1 | exec ' + shlex.join(tee),
                     *cmd]
        stdout_target = subprocess.DEVNULL
    proc = subprocess.Popen(
        spawn_cmd,
        cwd=cwd,
        env=env,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group: signals to uvicorn don't hit the child
    )

    timeout_s = settings.tts_health_timeout_s
    if slow_first_boot and "TTS_HEALTH_TIMEOUT_S" not in os.environ:
        # a FIRST boot pays engine init + compile-cache/TRT-plan population and
        # can exceed the sidecar-tuned 240s default; an explicit env still wins
        timeout_s = max(timeout_s, 600.0)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.error("TTS sidecar exited during startup (rc=%s) — see %s", proc.returncode, log_path)
            return None
        if _is_healthy(base_url):
            log.info("TTS sidecar healthy at %s (pid=%s)", base_url, proc.pid)
            return proc
        time.sleep(2.0)

    log.error(
        "TTS sidecar not healthy within %.0fs — terminating it; voice runs degraded (see %s)",
        timeout_s,
        log_path,
    )
    stop_tts_sidecar(proc)
    return None


def stop_tts_sidecar(proc: Optional[subprocess.Popen]) -> None:
    """Terminate an owned sidecar (process group), escalating after 10s."""
    if proc is None or proc.poll() is not None:
        return
    log.info("Stopping TTS sidecar (pid=%s)", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log.warning("TTS sidecar ignored SIGTERM; killing")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
