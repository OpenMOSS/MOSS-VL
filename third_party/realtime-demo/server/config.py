"""Env-driven settings (single source of truth).

Plain-dataclass config loaded from environment variables (no pydantic-settings
dependency). Defaults point at the reference board's on-disk models so the demo
backend runs against the same checkpoints.

Four config layers, highest priority first (same model as docker-compose's
CLI > environment > env_file > image-ENV):

  1. in-command env        VAR=x demo.sh up   /   demo.sh up VAR=x
  2. startup-script pins   the `: "${VAR:=x}"; export VAR` blocks in
                           scripts/deploy/run_*.sh (yield to 1, beat 3)
  3. .env.deploy           plain KEY=VALUE file at the repo root (gitignored;
                           see .env.deploy.example). Loaded by get_settings()
                           below AND by scripts/deploy/env_lib.sh, both with
                           setdefault semantics — a set env var always wins.
                           NOT bash-sourced: no $VAR expansion, values are
                           literal; optional `export ` prefixes are accepted.
                           ENV_DEPLOY_FILE overrides the path; empty disables.
  4. code defaults         the dataclass fields below — THE place defaults live
                           (deploy scripts are orchestration-only).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, fields
from typing import Dict, Optional


# Repo root (…/MOSS-VL-Realtime_Demo_App). Durable data defaults live under it because
# the repo sits on the shared FS that survives pod restarts (unlike /tmp).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _models_dir() -> str:
    """Model-zoo root — the repo-local `models/` tree (gitignored; on the shared
    FS, so it survives pod restarts). Grouped by kind: models/asr, models/tts,
    models/vlms (see _asr_dir/_tts_dir/_vlms_dir). A default PREFIX only — every
    model below still has its own env knob, so one model can be repointed by its
    var or the whole zoo by MODELS_DIR. A function, not a module constant: it
    must see a MODELS_DIR that .env.deploy provides."""
    return os.getenv("MODELS_DIR", os.path.join(REPO_ROOT, "models"))


def _asr_dir() -> str:
    return os.path.join(_models_dir(), "asr")   # models/asr/*


def _tts_dir() -> str:
    return os.path.join(_models_dir(), "tts")   # models/tts/*


def _vlms_dir() -> str:
    return os.path.join(_models_dir(), "vlms")  # models/vlms/*


def _memory_dir() -> str:
    return os.path.join(_models_dir(), "memory")  # models/memory/*


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _default_memory_pi_url() -> str:
    return f"http://127.0.0.1:{_env('PI_PORT') or '38082'}"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_opt_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Unset → `default`; explicit empty string → None (the escape hatch back to
    'no override' for fields whose code default is a real value)."""
    v = os.getenv(name)
    if v is None:
        return default
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return default


def _env_opt_float(name: str) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --------------------- .env.deploy (config layer 3) ---------------------

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPORT_RE = re.compile(r"^export[ \t]+")

_env_deploy_loaded = False


def _parse_env_file(path: str) -> Dict[str, str]:
    """Plain KEY=VALUE parser (see the module docstring for the contract).

    Mirrors scripts/deploy/env_lib.sh line for line — change BOTH together.
    Accepts `export ` prefixes and CRLF; values keep any further `=`; one
    matching surrounding quote pair is stripped (no escapes, no $ expansion);
    unquoted values lose a trailing ` # comment` (bash-source parity);
    duplicate keys: last wins. Malformed lines are skipped, never fatal.
    """
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            line = _EXPORT_RE.sub("", line)
            key, _, value = line.partition("=")
            key = key.strip()
            if not _KEY_RE.match(key):
                continue
            value = value.strip()
            if value[:1] in ("\"", "'"):
                close = value.find(value[0], 1)
                if close > 0:  # quoted: take the inner text, drop any trailer
                    value = value[1:close]
                # unterminated quote: keep the raw text (quote char included)
            else:
                value = re.sub(r"\s+#.*$", "", value).strip()
            out[key] = value
    return out


def _load_env_deploy() -> None:
    """Merge .env.deploy into os.environ with setdefault semantics (a var set by
    layers 1-2 is never touched). One attempt per process; mutating os.environ
    is deliberate — direct-env readers (logging_conf, TTS adapters) and spawned
    children (workers/sidecars copy os.environ) all see layer 3 through it."""
    global _env_deploy_loaded
    if _env_deploy_loaded:
        return
    _env_deploy_loaded = True
    path = os.environ.get("ENV_DEPLOY_FILE")
    if path == "":  # explicit kill-switch
        return
    if path is None:
        path = os.path.join(REPO_ROOT, ".env.deploy")
    if not os.path.isfile(path):
        return
    try:
        for key, value in _parse_env_file(path).items():
            os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"WARNING: .env.deploy unreadable ({exc}) — continuing without it",
              file=sys.stderr)


def _reset_settings_for_tests() -> None:
    global _settings, _env_deploy_loaded
    _settings = None
    _env_deploy_loaded = False


def _default_tts_provider() -> str:
    """Prefer MOSS-TTS-Realtime (fast: ~180ms TTFB, RTF~0.5, and correctly
    sampled — no garbling) when its checkpoint is present and .venv-vllm is
    built; else the Nano-on-vLLM engine; else the pytorch sidecar. A box
    without the engine venv/checkpoints never lands on a broken TTS. Explicit
    TTS_PROVIDER always wins (e.g. moss_tts_realtime_native for streaming)."""
    explicit = os.getenv("TTS_PROVIDER")
    if explicit:
        return explicit
    vllm_bin = os.getenv("TTS_VLLM_BIN", os.path.join(REPO_ROOT, ".venv-vllm", "bin", "vllm"))
    if os.access(vllm_bin, os.X_OK):
        mossrt_model = os.getenv("MOSSRT_MODEL", os.path.join(_tts_dir(), "MOSS-TTS-Realtime"))
        return "moss_tts_realtime" if os.path.isdir(mossrt_model) else "vllm_omni"
    return "moss_tts_nano"


def _default_offline_provider() -> str:
    """Same venv-gated auto-default as TTS: without .venv-sglang (build:
    scripts/build_venv_sglang.sh) the offline plane is off and offline chat
    falls back to the online HF workers. Explicit OFFLINE_PROVIDER wins."""
    explicit = os.getenv("OFFLINE_PROVIDER")
    if explicit:
        return explicit
    sglang_py = os.getenv("SGLANG_PYTHON", os.path.join(REPO_ROOT, ".venv-sglang", "bin", "python"))
    return "sglang" if os.access(sglang_py, os.X_OK) else "none"


@dataclass(frozen=True)
class Settings:
    # ---- gateway ----
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))

    # ---- VLM (online streaming + offline chat, HF path) ----
    # ONLINE streaming checkpoint (models/vlms/hf_mossvl_streaming_processor) — a
    # self-contained HF bundle (weights + processor); it also supplies the fixed
    # canonical streaming processor below. Env vars override either.
    model_path: str = field(default_factory=lambda: _env(
        "MODEL_PATH", os.path.join(_vlms_dir(), "hf_mossvl_streaming_processor")))
    gpu_id: int = field(default_factory=lambda: _env_int("GPU_ID", 0))
    autoload_vlm: bool = field(default_factory=lambda: _env_flag("AUTOLOAD_VLM", True))
    hf_mode: str = field(default_factory=lambda: _env("HF_MODE", "online_streaming"))
    mossvl_streaming_processor_path: str = field(
        default_factory=lambda: _env(
            "MOSSVL_STREAMING_PROCESSOR_PATH",
            os.path.join(_vlms_dir(), "hf_mossvl_streaming_processor")))
    # auto = pick by compute capability (sm_120/Blackwell has no flash-attn
    # cubins → sdpa; everything else → flash_attention_2). Explicit value wins.
    attn_impl: str = field(default_factory=lambda: _env("ATTN_IMPL", "auto"))
    attn_impl_fallback: str = field(default_factory=lambda: _env("ATTN_IMPL_FALLBACK", "sdpa"))
    # vendored FFmpeg .so closure for torchcodec (the ckpt's video remote code):
    # the supervisors prepend it to every child's LD_LIBRARY_PATH themselves, so
    # workers/sidecars decode video no matter how the gateway was launched
    # (run_backend.sh exports the same default for the gateway's own process)
    ffmpeg_libs: str = field(default_factory=lambda: _env(
        "FFMPEG_LIBS", os.path.join(REPO_ROOT, ".venv", "lib", "ffmpeg")))

    # ---- multi-GPU deployment (server/gpu/, server/vlm_worker/) ----
    # workers = one VLM worker process per detected GPU (replica-per-GPU);
    # inproc = the pre-multi-GPU single-process path (rollback lever).
    vlm_deploy: str = field(default_factory=lambda: _env("VLM_DEPLOY", "workers"))
    vlm_worker_base_port: int = field(default_factory=lambda: _env_int("VLM_WORKER_BASE_PORT", 9000))
    # comma list of GPU indices ("0,1", repeats allowed e.g. "0,0" for fake
    # multi-GPU tests); empty = auto (every GPU with enough free VRAM)
    vlm_worker_gpus: str = field(default_factory=lambda: _env("VLM_WORKER_GPUS", ""))
    vlm_min_free_mib: int = field(default_factory=lambda: _env_int("VLM_MIN_FREE_MIB", 22000))
    vlm_health_timeout_s: float = field(default_factory=lambda: _env_float("VLM_HEALTH_TIMEOUT_S", 600.0))
    # stagger worker spawns: N parallel bf16 loads thrash the shared FS
    vlm_spawn_stagger_s: float = field(default_factory=lambda: _env_float("VLM_SPAWN_STAGGER_S", 15.0))
    # worker logs live in-repo (shared FS: survives pod restarts, readable from
    # the CPU box): worker_<i>.log = rotating python log (incl. uvicorn access
    # lines). Raw worker stdout is discarded at spawn. logs/ is gitignored.
    vlm_worker_log_dir: str = field(default_factory=lambda: _env(
        "VLM_WORKER_LOG_DIR",
        os.path.join(REPO_ROOT, "logs", "handler", "backend", "workers")))
    # fake worker mode: no model load, scripted token stream behind the same RPC
    # (pool/manager/WS tests on boxes that cannot fit N model replicas)
    vlm_worker_fake: bool = field(default_factory=lambda: _env_flag("VLM_WORKER_FAKE", False))

    # ---- sglang-omni remote realtime backend (VLM_DEPLOY=sglang_omni) ----
    # adapters/vlm/moss_vl_sglang_omni/: the realtime VLM lives on remote
    # sglang-omni servers; no local worker processes are spawned in this mode.
    # comma-separated http base URLs, one replica per entry; empty = unconfigured
    sglang_omni_urls: str = field(default_factory=lambda: _env("SGLANG_OMNI_URLS", ""))
    # concurrent realtime sessions each remote replica may host (pool slots per
    # replica). MUST equal the remote omni instance's --max-running-requests
    # (deploy.conf MAX_RUNNING_REQUESTS): single-GPU KV pool ÷ N = per-session
    # safe context, so keep 1 until the P5 load test establishes a safe N.
    sglang_omni_sessions_per_replica: int = field(
        default_factory=lambda: _env_int("SGLANG_OMNI_SESSIONS_PER_REPLICA", 1))
    # server-side in-flight input slots; full → the gateway waits briefly then
    # drops PURE frames (prompts never drop)
    sglang_omni_input_queue_capacity: int = field(
        default_factory=lambda: _env_int("SGLANG_OMNI_INPUT_QUEUE_CAPACITY", 4))
    # how long a pure frame waits for an input credit before being dropped
    sglang_omni_input_drop_wait_seconds: float = field(
        default_factory=lambda: _env_float("SGLANG_OMNI_INPUT_DROP_WAIT_SECONDS", 0.5))
    sglang_omni_connect_timeout_s: float = field(
        default_factory=lambda: _env_float("SGLANG_OMNI_CONNECT_TIMEOUT_S", 10.0))
    # Older backends without negotiated usage need a conservative token budget.
    sglang_omni_context_length: int = field(
        default_factory=lambda: _env_int("SGLANG_OMNI_CONTEXT_LENGTH", 131072))
    sglang_omni_fallback_frame_tokens: int = field(
        default_factory=lambda: _env_int("SGLANG_OMNI_FALLBACK_FRAME_TOKENS", 2048))
    sglang_omni_context_reserve_tokens: int = field(
        default_factory=lambda: _env_int("SGLANG_OMNI_CONTEXT_RESERVE_TOKENS", 4096))
    # DOWN-replica re-probe cadence (capacity-exceeded quarantine / health recovery)
    sglang_omni_health_interval_s: float = field(
        default_factory=lambda: _env_float("SGLANG_OMNI_HEALTH_INTERVAL_S", 10.0))

    # ---- gateway plane (server/gateway/): thin passthrough in front of sglang-omni ----
    # one-shot token binding a created session to its WSS attach
    gateway_ws_token_ttl_s: float = field(
        default_factory=lambda: _env_float("GATEWAY_WS_TOKEN_TTL_S", 60.0))
    # a created session that no client attaches to within this window is destroyed
    gateway_attach_timeout_s: float = field(
        default_factory=lambda: _env_float("GATEWAY_ATTACH_TIMEOUT_S", 90.0))
    # Both layers enforce their own cap; advertise the smaller effective limit.
    # The gateway terminates oversized-frame sessions with error + close 1009.
    gateway_max_frame_bytes: int = field(
        default_factory=lambda: _env_int("GATEWAY_MAX_FRAME_BYTES", 32 * 1024 * 1024))
    gateway_create_timeout_s: float = field(
        default_factory=lambda: _env_float("GATEWAY_CREATE_TIMEOUT_S", 30.0))
    gateway_model_version: str = field(
        default_factory=lambda: _env("GATEWAY_MODEL_VERSION", ""))
    gateway_model_versions: str = field(
        default_factory=lambda: _env("GATEWAY_MODEL_VERSIONS", "{}"))
    # per-session reconciliation ledger (P3): one JSONL record per session
    # teardown; "" → {data_dir}/gateway_usage.jsonl
    gateway_usage_log: str = field(default_factory=lambda: _env("GATEWAY_USAGE_LOG", ""))

    # ---- offline chat backend (sglang sidecars; adapters/vlm/moss_vl_sglang/adapter.py) ----
    # The GPU fleet splits online/offline: offline gets round(n * ratio) of the
    # eligible GPUs (min 1 when n >= 2, never all of them, 0 on 1-GPU boxes), so
    # 8 -> 2 offline + 6 online, 4 -> 1+3, 2 -> 1+1, 1 -> online only. Each
    # offline GPU runs one `sglang.launch_server` sidecar (the fnlp-vision fork —
    # stock sglang cannot serve model_type moss_vl) from .venv-sglang; chat
    # requests fall back to the online HF pool whenever this plane is down.
    offline_provider: str = field(default_factory=_default_offline_provider)  # sglang | none
    offline_model_path: str = field(default_factory=lambda: _env(
        "OFFLINE_MODEL_PATH", os.path.join(_vlms_dir(), "offline")))
    offline_gpu_ratio: float = field(default_factory=lambda: _env_float("OFFLINE_GPU_RATIO", 0.25))
    # explicit offline GPU count (clamped to n_eligible - 1); None = ratio rule
    offline_gpu_count: Optional[int] = field(default_factory=lambda: _env_opt_int("OFFLINE_GPU_COUNT"))
    # explicit offline GPU indices (comma-separated) — wins over count/ratio;
    # symmetric with VLM_WORKER_GPUS. On NPU boxes the highest-index default
    # can land on a card the sglang engine cannot address.
    offline_gpus: str = field(default_factory=lambda: _env("OFFLINE_GPUS", ""))
    sglang_python: str = field(default_factory=lambda: _env(
        "SGLANG_PYTHON", os.path.join(REPO_ROOT, ".venv-sglang", "bin", "python")))
    # NOT sglang's default 30000: this box is shared, and adopt-if-healthy must
    # never mistake someone else's engine for ours (identity-checked besides)
    sglang_base_port: int = field(default_factory=lambda: _env_int("SGLANG_BASE_PORT", 30800))
    # offline GPUs are dedicated (no VLM worker / TTS / ASR colocation), so the
    # engine can take most of the card — board's 0.35 was for shared GPUs
    sglang_mem_fraction: float = field(default_factory=lambda: _env_float("SGLANG_MEM_FRACTION", 0.80))
    sglang_tp_size: int = field(default_factory=lambda: _env_int("SGLANG_TP_SIZE", 1))
    sglang_extra_args: str = field(default_factory=lambda: _env("SGLANG_EXTRA_ARGS", ""))
    # first boot pays weight load + flashinfer JIT/warmup (gated on /health_generate)
    sglang_health_timeout_s: float = field(default_factory=lambda: _env_float("SGLANG_HEALTH_TIMEOUT_S", 900.0))
    sglang_log_dir: str = field(default_factory=lambda: _env(
        "SGLANG_LOG_DIR", os.path.join(REPO_ROOT, "logs", "handler", "sidecar")))

    # ---- per-session KV budget (server/gpu/kv_budget.py) ----
    # hard floor: every session must be able to stream video this long
    kv_session_min_minutes: float = field(default_factory=lambda: _env_float("KV_SESSION_MIN_MINUTES", 3.0))
    # when memory is abundant, allocate this ratio of post-weights free VRAM
    kv_memory_ratio: float = field(default_factory=lambda: _env_float("KV_MEMORY_RATIO", 0.9))
    kv_safety_margin_mib: int = field(default_factory=lambda: _env_int("KV_SAFETY_MARGIN_MIB", 2048))
    # KV growth rate for budgeting (tokens/s of session time). Default from the
    # sizing estimate (~250 vision tok/frame @1 fps + 20 text tok/s); replace
    # with empirical measurements for the served checkpoint.
    kv_tokens_per_second_est: float = field(default_factory=lambda: _env_float("KV_TOKENS_PER_SECOND_EST", 270.0))
    # hard = stop the session at budget exhaustion; soft = warn only;
    # auto = hard on <32 GB GPUs (OOM-real), soft on big boxes
    kv_enforce: str = field(default_factory=lambda: _env("KV_ENFORCE", "auto"))
    # Realtime frame window: keep only the last N MINUTES of video frames, evicting
    # older frames' vision KV (the ~250 tok/frame that dominate the cache) while the
    # conversation words/turns stay in the text self-attention cache untouched. MOSS-VL
    # is cross-attention, so frames live in a separate KV lane — a clean FIFO prefix
    # drop, no RoPE re-indexing (see server/realtime/mossvl_patches.py). Bounds KV
    # growth, so KV_ENFORCE above becomes a rarely-tripped safety net. 0 = off (grow
    # unbounded, pre-window behavior). Frame budget = minutes*60*fps (VIDEO_FPS or 2).
    realtime_frame_window_minutes: float = field(
        default_factory=lambda: _env_float("REALTIME_FRAME_WINDOW_MINUTES", 5.0))

    # session capacity backstop (0 = auto: the VLM replica pool's capacity)
    max_sessions: int = field(default_factory=lambda: _env_int("MAX_SESSIONS", 0))
    # Override for the streaming config's vision_seq_pad_multiple. Default 1:
    # SFT ckpts ship 8, whose ragged-pad branch crashes the KV/mask on barge-in
    # when a frame's vision-token count isn't a multiple of 8 ("expanded size
    # (288) must match (285)", reproduced live) — webcam frames hit arbitrary
    # token counts, so force the validated pad=1 path. Set to the EMPTY string
    # (VISION_SEQ_PAD_MULTIPLE=) to leave the ckpt value untouched.
    vision_seq_pad_multiple_override: Optional[int] = field(
        default_factory=lambda: _env_opt_int("VISION_SEQ_PAD_MULTIPLE", 1))

    # realtime generation params. NOTE: realtime sessions take do_sample/
    # temperature/top_k/top_p from SessionConfig.params (server/schemas.py),
    # NOT these GEN_* envs — the effective realtime default lives there
    # (do_sample=True, board parity). GEN_* feeds the offline-chat Settings
    # path; kept aligned for consistency.
    temperature: float = field(default_factory=lambda: _env_float("GEN_TEMPERATURE", 0.7))
    top_k: int = field(default_factory=lambda: _env_int("GEN_TOP_K", 20))
    top_p: float = field(default_factory=lambda: _env_float("GEN_TOP_P", 0.8))
    do_sample: bool = field(default_factory=lambda: _env_flag("GEN_DO_SAMPLE", True))
    repetition_penalty: float = field(default_factory=lambda: _env_float("GEN_REPETITION_PENALTY", 1.0))
    # Selective repetition penalty (server/adapters/vlm/moss_vl_hf/rep_penalty.py).
    # Active ONLY when the effective penalty != 1.0 (default 1.0 = OFF — zero
    # behavior change). Window = most-recent NON-EXEMPT (real-text) ids
    # penalized; 0 = full context. Filter-then-window so per-frame scaffold /
    # timestamp splices can't consume the window. llama.cpp practice:
    # repeat_last_n 64-128; 256 here because scaffold-free text accrues slowly.
    rep_penalty_window: int = field(default_factory=lambda: _env_int("GEN_REP_PENALTY_WINDOW", 256))
    # 1 (default): exempt control/added tokens — <|silence|> above all
    # (penalizing the idle channel suppresses turn-ending and CAUSES
    # hallucinated rounds; cf. VideoLLM-online's protected EOS channel) — plus
    # punctuation/digits/whitespace and the " seconds" timestamp literal.
    # 0 = legacy stock HF full-context penalty (unsafe; debugging escape hatch).
    rep_penalty_exempt: bool = field(default_factory=lambda: _env_flag("GEN_REP_PENALTY_EXEMPT", True))
    max_new_tokens: int = field(default_factory=lambda: _env_int("GEN_MAX_NEW_TOKENS", 4096))
    # max_tokens_per_turn is a tokens-per-SECOND pacing knob in real_time_generate
    # (wait = 1/N - cost). It is now UNCAPPED by default (86400 = no pacing):
    # the frontend owns display pacing, and capping generation only starved the
    # (TPS-bound) TTS of input text. A throttle remains available for shared-GPU
    # debugging — a low cap free-runs fewer narration rounds and keeps KV smaller,
    # at the cost of slower replies and longer TTFA. Per-session override rides
    # GenerationParams.max_tokens_per_turn; the old UI default was 4.
    max_tokens_per_turn: int = field(default_factory=lambda: _env_int("GEN_MAX_TOKENS_PER_TURN", 86400))
    frame_queue_size: int = field(default_factory=lambda: _env_int("FRAME_QUEUE_SIZE", 256))
    system_prompt: Optional[str] = field(default_factory=lambda: os.getenv("REALTIME_SYSTEM_PROMPT"))
    initial_prompt: str = field(default_factory=lambda: _env("REALTIME_INITIAL_PROMPT", ""))

    # processor pixel / fps overrides (None = model default)
    min_pixels: Optional[int] = field(default_factory=lambda: _env_opt_int("MIN_PIXELS"))
    max_pixels: Optional[int] = field(default_factory=lambda: _env_opt_int("MAX_PIXELS"))
    video_fps: Optional[float] = field(default_factory=lambda: _env_opt_float("VIDEO_FPS"))
    min_frames: Optional[int] = field(default_factory=lambda: _env_opt_int("MIN_FRAMES"))
    max_frames: Optional[int] = field(default_factory=lambda: _env_opt_int("MAX_FRAMES"))
    multi_image_max_pixels: Optional[int] = field(default_factory=lambda: _env_opt_int("MULTI_IMAGE_MAX_PIXELS"))
    video_max_pixels: Optional[int] = field(default_factory=lambda: _env_opt_int("VIDEO_MAX_PIXELS"))

    # ---- ASR ----
    asr_provider: str = field(default_factory=lambda: _env("ASR_PROVIDER", "funasr_sensevoice"))
    asr_enabled: bool = field(default_factory=lambda: _env_flag("ASR_ENABLED", True))
    sensevoice_model: str = field(
        default_factory=lambda: _env("SENSEVOICE_MODEL", os.path.join(_asr_dir(), "SenseVoiceSmall"))
    )
    sensevoice_vad_model: str = field(
        default_factory=lambda: _env("SENSEVOICE_VAD_MODEL", os.path.join(_asr_dir(), "fsmn-vad"))
    )
    # auto = placement plan picks the highest-index eligible GPU (server/gpu/
    # placement.py); an explicit "cuda:N" / "cpu" always wins
    asr_device: str = field(default_factory=lambda: _env("ASR_DEVICE", "auto"))
    asr_fp16: bool = field(default_factory=lambda: _env_flag("ASR_FP16", True))
    asr_language: str = field(default_factory=lambda: _env("ASR_LANGUAGE", "zh"))
    asr_sample_rate: int = field(default_factory=lambda: _env_int("ASR_SAMPLE_RATE", 16000))
    asr_use_itn: bool = field(default_factory=lambda: _env_flag("SENSEVOICE_USE_ITN", True))
    asr_warmup: bool = field(default_factory=lambda: _env_flag("ASR_WARMUP", True))
    asr_min_pcm_bytes: int = field(default_factory=lambda: _env_int("ASR_MIN_PCM_BYTES", 16000 // 5 * 2))
    # Realtime partial captions: SenseVoice has no native streaming, so the
    # stream re-decodes its buffered audio every N ms and emits the hypothesis
    # via the standard on_partial callback (→ input.transcription.delta). A
    # genuinely streaming engine later just calls on_partial itself. 0 = off.
    asr_partial_interval_ms: int = field(default_factory=lambda: _env_int("ASR_PARTIAL_INTERVAL_MS", 800))
    # Partial re-decode window: a turn-level engine re-decodes to produce live
    # captions, but re-decoding the WHOLE (unbounded) turn every interval is
    # O(n²) and stalls long turns. Partials now decode only the last N seconds
    # of audio (the recent-hypothesis window) so their cost is O(1) regardless
    # of turn length; the FINAL still decodes the full turn (SenseVoice
    # VAD-segments it internally). 0 = whole buffer (legacy).
    asr_partial_window_s: float = field(default_factory=lambda: _env_float("ASR_PARTIAL_WINDOW_S", 15.0))
    # Memory safety ceiling on the retained per-turn PCM buffer — NOT a turn cap
    # (the user may hold the mic arbitrarily long). Only the oldest audio beyond
    # this many seconds is dropped so a runaway turn can't OOM; the final decode
    # still covers everything within the ceiling. 0 = unbounded.
    asr_buffer_max_s: float = field(default_factory=lambda: _env_float("ASR_BUFFER_MAX_S", 600.0))
    # OpenAI-realtime-style ASR (voxtral / qwen3-asr) — optional
    asr_ws_base_url: str = field(default_factory=lambda: _env("ASR_WS_BASE_URL", ""))
    asr_ws_model: str = field(default_factory=lambda: _env("ASR_WS_MODEL", ""))

    # ---- TTS ----
    tts_provider: str = field(default_factory=_default_tts_provider)
    tts_enabled: bool = field(default_factory=lambda: _env_flag("TTS_ENABLED", True))
    moss_tts_nano_base_url: str = field(
        default_factory=lambda: _env("MOSS_TTS_NANO_BASE_URL", "http://127.0.0.1:18100")
    )
    # -- sidecar lifecycle: the backend SPAWNS the TTS sidecar at startup (one
    # entry point brings the whole voice stack up; process isolation, not env
    # isolation — both default to THIS venv's interpreter). A sidecar already
    # answering /health is adopted instead (standalone runs, orphans).
    tts_spawn: bool = field(default_factory=lambda: _env_flag("TTS_SPAWN", True))
    # empty → sys.executable (same venv); set to another venv's python to split envs
    tts_python: str = field(default_factory=lambda: _env("TTS_PYTHON", ""))
    # the pytorch sidecar code is VENDORED in-repo (server/adapters/tts/moss_tts_nano/sidecar/
    # README.md) — no board dependency; spawn cwd stays env-overridable
    tts_sidecar_dir: str = field(
        default_factory=lambda: _env("TTS_SIDECAR_DIR", os.path.join(
            REPO_ROOT, "server", "adapters", "tts", "moss_tts_nano", "sidecar", "backend"))
    )
    tts_sidecar_gpu: str = field(default_factory=lambda: _env("TTS_GPU", ""))  # empty → inherit
    tts_sidecar_device: str = field(default_factory=lambda: _env("MOSS_TTS_NANO_DEVICE", "cuda"))
    tts_health_timeout_s: float = field(default_factory=lambda: _env_float("TTS_HEALTH_TIMEOUT_S", 240.0))
    tts_sidecar_log: str = field(default_factory=lambda: _env(
        "TTS_SIDECAR_LOG",
        os.path.join(REPO_ROOT, "logs", "handler", "sidecar", "tts.log")))
    # sidecar pool sizing: ceil(n_vlm_workers / this) sidecars, ports 18100+
    tts_sessions_per_sidecar: int = field(default_factory=lambda: _env_int("TTS_SESSIONS_PER_SIDECAR", 2))
    # explicit override for the TTS sidecar/engine COUNT (0 = auto, derived from
    # workers by the ceil formula in placement). Lets a low-worker box (e.g. 1
    # online GPU) serve N concurrent sessions: each native fast_api sidecar is
    # batch-1, so N sidecars = N concurrent sessions (the pool routes each to
    # the least-loaded one). They land on the online GPU(s); size N so
    # N * per-sidecar-VRAM fits (native ~13GiB; vLLM engines are far larger).
    tts_sidecar_count: int = field(default_factory=lambda: _env_int("TTS_SIDECAR_COUNT", 0))
    # -- vLLM-Omni TTS engine: TTS_PROVIDER=
    # vllm_omni swaps each sidecar for `vllm serve <ckpt> --omni` — same
    # backend-owned spawn/health-gate lifecycle, OpenAI /v1/audio/speech client.
    tts_vllm_bin: str = field(default_factory=lambda: _env(
        "TTS_VLLM_BIN", os.path.join(REPO_ROOT, ".venv-vllm", "bin", "vllm")))
    tts_vllm_model: str = field(default_factory=lambda: _env(
        "TTS_VLLM_MODEL",
        os.path.join(_tts_dir(), "MOSS-TTS-Nano-100M")))
    tts_vllm_served_name: str = field(default_factory=lambda: _env("TTS_VLLM_SERVED_NAME", "moss-tts-nano"))
    # local audio tokenizer for the engine — patched into a shadow config.json
    # (sidecars._ensure_vllm_model_dir) so no HF hub fetch happens offline
    tts_vllm_codec: str = field(default_factory=lambda: _env(
        "TTS_VLLM_CODEC",
        os.path.join(_tts_dir(), "MOSS-Audio-Tokenizer-Nano")))
    # co-located with a VLM worker. Raised 0.10 -> 0.30 (matches vllm-omni's own
    # moss_tts_nano.yaml recommendation): 0.10 starved the engine and inflated
    # TTFT/ITL under VLM contention (observed vllm_ttft ~1.3s → sentence drops).
    tts_vllm_gpu_mem_util: float = field(default_factory=lambda: _env_float("TTS_GPU_MEM_UTIL", 0.30))
    tts_vllm_extra_args: str = field(default_factory=lambda: _env("TTS_VLLM_EXTRA_ARGS", ""))
    # continuous batching absorbs whole-session bursts → more sessions per
    # engine than per pytorch sidecar (pool sizing when provider is vllm_omni)
    tts_sessions_per_engine: int = field(default_factory=lambda: _env_int("TTS_SESSIONS_PER_ENGINE", 4))
    # -- Fun-CosyVoice3-0.5B via vLLM-Omni (TTS_PROVIDER=cosyvoice3): same
    # backend-owned `vllm serve --omni` lifecycle as vllm_omni, different ckpt.
    # CosyVoice3 is cloning-only upstream: every request carries ref_audio AND
    # ref_text (transcripts resolved from the vendored Nano assets/demo.jsonl).
    tts_cosy3_model: str = field(default_factory=lambda: _env(
        "COSY3_MODEL", os.path.join(_tts_dir(), "Fun-CosyVoice3-0.5B-2512")))
    tts_cosy3_served_name: str = field(default_factory=lambda: _env("COSY3_SERVED_NAME", "fun-cosyvoice3"))
    # 0.5B LLM + flow + vocoder colocated with a VLM worker
    tts_cosy3_gpu_mem_util: float = field(default_factory=lambda: _env_float("COSY3_GPU_MEM_UTIL", 0.15))
    tts_cosy3_extra_args: str = field(default_factory=lambda: _env("COSY3_VLLM_EXTRA_ARGS", ""))
    # native fallback sidecar (TTS_PROVIDER=cosyvoice3_native): vendored
    # CosyVoice repo in its own venv (scripts/build_venv_cosyvoice.sh). NOTE the
    # native ckpt is the ModelScope layout (llm.pt/flow.pt/hift.pt + yaml, zoo
    # dir Fun-CosyVoice3-0.5B) — NOT the HF-format -2512 dir vllm serves.
    tts_cosy3_native_model: str = field(default_factory=lambda: _env(
        "COSY3_NATIVE_MODEL", os.path.join(_tts_dir(), "Fun-CosyVoice3-0.5B")))
    tts_cosy3_python: str = field(default_factory=lambda: _env(
        "COSY3_PYTHON", os.path.join(REPO_ROOT, ".venv-cosy", "bin", "python")))
    tts_cosy3_sidecar_dir: str = field(default_factory=lambda: _env(
        "COSY3_SIDECAR_DIR", os.path.join(
            REPO_ROOT, "server", "adapters", "tts", "cosyvoice3", "sidecar", "backend")))
    # -- MOSS-TTS-Realtime (1.7B) via vLLM-Omni (TTS_PROVIDER=moss_tts_realtime):
    # codec is the FULL MOSS-Audio-Tokenizer (not the Nano one) — patched into a
    # shadow config.json like the nano-vllm path so nothing fetches from the hub.
    tts_mossrt_model: str = field(default_factory=lambda: _env(
        "MOSSRT_MODEL", os.path.join(_tts_dir(), "MOSS-TTS-Realtime")))
    tts_mossrt_codec: str = field(default_factory=lambda: _env(
        "MOSSRT_CODEC", os.path.join(_tts_dir(), "MOSS-Audio-Tokenizer")))
    tts_mossrt_served_name: str = field(default_factory=lambda: _env("MOSSRT_SERVED_NAME", "moss-tts-realtime"))
    # 1.7B bf16 weights + codec + paged audio-token KV ≈ 6-8 GB budget
    tts_mossrt_gpu_mem_util: float = field(default_factory=lambda: _env_float("MOSSRT_GPU_MEM_UTIL", 0.25))
    tts_mossrt_extra_args: str = field(default_factory=lambda: _env("MOSSRT_VLLM_EXTRA_ARGS", ""))
    # native fallback sidecar (TTS_PROVIDER=moss_tts_realtime_native): the
    # UPSTREAM fast_api.py session server run from the vendored repo dir,
    # transformers 5.0 stack in its own venv (scripts/build_venv_mossrt.sh)
    tts_mossrt_python: str = field(default_factory=lambda: _env(
        "MOSSRT_PYTHON", os.path.join(REPO_ROOT, ".venv-mossrt", "bin", "python")))
    tts_mossrt_sidecar_dir: str = field(default_factory=lambda: _env(
        "MOSSRT_SIDECAR_DIR", os.path.join(
            REPO_ROOT, "server", "adapters", "tts", "moss_tts_realtime", "sidecar",
            "third_party", "MOSS-TTS", "moss_tts_realtime")))
    # builtin voice prompts (name → wav) fed to vLLM-Omni as per-request
    # ref_audio — VENDORED with the sidecar runtime (both providers read them)
    tts_voice_prompt_dir: str = field(default_factory=lambda: _env(
        "TTS_VOICE_PROMPT_DIR",
        os.path.join(REPO_ROOT, "server", "adapters", "tts", "moss_tts_nano", "sidecar",
                     "third_party", "MOSS-TTS-Nano", "assets", "audio")))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "Yuewen"))
    tts_sample_rate: int = field(default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 48000))
    tts_channels: int = field(default_factory=lambda: _env_int("TTS_CHANNELS", 2))
    # 12/75/150 mirrors the board's tuned values (asr-tts_research.md §1A T4): a
    # 12-char min cut starts the first synthesis ~2-4x sooner than 32.
    tts_seg_min_chars: int = field(default_factory=lambda: _env_int("TTS_SEGMENT_MIN_CHARS", 12))
    tts_seg_soft_chars: int = field(default_factory=lambda: _env_int("TTS_SEGMENT_SOFT_CHARS", 75))
    tts_seg_max_chars: int = field(default_factory=lambda: _env_int("TTS_SEGMENT_MAX_CHARS", 150))

    # -- ElevenLabs cloud TTS (external API — no GPU, no sidecar). Coexists
    # with the local pool: when an API key is present the engine is built at
    # boot alongside the local provider and sessions pick it via config
    # tts_engine=elevenlabs (UI: TTS 合成引擎 select). TTS_PROVIDER=elevenlabs
    # makes it the sole provider instead.
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY", ""))
    elevenlabs_base_url: str = field(default_factory=lambda: _env(
        "ELEVENLABS_BASE_URL", "https://api.elevenlabs.io"))
    # eleven_flash_v2_5: lowest-latency model (~75ms TTFB in-region), 32 langs
    # incl. zh; eleven_multilingual_v2 trades latency for quality
    elevenlabs_model_id: str = field(default_factory=lambda: _env(
        "ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"))
    # empty → the account's first voice (from GET /v1/voices at start())
    elevenlabs_voice_id: str = field(default_factory=lambda: _env("ELEVENLABS_VOICE_ID", ""))
    # pcm_24000 is the free-tier cap and needs no client-side resampling
    # (the session plane relays the engine's sample_rate to the player)
    elevenlabs_output_format: str = field(default_factory=lambda: _env(
        "ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"))
    # off by default: a warmup synth spends API credits on every boot
    elevenlabs_warmup: bool = field(default_factory=lambda: _env_flag("ELEVENLABS_WARMUP", False))
    # optional egress proxy for the API ("socks5://127.0.0.1:1080" — this box's
    # ssh -D tunnel is the only route to elevenlabs.io; empty = direct)
    elevenlabs_proxy: str = field(default_factory=lambda: _env("ELEVENLABS_PROXY", ""))
    # voice_settings.stability pinned per request: left unset it floats with the
    # voice's stored defaults, which reads as timbre drift across segments
    elevenlabs_stability: float = field(default_factory=lambda: _env_float("ELEVENLABS_STABILITY", 0.65))

    # -- MiniMax cloud TTS (T2A v2, speech-02 family — Chinese-native quality,
    # mainland endpoint is directly reachable: no tunnel, no TLS quirks). Same
    # coexist shape as elevenlabs: built at boot when MINIMAX_API_KEY is set,
    # sessions pick it via config tts_engine=minimax.
    minimax_api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY", ""))
    # api.minimaxi.com = the DOCUMENTED host (platform.minimaxi.com/docs);
    # api.minimaxi.chat / api.minimax.io are region mirrors that answer 2049
    # to valid keys. Direct egress is blocked on this box → MINIMAX_PROXY
    minimax_base_url: str = field(default_factory=lambda: _env(
        "MINIMAX_BASE_URL", "https://api.minimaxi.com"))
    # some keys are issued per-group; empty = let the key imply it
    minimax_group_id: str = field(default_factory=lambda: _env("MINIMAX_GROUP_ID", ""))
    # speech-02-hd = quality; speech-02-turbo = latency
    minimax_model: str = field(default_factory=lambda: _env("MINIMAX_MODEL", "speech-02-hd"))
    minimax_voice_id: str = field(default_factory=lambda: _env("MINIMAX_VOICE_ID", "male-qn-qingse"))
    minimax_sample_rate: int = field(default_factory=lambda: _env_int("MINIMAX_SAMPLE_RATE", 24000))
    minimax_proxy: str = field(default_factory=lambda: _env("MINIMAX_PROXY", ""))
    minimax_warmup: bool = field(default_factory=lambda: _env_flag("MINIMAX_WARMUP", False))

    # ---- voice capture / VAD ----
    capture_mode: str = field(default_factory=lambda: _env("CAPTURE_MODE", "ptt"))  # ptt | auto
    vad_rms_threshold: int = field(default_factory=lambda: _env_int("VOICE_ASR_AUTO_RMS_THRESHOLD", 350))
    vad_min_speech_ms: float = field(default_factory=lambda: _env_float("VOICE_ASR_AUTO_MIN_SPEECH_MS", 160))
    vad_silence_ms: float = field(default_factory=lambda: _env_float("VOICE_ASR_AUTO_SILENCE_MS", 500))
    # minimum captured speech for a PTT turn to be transcribed (reject accidental taps)
    ptt_min_speech_ms: float = field(default_factory=lambda: _env_float("PTT_MIN_SPEECH_MS", 120))

    # ---- session layer (backend_overhaul.md §3) ----
    # reconnect grace: how long a detached (or never-attached) session survives
    session_grace_seconds: float = field(default_factory=lambda: _env_float("SESSION_GRACE_SECONDS", 45.0))
    # server→client event ring buffer for ?last_seq replay
    session_replay_buffer: int = field(default_factory=lambda: _env_int("SESSION_REPLAY_BUFFER", 1024))
    # live out-queue bound (drop-oldest beyond this while detached)
    session_out_queue: int = field(default_factory=lambda: _env_int("SESSION_OUT_QUEUE", 4096))
    status_interval_s: float = field(default_factory=lambda: _env_float("STATUS_INTERVAL_S", 1.0))
    # attach a stored camera frame to a user turn only if it is at most this old
    frame_max_age_s: float = field(default_factory=lambda: _env_float("FRAME_MAX_AGE_S", 5.0))

    # ---- TTS back-pressure / drop-stale policy (asr-tts_research.md §1C) ----
    # pause feeding TTS units while estimated unplayed client audio exceeds high;
    # resume once it drains below low
    audio_buffer_high_s: float = field(default_factory=lambda: _env_float("AUDIO_BUFFER_HIGH_S", 2.5))
    audio_buffer_low_s: float = field(default_factory=lambda: _env_float("AUDIO_BUFFER_LOW_S", 1.0))
    # master switch for the whole drop-stale policy: 0 keeps EVERY unit (speech
    # never skips sentences, at the cost of narrating further behind if the
    # engine can't keep up — safe once TTS is fast, e.g. moss_tts_realtime).
    tts_drop_stale: bool = field(default_factory=lambda: _env_flag("TTS_DROP_STALE", True))
    # drop the oldest un-synthesized unit beyond this backlog, and any unit older
    # than max_age at feed time (0 disables either check). Relaxed 6 -> 24 so a
    # normal turn's sentences are not discarded as "backlog" when TTS lags.
    tts_max_pending_units: int = field(default_factory=lambda: _env_int("TTS_MAX_PENDING_UNITS", 24))
    # backlog coalescing (tts_serving_plan.md Stage 0): consecutive queued units
    # of one response merge into a single engine job up to this many chars, so a
    # backlog rides the engine's internal chunk batching instead of N serial
    # decodes (measured ~3x on the pytorch sidecar). 0 disables.
    tts_coalesce_max_chars: int = field(default_factory=lambda: _env_int("TTS_COALESCE_MAX_CHARS", 600))
    # relaxed 8s -> 30s: only expire units that are genuinely stale (a full turn
    # older), not merely queued behind a slow engine
    tts_unit_max_age_s: float = field(default_factory=lambda: _env_float("TTS_UNIT_MAX_AGE_S", 30.0))
    # cut the FIRST unit of a response at the first clause boundary once this many
    # chars accumulate (asr-tts_research.md §1A T4; 0 disables). Set >= the TTS
    # model's text->audio delay so the first push actually emits audio instead of
    # silence: MOSS-TTS-Realtime prefills at delay_tokens_len=12 TOKENS, so a
    # 12-CHAR clause can sit just under the gate (no audio until the 2nd clause).
    # 16 clears 12 tokens for the primary zh path at ~+0.2 s of accumulation;
    # English still relies on cross-clause accumulation to reach the gate.
    tts_first_clause_chars: int = field(default_factory=lambda: _env_int("TTS_FIRST_CLAUSE_CHARS", 16))

    # ---- persistence: history + media (durable archives, no auto-eviction) ----
    # history = append-only JSONL journal (source of truth) + SQLite index
    # (derived, rebuildable via scripts/history_prune.py --rebuild); media = CAS
    # blob store keyed by content hash. Growth is bounded by the manual prune
    # tool only — both stores are archives, not caches.
    history_enabled: bool = field(default_factory=lambda: _env_flag("HISTORY_ENABLED", True))
    media_enabled: bool = field(default_factory=lambda: _env_flag("MEDIA_ENABLED", True))
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", os.path.join(REPO_ROOT, "data")))
    history_db_path: str = field(default_factory=lambda: _env("HISTORY_DB_PATH", ""))  # "" → {data_dir}/index.db
    history_queue_bytes: int = field(default_factory=lambda: _env_int("HISTORY_QUEUE_BYTES", 16 * 1024**2))
    history_queue_items: int = field(default_factory=lambda: _env_int("HISTORY_QUEUE_ITEMS", 2048))
    history_pending_sessions: int = field(default_factory=lambda: _env_int("HISTORY_PENDING_SESSIONS", 128))
    upload_max_bytes: int = field(default_factory=lambda: _env_int("UPLOAD_MAX_BYTES", 512 * 1024 * 1024))
    media_hash_algo: str = field(default_factory=lambda: _env("MEDIA_HASH_ALGO", "sha256"))
    media_thumb_max_edge: int = field(default_factory=lambda: _env_int("MEDIA_THUMB_MAX_EDGE", 512))
    media_mime_allow: str = field(default_factory=lambda: _env(
        "MEDIA_MIME_ALLOW",
        "image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime"))
    # persist the camera frame attached to each realtime user turn (decision: off)
    history_keep_frames: bool = field(default_factory=lambda: _env_flag("HISTORY_KEEP_FRAMES", False))

    # ---- memory: L2 multimodal vector recall ----
    # Master flag. Off = the memory package is never constructed and the
    # orchestrator behaves exactly as before (no writes, no recall, no tokens).
    memory_enabled: bool = field(default_factory=lambda: _env_flag("MEMORY_ENABLED", False))
    # own DB, NOT index.db: memory rows carry embeddings and generated captions,
    # so they are not reconstructible by `history_prune.py --rebuild` and must
    # not be collateral damage of one. "" → {data_dir}/memory.db
    memory_db_path: str = field(default_factory=lambda: _env("MEMORY_DB_PATH", ""))
    memory_maintenance_interval_s: float = field(default_factory=lambda: _env_float("MEMORY_MAINTENANCE_INTERVAL_S", 30.0))
    memory_min_free_bytes: int = field(default_factory=lambda: _env_int("MEMORY_MIN_FREE_BYTES", 1024 ** 3))
    # Soft admission bound on occupied SQLite pages, not historical file size.
    memory_max_db_bytes: int = field(default_factory=lambda: _env_int("MEMORY_MAX_DB_BYTES", 2 * 1024 ** 3))
    memory_temporal_scan_items: int = field(default_factory=lambda: _env_int("MEMORY_TEMPORAL_SCAN_ITEMS", 128))
    memory_temporal_scan_bytes: int = field(default_factory=lambda: _env_int("MEMORY_TEMPORAL_SCAN_BYTES", 128 * 1024))
    memory_session_max_items: int = field(default_factory=lambda: _env_int("MEMORY_SESSION_MAX_ITEMS", 5000))
    memory_total_max_items: int = field(default_factory=lambda: _env_int("MEMORY_TOTAL_MAX_ITEMS", 20000))
    memory_session_max_bytes: int = field(default_factory=lambda: _env_int("MEMORY_SESSION_MAX_BYTES", 64 * 1024**2))
    memory_total_max_bytes: int = field(default_factory=lambda: _env_int("MEMORY_TOTAL_MAX_BYTES", 256 * 1024**2))
    memory_cache_max_bytes: int = field(default_factory=lambda: _env_int("MEMORY_CACHE_MAX_BYTES", 128 * 1024**2))
    memory_queue_max_bytes: int = field(default_factory=lambda: _env_int("MEMORY_QUEUE_MAX_BYTES", 32 * 1024**2))
    memory_item_max_chars: int = field(default_factory=lambda: _env_int("MEMORY_ITEM_MAX_CHARS", 8192))
    memory_session_frame_bytes: int = field(default_factory=lambda: _env_int("MEMORY_SESSION_FRAME_BYTES", 128 * 1024**2))
    memory_total_frame_bytes: int = field(default_factory=lambda: _env_int("MEMORY_TOTAL_FRAME_BYTES", 512 * 1024**2))
    memory_max_wal_bytes: int = field(default_factory=lambda: _env_int("MEMORY_MAX_WAL_BYTES", 128 * 1024**2))
    memory_queue_session_bytes: int = field(default_factory=lambda: _env_int("MEMORY_QUEUE_SESSION_BYTES", 8 * 1024**2))
    memory_frame_max_bytes: int = field(default_factory=lambda: _env_int("MEMORY_FRAME_MAX_BYTES", 2 * 1024**2))
    memory_background_tasks: int = field(default_factory=lambda: _env_int("MEMORY_BACKGROUND_TASKS", 8))
    memory_uncommitted_bytes: int = field(default_factory=lambda: _env_int("MEMORY_UNCOMMITTED_BYTES", 64 * 1024))
    memory_tracked_sessions: int = field(default_factory=lambda: _env_int("MEMORY_TRACKED_SESSIONS", 128))
    # Text space: BGE-M3 (multilingual, so an English query still hits a Chinese
    # memory). Image space: Chinese-CLIP — ONE space for frames and zh text, so
    # "我刚才给你看的那个" resolves against frames directly, no captioner needed.
    # Both default into the models/ zoo (symlinks, like asr/tts/vlms); a missing
    # path degrades to the deterministic fallbacks instead of failing the boot.
    memory_embed_text_model: str = field(default_factory=lambda: _env(
        "MEMORY_EMBED_TEXT_MODEL", os.path.join(_memory_dir(), "bge-m3")))
    memory_embed_image_model: str = field(default_factory=lambda: _env(
        "MEMORY_EMBED_IMAGE_MODEL", os.path.join(_memory_dir(), "chinese-clip")))
    memory_embed_device: str = field(default_factory=lambda: _env("MEMORY_EMBED_DEVICE", "cpu"))
    memory_text_dim: int = field(default_factory=lambda: _env_int("MEMORY_TEXT_DIM", 512))
    # keyframe capture: hot-path throttle, forced keep (a static scene must still
    # leave a trace), and the descriptor-cosine skip threshold
    memory_keyframe_min_interval_s: float = field(
        default_factory=lambda: _env_float("MEMORY_KEYFRAME_MIN_INTERVAL_S", 1.0))
    memory_keyframe_force_s: float = field(
        default_factory=lambda: _env_float("MEMORY_KEYFRAME_FORCE_S", 8.0))
    memory_keyframe_sim_threshold: float = field(
        default_factory=lambda: _env_float("MEMORY_KEYFRAME_SIM_THRESHOLD", 0.92))
    # speculative retrieval on ASR partials → the turn itself pays ~nothing
    memory_prefetch_on_partials: bool = field(
        default_factory=lambda: _env_flag("MEMORY_PREFETCH_ON_PARTIALS", True))
    memory_retrieval_topk: int = field(default_factory=lambda: _env_int("MEMORY_RETRIEVAL_TOPK", 4))
    # fusion weight of the text space; the image space gets (1 - this), and only
    # participates at all when the image embedder is cross-modal
    memory_fusion_text_weight: float = field(
        default_factory=lambda: _env_float("MEMORY_FUSION_TEXT_WEIGHT", 0.6))
    memory_recency_halflife_h: float = field(
        default_factory=lambda: _env_float("MEMORY_RECENCY_HALFLIFE_H", 24.0))
    # Admission threshold, as an ABSOLUTE cosine in the embedder's own space —
    # not the fused/normalized rank score, which maps the best of a bad hit set
    # to 1.0 and would admit something every turn. Skip-biased on purpose: an
    # injected token can never leave the KV cache, and a related-but-wrong
    # memory hurts more than no memory. A turn with no explicit past-reference
    # marker needs +0.15 (and +0.10 more if it is not even a question).
    # 0 (default) = use the embedder's own declared floor, because the cosine
    # scale is embedder-specific (n-gram fallback ~0.22, trained encoder ~0.45);
    # set a value to pin it explicitly after calibrating on the probe suite.
    memory_gate_min_score: float = field(
        default_factory=lambda: _env_float("MEMORY_GATE_MIN_SCORE", 0.0))
    memory_inject_max_items: int = field(default_factory=lambda: _env_int("MEMORY_INJECT_MAX_ITEMS", 3))
    memory_inject_max_tokens: int = field(
        default_factory=lambda: _env_int("MEMORY_INJECT_MAX_TOKENS", 180))
    # lifetime cap on injected recall tokens for one session (~5% of the text KV).
    # Exhaustion stops recalling SILENTLY — it is a rollover signal, not a reason
    # to raise the cap (design §5).
    memory_inject_session_max_tokens: int = field(
        default_factory=lambda: _env_int("MEMORY_INJECT_SESSION_MAX_TOKENS", 1200))
    # re-pushing a recalled frame through the frame queue (channel V): off until
    # the caption-vs-frame A/B says it earns its ~250 vision tokens
    memory_inject_frames: bool = field(default_factory=lambda: _env_flag("MEMORY_INJECT_FRAMES", False))
    # Re-injection: an already-shown item may be injected again only after this
    # many text-KV tokens have passed (below that it is still in the effective
    # window), and at most MAX_COPIES times per session. Retrieval ranks
    # injected items WITH the rest and applies this gate afterwards — excluding
    # them up front makes an explicit re-request silently unanswerable.
    memory_reinject_distance: int = field(
        default_factory=lambda: _env_int("MEMORY_REINJECT_DISTANCE", 2000))
    memory_reinject_max_copies: int = field(
        default_factory=lambda: _env_int("MEMORY_REINJECT_MAX_COPIES", 3))
    # per-space z-normalization background sample size (raw CLIP and BGE cosines
    # are not comparable; each space is normalized against its own score
    # distribution before fusion)
    memory_znorm_samples: int = field(default_factory=lambda: _env_int("MEMORY_ZNORM_SAMPLES", 1000))
    # BGE-M3 late interaction: score text↔text by max-sim over token vectors
    # (colbert head) instead of one pooled vector. Needs the checkpoint's
    # colbert_linear weights; the hashing fallback never supports it.
    memory_late_interaction: bool = field(
        default_factory=lambda: _env_flag("MEMORY_LATE_INTERACTION", True))
    memory_late_pool: str = field(default_factory=lambda: _env("MEMORY_LATE_POOL", "maxsim"))
    # no reranker by design (§1); the key exists so the surface matches the doc
    memory_reranker: str = field(default_factory=lambda: _env("MEMORY_RERANKER", "none"))
    # rollover (compaction): trigger on TEXT tokens, not KV ratio — ~80% of text
    # growth is timestamp wrappers for frames whose vision KV is already evicted.
    # Idle fires only at a <|silence|> idle moment; hard fires regardless.
    memory_rollover_idle_tokens: int = field(
        default_factory=lambda: _env_int("MEMORY_ROLLOVER_IDLE_TOKENS", 8000))
    memory_rollover_hard_tokens: int = field(
        default_factory=lambda: _env_int("MEMORY_ROLLOVER_HARD_TOKENS", 12000))
    memory_rollover_min_progress: float = field(
        default_factory=lambda: _env_float("MEMORY_ROLLOVER_MIN_PROGRESS", 0.10))
    memory_rollover_tail_turns: int = field(
        default_factory=lambda: _env_int("MEMORY_ROLLOVER_TAIL_TURNS", 6))
    # summary + fact extraction run on the offline sglang plane ("offline"), on
    # the pi_agent sidecar ("pi": journal -> POST {MEMORY_PI_URL}/compact ->
    # {summary, pins}); "none" degrades rollover to verbatim-tail-only and facts
    # to off (1-GPU boxes have no offline plane)
    memory_summary_provider: str = field(
        default_factory=lambda: _env("MEMORY_SUMMARY_PROVIDER", "offline"))
    memory_summary_max_tokens: int = field(
        default_factory=lambda: _env_int("MEMORY_SUMMARY_MAX_TOKENS", 200))
    # pi_agent memory sidecar (board parity): an HTTP service that LLM-gates
    # retrieval (/decide) and compacts the rollover journal (/compact). An
    # unreachable or failing pi degrades every dependent path to the
    # local-vector / verbatim-tail behavior — never a broken turn.
    memory_pi_url: str = field(default_factory=lambda: _env("MEMORY_PI_URL", _default_memory_pi_url()))
    memory_pi_decide_timeout_s: float = field(
        default_factory=lambda: _env_float("MEMORY_PI_DECIDE_TIMEOUT_S", 8.0))
    memory_pi_compact_timeout_s: float = field(
        default_factory=lambda: _env_float("MEMORY_PI_COMPACT_TIMEOUT_S", 120.0))
    # retrieval decision gate ahead of MemorySession.recall_for_turn:
    #   vector — local score gates only (pre-pi behavior);
    #   llm    — pi /decide every turn (recent_turns + the current user text);
    #   hybrid — explicit history references use pi /decide + query rewriting
    #            first; other turns retain the local raw-score prefilter.
    memory_decision_mode: str = field(default_factory=lambda: _env("MEMORY_DECISION_MODE", "hybrid"))
    # hybrid stage-1: the best ABSOLUTE raw cosine/maxsim must clear this before
    # pi /decide is consulted (board's 0.75; retro questions run 0.10 looser)
    memory_retrieval_prefilter_score: float = field(
        default_factory=lambda: _env_float("MEMORY_RETRIEVAL_PREFILTER_SCORE", 0.75))
    # rollover compact prefetch: once text tokens cross idle_tokens * ratio, a
    # background thread pre-runs the pi /compact so build_prefix usually finds
    # it ready (in-flight join capped at 30s; pi provider only)
    memory_rollover_prefetch_ratio: float = field(
        default_factory=lambda: _env_float("MEMORY_ROLLOVER_PREFETCH_RATIO", 0.6))
    # facts are extracted from user turns only (assistant turns may interpret an
    # elliptical user turn as context, never become facts) and go into the index
    # key, never to the model
    memory_fact_scope: str = field(default_factory=lambda: _env("MEMORY_FACT_SCOPE", "user"))
    memory_bg_concurrency: int = field(default_factory=lambda: _env_int("MEMORY_BG_CONCURRENCY", 1))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    _load_env_deploy()  # layer 3 — setdefault only, so set env (layers 1-2) wins
    if _settings is None:
        _settings = Settings()
    return _settings
