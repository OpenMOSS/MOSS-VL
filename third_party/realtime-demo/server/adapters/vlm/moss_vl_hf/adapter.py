"""HF MOSS-VL adapter — online streaming + offline chat (dual-mode fallback).

This in-process HF path backs the ONLINE worker fleet and doubles as the
offline-chat fallback when the dedicated sglang plane is unavailable; the
primary offline backend is adapters/vlm/moss_vl_sglang/adapter.py.

Faithful port of the streaming subset of board inference.py:ModelManager (drops
the GPU-occupy / batch / sglang / dp-worker / convert machinery). Loads the
streaming MOSS-VL class CPU-first in bf16 then moves to the GPU, installs the two
realtime patches, and runs `model.real_time_generate` in a daemon thread over the
per-session queues. Offline chat streams a standard HF generate via
TextIteratorStreamer.
"""
from __future__ import annotations

import importlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import torch

from ....config import Settings
from ....logging_conf import get_logger
from ....device_compat import device_str
from .serving_policy import (
    auto_attn_impl,
    clamp_attn_override,
    launch_decode_worker,
    offline_chat_attn_impl,
    prepare_vision_prefill,
    realtime_attn_impl,
    run_processor,
)
from ....realtime.mossvl_patches import (
    DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT,
    decode_prefill_messages,
    install_realtime_frame_window_patch,
    install_realtime_initial_prompt_patch,
    install_realtime_token_counter_patch,
    install_realtime_turn_interrupt_patch,
)
from ....realtime.session import (
    RealTimeFrameQueue,
    RealTimeOutputQueue,
    RealtimeSession,
)
from ...base import VlmCaps

log = get_logger(__name__)

HF_MODE_OFFLINE = "offline"
HF_MODE_ONLINE_STREAMING = "online_streaming"


# --------------------------- model resolution (ported) ---------------------------


def read_model_config(model_path: str) -> dict:
    try:
        with open(os.path.join(model_path, "config.json"), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def is_mossvl_model(model_config: dict) -> bool:
    return model_config.get("model_type") == "moss_vl"


def resolve_streaming_mossvl_classes(streaming_processor_path: str):
    processor_path = os.path.abspath(streaming_processor_path)
    if not os.path.isdir(processor_path):
        raise RuntimeError(f"Streaming MOSS-VL processor path not found: {processor_path}")
    # Import the checkpoint's remote code as a package derived from the processor
    # dir ITSELF (its parent on sys.path, its own name as the package), so it can
    # live anywhere — e.g. models/vlms/hf_mossvl_streaming_processor — not only
    # under a fixed .../moss/ tree. The dir is a namespace package (no
    # __init__.py); modeling_moss_vl's `from .configuration_moss_vl import`
    # resolves relative to it. (Back-compat: the old .../moss/… path still works
    # — it just imports as `hf_mossvl_streaming_processor` instead of `moss.…`.)
    parent = os.path.dirname(processor_path)
    pkg = os.path.basename(processor_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    modeling_module = importlib.import_module(f"{pkg}.modeling_moss_vl")
    config_module = importlib.import_module(f"{pkg}.configuration_moss_vl")
    return modeling_module.MossVLForConditionalGeneration, config_module.MossVLConfig


class RealTimeProcessorProxy:
    """Inject per-session pixel/fps overrides into processor calls (ported)."""

    def __init__(self, processor: Any, *, min_pixels=None, max_pixels=None, video_fps=None,
                 min_frames=None, max_frames=None, multi_image_max_pixels=None, video_max_pixels=None):
        self._processor = processor
        self._call_overrides = {
            "min_pixels": min_pixels, "max_pixels": max_pixels, "video_fps": video_fps,
            "min_frames": min_frames, "max_frames": max_frames,
        }
        self._multi_image_max_pixels = multi_image_max_pixels
        self._video_max_pixels = video_max_pixels

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for key, value in self._call_overrides.items():
            if value is not None and key not in kwargs:
                kwargs[key] = value
        image_proc = getattr(self._processor, "image_processor", None)
        video_proc = getattr(self._processor, "video_processor", None)
        orig_img = orig_vid = None
        try:
            if self._multi_image_max_pixels is not None and image_proc is not None:
                orig_img = getattr(image_proc, "multi_image_max_pixels", None)
                image_proc.multi_image_max_pixels = self._multi_image_max_pixels
            if self._video_max_pixels is not None and video_proc is not None:
                orig_vid = getattr(video_proc, "video_max_pixels", None)
                video_proc.video_max_pixels = self._video_max_pixels
            return self._processor(*args, **kwargs)
        finally:
            if image_proc is not None and orig_img is not None:
                image_proc.multi_image_max_pixels = orig_img
            if video_proc is not None and orig_vid is not None:
                video_proc.video_max_pixels = orig_vid


# --------------------------- adapter ---------------------------


class HfMossVlAdapter:
    def __init__(self, settings: Settings):
        self.s = settings
        self.caps = VlmCaps(modes=(HF_MODE_ONLINE_STREAMING, HF_MODE_OFFLINE))
        self.model: Any = None
        self.processor: Any = None
        self.model_path: str = ""
        self.gpu_id: int = settings.gpu_id
        self.hf_mode: str = ""
        self.model_config: dict = {}
        self._attn_impl: Optional[str] = None    # resolved at load()
        self._infer_lock = threading.Lock()      # one active realtime session per model
        self._sessions: Dict[str, RealtimeSession] = {}
        self._sessions_lock = threading.Lock()

    # ---- load ----

    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return device_str(self.gpu_id)

    def _resolve_attn_impl(self, gpu_id: int, override: Optional[str]) -> str:
        """Explicit override > explicit setting > auto by backend."""

        if override and override.strip().lower() != "auto":
            return clamp_attn_override(override)
        return auto_attn_impl(gpu_id, self.s.attn_impl)

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        hf_mode = hf_mode if hf_mode in (HF_MODE_OFFLINE, HF_MODE_ONLINE_STREAMING) else HF_MODE_ONLINE_STREAMING
        self.gpu_id = gpu_id
        self._attn_impl = self._resolve_attn_impl(gpu_id, attn_impl_override)
        device = device_str(gpu_id)
        model_config = read_model_config(model_path)
        if not is_mossvl_model(model_config):
            raise RuntimeError(f"Expected a MOSS-VL checkpoint (model_type=moss_vl) at {model_path}")

        log.info("Loading MOSS-VL (%s) from %s to %s", hf_mode, model_path, device)
        if hf_mode == HF_MODE_ONLINE_STREAMING:
            from transformers import AutoModelForCausalLM, AutoConfig
            streaming_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            ovr = self.s.vision_seq_pad_multiple_override
            if ovr is not None and getattr(streaming_config, "vision_seq_pad_multiple", None) != ovr:
                log.warning("Overriding vision_seq_pad_multiple %s -> %s (validated realtime path is 1)",
                            getattr(streaming_config, "vision_seq_pad_multiple", None), ovr)
                streaming_config.vision_seq_pad_multiple = ovr
            processor_path = self.s.mossvl_streaming_processor_path
            model = self._from_pretrained(AutoModelForCausalLM, model_path, config=streaming_config)
        else:
            from transformers import AutoModelForCausalLM
            processor_path = model_path  # offline processor bundled with the ckpt (or canonical)
            model = self._from_pretrained(AutoModelForCausalLM, model_path)

        log.info("Model loaded to CPU, moving to %s", device)
        model = model.to(device)
        model.eval()

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

        self.model = model
        self.processor = processor
        self.model_path = model_path
        self.hf_mode = hf_mode
        self.model_config = model_config
        log.info("MOSS-VL ready on %s (mode=%s)", device, hf_mode)

    def _from_pretrained(self, ModelClass, model_path: str, **extra):
        """CPU-first bf16 load with flash-attn2 -> sdpa fallback.

        This only catches LOAD-time failures (missing package, unsupported
        impl); a wrong-arch flash-attn build (e.g. sm_90-only cubins on a
        4090) loads fine and dies at the first forward — the worker's warmup
        forward (server/vlm_worker/app.py) covers that gap.
        """
        base = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                    device_map=None, low_cpu_mem_usage=False, **extra)
        primary = getattr(self, "_attn_impl", None) or self._resolve_attn_impl(self.gpu_id, None)
        attempts = [primary]
        if self.s.attn_impl_fallback and self.s.attn_impl_fallback != primary:
            attempts.append(self.s.attn_impl_fallback)
        for attn in attempts:
            try:
                model = ModelClass.from_pretrained(model_path, attn_implementation=attn, **base)
                self._attn_impl = attn
                return model
            except (ImportError, ValueError, RuntimeError) as exc:
                if attn == attempts[-1]:
                    raise
                log.warning("attn_implementation=%s failed (%s); retrying with %s",
                            attn, exc, attempts[-1])
        raise RuntimeError("unreachable")

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded(),
            "model_path": self.model_path,
            "gpu_id": self.gpu_id,
            "hf_mode": self.hf_mode,
            "attn_impl": self._attn_impl,
            "active_sessions": len(self._sessions),
            "modes": list(self.caps.modes),
        }

    # ---- realtime ----

    def _ensure_realtime_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("No model loaded")
        if self.hf_mode != HF_MODE_ONLINE_STREAMING:
            raise RuntimeError("Realtime requires hf_mode=online_streaming; reload the model")
        if not callable(getattr(self.model, "real_time_generate", None)):
            raise RuntimeError("Loaded MOSS-VL class does not expose real_time_generate")
        # eager on NPU (SDPA decode hang); CUDA keeps its load-time impl — serving_policy
        for cfg in self._attn_configs():
            cfg._attn_implementation = realtime_attn_impl(cfg._attn_implementation)

    def start_realtime_session(
        self,
        *,
        prompt: str = "",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        max_tokens_per_turn: Optional[int] = None,
        frame_queue_size: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_fps: Optional[float] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        multi_image_max_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        # rollover re-seat (design §6): a JSON string of [{role, content}] that
        # REPLACES the legacy system+user prefill pair. A str (not a list) so it
        # survives the worker proxy's scalar-only JSON seam unchanged.
        prefill_messages: Optional[str] = None,
    ) -> RealtimeSession:
        self._ensure_realtime_ready()
        if not self._infer_lock.acquire(blocking=False):
            raise RuntimeError("A realtime session is already running on this model")

        try:
            s = self.s
            proxy = RealTimeProcessorProxy(
                self.processor,
                min_pixels=min_pixels if min_pixels is not None else s.min_pixels,
                max_pixels=max_pixels if max_pixels is not None else s.max_pixels,
                video_fps=video_fps if video_fps is not None else s.video_fps,
                min_frames=min_frames if min_frames is not None else s.min_frames,
                max_frames=max_frames if max_frames is not None else s.max_frames,
                multi_image_max_pixels=multi_image_max_pixels if multi_image_max_pixels is not None else s.multi_image_max_pixels,
                video_max_pixels=video_max_pixels if video_max_pixels is not None else s.video_max_pixels,
            )
            install_realtime_initial_prompt_patch(self.model)
            install_realtime_turn_interrupt_patch(self.model)
            # Frame window: keep only the last N minutes of frames (evict old vision KV,
            # keep the words). Budget = minutes*60*fps; fps falls back to 2 (design 1-2 fps).
            # Reset the cumulative evicted counter every session (state lives on the reused
            # model instance). Installed AFTER turn_interrupt so eviction runs on the final
            # per-step state; keep_frames<=0 disables (and disables a prior session's window).
            eff_fps = (video_fps if video_fps is not None else s.video_fps) or 2.0
            keep_frames = round(max(0.0, s.realtime_frame_window_minutes) * 60.0 * float(eff_fps))
            self.model._frame_window_evicted_total = 0
            install_realtime_frame_window_patch(self.model, keep_frames)
            # Text-token counter (rollover trigger input): reset per session and
            # installed LAST so it wraps the frame-window patch and reads the
            # final per-step input_ids length.
            self.model._rt_text_tokens = 0
            install_realtime_token_counter_patch(self.model)

            session_id = str(uuid.uuid4())
            frame_queue = RealTimeFrameQueue(maxsize=max(1, frame_queue_size or s.frame_queue_size))
            prompt_queue: "queue.Queue[str]" = queue.Queue()
            stop_event = threading.Event()
            created_at = time.time()
            output_queue = RealTimeOutputQueue(created_at=created_at)

            sys_prompt = DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT if system_prompt is None else str(system_prompt)
            setattr(self.model, "_board_realtime_initial_prompt", prompt or s.initial_prompt or "")
            setattr(self.model, "_board_realtime_system_prompt", sys_prompt)
            # rollover prefill (validated list or None → legacy pair). Cleared
            # again in the runner's finally so a later plain session on this
            # reused model never inherits a stale rebuilt prefix.
            setattr(self.model, "_board_realtime_prefill_messages",
                    decode_prefill_messages(prefill_messages))

            gen = dict(
                max_new_tokens=max_new_tokens if max_new_tokens is not None else s.max_new_tokens,
                temperature=temperature if temperature is not None else s.temperature,
                top_k=top_k if top_k is not None else s.top_k,
                top_p=top_p if top_p is not None else s.top_p,
                do_sample=do_sample if do_sample is not None else s.do_sample,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else s.repetition_penalty,
            )
            # Selective repetition penalty (rep_penalty.py): when active, the
            # stock full-context processor is neutralized and a control-token-
            # exempt windowed processor is injected as a caller-supplied
            # logits_processor (merged by the checkpoint's _get_logits_processor
            # — no vendored-code edit). No-op at penalty=1.0 (the default).
            from transformers import LogitsProcessorList  # local: load() imported transformers already

            from .rep_penalty import configure_generation
            gen = configure_generation(
                gen, self.processor.tokenizer,
                window=s.rep_penalty_window, exempt=s.rep_penalty_exempt,
                wrap=LogitsProcessorList,
            )
            mtpt = max_tokens_per_turn if max_tokens_per_turn is not None else s.max_tokens_per_turn

            session = RealtimeSession(
                session_id=session_id,
                gpu_id=self.gpu_id,
                frame_queue=frame_queue,
                prompt_queue=prompt_queue,
                output_queue=output_queue,
                stop_event=stop_event,
                created_at=created_at,
                model=self.model,
            )
            session._stopper = self._stop_session
            frame_queue.on_get = session.mark_consumed

            def _runner():
                try:
                    log.info("Realtime session %s entering model loop on %s", session_id, self.device)
                    self.model.real_time_generate(
                        new_video_frames=frame_queue,
                        new_prompts=prompt_queue,
                        output_text_queue=output_queue,
                        processor=proxy,
                        max_tokens_per_turn=mtpt,
                        **gen,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("Realtime session %s failed: %s", session_id, exc)
                    output_queue.put(f"[ERROR] {exc}")
                finally:
                    setattr(self.model, "_board_realtime_initial_prompt", "")
                    setattr(self.model, "_board_realtime_system_prompt", DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT)
                    setattr(self.model, "_board_realtime_prefill_messages", None)
                    try:
                        stop_fn = getattr(self.model, "stop_real_time_generate", None)
                        if callable(stop_fn):
                            stop_fn()
                    except Exception:  # noqa: BLE001
                        pass
                    stop_event.set()
                    try:
                        self._infer_lock.release()
                    except RuntimeError:
                        pass
                    log.info(
                        "Realtime session %s exited (frames=%d consumed=%d outputs=%d non_silence=%d)",
                        session_id, session.frames_received, session.frames_consumed,
                        session.outputs_emitted, session.non_silence_outputs,
                    )

            thread = threading.Thread(target=_runner, name=f"mossvl-rt-{session_id[:8]}", daemon=True)
            session.thread = thread
            with self._sessions_lock:
                self._sessions[session_id] = session
            thread.start()
            log.info("Started realtime session %s on %s", session_id, self.device)
            return session
        except Exception:
            try:
                self._infer_lock.release()
            except RuntimeError:
                pass
            raise

    def get_session(self, session_id: str) -> RealtimeSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Realtime session not found: {session_id}")
        return session

    def _stop_session(self, session_id: str, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise KeyError(f"Realtime session not found: {session_id}")
        session.stop_event.set()
        try:
            stop_fn = getattr(self.model, "stop_real_time_generate", None)
            if callable(stop_fn):
                stop_fn()
        except Exception:  # noqa: BLE001
            pass
        if session.thread is not None:
            session.thread.join(timeout=timeout_seconds)
        return session.status_payload()

    # ---- offline chat ----

    def _attn_configs(self) -> list:
        """Every config object whose `_attn_implementation` the attention modules read."""
        configs = {id(self.model.config): self.model.config}
        for module in self.model.modules():
            cfg = getattr(module, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                configs[id(cfg)] = cfg
        return list(configs.values())

    def _offline_eos_ids(self, tokenizer: Any) -> set:
        eos: set = set()
        gc_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        for value in (gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos]):
            if isinstance(value, int):
                eos.add(value)
        if tokenizer.eos_token_id is not None:
            eos.add(int(tokenizer.eos_token_id))
        for token in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                eos.add(tid)
        return eos

    @staticmethod
    def _sample_token(logits: "torch.Tensor", params: Any, generated: list) -> int:
        if params.repetition_penalty and params.repetition_penalty != 1.0 and generated:
            prev = torch.tensor(sorted(set(generated)), device=logits.device)
            scores = logits[0, prev]
            scores = torch.where(scores > 0, scores / params.repetition_penalty, scores * params.repetition_penalty)
            logits[0, prev] = scores
        if not params.do_sample:
            return int(logits.argmax(dim=-1).item())
        logits = logits / max(float(params.temperature or 1.0), 1e-5)
        if params.top_k and params.top_k > 0:
            kth = torch.topk(logits, min(int(params.top_k), logits.shape[-1]))[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if params.top_p and 0.0 < params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            cut = cum - torch.softmax(sorted_logits, dim=-1) > params.top_p
            sorted_logits[cut] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _offline_decode_loop(self, tokenizer: Any, input_ids: "torch.Tensor",
                             params: Any, emit, vision: Optional[dict] = None) -> None:
        """Hand-rolled decode for offline chat (runs in a worker thread).

        The streaming MOSS-VL class was never exercised with `generate()`:
        its decode path recomputes 3-D M-RoPE position_ids over the FULL
        attention mask each step, so `apply_rotary_pos_emb` broadcasts the
        1-token K against a full-length cos/sin and the cache double-appends K
        ("Expected size ... [32, 51] but got: [32, 26]"). We drive the forward
        directly with the same explicit-position contract the (working)
        realtime loop uses: text-only M-RoPE positions are arange stacked on
        all three planes, one position per step.

        With `vision` (processor outputs: pixel_values / grid_thw /
        cross_attention_mask / media_nums_per_sample), the PREFILL instead
        passes position_ids=None so the model's own vision-aware path runs
        (compute_position_ids + compute_vision_position_ids) and caches
        rope_deltas; decode steps then use position = cache_index + delta —
        the model's documented decode contract. Decode steps pass NO
        cross_attention_mask: mask-less means "attend all cached vision
        tokens", exactly what _update_model_kwargs_for_generation's
        copy-last-row extension produces for offline single-shot media.
        """
        device = self.device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]
        eos_ids = self._offline_eos_ids(tokenizer)
        max_new = max(1, min(int(params.max_new_tokens or 1024), 4096))

        attention_mask = torch.ones(1, prompt_len, dtype=torch.long, device=device)
        cache_position = torch.arange(prompt_len, dtype=torch.long, device=device)

        rope_delta = 0
        generated: list = []
        emitted = ""
        with torch.no_grad():
            if vision and "_prefill_outputs" in vision:
                # Prefill was already run in the main thread (NPU thread safety)
                outputs = vision["_prefill_outputs"]
                # Same rope_delta contract as the elif branch below — the
                # model's decode position is cache_index + delta (delta is
                # NON-zero for vision prompts; measured 20 on a test image)
                deltas = getattr(outputs, "rope_deltas", None)
                if deltas is None:  # fall back to the module-level cache
                    deltas = getattr(getattr(self.model, "model", None), "rope_deltas", None)
                rope_delta = int(deltas.reshape(-1)[0].item()) if deltas is not None else 0
            elif vision:
                vision_kwargs = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in vision.items()
                    if v is not None
                }
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=None,
                    cache_position=cache_position, use_cache=True, **vision_kwargs,
                )
                deltas = getattr(outputs, "rope_deltas", None)
                if deltas is None:  # fall back to the module-level cache
                    deltas = getattr(getattr(self.model, "model", None), "rope_deltas", None)
                rope_delta = int(deltas.reshape(-1)[0].item()) if deltas is not None else 0
            else:
                positions = (
                    torch.arange(prompt_len, dtype=torch.long, device=device)
                    .view(1, 1, prompt_len).expand(3, 1, prompt_len).contiguous()
                )
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                    cache_position=cache_position, use_cache=True,
                )
            past_key_values = outputs.past_key_values
            for step in range(max_new):
                token = self._sample_token(outputs.logits[:, -1, :].float(), params, generated)
                if token in eos_ids:
                    break
                generated.append(token)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if len(text) > len(emitted) and not text.endswith("�"):
                    emit(text[len(emitted):])
                    emitted = text
                cache_idx = prompt_len + step
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=1)
                outputs = self.model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                    attention_mask=attention_mask,
                    position_ids=torch.full((3, 1, 1), cache_idx + rope_delta, dtype=torch.long, device=device),
                    cache_position=torch.tensor([cache_idx], dtype=torch.long, device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

    @staticmethod
    def _decode_chat_image(payload: str):
        """One uploaded chat image → RGB PIL image.

        Accepts a data-URL, raw base64, or a CAS handle (`sha256:<hex>` / bare
        64-hex) minted by POST /api/media — the handle form keeps large images
        out of the JSON body. (A 64-char pure-hex string can't be a real base64
        image, so handle detection can safely run first.)
        """
        import base64
        from io import BytesIO

        from PIL import Image

        from ....persistence.media import maybe_get_media_store, normalize_hash

        s = (payload or "").strip()
        if not s.startswith("data:"):
            hex_ = normalize_hash(s)
            if hex_ is not None:
                store = maybe_get_media_store()
                if store is None:
                    raise RuntimeError("media store unavailable — cannot resolve media handle")
                return Image.open(BytesIO(store.load_bytes(hex_))).convert("RGB")
        if s.startswith("data:"):
            s = s.split(",", 1)[1] if "," in s else ""
        raw = base64.b64decode(s)
        return Image.open(BytesIO(raw)).convert("RGB")

    @staticmethod
    def _resolve_chat_video(payload: str) -> dict:
        """One uploaded chat video → `{"video_path": …}` for the processor.

        Videos are accepted ONLY as CAS handles (`sha256:<hex>` / bare 64-hex)
        minted by POST /api/media — never raw paths or inline base64, so a
        chat request can't point the decoder at arbitrary files. Resolution is
        pure path math (persistence.media.resolve_blob_path), so it works in
        VLM worker processes too; the video processor decodes the blob itself
        (torchcodec sniffs the container, the extension-less name is fine).
        """
        from ....persistence.media import normalize_hash, resolve_blob_path

        s = (payload or "").strip()
        if normalize_hash(s) is None:
            raise ValueError(
                "videos must be CAS media handles (sha256:<hex> from POST /api/media)")
        path = resolve_blob_path(s)
        if path is None:
            raise ValueError(f"unknown video media: {s[:19]}…")
        return {"video_path": path}

    @classmethod
    def _prepare_chat_messages(cls, req: Any) -> Tuple[list, list, list]:
        """Normalize a ChatRequest into (template_messages, images, videos).

        Multi-turn: any message's `content` may be a parts list mixing
        `{"type":"text","text":…}`, `{"type":"image","media"|"image":…}` and
        `{"type":"video","media"|"video":…}` — image payloads are CAS handles
        (`sha256:<hex>`) or data-URL/base64; video payloads are CAS handles
        only (see `_resolve_chat_video`). Media are collected in DOCUMENT
        order (messages top-down, parts left-to-right), matching the
        `<|image|>`/`<|video|>` placeholder order the chat template emits.
        Legacy top-level `req.images`/`req.videos` still attach to the LAST
        user message (prepended as parts) for backward compatibility.

        Blocking (PIL + disk for handles) — call via `asyncio.to_thread`.
        """
        messages = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in req.messages]

        extra_images = list(getattr(req, "images", None) or [])
        extra_videos = [v for v in (getattr(req, "videos", None) or []) if isinstance(v, str)]
        if extra_images or extra_videos:
            target = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if target is None:
                raise ValueError("media provided but no user message to attach it to")
            body = target.get("content")
            media_parts = [{"type": "image", "media": s} for s in extra_images] + \
                [{"type": "video", "media": s} for s in extra_videos]
            if isinstance(body, list):
                target["content"] = media_parts + list(body)
            else:
                target["content"] = media_parts + [
                    {"type": "text", "text": body if isinstance(body, str) else str(body or "")}]

        images: list = []
        videos: list = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                ptype = part.get("type")
                if ptype == "image":
                    payload = part.get("media") or part.get("image") or part.get("data") or ""
                    images.append(cls._decode_chat_image(str(payload)))
                    parts.append({"type": "image"})
                elif ptype == "video":
                    payload = part.get("media") or part.get("video") or ""
                    videos.append(cls._resolve_chat_video(str(payload)))
                    parts.append({"type": "video"})
                elif ptype == "text":
                    parts.append({"type": "text", "text": str(part.get("text") or "")})
                # unknown part types are dropped from the template
            m["content"] = parts
        return messages, images, videos

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        """Stream an offline (non-realtime) chat response.

        Runs under SDPA for the duration: with mask-less decode steps,
        transformers' flash wrapper misreads 3-D M-RoPE position_ids as a
        packed batch (`_is_packed_sequence`) and crashes in varlen flash.
        The modeling code reads `_attn_implementation` per forward, so the
        swap is safe even with a live realtime session (a step may mix
        kernels — same math either way).

        The chat template is applied over the FULL `req.messages` conversation
        (multi-turn); `_prepare_chat_messages` documents the media-part wire
        shapes. Each `{"type":"image"}` part becomes one `<|image|>`
        placeholder, and the processor swaps each for ONE `<|image_pad|>`
        token (cross-attention design — no token expansion). Each
        `{"type":"video"}` part becomes one `<|video|>` placeholder, which the
        processor expands to per-frame `<|time_start|>…<|time_end|><|image_pad|>`
        runs after sampling the file (fps/min/max_frames — request params
        override the processor defaults via RealTimeProcessorProxy). Either
        way it emits the same unified pixel_values / grid_thw /
        cross_attention_mask / media_nums_per_sample for the vision prefill.
        """
        import asyncio
        import queue as queue_mod

        if self.model is None:
            raise RuntimeError("No model loaded")

        processor = self.processor
        tokenizer = getattr(processor, "tokenizer", processor)
        params = req.params

        # media resolution is blocking (PIL + disk for CAS handles) — off-loop
        messages, images, videos = await asyncio.to_thread(self._prepare_chat_messages, req)
        vision: Optional[dict] = None
        if images or videos:
            template_owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
            text = template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            proxy = RealTimeProcessorProxy(
                processor,
                min_pixels=params.min_pixels, max_pixels=params.max_pixels,
                video_fps=params.video_fps, min_frames=params.min_frames,
                max_frames=params.max_frames,
                multi_image_max_pixels=params.multi_image_max_pixels,
                video_max_pixels=params.video_max_pixels,
            )
            inputs = await run_processor(proxy, text, images, videos)
            input_ids = inputs["input_ids"]
            vision = {
                key: inputs.get(key)
                for key in ("pixel_values", "grid_thw", "cross_attention_mask", "media_nums_per_sample")
            }
            attn_configs = self._attn_configs()
            saved_impls = [(cfg, cfg._attn_implementation) for cfg in attn_configs]
            vision, input_ids = prepare_vision_prefill(
                self.model, self.device, input_ids, vision)
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            attn_configs = self._attn_configs()
            saved_impls = [(cfg, cfg._attn_implementation) for cfg in attn_configs]

        # sdpa on CUDA (upstream: FA2 varlen misreads 3-D M-RoPE as packed
        # batch), unchanged on NPU — serving_policy
        for cfg in attn_configs:
            cfg._attn_implementation = offline_chat_attn_impl(cfg._attn_implementation)

        deltas: "queue_mod.Queue" = queue_mod.Queue()

        loop = asyncio.get_running_loop()
        decode_exc: list = []

        def _decode():
            try:
                self._offline_decode_loop(tokenizer, input_ids, params, deltas.put, vision=vision)
            except BaseException as exc:
                decode_exc.append(exc)
                log.exception("offline generate failed: %s", exc)
            finally:
                deltas.put(None)

        worker = launch_decode_worker(loop, _decode)
        try:
            while True:
                try:
                    delta = await loop.run_in_executor(None, deltas.get, True, 5.0)
                except queue_mod.Empty:
                    if worker.still_running():
                        continue  # slow step on a contended GPU — keep waiting
                    break
                if delta is None:
                    break
                if delta:
                    yield delta
        finally:
            await worker.shutdown()
            for cfg, impl in saved_impls:
                cfg._attn_implementation = impl
        if decode_exc:
            raise RuntimeError(f"generation failed: {decode_exc[0]}") from decode_exc[0]
