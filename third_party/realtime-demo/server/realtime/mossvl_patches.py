"""Runtime monkeypatches for the streaming MOSS-VL model.

Faithful port of board inference.py:447-589. The streaming model's
`real_time_generate` (shipped in the checkpoint's remote code) is patched on the
loaded instance so that:

  1. the initial system + user turn is prefilled into the KV cache before frames
     start streaming (install_realtime_initial_prompt_patch), and
  2. a cooperative barge-in injects <|eot_id|> and waits for new input WITHOUT
     tearing down the KV cache (install_realtime_turn_interrupt_patch).

These two behaviours are load-bearing for the wire protocol the frontend speaks
(<|round_start|> to open a turn, <|silence|> when idle, <|eot_id|> on interrupt).
"""
from __future__ import annotations

import json
import queue
from types import MethodType
from typing import Any, List, Optional

import torch

from ..logging_conf import get_logger

log = get_logger(__name__)

_REQUIRED = object()


def _config_attr(config: Any, name: str, default: Any = _REQUIRED) -> Any:
    """Read a model-config attribute from the top level, falling back to a
    nested ``text_config``. The streaming MOSS-VL checkpoint nests
    ``cross_attention_layers`` under ``config.text_config`` (NOT top-level),
    while ``image_token_id`` sits at the top level — this handles either layout.
    Raises AttributeError when required (no default) and absent in both."""
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    if default is _REQUIRED:
        raise AttributeError(f"config has no {name!r} (checked top-level and text_config)")
    return default


EOT_TOKEN = "<|eot_id|>"

DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)


def prompt_log_preview(prompt: str, limit: int = 360) -> str:
    compact = " ".join((prompt or "").split())
    if len(compact) <= limit:
        return compact
    head = compact[: max(80, limit // 2)]
    tail = compact[-max(80, limit // 3):]
    return f"{head} ... {tail}"


_PREFILL_ROLES = frozenset({"system", "user", "assistant"})


def decode_prefill_messages(raw: Any) -> Optional[List[dict]]:
    """Decode + validate a `prefill_messages` JSON string (rollover re-seat).

    Returns a clean `[{"role":…, "content":…}]` list, or None on ANY problem
    (bad JSON, wrong shape, unknown role, non-str content) — the caller then
    falls back to the legacy system+user prefill pair. Validation matters
    because this string crosses the worker HTTP boundary and lands directly in
    `apply_chat_template`.
    """
    if raw is None:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        log.warning("prefill_messages rejected (bad JSON): %s", exc)
        return None
    if not isinstance(data, list) or not data:
        log.warning("prefill_messages rejected: not a non-empty list")
        return None
    out: List[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            log.warning("prefill_messages rejected: entry is not an object")
            return None
        role = entry.get("role")
        content = entry.get("content")
        if role not in _PREFILL_ROLES or not isinstance(content, str):
            log.warning("prefill_messages rejected: bad role/content (%r)", role)
            return None
        out.append({"role": role, "content": content})
    return out


def install_realtime_initial_prompt_patch(model_instance: Any) -> None:
    """Prefill a system + initial-user turn before realtime frames arrive.

    The upstream streaming impl hard-codes an empty user message; the board SFT
    data instead puts the task in the initial user turn. This patch uses the
    training-style prefill when an initial prompt is provided and preserves
    upstream behaviour otherwise.

    Rollover extension (design §6): when `_board_realtime_prefill_messages` is
    set on the model (a validated `[{role, content}]` list from a memory
    re-seat), it REPLACES the legacy system+user pair wholesale — the gateway
    controls the rebuilt prefix completely. Unset/None keeps the default path
    byte-identical.
    """
    if getattr(model_instance, "_board_initial_prompt_patch_installed", False):
        return

    original_func = getattr(model_instance, "real_time_generate", None)
    if not callable(original_func):
        raise RuntimeError("Realtime model does not expose real_time_generate")

    def _patched(
        model_self: Any,
        new_video_frames: queue.Queue,
        new_prompts: queue.Queue,
        output_text_queue: queue.Queue,
        processor: Any,
        max_tokens_per_turn: int = 86400,
        **generate_kwargs: Any,
    ):
        prefill = decode_prefill_messages(
            getattr(model_self, "_board_realtime_prefill_messages", None))
        if prefill is not None:
            initial_messages = prefill
            log.info(
                "Realtime rollover prefill: %d message(s), chars=%d, preview=%r",
                len(initial_messages),
                sum(len(m["content"]) for m in initial_messages),
                prompt_log_preview(initial_messages[0]["content"]),
            )
        else:
            initial_prompt = str(getattr(model_self, "_board_realtime_initial_prompt", "") or "")
            system_prompt = str(
                getattr(model_self, "_board_realtime_system_prompt", DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT)
                or ""
            )
            initial_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt},
            ]
            log.info(
                "Realtime initial prompt prefilled: chars=%d, system_chars=%d, preview=%r",
                len(initial_prompt), len(system_prompt), prompt_log_preview(initial_prompt),
            )

        initial_input_ids = processor.tokenizer.apply_chat_template(
            initial_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model_self.device)
        initial_attention_mask = torch.ones_like(initial_input_ids)
        prefill_len = initial_input_ids.shape[1]
        prefill_positions = (
            torch.arange(prefill_len, dtype=torch.long, device=model_self.device)
            .view(1, 1, prefill_len)
            .expand(3, 1, prefill_len)
            .contiguous()
        )

        model_self.start_real_time_generate()
        try:
            return model_self._real_time_generate(
                new_video_frames=new_video_frames,
                new_prompts=new_prompts,
                output_text_queue=output_text_queue,
                processor=processor,
                inputs=initial_input_ids,
                attention_mask=initial_attention_mask,
                position_ids=prefill_positions,
                realtime_next_position=prefill_len,
                full_vision_token_info=None,
                max_tokens_per_turn=max_tokens_per_turn,
                **generate_kwargs,
            )
        finally:
            model_self.stop_real_time_generate()

    model_instance.real_time_generate = MethodType(_patched, model_instance)
    model_instance._board_initial_prompt_patch_installed = True
    log.info("Realtime initial prompt patch installed")


def install_realtime_turn_interrupt_patch(model_instance: Any) -> None:
    """Cooperative end-of-turn (barge-in) that keeps the KV cache alive.

    stop_real_time_generate tears down the loop; for barge-in we only stop the
    current assistant turn, emit <|eot_id|> to the UI, and wait for new input so
    the next prompt continues in the same realtime context.
    """
    if getattr(model_instance, "_board_turn_interrupt_patch_installed", False):
        return

    original_func = getattr(model_instance, "_update_model_kwargs_for_real_time_generation", None)
    if not callable(original_func):
        raise RuntimeError("Realtime model does not expose _update_model_kwargs_for_real_time_generation")

    model_instance.turn_interrupt_requested = False

    def _patched(
        model_self: Any,
        outputs: Any,
        input_ids: Any,
        model_kwargs: Any,
        should_wait_for_new_input: bool,
        new_video_frames: Any,
        new_prompts: Any,
        output_text_queue: Any,
        token_buffer: Any,
        processor: Any,
        **kwargs: Any,
    ):
        if getattr(model_self, "turn_interrupt_requested", False):
            log.info("Realtime soft interrupt requested; injecting <|eot_id|> and waiting for new input")
            model_self.turn_interrupt_requested = False
            try:
                if output_text_queue is not None:
                    output_text_queue.put(EOT_TOKEN)
            except Exception as exc:  # noqa: BLE001
                log.debug("Realtime interrupt failed to enqueue eot: %s", exc)
            try:
                token_buffer.clear()
            except Exception:  # noqa: BLE001
                pass
            should_wait_for_new_input = True

        return original_func(
            outputs,
            input_ids,
            model_kwargs,
            should_wait_for_new_input,
            new_video_frames,
            new_prompts,
            output_text_queue,
            token_buffer,
            processor,
            **kwargs,
        )

    model_instance._update_model_kwargs_for_real_time_generation = MethodType(_patched, model_instance)
    model_instance._board_turn_interrupt_patch_installed = True
    log.info("Realtime soft turn interrupt patch installed")


def install_realtime_frame_window_patch(model_instance: Any, keep_frames: int) -> None:
    """Keep only the most recent ``keep_frames`` video frames in the KV cache.

    MOSS-VL is a gated cross-attention VLM: the ~250 tokens/frame of vision K/V live
    in a SEPARATE KV lane (the ``config.cross_attention_layers``), while the
    conversation words sit in the self-attention lane with their own M-RoPE positions.
    So "evict old frames, keep the words" is a contiguous FIFO prefix-drop of the
    cross-attention layers only — the text stream is never cut, so no StreamingLLM-style
    position re-indexing is needed, and because surviving frames are always recent their
    query<->key RoPE offsets stay in-distribution (the divergence fix).

    This wraps ``_update_model_kwargs_for_real_time_generation`` (the per-step hook that
    already grows ``full_vision_token_info`` and rebuilds the frame-level
    ``cross_attention_mask``): it calls the original, then evicts. ``keep_frames <= 0``
    disables (installs nothing). Mirrors the eviction primitive in StreamingVLM
    (mit-han-lab/streaming-vlm, arXiv 2510.09608), adapted to the cross-attention lane.
    """
    if getattr(model_instance, "_frame_window_patch_installed", False):
        # already installed on this reused model — just refresh the budget for the new
        # session (may raise, lower, or disable the window). keep_frames<=0 => off.
        model_instance._frame_window_keep_frames = max(0, int(keep_frames))
        return
    if keep_frames <= 0:
        return  # first-ever install with the window off — nothing to wrap

    original_func = getattr(model_instance, "_update_model_kwargs_for_real_time_generation", None)
    if not callable(original_func):
        raise RuntimeError("Realtime model does not expose _update_model_kwargs_for_real_time_generation")

    cross_layers = list(_config_attr(model_instance.config, "cross_attention_layers", []) or [])
    if not cross_layers:
        raise RuntimeError("Model config has no cross_attention_layers (checked top-level and "
                           "text_config); frame-window eviction cannot target vision KV")

    image_token_id = int(_config_attr(model_instance.config, "image_token_id"))
    model_instance._frame_window_keep_frames = int(keep_frames)
    model_instance._frame_window_evicted_total = 0

    def _patched(model_self: Any, outputs, input_ids, model_kwargs, *args, **kwargs):
        input_ids, model_kwargs = original_func(outputs, input_ids, model_kwargs, *args, **kwargs)

        keep = int(getattr(model_self, "_frame_window_keep_frames", 0))
        if keep <= 0:
            return input_ids, model_kwargs

        fvti = model_kwargs.get("full_vision_token_info")
        pkv = model_kwargs.get("past_key_values")
        if not fvti or not fvti[0].get("medias") or pkv is None:
            return input_ids, model_kwargs

        medias = fvti[0]["medias"]
        # One media == one frame in realtime mode; be robust if that ever changes.
        frames_in_cache = sum(int(m.get("num_frames", 1)) for m in medias)

        if frames_in_cache > keep:
            # Drop the oldest whole frames so exactly `keep` remain. medias are
            # append-ordered and their [start,end) offsets are contiguous vision-token
            # coordinates == positions in the cross-attention KV cache.
            drop_frames = frames_in_cache - keep
            # find the media boundary that leaves `keep` frames, counting num_frames
            acc = 0
            drop_medias = 0
            for m in medias:
                if acc >= drop_frames:
                    break
                acc += int(m.get("num_frames", 1))
                drop_medias += 1
            cut = int(medias[drop_medias - 1]["end"])  # vision tokens to remove from the front

            for li in cross_layers:
                try:
                    layer = pkv.layers[li]
                except (AttributeError, IndexError, TypeError):
                    layer = None
                if layer is None or getattr(layer, "keys", None) is None:
                    continue
                if layer.keys.shape[-2] <= cut:
                    continue  # nothing to trim on this layer (shouldn't happen — guard anyway)
                layer.keys = layer.keys[:, :, cut:, :].contiguous()
                layer.values = layer.values[:, :, cut:, :].contiguous()

            kept = medias[drop_medias:]
            for m in kept:
                m["start"] = int(m["start"]) - cut
                m["end"] = int(m["end"]) - cut
            fvti[0]["medias"] = kept
            fvti[0]["total_length"] = int(fvti[0]["total_length"]) - cut
            fvti[0]["pad_start"] = int(fvti[0]["pad_start"]) - cut
            fvti[0]["pad_end"] = int(fvti[0]["pad_end"]) - cut
            model_self._frame_window_evicted_total += acc
            log.debug(
                "Frame window: evicted %d frame(s) (%d vision tokens), %d kept",
                acc, cut, sum(int(m.get("num_frames", 1)) for m in kept),
            )

        # Once anything has been evicted the frame-level cross_attention_mask that the
        # original just rebuilt over-counts: input_ids keeps EVERY frame's <|image_pad|>
        # forever (text is never cut), so the cumulative pad count still includes evicted
        # frames. Re-derive the mask against only the frames still in the cache.
        evicted = int(getattr(model_self, "_frame_window_evicted_total", 0))
        if evicted > 0 and fvti[0].get("medias"):
            n = sum(int(m.get("num_frames", 1)) for m in fvti[0]["medias"])
            cum = (input_ids == image_token_id).cumsum(dim=1)          # (1, T) frames-ever-seen
            fi = torch.arange(n, device=input_ids.device)
            visible = (cum.unsqueeze(-1) - evicted) > fi               # (1, T, n)
            model_kwargs["cross_attention_mask"] = (~visible).unsqueeze(1)

        return input_ids, model_kwargs

    model_instance._update_model_kwargs_for_real_time_generation = MethodType(_patched, model_instance)
    model_instance._frame_window_patch_installed = True
    log.info("Realtime frame-window patch installed (keep_frames=%d, cross_layers=%d)",
             keep_frames, len(cross_layers))


def install_realtime_token_counter_patch(model_instance: Any) -> None:
    """Stash the exact TEXT-token count (`model._rt_text_tokens`) every step.

    The rollover trigger (design §6) fires on text tokens, not KV ratio — ~80%
    of text growth is timestamp wrappers for frames whose vision KV the frame
    window already evicted. This wraps the same per-step hook as the
    frame-window patch and must be installed LAST (outermost), so it reads the
    FINAL `input_ids` length after every other wrapper ran. `input_ids` keeps
    every frame's <|image_pad|> forever (text is never cut), so its length IS
    the session's text-token count. Read by RealtimeSession.status_payload and
    reset by the adapter on each start_realtime_session.
    """
    if getattr(model_instance, "_token_counter_patch_installed", False):
        return

    original_func = getattr(model_instance, "_update_model_kwargs_for_real_time_generation", None)
    if not callable(original_func):
        raise RuntimeError("Realtime model does not expose _update_model_kwargs_for_real_time_generation")

    def _patched(model_self: Any, outputs: Any, input_ids: Any, model_kwargs: Any, *args: Any, **kwargs: Any):
        input_ids, model_kwargs = original_func(outputs, input_ids, model_kwargs, *args, **kwargs)
        try:
            model_self._rt_text_tokens = int(input_ids.shape[1])
        except Exception:  # noqa: BLE001 — a counter must never break a step
            pass
        return input_ids, model_kwargs

    model_instance._update_model_kwargs_for_real_time_generation = MethodType(_patched, model_instance)
    model_instance._token_counter_patch_installed = True
    log.info("Realtime text-token counter patch installed")
