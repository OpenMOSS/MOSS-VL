"""Backend serving policy for the MOSS-VL HF adapter.

The ONE module in this adapter allowed to ask which accelerator we are on
(via ``device_compat.is_npu`` — never by sniffing ``torch.cuda`` directly).
Every backend divergence — value decisions AND structural ones (threading,
dispatch) — lives here as a named, self-documenting function;
``adapter.py`` calls the policy and contains zero backend branches.

All functions preserve upstream behavior verbatim on CUDA boxes.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, Tuple

from ....device_compat import is_npu


# ---------------------------------------------------------------------------
# Attention implementation selection
# ---------------------------------------------------------------------------

def clamp_attn_override(override: str) -> str:
    """Explicit attn overrides win — except flash_attention_2 on NPU, where
    no flash-attn kernels exist (clamp to eager rather than fail at load)."""
    if override == "flash_attention_2" and is_npu():
        return "eager"
    return override


def auto_attn_impl(gpu_id: int = 0, settings_impl: str = "auto") -> str:
    """Auto attn resolution.

    NPU: eager — CANN 9.0 SDPA hangs on 3-D M-RoPE incremental decode.
    Elsewhere: the upstream compute-capability rule via gpu.topology
    (Blackwell sm_120+ -> sdpa, otherwise flash_attention_2; a CPU box
    probes cc (0,0) -> FA2, which the load-time sdpa fallback then
    catches — upstream behavior)."""
    if is_npu():
        return "eager"
    import torch
    cc = (0, 0)
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(gpu_id)
    except Exception:  # noqa: BLE001 — unknown cc → FA-first with sdpa fallback
        pass
    from ....gpu.topology import select_attn_impl as _by_cc
    return _by_cc(cc, settings_impl)


def realtime_attn_impl(current: str) -> str:
    """Attn impl for realtime sessions.

    NPU: eager (SDPA hangs on 3-D M-RoPE decode steps). CUDA: unchanged
    (flash-attn 2 / sdpa on Blackwell, as loaded)."""
    return "eager" if is_npu() else current


def offline_chat_attn_impl(current: str) -> str:
    """Attn impl for the duration of an offline chat.

    CUDA: sdpa — flash-attn's varlen wrapper misreads 3-D M-RoPE
    position_ids as a packed batch and crashes (upstream forces sdpa
    for exactly this reason). NPU: unchanged (load-time eager)."""
    return current if is_npu() else "sdpa"


# ---------------------------------------------------------------------------
# Structural serving policy: where tensor work runs
# ---------------------------------------------------------------------------

async def run_processor(proxy: Any, text: str, images: list, videos: list) -> Any:
    """Run the chat processor (image/video preprocessing).

    NPU: inline on the loop thread — tensors created in worker threads can
    hit inconsistent NPU allocator state (NaN). CUDA: the upstream
    off-loop dispatch (video decode is blocking CPU work)."""
    if is_npu():
        return proxy(text=[text], images=images or None,
                     videos=videos or None, return_tensors="pt")
    return await asyncio.to_thread(
        lambda: proxy(text=[text], images=images or None,
                     videos=videos or None, return_tensors="pt"))


def prepare_vision_prefill(model: Any, device: str, input_ids: Any,
                           vision: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """NPU: move vision tensors to device and run the vision prefill on the
    caller's (loop) thread — the largest tensor op must not come from a
    worker thread (allocator thread-safety). The dict is repacked as a
    results envelope so the decode worker skips its own prefill.

    CUDA: unchanged — the prefill runs inside the decode worker, as
    upstream does."""
    if not is_npu():
        return vision, input_ids
    for k, v in vision.items():
        if hasattr(v, "to"):
            vision[k] = v.to(device)
    input_ids = input_ids.to(device)
    import torch
    attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=None,
            cache_position=cache_position, use_cache=True, **vision,
        )
    return {"_prefill_outputs": prefill_outputs}, input_ids


class _ExecutorDecodeWorker:
    """Decode worker on the loop's default executor (NPU)."""

    def __init__(self, task: Any) -> None:
        self._task = task

    def still_running(self) -> bool:
        return not self._task.done()

    async def shutdown(self) -> None:
        await self._task


class _ThreadDecodeWorker:
    """Decode worker on a dedicated daemon thread (CUDA, upstream)."""

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread

    def still_running(self) -> bool:
        return self._thread.is_alive()

    async def shutdown(self) -> None:
        self._thread.join(timeout=2.0)


def launch_decode_worker(loop: asyncio.AbstractEventLoop, target: Callable[[], None]) -> Any:
    """Start the offline-chat decode worker; returns a handle with
    still_running() (feeds the queue-wait loop) and shutdown() (joins it).

    NPU: the loop's default executor (not a raw thread) keeps decode ops
    stable; safe because live sessions are capped per replica. CUDA: the
    upstream dedicated daemon thread."""
    if is_npu():
        return _ExecutorDecodeWorker(loop.run_in_executor(None, target))
    thread = threading.Thread(target=target, name="offline-chat", daemon=True)
    thread.start()
    return _ThreadDecodeWorker(thread)
