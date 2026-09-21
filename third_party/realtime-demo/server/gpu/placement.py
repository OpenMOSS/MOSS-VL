"""Turn a GPU topology into a process placement plan.

One auto-detected code path covers every deployment class. Eligible GPUs are
first split online/offline (offline ≈ ratio of the fleet, highest indices,
dedicated to sglang offline-chat sidecars), then the online side hosts the
realtime workers plus ASR/TTS:
- 8×H200  → 6 VLM workers (FA2) on GPUs 0-5 + 2 offline sglang GPUs (6-7);
  ASR and the TTS sidecars stay on the high ONLINE GPUs (away from the busiest
  VLM contexts, off the dedicated offline cards);
- 2×RTX 6000 Blackwell → 1 worker (sdpa — no FA cubins for sm_120) + 1 offline;
- 1×RTX 4090 → 1 worker, no offline, everything colocated on GPU 0.

`python -m server.gpu.placement` prints the plan for the current box (dry run).
"""
from __future__ import annotations

import math
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from ..device_compat import device_str as _device_str, is_npu
from .topology import GpuInfo, probe_topology, select_attn_impl

log = get_logger(__name__)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    gpu_index: int          # host GPU index (worker sees it as cuda:0)
    port: int
    attn_impl: str          # pre-resolved for this GPU's compute capability
    fake: bool = False      # VLM_WORKER_FAKE: scripted stream, no model load

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class TtsSpec:
    sidecar_id: int
    port: int
    gpu_index: Optional[int]  # None = inherit the parent's CUDA visibility

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class OfflineSpec:
    """One sglang offline-chat sidecar on a dedicated GPU."""
    replica_id: int
    gpu_index: int
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class PlacementPlan:
    gpus: Tuple[GpuInfo, ...]
    workers: Tuple[WorkerSpec, ...]
    tts: Tuple[TtsSpec, ...]
    asr_device: str
    warnings: Tuple[str, ...] = field(default=())
    offline: Tuple[OfflineSpec, ...] = field(default=())

    @property
    def capacity(self) -> int:
        return len(self.workers)

    @property
    def offline_capacity(self) -> int:
        return len(self.offline)


def _tts_base_port(settings: Settings) -> int:
    return urllib.parse.urlparse(settings.moss_tts_nano_base_url).port or 18100


def _explicit_worker_gpus(settings: Settings) -> Optional[List[int]]:
    raw = (settings.vlm_worker_gpus or "").strip()
    if not raw:
        return None
    return [int(part) for part in raw.split(",") if part.strip() != ""]


def offline_gpu_count(n_eligible: int, settings: Settings,
                      warnings: Optional[List[str]] = None) -> int:
    """How many of `n_eligible` GPUs the offline sglang plane gets.

    `floor(n * ratio + 0.5)` (round-half-up) with two guards — never all GPUs
    (realtime is the primary product) and none on 1-GPU boxes — reproduces the
    whole spec table at the default 1:3 ratio: 1→0, 2→1, 3→1, 4→1, 8→2.
    An explicit OFFLINE_GPU_COUNT wins, clamped to the same guards.
    """
    if settings.offline_provider.strip().lower() in {"", "none", "off", "0"}:
        return 0
    if n_eligible <= 1:
        return 0
    cap = n_eligible - 1
    if settings.offline_gpu_count is not None:
        n = max(0, min(cap, settings.offline_gpu_count))
        if warnings is not None and n != settings.offline_gpu_count:
            warnings.append(
                f"OFFLINE_GPU_COUNT={settings.offline_gpu_count} clamped to {n} "
                f"({n_eligible} eligible GPUs)")
        return n
    ratio = max(0.0, settings.offline_gpu_ratio)
    return max(1, min(cap, int(n_eligible * ratio + 0.5)))


def plan_placement(topology: List[GpuInfo], settings: Settings) -> PlacementPlan:
    warnings: List[str] = []
    by_index = {g.index: g for g in topology}

    # ---- split eligible GPUs: offline sglang plane first, workers on the rest ----
    offline_gpus: List[int] = []
    explicit = _explicit_worker_gpus(settings)
    if explicit is not None:
        worker_gpus = explicit  # repeats allowed (fake multi-GPU testing)
        missing = [i for i in worker_gpus if i not in by_index]
        if missing and topology:
            warnings.append(f"VLM_WORKER_GPUS names unknown GPU indices {missing}")
        # hand-placed layout: only carve an offline plane when explicitly sized
        # (auto-carving would surprise VLM_WORKER_GPUS=0,0,0,0-style test rigs)
        if settings.offline_gpu_count:
            leftover = sorted((i for i in by_index if i not in set(worker_gpus)),
                              reverse=True)
            offline_gpus = sorted(leftover[:settings.offline_gpu_count])
            if len(offline_gpus) < settings.offline_gpu_count:
                warnings.append(
                    f"OFFLINE_GPU_COUNT={settings.offline_gpu_count} but only "
                    f"{len(offline_gpus)} GPUs left beside VLM_WORKER_GPUS")
    else:
        eligible = [g.index for g in sorted(topology, key=lambda g: g.index)
                    if g.mem_free_mib >= settings.vlm_min_free_mib]
        skipped = [g.index for g in topology if g.index not in eligible]
        if skipped:
            warnings.append(
                f"GPUs {skipped} skipped for VLM (free < {settings.vlm_min_free_mib} MiB)")
        # explicit OFFLINE_GPUS pins the plane (highest-index default can land
        # on cards the sglang NPU engine cannot address)
        pinned = [int(p) for p in (settings.offline_gpus or "").split(",") if p.strip() != ""]
        if pinned:
            missing = [i for i in pinned if i not in by_index]
            if missing:
                warnings.append(f"OFFLINE_GPUS names unknown GPU indices {missing}")
            offline_gpus = [i for i in pinned if i in by_index]
            worker_gpus = [i for i in eligible if i not in set(offline_gpus)] or [settings.gpu_id]
        else:
            n_off = offline_gpu_count(len(eligible), settings, warnings)
            # offline takes the HIGHEST indices: online workers keep their
            # lowest-first ordering and the low-GPU session affinity stays intact
            offline_gpus = eligible[len(eligible) - n_off:]
            worker_gpus = eligible[:len(eligible) - n_off]
    if not worker_gpus:
        # degenerate fallback: keep today's single-GPU behavior alive rather
        # than refusing to start (dev boxes, busy GPUs, missing nvidia-smi)
        worker_gpus = [settings.gpu_id]
        warnings.append(
            f"no eligible GPUs detected — falling back to one worker on GPU {settings.gpu_id}")

    offline = tuple(
        OfflineSpec(replica_id=i, gpu_index=gpu, port=settings.sglang_base_port + i)
        for i, gpu in enumerate(offline_gpus)
    )

    workers = tuple(
        WorkerSpec(
            worker_id=i,
            gpu_index=gpu,
            port=settings.vlm_worker_base_port + i,
            attn_impl=select_attn_impl(
                by_index[gpu].compute_cap if gpu in by_index else (0, 0),
                settings.attn_impl),
            fake=settings.vlm_worker_fake,
        )
        for i, gpu in enumerate(worker_gpus)
    )

    # ASR/TTS colocate with ONLINE workers only — the offline GPUs are dedicated
    # to sglang (mem_fraction_static claims most of the card at boot)
    online_indices = sorted({gpu for gpu in worker_gpus if gpu in by_index}) \
        or sorted(by_index)

    # ---- ASR: highest-index online GPU (explicit ASR_DEVICE wins) ----
    asr_device = (settings.asr_device or "auto").strip()
    if asr_device.lower() == "auto":
        if online_indices:
            asr_device = _device_str(online_indices[-1])
        else:
            asr_device = _device_str(0)

    # ---- TTS sidecars: ceil(workers / sessions_per_sidecar), high GPUs first ----
    # a vLLM-Omni engine continuous-batches concurrent streams, so it carries
    # more sessions than a serialize-per-job pytorch/native sidecar
    from ..adapters.tts.providers import is_external_provider, is_vllm_engine_provider
    if is_external_provider(settings.tts_provider):
        # external cloud API (elevenlabs): no GPU, no sidecar process at all
        tts: Tuple[TtsSpec, ...] = ()
        plan = PlacementPlan(gpus=tuple(topology), workers=workers, tts=tts,
                             asr_device=asr_device, warnings=tuple(warnings),
                             offline=offline)
        for w in warnings:
            log.warning("placement: %s", w)
        return plan
    if is_vllm_engine_provider(settings.tts_provider):
        per = max(1, settings.tts_sessions_per_engine)
    else:
        per = max(1, settings.tts_sessions_per_sidecar)
    # TTS sidecar COUNT:
    #  - explicit TTS_SIDECAR_COUNT wins.
    #  - vLLM-Omni engine: continuous-batches many streams per engine AND each
    #    reserves a big KV pool, so derive from workers (don't pile engines).
    #  - native/pytorch: each sidecar is BATCH-1 (one concurrent session), so
    #    scale the count to the live (online) card amount, one sidecar per card:
    #        online cards  >=4 -> 4 | 3 -> 3 | 2 -> 2 | 1 -> 2 (piled on the card)
    #    min 2 so even a single online card serves 2 concurrent sessions (the
    #    other card is the offline plane); cap 4. The round-robin below places
    #    one per online card, piling both onto the single card when there is one.
    live_cards = len(online_indices)
    if settings.tts_sidecar_count > 0:
        n_sidecars = settings.tts_sidecar_count
    elif is_vllm_engine_provider(settings.tts_provider):
        n_sidecars = max(1, math.ceil(len(workers) / per))
    else:
        n_sidecars = min(4, max(2, live_cards))
    base_port = _tts_base_port(settings)
    if settings.tts_sidecar_gpu:
        # explicit TTS_GPU pins every sidecar (pre-multi-GPU knob, still honored)
        tts_gpus: List[Optional[int]] = [int(settings.tts_sidecar_gpu)] * n_sidecars
    elif online_indices:
        descending = list(reversed(online_indices))
        tts_gpus = [descending[i % len(descending)] for i in range(n_sidecars)]
    else:
        tts_gpus = [None] * n_sidecars  # inherit (today's single-box behavior)
    tts = tuple(
        TtsSpec(sidecar_id=i, port=base_port + i, gpu_index=tts_gpus[i])
        for i in range(n_sidecars)
    )

    plan = PlacementPlan(gpus=tuple(topology), workers=workers, tts=tts,
                         asr_device=asr_device, warnings=tuple(warnings),
                         offline=offline)
    for w in warnings:
        log.warning("placement: %s", w)
    return plan


def format_plan(plan: PlacementPlan) -> str:
    lines = [f"placement: {len(plan.gpus)} GPU(s), {plan.capacity} VLM worker(s), "
             f"{plan.offline_capacity} offline sglang, "
             f"{len(plan.tts)} TTS sidecar(s), asr={plan.asr_device}"]
    for g in plan.gpus:
        lines.append(f"  gpu {g.index}: {g.name}  {g.mem_free_mib}/{g.mem_total_mib} MiB free"
                     f"  cc{g.compute_cap[0]}.{g.compute_cap[1]}")
    for w in plan.workers:
        lines.append(f"  vlm worker {w.worker_id}: gpu {w.gpu_index}  :{w.port}"
                     f"  attn={w.attn_impl}{'  FAKE' if w.fake else ''}")
    for o in plan.offline:
        lines.append(f"  offline sglang replica {o.replica_id}: gpu {o.gpu_index}  :{o.port}")
    for t in plan.tts:
        dev = "inherit" if t.gpu_index is None else f"gpu {t.gpu_index}"
        lines.append(f"  tts sidecar {t.sidecar_id}: {dev}  :{t.port}")
    for w in plan.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


if __name__ == "__main__":
    from ..config import get_settings

    topo = probe_topology()
    print(format_plan(plan_placement(topo, get_settings())))
