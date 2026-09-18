"""SGLang 0.5.14/0.5.16 API bridges; modern runtimes retain native behavior.

Scheduler calls are adapted on Omni instances, never by replacing methods on
the upstream Scheduler class. Missing request/forward metadata APIs are added
only on the legacy runtime, before Omni imports their consumers.
"""

from __future__ import annotations

import functools
import inspect
import sys
import types
from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass
class NextBatchPlan:
    batch_to_run: Any = None
    running_batch: Any = None


class ExtendRange(NamedTuple):
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def uses_legacy_runner(runner_class) -> bool:
    parameters = inspect.signature(runner_class.__init__).parameters
    if "ps" in parameters:
        return False
    if {"tp_rank", "tp_size"} <= parameters.keys():
        return True
    raise RuntimeError(
        "unsupported SGLang ModelRunner constructor; expected 0.5.14 or 0.5.16 API"
    )


def needs_bridge() -> bool:
    from sglang.srt.model_executor.model_runner import ModelRunner

    return uses_legacy_runner(ModelRunner)


def get_next_batch_plan(upstream, scheduler, running_batch, last_batch):
    method = upstream.get_next_batch_to_run
    if "running_batch" in inspect.signature(method).parameters:
        return method(scheduler, running_batch, last_batch)
    # None is an explicit state update, not an instruction to retain a stale batch.
    scheduler.running_batch = running_batch
    scheduler.last_batch = last_batch
    prefill = scheduler.get_new_batch_prefill
    missing = object()
    previous = scheduler.__dict__.get("get_new_batch_prefill", missing)

    def legacy_prefill():
        plan = prefill(scheduler.running_batch)
        scheduler.running_batch = plan.running_batch
        return plan.batch_to_run

    scheduler.get_new_batch_prefill = legacy_prefill
    try:
        batch = method(scheduler)
    finally:
        if previous is missing:
            scheduler.__dict__.pop("get_new_batch_prefill", None)
        else:
            scheduler.get_new_batch_prefill = previous
    return NextBatchPlan(batch, scheduler.running_batch)


def get_prefill_plan(upstream, scheduler, running_batch):
    method = upstream.get_new_batch_prefill
    if "running_batch" in inspect.signature(method).parameters:
        return method(scheduler, running_batch)
    scheduler.running_batch = running_batch
    return NextBatchPlan(method(scheduler), scheduler.running_batch)


def init_metrics_collector(upstream, scheduler, tp_rank, pp_rank, dp_rank):
    method = getattr(upstream, "init_metrics_collector", None)
    if method is not None:
        return method(scheduler, tp_rank, pp_rank, dp_rank)
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

    scheduler.metrics_collector_context = SchedulerMetricsCollector.init_new(
        server_args=scheduler.server_args,
        ps=scheduler.ps,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        enable_priority_scheduling=scheduler.enable_priority_scheduling,
        enable_lora=scheduler.enable_lora,
        enable_hierarchical_cache=scheduler.enable_hierarchical_cache,
    )
    scheduler.metrics_collector = scheduler.metrics_collector_context.collector


def init_metrics_reporter(upstream, scheduler, tp_rank, pp_rank, dp_rank):
    method = getattr(upstream, "init_metrics_reporter", None)
    if method is not None:
        return method(scheduler, tp_rank, pp_rank, dp_rank)
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )

    scheduler.metrics_reporter = SchedulerMetricsReporter(
        scheduler=scheduler,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        metrics_collector_context=scheduler.metrics_collector_context,
        metrics_collector=scheduler.metrics_collector,
    )


def install_req_extend_range(req_class) -> None:
    if hasattr(req_class, "set_extend_range"):
        return

    def get_range(self):
        end = getattr(self, "fill_len", 0)
        return ExtendRange(end - getattr(self, "extend_input_len", 0), end)

    def set_range(self, value):
        if value is None:
            self.fill_len = self.extend_input_len = 0
        else:
            self.fill_len = int(value.end)
            self.extend_input_len = int(value.length)

    def set_extend_range(self, start, end):
        self.extend_range = ExtendRange(int(start), int(end))

    # The old prefill planner still writes fill_len/extend_input_len directly.
    # A property keeps both API views synchronized through admission/rollback.
    req_class.extend_range = property(get_range, set_range)
    req_class.set_extend_range = set_extend_range


def install_forward_batch_shim(forward_class) -> None:
    original = forward_class.init_new.__func__
    if getattr(original, "_omni_legacy_forward", False):
        return
    parameters = inspect.signature(original).parameters
    overrides = (
        "capture_hidden_mode",
        "return_hidden_states_before_norm",
        "seq_lens_cpu_cache",
    )
    if all(name in parameters for name in overrides[:2]):
        return

    @functools.wraps(original)
    def init_new(cls, batch, model_runner, **kwargs):
        unsupported = set(kwargs) - set(parameters) - set(overrides)
        if unsupported:
            raise TypeError(
                f"unsupported ForwardBatch overrides: {sorted(unsupported)}"
            )
        saved = {}
        forwarded = {}
        try:
            for name, value in kwargs.items():
                if name in parameters:
                    forwarded[name] = value
                else:
                    if not hasattr(batch, name):
                        raise RuntimeError(
                            f"legacy ScheduleBatch has no {name} override slot"
                        )
                    saved[name] = getattr(batch, name)
                    setattr(batch, name, value)
            return original(cls, batch, model_runner, **forwarded)
        finally:
            for name, value in saved.items():
                setattr(batch, name, value)

    init_new._omni_legacy_forward = True
    forward_class.init_new = classmethod(init_new)


def install_kv_cache_configurator_shim() -> None:
    name = "sglang.srt.mem_cache.kv_cache_configurator"
    try:
        __import__(name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    import sglang.srt.mem_cache as package
    from sglang.srt.model_executor.model_runner import ModelRunner

    profile = ModelRunner._profile_available_bytes

    @dataclass
    class KVCacheConfigurator:
        model_runner: Any

        def __getattr__(self, field):
            return getattr(self.model_runner, field)

        def _profile_available_bytes(self, pre_model_load_memory):
            # Call the saved upstream implementation, not the Omni override.
            return profile(self.model_runner, pre_model_load_memory)

    module = types.ModuleType(name)
    module.KVCacheConfigurator = KVCacheConfigurator
    sys.modules[name] = module
    package.kv_cache_configurator = module


def apply_all() -> None:
    if not needs_bridge():
        return
    from sglang.srt import runtime_context
    from sglang.srt.environ import envs
    from sglang.srt.managers import schedule_batch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs

    if not hasattr(schedule_batch, "NextBatchPlan"):
        schedule_batch.NextBatchPlan = NextBatchPlan
    if not hasattr(ServerArgs, "dcp_size"):
        ServerArgs.dcp_size = 1
    install_req_extend_range(schedule_batch.Req)
    install_forward_batch_shim(ForwardBatch)
    install_kv_cache_configurator_shim()
    if not hasattr(runtime_context, "get_flags"):
        # Only this explicit capture flag is read/written by Omni on 0.5.14.
        flags = types.SimpleNamespace(
            capture=types.SimpleNamespace(enable_torch_compile=False)
        )
        runtime_context.get_flags = lambda: flags
    if not hasattr(envs, "SGLANG_MAX_NEW_TOKENS_LIMIT"):
        from sglang.srt.environ import EnvInt

        field = EnvInt(65536)
        field.name = "SGLANG_MAX_NEW_TOKENS_LIMIT"
        envs.SGLANG_MAX_NEW_TOKENS_LIMIT = field
