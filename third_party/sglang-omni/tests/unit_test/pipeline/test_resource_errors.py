"""NPU termination policy must not change CUDA batch or thread recovery."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.pipeline.stage import runtime
from sglang_omni.platforms.errors import is_fatal_npu_oom
from sglang_omni.scheduling import omni_scheduler
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler
from tests.unit_test.pipeline.helpers import make_stage


@pytest.mark.parametrize("device", ["cuda", "cpu", "xpu", "npu"])
@pytest.mark.parametrize(
    "error,oom",
    [
        (torch.OutOfMemoryError("allocation failed"), True),
        (RuntimeError("NPU out of memory"), True),
        (RuntimeError("forward failed"), False),
        (ValueError("out of memory in user input"), False),
    ],
)
def test_platform_error_policy(monkeypatch, device, error, oom):
    monkeypatch.setattr(platforms.current_platform, "device_type", device)
    assert is_fatal_npu_oom(error) is (device == "npu" and oom)


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_batch_oom_policy_survives_reporting_failure(
    monkeypatch, device, cleanup_fails
):
    monkeypatch.setattr(platforms.current_platform, "device_type", device)
    events = []
    monkeypatch.setattr(
        omni_scheduler.os, "_exit", lambda code: events.append(("exit", code))
    )
    scheduler = object.__new__(omni_scheduler.OmniScheduler)

    def report(rid, error):
        events.append(("error", rid))
        if cleanup_fails:
            raise RuntimeError("report failed")

    scheduler._emit_request_error = report
    scheduler._emit_model_path_end_once = lambda rid, **kw: events.append(("end", rid))
    scheduler.abort = lambda rid, **kw: events.append(("abort", rid))
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="failed")])
    error = torch.OutOfMemoryError("allocation failed")
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="report failed"):
            scheduler._handle_batch_failure(batch, error)
    else:
        scheduler._handle_batch_failure(batch, error)
        assert events[:3] == [
            ("error", "failed"),
            ("end", "failed"),
            ("abort", "failed"),
        ]
    assert (("exit", 1) in events) is (device == "npu")


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize(
    "flush_error", [None, RuntimeError("send failed"), TimeoutError()]
)
def test_scheduler_thread_exit_is_platform_scoped_and_bounded(
    monkeypatch, device, flush_error
):
    monkeypatch.setattr(platforms.current_platform, "device_type", device)
    exits, waits = [], []
    monkeypatch.setattr(runtime.os, "_exit", exits.append)

    class Flush:
        def result(self, timeout):
            waits.append(timeout)
            if flush_error is not None:
                raise flush_error

    def post(coro, loop):
        coro.close()
        return Flush()

    monkeypatch.setattr(runtime.asyncio, "run_coroutine_threadsafe", post)

    async def check():
        stage = make_stage(
            scheduler=FakeScheduler(fail_start=torch.OutOfMemoryError("OOM"))
        )
        try:
            await stage.start()
            thread = stage._scheduler_thread
            await asyncio.to_thread(thread.join, 2)
            assert not thread.is_alive()
        finally:
            await stage.stop()

    asyncio.run(check())
    assert exits == ([1] if device == "npu" else [])
    assert waits == ([30] if device == "npu" else [])


def test_npu_thread_exits_when_error_notification_cannot_be_submitted(monkeypatch):
    monkeypatch.setattr(platforms.current_platform, "device_type", "npu")
    exits = []
    notifications = []
    monkeypatch.setattr(runtime.os, "_exit", exits.append)

    def post(coro, loop):
        notifications.append(coro)
        raise RuntimeError("event loop already closed")

    monkeypatch.setattr(runtime.asyncio, "run_coroutine_threadsafe", post)

    async def check():
        stage = make_stage(
            scheduler=FakeScheduler(fail_start=torch.OutOfMemoryError("OOM"))
        )
        try:
            await stage.start()
            thread = stage._scheduler_thread
            await asyncio.to_thread(thread.join, 2)
            assert not thread.is_alive()
        finally:
            await stage.stop()

    asyncio.run(check())
    assert exits == [1]
    assert len(notifications) == 1 and notifications[0].cr_frame is None
