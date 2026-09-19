"""Contract regressions for both SGLang APIs, without model construction."""

from types import SimpleNamespace

import pytest

from sglang_omni import compat


def test_runner_api_detection():
    class Legacy:
        def __init__(self, tp_rank, tp_size):
            pass

    class Modern:
        def __init__(self, ps):
            pass

    class Unknown:
        def __init__(self, unknown):
            pass

    assert compat.uses_legacy_runner(Legacy)
    assert not compat.uses_legacy_runner(Modern)
    with pytest.raises(RuntimeError, match="unsupported"):
        compat.uses_legacy_runner(Unknown)


def test_legacy_scheduler_clears_last_batch_and_keeps_upstream_unchanged():
    class Upstream:
        def get_next_batch_to_run(self):
            assert self.last_batch is None
            return self.get_new_batch_prefill()

    class Omni:
        last_batch = "stale"

        def get_new_batch_prefill(self, running_batch):
            assert running_batch == "running"
            return compat.NextBatchPlan("new-work", "reconciled")

    original = Upstream.get_next_batch_to_run
    scheduler = Omni()
    plan = compat.get_next_batch_plan(Upstream, scheduler, "running", None)
    assert plan == compat.NextBatchPlan("new-work", "reconciled")
    assert "get_new_batch_prefill" not in scheduler.__dict__
    assert Upstream.get_next_batch_to_run is original


def test_legacy_scheduler_restores_instance_override_after_error():
    class Upstream:
        def get_next_batch_to_run(self):
            self.get_new_batch_prefill()

    def broken(running_batch):
        raise RuntimeError("prefill failed")

    scheduler = SimpleNamespace(get_new_batch_prefill=broken)
    with pytest.raises(RuntimeError, match="prefill failed"):
        compat.get_next_batch_plan(Upstream, scheduler, "running", None)
    assert scheduler.get_new_batch_prefill is broken


def test_modern_scheduler_plan_is_returned_unchanged():
    expected = object()

    class Upstream:
        def get_next_batch_to_run(self, running_batch, last_batch):
            assert running_batch == "running" and last_batch == "last"
            return expected

        def get_new_batch_prefill(self, running_batch):
            assert running_batch == "running"
            return expected

    scheduler = SimpleNamespace()
    assert (
        compat.get_next_batch_plan(Upstream, scheduler, "running", "last") is expected
    )
    assert compat.get_prefill_plan(Upstream, scheduler, "running") is expected
    assert vars(scheduler) == {}


def test_legacy_prefill_returns_updated_running_batch():
    class Upstream:
        def get_new_batch_prefill(self):
            self.running_batch = "updated"
            return "work"

    plan = compat.get_prefill_plan(Upstream, SimpleNamespace(), "old")
    assert plan == compat.NextBatchPlan("work", "updated")


def test_legacy_extend_range_tracks_both_api_views():
    class Req:
        fill_len = 8
        extend_input_len = 3

    compat.install_req_extend_range(Req)
    req = Req()
    assert req.extend_range == compat.ExtendRange(5, 8)
    req.set_extend_range(2, 7)
    assert (req.fill_len, req.extend_input_len) == (7, 5)
    req.extend_input_len = 4
    assert req.extend_range == compat.ExtendRange(3, 7)
    req.extend_range = req.extend_range._replace(start=4)
    assert (req.fill_len, req.extend_input_len) == (7, 3)
    req.extend_range = None
    assert (req.fill_len, req.extend_input_len) == (0, 0)
    original = Req.set_extend_range
    compat.install_req_extend_range(Req)
    assert Req.set_extend_range is original


def test_modern_req_class_is_not_modified():
    class Req:
        def set_extend_range(self, start, end):
            pass

    before = dict(vars(Req))
    compat.install_req_extend_range(Req)
    assert dict(vars(Req)) == before


@pytest.mark.parametrize("raises", [False, True])
def test_legacy_forward_overrides_are_forwarded_and_restored(raises):
    class Forward:
        @classmethod
        def init_new(cls, batch, model_runner):
            assert batch.capture_hidden_mode == "full"
            assert batch.return_hidden_states_before_norm is True
            assert batch.seq_lens_cpu_cache == "lengths"
            batch.capture_hidden_mode = None
            batch.return_hidden_states_before_norm = False
            batch.seq_lens_cpu_cache = None
            if raises:
                raise RuntimeError("forward failed")
            return "forward-batch"

    compat.install_forward_batch_shim(Forward)
    batch = SimpleNamespace(
        capture_hidden_mode="old",
        return_hidden_states_before_norm=False,
        seq_lens_cpu_cache="old-lengths",
    )
    kwargs = dict(
        capture_hidden_mode="full",
        return_hidden_states_before_norm=True,
        seq_lens_cpu_cache="lengths",
    )
    if raises:
        with pytest.raises(RuntimeError):
            Forward.init_new(batch, None, **kwargs)
    else:
        assert Forward.init_new(batch, None, **kwargs) == "forward-batch"
    assert vars(batch) == dict(
        capture_hidden_mode="old",
        return_hidden_states_before_norm=False,
        seq_lens_cpu_cache="old-lengths",
    )
    with pytest.raises(TypeError, match="unsupported"):
        Forward.init_new(batch, None, unknown=True)
    original = Forward.init_new.__func__
    compat.install_forward_batch_shim(Forward)
    assert Forward.init_new.__func__ is original


def test_modern_forward_class_is_not_modified():
    class Forward:
        @classmethod
        def init_new(
            cls,
            batch,
            model_runner,
            *,
            capture_hidden_mode=None,
            return_hidden_states_before_norm=False,
        ):
            return "native"

    original = Forward.init_new.__func__
    compat.install_forward_batch_shim(Forward)
    assert Forward.init_new.__func__ is original


def test_modern_runtime_does_not_install_legacy_patches(monkeypatch):
    monkeypatch.setattr(compat, "needs_bridge", lambda: False)

    def unexpected():
        raise AssertionError("modern runtime must not install legacy modules")

    monkeypatch.setattr(compat, "install_kv_cache_configurator_shim", unexpected)
    compat.apply_all()
