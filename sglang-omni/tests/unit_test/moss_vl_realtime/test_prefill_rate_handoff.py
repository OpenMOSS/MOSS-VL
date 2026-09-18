"""Exercise the real SGLang planner at the prefill/rate-limit boundary."""

from types import SimpleNamespace

import pytest
from sglang.srt.managers.schedule_batch import NextBatchPlan

from sglang_omni.models.moss_vl_realtime.runtime_state import MossVLRealtimeRuntimeState
from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class Request:
    def __init__(self, rid, done=False):
        self.rid = rid
        self.done = done
        self._moss_vl_realtime_state = MossVLRealtimeRuntimeState(
            request_id=rid,
            session_id=rid,
            max_tokens_per_turn=4,
            next_decode_not_before=10.25,
        )
        self._omni_data = SimpleNamespace(runtime_state=self._moss_vl_realtime_state)
        self.extend_range = SimpleNamespace(end=10)
        self.prefix_indices = [1]

    def finished(self):
        return self.done


class Batch:
    def __init__(self, reqs, extend=False, chunked=None):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.chunked_req = chunked
        self.batch_is_full = False
        self.is_prefill_only = False

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, chunked_req_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in chunked_req_to_exclude
        ]

    def merge_batch(self, other):
        assert not {r.rid for r in self.reqs} & {
            r.rid for r in other.reqs
        }, "duplicate handoff"
        self.reqs.extend(other.reqs)


@pytest.fixture
def scheduler(monkeypatch):
    s = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    s.running_batch = Batch([Request("running")])
    s.last_batch = Batch([Request("prefilled")], extend=True)
    s._realtime_extend_batch = None
    s._async_pending = None
    s.parked_reqs = {}
    s.waiting_queue = []
    s.chunked_req = None
    s.dllm_config = None
    s.enable_fpm = False
    s.enable_hisparse = False
    s.require_mlp_sync = False
    s.tp_size = 1
    s.prepares = 0
    s.stashes = []
    s.order = []
    s._expire_parked_requests = lambda: None
    s._abort_retracted_realtime_requests = lambda: None
    s._evaluate_frame_window = lambda: s.order.append(
        ("window", [r.rid for r in s.running_batch.reqs])
    )
    s._materialize_realtime_extensions = lambda: None
    s._preempt_decode_memory_pressure = lambda: s.order.append(
        ("pressure", [r.rid for r in s.running_batch.reqs])
    )
    s.process_pending_chunked_abort = lambda: None
    s._abort_on_waiting_timeout = lambda: None
    s._abort_on_running_timeout = lambda batch: None
    s.stash_chunked_request = lambda req: s.stashes.append(req.rid)
    s.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda value, **kw: value
    )
    s.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda value, **kw: value
    )

    def prepare(batch):
        s.prepares += 1
        return batch

    s.update_running_batch = prepare
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.set_schedule_time_batch", lambda batch: None
    )
    monkeypatch.setattr(
        OmniScheduler,
        "get_new_batch_prefill",
        lambda self, batch: NextBatchPlan(batch_to_run=None, running_batch=batch),
    )
    return s


def ids(batch):
    return [r.rid for r in batch.reqs]


def test_prefill_survives_repeated_rate_limited_idle(scheduler, monkeypatch):
    s = scheduler
    for _ in range(5):
        batch = s.get_next_batch_to_run()
        assert batch is None
        s.cur_batch = batch
        s.last_batch = batch
        assert ids(s.running_batch) == ["running", "prefilled"]
        assert s.prepares == 0
    assert s.order[0] == ("window", ["running", "prefilled"])
    monkeypatch.setattr("time.monotonic", lambda: 10.25)
    assert ids(s.get_next_batch_to_run()) == ["running", "prefilled"]
    assert s.prepares == 1


def test_new_prefill_is_not_delayed_by_existing_decode_rate(scheduler, monkeypatch):
    s = scheduler
    fresh = Batch([Request("new")], extend=True)
    monkeypatch.setattr(
        OmniScheduler,
        "get_new_batch_prefill",
        lambda self, batch: NextBatchPlan(batch_to_run=fresh, running_batch=batch),
    )
    assert s.get_next_batch_to_run() is fresh
    assert ids(s.running_batch) == ["running", "prefilled"]
    assert s.prepares == 0


def test_incremental_input_sees_just_prefilled_request(scheduler):
    s = scheduler
    selected = []

    def extend():
        selected.extend(r for r in s.running_batch.reqs if r.rid == "prefilled")
        s.running_batch.reqs = [r for r in s.running_batch.reqs if r.rid != "prefilled"]
        return Batch(selected, extend=True)

    s._materialize_realtime_extensions = extend
    assert ids(s.get_next_batch_to_run()) == ["prefilled"]
    assert ids(s.running_batch) == ["running"]
    assert s.prepares == 0


def test_finished_and_chunked_requests_keep_upstream_filtering(scheduler):
    s = scheduler
    chunked = Request("chunked")
    s.chunked_req = chunked
    s.last_batch.reqs.extend([Request("finished", done=True), chunked])
    assert s.get_next_batch_to_run() is None
    assert ids(s.running_batch) == ["running", "prefilled"]
    assert s.chunked_req is chunked
    assert s.stashes == ["chunked"]
    assert s.prepares == 0


def test_abort_can_find_request_after_deferred_handoff(scheduler, monkeypatch):
    s = scheduler
    assert s.get_next_batch_to_run() is None
    # Exercise the real request lookup used by abort/update routing.
    s.cur_batch = None
    s.last_batch = None
    found = s._find_request_data("prefilled")
    assert found is not None
    assert found.runtime_state.request_id == "prefilled"


@pytest.mark.parametrize(
    "loop_name", ["_event_loop_normal", "_event_loop_async_decode"]
)
def test_both_event_loops_keep_deferred_prefill_owned(scheduler, loop_name):
    s = scheduler
    s._running = True
    s._engine_paused = False
    s._model_runner = None
    s._process_admin_requests = lambda: None
    s.recv_requests = lambda: []
    s._take_deferred_request_payloads = lambda: []
    s.process_input_requests = lambda reqs: None
    s.self_check_during_idle = lambda: None
    rounds = []

    def idle():
        rounds.append(ids(s.running_batch))
        if len(rounds) == 3:
            s._running = False

    s._sleep_during_idle = idle
    getattr(s, loop_name)()
    assert rounds == [["running", "prefilled"]] * 3
    assert s.last_batch is None
    assert s.prepares == 0


def test_unexpected_planning_error_is_not_swallowed(scheduler, monkeypatch):
    def fail(self, batch):
        raise RuntimeError("planning failure")

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", fail)
    with pytest.raises(RuntimeError, match="planning failure"):
        scheduler.get_next_batch_to_run()


@pytest.mark.parametrize("rank,local_time", [(0, 10.0), (1, 11.0)])
def test_tp_followers_use_shared_rate_decision_after_handoff(
    scheduler, monkeypatch, rank, local_time
):
    s = scheduler
    s.tp_size = 2
    s.tp_group = SimpleNamespace(rank=rank, ranks=[0, 1])
    s.tp_cpu_group = object()
    monkeypatch.setattr("time.monotonic", lambda: local_time)
    monkeypatch.setattr(
        "sglang_omni.models.moss_vl_realtime.scheduler.broadcast_pyobj",
        lambda *args, **kwargs: [True],
    )
    assert s.get_next_batch_to_run() is None
    assert ids(s.running_batch) == ["running", "prefilled"]
    assert s.prepares == 0


def test_async_update_barrier_precedes_handoff_and_eviction(scheduler):
    s = scheduler
    s._async_pending = object()
    s._has_pending_realtime_events = lambda: True

    def resolve():
        s.order.append(("resolve", []))
        s._async_pending = None

    s._resolve_pending_async = resolve
    assert s.get_next_batch_to_run() is None
    assert s.order[:2] == [("resolve", []), ("window", ["running", "prefilled"])]
