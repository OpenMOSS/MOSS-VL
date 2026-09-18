import queue
import threading

import pytest

from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.mark.parametrize("entry", [True, False])
def test_external_abort_never_mutates_or_releases_kv_on_listener_thread(entry):
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler._scheduler_thread_id = threading.get_ident()
    scheduler._running = True
    scheduler.is_entry_rank = entry
    scheduler.inbox = queue.Queue()
    # No other scheduler state is initialized: touching the release path fails.
    errors = []

    def listener():
        try:
            scheduler.abort("r", defer_running_cleanup=False)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=listener)
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and not errors
    assert scheduler.inbox.qsize() == int(entry)
    if entry:
        message = scheduler.inbox.get_nowait()
        assert message.type == "abort"
        assert message.data == {"defer_running_cleanup": False}


def test_scheduler_consumes_abort_in_order_before_next_batch():
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._aborted_request_ids = set()
    calls = []
    scheduler.abort = lambda rid, **kwargs: (
        calls.append((rid, kwargs)),
        scheduler._aborted_request_ids.add(rid),
    )
    scheduler._on_request_update = lambda *args: pytest.fail(
        "update after abort must be discarded"
    )
    scheduler._recv_scheduler_messages = lambda: [
        IncomingMessage("r", "abort", {"defer_running_cleanup": False}),
        IncomingMessage("r", "request_update", {}),
        IncomingMessage("r", "abort", {}),
    ]
    assert scheduler.recv_requests() == []
    assert calls == [("r", {"defer_running_cleanup": False})]
