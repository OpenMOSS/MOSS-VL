"""Regression coverage for complete journals, stale prefetch and output bounds."""
import asyncio

from server.memory.rollover import RolloverManager, _Prefetch
from server.memory import inject
from server.memory.pi_client import PiAgentClient
from server.tests.test_rollover import _make_store, _add_turns


class Pi:
    def __init__(self, result=None):
        self.journals = []
        self.result = result or {"summary": "short summary", "pins": []}

    def compact(self, conversation_id, journal):
        self.journals.append(journal)
        return self.result


def test_client_rejects_string_booleans(tmp_path, monkeypatch):
    from server.memory import pi_client
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        monkeypatch.setattr(pi_client, "_post_json", lambda *args:
                            {"retrieve": "false", "query": None, "reason": "bad type"})
        assert PiAgentClient(settings).decide("c", "recent", "current") is None
    finally:
        store.close()


def test_input_budget_rejection_is_not_retried(monkeypatch):
    from server.memory import pi_client
    calls = []
    class Rejected:
        status = 413
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
    def reject(self, *args, **kwargs):
        calls.append(args)
        return Rejected()
    monkeypatch.setattr(pi_client.aiohttp.ClientSession, "post", reject)
    assert pi_client._post_json("http://localhost/compact", {"journal": "x"}, 1) is None
    assert len(calls) == 1


def test_full_journal_and_interrupted_tail_are_serialized(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        _add_turns(store, "c", [(f"question-{i}", f"answer-{i}") for i in range(1002)])
        pi = Pi()
        manager = RolloverManager(settings, store, "c", pi=pi,
            journal_extra=lambda: [("assistant", "new interrupted output")])
        journal, _, _ = manager._collect()
        assert len(journal) == 2004
        assert manager._summarize_pi(journal)[0]
        assert "question-0" in pi.journals[0]
        assert "answer-1001" in pi.journals[0]
        assert pi.journals[0].endswith("new interrupted output [interrupted]")
    finally:
        store.close()


def test_new_turn_invalidates_prefetched_summary(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        _add_turns(store, "c", [("old question", "old answer")])
        pi = Pi()
        manager = RolloverManager(settings, store, "c", pi=pi)
        manager._prefetch = _Prefetch(status="ready",
            result=(manager._collect(), "stale summary", []))
        _add_turns(store, "c", [("latest correction FM2", "confirmed")])
        messages, _, _ = asyncio.run(manager.build_prefix())
        assert "latest correction FM2" in pi.journals[0]
        assert "stale summary" not in messages[0]["content"]
    finally:
        store.close()


def test_finished_old_worker_cannot_publish_into_new_prefetch(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        _add_turns(store, "c", [("question", "answer")])
        manager = RolloverManager(settings, store, "c", pi=Pi())
        old = _Prefetch()
        manager._prefetch = _Prefetch()
        manager._prefetch_worker(old)
        assert manager._prefetch.status == "running"
        assert manager._prefetch.result is None
    finally:
        store.close()


def test_invalid_summary_types_and_lengths_degrade(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        _add_turns(store, "c", [("question", "answer")])
        for output in [{"summary": {}, "pins": []}, {"summary": "x" * 201, "pins": []},
                       {"summary": "ok", "pins": [False]}, {"summary": "ok", "pins": ["x"] * 17}]:
            manager = RolloverManager(settings, store, "c", pi=Pi(output))
            assert manager._summarize_pi(manager._collect()[0]) == (None, [])
    finally:
        store.close()


def test_generated_pins_share_prefix_layer_budget(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="pi")
    try:
        manager = RolloverManager(settings, store, "c")
        messages, _, _ = manager._assemble(None, pins=[f"pin-{i}-" + "x" * 240 for i in range(16)])
        pinned = messages[0]["content"].split("置顶记忆 / Pinned:\n", 1)[1].split("\n\n", 1)[0]
        assert inject.estimate_tokens(pinned) <= 245
    finally:
        store.close()


def test_offline_compaction_covers_late_qa(tmp_path):
    settings, store = _make_store(str(tmp_path), MEMORY_SUMMARY_PROVIDER="offline")
    class Plane:
        prompts = []
        def is_loaded(self):
            return True
        async def generate_stream(self, req):
            self.prompts.append(req.messages[0].content)
            yield "short summary"
    try:
        pairs = [(f"question-{i} " + "word " * 700, f"answer-{i}") for i in range(5)]
        _add_turns(store, "c", pairs)
        plane = Plane()
        manager = RolloverManager(settings, store, "c", plane=plane)
        assert asyncio.run(manager._summarize(manager._collect()[0]))
        assert len(plane.prompts) > 1
        for question, answer in pairs:
            assert any(question.strip() in prompt and answer in prompt for prompt in plane.prompts)
    finally:
        store.close()
