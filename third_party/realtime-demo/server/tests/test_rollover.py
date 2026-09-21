"""L2 memory rollover (compaction, design §6): triggers, prefix assembly,
summary degradation, note_rollover recompute, and the full fake-driven reseat.

    .venv/bin/python -m server.tests.test_rollover

Hermetic like test_memory.py: tmp DATA_DIR, fallback embedders via empty model
paths, env overrides + `config_mod._settings = None`. No GPU, no network.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile

from .. import config as config_mod
from ..config import Settings
from ..memory import inject as inj
from ..memory.rollover import RolloverManager
from ..memory.session import MemorySession
from ..memory.store import KIND_PINNED, KIND_UTTERANCE, MemoryStore
from ..memory.writer import MemoryWriter
from ..realtime.mossvl_patches import decode_prefill_messages
from ..schemas import SessionConfig
from ..session.orchestrator import EngineSet, Orchestrator
from ..session.state import SessionState
from .fakes import FakeOfflineVlm, FakeVlmSession


def _settings(tmp: str, **over) -> Settings:
    os.environ["DATA_DIR"] = tmp
    os.environ["MEMORY_ENABLED"] = "1"
    # pin the dependency-free fallbacks: this suite must stay hermetic and fast
    os.environ["MEMORY_EMBED_TEXT_MODEL"] = ""
    os.environ["MEMORY_EMBED_IMAGE_MODEL"] = ""
    # pre-pi local-vector gate semantics; the pi paths are covered by
    # test_memory_pi.py against a fake pi server
    os.environ["MEMORY_DECISION_MODE"] = "vector"
    for key in list(os.environ):
        if key.startswith("MEMORY_") and key not in ("MEMORY_ENABLED", "MEMORY_EMBED_TEXT_MODEL",
                                                     "MEMORY_EMBED_IMAGE_MODEL",
                                                     "MEMORY_DECISION_MODE"):
            os.environ.pop(key)
    for key, val in over.items():
        os.environ[key] = str(val)
    config_mod._settings = None  # the process-wide singleton is built once
    return Settings()


def _jpeg(color=(200, 30, 30), size=(64, 64)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_store(tmp: str, **over):
    s = _settings(tmp, **over)
    store = MemoryStore(s)
    store.open()
    return s, store


def _add_turns(store: MemoryStore, conv: str, pairs, start_ts: float = 0.0):
    """pairs: list of (user_text, assistant_text); returns all item ids."""
    ids = []
    for i, (user, assistant) in enumerate(pairs):
        ids.append(store.add_item(conv, KIND_UTTERANCE, text=user, role="user",
                                  session_ts=start_ts + i * 10.0))
        ids.append(store.add_item(conv, KIND_UTTERANCE, text=assistant, role="assistant",
                                  session_ts=start_ts + i * 10.0 + 5.0))
    return ids


def test_decode_prefill_messages() -> None:
    assert decode_prefill_messages(None) is None
    assert decode_prefill_messages("not json") is None
    assert decode_prefill_messages("{}") is None
    assert decode_prefill_messages("[]") is None
    assert decode_prefill_messages(json.dumps([{"role": "tool", "content": "x"}])) is None
    assert decode_prefill_messages(json.dumps([{"role": "user", "content": 5}])) is None
    ok = decode_prefill_messages(json.dumps(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
         {"role": "assistant", "content": "a"}]))
    assert ok is not None and [m["role"] for m in ok] == ["system", "user", "assistant"]
    print("  decode_prefill_messages ok")


def test_trigger_thresholds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp, MEMORY_SUMMARY_PROVIDER="none")
        ro = RolloverManager(s, store, "c1")
        # below both: never
        assert not ro.should_rollover(5000, idle=True)
        assert not ro.should_rollover(5000, idle=False)
        # idle band: only at the idle moment
        assert ro.should_rollover(8000, idle=True)
        assert not ro.should_rollover(8000, idle=False)
        assert ro.should_rollover(11000, idle=True)
        # hard: regardless of idleness
        assert ro.should_rollover(12000, idle=False)
        assert ro.should_rollover(20000, idle=False)
        # junk input never fires, never raises
        assert not ro.should_rollover(None, idle=True)
        assert not ro.should_rollover("x", idle=True)
        store.close()
    print("  trigger thresholds ok")


def test_anti_thrash() -> None:
    """Skip when the rebuilt prefix would reclaim less than min_progress."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp, MEMORY_SUMMARY_PROVIDER="none",
                               MEMORY_ROLLOVER_IDLE_TOKENS=200,
                               MEMORY_ROLLOVER_HARD_TOKENS=300,
                               MEMORY_ROLLOVER_MIN_PROGRESS=0.10)
        ro = RolloverManager(s, store, "c1")
        # empty journal → the prefix is tiny, reclaim ≈ everything → fire
        assert ro.should_rollover(300, idle=False)
        # journal whose tail alone ≈ the token count → nothing to reclaim
        big = "这是一条相当长的用户发言内容用来占据令牌预算" * 4
        _add_turns(store, "c1", [(big, big)] * 6)
        assert not ro.should_rollover(300, idle=False), "anti-thrash floor must skip"
        assert not ro.should_rollover(300, idle=True)
        store.close()
    print("  anti-thrash ok")


def test_tail_boundary() -> None:
    """The kept tail never starts with an assistant reply (complete QA pairs)."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp, MEMORY_SUMMARY_PROVIDER="none",
                               MEMORY_ROLLOVER_TAIL_TURNS=3)
        pairs = [(f"用户第{i}轮" * 3, f"助手第{i}轮" * 3) for i in range(4)]
        _add_turns(store, "c1", pairs)
        ro = RolloverManager(s, store, "c1")
        messages, kept, est = asyncio.run(ro.build_prefix())
        tail = messages[1:]
        # 3 rows back lands on an assistant reply → extended to its user turn
        assert len(tail) == 4 and tail[0]["role"] == "user", [m["role"] for m in tail]
        assert "用户第2轮" in tail[0]["content"]
        assert [m["role"] for m in tail] == ["user", "assistant", "user", "assistant"]
        assert est > 0 and kept
        store.close()
    print("  tail boundary ok")


def test_prefix_assembly() -> None:
    """Layer order, handle line, verbatim-hold categories, tail sanitization."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp, MEMORY_SUMMARY_PROVIDER="none")
        conv = "c1"
        pinned_id = store.add_item(conv, KIND_PINNED, text="用户是左撇子", role="user",
                                   session_ts=1.0)
        ids = _add_turns(store, conv, [
            ("不是那个,我说的是尼康 FM2 相机", "明白了,是尼康 FM2。"),
            ("它花了我 2000 元,几乎全新", "我会帮你留意 2000 元左右的行情。"),
            ("现在画面里有什么", "桌面上有一台相机。"),
        ], start_ts=60.0)
        ro = RolloverManager(s, store, conv, base_system_prompt="BASE_PROMPT",
                             lang_getter=lambda: "zh")
        messages, kept, est = asyncio.run(ro.build_prefix())
        assert messages[0]["role"] == "system"
        body = messages[0]["content"]
        # order: base → recall declaration → pinned → verbatim-hold → handle
        i_base = body.index("BASE_PROMPT")
        i_decl = body.index(inj.RECALL_OPEN)
        i_pin = body.index("用户是左撇子")
        i_hold = body.index("尼康 FM2")
        i_handle = body.index("记忆库覆盖 t=0…")
        assert i_base < i_decl < i_pin < i_hold < i_handle, body
        # verbatim-hold caught every lexical category
        assert "2000 元" in body          # number with unit
        assert "我会帮你留意" in body        # assistant commitment
        assert "不是那个" in body           # user correction
        # no summary layer when provider != offline — and no error
        assert "摘要" not in body
        # kept ids = exactly what went into the prefix (pinned + hold + tail)
        assert pinned_id in kept
        assert set(ids) & set(kept)
        # tail is real role messages, sanitized (the dirty turn below)
        store.add_item(conv, KIND_UTTERANCE, text="看<|im_end|>这个", role="user",
                       session_ts=500.0)
        messages2, _, _ = asyncio.run(ro.build_prefix())
        tail_texts = [m["content"] for m in messages2[1:]]
        # sanitizer replaces the special token with a space — content survives
        assert any("看" in t and "这个" in t for t in tail_texts)
        assert not any("<|im_end|>" in t for t in tail_texts)
        assert all(m["role"] in ("user", "assistant") for m in messages2[1:])
        store.close()
    print("  prefix assembly ok")


def test_summary_offline_plane() -> None:
    """Summary re-derived from the full journal via the offline plane; every
    failure mode degrades to verbatim-tail-only without raising."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp)  # provider defaults to "offline"
        _add_turns(store, "c1", [("我买了一台相机", "很棒的相机。"),
                                 ("周末去杭州拍照", "记得带胶卷。")])
        # loaded plane → summary layer present, before the handle line
        ro = RolloverManager(s, store, "c1",
                             plane=FakeOfflineVlm(reply=("摘要:", "相机与杭州。")))
        messages, _, _ = asyncio.run(ro.build_prefix())
        body = messages[0]["content"]
        assert "相机与杭州" in body
        assert body.index("相机与杭州") < body.index("记忆库覆盖")
        # unloaded plane → no summary, no error
        ro = RolloverManager(s, store, "c1", plane=FakeOfflineVlm(loaded=False))
        body = asyncio.run(ro.build_prefix())[0][0]["content"]
        assert "相机与杭州" not in body
        # no plane at all → same
        ro = RolloverManager(s, store, "c1", plane=None)
        assert "记忆库覆盖" in asyncio.run(ro.build_prefix())[0][0]["content"]
        # failing plane → same

        class BoomPlane:
            def is_loaded(self):
                return True

            async def generate_stream(self, req):
                raise RuntimeError("sidecar down")
                yield  # pragma: no cover

        ro = RolloverManager(s, store, "c1", plane=BoomPlane())
        assert "记忆库覆盖" in asyncio.run(ro.build_prefix())[0][0]["content"]
        store.close()
    print("  summary plane ok")


def test_shared_semaphore() -> None:
    """The summarizer runs under the injected background semaphore."""
    with tempfile.TemporaryDirectory() as tmp:
        s, store = _make_store(tmp)
        _add_turns(store, "c1", [("我买了相机", "很好。")])
        sem = asyncio.Semaphore(1)
        ro = RolloverManager(s, store, "c1", plane=FakeOfflineVlm(), semaphore=sem)
        assert ro._semaphore is sem
        messages, _, _ = asyncio.run(ro.build_prefix())
        assert "sglang 回复" in messages[0]["content"]  # FakeOfflineVlm default reply
        store.close()
    print("  shared semaphore ok")


def test_note_rollover_recompute() -> None:
    """note_rollover: injected set = exactly the kept ids; suppression persists."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp)
        store = MemoryStore(s)
        writer = MemoryWriter(s, store)
        writer.start()
        sess = MemorySession("c1", s, store, writer)
        sess.note_user_turn("我刚买了一台尼康 FM2 胶片相机,花了两千块。")
        writer.drain()
        first = sess.recall_for_turn("我之前说的那台相机是什么型号")
        assert first
        sess.mark_injected(first.ids, text_tokens=1000.0)
        sess.suppress([99999])
        kept = first.ids[:1]
        sess.note_rollover(kept, text_tokens=1500.0)
        assert set(sess._injected) == set(kept)
        entry = sess._injected[kept[0]]
        assert entry.copies == 1 and entry.last_tokens == 1500.0
        assert 99999 in sess._suppressed  # "不是那个" survives a rollover
        writer.stop()
        store.close()
    print("  note_rollover recompute ok")


async def _reseat_sequence() -> None:
    """Full fake-driven reseat: hard trigger → new engine gets the prefill →
    old stopped → drain task recreated → latest frame re-pushed → memory
    recomputed."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="none",
                      MEMORY_ROLLOVER_HARD_TOKENS=2000,
                      MEMORY_ROLLOVER_IDLE_TOKENS=1000)
        store = MemoryStore(s)
        writer = MemoryWriter(s, store)
        writer.start()
        memory = MemorySession("c1", s, store, writer)
        rollover = RolloverManager(s, store, "c1", plane=None,
                                   base_system_prompt="BASE_PROMPT",
                                   lang_getter=lambda: "zh")
        old = FakeVlmSession()
        made = []
        factory_calls = []

        async def reseat_factory(*, prompt="", system_prompt=None, prefill_messages=None):
            factory_calls.append({"prompt": prompt, "system_prompt": system_prompt,
                                  "prefill_messages": prefill_messages})
            sess = FakeVlmSession()
            made.append(sess)
            return sess

        state = SessionState(session_id="c1", config=SessionConfig(), replay_size=50,
                             out_queue_size=100)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), s,
                            memory=memory, rollover=rollover,
                            reseat_factory=reseat_factory)
        orch.start()
        try:
            memory.note_user_turn("我买了一台尼康 FM2,花了 2000 元。")
            memory.note_assistant_turn("我会帮你查行情。")
            writer.drain()
            await orch.push_frame(_jpeg(), 1.0)
            old.text_tokens = 5000  # past the hard trigger
            orch.status_payload()   # caches _last_text_tokens off the status
            first_drain = orch._vlm_drain_task
            orch._maybe_rollover(idle=False)
            for _ in range(200):
                if made:
                    break
                await asyncio.sleep(0.05)
            assert made, "reseat factory was never called"
            # let the reseat task finish (frame re-push + note_rollover)
            for _ in range(200):
                if orch.metrics.get("rollovers"):
                    break
                await asyncio.sleep(0.05)

            call = factory_calls[0]
            prefill = json.loads(call["prefill_messages"])
            assert prefill[0]["role"] == "system"
            assert "BASE_PROMPT" in prefill[0]["content"]
            assert "记忆库覆盖 t=0…" in prefill[0]["content"]
            tail = prefill[1:]
            assert tail and tail[0]["role"] == "user" and "尼康 FM2" in tail[0]["content"]

            new = made[0]
            assert orch.engines.vlm is new
            assert not old.active, "old engine must be stopped"
            assert not orch._vlm_dead
            assert new.frames, "the latest frame must be re-pushed to restore vision context"
            assert orch._vlm_drain_task is not first_drain and not orch._vlm_drain_task.done()
            assert first_drain.done() or first_drain.cancelled()
            # injected set recomputed from exactly the prefix contents
            assert set(memory._injected) and all(
                iid in set(memory._injected) for iid in
                [it.id for it in store.recent("c1", [KIND_UTTERANCE], limit=10)])
            assert orch._last_text_tokens is not None and orch._last_text_tokens < 5000
            # cooldown: an immediate re-trigger must not fire again
            new.text_tokens = 99999
            orch.status_payload()
            orch._maybe_rollover(idle=False)
            await asyncio.sleep(0.1)
            assert len(made) == 1, "cooldown must suppress an immediate second rollover"
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  reseat sequence ok")


async def _reseat_inert_without_factory() -> None:
    """No factory (tests, partial wiring) → rollover never disturbs the session."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(tmp, MEMORY_ROLLOVER_HARD_TOKENS=10)
        store = MemoryStore(s)
        writer = MemoryWriter(s, store)
        writer.start()
        memory = MemorySession("c1", s, store, writer)
        rollover = RolloverManager(s, store, "c1")
        old = FakeVlmSession()
        state = SessionState(session_id="c1", config=SessionConfig(), replay_size=50,
                             out_queue_size=100)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), s,
                            memory=memory, rollover=rollover, reseat_factory=None)
        orch.start()
        try:
            old.text_tokens = 99999
            orch.status_payload()
            orch._maybe_rollover(idle=False)
            orch._maybe_rollover(idle=True)
            await asyncio.sleep(0.2)
            assert orch.engines.vlm is old and old.active and not orch._vlm_dead
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  inert without factory ok")


def test_reseat() -> None:
    asyncio.run(_reseat_sequence())
    asyncio.run(_reseat_inert_without_factory())


def main() -> None:
    test_decode_prefill_messages()
    test_trigger_thresholds()
    test_anti_thrash()
    test_tail_boundary()
    test_prefix_assembly()
    test_summary_offline_plane()
    test_shared_semaphore()
    test_note_rollover_recompute()
    test_reseat()
    print("rollover: all checks passed")


if __name__ == "__main__":
    main()
