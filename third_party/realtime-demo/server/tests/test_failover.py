"""Failure relay (GATEWAY_PLAN P2): a dead VLM transport gets ONE memory-prefix
re-seat attempt before the terminal vlm_stopped state.

    .venv/bin/python -m server.tests.test_failover

Hermetic like test_rollover.py: tmp DATA_DIR, fallback embedders via empty model
paths, env overrides + `config_mod._settings = None`. No GPU, no network.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time

from .. import config as config_mod
from ..config import Settings
from ..memory.rollover import RolloverManager
from ..memory.session import MemorySession
from ..memory.store import MemoryStore
from ..memory.writer import MemoryWriter
from ..schemas import SessionConfig
from ..session.orchestrator import EngineSet, Orchestrator
from ..session.state import SessionState
from .fakes import FakeVlmSession


def _settings(tmp: str, **over) -> Settings:
    os.environ["DATA_DIR"] = tmp
    os.environ["MEMORY_ENABLED"] = "1"
    os.environ["MEMORY_EMBED_TEXT_MODEL"] = ""
    os.environ["MEMORY_EMBED_IMAGE_MODEL"] = ""
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


async def _wait_for(cond, timeout: float = 10.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.05)
    return False


def _errors(state: SessionState, code: str) -> list:
    return [item for item in state.replay
            if '"error"' in item.text and f'"{code}"' in item.text]


def _make_stack(tmp: str, factory, *, memory_on: bool = True):
    s = _settings(tmp, MEMORY_SUMMARY_PROVIDER="none")
    store = MemoryStore(s)
    writer = MemoryWriter(s, store)
    writer.start()
    memory = MemorySession("c1", s, store, writer) if memory_on else None
    rollover = (RolloverManager(s, store, "c1", plane=None,
                                base_system_prompt="BASE_PROMPT",
                                lang_getter=lambda: "zh")
                if memory_on else None)
    old = FakeVlmSession()
    state = SessionState(session_id="c1", config=SessionConfig(), replay_size=50,
                         out_queue_size=100)
    orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), s,
                        memory=memory, rollover=rollover, reseat_factory=factory)
    orch.start()
    return s, store, writer, memory, old, orch


async def _relay_recovers() -> None:
    """VLM dies with memory/rollover available → one relay re-seat revives the
    session: factory gets the memory prefix, engines.vlm swapped, drain loop
    restarted, _vlm_dead reset, latest frame re-pushed, no terminal error."""
    with tempfile.TemporaryDirectory() as tmp:
        made = []
        factory_calls = []

        async def reseat_factory(*, prompt="", system_prompt=None, prefill_messages=None):
            factory_calls.append({"prefill_messages": prefill_messages})
            sess = FakeVlmSession()
            made.append(sess)
            return sess

        s, store, writer, memory, old, orch = _make_stack(tmp, reseat_factory)
        try:
            memory.note_user_turn("我买了一台尼康 FM2,花了 2000 元。")
            memory.note_assistant_turn("我会帮你查行情。")
            writer.drain()
            await orch.push_frame(_jpeg(), 1.0)
            first_drain = orch._vlm_drain_task

            old.active = False  # transport death: the next poll reports inactive
            assert await _wait_for(lambda: made and not orch._vlm_dead), \
                "failure relay never revived the session"

            call = factory_calls[0]
            prefill = json.loads(call["prefill_messages"])
            assert prefill[0]["role"] == "system"
            assert "BASE_PROMPT" in prefill[0]["content"]
            tail = prefill[1:]
            assert tail and any("尼康 FM2" in m["content"] for m in tail), prefill

            new = made[0]
            assert orch.engines.vlm is new
            assert not old.active
            assert not orch._closed, "the client session must survive the relay"
            assert not _errors(orch.state, "vlm_stopped"), \
                "no terminal error while the relay succeeds"
            assert new.frames, "the latest frame must be re-pushed"
            assert orch._vlm_drain_task is not first_drain and not orch._vlm_drain_task.done()
            assert first_drain.done() or first_drain.cancelled()

            # model deltas flow again on the new engine
            new.narrate("接力成功,继续。")
            assert await _wait_for(
                lambda: any("接力成功" in item.text for item in orch.state.replay)), \
                "the revived drain loop never routed model text"
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  relay recovers ok")


async def _relay_failure_is_terminal() -> None:
    """Relay factory raises → terminal vlm_stopped, exactly one attempt, no
    recursion (the reseat failure path must not schedule another relay)."""
    with tempfile.TemporaryDirectory() as tmp:
        calls = []

        async def reseat_factory(*, prompt="", system_prompt=None, prefill_messages=None):
            calls.append(1)
            raise RuntimeError("no replica available anywhere")

        s, store, writer, memory, old, orch = _make_stack(tmp, reseat_factory)
        try:
            old.active = False
            assert await _wait_for(lambda: bool(_errors(orch.state, "vlm_stopped"))), \
                "failed relay never reached the terminal state"
            await asyncio.sleep(0.5)  # give any (buggy) recursion a chance to fire
            assert orch._vlm_dead
            assert len(calls) == 1, "the relay must be attempted exactly once"
            assert orch._failure_relay_attempted
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  relay failure terminal ok")


async def _no_memory_goes_terminal() -> None:
    """Memory disabled → rollover/relay inert: death goes straight to the
    terminal state, the factory is never called."""
    with tempfile.TemporaryDirectory() as tmp:
        calls = []

        async def reseat_factory(*, prompt="", system_prompt=None, prefill_messages=None):
            calls.append(1)
            return FakeVlmSession()

        s, store, writer, memory, old, orch = _make_stack(tmp, reseat_factory,
                                                          memory_on=False)
        try:
            old.active = False
            assert await _wait_for(lambda: bool(_errors(orch.state, "vlm_stopped"))), \
                "death without memory never reached the terminal state"
            assert orch._vlm_dead
            assert not calls, "no relay attempt without memory"
            assert not orch._failure_relay_attempted
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  no memory terminal ok")


async def _relay_rearms_after_success() -> None:
    """A successful relay resets the one-shot flag: a SECOND transport death
    gets its own relay attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        made = []

        async def reseat_factory(*, prompt="", system_prompt=None, prefill_messages=None):
            sess = FakeVlmSession()
            made.append(sess)
            return sess

        s, store, writer, memory, old, orch = _make_stack(tmp, reseat_factory)
        try:
            memory.note_user_turn("第一轮的上下文。")
            writer.drain()

            old.active = False
            assert await _wait_for(lambda: made and not orch._vlm_dead), \
                "first relay never revived the session"
            assert not orch._failure_relay_attempted, "success must re-arm the relay"

            made[0].active = False  # the replacement engine dies too
            assert await _wait_for(lambda: len(made) >= 2 and not orch._vlm_dead), \
                "second relay never revived the session"
            assert orch.engines.vlm is made[1]
            assert not _errors(orch.state, "vlm_stopped")
        finally:
            await orch.close()
            writer.stop()
            store.close()
    print("  relay re-arms ok")


def test_failure_relay() -> None:
    asyncio.run(_relay_recovers())
    asyncio.run(_relay_failure_is_terminal())
    asyncio.run(_no_memory_goes_terminal())
    asyncio.run(_relay_rearms_after_success())


def main() -> None:
    test_failure_relay()
    print("failover: all checks passed")


if __name__ == "__main__":
    main()
