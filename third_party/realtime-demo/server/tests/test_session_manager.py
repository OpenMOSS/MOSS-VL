"""B2 test: session manager create/get/attach/detach/close + grace expiry.

Run:  <repo>/.venv/bin/python -m server.tests.test_session_manager
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys

from server.config import Settings
from server.schemas import SessionConfig
from server.session.manager import SessionConflict, SessionManager
from server.session.orchestrator import EngineSet
from server.session.state import PHASE_ACTIVE, PHASE_CLOSED, PHASE_DETACHED
from server.tests.fakes import FakeAsrAdapter, FakeVlmSession
from server.voice.tts_session import TtsSession
from server.tests.fakes import FakeTtsEngine


def make_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(), **overrides)


async def make_engines() -> EngineSet:
    tts = TtsSession(FakeTtsEngine(), "test", lambda payload: None)
    return EngineSet(vlm=FakeVlmSession(), asr=FakeAsrAdapter(), tts=tts)


async def test_lifecycle() -> None:
    manager = SessionManager(make_settings(session_grace_seconds=30.0))
    state = await manager.create(SessionConfig(), make_engines)
    assert manager.get(state.session_id) is state
    assert manager.active_count == 1

    # multi-session: capacity now lives in the VLM replica pool; the manager
    # happily hosts concurrent sessions up to the MAX_SESSIONS backstop
    second = await manager.create(SessionConfig(), make_engines)
    assert manager.active_count == 2 and second.session_id != state.session_id
    assert await manager.close(second.session_id) is True
    assert manager.active_count == 1

    # attach / emit / detach / replay
    st, token, close_event, replay = await manager.attach_ws(state.session_id)
    assert st.phase == PHASE_ACTIVE and replay == [] and not close_event.is_set()
    seq1 = state.emit("response.text.delta", response_id="r1", delta="a")
    seq2 = state.emit("response.text.delta", response_id="r1", delta="b")
    state.emit("status", transient=True)  # transient → never replayed
    await manager.detach_ws(state.session_id, token)
    assert st.phase == PHASE_DETACHED and close_event.is_set()

    _, token2, close2, replay2 = await manager.attach_ws(state.session_id, last_seq=seq1)
    assert [i.seq for i in replay2] == [seq2], [i.seq for i in replay2]

    # supersede: a third attach releases the second pump
    _, token3, close3, _ = await manager.attach_ws(state.session_id, last_seq=seq2)
    assert close2.is_set() and not close3.is_set()
    # the superseded socket's detach must not touch the new binding
    await manager.detach_ws(state.session_id, token2)
    assert st.phase == PHASE_ACTIVE and st.ws_token == token3

    vlm = st.orchestrator.engines.vlm
    assert await manager.close(state.session_id) is True
    assert st.phase == PHASE_CLOSED and manager.active_count == 0
    assert vlm.active is False, "close must stop the VLM session"
    try:
        manager.get(state.session_id)
    except KeyError:
        pass
    else:
        raise AssertionError("closed session must be gone")
    print("lifecycle: OK")


async def test_max_sessions_backstop() -> None:
    # explicit MAX_SESSIONS wins over the auto (capacity + headroom) limit
    manager = SessionManager(make_settings(max_sessions=1), capacity=lambda: 4)
    state = await manager.create(SessionConfig(), make_engines)
    try:
        await manager.create(SessionConfig(), make_engines)
    except SessionConflict:
        pass
    else:
        raise AssertionError("create beyond MAX_SESSIONS must raise SessionConflict")
    await manager.close(state.session_id)

    # auto limit = capacity + 8 (headroom for pool-bypassing scripted sessions)
    manager = SessionManager(make_settings(), capacity=lambda: 1)
    states = [await manager.create(SessionConfig(), make_engines) for _ in range(9)]
    try:
        await manager.create(SessionConfig(), make_engines)
    except SessionConflict:
        pass
    else:
        raise AssertionError("auto backstop must cap runaway session counts")
    for st in states:
        await manager.close(st.session_id)
    print("max-sessions backstop: OK")


async def test_grace_expiry() -> None:
    manager = SessionManager(make_settings(session_grace_seconds=1.0))

    # never-attached session GCs after grace
    state = await manager.create(SessionConfig(), make_engines)
    await asyncio.sleep(1.4)
    assert manager.active_count == 0, "unattached session must be GC'd"
    assert state.phase == PHASE_CLOSED

    # attach cancels the timer; detach re-arms it
    state = await manager.create(SessionConfig(), make_engines)
    _, token, _, _ = await manager.attach_ws(state.session_id)
    await asyncio.sleep(1.4)
    assert manager.active_count == 1, "attached session must survive grace"
    await manager.detach_ws(state.session_id, token)
    await asyncio.sleep(1.4)
    assert manager.active_count == 0, "detached session must be GC'd after grace"
    print("grace expiry: OK")


async def test_shutdown() -> None:
    manager = SessionManager(make_settings(session_grace_seconds=30.0))
    state = await manager.create(SessionConfig(), make_engines)
    await manager.aclose()
    assert manager.active_count == 0 and state.phase == PHASE_CLOSED
    try:
        await manager.create(SessionConfig(), make_engines)
    except RuntimeError:
        pass
    else:
        raise AssertionError("create after shutdown must fail")
    print("shutdown: OK")


def main() -> int:
    asyncio.run(test_lifecycle())
    asyncio.run(test_max_sessions_backstop())
    asyncio.run(test_grace_expiry())
    asyncio.run(test_shutdown())
    print("\nSESSION MANAGER TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
