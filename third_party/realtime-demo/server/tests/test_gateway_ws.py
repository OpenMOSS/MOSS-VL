"""Gateway plane WSS tests (GATEWAY_PLAN.md §2-P1 acceptance).

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_gateway_ws.py -q

Drives WS /v1/realtime end-to-end against the harness from test_gateway_rest:
token gate (missing/bogus/expired), the attach + verbatim passthrough loop
(configure → configured/ready → two-phase frame → delta), the 1009 oversize
frame policy, double-attach 1008, client-disconnect teardown, omni-death 1011,
and the never-attached janitor GC.
"""
from __future__ import annotations

import asyncio
import json
import sys

import websockets

from server.gateway.pool import DOWN, READY
from server.tests.test_gateway_rest import (
    GatewayFakeOmni,
    rest,
    start_gateway,
    wait_until,
)
from server.tests.test_sglang_omni_adapter import JPEG


async def recv_json(ws, timeout: float = 5.0) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout)
    assert isinstance(raw, str), f"expected a text event, got {type(raw)}"
    return json.loads(raw)


async def expect_close(ws, code: int, timeout: float = 5.0) -> None:
    """Drain until the peer closes; assert the close code."""
    try:
        while True:
            await asyncio.wait_for(ws.recv(), timeout)
    except websockets.exceptions.ConnectionClosed as exc:
        assert exc.rcvd is not None and exc.rcvd.code == code, \
            f"want close {code}, got {exc.rcvd and exc.rcvd.code}"
        return
    raise AssertionError("connection never closed")


# ---------------------------------------------------------------- token gate


async def _test_ws_token_gate() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, gateway_ws_token_ttl_s=0.3)
    try:
        # missing / bogus token → error{ws_token_invalid} + 1008
        for url in (f"{rig.ws_base}/v1/realtime",
                    f"{rig.ws_base}/v1/realtime?ws_token=nope"):
            async with websockets.connect(url) as ws:
                err = await recv_json(ws)
                assert err["type"] == "error" and err["code"] == "ws_token_invalid", err
                await expect_close(ws, 1008)

        # expired token → error{ws_token_expired} + 1008
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        await asyncio.sleep(0.5)  # ttl 0.3s
        async with websockets.connect(
                f"{rig.ws_base}/v1/realtime?ws_token={body['ws_token']}") as ws:
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "ws_token_expired", err
            await expect_close(ws, 1008)
        # the never-attached session itself is still alive until the janitor/DELETE
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{body['session_id']}")
        assert code == 200, (code, snap)
        await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{body['session_id']}")
        print("WS token gate (invalid/expired → error + 1008): OK")
    finally:
        await rig.stop()
        fake.close()


def test_ws_token_gate() -> None:
    asyncio.run(_test_ws_token_gate())


# ---------------------------------------------------------------- happy path


async def _open_ready_session(rig, fake):
    """REST create → WSS attach → configure → ready. Returns (sid, ws)."""
    code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
    assert code == 201, (code, body)
    ws = await websockets.connect(f"{rig.ws_base}/v1/realtime?ws_token={body['ws_token']}")
    created = await recv_json(ws)
    assert created["type"] == "session.created", created
    assert created["session_id"] == "fake-omni-session"  # omni's id, verbatim
    await ws.send(json.dumps({"type": "session.configure", "max_new_tokens": 64}))
    configured = await recv_json(ws)
    ready = await recv_json(ws)
    assert configured["type"] == "session.configured", configured
    assert ready["type"] == "session.ready", ready
    return body["session_id"], ws


async def _test_ws_attach_passthrough() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            # configure crossed verbatim
            assert fake.configure_payload is not None
            assert fake.configure_payload["max_new_tokens"] == 64

            # two-phase frame upload: metadata → ready → binary → accepted/processed
            await ws.send(json.dumps({
                "type": "input.frame", "seq_no": 0, "timestamp": 1.0,
                "mime_type": "image/jpeg"}))
            frame_ready = await recv_json(ws)
            assert frame_ready["type"] == "input.frame.ready" and frame_ready["seq_no"] == 0
            await ws.send(JPEG)
            accepted = await recv_json(ws)
            processed = await recv_json(ws)
            assert accepted["type"] == "input.frame.accepted", accepted
            assert processed["type"] == "input.frame.processed", processed
            assert fake.binaries == [JPEG], "binary frame must cross verbatim"

            # downstream delta crosses byte-for-byte (pass_raw passthrough)
            want = {"type": "response.text.delta", "delta": "你好", "turn_id": 0}
            fake.send_delta("你好", turn_id=0)
            raw = await asyncio.wait_for(ws.recv(), 5.0)
            assert raw == json.dumps(want), (raw, json.dumps(want))

            # state tracker followed along on its own parsed copy
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] == "streaming", snap
            assert snap["frames_accepted"] == 1 and snap["text_deltas"] == 1
            assert snap["text_chars"] == 2 and snap["turn_id"] == 0

            # prompt → accepted/processed; silence → parked
            await ws.send(json.dumps({
                "type": "input.prompt", "seq_no": 1, "prompt": "看到了什么？"}))
            pa = await recv_json(ws)
            pp = await recv_json(ws)
            assert pa["type"] == "input.prompt.accepted", pa
            assert pp["type"] == "input.prompt.processed", pp
            fake.send_silence(turn_id=0)
            silence = await recv_json(ws)
            assert silence["type"] == "response.turn.silence", silence
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert snap["status"] == "parked" and snap["prompts"] == 1, snap
        finally:
            await ws.close()

        # client disconnect destroys the session and frees the replica slot
        assert await wait_until(lambda: rig.registry.get(sid) is None), \
            "session must be destroyed on client disconnect"
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY)
        code, _ = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 404
        print("WS attach + two-phase frame + verbatim delta + disconnect teardown: OK")
    finally:
        await rig.stop()
        fake.close()


def test_ws_attach_passthrough() -> None:
    asyncio.run(_test_ws_attach_passthrough())


# ---------------------------------------------------------------- policies


async def _test_ws_oversize_frame() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, gateway_max_frame_bytes=64)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await ws.send(json.dumps({
                "type": "input.frame", "seq_no": 0, "timestamp": 1.0,
                "mime_type": "image/jpeg"}))
            frame_ready = await recv_json(ws)
            assert frame_ready["type"] == "input.frame.ready"
            await ws.send(b"\x00" * 100)  # 100B > 64B cap
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "invalid_request", err
            assert err["message"] == "frame exceeds max_frame_bytes", err
            await expect_close(ws, 1009)
        finally:
            await ws.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY)
        print("WS oversize frame → error{invalid_request} + 1009: OK")
    finally:
        await rig.stop()
        fake.close()


def test_ws_oversize_frame() -> None:
    asyncio.run(_test_ws_oversize_frame())


async def _test_ws_double_attach() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws1 = await _open_ready_session(rig, fake)
        try:
            # a second token for the same session (reset is the only REST way;
            # here we mint directly to keep ws1 attached)
            token2 = rig.tokens.mint(sid)
            async with websockets.connect(
                    f"{rig.ws_base}/v1/realtime?ws_token={token2}") as ws2:
                err = await recv_json(ws2)
                assert err["type"] == "error" and err["code"] == "session_already_attached", err
                await expect_close(ws2, 1008)
            # the first socket is untouched
            fake.send_delta("仍在", turn_id=0)
            raw = await asyncio.wait_for(ws1.recv(), 5.0)
            assert json.loads(raw)["delta"] == "仍在"
        finally:
            await ws1.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        print("WS double attach → error + 1008, first socket unaffected: OK")
    finally:
        await rig.stop()
        fake.close()


def test_ws_double_attach() -> None:
    asyncio.run(_test_ws_double_attach())


async def _test_ws_omni_death() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        # downstream transport dies → client gets 1011, slot is quarantined
        asyncio.run_coroutine_threadsafe(fake._ws.close(), fake._loop).result(timeout=5)
        await expect_close(ws, 1011)
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        assert await wait_until(lambda: rig.pool.replicas[0].state == DOWN), \
            "transport-dead release must quarantine the replica"
        print("WS omni death → client 1011 + session destroyed + replica DOWN: OK")
    finally:
        await rig.stop()
        fake.close()


def test_ws_omni_death() -> None:
    asyncio.run(_test_ws_omni_death())


# ---------------------------------------------------------------- janitor


async def _test_janitor_attach_timeout() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, gateway_attach_timeout_s=0.6)
    try:
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        sid = body["session_id"]
        assert rig.registry.get(sid) is not None
        # never attach → the janitor destroys it and frees the slot
        assert await wait_until(lambda: rig.registry.get(sid) is None, timeout=8.0), \
            "janitor must GC the never-attached session"
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY)
        code, _ = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 404
        print("janitor attach-timeout GC: OK")
    finally:
        await rig.stop()
        fake.close()


def test_janitor_attach_timeout() -> None:
    asyncio.run(_test_janitor_attach_timeout())


def main() -> int:
    asyncio.run(_test_ws_token_gate())
    asyncio.run(_test_ws_attach_passthrough())
    asyncio.run(_test_ws_oversize_frame())
    asyncio.run(_test_ws_double_attach())
    asyncio.run(_test_ws_omni_death())
    asyncio.run(_test_janitor_attach_timeout())
    print("\nGATEWAY WS TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
