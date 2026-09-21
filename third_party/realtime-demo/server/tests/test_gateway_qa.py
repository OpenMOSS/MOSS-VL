"""Gateway QA acceptance tests (评审文档 §8 × GATEWAY_PLAN.md §2-P6).

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_gateway_qa.py -q

Fills the §8 acceptance gaps left by test_gateway_rest/ws/metrics: mid-stream
prompting, response.done/session.done termination, silence-not-an-answer,
malformed JSON, seq/timestamp rejections, delayed-ready backpressure, and the
park-timeout response_failed path. Every docstring names its §8.x item;
docs/qa_acceptance.md holds the full item → test mapping.

Reuses the harness from test_gateway_rest (in-process uvicorn +
GatewayFakeOmni); the fake's scripted abilities (reject_next_input,
ready_delay_s, malformed-JSON invalid_request) live in
test_sglang_omni_adapter.FakeSglangOmniServer.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import websockets

from server.gateway.pool import DOWN, READY
from server.tests.test_gateway_rest import (
    GatewayFakeOmni,
    rest,
    start_gateway,
    wait_until,
)
from server.tests.test_gateway_ws import (
    _open_ready_session,
    expect_close,
    recv_json,
)
from server.tests.test_sglang_omni_adapter import JPEG

JPEG2 = b"\xff\xd8\xff\xe0" + b"\x11" * 32 + b"\xff\xd9"


async def _push_frame(ws, seq_no: int, timestamp: float,
                      payload: bytes = JPEG) -> None:
    """Two-phase upload, asserting the accepted/processed pair."""
    await ws.send(json.dumps({
        "type": "input.frame", "seq_no": seq_no, "timestamp": timestamp,
        "mime_type": "image/jpeg"}))
    ready = await recv_json(ws)
    assert ready["type"] == "input.frame.ready" and ready["seq_no"] == seq_no, ready
    await ws.send(payload)
    accepted = await recv_json(ws)
    processed = await recv_json(ws)
    assert accepted["type"] == "input.frame.accepted" and accepted["seq_no"] == seq_no
    assert processed["type"] == "input.frame.processed" and processed["seq_no"] == seq_no


async def _send_prompt(ws, seq_no: int, text: str) -> None:
    await ws.send(json.dumps({"type": "input.prompt", "seq_no": seq_no, "prompt": text}))
    accepted = await recv_json(ws)
    processed = await recv_json(ws)
    assert accepted["type"] == "input.prompt.accepted" and accepted["seq_no"] == seq_no
    assert processed["type"] == "input.prompt.processed" and processed["seq_no"] == seq_no


# ---------------------------------------------------------------- §8.1 功能


async def _test_qa_8_1_prompt_anytime() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _push_frame(ws, 0, 1.0)  # streaming first …
            await _send_prompt(ws, 1, "现在画面里有什么？")  # … then ask anytime
            want = {"type": "response.text.delta", "delta": "有一只猫", "turn_id": 0}
            fake.send_delta("有一只猫", turn_id=0)
            raw = await asyncio.wait_for(ws.recv(), 5.0)
            assert raw == json.dumps(want), raw  # verbatim, default ensure_ascii
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] == "streaming", snap
            assert snap["frames_accepted"] == 1 and snap["prompts"] == 1, snap
        finally:
            await ws.close()
        print("§8.1 随时提问 (mid-stream input.prompt → delta 透传): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_1_prompt_anytime() -> None:
    """§8.1 随时提问：推帧途中发 input.prompt，accepted/processed 与 delta 透传。"""
    asyncio.run(_test_qa_8_1_prompt_anytime())


async def _test_qa_8_1_done_events() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _send_prompt(ws, 0, "描述一下")
            fake.send_delta("增量一", turn_id=0)
            fake.send_delta("增量二", turn_id=0)
            fake.send_event({"type": "response.done", "finish_reason": "stop", "turn_id": 0})
            fake.send_event({"type": "session.done", "aborted": False})
            got = [json.loads(await asyncio.wait_for(ws.recv(), 5.0)) for _ in range(4)]
            assert [e["type"] for e in got] == [
                "response.text.delta", "response.text.delta",
                "response.done", "session.done"], got
            assert got[2]["finish_reason"] == "stop" and got[3]["aborted"] is False
            # tracker observed the terminal events before any teardown
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] == "done", snap
            assert snap["text_deltas"] == 2, snap
            # omni ends the session server-side → session destroyed, slot freed.
            # A close AFTER session.done is the clean end of the wire protocol:
            # the client gets 1000 (not 1011) and the slot frees WITHOUT
            # quarantine (mid-session deaths still go 1011 + DOWN).
            asyncio.run_coroutine_threadsafe(fake._ws.close(), fake._loop).result(timeout=5)
            await expect_close(ws, 1000)
        finally:
            await ws.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        # clean done → slot returns READY, no DOWN quarantine
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY)
        code, _ = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 404
        print("§8.1 增量+正确结束 (delta…→response.done→session.done 透传): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_1_done_events() -> None:
    """§8.1 增量输出与正确结束：delta 逐条透传，response.done/session.done
    原样到达且状态落 done；omni 关闭后会话销毁、槽位释放。"""
    asyncio.run(_test_qa_8_1_done_events())


async def _test_qa_8_1_silence_not_an_answer() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _send_prompt(ws, 0, "看到了吗？")
            fake.send_silence(turn_id=0)
            silence = await recv_json(ws)
            assert silence["type"] == "response.turn.silence", silence
            # silence is an event, never a text answer: no delta may follow
            try:
                raw = await asyncio.wait_for(ws.recv(), 0.5)
                raise AssertionError(f"unexpected event after silence: {raw!r}")
            except asyncio.TimeoutError:
                pass
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] == "parked", snap
            assert snap["text_deltas"] == 0 and snap["text_chars"] == 0, snap
        finally:
            await ws.close()
        print("§8.1 静默不展示为普通回答 (silence 透传且无文本): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_1_silence_not_an_answer() -> None:
    """§8.1 静默：response.turn.silence 透传为事件，无 response.text.delta，
    状态落 parked。"""
    asyncio.run(_test_qa_8_1_silence_not_an_answer())


# ---------------------------------------------------------------- §8.2 边界异常


async def _test_qa_8_2_malformed_json() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await ws.send("{not valid json")  # crosses verbatim to omni
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "invalid_request", err
            # one bad message rejects the message, not the session
            await _send_prompt(ws, 0, "还活着吗？")
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] != "done", snap
        finally:
            await ws.close()
        print("§8.2 非法格式 (坏 JSON → invalid_request 透传，会话保活): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_2_malformed_json() -> None:
    """§8.2 非法格式：坏 JSON 经网关原样到 omni，error{invalid_request}
    透传回客户端，会话保活可继续提问。"""
    asyncio.run(_test_qa_8_2_malformed_json())


async def _test_qa_8_2_seq_out_of_order() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _push_frame(ws, 0, 1.0)
            # omni rejects the out-of-order seq_no; the error must pass through
            fake.reject_next_input("invalid_request")
            await ws.send(json.dumps({
                "type": "input.frame", "seq_no": 5, "timestamp": 2.0,
                "mime_type": "image/jpeg"}))
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "invalid_request", err
            assert "seq_no" not in err, err  # Current omni errors carry no sequence.
            # session survives: the next in-order frame flows normally
            await _push_frame(ws, 1, 3.0)
            seqs = [m["seq_no"] for m in fake.received if m.get("type") == "input.frame"]
            assert seqs == [0, 5, 1], seqs  # rejected frame crossed, nothing dropped
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["status"] == "streaming", snap
        finally:
            await ws.close()
        print("§8.2 seq 乱序 (omni invalid_request 透传，会话保活): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_2_seq_out_of_order() -> None:
    """§8.2 seq 乱序：omni 拒绝乱序 seq_no（invalid_request），错误原样透传，
    会话保活、后续帧正常。"""
    asyncio.run(_test_qa_8_2_seq_out_of_order())


async def _test_qa_8_2_timestamp_regression() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _push_frame(ws, 0, 10.0)
            # omni rejects the regressed timestamp; the error must pass through
            fake.reject_next_input("invalid_request")
            await ws.send(json.dumps({
                "type": "input.frame", "seq_no": 1, "timestamp": 5.0,
                "mime_type": "image/jpeg"}))
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "invalid_request", err
            assert "seq_no" not in err, err
            await _push_frame(ws, 2, 11.0)  # recovered: monotonic again
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["frames_accepted"] == 2, snap
        finally:
            await ws.close()
        print("§8.2 timestamp 回退 (omni invalid_request 透传，会话保活): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_2_timestamp_regression() -> None:
    """§8.2 timestamp 回退：omni 拒绝回退的时间戳（invalid_request），错误
    原样透传，会话保活、后续帧正常。"""
    asyncio.run(_test_qa_8_2_timestamp_regression())


async def _test_qa_8_2_backpressure_delayed_ready() -> None:
    fake = GatewayFakeOmni(ready_delay_s=0.4).start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            # input queue full → omni delays frame.ready; while the client
            # holds the binary waiting for the credit, the gateway must
            # neither drop nor reorder anything (protocol: one frame in
            # flight — ready is the per-frame credit)
            t0 = time.monotonic()
            await _push_frame(ws, 0, 1.0, JPEG)
            first_rtt = time.monotonic() - t0
            await _push_frame(ws, 1, 2.0, JPEG2)
            elapsed = time.monotonic() - t0
            assert first_rtt >= 0.3 and elapsed >= 0.7, \
                f"ready must actually be delayed (first {first_rtt:.2f}s, total {elapsed:.2f}s)"
            # nothing dropped, nothing reordered, payloads verbatim
            seqs = [m["seq_no"] for m in fake.received if m.get("type") == "input.frame"]
            assert seqs == [0, 1], seqs
            assert fake.binaries == [JPEG, JPEG2]
            code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
            assert code == 200 and snap["frames_accepted"] == 2, snap
        finally:
            await ws.close()
        print("§8.2 帧队列满背压 (ready 延迟：不丢不改序): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_2_backpressure_delayed_ready() -> None:
    """§8.2 帧队列满背压：omni 延迟 input.frame.ready 期间网关不丢帧、不
    改序，两段式上传按 seq 完成。"""
    asyncio.run(_test_qa_8_2_backpressure_delayed_ready())


async def _test_qa_8_2_park_timeout_response_failed() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        sid, ws = await _open_ready_session(rig, fake)
        try:
            await _send_prompt(ws, 0, "长时间无输入场景")
            # omni park 超时（长时间无输入同路径）：先 error[response_failed]，
            # 再由服务端结束会话 — 两者都必须到达客户端
            fake.send_event({
                "type": "error", "code": "response_failed",
                "message": "turn parked without input", "turn_id": 0})
            err = await recv_json(ws)
            assert err["type"] == "error" and err["code"] == "response_failed", err
            assert err["message"] == "turn parked without input", err
            asyncio.run_coroutine_threadsafe(fake._ws.close(), fake._loop).result(timeout=5)
            await expect_close(ws, 1011)
        finally:
            await ws.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        code, _ = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 404
        print("§8.2 park 超时/长时间无输入 (response_failed 透传 → 关闭): OK")
    finally:
        await rig.stop()
        fake.close()


def test_qa_8_2_park_timeout_response_failed() -> None:
    """§8.2 park 超时 / 长时间无输入：omni error{response_failed} 原样透传，
    服务端关闭后客户端收 1011、会话销毁。真实 300s park 计时属真机验收
    （docs/qa_acceptance.md 真机清单）。"""
    asyncio.run(_test_qa_8_2_park_timeout_response_failed())


def main() -> int:
    asyncio.run(_test_qa_8_1_prompt_anytime())
    asyncio.run(_test_qa_8_1_done_events())
    asyncio.run(_test_qa_8_1_silence_not_an_answer())
    asyncio.run(_test_qa_8_2_malformed_json())
    asyncio.run(_test_qa_8_2_seq_out_of_order())
    asyncio.run(_test_qa_8_2_timestamp_regression())
    asyncio.run(_test_qa_8_2_backpressure_delayed_ready())
    asyncio.run(_test_qa_8_2_park_timeout_response_failed())
    print("\nGATEWAY QA ACCEPTANCE TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
