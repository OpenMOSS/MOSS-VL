"""Gateway plane P3/P4 tests (GATEWAY_PLAN.md §2-P3/P4 acceptance).

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_gateway_metrics.py -q

P3 计量与日志: trace_id 三元映射（snapshot + 对账记录）、会话终结 JSONL 落库
（client_disconnect / client_delete / reset / omni_dead / attach_timeout 五种
end_reason）、reset 旧会话记一条且新会话继承同一 trace_id。

P4 监控指标: GET /v1/realtime/metrics 的 gauges/counters 形状与事件点更新；
告警规则的「触发条件 → 指标变化」自动化记录（对应 docs/gateway_alerting.md）：
副本 DOWN → replicas_down=1、omni 死亡 → abnormal_disconnects+1、error 事件
透传 → 按 code 计数、janitor 超时 → attach_timeouts+1。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import websockets

from server.gateway.pool import DOWN, READY
from server.tests.test_gateway_rest import (
    GatewayFakeOmni,
    rest,
    start_gateway,
    wait_until,
)
from server.tests.test_gateway_ws import expect_close, recv_json
from server.tests.test_sglang_omni_adapter import JPEG


def read_ledger(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def get_metrics(rig) -> dict:
    code, body = await rest("GET", f"{rig.base}/v1/realtime/metrics")
    assert code == 200, (code, body)
    return body


async def _open_ready(rig):
    """REST create → WSS attach → configure → ready. Returns (sid, ws, create_body)."""
    code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
    assert code == 201, (code, body)
    ws = await websockets.connect(f"{rig.ws_base}/v1/realtime?ws_token={body['ws_token']}")
    created = await recv_json(ws)
    assert created["type"] == "session.created", created
    await ws.send(json.dumps({"type": "session.configure", "max_new_tokens": 64}))
    assert (await recv_json(ws))["type"] == "session.configured"
    assert (await recv_json(ws))["type"] == "session.ready"
    return body["session_id"], ws, body


# ---------------------------------------------------------------- P4 endpoint


async def _test_metrics_endpoint_counters_and_gauges(tmp_path) -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path))
    try:
        m = await get_metrics(rig)
        assert m["gauges"] == {
            "gateway_active_sessions": 0,
            "gateway_replica_slots_total": 1,
            "gateway_replica_slots_used": 0,
            "gateway_replicas_down": 0,
        }, m["gauges"]
        for name in ("gateway_sessions_created_total", "gateway_frames_accepted_total",
                     "gateway_text_chars_total", "gateway_abnormal_disconnects_total",
                     "gateway_attach_timeouts_total"):
            assert m["counters"][name] == 0, (name, m["counters"])
        assert m["counters"]["gateway_errors_total"] == {}
        assert m["pool"]["instances"] == 1 and m["pool"]["capacity"] == 1, m["pool"]

        # create → sessions_created+1, active gauge +1, slot used
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        sid = body["session_id"]
        m = await get_metrics(rig)
        assert m["counters"]["gateway_sessions_created_total"] == 1
        assert m["gauges"]["gateway_active_sessions"] == 1
        assert m["gauges"]["gateway_replica_slots_used"] == 1

        # capacity full (watermark rule 2: used/total == 1.0) → 503 counted by code
        code, err = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 503, (code, err)
        m = await get_metrics(rig)
        assert m["counters"]["gateway_errors_total"] == {"session_capacity_exceeded": 1}

        # unknown session → 404 counted by code
        code, _ = await rest("GET", f"{rig.base}/v1/realtime/sessions/gws-nope")
        assert code == 404
        m = await get_metrics(rig)
        assert m["counters"]["gateway_errors_total"]["session_not_found"] == 1

        # delete → active gauge back to 0, slot freed
        code, _ = await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200
        m = await get_metrics(rig)
        assert m["gauges"]["gateway_active_sessions"] == 0
        assert m["gauges"]["gateway_replica_slots_used"] == 0
        print("metrics endpoint (gauges/counters/error codes/pool summary): OK")
    finally:
        await rig.stop()
        fake.close()


def test_metrics_endpoint_counters_and_gauges(tmp_path) -> None:
    asyncio.run(_test_metrics_endpoint_counters_and_gauges(tmp_path))


# ---------------------------------------------------------------- P3 ledger


async def _test_usage_ledger_client_disconnect(tmp_path) -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path))
    try:
        sid, ws, _ = await _open_ready(rig)
        # one frame + one prompt + one 2-char delta, then client disconnect
        await ws.send(json.dumps({
            "type": "input.frame", "seq_no": 0, "timestamp": 1.0,
            "mime_type": "image/jpeg"}))
        assert (await recv_json(ws))["type"] == "input.frame.ready"
        await ws.send(JPEG)
        assert (await recv_json(ws))["type"] == "input.frame.accepted"
        assert (await recv_json(ws))["type"] == "input.frame.processed"
        await ws.send(json.dumps({
            "type": "input.prompt", "seq_no": 1, "prompt": "看到了什么？"}))
        assert (await recv_json(ws))["type"] == "input.prompt.accepted"
        assert (await recv_json(ws))["type"] == "input.prompt.processed"
        fake.send_delta("你好", turn_id=0)
        assert json.loads(await asyncio.wait_for(ws.recv(), 5.0))["delta"] == "你好"

        # trace_id is part of the status snapshot (session_id/request_id 三元映射)
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200 and len(snap["trace_id"]) == 32, snap

        await ws.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)

        ledger = read_ledger(os.path.join(str(tmp_path), "gateway_usage.jsonl"))
        assert len(ledger) == 1, ledger
        rec = ledger[0]
        assert rec["trace_id"] == snap["trace_id"]
        assert rec["session_id"] == sid
        assert rec["request_id"] == "req-1" and rec["model"] == "fake-moss-vl"
        assert rec["replica_url"] == fake.url
        assert rec["end_reason"] == "client_disconnect", rec
        assert rec["frames_accepted"] == 1 and rec["prompts"] == 1
        assert rec["text_deltas"] == 1 and rec["text_chars"] == 2  # chars, not tokens
        assert 0 < rec["created_at"] <= rec["ended_at"]
        assert rec["duration_s"] >= 0
        print("usage ledger (client_disconnect record, trace_id mapping): OK")
    finally:
        await rig.stop()
        fake.close()


def test_usage_ledger_client_disconnect(tmp_path) -> None:
    asyncio.run(_test_usage_ledger_client_disconnect(tmp_path))


async def _test_usage_ledger_reset_and_delete(tmp_path) -> None:
    """reset: old meter closes with end_reason=reset, the new meter inherits the
    SAME trace_id; DELETE then closes it with client_delete."""
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path))
    try:
        sid, ws, _ = await _open_ready(rig)
        fake.send_delta("旧", turn_id=0)
        assert json.loads(await asyncio.wait_for(ws.recv(), 5.0))["delta"] == "旧"

        # reset while attached: old socket gets 1012, old meter is written
        code, reset = await rest("POST", f"{rig.base}/v1/realtime/sessions/{sid}/reset")
        assert code == 200, (code, reset)
        await expect_close(ws, 1012)

        # reattach with the fresh token, then DELETE
        ws2 = await websockets.connect(
            f"{rig.ws_base}/v1/realtime?ws_token={reset['ws_token']}")
        assert (await recv_json(ws2))["type"] == "session.created"
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert snap["text_chars"] == 0, "new meter must start zeroed"
        await ws2.close()  # client_disconnect of the new meter…
        # …but to exercise client_delete deterministically, DELETE before the
        # disconnect teardown races: recreate cleanly instead
        assert await wait_until(lambda: rig.registry.get(sid) is None)

        code, body2 = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body2)
        code, _ = await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{body2['session_id']}")
        assert code == 200

        ledger = read_ledger(os.path.join(str(tmp_path), "gateway_usage.jsonl"))
        reasons = [r["end_reason"] for r in ledger]
        assert reasons == ["reset", "client_disconnect", "client_delete"], ledger
        # reset record: old meter's counters, closed at reset time
        assert ledger[0]["session_id"] == sid and ledger[0]["text_chars"] == 1
        # new meter: same trace_id, zeroed counters, later created_at
        assert ledger[1]["trace_id"] == ledger[0]["trace_id"]
        assert ledger[1]["text_chars"] == 0
        assert ledger[1]["created_at"] >= ledger[0]["created_at"]
        assert ledger[2]["end_reason"] == "client_delete"
        print("usage ledger (reset record + trace_id inheritance + client_delete): OK")
    finally:
        await rig.stop()
        fake.close()


def test_usage_ledger_reset_and_delete(tmp_path) -> None:
    asyncio.run(_test_usage_ledger_reset_and_delete(tmp_path))


# ---------------------------------------------------------------- P4 triggers


async def _test_metrics_omni_death_alerts(tmp_path) -> None:
    """Alert rules 1+3: omni transport dies → replica quarantined (DOWN) →
    gateway_replicas_down=1, gateway_abnormal_disconnects_total+1, ledger
    record end_reason=omni_dead."""
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path))
    try:
        sid, ws, _ = await _open_ready(rig)
        asyncio.run_coroutine_threadsafe(fake._ws.close(), fake._loop).result(timeout=5)
        await expect_close(ws, 1011)
        assert await wait_until(lambda: rig.pool.replicas[0].state == DOWN)
        m = await get_metrics(rig)
        assert m["gauges"]["gateway_replicas_down"] == 1, m["gauges"]
        assert m["gauges"]["gateway_replica_slots_used"] == 0  # slot released
        assert m["counters"]["gateway_abnormal_disconnects_total"] == 1, m["counters"]
        assert m["gauges"]["gateway_active_sessions"] == 0
        ledger = read_ledger(os.path.join(str(tmp_path), "gateway_usage.jsonl"))
        assert [r["end_reason"] for r in ledger] == ["omni_dead"], ledger
        assert ledger[0]["session_id"] == sid
        print("alert trigger: omni death → replicas_down=1 + abnormal_disconnects+1: OK")
    finally:
        await rig.stop()
        fake.close()


def test_metrics_omni_death_alerts(tmp_path) -> None:
    asyncio.run(_test_metrics_omni_death_alerts(tmp_path))


async def _test_metrics_error_event_passthrough_counted(tmp_path) -> None:
    """Alert rule 5: an omni error event crosses verbatim AND is counted under
    gateway_errors_total{code=...}."""
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path))
    try:
        sid, ws, _ = await _open_ready(rig)
        fake.send_event({"type": "error", "code": "response_failed",
                         "message": "fake boom", "turn_id": 0})
        err = await recv_json(ws)
        assert err["type"] == "error" and err["code"] == "response_failed", err
        assert await wait_until(
            lambda: rig.registry.metrics.snapshot()["counters"]
                    ["gateway_errors_total"].get("response_failed") == 1)
        # a second code lands alongside, not over
        fake.send_event({"type": "error", "code": "input_submission_failed",
                         "message": "fake"})
        assert (await recv_json(ws))["code"] == "input_submission_failed"
        await ws.close()
        assert await wait_until(lambda: rig.registry.get(sid) is None)
        m = await get_metrics(rig)
        assert m["counters"]["gateway_errors_total"] == {
            "response_failed": 1, "input_submission_failed": 1}, m["counters"]
        print("alert trigger: error event passthrough → per-code counter: OK")
    finally:
        await rig.stop()
        fake.close()


def test_metrics_error_event_passthrough_counted(tmp_path) -> None:
    asyncio.run(_test_metrics_error_event_passthrough_counted(tmp_path))


async def _test_metrics_janitor_attach_timeout(tmp_path) -> None:
    """Alert rule 4: created session never attached → janitor GC →
    gateway_attach_timeouts_total+1, ledger record end_reason=attach_timeout."""
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake, data_dir=str(tmp_path),
                              gateway_attach_timeout_s=0.6)
    try:
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        sid = body["session_id"]
        assert await wait_until(lambda: rig.registry.get(sid) is None, timeout=8.0)
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY)
        m = await get_metrics(rig)
        assert m["counters"]["gateway_attach_timeouts_total"] == 1, m["counters"]
        assert m["gauges"]["gateway_active_sessions"] == 0
        ledger = read_ledger(os.path.join(str(tmp_path), "gateway_usage.jsonl"))
        assert [r["end_reason"] for r in ledger] == ["attach_timeout"], ledger
        assert ledger[0]["session_id"] == sid
        print("alert trigger: attach timeout → attach_timeouts+1 + ledger record: OK")
    finally:
        await rig.stop()
        fake.close()


def test_metrics_janitor_attach_timeout(tmp_path) -> None:
    asyncio.run(_test_metrics_janitor_attach_timeout(tmp_path))


def main() -> int:
    import tempfile
    for fn in (_test_metrics_endpoint_counters_and_gauges,
               _test_usage_ledger_client_disconnect,
               _test_usage_ledger_reset_and_delete,
               _test_metrics_omni_death_alerts,
               _test_metrics_error_event_passthrough_counted,
               _test_metrics_janitor_attach_timeout):
        asyncio.run(fn(tempfile.mkdtemp()))
    print("\nGATEWAY METRICS TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
