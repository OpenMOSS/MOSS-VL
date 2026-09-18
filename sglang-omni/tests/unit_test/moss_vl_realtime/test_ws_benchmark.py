"""WebSocket DP benchmark: argument handling and the dry-run plan (no GPU).

Covers only the CPU-safe surface of ``deployment/moss_vl_realtime/ws_benchmark.py``:
argument validation, the in-memory CONFIG override without touching config.json,
the resolved server command shape for dp=1/dp=2, and per-session aggregation.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def ws_benchmark(monkeypatch):
    import sys

    spec = importlib.util.spec_from_file_location(
        "ws_benchmark_test", HERE / "ws_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ws_benchmark_test", module)
    spec.loader.exec_module(module)
    return module


def test_dry_run_dp2_plans_two_device_local_replicas(ws_benchmark, capsys):
    rc = ws_benchmark.main(
        [
            "/models/x",
            "--dp-size",
            "2",
            "--sessions",
            "4",
            "--repeats",
            "2",
            "--dry-run",
        ]
    )

    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dp_size"] == 2
    # The per-replica cap never drops below the config.json default.
    assert plan["max_sessions_per_replica"] == 4
    command = plan["command"]
    assert command[command.index("--dp-size") + 1] == "2"
    # NVIDIA visibility is narrowed by environment(); the child keeps
    # device-local ids 0..dp_size-1. "--gpu 0" stays for replica default.
    assert command[command.index("--gpus") + 1] == "0,1"
    assert command[command.index("--max-running-requests") + 1] == "4"
    assert plan["sessions"] == [4]
    assert plan["events_per_session"] == 14


def test_dry_run_dp1_keeps_single_gpu_command(ws_benchmark, capsys):
    rc = ws_benchmark.main(["/models/x", "--sessions", "2", "--dry-run"])

    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dp_size"] == 1
    assert "--dp-size" not in plan["command"]
    assert "--gpus" not in plan["command"]


@pytest.mark.parametrize(
    "argv",
    [
        (["--dp-size", "0"]),
        (["--max-sessions-per-replica", "0"]),
        (["--sessions", "4", "--dp-size", "1", "--max-sessions-per-replica", "2"]),
        (["--token-rate", "0"]),
        (["--repeats", "0"]),
    ],
)
def test_invalid_benchmark_arguments_are_rejected(ws_benchmark, argv) -> None:
    with pytest.raises(SystemExit):
        ws_benchmark.main(["/models/x", "x", *argv, "--dry-run"])


def test_workload_events_are_scheduled_frames_and_prompts(ws_benchmark) -> None:
    case = ws_benchmark._load_case(HERE / "cases")
    events = ws_benchmark.workload(case, fps=1.0)

    assert len(events) == 14
    assert sum(e["type"] == "frame" for e in events) == 12
    prompts = [e for e in events if e["type"] == "prompt"]
    assert len(prompts) == 2
    assert events[-1]["final"]


def test_lane_metrics_pair_turns_and_token_usage(ws_benchmark) -> None:
    epoch = 100.0

    def frame_event(seq, offset, idx):
        return {
            "type": "frame",
            "seq_no": seq,
            "offset": offset,
            "frame_index": idx,
            "timestamp": float(offset),
            "final": False,
        }

    def prompt_event(seq, offset, final=False):
        return {
            "type": "prompt",
            "seq_no": seq,
            "offset": offset,
            "prompt": "Q",
            "final": final,
        }

    events = [
        frame_event(0, 0.0, 0),
        frame_event(1, 1.0, 1),
        frame_event(2, 2.0, 2),
        prompt_event(3, 3.25),
        prompt_event(4, 5.0, final=True),
    ]
    sent = [
        {
            "seq_no": index,
            "planned_at": epoch + event["offset"],
            "sent_at": epoch + event["offset"] + 0.01,
        }
        for index, event in enumerate(events)
    ]

    def received_at(type_, at, **rest):
        return {
            "payload": {"type": type_, **rest},
            "arrived_at": epoch + at,
        }

    received = [
        received_at("input.frame.processed", 0.4, seq_no=0),
        received_at("input.frame.processed", 1.4, seq_no=1),
        received_at("input.frame.processed", 2.4, seq_no=2),
        received_at("input.prompt.processed", 3.5, seq_no=3),
        received_at("response.text.delta", 4.0, turn_id=0),
        received_at("response.text.delta", 4.2, turn_id=0),
        received_at("input.prompt.processed", 5.0, seq_no=4),
        received_at("response.text.delta", 5.6, turn_id=1),
        received_at("session.usage", 6.0, decoder_tokens=5),
        received_at("session.done", 6.1, seq_no=4),
    ]

    lane, raw_gaps = ws_benchmark._lane_metrics(
        lane_index=0, epoch=epoch, events=events, sent=sent, received=received
    )

    # Token span: first/last deltas at +4.0/+5.6 -> a 1.6s decode window; the
    # sampled decoder count (usage) wins over delta counting.
    assert lane["ttft_s"] == pytest.approx(4.0)
    assert lane["generated_tokens"] == 5
    assert lane["tokens_per_second"] == pytest.approx(5 / 1.6)
    assert lane["questions"][0]["turn_id"] == 0
    assert lane["questions"][0]["ttft_s"] == pytest.approx(4.0 - 3.26)
    assert lane["questions"][1]["turn_id"] == 1
    assert lane["questions"][1]["ttft_s"] == pytest.approx(5.6 - 5.01)
    assert lane["token_gaps"]["mean"] == pytest.approx(0.2)
    assert raw_gaps == pytest.approx([0.2])
    # WARMUP_FRAMES=2 -> only the third frame contributes processing delay.
    assert lane["frame_delay"]["count"] == 1
    assert lane["errors"] == []


def test_summary_groups_lanes_by_session_count_with_dp_label(ws_benchmark) -> None:

    def lane(i, tps, ttft):
        return {
            "lane": i,
            "ttft_s": ttft,
            "questions": [],
            "generated_tokens": 10,
            "text_delta_events": 10,
            "elapsed_s": 1.0,
            "decode_span_s": 1.0,
            "tokens_per_second": tps,
            "token_gaps": ws_benchmark.stats([0.1] if i % 2 else []),
            "frame_delay": ws_benchmark.stats([]),
            "send_lag": ws_benchmark.stats([0.01]),
            "errors": [],
        }

    trials = [
        {
            "sessions": 2,
            "repeat": 0,
            "lanes": [lane(0, 5.0, 1.0), lane(1, 3.0, 2.0)],
            "lane_gaps": [[0.1], [0.2]],
        },
        {
            "sessions": 2,
            "repeat": 1,
            "lanes": [lane(0, 4.0, 1.5), lane(1, 2.0, 2.5)],
            "lane_gaps": [[0.3], []],
        },
    ]
    rows = ws_benchmark._summarize(trials, dp_size=2)

    assert rows[0]["dp_size"] == 2
    assert rows[0]["label"] == "dp2"
    assert rows[0]["lane_tokens_per_second_mean"] == pytest.approx(3.5)
    assert rows[0]["lane_tokens_per_second_min"] == 2.0
    assert rows[0]["ttft_s_mean"] == pytest.approx(1.75)
    assert rows[0]["token_gap_stats"]["count"] == 3
