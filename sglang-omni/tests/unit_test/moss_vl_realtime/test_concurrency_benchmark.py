"""Independent sender scheduling, turn attribution and latency accounting."""

import copy
import importlib.util
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    for name in ("semantic_checks", "concurrency_benchmark"):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
    return loaded


def trials(counts=(1, 2, 4, 8), repeats=2):
    rows = []
    for n in counts:
        for trial in range(repeats):
            lanes = []
            for i in range(n):
                start = 10 + i * 0.01
                lanes.append(
                    dict(
                        lane=i,
                        epoch=start,
                        ended_at=start + 12,
                        frames=[
                            dict(
                                seq_no=j,
                                frame_index=j,
                                sha256=str(j),
                                planned_at=start + j,
                                sent_at=start + j + 0.001,
                                processed_at=start + j + 0.101 * n,
                            )
                            for j in range(12)
                        ],
                        questions=[
                            dict(
                                seq_no=j + 20,
                                turn_id=j + 1,
                                planned_at=start + j * 4 + 3.25,
                                sent_at=start + j * 4 + 3.251,
                                first_text_at=start + j * 4 + 3.251 + 0.02 * n,
                                text="Answer",
                            )
                            for j in range(2)
                        ],
                        decode_intervals=[0.01 * n] * 18,
                        generated_tokens=20,
                        decode_batch_sizes=[1] * 20,
                        peak_pending_events=1,
                        errors=[],
                        missing_events=[],
                        unexpected_events=[],
                        scheduled_events=14,
                        sent_events=14,
                        processed_events=14,
                    )
                )
            rows.append(
                dict(
                    protocol="independent_realtime_v1",
                    sessions=n,
                    trial=trial,
                    kv_recovered=True,
                    fps=1.0,
                    token_rate=10.0,
                    lanes=lanes,
                )
            )
    return rows


def test_independent_metrics_and_non_full_batches_are_valid(module):
    result = module.summarize(trials(), [1, 2, 4, 8], 2)
    for r in result:
        n = r["sessions"]
        assert r["tpot_ratio"] == pytest.approx(n)
        assert r["mean_lane_tokens_per_second"] == pytest.approx(100 / n)
        assert r["aggregate_wall_tokens_per_second"] == pytest.approx(
            n * 20 / (12 + (n - 1) * 0.01)
        )
        assert r["batch_histogram"]["1"] == n * 40
        assert r["frame"]["count"] == n * 20
        assert len(r["lanes"]) == n


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "lane", "protocol", "serialized", "hash", "nan"],
)
def test_bad_data_and_old_barrier_protocol_are_rejected(module, mutation):
    rows = trials()
    if mutation == "missing":
        rows.pop()
    if mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    if mutation == "lane":
        rows[-1]["lanes"].pop()
    if mutation == "protocol":
        rows[-1]["protocol"] = "synchronized_v0"
    if mutation == "serialized":
        rows[-1]["lanes"][0]["epoch"] = 100
    if mutation == "hash":
        rows[-1]["lanes"][0]["frames"][0]["sha256"] = "different"
    if mutation == "nan":
        rows[-1]["lanes"][0]["decode_intervals"][0] = float("nan")
    with pytest.raises(ValueError):
        module.summarize(rows, [1, 2, 4, 8], 2)


def test_silent_answer_and_missing_frame_are_counted(module):
    rows = trials()
    lane = rows[-1]["lanes"][0]
    lane["questions"][0]["first_text_at"] = None
    lane["frames"][3]["processed_at"] = None
    lane["errors"] = ["missing"]
    result = module.summarize(rows, [1, 2, 4, 8], 2)[-1]
    assert (
        result["unanswered_prompts"]
        == result["missing_frames"]
        == result["failed_lanes"]
        == 1
    )


def state():
    return dict(
        ready=threading.Event(),
        cancel=threading.Event(),
        sender_done=threading.Event(),
        lock=threading.Lock(),
        sent=[],
        errors=[],
    )


def test_one_sender_never_waits_for_another_sender_or_output(module):
    a, b = state(), state()
    a["ready"].set()
    calls = []
    engine = SimpleNamespace(
        update=lambda rid, event, phase, final: calls.append((rid, phase))
    )
    events = [
        dict(seq_no=0, offset=0, final=False),
        dict(seq_no=1, offset=0.01, final=True),
    ]
    threads = [
        threading.Thread(
            target=module.send_lane, args=(engine, rid, lane, events, 0, 1)
        )
        for rid, lane in (("a", a), ("b", b))
    ]
    for thread in threads:
        thread.start()
    try:
        assert a["sender_done"].wait(1)
        assert not b["sender_done"].is_set()
        assert calls == [("a", 0), ("a", 1)]
    finally:
        b["ready"].set()
        for thread in threads:
            thread.join(1)
    assert calls[-2:] == [("b", 0), ("b", 1)]


def test_turn_mapping_excludes_old_text_and_silence_gaps(module):
    lane = dict(
        sent=[
            dict(type="prompt", seq_no=0, planned_at=1, sent_at=1.0),
            dict(
                type="frame",
                seq_no=1,
                frame_index=0,
                sha256="x",
                planned_at=2,
                sent_at=2,
            ),
        ],
        received=[
            dict(event="input.prompt.processed", seq_no=0, turn_id=1, received_at=1.1),
            dict(modality="text", turn_id=0, text="Old answer", received_at=1.15),
            dict(modality="text", turn_id=1, text=" ", received_at=1.19),
            dict(modality="text", turn_id=1, text="New answer", received_at=1.2),
            dict(event="input.frame.processed", seq_no=1, received_at=2.1),
        ],
        errors=[],
        epoch=1,
        ended_at=4,
        peak_pending=1,
        schedule=[0, 1],
    )
    records = [
        dict(token=t, time=when, turn_id=turn, batch_size=1)
        for t, when, turn in [
            (3, 1.0, 0),
            (3, 1.1, 1),
            (4, 1.2, 1),
            (0, 1.3, 1),
            (3, 3.1, 1),
            (4, 3.2, 1),
        ]
    ]
    result = module.lane_result(0, lane, records, {0})
    assert result["questions"][0]["text"] == " New answer"
    assert result["questions"][0]["first_text_at"] == 1.2
    assert result["decode_intervals"] == pytest.approx([0.1, 0.1])
    assert result["generated_tokens"] == 4


def test_full_measurement_uses_natural_generation_and_all_scheduled_inputs(module):
    class Fake:
        def __init__(self):
            self.scheduler = SimpleNamespace(
                outbox=queue.Queue(), abort=lambda rid: None
            )
            self.thread = SimpleNamespace(is_alive=lambda: True)
            self.tokenizer = SimpleNamespace(all_special_ids=[0])
            self.silence = 0
            self.streams = {}

        def emit(self, rid, data, kind="stream"):
            self.scheduler.outbox.put(
                SimpleNamespace(request_id=rid, type=kind, data=data)
            )

        def new(self, case, force, allowance, benchmark, token_rate):
            assert force is None and benchmark is False
            assert token_rate == 10
            rid = str(len(self.streams))
            self.streams[rid] = dict(records=[], turn=0)
            self.emit(rid, dict(event="session.ready"))
            return rid

        def update(self, rid, event, phase, final):
            stream = self.streams[rid]
            data = dict(seq_no=event["seq_no"])
            if event["type"] == "prompt":
                stream["turn"] += 1
                data.update(event="input.prompt.processed", turn_id=stream["turn"])
            else:
                data["event"] = "input.frame.processed"
            self.emit(rid, data)
            if event["type"] == "prompt":
                now = time.perf_counter()
                stream["records"].extend(
                    dict(
                        token=2,
                        time=now + i * 0.001,
                        turn_id=stream["turn"],
                        batch_size=1,
                    )
                    for i in range(5)
                )
                stream["records"].append(
                    dict(
                        token=0, time=now + 0.006, turn_id=stream["turn"], batch_size=1
                    )
                )
                self.emit(
                    rid, dict(modality="text", turn_id=stream["turn"], text="Answer")
                )
            if final:
                self.emit(rid, {}, "result")

        def recover(self, rids):
            return len(rids) == 2

    case = dict(
        events=[
            dict(type="frame", timestamp=i, frame_path="unused", sha256=str(i))
            for i in range(12)
        ]
    )
    result = module.measure(Fake(), case, 2, fps=100)
    assert all(
        l["sent_events"] == l["processed_events"] == 14 and not l["errors"]
        for l in result["lanes"]
    )
    assert all(len(l["questions"]) == 2 for l in result["lanes"])


def test_latency_cli_enables_independent_mode(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    from evaluation import parse_args

    args = parse_args(["latency", "/model", "--sessions", "1", "2", "4", "8"])
    assert args.suite == "concurrency" and args.fps == 1 and args.repeats == 3
    assert args.token_rate == 10
    assert parse_args(["latency", "/model"]).suite == "latency"
    for flags in (["--fps", "0"], ["--fps", "nan"], ["--token-rate", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["latency", "/model", "--sessions", "1", *flags])
